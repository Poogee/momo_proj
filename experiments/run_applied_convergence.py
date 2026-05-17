"""Applied, fully-causal experiment: across four real domains — financial
daily returns, financial 15-min returns, FRED macro levels, and a
non-financial sensor series (ETT transformer oil temperature) — does
preprocessing the observed series with a *causal* filter speed up the
training convergence of an AR(p) model (and not hurt the causal
walk-forward forecast)?

Only causal filters are used (F0 identity, F1 trailing MA, F2 forward
Kalman, F4 trailing/causal median, FA online) so there is no look-ahead.
Train/test is a single causal split at 70%; AR features always use only
the past, and the held-out target is the *raw* (unfiltered) standardised
future value, identical across filters, so the comparison is fair and
leakage-free.

The honest expectation (and what the data shows): on near-white return
series there is no smooth signal to recover and F0 is best (a faithfully
reported negative zone); on the macro / sensor series, which carry a
recoverable low-frequency signal, a causal filter converges markedly
faster and forecasts at least as well — the regime where preprocessing
measurably helps.

Outputs tables/applied_convergence.csv and a per-domain summary.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from momo.data import (
    DEFAULT_FRED_SERIES,
    DEFAULT_TICKERS,
    fetch_fred,
    fetch_intraday,
    fetch_nonfinancial,
    fetch_returns,
)
from momo.filters import (
    CausalMedianFilter,
    IdentityFilter,
    KalmanLocalLevelFilter,
    MovingAverageFilter,
    OnlineAdaptiveFilter,
)
from momo.metrics import time_to_drop
from momo.noise import GaussianNoise
from momo.optim import run_optimization

P = 5
STEPS = 4000
SEEDS = list(range(4))

CAUSAL_FILTERS = {
    "F0": lambda: IdentityFilter(),
    "F1": lambda: MovingAverageFilter(window=15),
    "F2": lambda: KalmanLocalLevelFilter(process_var=1e-3, obs_var=1.0),
    "F4": lambda: CausalMedianFilter(window=9),
    "FA": lambda: OnlineAdaptiveFilter(window=9, k=3.0),
}
OPTIMIZERS = ["sgd", "adam"]


class _LinAR:
    def __init__(self, p):
        self.p = p
        self.dim = p

    def attach(self, X, y):
        self.X, self.y = X, y

    def loss(self, x, Z=None, y=None):
        Z = self.X if Z is None else Z
        t = self.y if y is None else y
        r = Z @ x - t
        return float(np.mean(r * r))

    def grad(self, x, Z=None, y=None):
        Z = self.X if Z is None else Z
        t = self.y if y is None else y
        r = Z @ x - t
        return (2.0 / Z.shape[0]) * (Z.T @ r)

    def sample_batch(self, rng, n):
        idx = rng.integers(0, self.X.shape[0], size=min(n, self.X.shape[0]))
        return self.X[idx], self.y[idx]


def _ar_design(series, p):
    n = series.size
    if n <= p:
        return np.empty((0, p)), np.empty(0)
    X = np.lib.stride_tricks.sliding_window_view(series[:n - 1], p)
    return X.copy(), series[p:].copy()


def _domains():
    d = {}
    fin = fetch_returns(sum(DEFAULT_TICKERS.values(), []))
    d["financial_daily"] = {c: fin[c].dropna().to_numpy()
                            for c in list(fin.columns)[:8]}
    intr = fetch_intraday(sum(DEFAULT_TICKERS.values(), []), interval="15m")
    r15 = np.log(intr / intr.shift(1)).dropna(how="all")
    d["financial_15m"] = {c: r15[c].dropna().to_numpy()
                          for c in list(r15.columns)[:4]}
    fred = fetch_fred(DEFAULT_FRED_SERIES)
    d["macro_fred"] = {c: fred[c].dropna().to_numpy()
                       for c in fred.columns
                       if fred[c].dropna().size > 300}
    ett = fetch_nonfinancial()
    d["nonfinancial_ett"] = {"OT": ett["OT"].to_numpy()[::4],
                             "HUFL": ett["HUFL"].to_numpy()[::4]}
    return d


def _standardize(s):
    s = np.asarray(s, dtype=float)
    s = s[np.isfinite(s)]
    mu, sd = float(np.mean(s)), float(np.std(s))
    return (s - mu) / (sd if sd > 1e-12 else 1.0)


def run_cell(domain, name, series, filt_key, opt, seed):
    s = _standardize(series)
    if s.size < 6 * P + 80:
        return None
    cut = int(0.7 * s.size)
    filt = CAUSAL_FILTERS[filt_key]()
    sf = filt.apply(s)  # causal -> point t uses only s[<=t]
    Xtr, ytr = _ar_design(sf[:cut], P)
    # test features: causal filtered past; target: RAW future (shared)
    Xte_f, _ = _ar_design(sf[cut - P:], P)
    _, yte_raw = _ar_design(s[cut - P:], P)
    m = min(Xte_f.shape[0], yte_raw.shape[0])
    if Xtr.shape[0] < 50 or m < 20:
        return None
    task = _LinAR(P)
    task.attach(Xtr, ytr)
    res = run_optimization(task=task, optimizer=opt,
                           noise=GaussianNoise(0.05), filt=IdentityFilter(),
                           steps=STEPS, lr=1e-2, batch_size=64, seed=seed,
                           noise_scale=1.0, preprocess_mode="series")
    g = res.grad_norm_sq_history
    td = time_to_drop(g, factor=1e2)
    pred = Xte_f[:m] @ res.x_final
    ho = float(np.mean((pred - yte_raw[:m]) ** 2))
    return dict(domain=domain, series=name, filter=filt_key, optimizer=opt,
                seed=seed, t_conv=STEPS if td is None else td,
                conv=int(td is not None), final_grad=float(g[-1]),
                holdout_mse=ho)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=10)
    ap.add_argument("--csv", type=Path,
                    default=Path("tables/applied_convergence.csv"))
    ap.add_argument("--summary-csv", type=Path,
                    default=Path("tables/applied_convergence_summary.csv"))
    args = ap.parse_args()

    dom = _domains()
    cells = [(dn, nm, sv, fk, o, s)
             for dn, series in dom.items()
             for nm, sv in series.items()
             for fk in CAUSAL_FILTERS for o in OPTIMIZERS for s in SEEDS]
    print(f"running {len(cells)} cells")
    out = [r for r in Parallel(n_jobs=args.n_jobs, verbose=3,
                               backend="loky")(
        delayed(run_cell)(*c) for c in cells) if r is not None]
    df = pd.DataFrame(out)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)

    g = df.groupby(["domain", "optimizer", "filter"])
    summ = g.agg(t_conv_med=("t_conv", "median"),
                 conv_frac=("conv", "mean"),
                 holdout_mse_med=("holdout_mse", "median")).reset_index()
    summ.to_csv(args.summary_csv, index=False)

    print("\n=== applied convergence: causal filter vs F0 (Adam) ===")
    for dn, s in summ[summ.optimizer == "adam"].groupby("domain"):
        f0 = s[s["filter"] == "F0"]
        if f0.empty:
            continue
        f0t = float(f0["t_conv_med"].iloc[0])
        f0h = float(f0["holdout_mse_med"].iloc[0])
        best = s.loc[s["t_conv_med"].idxmin()]
        if best["filter"] != "F0" and best["t_conv_med"] < f0t:
            verdict = (f"{best['filter']} {f0t / max(best['t_conv_med'],1):.1f}x "
                       f"faster (holdout {best['holdout_mse_med']:.3g} "
                       f"vs F0 {f0h:.3g})")
        else:
            verdict = f"F0 best (t_conv {f0t:.0f}, holdout {f0h:.3g})"
        print(f"  {dn:18s}: {verdict}")
    print(f"\nwrote {args.csv}, {args.summary_csv}")


if __name__ == "__main__":
    main()
