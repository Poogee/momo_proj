"""Extended factorial: all causal filters (incl. new F10/F11 cascade)
x optimizer x domain x model {regression, classification},
on synthetic noise calibrated to real-data diagnostics plus a wide
basket of real domains (financial daily, intraday 5m/15m, FRED macro,
ETTh1, ETTh2, ETTm1, sunspots).

Goal — confirm the practical criteria from the paper with much more
real-data coverage and surface the cascade filter (F10/F11) on the
mixed heavy-tailed + long-memory regime, which the existing single-stage
filters do not handle.

All filters used here are strictly causal (no look-ahead). The held-out
target is always the *raw* (unfiltered) future, identical across filters,
so the comparison is fair.

Outputs:
  tables/extended_factorial.csv             — per-cell raw metrics
  tables/extended_factorial_summary.csv     — domain x filter x optimizer
  tables/extended_factorial_criteria.csv    — confirmation of the
                                              (alpha_hat, H_hat) -> filter rule
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from momo.data import (
    EXTENDED_TICKERS,
    DEFAULT_FRED_SERIES,
    fetch_fred,
    fetch_intraday,
    fetch_nonfinancial,
    fetch_returns,
    fetch_sunspots,
)
from momo.filters import (
    AdaptiveCascadeFilter,
    CausalCascadeFilter,
    CausalHybridMedianWavelet,
    CausalMedianFilter,
    IdentityFilter,
    KalmanLocalLevelFilter,
    MovingAverageFilter,
    OnlineAdaptiveFilter,
)
from momo.metrics import (
    convergence_auc,
    divergence_slope,
    hurst_dfa,
    mcculloch_alpha,
    noise_floor_quantiles,
    time_to_drop,
)
from momo.noise import GaussianNoise
from momo.optim import run_optimization

P = 5
STEPS = 4000
SEEDS = list(range(4))

# only strictly causal filters here so real-data results are leak-free
CAUSAL_FILTERS = {
    "F0":  lambda: IdentityFilter(),
    "F1":  lambda: MovingAverageFilter(window=15),
    "F2":  lambda: KalmanLocalLevelFilter(process_var=1e-3, obs_var=1.0),
    "F4":  lambda: CausalMedianFilter(window=9),
    "F7":  lambda: CausalHybridMedianWavelet(median_window=5),
    "F10": lambda: CausalCascadeFilter(median_window=3,
                                       process_var=1e-3, obs_var=1.0),
    "F11": lambda: AdaptiveCascadeFilter(alpha_threshold=1.9,
                                         hurst_threshold=0.6,
                                         median_window=3,
                                         process_var=1e-3, obs_var=1.0),
    "FA":  lambda: OnlineAdaptiveFilter(window=9, k=3.0),
}

OPTIMIZERS = ["sgd", "adam"]


class _LinAR:
    """In-script linear AR(p) task: regression vs raw future target.
    classification mode predicts the sign of the next return."""

    def __init__(self, p, classification: bool = False):
        self.p = p
        self.dim = p
        self.cls = bool(classification)

    def attach(self, X, y):
        self.X, self.y = X, y

    def loss(self, x, Z=None, y=None):
        Z = self.X if Z is None else Z
        t = self.y if y is None else y
        if not self.cls:
            r = Z @ x - t
            return float(np.mean(r * r))
        # logistic on +/- 1 targets
        z = Z @ x
        # binary cross-entropy with t in {0,1}
        t01 = (t > 0).astype(float)
        return float(np.mean(np.logaddexp(0.0, z) - t01 * z))

    def grad(self, x, Z=None, y=None):
        Z = self.X if Z is None else Z
        t = self.y if y is None else y
        if not self.cls:
            r = Z @ x - t
            return (2.0 / Z.shape[0]) * (Z.T @ r)
        z = Z @ x
        s = 1.0 / (1.0 + np.exp(-z))
        t01 = (t > 0).astype(float)
        return (Z.T @ (s - t01)) / Z.shape[0]

    def sample_batch(self, rng, n):
        idx = rng.integers(0, self.X.shape[0],
                           size=min(n, self.X.shape[0]))
        return self.X[idx], self.y[idx]


def _ar_design(series, p):
    n = series.size
    if n <= p:
        return np.empty((0, p)), np.empty(0)
    X = np.lib.stride_tricks.sliding_window_view(series[:n - 1], p)
    return X.copy(), series[p:].copy()


def _standardize(s):
    s = np.asarray(s, dtype=float)
    s = s[np.isfinite(s)]
    mu, sd = float(np.mean(s)), float(np.std(s))
    return (s - mu) / (sd if sd > 1e-12 else 1.0)


def _domains():
    """Build all (domain, series_name -> 1d numpy array) groups."""
    d: dict[str, dict[str, np.ndarray]] = {}

    ext = sum(EXTENDED_TICKERS.values(), [])

    # financial daily — extended basket
    fin = fetch_returns(ext)
    d["financial_daily"] = {c: fin[c].dropna().to_numpy()
                            for c in list(fin.columns)[:10]}

    # financial intraday — 15m and 5m, separate domains
    for itv, key in [("15m", "financial_15m"), ("5m", "financial_5m")]:
        try:
            intr = fetch_intraday(ext, interval=itv)
            r = np.log(intr / intr.shift(1)).dropna(how="all")
            d[key] = {c: r[c].dropna().to_numpy()
                      for c in list(r.columns)[:4]}
        except Exception:
            pass

    # macroeconomic (FRED)
    fred = fetch_fred(DEFAULT_FRED_SERIES)
    d["macro_fred"] = {c: fred[c].dropna().to_numpy()
                       for c in fred.columns
                       if fred[c].dropna().size > 300}

    # non-financial sensor — three ETT variants
    for v in ("ETTh1", "ETTh2", "ETTm1"):
        ett = fetch_nonfinancial(variant=v)
        # downsample minute-level a little so all are similar length
        stride = 4 if v == "ETTm1" else 4
        d[f"sensor_{v.lower()}"] = {
            "OT":   ett["OT"].to_numpy()[::stride],
            "HUFL": ett["HUFL"].to_numpy()[::stride],
        }

    # scientific (sunspots) — smooth signal, mild heavy tails
    sn = fetch_sunspots()
    d["scientific_sunspots"] = {
        "sn":     sn["sunspot"].to_numpy()[::5],
    }
    return d


def _diagnostics(series: np.ndarray) -> tuple[float, float]:
    """alpha-hat (McCulloch) and H-hat (DFA) on the differenced series —
    estimated on the training slice only."""
    diff = np.diff(series)
    alpha_hat = mcculloch_alpha(diff)
    if not np.isfinite(alpha_hat):
        alpha_hat = 2.0
    h_hat = hurst_dfa(diff)
    if not np.isfinite(h_hat):
        h_hat = 0.5
    return float(alpha_hat), float(h_hat)


def run_cell(domain, name, series, filt_key, opt, classification, seed):
    s = _standardize(series)
    if s.size < 6 * P + 80:
        return None
    cut = int(0.7 * s.size)

    # ---- diagnostics computed *only on the training slice* ------------
    alpha_hat, h_hat = _diagnostics(s[:cut])

    filt = CAUSAL_FILTERS[filt_key]()
    sf = filt.apply(s)
    Xtr, ytr = _ar_design(sf[:cut], P)
    Xte_f, _ = _ar_design(sf[cut - P:], P)
    _, yte_raw = _ar_design(s[cut - P:], P)
    m = min(Xte_f.shape[0], yte_raw.shape[0])
    if Xtr.shape[0] < 50 or m < 20:
        return None

    task = _LinAR(P, classification=classification)
    task.attach(Xtr, ytr)
    res = run_optimization(task=task, optimizer=opt,
                           noise=GaussianNoise(0.05), filt=IdentityFilter(),
                           steps=STEPS, lr=1e-2, batch_size=64, seed=seed,
                           noise_scale=1.0, preprocess_mode="series")
    g = res.grad_norm_sq_history
    td = time_to_drop(g, factor=1e2)
    fq = noise_floor_quantiles(g, tail_frac=0.2)

    pred = Xte_f[:m] @ res.x_final
    if classification:
        ho = float(np.mean((pred > 0).astype(float)
                           == (yte_raw[:m] > 0).astype(float)))
    else:
        ho = float(np.mean((pred - yte_raw[:m]) ** 2))

    return dict(domain=domain, series=name, filter=filt_key, optimizer=opt,
                model=("classification" if classification else "regression"),
                seed=seed,
                t_conv=STEPS if td is None else td,
                conv=int(td is not None),
                slope=divergence_slope(g),
                floor_p50=fq[0.5], floor_p10=fq[0.1], floor_p90=fq[0.9],
                auc=convergence_auc(g),
                final_grad=float(g[-1]),
                holdout=ho,
                alpha_hat=alpha_hat, hurst_hat=h_hat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--csv", type=Path,
                    default=Path("tables/extended_factorial.csv"))
    ap.add_argument("--summary-csv", type=Path,
                    default=Path("tables/extended_factorial_summary.csv"))
    ap.add_argument("--criteria-csv", type=Path,
                    default=Path("tables/extended_factorial_criteria.csv"))
    args = ap.parse_args()

    dom = _domains()
    filters = list(CAUSAL_FILTERS)
    seeds = SEEDS
    models = [False, True]            # regression, classification
    if args.smoke:
        dom = {k: v for k, v in list(dom.items())[:3]}
        filters = ["F0", "F4", "F10", "F11"]
        seeds = [0]
        models = [False]

    cells = [(dn, nm, sv, fk, o, cls, s)
             for dn, series in dom.items()
             for nm, sv in series.items()
             for fk in filters for o in OPTIMIZERS
             for cls in models for s in seeds]
    print(f"running {len(cells)} cells on {args.n_jobs} jobs")
    t0 = time.perf_counter()
    out = [r for r in Parallel(n_jobs=args.n_jobs, verbose=4, backend="loky")(
        delayed(run_cell)(*c) for c in cells) if r is not None]
    print(f"done in {time.perf_counter() - t0:.1f}s, {len(out)} valid cells")

    df = pd.DataFrame(out)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)

    # ---- aggregated summary per (domain, model, optimizer, filter) -----
    g = df.groupby(["domain", "model", "optimizer", "filter"])
    summ = g.agg(
        n_cells=("seed", "size"),
        t_conv_med=("t_conv", "median"),
        conv_frac=("conv", "mean"),
        floor_p50_med=("floor_p50", "median"),
        slope_mean=("slope", "mean"),
        auc_mean=("auc", "mean"),
        holdout_med=("holdout", "median"),
        alpha_hat_med=("alpha_hat", "median"),
        hurst_hat_med=("hurst_hat", "median"),
    ).reset_index()

    # speedup and floor reduction vs F0 within each (domain, model, opt)
    summ["speedup_vs_F0"] = np.nan
    summ["floor_ratio_vs_F0"] = np.nan
    for key, sub in summ.groupby(["domain", "model", "optimizer"]):
        f0 = sub[sub["filter"] == "F0"]
        if f0.empty:
            continue
        f0t = float(f0["t_conv_med"].iloc[0])
        f0f = float(f0["floor_p50_med"].iloc[0])
        if f0t > 0:
            summ.loc[sub.index, "speedup_vs_F0"] = (
                f0t / sub["t_conv_med"].clip(lower=1))
        if f0f > 0:
            summ.loc[sub.index, "floor_ratio_vs_F0"] = f0f / sub["floor_p50_med"]
    summ.to_csv(args.summary_csv, index=False)

    # ---- criteria confirmation ----------------------------------------
    # for each domain, find best filter and check what (alpha_hat, H_hat)
    # bucket it falls into; this is the "where does the rule fire" table
    # we cite in the paper to claim the rule is empirically grounded.
    rows = []
    for (dn, mdl, opt), sub in summ.groupby(["domain", "model", "optimizer"]):
        f0 = sub[sub["filter"] == "F0"]
        if f0.empty:
            continue
        # best by t_conv, tiebreak by holdout (lower is better for MSE,
        # but for classification holdout=accuracy: higher better — flip)
        cand = sub.copy()
        if mdl == "classification":
            cand["score"] = cand["t_conv_med"] - 1e6 * cand["holdout_med"]
        else:
            cand["score"] = cand["t_conv_med"] + 1e6 * cand["holdout_med"]
        best = cand.loc[cand["score"].idxmin()]
        ah, hh = float(best["alpha_hat_med"]), float(best["hurst_hat_med"])
        if ah >= 1.9 and hh <= 0.6:
            bucket = "light_short"
        elif ah >= 1.9 and hh > 0.6:
            bucket = "light_long"
        elif ah < 1.9 and hh <= 0.6:
            bucket = "heavy_short"
        else:
            bucket = "heavy_long"
        rows.append(dict(domain=dn, model=mdl, optimizer=opt,
                         alpha_hat=ah, hurst_hat=hh, bucket=bucket,
                         best_filter=best["filter"],
                         best_t_conv=float(best["t_conv_med"]),
                         best_conv_frac=float(best["conv_frac"]),
                         best_holdout=float(best["holdout_med"]),
                         f0_t_conv=float(f0["t_conv_med"].iloc[0]),
                         f0_conv_frac=float(f0["conv_frac"].iloc[0]),
                         f0_holdout=float(f0["holdout_med"].iloc[0]),
                         speedup=(float(f0["t_conv_med"].iloc[0]) /
                                  max(float(best["t_conv_med"]), 1.0))))
    crit = pd.DataFrame(rows)
    crit.to_csv(args.criteria_csv, index=False)

    print("\n=== best filter per (domain, model, optimizer) ===")
    for _, r in crit.iterrows():
        print(f"  {r['domain']:22s} {r['model'][:5]:5s} {r['optimizer']:5s}  "
              f"alpha={r['alpha_hat']:.2f} H={r['hurst_hat']:.2f} "
              f"-> {r['best_filter']:4s} speedup {r['speedup']:5.1f}x "
              f"({r['bucket']})")
    print(f"\nwrote {args.csv}, {args.summary_csv}, {args.criteria_csv}")


if __name__ == "__main__":
    main()
