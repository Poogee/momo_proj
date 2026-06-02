"""June 2026 dataset sweep with a *lean* causal filter set.

Motivation
----------
The earlier factorial carried 14 filters (F0--F11, FA, FE), several of
which were slow (learnable CNNs F5/F9, ensemble FE) or simply tracked the
identity baseline on every real series (the linear F1/F2/F3 family, the
hybrid wavelet F7). For this sweep we keep only the filters that (a) are
strictly causal, (b) run in O(T), and (c) are theoretically motivated for
the heavy-tailed / mixed regime where filtering actually pays off:

    F0   identity (control / baseline)
    F4   causal median               (Lemma: finite-variance surrogate)
    FA   online median/EMA switch     (per-step robust/efficient choice)
    F10  causal cascade median->Kalman (mixed heavy-tail + drift)
    F11  adaptive cascade (alpha,H routed)

Datasets (only freely obtainable sources; LOBSTER full feed, NYSE TAQ and
CRSP are subscription/portal-gated and are dropped -- see DECISIONS.md):

    financial:      Binance BTCUSDT, ETHUSDT (1h log-returns, 2024)
    non-financial:  ETTm2, UCI/LSTNet electricity, LSTNet PEMS traffic,
                    NOAA GHCN-daily weather, NASA C-MAPSS turbofan,
                    daily ATM cash withdrawals

Every cell trains AR(5) regression and sign classification under SGD and
Adam, 4 seeds, with strictly causal pre-filtering; the holdout target is
the *raw* future, identical across filters. Domains where nothing
converges are dropped from the reported tables (see ``--min-conv``).

Outputs:
  tables/new_datasets.csv           per-cell raw metrics
  tables/new_datasets_summary.csv   domain x model x optimizer x filter
  tables/new_datasets_criteria.csv  best filter + (alpha_hat,H_hat) bucket
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
    fetch_atm,
    fetch_binance,
    fetch_cmapss,
    fetch_electricity,
    fetch_nonfinancial,
    fetch_traffic,
    fetch_weather_noaa,
)
from momo.filters import (
    AdaptiveCascadeFilter,
    CausalCascadeFilter,
    CausalMedianFilter,
    IdentityFilter,
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

# ---- lean causal filter set -----------------------------------------
LEAN_FILTERS = {
    "F0":  lambda: IdentityFilter(),
    "F4":  lambda: CausalMedianFilter(window=9),
    "FA":  lambda: OnlineAdaptiveFilter(window=9, k=3.0),
    "F10": lambda: CausalCascadeFilter(median_window=3,
                                       process_var=1e-3, obs_var=1.0),
    "F11": lambda: AdaptiveCascadeFilter(alpha_threshold=1.9,
                                         hurst_threshold=0.6,
                                         median_window=3,
                                         process_var=1e-3, obs_var=1.0),
}
OPTIMIZERS = ["sgd", "adam"]


class _LinAR:
    """Linear AR(p): regression vs the raw future, or sign classification."""

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
        z = Z @ x
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


def _diagnostics(series: np.ndarray) -> tuple[float, float]:
    diff = np.diff(series)
    alpha_hat = mcculloch_alpha(diff)
    if not np.isfinite(alpha_hat):
        alpha_hat = 2.0
    h_hat = hurst_dfa(diff)
    if not np.isfinite(h_hat):
        h_hat = 0.5
    return float(alpha_hat), float(h_hat)


def _logret(s: pd.Series) -> np.ndarray:
    s = pd.to_numeric(s, errors="coerce").replace(0.0, np.nan).dropna()
    return np.log(s).diff().dropna().to_numpy()


def _domains() -> dict[str, dict[str, np.ndarray]]:
    d: dict[str, dict[str, np.ndarray]] = {}

    # ---- financial: Binance crypto 1h log-returns -------------------
    fin = {}
    for sym in ("BTCUSDT", "ETHUSDT"):
        try:
            fin[sym] = _logret(fetch_binance(sym)["close"])
        except Exception:
            pass
    if fin:
        d["binance_crypto_1h"] = fin

    # ---- ETTm2 (electricity transformer temperature, 15-min) --------
    try:
        et = fetch_nonfinancial("ETTm2")
        d["sensor_ettm2"] = {c: et[c].to_numpy()[::4] for c in et.columns}
    except Exception:
        pass

    # ---- UCI/LSTNet electricity load --------------------------------
    try:
        el = fetch_electricity(n_series=6)
        d["electricity_load"] = {c: el[c].to_numpy() for c in el.columns}
    except Exception:
        pass

    # ---- LSTNet PEMS traffic occupancy ------------------------------
    try:
        tr = fetch_traffic(n_series=6)
        d["traffic_pems"] = {c: tr[c].to_numpy() for c in tr.columns}
    except Exception:
        pass

    # ---- NOAA GHCN-daily weather ------------------------------------
    try:
        wx = fetch_weather_noaa()
        d["noaa_weather"] = {c: wx[c].dropna().to_numpy() for c in wx.columns}
    except Exception:
        pass

    # ---- NASA C-MAPSS turbofan degradation --------------------------
    try:
        cm = fetch_cmapss()
        cols = list(cm.columns)[:8]
        d["cmapss_turbofan"] = {c: cm[c].dropna().to_numpy() for c in cols}
    except Exception:
        pass

    # ---- daily ATM cash withdrawals ---------------------------------
    try:
        atm = fetch_atm()
        d["atm_withdrawals"] = {"withdrawn": atm["withdrawn"].to_numpy()}
    except Exception:
        pass

    return d


def run_cell(domain, name, series, filt_key, opt, classification, seed):
    s = _standardize(series)
    if s.size < 6 * P + 80:
        return None
    cut = int(0.7 * s.size)
    alpha_hat, h_hat = _diagnostics(s[:cut])

    filt = LEAN_FILTERS[filt_key]()
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
    ap.add_argument("--min-conv", type=float, default=0.25,
                    help="drop a domain if its best (filter,opt,model) "
                         "convergence fraction is below this")
    ap.add_argument("--csv", type=Path, default=Path("tables/new_datasets.csv"))
    ap.add_argument("--summary-csv", type=Path,
                    default=Path("tables/new_datasets_summary.csv"))
    ap.add_argument("--criteria-csv", type=Path,
                    default=Path("tables/new_datasets_criteria.csv"))
    args = ap.parse_args()

    dom = _domains()
    print("domains built:", {k: len(v) for k, v in dom.items()})
    filters = list(LEAN_FILTERS)
    seeds = SEEDS
    models = [False, True]
    if args.smoke:
        dom = {k: {n: s for n, s in list(v.items())[:1]}
               for k, v in list(dom.items())}
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

    # ---- drop domains where nothing converges -----------------------
    best_conv = summ.groupby("domain")["conv_frac"].max()
    keep = set(best_conv[best_conv >= args.min_conv].index)
    dropped = sorted(set(best_conv.index) - keep)
    if dropped:
        print(f"DROPPED domains (best conv_frac < {args.min_conv}): {dropped}")
    summ = summ[summ["domain"].isin(keep)].copy()
    summ.to_csv(args.summary_csv, index=False)

    # ---- best filter per (domain, model, optimizer) -----------------
    rows = []
    for (dn, mdl, opt), sub in summ.groupby(["domain", "model", "optimizer"]):
        f0 = sub[sub["filter"] == "F0"]
        if f0.empty:
            continue
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
        print(f"  {r['domain']:20s} {r['model'][:5]:5s} {r['optimizer']:4s}  "
              f"alpha={r['alpha_hat']:.2f} H={r['hurst_hat']:.2f} "
              f"-> {r['best_filter']:4s} speedup {r['speedup']:6.1f}x "
              f"({r['bucket']})")
    print(f"\nwrote {args.csv}, {args.summary_csv}, {args.criteria_csv}")


if __name__ == "__main__":
    main()
