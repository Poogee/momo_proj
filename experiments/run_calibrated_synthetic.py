"""Does the heavy-tailed noise-floor rescue survive when the synthetic
noise is *calibrated to real data* (not hand-picked alpha)?

Step 1 — diagnose the tail index alpha-hat and the long-memory parameter
d-hat = clip(H-0.5) on several genuinely real series (financial daily
basket, 5-min intraday, the non-financial ETT sensor). All causal /
in-sample diagnostics, no look-ahead.

Step 2 — build StableNoise(alpha-hat) and MixedFARIMAStableNoise(d-hat,
alpha-hat) at the *median* diagnosed values and re-run the SGD floor
rescue (quadratic / logistic / AR x F0 vs F4 vs F7, 8 seeds). If the
1-2 order-of-magnitude floor reduction persists at realistically
calibrated tails, the effect is not an artifact of a convenient alpha.

Outputs tables/calibrated_synthetic.csv and prints the calibration.
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
    DEFAULT_TICKERS,
    fetch_intraday,
    fetch_nonfinancial,
    fetch_returns,
    make_ar_forecast_task,
)
from momo.filters import CausalMedianFilter, HybridMedianWaveletFilter, IdentityFilter
from momo.metrics import (
    convergence_auc,
    divergence_slope,
    hurst_dfa,
    mcculloch_alpha,
    noise_floor_quantiles,
    time_to_drop,
)
from momo.noise import MixedFARIMAStableNoise, StableNoise
from momo.optim import run_optimization
from momo.tasks import make_logistic, make_quadratic

SEEDS = list(range(8))


def _diagnose() -> pd.DataFrame:
    rows = []
    # genuinely real & sandbox-verified domains
    fin = fetch_returns(sum(DEFAULT_TICKERS.values(), []))
    for c in fin.columns:
        s = fin[c].dropna().to_numpy()
        if s.size > 400:
            rows.append(("financial_daily", c, s))
    intr = fetch_intraday(sum(DEFAULT_TICKERS.values(), []), interval="5m")
    r5 = np.log(intr / intr.shift(1)).dropna(how="all")
    for c in r5.columns[:4]:
        s = r5[c].dropna().to_numpy()
        if s.size > 400:
            rows.append(("financial_5m", c, s))
    ett = fetch_nonfinancial()
    rows.append(("nonfinancial_ett", "OT", np.diff(ett["OT"].to_numpy())))

    out = []
    for domain, name, s in rows:
        s = s[np.isfinite(s)]
        a = mcculloch_alpha(s)
        h_abs = hurst_dfa(np.abs(s))
        out.append(dict(domain=domain, series=name, n=s.size,
                        alpha_hat=a, hurst_abs=h_abs,
                        d_hat=float(np.clip((h_abs - 0.5), 0.0, 0.49))
                        if np.isfinite(h_abs) else np.nan))
    return pd.DataFrame(out)


def _make_task(model, seed):
    if model == "quadratic":
        return make_quadratic(dim=20, condition_number=5.0, seed=seed)
    if model == "logistic":
        return make_logistic(n=3000, dim=15, n_test=1500,
                             noise_scale=0.4, seed=seed)
    rng = np.random.default_rng(1234 + seed)
    n = 4200
    e = rng.normal(0, 1.0, n)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = 0.6 * y[t - 1] - 0.3 * y[t - 2] + e[t]
    return make_ar_forecast_task(y, p=5, train_frac=0.7)


CFG = {"quadratic": dict(steps=4000, lr=5e-3, sc=0.4),
       "logistic": dict(steps=3000, lr=3e-2, sc=0.25),
       "ar": dict(steps=3000, lr=2e-2, sc=0.3)}


def run_cell(model, noise_key, alpha, d, filt_key, seed):
    c = CFG[model]
    task = _make_task(model, seed)
    noise = (StableNoise(alpha=alpha, sigma=0.3) if noise_key == "N3cal"
             else MixedFARIMAStableNoise(d=d, alpha=alpha, sigma=0.3))
    filt = {"F0": IdentityFilter(),
            "F4": CausalMedianFilter(window=9),
            "F7": HybridMedianWaveletFilter(median_window=5)}[filt_key]
    res = run_optimization(task=task, optimizer="sgd", noise=noise,
                           filt=filt, steps=c["steps"], lr=c["lr"],
                           seed=seed, noise_scale=c["sc"],
                           preprocess_mode="series")
    g = res.grad_norm_sq_history
    td = time_to_drop(g, factor=1e2)
    return dict(model=model, noise=noise_key, alpha=round(alpha, 3),
                d=round(d, 3), filter=filt_key, seed=seed,
                conv100=int(td is not None),
                slope=divergence_slope(g),
                floor_p50=noise_floor_quantiles(g)[0.5],
                auc=convergence_auc(g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=10)
    ap.add_argument("--csv", type=Path,
                    default=Path("tables/calibrated_synthetic.csv"))
    ap.add_argument("--diag-csv", type=Path,
                    default=Path("tables/real_noise_calibration.csv"))
    args = ap.parse_args()

    diag = _diagnose()
    args.diag_csv.parent.mkdir(parents=True, exist_ok=True)
    diag.to_csv(args.diag_csv, index=False)
    print("real-data noise diagnostics:")
    print(diag.round(3).to_string(index=False))

    a_med = float(np.nanmedian(diag["alpha_hat"]))
    d_med = float(np.nanmedian(diag["d_hat"]))
    a_med = float(np.clip(a_med, 1.05, 1.95))
    d_med = float(np.clip(d_med, 0.05, 0.49))
    print(f"\ncalibrated: alpha-hat (median) = {a_med:.3f},  "
          f"d-hat (median) = {d_med:.3f}")

    cells = [(m, nk, a_med, d_med, f, s)
             for m in CFG for nk in ("N3cal", "N4cal")
             for f in ("F0", "F4", "F7") for s in SEEDS]
    out = Parallel(n_jobs=args.n_jobs, verbose=3, backend="loky")(
        delayed(run_cell)(*c) for c in cells)
    df = pd.DataFrame(out)
    df.to_csv(args.csv, index=False)

    print("\n=== calibrated heavy-tailed SGD floor: F0 vs F4/F7 "
          f"(alpha={a_med:.2f}, d={d_med:.2f}) ===")
    for (m, nk), s in df.groupby(["model", "noise"]):
        piv = s.groupby("filter")["floor_p50"].median()
        f0 = float(piv.get("F0", np.nan))
        for fk in ("F4", "F7"):
            if fk in piv and piv[fk] > 0:
                print(f"  {m:9s} {nk}: F0 {f0:.2e} -> {fk} "
                      f"{piv[fk]:.2e}  ({f0 / piv[fk]:.0f}x lower floor)")
    print(f"\nwrote {args.csv}, {args.diag_csv}")


if __name__ == "__main__":
    main()
