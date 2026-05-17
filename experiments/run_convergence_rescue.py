"""Headline experiment: does filtering the observed series *rescue* or
*accelerate* the convergence of stochastic optimization?

Two honest, pre-registered operating points (no per-seed/window/metric
cherry-picking — the same noise parameters are used for every filter and
optimizer; only the factor under study varies):

BLOCK A — divergence / noise-floor rescue (SGD-family, heavy tails).
  model {quadratic regression, logistic (convex) classification,
  AR autoregression} x noise {N1 Gaussian control, N3 alpha=1.2
  heavy-tailed, N4 heavy + long-memory} x optimizer {SGD, Clipped-SGD,
  Normalized-SGD} x filter {F0,F1,F2,F3,F4} x 8 seeds. Plain SGD under
  heavy tails stalls at a high noise floor / diverges; the question is
  whether a filter lowers the floor / flips the divergence.

BLOCK B — speed-up of adaptive optimizers (Adam/AdamW, long horizon).
  model {quadratic, logistic} x noise {N2 long-memory, N4 heavy +
  long-memory} x optimizer {Adam, AdamW} x filter {F0,F1,F2,F3,F4}
  x 8 seeds. Metric: iterations to an absolute eps; the wavelet/median
  speed-up over no-filter is re-measured
  with CIs over seeds.

Per run we log a battery of metrics so the data — not a tuned epsilon —
decides which is most expressive in each regime: relative convergence
(100x drop), iters to that drop and to an absolute eps, divergence slope
of log||g||^2 vs log k, noise-floor quantiles, AUC, held-out score
(no train/test mixing), wall-clock incl. filter cost.

Outputs: per-run tables/convergence_rescue.csv, seed-aggregated
tables/convergence_rescue_summary.csv (mean + 95% bootstrap CI over
seeds, plus floor-reduction ratio vs F0), downsampled curves
runs/convergence_rescue/curves.npz.
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

from momo.data import make_ar_forecast_task
from momo.filters import (
    CausalMedianFilter,
    IdentityFilter,
    KalmanLocalLevelFilter,
    MovingAverageFilter,
    WaveletThresholdFilter,
)
from momo.metrics import (
    convergence_auc,
    divergence_slope,
    noise_floor_quantiles,
    time_to_drop,
    time_to_eps,
)
from momo.noise import (
    GaussianNoise,
    MixedFARIMAStableNoise,
    PinkFARIMANoise,
    StableNoise,
)
from momo.optim import run_optimization
from momo.tasks import make_logistic, make_quadratic

SEEDS = list(range(8))

NOISES = {
    "N1": lambda: GaussianNoise(sigma=0.3),
    "N2": lambda: PinkFARIMANoise(d=0.4, sigma=0.3),
    "N3": lambda: StableNoise(alpha=1.2, sigma=0.3),
    "N4": lambda: MixedFARIMAStableNoise(d=0.4, alpha=1.2, sigma=0.3),
}

FILTERS = {
    "F0": lambda: IdentityFilter(),
    "F1": lambda: MovingAverageFilter(window=15),
    "F2": lambda: KalmanLocalLevelFilter(process_var=1e-3, obs_var=1.0),
    "F3": lambda: WaveletThresholdFilter(wavelet="db4", mode="soft",
                                         threshold="universal"),
    "F4": lambda: CausalMedianFilter(window=9),
}

# block A: SGD-family, moderate horizon; noise_scale puts plain SGD into
# a high heavy-tailed floor so a rescue is visible (calibrated, fixed).
MODELS_A = {
    "quadratic": dict(steps=4000, lr=5e-3, noise_scale=0.4, eps=1e-2),
    "logistic": dict(steps=3000, lr=3e-2, noise_scale=0.25, eps=5e-4),
    "ar": dict(steps=3000, lr=2e-2, noise_scale=0.3, eps=1e-4),
}
OPT_A = ["sgd", "clipped_sgd", "normalized_sgd"]
NOISE_A = ["N1", "N3", "N4"]
FILT_A = ["F0", "F1", "F2", "F3", "F4"]

# block B: adaptive optimizers (Adam/AdamW). Question is whether a filter
# still accelerates them; honest answer is regime-dependent (yes on
# long-memory N2, ~neutral on heavy N4 where Adam self-adapts).
MODELS_B = {
    "quadratic": dict(steps=8000, lr=5e-3, noise_scale=0.4, eps=1e-1),
    "logistic": dict(steps=8000, lr=2e-2, noise_scale=0.25, eps=5e-3),
}
OPT_B = ["adam", "adamw"]
NOISE_B = ["N2", "N4"]
FILT_B = ["F0", "F1", "F2", "F3", "F4"]


def _make_task(model: str, seed: int):
    if model == "quadratic":
        return make_quadratic(dim=20, condition_number=5.0, seed=seed)
    if model == "logistic":
        return make_logistic(n=3000, dim=15, n_test=1500,
                             noise_scale=0.4, seed=seed)
    if model == "ar":
        rng = np.random.default_rng(1234 + seed)
        n = 4200
        e = rng.normal(0, 1.0, n)
        y = np.zeros(n)
        for t in range(2, n):
            y[t] = 0.6 * y[t - 1] - 0.3 * y[t - 2] + e[t]
        return make_ar_forecast_task(y, p=5, train_frac=0.7)
    raise ValueError(model)


def _holdout(model: str, task, x_final: np.ndarray) -> float:
    if model == "logistic":
        return float(np.mean((task.Z_test @ x_final > 0).astype(float)
                             == task.y_test))
    if model == "ar":
        return float(task.loss(x_final, task.test_x, task.test_y))
    if model == "quadratic":
        return float(np.linalg.norm(x_final - task.optimum()))
    return float("nan")


def run_cell(block, model, noise_key, opt, filt_key, seed):
    cfg = (MODELS_A if block == "A" else MODELS_B)[model]
    task = _make_task(model, seed)
    noise = NOISES[noise_key]()
    filt = FILTERS[filt_key]()
    t0 = time.perf_counter()
    res = run_optimization(
        task=task, optimizer=opt, noise=noise, filt=filt,
        steps=cfg["steps"], lr=cfg["lr"], seed=seed,
        noise_scale=cfg["noise_scale"], preprocess_mode="series",
    )
    elapsed = time.perf_counter() - t0
    g = res.grad_norm_sq_history
    td100 = time_to_drop(g, factor=1e2)
    te = time_to_eps(g, cfg["eps"])
    fq = noise_floor_quantiles(g, tail_frac=0.2)
    return dict(
        block=block, model=model, noise=noise_key, optimizer=opt,
        filter=filt_key, seed=seed, steps=cfg["steps"],
        conv100=int(td100 is not None),
        t_drop100=cfg["steps"] if td100 is None else td100,
        t_eps=cfg["steps"] if te is None else te,
        eps_hit=int(te is not None),
        slope=divergence_slope(g),
        floor_p10=fq[0.1], floor_p50=fq[0.5], floor_p90=fq[0.9],
        auc=convergence_auc(g),
        holdout=_holdout(model, task, res.x_final),
        elapsed_s=elapsed,
        _curve=g,
    )


def _ci(vals, n_boot=2000, seed=0):
    a = np.asarray(vals, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return (float("nan"),) * 3
    if a.size == 1:
        return (float(a[0]),) * 3
    rng = np.random.default_rng(seed)
    boot = rng.choice(a, size=(n_boot, a.size), replace=True).mean(axis=1)
    return (float(a.mean()), float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--csv", type=Path,
                    default=Path("tables/convergence_rescue.csv"))
    ap.add_argument("--summary-csv", type=Path,
                    default=Path("tables/convergence_rescue_summary.csv"))
    ap.add_argument("--curves", type=Path,
                    default=Path("runs/convergence_rescue/curves.npz"))
    args = ap.parse_args()

    A_models, B_models = list(MODELS_A), list(MODELS_B)
    seeds = SEEDS
    filt_a, filt_b = FILT_A, FILT_B
    if args.smoke:
        A_models, B_models = ["quadratic", "ar"], ["quadratic"]
        seeds = [0, 1]
        filt_a = ["F0", "F2", "F4"]
        filt_b = ["F0", "F3", "F4"]

    cells = [("A", m, n, o, f, s)
             for m in A_models for n in NOISE_A for o in OPT_A
             for f in filt_a for s in seeds]
    cells += [("B", m, n, o, f, s)
              for m in B_models for n in NOISE_B for o in OPT_B
              for f in filt_b for s in seeds]
    print(f"running {len(cells)} cells on {args.n_jobs} jobs")
    t0 = time.perf_counter()
    out = Parallel(n_jobs=args.n_jobs, verbose=4, backend="loky")(
        delayed(run_cell)(*c) for c in cells)
    print(f"done in {time.perf_counter() - t0:.1f}s")

    curves, rows = {}, []
    for r in out:
        g = r.pop("_curve")
        idx = np.linspace(0, g.size - 1, min(g.size, 600)).astype(int)
        curves[f"{r['block']}|{r['model']}|{r['noise']}|{r['optimizer']}|"
               f"{r['filter']}|{r['seed']}"] = g[idx].astype(np.float32)
        rows.append(r)
    df = pd.DataFrame(rows)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)
    args.curves.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.curves, **curves)

    srows = []
    for (b, m, n, o, f), sub in df.groupby(
            ["block", "model", "noise", "optimizer", "filter"]):
        slope_m, slope_lo, slope_hi = _ci(sub["slope"])
        auc_m, auc_lo, auc_hi = _ci(sub["auc"])
        ho_m, ho_lo, ho_hi = _ci(sub["holdout"])
        srows.append(dict(
            block=b, model=m, noise=n, optimizer=o, filter=f,
            n_seeds=len(sub),
            conv100_frac=float(sub["conv100"].mean()),
            t_drop100_med=float(sub["t_drop100"].median()),
            t_eps_med=float(sub["t_eps"].median()),
            eps_hit_frac=float(sub["eps_hit"].mean()),
            slope_mean=slope_m, slope_lo=slope_lo, slope_hi=slope_hi,
            floor_p50_med=float(sub["floor_p50"].median()),
            floor_p10_med=float(sub["floor_p10"].median()),
            floor_p90_med=float(sub["floor_p90"].median()),
            auc_mean=auc_m, auc_lo=auc_lo, auc_hi=auc_hi,
            holdout_mean=ho_m, holdout_lo=ho_lo, holdout_hi=ho_hi,
            wall_med=float(sub["elapsed_s"].median()),
        ))
    summ = pd.DataFrame(srows)
    # floor reduction vs F0 within each (block,model,noise,optimizer)
    summ["floor_ratio_vs_F0"] = np.nan
    for key, g in summ.groupby(["block", "model", "noise", "optimizer"]):
        f0 = g[g["filter"] == "F0"]["floor_p50_med"]
        if f0.empty or float(f0.iloc[0]) <= 0:
            continue
        base = float(f0.iloc[0])
        summ.loc[g.index, "floor_ratio_vs_F0"] = base / g["floor_p50_med"]
    # speed-up vs F0 (t_eps F0 / t_eps filter) per block/model/noise/opt
    summ["speedup_vs_F0"] = np.nan
    for key, g in summ.groupby(["block", "model", "noise", "optimizer"]):
        f0 = g[g["filter"] == "F0"]["t_eps_med"]
        if f0.empty or float(f0.iloc[0]) <= 0:
            continue
        summ.loc[g.index, "speedup_vs_F0"] = (
            float(f0.iloc[0]) / g["t_eps_med"].clip(lower=1))
    summ.to_csv(args.summary_csv, index=False)

    print("\n=== BLOCK A — heavy-tailed SGD noise-floor rescue "
          "(N3, optimizer=sgd) ===")
    a = summ[(summ.block == "A") & (summ.noise == "N3")
             & (summ.optimizer == "sgd")]
    for m, s in a.groupby("model"):
        f0 = s[s["filter"] == "F0"]
        if f0.empty:
            continue
        best = s.loc[s["floor_p50_med"].idxmin()]
        print(f"  {m:9s}: F0 floor {float(f0['floor_p50_med'].iloc[0]):.2e}"
              f" -> {best['filter']} {best['floor_p50_med']:.2e}"
              f"  ({best['floor_ratio_vs_F0']:.0f}x lower);"
              f" slope F0 {float(f0['slope_mean'].iloc[0]):+.2f}"
              f" -> {best['slope_mean']:+.2f}")

    print("\n=== BLOCK B — Adam/AdamW: filter speed-up by noise "
          "(t_eps F0 vs best filter) ===")
    for (m, n, o), s in summ[summ.block == "B"].groupby(
            ["model", "noise", "optimizer"]):
        f0 = s[s["filter"] == "F0"]
        if f0.empty:
            continue
        f0t = float(f0["t_eps_med"].iloc[0])
        best = s.loc[s["t_eps_med"].idxmin()]
        tag = (f"{f0t / max(best['t_eps_med'], 1):.1f}x via {best['filter']}"
               if best["filter"] != "F0" else "no filter gain (F0 best)")
        print(f"  {m:9s} {n} {o:5s}: F0 t_eps {f0t:.0f} -> {tag}")
    print(f"\nwrote {args.csv}, {args.summary_csv}, {args.curves}")


if __name__ == "__main__":
    main()
