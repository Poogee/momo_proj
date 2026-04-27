from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from momo.clipping import AlphaAwareClipper
from momo.filters import (
    HybridMedianWaveletFilter,
    IdentityFilter,
    KalmanLocalLevelFilter,
    MedianFilter,
)
from momo.metrics import time_to_eps
from momo.noise import MixedFARIMAStableNoise, StableNoise
from momo.optim import run_optimization
from momo.tasks import make_logistic, make_quadratic


FILTERS = {
    "F0": lambda: IdentityFilter(),
    "F2": lambda: KalmanLocalLevelFilter(),
    "F4": lambda: MedianFilter(window=9),
    "F7": lambda: HybridMedianWaveletFilter(),
}

NOISES = {
    "N3": lambda: StableNoise(alpha=1.7, sigma=0.5),
    "N4": lambda: MixedFARIMAStableNoise(d=0.25, alpha=1.7, sigma=0.5),
}


def run_one(task_name, filt_name, noise_name, opt_name, clip_on, seed, steps, runs_dir):
    if task_name == "quadratic":
        task = make_quadratic(dim=50, condition_number=10.0, seed=seed)
        lr = 1e-2
    else:
        task = make_logistic(n=4000, dim=20, n_test=1000, noise_scale=0.5, seed=seed)
        lr = 5e-2
    filt = FILTERS[filt_name]()
    noise = NOISES[noise_name]()
    clipper = AlphaAwareClipper() if clip_on else None
    t0 = time.perf_counter()
    res = run_optimization(
        task=task, optimizer=opt_name, noise=noise, filt=filt,
        steps=steps, lr=lr, seed=seed, buffer_size=32,
        clipper=clipper,
    )
    elapsed = time.perf_counter() - t0
    g_curve = res.grad_norm_sq_history
    t_eps = time_to_eps(g_curve, 1e-2)
    cell_dir = runs_dir / task_name / f"{filt_name}_{noise_name}_{opt_name}_clip{int(clip_on)}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cell_dir / f"seed{seed:02d}.npz",
        grad_norm_sq=g_curve.astype(np.float32),
        loss=res.loss_history.astype(np.float32),
    )
    final_alpha = clipper.alpha_hat if clipper is not None else float("nan")
    return dict(
        task=task_name, filter=filt_name, noise=noise_name,
        optimizer=opt_name, clip=clip_on, seed=seed,
        t_eps=(-1 if t_eps is None else t_eps),
        final_grad_norm_sq=float(g_curve[-1]),
        final_loss=float(res.loss_history[-1]),
        final_alpha_hat=final_alpha,
        elapsed_s=elapsed,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/clipping"))
    parser.add_argument("--summary-csv", type=Path, default=Path("tables/clipping_summary.csv"))
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--n-seeds", type=int, default=8)
    args = parser.parse_args()

    args.runs_dir.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)

    cells = []
    for task_name in ["quadratic", "logistic"]:
        for filt_name in FILTERS:
            for noise_name in NOISES:
                for opt_name in ["sgd", "adam"]:
                    for clip_on in [False, True]:
                        for seed in range(args.n_seeds):
                            cells.append((task_name, filt_name, noise_name,
                                          opt_name, clip_on, seed))
    print(f"Running {len(cells)} cells")
    t0 = time.perf_counter()
    rows = Parallel(n_jobs=args.n_jobs, verbose=5, backend="loky")(
        delayed(run_one)(tn, fn, nn, on, ck, sd, args.steps, args.runs_dir)
        for (tn, fn, nn, on, ck, sd) in cells
    )
    elapsed = time.perf_counter() - t0
    df = pd.DataFrame(rows)
    df.to_csv(args.summary_csv, index=False)
    print(f"Done in {elapsed:.1f}s. Wrote {args.summary_csv}")
    print("\nFinal grad-norm^2 (median) by (filter, noise, optimizer, clip):")
    print(df.groupby(["task", "filter", "noise", "optimizer", "clip"])["final_grad_norm_sq"].median().unstack(["clip"]).round(3))


if __name__ == "__main__":
    main()
