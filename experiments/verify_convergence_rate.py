from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy.stats import linregress

from momo.filters import IdentityFilter, MedianFilter, HybridMedianWaveletFilter
from momo.noise import StableNoise
from momo.optim import run_optimization
from momo.tasks import make_quadratic


def run(alpha, filt_kind, optimizer, seed, steps):
    task = make_quadratic(dim=20, condition_number=5.0, seed=seed)
    noise = StableNoise(alpha=alpha, sigma=0.3)
    if filt_kind == "F0":
        filt = IdentityFilter()
    elif filt_kind == "F4":
        filt = MedianFilter(window=9)
    else:
        filt = HybridMedianWaveletFilter()
    res = run_optimization(
        task=task, optimizer=optimizer, noise=noise, filt=filt,
        steps=steps, lr=5e-3, seed=seed,
    )
    return alpha, filt_kind, optimizer, seed, res.grad_norm_sq_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--out-fig", type=Path, default=Path("figures/convergence_rate_alpha.pdf"))
    parser.add_argument("--out-csv", type=Path, default=Path("tables/convergence_rates.csv"))
    parser.add_argument("--n-jobs", type=int, default=12)
    args = parser.parse_args()

    alphas = [1.2, 1.4, 1.6, 1.8, 2.0]
    cells = [(a, fk, opt, s) for a in alphas for fk in ["F0", "F4", "F7"]
             for opt in ["sgd", "clipped_sgd", "normalized_sgd"]
             for s in range(args.n_seeds)]
    print(f"Running {len(cells)} cells")
    t0 = time.perf_counter()
    out = Parallel(n_jobs=args.n_jobs, verbose=5, backend="loky")(
        delayed(run)(a, fk, opt, s, args.steps) for (a, fk, opt, s) in cells
    )
    print(f"done in {time.perf_counter()-t0:.1f}s")

    sns.set_theme(context="paper", style="whitegrid", font_scale=0.95)
    by_cell = {}
    for a, fk, opt, s, curve in out:
        by_cell.setdefault((a, fk, opt), []).append(curve)

    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, opt in zip(axes, ["sgd", "clipped_sgd", "normalized_sgd"]):
        for fk, color in zip(["F0", "F4", "F7"], ["C0", "C1", "C2"]):
            for a, ls in zip(alphas, ["-", "--", "-.", ":", (0, (1, 1))]):
                arr = np.stack(by_cell[(a, fk, opt)])
                med = np.median(arr, axis=0)
                tail_start = int(0.3 * args.steps)
                k = np.arange(tail_start, args.steps)
                logk = np.log(k + 1)
                logy = np.log(np.maximum(med[tail_start:], 1e-12))
                slope, _, _, _, _ = linregress(logk, logy)
                rows.append(dict(alpha=a, filter=fk, optimizer=opt, slope=slope))
                if a in (1.2, 2.0):
                    ax.plot(np.arange(args.steps), med, color=color, ls=ls, lw=1.0, alpha=0.85,
                            label=f"{fk}, α={a}" if opt == "sgd" else None)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_title(f"{opt}")
        ax.set_xlabel("итерация k")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel(r"$\|\nabla f(x_k)\|^2$ (медиана)")
    axes[0].legend(fontsize=7, loc="lower left", ncol=2)
    fig.suptitle("Сходимость на квадратичной задаче при разных α (heavy-tail) и фильтрах")
    fig.tight_layout()
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, dpi=150)
    plt.close(fig)
    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print("Slopes (log||g||^2 vs log k) — closer to 0 = floor; more negative = faster convergence:")
    print(df.groupby(["optimizer", "filter", "alpha"])["slope"].mean().unstack(["alpha"]).round(2))


if __name__ == "__main__":
    main()
