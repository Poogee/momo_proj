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

from momo.filters import (
    AdaptiveMetaFilter,
    HybridMedianWaveletFilter,
    IdentityFilter,
    KalmanLocalLevelFilter,
    MedianFilter,
)
from momo.noise import StableNoise
from momo.optim import run_optimization
from momo.tasks import _sigmoid, make_logistic


FILTERS = {
    "F0": IdentityFilter(),
    "F2": KalmanLocalLevelFilter(),
    "F4": MedianFilter(window=15),
    "F7": HybridMedianWaveletFilter(median_window=7),
    "F8": AdaptiveMetaFilter(),
}

OPTS = ["sgd", "clipped_sgd", "normalized_sgd", "adam", "adamw"]
ALPHAS = [1.1, 1.3, 1.5, 1.7, 1.9]


def run_one(alpha, fk, opt, seed):
    task = make_logistic(n=4000, dim=20, n_test=1000, noise_scale=0.5, seed=seed)
    res = run_optimization(
        task=task, optimizer=opt, noise=StableNoise(alpha=alpha, sigma=0.5),
        filt=FILTERS[fk], steps=1500, lr=5e-2, seed=seed,
    )
    preds = (_sigmoid(task.Z_test @ res.x_final) >= 0.5).astype(int)
    return dict(alpha=alpha, filter=fk, optimizer=opt, seed=seed,
                final_grad=float(res.grad_norm_sq_history[-1]),
                holdout_acc=float(np.mean(preds == task.y_test)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-csv", type=Path, default=Path("tables/stress_test.csv"))
    parser.add_argument("--out-fig", type=Path, default=Path("figures/stress_test.pdf"))
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--n-seeds", type=int, default=4)
    args = parser.parse_args()

    cells = [(a, fk, op, sd) for a in ALPHAS for fk in FILTERS for op in OPTS for sd in range(args.n_seeds)]
    print(f"Running {len(cells)} cells")
    t0 = time.perf_counter()
    rows = Parallel(n_jobs=args.n_jobs, verbose=5, backend="loky")(
        delayed(run_one)(*c) for c in cells
    )
    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Done in {time.perf_counter()-t0:.1f}s")

    sns.set_theme(context="paper", style="whitegrid", font_scale=0.95)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey="row")
    flat = axes.flatten()
    for ax, opt in zip(flat[:5], OPTS):
        sub = df[df["optimizer"] == opt].groupby(["filter", "alpha"])["holdout_acc"].mean().reset_index()
        for fk in FILTERS:
            view = sub[sub["filter"] == fk]
            ax.plot(view["alpha"], view["holdout_acc"], marker="o", lw=1.5, label=fk)
        ax.set_title(opt)
        ax.set_xlabel(r"$\alpha$ (heavy-tail index)")
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
    flat[0].set_ylabel("test accuracy")
    flat[0].legend(loc="lower left", fontsize=9, frameon=False)
    flat[5].axis("off")
    fig.suptitle("Stress test: holdout accuracy vs heavy-tail index α (lower α = heavier tails)")
    fig.tight_layout()
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, dpi=150)
    plt.close(fig)
    print(f"Wrote {args.out_fig}")
    print()
    print(df.groupby(["alpha", "filter", "optimizer"])["holdout_acc"].mean().unstack(["optimizer"]).round(3))


if __name__ == "__main__":
    main()
