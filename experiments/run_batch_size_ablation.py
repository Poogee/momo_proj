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
    MedianFilter,
)
from momo.noise import StableNoise
from momo.optim import run_optimization
from momo.tasks import _sigmoid, make_logistic


FILTERS = {
    "F0": lambda: IdentityFilter(),
    "F4": lambda: MedianFilter(window=9),
    "F7": lambda: HybridMedianWaveletFilter(),
    "F8": lambda: AdaptiveMetaFilter(),
}


def run(batch, filt, opt, seed):
    task = make_logistic(n=4000, dim=20, n_test=1000, noise_scale=0.5, seed=seed)
    res = run_optimization(
        task=task, optimizer=opt, noise=StableNoise(alpha=1.7, sigma=0.5),
        filt=FILTERS[filt](), steps=1500, lr=5e-2, seed=seed, batch_size=batch,
    )
    preds = (_sigmoid(task.Z_test @ res.x_final) >= 0.5).astype(int)
    return dict(
        batch=batch, filter=filt, optimizer=opt, seed=seed,
        final_grad_norm_sq=float(res.grad_norm_sq_history[-1]),
        holdout_acc=float(np.mean(preds == task.y_test)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-csv", type=Path, default=Path("tables/batch_size_ablation.csv"))
    parser.add_argument("--out-fig", type=Path, default=Path("figures/batch_size_ablation.pdf"))
    parser.add_argument("--n-jobs", type=int, default=12)
    args = parser.parse_args()

    batches = [4, 8, 16, 32, 64, 128, 256]
    cells = [(b, f, o, s) for b in batches for f in FILTERS for o in ["sgd", "adam"] for s in range(4)]
    print(f"Running {len(cells)} cells")
    t0 = time.perf_counter()
    rows = Parallel(n_jobs=args.n_jobs, verbose=5, backend="loky")(
        delayed(run)(*c) for c in cells
    )
    print(f"done in {time.perf_counter()-t0:.1f}s")
    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    sns.set_theme(context="paper", style="whitegrid", font_scale=0.95)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, opt in zip(axes, ["sgd", "adam"]):
        sub = df[df["optimizer"] == opt].groupby(["filter", "batch"])["holdout_acc"].mean().reset_index()
        for filt in FILTERS:
            view = sub[sub["filter"] == filt]
            ax.plot(view["batch"], view["holdout_acc"], marker="o", lw=1.5, label=filt)
        ax.set_xscale("log")
        ax.set_xlabel("batch size")
        ax.set_title(f"{opt}, шум N3 (α=1.7)")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("holdout accuracy")
    axes[0].legend(fontsize=9, frameon=False)
    fig.tight_layout()
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, dpi=150)
    plt.close(fig)

    print("\n=== Holdout accuracy (Adam, mean), N3 ===")
    sub = df[df["optimizer"] == "adam"]
    print(sub.groupby(["filter", "batch"])["holdout_acc"].mean().unstack().round(3))


if __name__ == "__main__":
    main()
