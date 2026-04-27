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
from momo.noise import GaussianNoise, MixedFARIMAStableNoise, StableNoise
from momo.optim import run_optimization
from momo.tasks import _sigmoid, make_logistic


FILTERS = {
    "F0": lambda: IdentityFilter(),
    "F2": lambda: KalmanLocalLevelFilter(),
    "F4": lambda: MedianFilter(window=9),
    "F7": lambda: HybridMedianWaveletFilter(),
    "F8": lambda: AdaptiveMetaFilter(),
}


def run_one(noise_kind: str, sigma: float, filt_name: str, opt_name: str, seed: int):
    if noise_kind == "N1":
        noise = GaussianNoise(sigma=sigma)
    elif noise_kind == "N3":
        noise = StableNoise(alpha=1.7, sigma=sigma)
    elif noise_kind == "N4":
        noise = MixedFARIMAStableNoise(d=0.25, alpha=1.7, sigma=sigma)
    else:
        raise ValueError(noise_kind)
    task = make_logistic(n=4000, dim=20, n_test=1000, noise_scale=0.5, seed=seed)
    filt = FILTERS[filt_name]()
    res = run_optimization(
        task=task, optimizer=opt_name, noise=noise, filt=filt,
        steps=1500, lr=5e-2, seed=seed, batch_size=32,
    )
    preds = (_sigmoid(task.Z_test @ res.x_final) >= 0.5).astype(int)
    acc = float(np.mean(preds == task.y_test))
    return dict(
        noise=noise_kind, sigma=sigma, filter=filt_name, optimizer=opt_name, seed=seed,
        final_grad_norm_sq=float(res.grad_norm_sq_history[-1]),
        holdout_acc=acc,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("tables/sigma_scan.csv"))
    parser.add_argument("--n-jobs", type=int, default=12)
    args = parser.parse_args()

    sigmas = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
    cells = []
    for noise in ["N1", "N3"]:
        for sigma in sigmas:
            for filt in FILTERS:
                for opt in ["sgd", "adam"]:
                    for seed in range(4):
                        cells.append((noise, sigma, filt, opt, seed))
    print(f"Running {len(cells)} cells")
    t0 = time.perf_counter()
    rows = Parallel(n_jobs=args.n_jobs, verbose=5, backend="loky")(
        delayed(run_one)(*c) for c in cells
    )
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Done in {time.perf_counter()-t0:.1f}s. Wrote {args.out}")

    sns.set_theme(context="paper", style="whitegrid", font_scale=0.95)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, noise in zip(axes, ["N1", "N3"]):
        sub = df[(df["noise"] == noise) & (df["optimizer"] == "adam")]
        agg = sub.groupby(["filter", "sigma"])["holdout_acc"].mean().reset_index()
        for filt in FILTERS:
            view = agg[agg["filter"] == filt]
            ax.plot(view["sigma"], view["holdout_acc"], marker="o", label=filt, lw=1.5)
        ax.set_xlabel(r"$\sigma$ noise scale")
        ax.set_title(f"{noise}: holdout acc vs $\\sigma$ (Adam)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("holdout accuracy")
    axes[0].legend(loc="lower left", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig("figures/sigma_scan.pdf", dpi=150)
    plt.close(fig)
    print("wrote figures/sigma_scan.pdf")


if __name__ == "__main__":
    main()
