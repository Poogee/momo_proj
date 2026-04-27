from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from momo.filters import (
    AdaptiveMetaFilter,
    HybridMedianWaveletFilter,
    IdentityFilter,
    KalmanLocalLevelFilter,
    MedianFilter,
)
from momo.noise import GaussianNoise, StableNoise
from momo.optim import run_optimization
from momo.tasks import QuadraticTask


def make_2d_quadratic(condition: float = 8.0, seed: int = 0) -> QuadraticTask:
    eigs = np.array([1.0, 1.0 / condition])
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((2, 2)))
    A = Q @ np.diag(eigs) @ Q.T
    b = rng.standard_normal(2) * 0.5
    return QuadraticTask(A=A, b=b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("figures/2d_trajectory.pdf"))
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sns.set_theme(context="paper", style="white", font_scale=0.95)
    task = make_2d_quadratic(condition=8.0, seed=args.seed)
    x_star = task.optimum()
    grid = np.linspace(-3.0, 3.0, 80)
    X, Y = np.meshgrid(grid, grid)
    XY = np.stack([X.ravel(), Y.ravel()], axis=1)
    losses = np.array([task.loss(p) for p in XY]).reshape(X.shape)

    cases = [
        ("N1 + F0", GaussianNoise(0.5), IdentityFilter()),
        ("N1 + F2 Kalman", GaussianNoise(0.5), KalmanLocalLevelFilter()),
        ("N1 + F4 Median", GaussianNoise(0.5), MedianFilter(window=9)),
        ("N3 + F0", StableNoise(alpha=1.7, sigma=0.5), IdentityFilter()),
        ("N3 + F4 Median", StableNoise(alpha=1.7, sigma=0.5), MedianFilter(window=9)),
        ("N3 + F7 Hybrid", StableNoise(alpha=1.7, sigma=0.5), HybridMedianWaveletFilter()),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
    for ax, (label, noise, filt) in zip(axes.flatten(), cases):
        cs = ax.contour(X, Y, np.log(np.maximum(losses - losses.min() + 1e-3, 1e-3)),
                        levels=12, cmap="viridis", alpha=0.5)
        ax.plot(x_star[0], x_star[1], "k*", markersize=10, label=r"$x^*$")
        for seed in range(3):
            res = run_optimization(
                task=task, optimizer="adam", noise=noise, filt=filt,
                steps=args.steps, lr=5e-2, seed=args.seed + seed * 100,
                buffer_size=20,
            )
            xs = res.x_history
            ax.plot(xs[:, 0], xs[:, 1], lw=0.8, alpha=0.85)
            ax.plot(xs[-1, 0], xs[-1, 1], "o", markersize=4)
        ax.set_title(label, fontsize=11)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.grid(alpha=0.2)

    fig.suptitle("Траектории Adam в 2D-квадратичной задаче (3 запуска)")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
