from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_curve(runs_dir: Path, task: str, filt: str, noise: str, opt: str, clip: int) -> np.ndarray:
    cell = runs_dir / task / f"{filt}_{noise}_{opt}_clip{clip}"
    seeds = []
    for f in sorted(cell.glob("seed*.npz")):
        arr = np.load(f)
        seeds.append(arr["grad_norm_sq"])
    if not seeds:
        return None
    return np.stack(seeds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/clipping"))
    parser.add_argument("--summary-csv", type=Path, default=Path("tables/clipping_summary.csv"))
    parser.add_argument("--out", type=Path, default=Path("figures/clipping_ablation.pdf"))
    args = parser.parse_args()

    sns.set_theme(context="paper", style="whitegrid", font_scale=0.95)
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharex=True, sharey=True)
    pairs = [
        ("logistic", "F0", "N3"), ("logistic", "F2", "N3"),
        ("logistic", "F4", "N3"), ("logistic", "F7", "N3"),
        ("logistic", "F0", "N4"), ("logistic", "F2", "N4"),
        ("logistic", "F4", "N4"), ("logistic", "F7", "N4"),
    ]
    for ax, (task, filt, noise) in zip(axes.flatten(), pairs):
        for opt, ls in [("sgd", "-"), ("adam", "--")]:
            for clip, color, label in [(0, "C3", "no clip"), (1, "C2", "α-clip")]:
                arr = load_curve(args.runs_dir, task, filt, noise, opt, clip)
                if arr is None:
                    continue
                med = np.median(arr, axis=0)
                p10 = np.quantile(arr, 0.10, axis=0)
                p90 = np.quantile(arr, 0.90, axis=0)
                x = np.arange(med.size)
                ax.plot(x, med, ls=ls, lw=1.5, color=color,
                        label=f"{opt.upper()} {label}")
                ax.fill_between(x, p10, p90, alpha=0.10, color=color)
        ax.set_yscale("log")
        ax.set_title(f"{filt} · {noise}")
        ax.grid(alpha=0.3, which="both")
    axes[0, 0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[0, 0].set_ylabel("logistic\n" + r"$\|\nabla f\|^2$")
    axes[1, 0].set_ylabel("logistic\n" + r"$\|\nabla f\|^2$")
    for ax in axes[1]:
        ax.set_xlabel("итерация")
    fig.suptitle("Adaptive-α clipping on heavy-tailed noise (N3 верх, N4 низ)")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
