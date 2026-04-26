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

FILTER_ORDER = ["F0", "F1", "F2", "F3", "F4"]
OPT_ORDER = ["sgd", "adam", "adamw"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, default=Path("tables/real_summary.csv"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    df = pd.read_csv(args.summary_csv)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(context="paper", style="whitegrid", font_scale=0.95)

    pivot_mse = df.groupby(["filter", "optimizer"])["holdout_mse_raw"].mean().unstack().reindex(
        index=FILTER_ORDER, columns=OPT_ORDER
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(pivot_mse * 1e4, annot=True, fmt=".2f", cmap="viridis_r", ax=ax,
                cbar_kws={"label": "MSE × 10⁴"})
    ax.set_title("Реальные данные: holdout MSE по парам (фильтр, оптимизатор)")
    fig.tight_layout()
    fig.savefig(args.figures_dir / "real_holdout_mse.pdf", dpi=150)
    plt.close(fig)

    pivot_h = df.groupby("filter")[["hurst_raw", "hurst_smoothed"]].mean().reindex(FILTER_ORDER)
    pivot_a = df.groupby("filter")[["alpha_raw", "alpha_smoothed"]].mean().reindex(FILTER_ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    pivot_h.plot.bar(ax=axes[0], rot=0)
    axes[0].axhline(0.5, color="gray", lw=1, ls="--")
    axes[0].set_title("Хёрст до / после фильтрации (среднее по корзине)")
    axes[0].set_ylabel(r"$\hat{H}$")
    axes[0].legend(["до", "после"])
    pivot_a.plot.bar(ax=axes[1], rot=0)
    axes[1].axhline(2.0, color="gray", lw=1, ls="--")
    axes[1].set_title("Хвостовой индекс α до / после")
    axes[1].set_ylabel(r"$\hat{\alpha}$")
    axes[1].legend(["до", "после"])
    fig.tight_layout()
    fig.savefig(args.figures_dir / "real_hurst_alpha.pdf", dpi=150)
    plt.close(fig)

    pivot_snr = df.groupby(["ticker", "filter"])["snr_filter_db"].mean().unstack().reindex(
        columns=FILTER_ORDER
    )
    pivot_snr = pivot_snr.replace([np.inf, -np.inf], np.nan)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot_snr, annot=True, fmt=".1f", cmap="RdYlGn", center=0, ax=ax)
    ax.set_title("SNR (дБ) сглаженной серии относительно сырой (реальные данные)")
    fig.tight_layout()
    fig.savefig(args.figures_dir / "real_filter_snr.pdf", dpi=150)
    plt.close(fig)

    print(f"Wrote real-data figures to {args.figures_dir}")


if __name__ == "__main__":
    main()
