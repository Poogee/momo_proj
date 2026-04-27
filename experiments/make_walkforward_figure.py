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


FILTER_ORDER = ["F0", "F1", "F2", "F3", "F4", "F6", "F7"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, default=Path("tables/real_walkforward_summary.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    df = pd.read_csv(args.summary_csv)
    sns.set_theme(context="paper", style="whitegrid", font_scale=0.95)

    pivot = df[df["optimizer"] == "adam"].groupby(["ticker", "filter"])["holdout_mse_raw"].mean().unstack().reindex(columns=FILTER_ORDER)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot * 1e4, annot=True, fmt=".2f", cmap="viridis_r", ax=ax,
                cbar_kws={"label": "Walk-forward holdout MSE × 10⁴"})
    ax.set_title("Walk-forward holdout MSE (Adam) по тикерам и фильтрам")
    fig.tight_layout()
    fig.savefig(args.out_dir / "real_walkforward_mse.pdf", dpi=150)
    plt.close(fig)

    rel = pivot.div(pivot["F0"], axis=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    rel.drop(columns=["F0"]).plot.bar(ax=ax)
    ax.axhline(1.0, color="black", lw=0.8, ls="--")
    ax.set_ylabel(r"Holdout MSE / MSE(F0)")
    ax.set_title("Относительный ущерб от фильтрации (Adam, walk-forward)")
    ax.legend(title="Фильтр", bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    fig.savefig(args.out_dir / "real_walkforward_relative.pdf", dpi=150)
    plt.close(fig)
    print("wrote walk-forward figures")


if __name__ == "__main__":
    main()
