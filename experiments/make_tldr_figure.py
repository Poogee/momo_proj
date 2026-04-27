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


FILTER_ORDER = ["F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("figures/tldr_summary.pdf"))
    args = parser.parse_args()

    sns.set_theme(context="paper", style="whitegrid", font_scale=0.95)

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.4, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0:2])
    snr_csv = Path("tables/master_snr_table.csv")
    if snr_csv.exists():
        snr = pd.read_csv(snr_csv)
        pivot = snr.pivot(index="filter", columns="noise", values="delta_snr_db")
        pivot = pivot.reindex(index=FILTER_ORDER)
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", center=8, ax=ax1,
                    cbar_kws={"label": "ΔSNR (dB)"})
        ax1.set_title("(a) Synthetic SNR gain: F0–F9 × N1–N6 (10 seeds, T=4096)")
        ax1.set_xlabel("noise")
        ax1.set_ylabel("filter")

    ax2 = fig.add_subplot(gs[0, 2])
    real_csv = Path("tables/real_walkforward_full.csv")
    if real_csv.exists():
        real = pd.read_csv(real_csv)
        rmse = real[real["optimizer"] == "adam"].groupby("filter")["holdout_mse_raw"].mean() * 1e4
        rmse = rmse.reindex(FILTER_ORDER).dropna()
        bars = ax2.bar(rmse.index, rmse.values, color="steelblue")
        baseline = rmse["F0"]
        ax2.axhline(baseline, color="green", lw=1, ls="--", label=f"F0 = {baseline:.2f}")
        ax2.set_title("(b) Real-data: holdout MSE (Adam)\n— filtering hurts")
        ax2.set_ylabel("MSE × 10⁴")
        ax2.set_xlabel("filter")
        ax2.tick_params(axis="x", rotation=45)
        ax2.legend(fontsize=8, frameon=False)

    ax3 = fig.add_subplot(gs[1, 0])
    sigma_csv = Path("tables/sigma_scan.csv")
    if sigma_csv.exists():
        sigma_df = pd.read_csv(sigma_csv)
        sub = sigma_df[(sigma_df["noise"] == "N3") & (sigma_df["optimizer"] == "adam")]
        agg = sub.groupby(["filter", "sigma"])["holdout_acc"].mean().unstack()
        for fn in agg.index:
            ax3.plot(agg.columns, agg.loc[fn], marker="o", label=fn, lw=1.5)
        ax3.set_title("(c) σ-scan on α-stable noise\n— filters help more at high σ")
        ax3.set_xlabel("σ")
        ax3.set_ylabel("holdout accuracy (logistic)")
        ax3.legend(fontsize=8, frameon=False, loc="lower left")
        ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    rate_csv = Path("tables/convergence_rates.csv")
    if rate_csv.exists():
        rates = pd.read_csv(rate_csv)
        sub = rates[rates["filter"] == "F0"].groupby(["optimizer", "alpha"])["slope"].mean().unstack()
        for opt in sub.index:
            ax4.plot(sub.columns, sub.loc[opt], marker="o", label=opt, lw=1.5)
        ax4.axhline(0, color="black", lw=0.5)
        ax4.set_title("(d) F0 convergence slope vs α\n— SGD diverges at α=1.2")
        ax4.set_xlabel("α (heavy-tail index)")
        ax4.invert_xaxis()
        ax4.set_ylabel("slope log‖g‖² vs log k")
        ax4.legend(fontsize=8, frameon=False, loc="upper right")
        ax4.grid(alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 2])
    causal_csv = Path("tables/signpred_causal.csv")
    if causal_csv.exists():
        causal = pd.read_csv(causal_csv)
        centered_acc = causal[causal["kind"] == "centered"].groupby("window")["acc"].mean()
        causal_acc = causal[causal["kind"] == "causal"].groupby("window")["acc"].mean()
        identity_acc = causal[causal["kind"] == "identity"]["acc"].mean()
        ax5.plot(centered_acc.index, centered_acc.values, marker="o", lw=2, label="centered (lookahead)")
        ax5.plot(causal_acc.index, causal_acc.values, marker="s", lw=2, label="causal (no future)")
        ax5.axhline(identity_acc, color="green", lw=1, ls="--", label=f"F0 = {identity_acc:.3f}")
        ax5.set_xscale("log")
        ax5.set_title("(e) Sign prediction:\nlookahead vs causal — correction")
        ax5.set_xlabel("median window")
        ax5.set_ylabel("test accuracy")
        ax5.legend(fontsize=8, frameon=False, loc="lower left")
        ax5.grid(alpha=0.3, which="both")

    fig.suptitle("Filter denoising as preprocessing: TL;DR — five panels covering 44 iterations",
                 fontsize=12, fontweight="bold", y=1.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
