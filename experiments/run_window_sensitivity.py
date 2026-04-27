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
from scipy.ndimage import gaussian_filter1d

from momo.filters import HybridMedianWaveletFilter, MedianFilter, MovingAverageFilter
from momo.metrics import snr_db
from momo.noise import GaussianNoise, MixedFARIMAStableNoise, StableNoise


NOISES = {
    "N1 Gauss": GaussianNoise(0.5),
    "N3 α-stable": StableNoise(alpha=1.7, sigma=0.5),
    "N4 Mixed": MixedFARIMAStableNoise(d=0.25, alpha=1.7, sigma=0.5),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=4096)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--out-csv", type=Path, default=Path("tables/window_sensitivity.csv"))
    parser.add_argument("--out-fig", type=Path, default=Path("figures/window_sensitivity.pdf"))
    args = parser.parse_args()

    sns.set_theme(context="paper", style="whitegrid", font_scale=0.95)
    windows = [3, 5, 9, 15, 21, 31, 51, 81]
    rows = []
    for w in windows:
        filters = {
            "F1 MA": MovingAverageFilter(window=w),
            "F4 Median": MedianFilter(window=w),
            "F7 Hybrid (med w)": HybridMedianWaveletFilter(median_window=w),
        }
        for nname, N in NOISES.items():
            for seed in range(args.n_seeds):
                rng = np.random.default_rng(seed)
                signal = gaussian_filter1d(rng.standard_normal(args.T), 15)
                observed = signal + N.sample(args.T, rng)
                snr_in = snr_db(signal, observed)
                for fname, F in filters.items():
                    snr_out = snr_db(signal, F.apply(observed))
                    rows.append(dict(window=w, filter=fname, noise=nname,
                                     seed=seed, delta_snr_db=snr_out - snr_in))
    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, nname in zip(axes, NOISES):
        sub = df[df["noise"] == nname].groupby(["filter", "window"])["delta_snr_db"].mean().reset_index()
        for fname in sub["filter"].unique():
            view = sub[sub["filter"] == fname]
            ax.plot(view["window"], view["delta_snr_db"], marker="o", lw=1.5, label=fname)
        ax.set_xscale("log")
        ax.set_xlabel("window size")
        ax.set_title(f"{nname}")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel(r"$\Delta$ SNR (dB)")
    axes[0].legend(fontsize=9, frameon=False)
    fig.suptitle("Чувствительность F1, F4, F7 к выбору окна")
    fig.tight_layout()
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, dpi=150)
    plt.close(fig)
    print(f"wrote {args.out_fig}")
    print()
    print(df.groupby(["noise", "filter", "window"])["delta_snr_db"].mean().unstack().round(2))


if __name__ == "__main__":
    main()
