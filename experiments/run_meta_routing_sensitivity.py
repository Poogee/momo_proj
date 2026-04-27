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

from momo.filters import AdaptiveMetaFilter
from momo.metrics import snr_db
from momo.noise import (
    GaussianNoise,
    JumpDiffusionNoise,
    MixedFARIMAStableNoise,
    PinkFARIMANoise,
    RegimeSwitchNoise,
    StableNoise,
)


NOISES = {
    "N1": GaussianNoise(0.5),
    "N2": PinkFARIMANoise(d=0.3, sigma=0.5),
    "N3": StableNoise(alpha=1.7, sigma=0.5),
    "N4": MixedFARIMAStableNoise(d=0.25, alpha=1.7, sigma=0.5),
    "N5": RegimeSwitchNoise(sigma=0.5, alpha=1.6, block_length=128),
    "N6": JumpDiffusionNoise(sigma=0.3, jump_intensity=0.02, jump_scale=3.0),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=4096)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--out-csv", type=Path, default=Path("tables/meta_routing_sensitivity.csv"))
    parser.add_argument("--out-fig", type=Path, default=Path("figures/meta_routing_sensitivity.pdf"))
    args = parser.parse_args()

    sns.set_theme(context="paper", style="whitegrid", font_scale=0.95)
    alpha_thresholds = [1.5, 1.7, 1.85, 1.9, 1.95]
    rows = []
    for alpha_th in alpha_thresholds:
        meta = AdaptiveMetaFilter(alpha_threshold=alpha_th)
        for nname, N in NOISES.items():
            for seed in range(args.n_seeds):
                rng = np.random.default_rng(seed * 19)
                signal = gaussian_filter1d(rng.standard_normal(args.T), 15)
                observed = signal + N.sample(args.T, rng)
                snr_in = snr_db(signal, observed)
                snr_out = snr_db(signal, meta.apply(observed))
                rows.append(dict(alpha_threshold=alpha_th, noise=nname, seed=seed,
                                 delta_snr_db=snr_out - snr_in))
    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    agg = df.groupby(["alpha_threshold", "noise"])["delta_snr_db"].mean().reset_index()
    for noise in NOISES:
        view = agg[agg["noise"] == noise].sort_values("alpha_threshold")
        ax.plot(view["alpha_threshold"], view["delta_snr_db"],
                marker="o", lw=1.5, label=noise)
    avg = df.groupby("alpha_threshold")["delta_snr_db"].mean()
    ax.plot(avg.index, avg.values, marker="s", lw=2.5, color="black",
            label="среднее по шумам", linestyle="--")
    ax.set_xlabel(r"$\alpha$-threshold (routing decision)")
    ax.set_ylabel(r"$\Delta$ SNR (dB)")
    ax.set_title("Чувствительность F8 к параметру маршрутизации")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, frameon=False, loc="lower right", ncol=2)
    fig.tight_layout()
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, dpi=150)
    plt.close(fig)
    print(agg.pivot(index="noise", columns="alpha_threshold", values="delta_snr_db").round(2))


if __name__ == "__main__":
    main()
