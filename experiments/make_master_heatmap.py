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

from momo.filters import FILTER_REGISTRY
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
    "N1 Gauss": GaussianNoise(0.5),
    "N2 Pink (1/f)": PinkFARIMANoise(d=0.3, sigma=0.5),
    "N3 α-stable": StableNoise(alpha=1.7, sigma=0.5),
    "N4 Mixed": MixedFARIMAStableNoise(d=0.25, alpha=1.7, sigma=0.5),
    "N5 Regime-switch": RegimeSwitchNoise(sigma=0.5, alpha=1.6, block_length=128),
    "N6 Jump-diffusion": JumpDiffusionNoise(sigma=0.3, jump_intensity=0.02, jump_scale=3.0),
}

FILTER_ORDER = ["F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9"]
FILTER_LABELS = {
    "F0": "F0 Identity", "F1": "F1 MA(21)", "F2": "F2 Kalman",
    "F3": "F3 Wavelet", "F4": "F4 Median(21)", "F5": "F5 CNN small",
    "F6": "F6 Adapt.Wav.", "F7": "F7 Hybrid", "F8": "F8 Meta",
    "F9": "F9 CNN large",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=4096)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("figures/master_snr_heatmap.pdf"))
    parser.add_argument("--csv", type=Path, default=Path("tables/master_snr_table.csv"))
    args = parser.parse_args()

    sns.set_theme(context="paper", style="white", font_scale=0.95)
    rows = []
    for nname, N in NOISES.items():
        for fname in FILTER_ORDER:
            F = FILTER_REGISTRY[fname]()
            deltas = []
            for seed in range(args.n_seeds):
                rng = np.random.default_rng(seed * 17)
                signal = gaussian_filter1d(rng.standard_normal(args.T), 15)
                observed = signal + N.sample(args.T, rng)
                snr_in = snr_db(signal, observed)
                snr_out = snr_db(signal, F.apply(observed))
                deltas.append(snr_out - snr_in)
            rows.append(dict(noise=nname, filter=fname, delta_snr_db=float(np.mean(deltas))))

    df = pd.DataFrame(rows)
    df.to_csv(args.csv, index=False)
    pivot = df.pivot(index="filter", columns="noise", values="delta_snr_db").reindex(index=FILTER_ORDER, columns=list(NOISES.keys()))
    pivot["среднее"] = pivot.mean(axis=1)
    pivot = pivot.rename(index=FILTER_LABELS)

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", center=8, ax=ax,
                cbar_kws={"label": r"$\Delta$ SNR (dB)"})
    ax.set_title("Прирост SNR (дБ): все 10 фильтров × 6 шумов; столбец «среднее» итожит универсальность")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)

    print("=== SNR gain summary ===")
    print(pivot.round(2).to_string())
    print(f"\nWrote {args.out}, {args.csv}")


if __name__ == "__main__":
    main()
