from __future__ import annotations

import sys
import time
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
from momo.noise import GaussianNoise, MixedFARIMAStableNoise, StableNoise


NOISES = {
    "N1": GaussianNoise(0.5),
    "N3": StableNoise(alpha=1.7, sigma=0.5),
    "N4": MixedFARIMAStableNoise(d=0.25, alpha=1.7, sigma=0.5),
}

LENGTHS = [256, 1024, 4096, 16384]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-csv", type=Path, default=Path("tables/filter_speed.csv"))
    parser.add_argument("--out-fig", type=Path, default=Path("figures/filter_speed.pdf"))
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    rows = []
    for fname, F_cls in FILTER_REGISTRY.items():
        F = F_cls()
        for T in LENGTHS:
            for nname, N in NOISES.items():
                times = []
                snrs = []
                for r in range(args.repeats):
                    rng = np.random.default_rng(r * 100 + T + hash(fname) % 100)
                    signal = gaussian_filter1d(rng.standard_normal(T), max(8, T // 200))
                    observed = signal + N.sample(T, rng)
                    if r == 0 and fname == "F5":
                        F.apply(observed)
                    t0 = time.perf_counter()
                    out = F.apply(observed)
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    times.append(elapsed_ms)
                    snrs.append(snr_db(signal, out) - snr_db(signal, observed))
                rows.append(dict(
                    filter=fname, T=T, noise=nname,
                    median_ms=float(np.median(times)),
                    snr_gain_db=float(np.mean(snrs)),
                ))
    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")

    sns.set_theme(context="paper", style="whitegrid", font_scale=0.95)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, nname in zip(axes, NOISES):
        sub = df[df["noise"] == nname]
        for fname in sub["filter"].unique():
            view = sub[sub["filter"] == fname].sort_values("T")
            ax.loglog(view["T"], view["median_ms"], marker="o", lw=1.5, label=fname)
        ax.set_xlabel(r"$T$ (длина сигнала)")
        ax.set_title(f"Шум {nname}")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("медианное время, мс")
    axes[0].legend(fontsize=8, loc="upper left", ncol=2, frameon=False)
    fig.suptitle("Время выполнения фильтров (медиана 5 запусков)")
    fig.tight_layout()
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, dpi=150)
    plt.close(fig)

    pareto = df[df["T"] == 4096].pivot(index="filter", columns="noise", values=["median_ms", "snr_gain_db"])
    print("\n=== Speed (ms) and SNR gain at T=4096 ===")
    print(pareto.round(2))


if __name__ == "__main__":
    main()
