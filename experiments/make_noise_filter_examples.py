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
from scipy.ndimage import gaussian_filter1d

from momo.filters import (
    CausalMedianFilter,
    MovingAverageFilter,
    WaveletThresholdFilter,
)
from momo.noise import (
    GaussianNoise,
    MixedFARIMAStableNoise,
    PinkFARIMANoise,
    StableNoise,
)


# те же четыре класса шума, что и в экспериментах (alpha=1.2)
NOISES = {
    "N1 Гаусс": GaussianNoise(0.5),
    "N2 Память (1/f)": PinkFARIMANoise(d=0.4, sigma=0.5),
    "N3 Тяжёлый хвост": StableNoise(alpha=1.2, sigma=0.5),
    "N4 Смешанный": MixedFARIMAStableNoise(d=0.4, alpha=1.2, sigma=0.5),
}

# только реально используемые причинные фильтры, по одному на семейство
FILTERS = {
    "F1 скольз. среднее": MovingAverageFilter(window=15),
    "F3 вейвлет": WaveletThresholdFilter(wavelet="db4", mode="soft", threshold="universal"),
    "F4 медиана": CausalMedianFilter(window=9),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("figures/noise_filter_examples.pdf"))
    parser.add_argument("--T", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    sns.set_theme(context="paper", style="white", font_scale=0.85)
    n_rows = len(NOISES)
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 1.6 * n_rows), sharex=True)
    rng = np.random.default_rng(args.seed)
    signal_seed = rng.standard_normal(args.T)
    signal = gaussian_filter1d(signal_seed, 12)

    for row, (nname, N) in enumerate(NOISES.items()):
        local_rng = np.random.default_rng(args.seed * 100 + row)
        xi = N.sample(args.T, local_rng)
        observed = signal + xi
        ax_l, ax_r = axes[row]
        ax_l.plot(observed, color="C3", lw=0.6, alpha=0.8, label="наблюдаемое")
        ax_l.plot(signal, color="black", lw=1.0, label="сигнал")
        ax_l.set_ylabel(nname, fontsize=9)
        ax_l.grid(alpha=0.2)
        if row == 0:
            ax_l.set_title("Наблюдаемое = сигнал + шум", fontsize=10)
            ax_l.legend(loc="upper right", fontsize=7, frameon=False)
        for fname, F in FILTERS.items():
            recovered = F.apply(observed)
            ax_r.plot(recovered, lw=0.8, alpha=0.85, label=fname)
        ax_r.plot(signal, color="black", lw=1.0, ls="--", alpha=0.6, label="истинный сигнал")
        ax_r.grid(alpha=0.2)
        if row == 0:
            ax_r.set_title("Фильтрация: F1 скольз. / F3 вейвлет / F4 медиана", fontsize=10)
            ax_r.legend(loc="upper right", fontsize=7, frameon=False, ncol=2)
        # робастный масштаб: тяжёлые хвосты дают выбросы в сотни раз больше
        # сигнала; обрезаем ось, чтобы было видно, как медиана держится за
        # сигнал, а линейный/вейвлет тянутся к (улетающим за кадр) выбросам.
        lo, hi = np.percentile(observed, [1.0, 99.0])
        pad = 0.6 * (hi - lo) + 1e-9
        ax_l.set_ylim(lo - pad, hi + pad)
        ax_r.set_ylim(lo - pad, hi + pad)
        ax_l.set_xlim(0, args.T)
        ax_r.set_xlim(0, args.T)

    for ax in axes[-1]:
        ax.set_xlabel("$t$")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
