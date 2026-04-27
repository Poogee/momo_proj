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
    HybridMedianWaveletFilter,
    KalmanLocalLevelFilter,
    MedianFilter,
    MovingAverageFilter,
    WaveletThresholdFilter,
)


FILTERS = {
    "F1 MA(21)": MovingAverageFilter(window=21),
    "F2 Калман": KalmanLocalLevelFilter(process_var=1e-3, obs_var=1.0),
    "F3 Вейвлет": WaveletThresholdFilter(),
    "F4 Медиана": MedianFilter(window=21),
    "F7 Гибрид": HybridMedianWaveletFilter(),
}


def empirical_response(filt, T: int, n_freqs: int = 60, n_repeats: int = 3, seed: int = 0):
    freqs = np.linspace(0.005, 0.5, n_freqs)
    gains = []
    for f in freqs:
        amps = []
        for r in range(n_repeats):
            rng = np.random.default_rng(seed + r * 11)
            phase = rng.uniform(0, 2 * np.pi)
            t = np.arange(T)
            x = np.sin(2 * np.pi * f * t + phase)
            y = filt.apply(x)
            x_amp = float(np.std(x[T // 4 : 3 * T // 4]))
            y_amp = float(np.std(y[T // 4 : 3 * T // 4]))
            amps.append(y_amp / max(x_amp, 1e-9))
        gains.append(float(np.mean(amps)))
    return freqs, np.array(gains)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("figures/spectral_response.pdf"))
    parser.add_argument("--T", type=int, default=2048)
    args = parser.parse_args()

    sns.set_theme(context="paper", style="whitegrid", font_scale=0.95)
    fig, ax = plt.subplots(figsize=(9, 5))
    for fname, F in FILTERS.items():
        freqs, gains = empirical_response(F, args.T)
        ax.plot(freqs, gains, lw=1.5, label=fname)
    ax.set_xscale("log")
    ax.set_xlabel("нормированная частота $f / f_s$")
    ax.set_ylabel("коэффициент передачи (по амплитуде)")
    ax.set_title("Эмпирический амплитудный отклик фильтров на синусоиду")
    ax.axhline(1.0, color="gray", lw=0.5, ls="--")
    ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
