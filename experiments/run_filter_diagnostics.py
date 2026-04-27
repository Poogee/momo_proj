from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse

import numpy as np
import pandas as pd
import yaml

from momo.filters import (
    IdentityFilter,
    KalmanLocalLevelFilter,
    MedianFilter,
    MovingAverageFilter,
    WaveletThresholdFilter,
)
from momo.metrics import hill_alpha, hurst_dfa, mcculloch_alpha, mse, snr_db
from momo.noise import (
    GaussianNoise,
    MixedFARIMAStableNoise,
    PinkFARIMANoise,
    StableNoise,
)


def build_signal(T: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(T) / T
    if kind == "sine":
        return np.sin(2 * np.pi * 5 * t)
    if kind == "trend_step":
        s = np.zeros(T)
        s[: T // 3] = 0.0
        s[T // 3 : 2 * T // 3] = 1.0
        s[2 * T // 3 :] = -0.5
        s += 0.5 * t
        return s
    if kind == "smooth_random":
        x = rng.normal(0, 1, T)
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(x, sigma=15)
    raise ValueError(kind)


def build_noise(name: str, sigma: float = 1.0):
    if name == "N1":
        return GaussianNoise(sigma=sigma)
    if name == "N2":
        return PinkFARIMANoise(d=0.3, sigma=sigma)
    if name == "N3":
        return StableNoise(alpha=1.7, sigma=sigma)
    if name == "N4":
        return MixedFARIMAStableNoise(d=0.25, alpha=1.7, sigma=sigma)
    raise ValueError(name)


def build_filters(include_learnable: bool = True) -> dict:
    from momo.filters import AdaptiveMetaFilter, AdaptiveWaveletFilter, HybridMedianWaveletFilter
    base = {
        "F0": IdentityFilter(),
        "F1": MovingAverageFilter(window=21),
        "F2": KalmanLocalLevelFilter(process_var=1e-3, obs_var=1.0),
        "F3": WaveletThresholdFilter(wavelet="db4", mode="soft", threshold="universal"),
        "F4": MedianFilter(window=21),
        "F6": AdaptiveWaveletFilter(),
        "F7": HybridMedianWaveletFilter(),
        "F8": AdaptiveMetaFilter(),
    }
    if include_learnable:
        try:
            from momo.learnable import LearnableCNNFilter
            base["F5"] = LearnableCNNFilter()
        except Exception:
            pass
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=4096)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--signal", type=str, default="smooth_random")
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=Path("tables/filter_diagnostics.csv"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    filters = build_filters()
    rows = []
    for noise_name in ["N1", "N2", "N3", "N4"]:
        for seed in range(args.n_seeds):
            rng = np.random.default_rng(seed)
            signal = build_signal(args.T, args.signal, rng)
            noise = build_noise(noise_name, sigma=args.sigma)
            xi = noise.sample(args.T, rng)
            observed = signal + xi
            snr_in = snr_db(signal, observed)
            for f_name, filt in filters.items():
                hat = filt.apply(observed)
                snr_out = snr_db(signal, hat)
                rows.append(dict(
                    noise=noise_name, filter=f_name, seed=seed,
                    snr_in_db=snr_in, snr_out_db=snr_out,
                    delta_snr_db=snr_out - snr_in,
                    mse_in=mse(signal, observed),
                    mse_out=mse(signal, hat),
                    hurst_in=hurst_dfa(observed),
                    hurst_out=hurst_dfa(hat),
                    alpha_in=mcculloch_alpha(observed),
                    alpha_out=mcculloch_alpha(hat),
                ))
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    summary = df.groupby(["noise", "filter"]).agg(
        snr_in=("snr_in_db", "mean"),
        snr_out=("snr_out_db", "mean"),
        delta_snr=("delta_snr_db", "mean"),
        mse_in=("mse_in", "mean"),
        mse_out=("mse_out", "mean"),
    ).round(2)
    print(summary.to_string())


if __name__ == "__main__":
    main()
