from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from momo.learnable import LearnableCNNFilter
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
    parser.add_argument("--out-csv", type=Path, default=Path("tables/learnable_v1_vs_v2.csv"))
    args = parser.parse_args()

    f5_v1 = LearnableCNNFilter(
        weights_path=str(Path("models/learnable_filter.pt")),
        channels=48, kernel_size=9, n_blocks=2,
    )
    f5_v2 = LearnableCNNFilter(
        weights_path=str(Path("models/learnable_filter_v2.pt")),
        channels=96, kernel_size=11, n_blocks=4,
    )

    rows = []
    for nname, N in NOISES.items():
        for seed in range(args.n_seeds):
            rng = np.random.default_rng(seed * 13)
            signal = gaussian_filter1d(rng.standard_normal(args.T), 15)
            observed = signal + N.sample(args.T, rng)
            snr_in = snr_db(signal, observed)
            for fname, F in [("F5_v1", f5_v1), ("F5_v2", f5_v2)]:
                snr_out = snr_db(signal, F.apply(observed))
                rows.append(dict(noise=nname, filter=fname, seed=seed,
                                 delta_snr_db=snr_out - snr_in))
    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print("=== Mean SNR gain (dB) by noise ===")
    print(df.groupby(["noise", "filter"])["delta_snr_db"].mean().unstack().round(2))


if __name__ == "__main__":
    main()
