"""Microstructure experiment (#3): denoising the price before estimating IV.

For high-frequency data the y = s + xi model is literally correct: the
observed log-price is the (latent) efficient price plus microstructure
noise (bid-ask bounce, discreteness). The efficient price here is a
random walk -- so there is NO predictable signal in the returns, fully
consistent with our negative result on raw-return prediction. But the
quantity of interest is the integrated variance IV over a day. The naive
realized variance RV = sum (Delta p)^2 is biased upward by ~2 M Var(u);
denoising the price before computing RV removes most of this bias. Since
the efficient price (hence true IV) is known by construction, the metric
(relative error to true IV) is honest. This is a positive result for
filtering on a microstructure-flavoured real-finance task.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse

import numpy as np
import pandas as pd

from momo.filters import (
    CausalMedianFilter,
    IdentityFilter,
    KalmanLocalLevelFilter,
    MovingAverageFilter,
    WaveletThresholdFilter,
)

FILTERS = {
    "F0 raw": lambda: IdentityFilter(),
    "F1 MA": lambda: MovingAverageFilter(window=5),
    "F2 Kalman": lambda: KalmanLocalLevelFilter(process_var=1e-2, obs_var=1.0),
    "F3 Wavelet": lambda: WaveletThresholdFilter(wavelet="db4", mode="soft",
                                                 threshold="universal"),
    "F4 Median": lambda: CausalMedianFilter(window=5),
}


def simulate_day(M, sigma, gamma, heavy, rng):
    """Efficient log-price random walk + microstructure noise.

    M ticks; per-tick efficient innovation std = sigma; microstructure
    noise std = gamma * sigma. heavy=True -> Student-t(3) efficient
    innovations (ties to the heavy-tailed theme N3).
    """
    if heavy:
        e = rng.standard_t(3, size=M) / np.sqrt(3.0)
    else:
        e = rng.standard_normal(M)
    p_eff = np.cumsum(sigma * e)
    u = gamma * sigma * rng.standard_normal(M)
    p_obs = p_eff + u
    iv_true = float(np.sum(np.diff(p_eff) ** 2))
    return p_obs, iv_true


def rv(price):
    d = np.diff(price)
    return float(np.sum(d * d))


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", type=Path,
                    default=Path("tables/microstructure_summary.csv"))
    ap.add_argument("--days", type=int, default=200)
    ap.add_argument("--ticks", type=int, default=390)  # ~1 trading day, 1-min
    args = ap.parse_args()

    sigma = 1e-3
    rows = []
    for heavy in (False, True):
        for gamma in (0.5, 1.0, 2.0):  # microstructure noise-to-signal
            rng = np.random.default_rng(0)
            for day in range(args.days):
                p, iv = simulate_day(args.ticks, sigma, gamma, heavy, rng)
                for fname, fbuild in FILTERS.items():
                    phat = fbuild().apply(p)
                    est = rv(phat)
                    rows.append(dict(
                        regime="heavy" if heavy else "gauss",
                        gamma=gamma, day=day, filter=fname,
                        rv_est=est, iv_true=iv,
                        rel_err=abs(est - iv) / iv,
                    ))
    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv}  ({len(df)} rows)")
    for reg in ("gauss", "heavy"):
        sub = df[df.regime == reg]
        piv = sub.pivot_table(index="filter", columns="gamma",
                              values="rel_err", aggfunc="median")
        print(f"\nMedian |RV-IV|/IV, regime={reg} (lower = better), "
              f"columns = noise-to-signal gamma:")
        print(piv.round(3).to_string())


if __name__ == "__main__":
    run()
