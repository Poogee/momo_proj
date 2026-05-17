"""Intraday experiment (#3 on real data): denoise 1-min prices before RV.

For high-frequency data the y = s + xi model holds literally: observed
log-price = latent efficient price + microstructure noise. The true
integrated variance is latent for real data, so we use the textbook
microstructure-robust benchmark: the *sparse 5-minute* realized variance
(Andersen-Bollerslev; Liu-Patton-Sheppard show 5-min RV is hard to beat).
We compare, per session, the full 1-min RV (microstructure-contaminated)
against the same after denoising the 1-min price, measuring how close
each gets to the 5-min benchmark. Filtering helping here is a positive
result on real intraday financial data.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse

import numpy as np
import pandas as pd

from momo.data import DEFAULT_TICKERS, fetch_intraday
from momo.filters import (
    CausalMedianFilter,
    IdentityFilter,
    KalmanLocalLevelFilter,
    MovingAverageFilter,
    OnlineAdaptiveFilter,
    WaveletThresholdFilter,
)

FILTERS = {
    "F0 raw": lambda: IdentityFilter(),
    "F1 MA": lambda: MovingAverageFilter(window=5),
    "F2 Kalman": lambda: KalmanLocalLevelFilter(process_var=1e-2, obs_var=1.0),
    "F3 Wavelet": lambda: WaveletThresholdFilter(wavelet="db4", mode="soft",
                                                 threshold="universal"),
    "F4 Median": lambda: CausalMedianFilter(window=5),
    "FA online": lambda: OnlineAdaptiveFilter(window=9, k=3.0),
}


def rv(logprice: np.ndarray) -> float:
    d = np.diff(logprice)
    return float(np.sum(d * d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", type=Path,
                    default=Path("tables/intraday_summary.csv"))
    ap.add_argument("--min-bars", type=int, default=60)
    args = ap.parse_args()

    tickers = sum(DEFAULT_TICKERS.values(), [])
    px = fetch_intraday(tickers, interval="1m")
    rows = []
    for tk in px.columns:
        s = px[tk].dropna()
        for day, grp in s.groupby(s.index.date):
            p = np.log(grp.to_numpy())
            if p.size < args.min_bars:
                continue
            bench = rv(p[::5])              # sparse 5-min benchmark
            if bench <= 0:
                continue
            for fname, fbuild in FILTERS.items():
                phat = fbuild().apply(p)
                est = rv(phat)
                rows.append(dict(
                    ticker=tk, day=str(day), filter=fname,
                    rv_est=est, rv_bench=bench,
                    rel_err=abs(est - bench) / bench,
                ))
    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv}  ({len(df)} rows, "
          f"{df.day.nunique()} sessions, {df.ticker.nunique()} tickers)")
    g = (df.groupby("filter").rel_err.median()
         .reindex([k for k in FILTERS]))
    print("\nMedian |RV_1min - RV_5min| / RV_5min (lower = better):")
    print(g.round(3).to_string())


if __name__ == "__main__":
    main()
