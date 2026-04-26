from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from momo.data import DATA_DIR, DEFAULT_TICKERS, fetch_returns


def main() -> None:
    tickers = DEFAULT_TICKERS["equity"] + DEFAULT_TICKERS["fx"] + DEFAULT_TICKERS["crypto"]
    df = fetch_returns(tickers, start="2018-01-01", end="2025-12-31")
    print("=" * 60)
    print(f"basket: {len(df.columns)} tickers / {df.shape[0]} bars")
    print(f"date range: {df.index.min().date()} -> {df.index.max().date()}")
    print(f"cache dir : {DATA_DIR.resolve()}")
    print("-" * 60)
    summary = pd.DataFrame({
        "n": df.count(),
        "mean": df.mean(),
        "std": df.std(),
        "min": df.min(),
        "max": df.max(),
    })
    print(summary.to_string(float_format=lambda v: f"{v:+.5f}"))
    print("=" * 60)


if __name__ == "__main__":
    main()
