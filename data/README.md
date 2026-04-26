# Real financial data

## Basket

Default tickers (see `momo.data.DEFAULT_TICKERS`):

- equity: SPY, QQQ, AAPL, MSFT, JPM
- fx: EURUSD=X, GBPUSD=X, JPY=X
- crypto: BTC-USD, ETH-USD

## Date range

`2018-01-01` to `2025-12-31`, daily bars from Yahoo Finance via `yfinance`
(`auto_adjust=True`). Series shorter than 500 observations are dropped.
Returns are log-returns by default.

## Cache

- Layout: `data/cache/returns_<sha1-of-sorted-tickers-and-range>.parquet`
- A second `fetch_returns(...)` call with identical args reads the cache
  instead of hitting the network.
- The yfinance HTTP cache itself lives at `~/.cache/py-yfinance/`.
- This directory is git-ignored (`data/cache/`).

## Repopulating

```bash
python3 experiments/fetch_data.py
```

If yfinance fails (rate limit, no network) the loader falls back to a
deterministic synthetic basket so downstream experiments still run.

## Different tickers

Pass any `list[str]` to `fetch_returns`. To make a new basket the default,
edit `DEFAULT_TICKERS` in `src/momo/data.py`. To regenerate the cache for
a new range, also change `start`/`end` (the cache key includes them).
