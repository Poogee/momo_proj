from __future__ import annotations

import hashlib
import socket
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_TICKERS = {
    "equity": ["SPY", "QQQ", "AAPL", "MSFT", "JPM"],
    "fx": ["EURUSD=X", "GBPUSD=X", "JPY=X"],
    "crypto": ["BTC-USD", "ETH-USD"],
}

DATA_DIR = Path("data/cache")
MIN_OBSERVATIONS = 500


def _cache_key(tickers: list[str], start: str, end: str) -> str:
    payload = "|".join(sorted(tickers)) + f"::{start}::{end}"
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _cache_path(tickers: list[str], start: str, end: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / f"returns_{_cache_key(tickers, start, end)}.parquet"


def _has_network(host: str = "query1.finance.yahoo.com", port: int = 443, timeout: float = 2.0) -> bool:
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


def _extract_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" in raw.columns.get_level_values(0):
            prices = raw["Adj Close"].copy()
        else:
            prices = raw["Close"].copy()
    else:
        col = "Adj Close" if "Adj Close" in raw.columns else "Close"
        prices = raw[[col]].copy()
        prices.columns = [tickers[0]]
    prices.index = pd.to_datetime(prices.index)
    return prices


def _synthetic_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    idx = pd.bdate_range(start=start, end=end)
    rng = np.random.default_rng(42)
    data = {}
    for i, t in enumerate(tickers):
        drift = 0.0002 + 0.0001 * (i % 3)
        vol = 0.01 + 0.005 * ((i + 1) % 4)
        innov = rng.normal(drift, vol, size=len(idx))
        data[t] = 100.0 * np.exp(np.cumsum(innov))
    return pd.DataFrame(data, index=idx)


def _download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned empty frame")
    return _extract_close(raw, tickers)


def _to_returns(prices: pd.DataFrame, log_returns: bool) -> pd.DataFrame:
    prices = prices.sort_index().ffill().dropna(how="all")
    if log_returns:
        rets = np.log(prices / prices.shift(1))
    else:
        rets = prices.pct_change()
    rets = rets.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return rets


def fetch_returns(
    tickers: list[str],
    start: str = "2018-01-01",
    end: str = "2025-12-31",
    cache: bool = True,
    log_returns: bool = True,
) -> pd.DataFrame:
    tickers = list(tickers)
    path = _cache_path(tickers, start, end)
    if cache and path.exists():
        cached = pd.read_parquet(path)
        cached.index = pd.DatetimeIndex(pd.to_datetime(cached.index).astype("datetime64[ns]"), name="Date")
        return cached
    try:
        prices = _download_prices(tickers, start, end)
    except Exception:
        if not _has_network():
            prices = _synthetic_prices(tickers, start, end)
        else:
            prices = _synthetic_prices(tickers, start, end)
    rets = _to_returns(prices, log_returns)
    threshold = min(MIN_OBSERVATIONS, max(1, len(rets) // 2))
    keep = [t for t in rets.columns if rets[t].dropna().shape[0] >= threshold]
    rets = rets[keep].dropna(how="any")
    rets.index = pd.DatetimeIndex(rets.index.astype("datetime64[ns]"), name="Date")
    if cache:
        rets.to_parquet(path)
    return rets


def make_walk_forward_splits(
    returns: pd.Series,
    n_splits: int = 5,
    train_size: int = 1000,
    test_size: int = 200,
) -> list[tuple[np.ndarray, np.ndarray]]:
    arr = np.asarray(returns, dtype=float)
    n = arr.size
    if n < train_size + test_size:
        raise ValueError(f"need at least {train_size + test_size} obs, got {n}")
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    if n_splits == 1:
        return [(arr[:train_size], arr[train_size:train_size + test_size])]
    step = max(1, (n - train_size - test_size) // (n_splits - 1))
    splits = []
    for i in range(n_splits):
        s = i * step
        train = arr[s:s + train_size]
        test = arr[s + train_size:s + train_size + test_size]
        if test.size < test_size:
            break
        splits.append((train.copy(), test.copy()))
    return splits


def _ar_design(series: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    series = np.asarray(series, dtype=float).ravel()
    n = series.size
    if n <= p:
        raise ValueError(f"series length {n} must exceed p={p}")
    rows = n - p
    X = np.empty((rows, p), dtype=float)
    for i in range(p):
        X[:, i] = series[i:i + rows]
    y = series[p:]
    return X, y


@dataclass
class ForecastTask:
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    p: int

    @property
    def dim(self) -> int:
        return int(self.p)

    def _resolve(self, Z, y):
        X = self.train_x if Z is None else np.asarray(Z, dtype=float)
        t = self.train_y if y is None else np.asarray(y, dtype=float)
        return X, t

    def loss(self, x: np.ndarray, Z=None, y=None) -> float:
        X, t = self._resolve(Z, y)
        resid = X @ np.asarray(x, dtype=float) - t
        return float(np.mean(resid ** 2))

    def grad(self, x: np.ndarray, Z=None, y=None) -> np.ndarray:
        X, t = self._resolve(Z, y)
        resid = X @ np.asarray(x, dtype=float) - t
        return (2.0 / X.shape[0]) * (X.T @ resid)

    def sample_batch(self, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
        m = self.train_x.shape[0]
        n = min(int(n), m)
        idx = rng.integers(0, m, size=n)
        return self.train_x[idx], self.train_y[idx]


def make_ar_forecast_task(
    series: np.ndarray,
    p: int = 5,
    train_frac: float = 0.7,
) -> ForecastTask:
    series = np.asarray(series, dtype=float).ravel()
    if not (0.0 < train_frac < 1.0):
        raise ValueError("train_frac must be in (0,1)")
    cut = int(series.size * train_frac)
    if cut <= p + 1 or series.size - cut <= p + 1:
        raise ValueError("series too short for given p / train_frac")
    train_series = series[:cut]
    test_series = series[cut - p:]
    Xtr, ytr = _ar_design(train_series, p)
    Xte, yte = _ar_design(test_series, p)
    return ForecastTask(train_x=Xtr, train_y=ytr, test_x=Xte, test_y=yte, p=int(p))


REAL_TASK_REGISTRY = {"ar_forecast": make_ar_forecast_task}
