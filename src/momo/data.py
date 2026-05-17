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
    rets = rets.replace([np.inf, -np.inf], np.nan).dropna(how="all")
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


_INTRADAY_BARS_PER_DAY = {"1m": 390, "5m": 78, "15m": 26, "60m": 7}
_INTRADAY_DEFAULT_PERIOD = {"1m": "7d", "5m": "60d", "15m": "60d", "60m": "360d"}
_INTRADAY_STEP_MIN = {"1m": 1, "5m": 5, "15m": 15, "60m": 60}


def _synthetic_intraday(tickers: list[str], interval: str,
                        n_sessions: int) -> pd.DataFrame:
    """Reproducible intraday Close: efficient random walk with a U-shaped
    intraday volatility and additive bid-ask-bounce microstructure noise.
    """
    m = _INTRADAY_BARS_PER_DAY.get(interval, 78)
    rng = np.random.default_rng(123)
    u = np.linspace(0, 1, m)
    season = 0.6 + 1.4 * (u - 0.5) ** 2          # U-shaped vol over the day
    sigma = {"1m": 5e-4, "5m": 1.2e-3, "15m": 2.0e-3,
             "60m": 4.0e-3}.get(interval, 1.2e-3)  # per-bar efficient vol
    gamma = 1.5                                   # noise-to-signal
    idx, cols = [], {t: [] for t in tickers}
    start = pd.Timestamp("2026-01-05 09:30", tz="UTC")
    step = _INTRADAY_STEP_MIN.get(interval, 5)
    for d in range(n_sessions):
        day0 = start + pd.Timedelta(days=d)
        idx += [day0 + pd.Timedelta(minutes=step * k) for k in range(m)]
        for i, t in enumerate(tickers):
            lvl = 100.0 + 10.0 * i
            eff = lvl * np.exp(np.cumsum(sigma * season
                                         * rng.standard_normal(m)))
            noise = gamma * sigma * eff * rng.standard_normal(m)
            cols[t].extend((eff + noise).tolist())
    df = pd.DataFrame(cols, index=pd.DatetimeIndex(idx, name="Datetime"))
    return df


def fetch_intraday(
    tickers: list[str],
    interval: str = "5m",
    period: str | None = None,
    cache: bool = True,
) -> pd.DataFrame:
    """Intraday Close prices (columns=tickers, tz-aware Datetime index).

    interval in {"1m","5m"}. yfinance limits intraday history (1m ~ a few
    days, 5m ~ 60 days); we cap accordingly. Falls back to a reproducible
    synthetic intraday generator when the network/data is unavailable.
    """
    if interval not in _INTRADAY_BARS_PER_DAY:
        raise ValueError("interval must be one of "
                         f"{sorted(_INTRADAY_BARS_PER_DAY)}")
    tickers = list(tickers)
    period = period or _INTRADAY_DEFAULT_PERIOD[interval]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"intraday_{interval}_{_cache_key(tickers, period, '')}.parquet"
    if cache and path.exists():
        df = pd.read_parquet(path)
        df.index = pd.DatetimeIndex(pd.to_datetime(df.index), name="Datetime")
        return df
    df = None
    if _has_network():
        try:
            import yfinance as yf

            raw = yf.download(tickers, period=period, interval=interval,
                              auto_adjust=True, progress=False,
                              group_by="column", threads=True)
            if raw is not None and not raw.empty:
                df = _extract_close(raw, tickers)
        except Exception:
            df = None

    def _too_small(d: pd.DataFrame | None) -> bool:
        if d is None or d.empty or d.shape[0] < 200:
            return True
        idx = pd.to_datetime(d.index)
        return len(np.unique(idx.date)) < 5

    if _too_small(df):
        n_sessions = 30 if interval == "1m" else 60
        df = _synthetic_intraday(tickers, interval, n_sessions)
    df = df.sort_index().ffill().dropna(how="all")
    df = df.loc[:, [c for c in df.columns if df[c].notna().sum() > 50]]
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index), name="Datetime")
    if cache:
        df.to_parquet(path)
    return df


def _host_reachable(host: str, port: int = 443, timeout: float = 3.0) -> bool:
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


DEFAULT_FRED_SERIES = ["INDPRO", "UNRATE", "CPIAUCSL", "DGS10",
                       "DEXUSEU", "VIXCLS"]


def _synthetic_fred(series_ids: list[str], start: str, end: str) -> pd.DataFrame:
    idx = pd.bdate_range(start=start, end=end, freq="W")
    rng = np.random.default_rng(2026)
    out = {}
    for i, sid in enumerate(series_ids):
        drift = 0.0 + 0.02 * (i % 3)
        x = 50.0 + 10.0 * i + np.cumsum(rng.normal(drift, 1.0 + 0.3 * i,
                                                   size=len(idx)))
        out[sid] = x
    return pd.DataFrame(out, index=pd.DatetimeIndex(idx, name="Date"))


def fetch_fred(series_ids: list[str] | None = None,
               start: str = "2000-01-01", end: str = "2025-12-31",
               cache: bool = True) -> pd.DataFrame:
    """Macro series from FRED via the public ``fredgraph.csv`` endpoint
    (no API key). Returns one column per id on the union weekly grid
    (forward-filled). Deterministic synthetic fallback when offline so
    downstream experiments still run reproducibly. Cached to parquet."""
    series_ids = list(series_ids or DEFAULT_FRED_SERIES)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"fred_{_cache_key(series_ids, start, end)}.parquet"
    if cache and path.exists():
        df = pd.read_parquet(path)
        df.index = pd.DatetimeIndex(pd.to_datetime(df.index), name="Date")
        return df
    df: pd.DataFrame | None = None
    if _host_reachable("fred.stlouisfed.org"):
        import io
        import urllib.request

        cols: dict[str, pd.Series] = {}
        for sid in series_ids:
            # one request per id: single-series fredgraph.csv is an
            # unambiguous two-column ``date,<ID>`` levels file (the
            # multi-id endpoint can realign/transform columns).
            url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv?"
                   f"id={sid}&cosd={start}&coed={end}")
            req = urllib.request.Request(
                url, headers={"User-Agent": "momo-research/1.0"})
            for _attempt in range(2):
                try:
                    with urllib.request.urlopen(req, timeout=20) as resp:
                        one = pd.read_csv(io.BytesIO(resp.read()))
                    dcol = one.columns[0]
                    one[dcol] = pd.to_datetime(one[dcol], errors="coerce")
                    one = one.set_index(dcol)
                    s = pd.to_numeric(
                        one.iloc[:, 0].replace(".", np.nan), errors="coerce"
                    ).dropna()
                    if not s.empty:
                        cols[sid] = s
                    break
                except Exception:
                    continue
        if cols:
            df = pd.concat(cols, axis=1)
    if df is None or df.empty:
        df = _synthetic_fred(series_ids, start, end)
    df = df.sort_index().ffill().dropna(how="all")
    df.index = pd.DatetimeIndex(
        pd.to_datetime(df.index).to_numpy("datetime64[ns]"), name="Date")
    if cache:
        df.to_parquet(path)
    return df


_ETT_URL = ("https://raw.githubusercontent.com/zhouhaoyi/ETDataset/"
            "main/ETT-small/ETTh1.csv")


def _synthetic_ett(n: int = 9000) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2016-07-01", periods=n, freq="h", name="date")
    t = np.arange(n)
    daily = 6.0 * np.sin(2 * np.pi * t / 24.0)
    weekly = 3.0 * np.sin(2 * np.pi * t / (24.0 * 7))
    load = 20.0 + daily + weekly + np.cumsum(rng.normal(0, 0.15, n))
    ot = 10.0 + 0.4 * load + 4.0 * np.sin(2 * np.pi * t / (24.0 * 30)) \
        + rng.normal(0, 1.0, n)
    return pd.DataFrame({"HUFL": load + rng.normal(0, 1.0, n),
                         "OT": ot}, index=idx)


def fetch_nonfinancial(cache: bool = True) -> pd.DataFrame:
    """Non-financial open sensor series: the ETT (Electricity Transformer
    Temperature) hourly dataset — transformer oil temperature ``OT`` and
    high-useful-load ``HUFL``. Stable raw-GitHub source, deterministic
    synthetic fallback, cached to parquet."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "nonfinancial_etth1.parquet"
    if cache and path.exists():
        df = pd.read_parquet(path)
        df.index = pd.DatetimeIndex(pd.to_datetime(df.index), name="date")
        return df
    df: pd.DataFrame | None = None
    if _host_reachable("raw.githubusercontent.com"):
        try:
            raw = pd.read_csv(_ETT_URL)
            raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
            raw = raw.set_index("date")
            keep = [c for c in ("HUFL", "OT") if c in raw.columns]
            if keep and len(raw) > 1000:
                df = raw[keep].apply(pd.to_numeric, errors="coerce")
        except Exception:
            df = None
    if df is None or df.empty:
        df = _synthetic_ett()
    df = df.sort_index().ffill().dropna(how="all")
    df.index = pd.DatetimeIndex(
        pd.to_datetime(df.index).to_numpy("datetime64[ns]"), name="date")
    if cache:
        df.to_parquet(path)
    return df


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
