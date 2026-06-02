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

# Extended basket: same character as DEFAULT_TICKERS but covers more
# regimes (commodities, bonds, volatility index, more crypto) so the
# practical-criteria conclusions are not asset-specific.
EXTENDED_TICKERS = {
    "equity":     ["SPY", "QQQ", "DIA", "IWM", "AAPL", "MSFT", "JPM", "XOM"],
    "fx":         ["EURUSD=X", "GBPUSD=X", "JPY=X", "AUDUSD=X"],
    "crypto":     ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD"],
    "commodity":  ["GLD", "USO", "SLV"],   # gold, oil, silver ETFs
    "bond":       ["TLT", "IEF"],          # long / intermediate treasuries
    "volatility": ["^VIX"],                # equity vol index
}

DATA_DIR = Path("data/cache")
RAW_DIR = Path("data/raw")
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


_ETT_BASE = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/"
_ETT_VARIANTS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2")


def _synthetic_ett(n: int = 9000, freq: str = "h",
                   seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2016-07-01", periods=n, freq=freq, name="date")
    per_day = {"h": 24, "15min": 96, "T": 1440}.get(freq, 24)
    t = np.arange(n)
    daily = 6.0 * np.sin(2 * np.pi * t / per_day)
    weekly = 3.0 * np.sin(2 * np.pi * t / (per_day * 7))
    load = 20.0 + daily + weekly + np.cumsum(rng.normal(0, 0.15, n))
    ot = 10.0 + 0.4 * load + 4.0 * np.sin(2 * np.pi * t / (per_day * 30)) \
        + rng.normal(0, 1.0, n)
    return pd.DataFrame({"HUFL": load + rng.normal(0, 1.0, n),
                         "OT": ot}, index=idx)


def fetch_nonfinancial(variant: str = "ETTh1",
                       cache: bool = True) -> pd.DataFrame:
    """Non-financial open sensor series from the ETT collection
    (Electricity Transformer Temperature). ``variant`` is one of
    ``ETTh1``/``ETTh2`` (hourly) or ``ETTm1``/``ETTm2`` (15-minute).
    Returns columns ``HUFL`` (high-useful-load) and ``OT`` (operational
    transformer temperature). Stable raw-GitHub source with a
    deterministic synthetic fallback so experiments stay reproducible
    offline. Cached to parquet per variant.
    """
    if variant not in _ETT_VARIANTS:
        raise ValueError(f"variant must be one of {_ETT_VARIANTS}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"nonfinancial_{variant.lower()}.parquet"
    if cache and path.exists():
        df = pd.read_parquet(path)
        df.index = pd.DatetimeIndex(pd.to_datetime(df.index), name="date")
        return df
    df: pd.DataFrame | None = None
    if _host_reachable("raw.githubusercontent.com"):
        try:
            raw = pd.read_csv(_ETT_BASE + f"{variant}.csv")
            raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
            raw = raw.set_index("date")
            keep = [c for c in ("HUFL", "OT") if c in raw.columns]
            if keep and len(raw) > 1000:
                df = raw[keep].apply(pd.to_numeric, errors="coerce")
        except Exception:
            df = None
    if df is None or df.empty:
        freq = "h" if variant.startswith("ETTh") else "15min"
        n = 9000 if variant.startswith("ETTh") else 12000
        seed = 7 if "1" in variant else 11
        df = _synthetic_ett(n=n, freq=freq, seed=seed)
    df = df.sort_index().ffill().dropna(how="all")
    df.index = pd.DatetimeIndex(
        pd.to_datetime(df.index).to_numpy("datetime64[ns]"), name="date")
    if cache:
        df.to_parquet(path)
    return df


_SUNSPOT_URL = "https://www.sidc.be/SILSO/INFO/sndtotcsv.php"


def _synthetic_sunspots(n: int = 6000, seed: int = 13) -> pd.DataFrame:
    """Synthetic 11-year cycle + AR(1) residual, with sparse heavy spikes.
    The signal is smooth and recoverable, so filters should help — this is
    a positive non-financial control with a known answer."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1900-01-01", periods=n, freq="D", name="date")
    t = np.arange(n)
    cycle = 80.0 + 60.0 * np.sin(2 * np.pi * t / (11 * 365.25))
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.97 * ar[i - 1] + rng.normal(0, 5.0)
    spikes = (rng.uniform(size=n) < 0.005) * rng.normal(0, 40.0, size=n)
    sn = np.clip(cycle + ar + spikes, 0.0, None)
    return pd.DataFrame({"sunspot": sn}, index=idx)


def fetch_sunspots(cache: bool = True) -> pd.DataFrame:
    """Daily sunspot number — the canonical long-memory, mildly
    heavy-tailed scientific series. Falls back to a deterministic
    synthetic 11-year cycle if SIDC is unreachable. Cached to parquet."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "nonfinancial_sunspots.parquet"
    if cache and path.exists():
        df = pd.read_parquet(path)
        df.index = pd.DatetimeIndex(pd.to_datetime(df.index), name="date")
        return df
    df: pd.DataFrame | None = None
    if _host_reachable("www.sidc.be"):
        import io
        import urllib.request
        try:
            req = urllib.request.Request(
                _SUNSPOT_URL,
                headers={"User-Agent": "momo-research/1.0"},
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw = pd.read_csv(io.BytesIO(resp.read()), sep=";",
                                  header=None, engine="python")
            # SILSO daily file columns (positional): year;month;day;dec_yr;sn;...
            if raw.shape[1] >= 5:
                y, m, d = raw[0].astype(int), raw[1].astype(int), raw[2].astype(int)
                idx = pd.to_datetime(dict(year=y, month=m, day=d),
                                     errors="coerce")
                sn = pd.to_numeric(raw[4], errors="coerce")
                df = pd.DataFrame({"sunspot": sn.values},
                                  index=pd.DatetimeIndex(idx, name="date"))
                df = df[df["sunspot"] >= 0]
        except Exception:
            df = None
    if df is None or df.empty:
        df = _synthetic_sunspots()
    df = df.sort_index().ffill().dropna(how="all")
    df.index = pd.DatetimeIndex(
        pd.to_datetime(df.index).to_numpy("datetime64[ns]"), name="date")
    if cache:
        df.to_parquet(path)
    return df


# ======================================================================
# Additional real-world datasets (June 2026 sweep).
#
# Each loader follows the house pattern: parquet cache -> raw file in
# ``data/raw`` -> network download -> deterministic synthetic fallback,
# so every downstream experiment runs reproducibly offline. Sources that
# require a paid subscription / portal login (LOBSTER full feed, NYSE TAQ,
# CRSP) are intentionally NOT wrapped here -- they are not freely
# obtainable and are dropped from the study; see DECISIONS.md.
# ======================================================================


def _cache(name: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / name


def fetch_binance(symbol: str = "BTCUSDT", interval: str = "1h",
                  year: int = 2024, cache: bool = True) -> pd.DataFrame:
    """Binance spot klines (public data.binance.vision feed, no key).

    Returns a single ``close`` column indexed by close-time. Reads any
    monthly zips already present under ``data/raw`` first; otherwise
    downloads the 12 monthly zips for ``year``. Deterministic synthetic
    GBM fallback when the feed is unreachable. Cached to parquet.
    """
    path = _cache(f"binance_{symbol}_{interval}_{year}.parquet")
    if cache and path.exists():
        df = pd.read_parquet(path)
        df.index = pd.DatetimeIndex(pd.to_datetime(df.index), name="close_time")
        return df

    import io
    import urllib.request
    import zipfile

    def _read_month_bytes(buf: bytes) -> pd.DataFrame | None:
        try:
            with zipfile.ZipFile(io.BytesIO(buf)) as z:
                inner = z.namelist()[0]
                raw = pd.read_csv(io.BytesIO(z.read(inner)), header=None)
            # spot kline schema: open_time, o,h,l,c, vol, close_time, ...
            close = pd.to_numeric(raw.iloc[:, 4], errors="coerce")
            ct = pd.to_datetime(raw.iloc[:, 6].astype("int64"), unit="ms",
                                errors="coerce")
            return pd.DataFrame({"close": close.values},
                                index=pd.DatetimeIndex(ct, name="close_time"))
        except Exception:
            return None

    frames: list[pd.DataFrame] = []
    for m in range(1, 13):
        fn = f"{symbol}-{interval}-{year}-{m:02d}.zip"
        local = RAW_DIR / fn
        buf = None
        if local.exists():
            buf = local.read_bytes()
        elif _host_reachable("data.binance.vision"):
            url = (f"https://data.binance.vision/data/spot/monthly/klines/"
                   f"{symbol}/{interval}/{fn}")
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "momo-research/1.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    buf = resp.read()
            except Exception:
                buf = None
        if buf is not None:
            one = _read_month_bytes(buf)
            if one is not None and not one.empty:
                frames.append(one)

    if frames:
        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated(keep="first")].dropna()
    else:
        # synthetic hourly GBM fallback so experiments still run
        n = 24 * 365
        rng = np.random.default_rng(abs(hash(symbol)) % (2 ** 32))
        idx = pd.date_range(f"{year}-01-01", periods=n, freq="h",
                            name="close_time")
        px = 30000.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        df = pd.DataFrame({"close": px}, index=idx)
    if cache:
        df.to_parquet(path)
    return df


_LSTNET_BASE = ("https://raw.githubusercontent.com/laiguokun/"
                "multivariate-time-series-data/master/")


def _fetch_lstnet(name: str, url_sub: str, raw_name: str,
                  cache: bool = True) -> pd.DataFrame:
    """Shared loader for the laiguokun multivariate-time-series datasets
    (electricity, traffic) shipped as gzipped headerless CSV matrices
    (rows = hourly steps, columns = series)."""
    path = _cache(f"{name}.parquet")
    if cache and path.exists():
        return pd.read_parquet(path)
    import gzip
    import io
    import urllib.request

    raw = None
    local = RAW_DIR / raw_name
    if local.exists():
        raw = local.read_bytes()
    elif _host_reachable("raw.githubusercontent.com"):
        try:
            req = urllib.request.Request(
                _LSTNET_BASE + url_sub,
                headers={"User-Agent": "momo-research/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
        except Exception:
            raw = None
    if raw is not None:
        try:
            txt = gzip.decompress(raw)
            mat = pd.read_csv(io.BytesIO(txt), header=None)
        except Exception:
            mat = None
    else:
        mat = None
    if mat is None or mat.empty:
        rng = np.random.default_rng(99)
        n, k = 8000, 8
        t = np.arange(n)
        cols = {f"s{j}": 10.0 + 5.0 * np.sin(2 * np.pi * t / 24 + j)
                + np.cumsum(rng.normal(0, 0.2, n)) for j in range(k)}
        mat = pd.DataFrame(cols)
    mat.columns = [f"s{j}" for j in range(mat.shape[1])]
    if cache:
        mat.to_parquet(path)
    return mat


def fetch_electricity(n_series: int = 8, cache: bool = True) -> pd.DataFrame:
    """UCI/LSTNet electricity load (321 clients, hourly 2012--2014).
    Returns ``n_series`` columns spread across the client index."""
    df = _fetch_lstnet("nonfinancial_electricity",
                       "electricity/electricity.txt.gz",
                       "electricity.txt.gz", cache=cache)
    step = max(1, df.shape[1] // n_series)
    return df.iloc[:, ::step].iloc[:, :n_series]


def fetch_traffic(n_series: int = 8, cache: bool = True) -> pd.DataFrame:
    """LSTNet PEMS traffic occupancy (862 sensors, hourly).
    Returns ``n_series`` columns spread across the sensor index."""
    df = _fetch_lstnet("nonfinancial_traffic",
                       "traffic/traffic.txt.gz",
                       "traffic.txt.gz", cache=cache)
    step = max(1, df.shape[1] // n_series)
    return df.iloc[:, ::step].iloc[:, :n_series]


def fetch_weather_noaa(station: str = "USW00094728",
                       cache: bool = True) -> pd.DataFrame:
    """NOAA GHCN-daily station record (TMAX/TMIN in deg C, PRCP in mm).
    Default station USW00094728 = NY Central Park (1869--present).
    Reads ``data/raw/noaa_<station>.csv`` first, else NCEI access CSV."""
    path = _cache(f"nonfinancial_noaa_{station}.parquet")
    if cache and path.exists():
        df = pd.read_parquet(path)
        df.index = pd.DatetimeIndex(pd.to_datetime(df.index), name="date")
        return df
    import io
    import urllib.request

    raw = None
    local = RAW_DIR / f"noaa_{station}.csv"
    if local.exists():
        raw = pd.read_csv(local, usecols=lambda c: c in
                          ("DATE", "TMAX", "TMIN", "PRCP"))
    elif _host_reachable("www.ncei.noaa.gov"):
        url = ("https://www.ncei.noaa.gov/data/"
               "global-historical-climatology-network-daily/access/"
               f"{station}.csv")
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "momo-research/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = pd.read_csv(io.BytesIO(resp.read()),
                                  usecols=lambda c: c in
                                  ("DATE", "TMAX", "TMIN", "PRCP"))
        except Exception:
            raw = None
    if raw is not None and not raw.empty:
        raw["DATE"] = pd.to_datetime(raw["DATE"], errors="coerce")
        raw = raw.set_index("DATE").sort_index()
        out = pd.DataFrame(index=raw.index)
        # GHCN units: TMAX/TMIN in tenths of deg C, PRCP in tenths of mm
        if "TMAX" in raw:
            out["TMAX"] = pd.to_numeric(raw["TMAX"], errors="coerce") / 10.0
        if "TMIN" in raw:
            out["TMIN"] = pd.to_numeric(raw["TMIN"], errors="coerce") / 10.0
        df = out.dropna(how="all").ffill()
    else:
        rng = np.random.default_rng(31)
        n = 20000
        idx = pd.date_range("1960-01-01", periods=n, freq="D", name="date")
        t = np.arange(n)
        seas = 15.0 + 12.0 * np.sin(2 * np.pi * t / 365.25)
        df = pd.DataFrame({"TMAX": seas + rng.normal(0, 3.0, n),
                           "TMIN": seas - 8 + rng.normal(0, 3.0, n)},
                          index=idx)
    df.index = pd.DatetimeIndex(
        pd.to_datetime(df.index).to_numpy("datetime64[ns]"), name="date")
    if cache:
        df.to_parquet(path)
    return df


_CMAPSS_URL = ("https://raw.githubusercontent.com/edwardzjl/CMAPSSData/"
               "master/train_FD001.txt")
# sensors that actually vary on FD001 (constant channels excluded)
_CMAPSS_SENSORS = (2, 3, 4, 8, 11, 13)
_CMAPSS_UNITS = (1, 2, 3, 4, 5, 6)


def fetch_cmapss(units=_CMAPSS_UNITS, sensors=_CMAPSS_SENSORS,
                 cache: bool = True) -> pd.DataFrame:
    """NASA C-MAPSS turbofan degradation (FD001 training set). Returns one
    column ``u<unit>_s<sensor>`` per (engine unit, sensor) trajectory --
    a slowly degrading sensor signal with structural measurement noise.
    Columns have different lengths (per-unit run-to-failure) and are
    NaN-padded; callers drop NaNs per column."""
    path = _cache("nonfinancial_cmapss_fd001.parquet")
    if cache and path.exists():
        full = pd.read_parquet(path)
    else:
        import io
        import urllib.request

        arr = None
        local = RAW_DIR / "train_FD001.txt"
        if local.exists():
            arr = np.loadtxt(local)
        elif _host_reachable("raw.githubusercontent.com"):
            try:
                req = urllib.request.Request(
                    _CMAPSS_URL, headers={"User-Agent": "momo-research/1.0"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    arr = np.loadtxt(io.BytesIO(resp.read()))
            except Exception:
                arr = None
        if arr is None:
            rng = np.random.default_rng(17)
            rows = []
            for u in range(1, 11):
                T = int(rng.integers(150, 300))
                for c in range(1, T + 1):
                    base = [u, c, 0, 0, 0]
                    sens = [500 + 50 * np.exp(c / T) + rng.normal(0, 2.0)
                            for _ in range(21)]
                    rows.append(base + sens)
            arr = np.array(rows, dtype=float)
        # column layout: unit, cycle, 3 op-settings, 21 sensors
        cols = {}
        for u in range(1, int(arr[:, 0].max()) + 1):
            sub = arr[arr[:, 0] == u]
            order = np.argsort(sub[:, 1])
            sub = sub[order]
            for s in range(1, 22):
                cols[f"u{u}_s{s}"] = pd.Series(sub[:, 4 + s])
        full = pd.DataFrame(cols)
        if cache:
            full.to_parquet(path)
    keep = [f"u{u}_s{s}" for u in units for s in sensors
            if f"u{u}_s{s}" in full.columns]
    return full[keep]


def fetch_atm(cache: bool = True) -> pd.DataFrame:
    """Daily ATM cash-withdrawal totals (single ATM, ~2.2k days).
    Public Kaggle-derived CSV mirrored on GitHub. Returns one
    ``withdrawn`` column ordered chronologically by record id."""
    path = _cache("nonfinancial_atm.parquet")
    if cache and path.exists():
        return pd.read_parquet(path)
    import io
    import urllib.request

    raw = None
    local = RAW_DIR / "atm.csv"
    url = ("https://raw.githubusercontent.com/anjalysam/"
           "ATM_Transaction-data_analysis-/master/atm%20bank%20dataset.csv")
    if local.exists():
        raw = pd.read_csv(local)
    elif _host_reachable("raw.githubusercontent.com"):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "momo-research/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = pd.read_csv(io.BytesIO(resp.read()))
        except Exception:
            raw = None
    if raw is not None and "total_amount_withdrawn" in raw.columns:
        raw = raw.sort_values("id") if "id" in raw.columns else raw
        df = pd.DataFrame({"withdrawn": pd.to_numeric(
            raw["total_amount_withdrawn"], errors="coerce").to_numpy()})
        df = df.dropna()
    else:
        rng = np.random.default_rng(53)
        n = 2200
        t = np.arange(n)
        week = 1.0 + 0.4 * np.sin(2 * np.pi * t / 7)
        df = pd.DataFrame({"withdrawn":
                           5e5 * week * np.exp(rng.normal(0, 0.3, n))})
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
