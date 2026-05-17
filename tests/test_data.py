from __future__ import annotations

import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from momo.data import (
    DEFAULT_FRED_SERIES,
    DEFAULT_TICKERS,
    ForecastTask,
    fetch_fred,
    fetch_intraday,
    fetch_nonfinancial,
    fetch_returns,
    make_ar_forecast_task,
    make_walk_forward_splits,
)


def _online() -> bool:
    try:
        socket.create_connection(("query1.finance.yahoo.com", 443), timeout=2.0).close()
        return True
    except OSError:
        return False


needs_net = pytest.mark.skipif(not _online(), reason="no network for yfinance")


@needs_net
def test_fetch_returns_returns_dataframe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = fetch_returns(["SPY", "QQQ"], start="2022-01-01", end="2023-01-01")
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 100
    assert df.shape[1] >= 1
    assert np.isfinite(df.values).all()


@needs_net
def test_fetch_returns_cached(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tickers = ["SPY", "AAPL"]
    start, end = "2022-01-01", "2023-01-01"
    t0 = time.perf_counter()
    a = fetch_returns(tickers, start=start, end=end)
    t1 = time.perf_counter()
    b = fetch_returns(tickers, start=start, end=end)
    t2 = time.perf_counter()
    assert (t2 - t1) <= (t1 - t0)
    pd.testing.assert_frame_equal(a, b)
    cache_files = list(Path("data/cache").glob("returns_*.parquet"))
    assert len(cache_files) >= 1


@pytest.mark.parametrize("interval", ["1m", "5m", "15m", "60m"])
def test_fetch_intraday_shape_and_index(interval):
    # works offline via the synthetic intraday fallback
    df = fetch_intraday(["SPY", "AAPL"], interval=interval, cache=False)
    assert df.shape[0] > 100
    assert list(df.columns)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.notna().all().all()
    # spans several distinct trading sessions
    assert len(np.unique(df.index.date)) >= 5


def test_fetch_intraday_cached(monkeypatch):
    tk = ["SPY"]
    a = fetch_intraday(tk, interval="5m")
    b = fetch_intraday(tk, interval="5m")
    assert a.shape == b.shape
    files = list(Path("data/cache").glob("intraday_5m_*.parquet"))
    assert len(files) >= 1


def test_walk_forward_splits_sizes_and_disjoint_test_windows():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=3000))
    splits = make_walk_forward_splits(s, n_splits=5, train_size=1000, test_size=200)
    assert len(splits) == 5
    for tr, te in splits:
        assert tr.shape == (1000,)
        assert te.shape == (200,)
    arr = np.asarray(s)

    def find_start(window):
        for i in range(arr.size - window.size + 1):
            if np.array_equal(arr[i:i + window.size], window):
                return i
        return -1

    test_starts = [find_start(te) for _, te in splits]
    assert all(s_ >= 0 for s_ in test_starts)
    test_starts_sorted = sorted(test_starts)
    for a, b in zip(test_starts_sorted, test_starts_sorted[1:]):
        assert b >= a + 200


def test_walk_forward_too_short_raises():
    s = pd.Series(np.zeros(100))
    with pytest.raises(ValueError):
        make_walk_forward_splits(s, n_splits=3, train_size=200, test_size=50)


def test_forecast_task_dim_and_loss_ordering():
    rng = np.random.default_rng(0)
    series = rng.normal(size=2000)
    task = make_ar_forecast_task(series, p=5, train_frac=0.7)
    assert isinstance(task, ForecastTask)
    assert task.dim == 5
    x_zero = np.zeros(task.dim)
    loss_zero = task.loss(x_zero)
    assert loss_zero > 0
    beta_ols, *_ = np.linalg.lstsq(task.train_x, task.train_y, rcond=None)
    loss_ols = task.loss(beta_ols)
    assert loss_ols < loss_zero


def test_forecast_task_grad_finite_difference():
    rng = np.random.default_rng(1)
    series = rng.normal(size=600)
    task = make_ar_forecast_task(series, p=4, train_frac=0.7)
    x = rng.normal(size=task.dim)
    g = task.grad(x)
    h = 1e-6
    g_num = np.zeros_like(g)
    for i in range(task.dim):
        e = np.zeros_like(x)
        e[i] = h
        g_num[i] = (task.loss(x + e) - task.loss(x - e)) / (2 * h)
    rel = np.linalg.norm(g - g_num) / max(np.linalg.norm(g_num), 1e-12)
    assert rel < 1e-4


def test_forecast_task_sample_batch_shapes():
    rng = np.random.default_rng(2)
    series = rng.normal(size=500)
    task = make_ar_forecast_task(series, p=3)
    bx, by = task.sample_batch(rng, 32)
    assert bx.shape == (32, 3)
    assert by.shape == (32,)


def test_fetch_intraday_rejects_bad_interval():
    with pytest.raises(ValueError):
        fetch_intraday(["SPY"], interval="3m", cache=False)


def test_fetch_fred_structure_offline_safe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = fetch_fred(["INDPRO", "UNRATE"], start="2010-01-01",
                    end="2020-01-01", cache=False)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 50 and df.shape[1] >= 1
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing
    assert not df.dropna(how="all").empty


def test_fetch_fred_cached(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a = fetch_fred(["INDPRO"], start="2015-01-01", end="2020-01-01")
    b = fetch_fred(["INDPRO"], start="2015-01-01", end="2020-01-01")
    pd.testing.assert_frame_equal(a, b)
    assert list(Path("data/cache").glob("fred_*.parquet"))


def test_default_fred_series_nonempty():
    assert DEFAULT_FRED_SERIES and all(isinstance(s, str)
                                       for s in DEFAULT_FRED_SERIES)


def test_fetch_nonfinancial_structure_offline_safe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    df = fetch_nonfinancial(cache=False)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 1000
    assert "OT" in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)
    assert np.isfinite(df["OT"].to_numpy()).all()


def test_default_tickers_structure():
    assert "equity" in DEFAULT_TICKERS
    assert "fx" in DEFAULT_TICKERS
    assert "crypto" in DEFAULT_TICKERS
    assert all(isinstance(t, str) for group in DEFAULT_TICKERS.values() for t in group)
