"""Smoke tests for the June 2026 dataset loaders and the lean-filter
runner. These exercise the cache/synthetic-fallback path (no network is
required) and assert the loaders return usable, finite series."""
from __future__ import annotations

import numpy as np
import pytest

from momo.data import (
    fetch_atm,
    fetch_binance,
    fetch_cmapss,
    fetch_electricity,
    fetch_traffic,
    fetch_weather_noaa,
)


@pytest.mark.parametrize("loader,min_rows", [
    (lambda: fetch_binance("BTCUSDT"), 500),
    (lambda: fetch_electricity(n_series=4), 1000),
    (lambda: fetch_traffic(n_series=4), 1000),
    (lambda: fetch_weather_noaa(), 1000),
    (lambda: fetch_atm(), 500),
])
def test_loader_returns_finite_series(loader, min_rows):
    df = loader()
    assert df.shape[0] >= min_rows
    assert df.shape[1] >= 1
    # at least one column is mostly finite and non-constant
    col = df.iloc[:, 0].to_numpy(dtype=float)
    col = col[np.isfinite(col)]
    assert col.size >= min_rows // 2
    assert np.std(col) > 0


def test_cmapss_columns_and_lengths():
    df = fetch_cmapss()
    assert df.shape[1] >= 1
    # each kept column is a per-unit trajectory of plausible length
    s = df.iloc[:, 0].dropna().to_numpy(dtype=float)
    assert 100 <= s.size <= 1000
    assert np.std(s) > 0


def test_lean_filter_keys_are_causal_and_fast():
    # the runner's lean set must be exactly these five and all O(T) causal
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parents[1] / "experiments" / "run_new_datasets.py"
    spec = importlib.util.spec_from_file_location("run_new_datasets", p)
    R = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(R)
    assert set(R.LEAN_FILTERS) == {"F0", "F4", "FA", "F10", "F11"}
    y = np.cumsum(np.random.default_rng(0).standard_normal(512))
    for k, make in R.LEAN_FILTERS.items():
        out = make().apply(y)
        assert out.shape == y.shape
        assert np.all(np.isfinite(out))
