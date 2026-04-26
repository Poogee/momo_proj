from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import levy_stable

from momo.metrics import (
    gaussian_log_likelihood,
    hill_alpha,
    hurst_dfa,
    hurst_rs,
    mcculloch_alpha,
    mse,
    snr_db,
    time_to_eps,
)


def test_snr_known_ratio():
    rng = np.random.default_rng(0)
    signal = rng.normal(0, 1, 10_000)
    noise = rng.normal(0, 0.5, 10_000)
    val = snr_db(signal, signal + noise)
    expected = 10 * np.log10(1.0 / 0.25)
    assert abs(val - expected) < 0.5


def test_mse_zero_for_identity():
    y = np.linspace(-1, 1, 100)
    assert mse(y, y) == 0.0


def test_gaussian_log_likelihood_finite():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 1000)
    pred = np.zeros_like(y)
    val = gaussian_log_likelihood(y, pred)
    assert np.isfinite(val)


def test_hurst_brownian_motion_close_to_half():
    rng = np.random.default_rng(0)
    increments = rng.normal(0, 1, 8000)
    h_rs = hurst_rs(increments)
    h_dfa = hurst_dfa(increments)
    assert abs(h_rs - 0.5) < 0.1
    assert abs(h_dfa - 0.5) < 0.1


def test_hurst_persistent_above_half():
    rng = np.random.default_rng(0)
    n = 4000
    increments = rng.normal(0, 1, n)
    long_mem = np.cumsum(increments)
    h_rs = hurst_rs(long_mem)
    assert h_rs > 0.7


def test_hill_alpha_gaussian_above_one():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 10_000)
    a = hill_alpha(x)
    assert a > 2.0 or np.isnan(a) or a > 1.5


def test_hill_alpha_pareto_recovers_true():
    rng = np.random.default_rng(0)
    alpha = 1.5
    u = rng.uniform(0, 1, 50_000)
    x = u ** (-1.0 / alpha)
    est = hill_alpha(x, k=5000)
    assert abs(est - alpha) < 0.1


def test_mcculloch_alpha_recovers_stable():
    np.random.seed(0)
    x = levy_stable.rvs(alpha=1.7, beta=0, size=20_000, random_state=0)
    est = mcculloch_alpha(x)
    assert abs(est - 1.7) < 0.15


def test_mcculloch_alpha_gaussian_near_two():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 20_000)
    est = mcculloch_alpha(x)
    assert est > 1.85


def test_time_to_eps_hit():
    arr = np.array([10.0, 5.0, 2.0, 1.0, 0.5, 0.3])
    assert time_to_eps(arr, 1.0) == 3


def test_time_to_eps_no_hit():
    arr = np.array([10.0, 5.0, 2.0])
    assert time_to_eps(arr, 1.0) is None
