from __future__ import annotations

import numpy as np
import pytest

from momo.filters import (
    FILTER_REGISTRY,
    IdentityFilter,
    KalmanLocalLevelFilter,
    MedianFilter,
    MovingAverageFilter,
    WaveletThresholdFilter,
)
from momo.metrics import snr_db


def _all_filters():
    return [
        IdentityFilter(),
        MovingAverageFilter(window=20),
        KalmanLocalLevelFilter(process_var=1e-4, obs_var=1.0),
        WaveletThresholdFilter(wavelet="db4", level=None, mode="soft"),
        MedianFilter(window=21),
    ]


def test_length_preservation_all_filters():
    rng = np.random.default_rng(0)
    for n in [500, 1000, 1024, 2000, 8192]:
        y = rng.normal(0, 1, n)
        for f in _all_filters():
            out = f.apply(y)
            assert out.shape == y.shape, f"{type(f).__name__} length mismatch at n={n}"


def test_identity_returns_input_untouched():
    rng = np.random.default_rng(1)
    y = rng.normal(0, 1, 1000)
    out = IdentityFilter().apply(y)
    assert np.array_equal(out, y)


def test_ma_on_constant_returns_constant():
    y = np.full(500, 3.7)
    out = MovingAverageFilter(window=20).apply(y)
    assert np.allclose(out, 3.7, atol=1e-10)


def test_ma_reduces_noise_on_smooth_signal():
    rng = np.random.default_rng(42)
    T = 2000
    t = np.arange(T)
    signal = np.sin(2.0 * np.pi * t / 200.0)
    noisy = signal + rng.normal(0, 0.5, T)
    filtered = MovingAverageFilter(window=20).apply(noisy)
    mse_in = float(np.mean((noisy - signal) ** 2))
    mse_out = float(np.mean((filtered - signal) ** 2))
    assert mse_out < mse_in


def test_kalman_recovers_constant():
    rng = np.random.default_rng(7)
    T = 2000
    signal = np.ones(T)
    noisy = signal + rng.normal(0, 1.0, T)
    out = KalmanLocalLevelFilter(process_var=1e-6, obs_var=1.0).apply(noisy)
    burn_in = 200
    recovered_mean = float(out[burn_in:].mean())
    assert abs(recovered_mean - 1.0) < 0.1


def test_kalman_improves_snr_on_smooth_signal():
    rng = np.random.default_rng(123)
    T = 2000
    t = np.arange(T)
    signal = np.sin(2.0 * np.pi * t / 200.0)
    noisy = signal + rng.normal(0, 0.5, T)
    filt = KalmanLocalLevelFilter(process_var=1e-3, obs_var=1.0).apply(noisy)
    snr_in = snr_db(signal, noisy)
    snr_out = snr_db(signal, filt)
    assert snr_out > snr_in


def test_wavelet_recovers_blocky_signal():
    rng = np.random.default_rng(11)
    T = 2048
    signal = np.zeros(T)
    signal[300:800] = 1.0
    signal[1100:1500] = -0.5
    signal[1700:1900] = 2.0
    noisy = signal + rng.normal(0, 0.2, T)
    filt = WaveletThresholdFilter(wavelet="db4", mode="soft").apply(noisy)
    mse_in = float(np.mean((noisy - signal) ** 2))
    mse_out = float(np.mean((filt - signal) ** 2))
    assert mse_out < mse_in


def test_wavelet_handles_power_of_two_and_non_power_of_two():
    rng = np.random.default_rng(2)
    for n in [1024, 1000]:
        y = rng.normal(0, 1, n)
        out = WaveletThresholdFilter(wavelet="db4", mode="soft").apply(y)
        assert out.shape == y.shape


def test_wavelet_hard_thresholding_runs():
    rng = np.random.default_rng(3)
    y = rng.normal(0, 1, 1024)
    out = WaveletThresholdFilter(wavelet="db4", mode="hard").apply(y)
    assert out.shape == y.shape
    assert np.all(np.isfinite(out))


def test_wavelet_explicit_threshold():
    rng = np.random.default_rng(5)
    y = rng.normal(0, 1, 512)
    out = WaveletThresholdFilter(wavelet="db4", threshold=0.5).apply(y)
    assert out.shape == y.shape


def test_median_robust_to_spikes():
    rng = np.random.default_rng(99)
    T = 2000
    signal = np.zeros(T)
    observed = signal.copy()
    n_spikes = int(0.05 * T)
    idx = rng.choice(T, size=n_spikes, replace=False)
    signs = rng.choice([-1.0, 1.0], size=n_spikes)
    observed[idx] = 10.0 * signs
    w = 21
    med = MedianFilter(window=w).apply(observed)
    ma = MovingAverageFilter(window=w).apply(observed)
    mse_med = float(np.mean((med - signal) ** 2))
    mse_ma = float(np.mean((ma - signal) ** 2))
    assert mse_med < mse_ma


def test_all_filters_run_on_various_sizes():
    rng = np.random.default_rng(13)
    for n in [500, 8192]:
        y = rng.normal(0, 1, n)
        for f in _all_filters():
            out = f.apply(y)
            assert out.shape == y.shape
            assert np.all(np.isfinite(out))


def test_registry_keys_present():
    assert set(FILTER_REGISTRY.keys()) == {"F0", "F1", "F2", "F3", "F4"}
    assert FILTER_REGISTRY["F0"] is IdentityFilter
    assert FILTER_REGISTRY["F1"] is MovingAverageFilter
    assert FILTER_REGISTRY["F2"] is KalmanLocalLevelFilter
    assert FILTER_REGISTRY["F3"] is WaveletThresholdFilter
    assert FILTER_REGISTRY["F4"] is MedianFilter
