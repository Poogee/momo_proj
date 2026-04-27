from __future__ import annotations

import numpy as np

from momo.filters import (
    AdaptiveMetaFilter,
    AdaptiveWaveletFilter,
    HybridMedianWaveletFilter,
    KalmanLocalLevelFilter,
)
from momo.metrics import mse, snr_db
from momo.noise import HawkesClusteredJumpNoise, MixedFARIMAStableNoise, StableNoise


def _signal_plus_noise(T, rng, noise_cls):
    from scipy.ndimage import gaussian_filter1d
    signal = gaussian_filter1d(rng.standard_normal(T), 12)
    observed = signal + noise_cls.sample(T, rng)
    return signal, observed


def test_adaptive_wavelet_returns_finite():
    rng = np.random.default_rng(0)
    sig, obs = _signal_plus_noise(2048, rng, StableNoise(alpha=1.7, sigma=0.5))
    out = AdaptiveWaveletFilter().apply(obs)
    assert out.shape == obs.shape
    assert np.all(np.isfinite(out))


def test_adaptive_wavelet_improves_snr_on_stable():
    rng = np.random.default_rng(0)
    sig, obs = _signal_plus_noise(4096, rng, StableNoise(alpha=1.7, sigma=0.5))
    out = AdaptiveWaveletFilter().apply(obs)
    assert snr_db(sig, out) > snr_db(sig, obs)


def test_hybrid_outperforms_pure_wavelet_on_alpha_stable():
    rng = np.random.default_rng(0)
    sig, obs = _signal_plus_noise(4096, rng, StableNoise(alpha=1.6, sigma=0.5))
    f_wav = AdaptiveWaveletFilter()
    f_hyb = HybridMedianWaveletFilter(median_window=5)
    snr_wav = snr_db(sig, f_wav.apply(obs))
    snr_hyb = snr_db(sig, f_hyb.apply(obs))
    assert snr_hyb > snr_wav


def test_meta_filter_routes_to_kalman_on_gaussian():
    rng = np.random.default_rng(0)
    from momo.noise import GaussianNoise
    sig, obs = _signal_plus_noise(4096, rng, GaussianNoise(0.5))
    out = AdaptiveMetaFilter().apply(obs)
    out_kalman = KalmanLocalLevelFilter(process_var=1e-3, obs_var=1.0).apply(obs)
    assert np.allclose(out, out_kalman, atol=1e-6)


def test_meta_filter_routes_to_hybrid_on_alpha_stable():
    rng = np.random.default_rng(0)
    sig, obs = _signal_plus_noise(4096, rng, StableNoise(alpha=1.7, sigma=0.5))
    out_meta = AdaptiveMetaFilter().apply(obs)
    out_hybrid = HybridMedianWaveletFilter(median_window=5).apply(obs)
    assert np.allclose(out_meta, out_hybrid, atol=1e-6)


def test_meta_filter_competitive_on_mixed():
    rng = np.random.default_rng(0)
    sig, obs = _signal_plus_noise(4096, rng, MixedFARIMAStableNoise(d=0.25, alpha=1.7, sigma=0.5))
    snr_in = snr_db(sig, obs)
    snr_meta = snr_db(sig, AdaptiveMetaFilter().apply(obs))
    assert snr_meta > snr_in + 5.0


def test_hawkes_noise_length_match():
    rng = np.random.default_rng(0)
    n = HawkesClusteredJumpNoise().sample(2000, rng)
    assert n.size == 2000
    assert np.all(np.isfinite(n))


def test_hawkes_more_active_with_higher_excitation():
    rng = np.random.default_rng(0)
    low = HawkesClusteredJumpNoise(self_excitation=0.05, base_intensity=0.005).sample(8000, rng)
    rng = np.random.default_rng(0)
    high = HawkesClusteredJumpNoise(self_excitation=0.5, base_intensity=0.005).sample(8000, rng)
    assert np.std(high) > np.std(low)


def test_hawkes_determinism():
    a = HawkesClusteredJumpNoise().sample(500, np.random.default_rng(7))
    b = HawkesClusteredJumpNoise().sample(500, np.random.default_rng(7))
    assert np.allclose(a, b)
