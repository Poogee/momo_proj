from __future__ import annotations

import numpy as np
import pytest

from momo.metrics import hurst_dfa, mcculloch_alpha
from momo.noise import (
    GaussianNoise,
    MixedFARIMAStableNoise,
    NOISE_REGISTRY,
    PinkFARIMANoise,
    StableNoise,
)


def test_registry_keys():
    assert NOISE_REGISTRY["N1"] is GaussianNoise
    assert NOISE_REGISTRY["N2"] is PinkFARIMANoise
    assert NOISE_REGISTRY["N3"] is StableNoise
    assert NOISE_REGISTRY["N4"] is MixedFARIMAStableNoise


def test_determinism_same_seed():
    gen = GaussianNoise(sigma=1.0)
    a = gen.sample(1000, np.random.default_rng(42))
    b = gen.sample(1000, np.random.default_rng(42))
    assert np.array_equal(a, b)


def test_determinism_different_seed():
    gen = GaussianNoise(sigma=1.0)
    a = gen.sample(1000, np.random.default_rng(1))
    b = gen.sample(1000, np.random.default_rng(2))
    assert not np.array_equal(a, b)


def test_pink_determinism():
    gen = PinkFARIMANoise(d=0.3, sigma=1.0, truncation=500)
    a = gen.sample(2000, np.random.default_rng(7))
    b = gen.sample(2000, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_stable_determinism():
    gen = StableNoise(alpha=1.7, sigma=1.0)
    a = gen.sample(500, np.random.default_rng(11))
    b = gen.sample(500, np.random.default_rng(11))
    assert np.array_equal(a, b)


def test_mixed_determinism():
    gen = MixedFARIMAStableNoise(d=0.3, alpha=1.7, sigma=1.0, truncation=500)
    a = gen.sample(500, np.random.default_rng(13))
    b = gen.sample(500, np.random.default_rng(13))
    assert np.array_equal(a, b)


def test_gaussian_moments():
    gen = GaussianNoise(sigma=2.0)
    x = gen.sample(20_000, np.random.default_rng(0))
    assert abs(x.mean()) < 0.1
    assert abs(x.std(ddof=1) - 2.0) / 2.0 < 0.05


def test_gaussian_variance_scaling():
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    a = GaussianNoise(sigma=1.0).sample(20_000, rng_a)
    b = GaussianNoise(sigma=2.0).sample(20_000, rng_b)
    ratio = b.std(ddof=1) / a.std(ddof=1)
    assert abs(ratio - 2.0) < 0.05


def test_pink_hurst_recovers():
    d = 0.3
    gen = PinkFARIMANoise(d=d, sigma=1.0, truncation=2000)
    x = gen.sample(8000, np.random.default_rng(0))
    h = hurst_dfa(x)
    assert abs(h - (0.5 + d)) < 0.1


def test_stable_alpha_recovers():
    alpha = 1.5
    gen = StableNoise(alpha=alpha, sigma=1.0)
    x = gen.sample(20_000, np.random.default_rng(0))
    est = mcculloch_alpha(x)
    assert abs(est - alpha) < 0.15


def test_mixed_joint_property():
    gen = MixedFARIMAStableNoise(d=0.25, alpha=1.7, sigma=1.0, truncation=2000)
    x = gen.sample(8000, np.random.default_rng(0))
    h = hurst_dfa(x)
    a = mcculloch_alpha(x)
    assert h > 0.5
    assert a < 1.85


def test_pink_extreme_d_finite():
    gen = PinkFARIMANoise(d=0.45, sigma=1.0, truncation=2000)
    x = gen.sample(2000, np.random.default_rng(0))
    assert x.shape == (2000,)
    assert np.all(np.isfinite(x))


def test_output_shapes():
    rng = np.random.default_rng(0)
    assert GaussianNoise().sample(123, rng).shape == (123,)
    assert PinkFARIMANoise(truncation=200).sample(123, rng).shape == (123,)
    assert StableNoise().sample(123, rng).shape == (123,)
    assert MixedFARIMAStableNoise(truncation=200).sample(123, rng).shape == (123,)
