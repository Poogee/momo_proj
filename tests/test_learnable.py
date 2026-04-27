from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from momo.learnable import LearnableCNNFilter, DEFAULT_WEIGHTS_PATH
from momo.filters import FILTER_REGISTRY


WEIGHTS_OK = Path(DEFAULT_WEIGHTS_PATH).exists()
pytestmark = pytest.mark.skipif(not WEIGHTS_OK, reason="trained weights file not present")


def test_registry_has_f5():
    assert "F5" in FILTER_REGISTRY
    assert FILTER_REGISTRY["F5"] is LearnableCNNFilter


@pytest.mark.parametrize("n", [100, 256, 1000, 4096])
def test_apply_preserves_length(n):
    rng = np.random.default_rng(0)
    y = rng.standard_normal(n).astype(np.float64)
    filt = LearnableCNNFilter()
    out = filt.apply(y)
    assert out.shape == y.shape
    assert np.isfinite(out).all()


def test_filter_reduces_mse_on_gaussian_noise():
    rng = np.random.default_rng(123)
    T = 2048
    t = np.arange(T)
    clean = np.sin(2.0 * np.pi * t / 200.0).astype(np.float64)
    noisy = clean + 0.4 * rng.standard_normal(T)
    filt = LearnableCNNFilter()
    out = filt.apply(noisy)
    mse_in = float(np.mean((noisy - clean) ** 2))
    mse_out = float(np.mean((out - clean) ** 2))
    assert mse_out < mse_in


def test_filter_is_deterministic():
    rng = np.random.default_rng(7)
    y = rng.standard_normal(1000).astype(np.float64)
    filt = LearnableCNNFilter()
    out1 = filt.apply(y)
    out2 = filt.apply(y)
    assert np.array_equal(out1, out2)
