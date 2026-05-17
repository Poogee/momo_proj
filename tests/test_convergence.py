from __future__ import annotations

import numpy as np

from momo.metrics import (
    converged,
    convergence_auc,
    convergence_fraction,
    divergence_slope,
    noise_floor_quantiles,
    time_to_drop,
)


def _decaying(n=400, rate=0.02, floor=1e-8, seed=0):
    rng = np.random.default_rng(seed)
    k = np.arange(n)
    base = np.exp(-rate * k) + floor
    return base * np.exp(rng.normal(0, 0.05, n))


def _diverging(n=400, rate=0.01, seed=0):
    rng = np.random.default_rng(seed)
    k = np.arange(n)
    return (0.1 * np.exp(rate * k)) * np.exp(rng.normal(0, 0.05, n))


def test_time_to_drop_detects_factor():
    h = _decaying()
    k = time_to_drop(h, factor=1e3)
    assert k is not None and 0 < k < h.size
    assert h[k] <= np.mean(h[:5]) / 1e3
    assert time_to_drop(_diverging(), factor=1e3) is None


def test_converged_uses_robust_tail():
    assert converged(_decaying(), eps=1e-3) is True
    assert converged(_diverging(), eps=1e-3) is False
    # single lucky last point does not flip a non-converged run
    bad = _diverging()
    bad[-1] = 1e-12
    assert converged(bad, eps=1e-6, patience=10) is False


def test_convergence_fraction_counts_seeds():
    good = [_decaying(seed=s) for s in range(5)]
    bad = [_diverging(seed=s) for s in range(5)]
    assert convergence_fraction(good, eps=1e-3) == 1.0
    assert convergence_fraction(bad, eps=1e-3) == 0.0
    assert convergence_fraction(good[:2] + bad[:2], eps=1e-3) == 0.5


def test_divergence_slope_sign():
    assert divergence_slope(_decaying()) < -0.1   # converging
    assert divergence_slope(_diverging()) > 0.1    # diverging
    flat = np.full(400, 0.5) * np.exp(np.random.default_rng(0).normal(0, 0.01, 400))
    assert abs(divergence_slope(flat)) < 0.1       # stalled at a floor


def test_convergence_auc_orders_runs():
    assert convergence_auc(_decaying()) < convergence_auc(_diverging())


def test_noise_floor_quantiles_ordered():
    q = noise_floor_quantiles(_decaying(), tail_frac=0.2)
    assert q[0.1] <= q[0.5] <= q[0.9]
