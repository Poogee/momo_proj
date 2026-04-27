from __future__ import annotations

import numpy as np
from scipy.stats import levy_stable

from momo.clipping import AlphaAwareClipper


def test_clipper_passes_through_when_no_history():
    c = AlphaAwareClipper()
    g = np.array([1.0, 2.0, 3.0])
    out = c.update(g)
    assert np.allclose(out, g)


def test_clipper_caps_outlier_after_warmup():
    c = AlphaAwareClipper(window=128, refresh_every=8, base_scale=2.0)
    rng = np.random.default_rng(0)
    for _ in range(200):
        c.update(rng.normal(0, 1, 5))
    huge = np.array([100.0, 0.0, 0.0, 0.0, 0.0])
    out = c.update(huge)
    assert np.linalg.norm(out) < np.linalg.norm(huge)


def test_clipper_estimates_alpha_lower_for_stable():
    c = AlphaAwareClipper(window=512, refresh_every=64, base_scale=2.0)
    samples = levy_stable.rvs(alpha=1.5, beta=0, size=600, random_state=0)
    for s in samples:
        c.update(np.array([s]))
    assert c.alpha_hat < 1.95


def test_clipper_estimates_alpha_high_for_gaussian():
    c = AlphaAwareClipper(window=512, refresh_every=64, base_scale=2.0)
    rng = np.random.default_rng(0)
    for _ in range(600):
        c.update(rng.normal(0, 1, 1))
    assert c.alpha_hat > 1.7
