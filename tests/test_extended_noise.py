from __future__ import annotations

import numpy as np

from momo.metrics import mcculloch_alpha
from momo.noise import JumpDiffusionNoise, RegimeSwitchNoise


def test_regime_switch_length_match():
    rng = np.random.default_rng(0)
    n = RegimeSwitchNoise(block_length=128).sample(2000, rng)
    assert n.size == 2000


def test_regime_switch_has_heavy_segments():
    rng = np.random.default_rng(0)
    x = RegimeSwitchNoise(sigma=0.5, alpha=1.5, block_length=200).sample(8000, rng)
    a = mcculloch_alpha(x)
    assert a < 1.95


def test_jump_diffusion_length_match():
    rng = np.random.default_rng(0)
    n = JumpDiffusionNoise(jump_intensity=0.1).sample(1000, rng)
    assert n.size == 1000


def test_jump_diffusion_more_heavy_with_jumps():
    rng = np.random.default_rng(0)
    a_low = mcculloch_alpha(JumpDiffusionNoise(sigma=0.5, jump_intensity=0.0, jump_scale=0.0).sample(8000, rng))
    rng = np.random.default_rng(0)
    a_high = mcculloch_alpha(JumpDiffusionNoise(sigma=0.5, jump_intensity=0.05, jump_scale=5.0).sample(8000, rng))
    assert a_high < a_low


def test_regime_switch_determinism():
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    a = RegimeSwitchNoise().sample(500, rng_a)
    b = RegimeSwitchNoise().sample(500, rng_b)
    assert np.allclose(a, b)
