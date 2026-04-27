from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import levy_stable


def _fractional_weights(d: float, length: int) -> np.ndarray:
    weights = np.empty(length, dtype=float)
    weights[0] = 1.0
    for j in range(1, length):
        weights[j] = weights[j - 1] * (j - 1 + d) / j
    return weights


def _stable_random_state(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 2**31 - 1))


@dataclass(frozen=True)
class GaussianNoise:
    sigma: float = 1.0

    def sample(self, T: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(loc=0.0, scale=self.sigma, size=T)


@dataclass(frozen=True)
class PinkFARIMANoise:
    d: float = 0.3
    sigma: float = 1.0
    truncation: int = 2000

    def sample(self, T: int, rng: np.random.Generator) -> np.ndarray:
        weights = _fractional_weights(self.d, self.truncation)
        innovations = rng.normal(loc=0.0, scale=self.sigma, size=T + self.truncation)
        filtered = np.convolve(innovations, weights, mode="full")
        return filtered[self.truncation : self.truncation + T]


@dataclass(frozen=True)
class StableNoise:
    alpha: float = 1.7
    sigma: float = 1.0

    def sample(self, T: int, rng: np.random.Generator) -> np.ndarray:
        seed = _stable_random_state(rng)
        return levy_stable.rvs(
            alpha=self.alpha,
            beta=0.0,
            scale=self.sigma,
            size=T,
            random_state=seed,
        )


@dataclass(frozen=True)
class MixedFARIMAStableNoise:
    d: float = 0.3
    alpha: float = 1.7
    sigma: float = 1.0
    truncation: int = 2000

    def sample(self, T: int, rng: np.random.Generator) -> np.ndarray:
        weights = _fractional_weights(self.d, self.truncation)
        seed = _stable_random_state(rng)
        innovations = levy_stable.rvs(
            alpha=self.alpha,
            beta=0.0,
            scale=self.sigma,
            size=T + self.truncation,
            random_state=seed,
        )
        filtered = np.convolve(innovations, weights, mode="full")
        return filtered[self.truncation : self.truncation + T]


@dataclass(frozen=True)
class RegimeSwitchNoise:
    sigma: float = 1.0
    alpha: float = 1.6
    block_length: int = 256

    def sample(self, T: int, rng: np.random.Generator) -> np.ndarray:
        out = np.empty(T, dtype=float)
        i = 0
        regime = 0
        while i < T:
            n = min(self.block_length, T - i)
            if regime == 0:
                out[i:i + n] = rng.normal(0.0, self.sigma, size=n)
            else:
                seed = _stable_random_state(rng)
                out[i:i + n] = levy_stable.rvs(
                    alpha=self.alpha, beta=0.0, scale=self.sigma,
                    size=n, random_state=seed,
                )
            regime = 1 - regime
            i += n
        return out


@dataclass(frozen=True)
class JumpDiffusionNoise:
    sigma: float = 1.0
    jump_intensity: float = 0.02
    jump_scale: float = 5.0

    def sample(self, T: int, rng: np.random.Generator) -> np.ndarray:
        diff = rng.normal(0.0, self.sigma, size=T)
        n_jumps = rng.binomial(1, self.jump_intensity, size=T)
        jumps = n_jumps * rng.normal(0.0, self.jump_scale, size=T)
        return diff + jumps


@dataclass(frozen=True)
class HawkesClusteredJumpNoise:
    sigma: float = 1.0
    base_intensity: float = 0.005
    self_excitation: float = 0.5
    decay: float = 0.05
    jump_scale: float = 4.0

    def sample(self, T: int, rng: np.random.Generator) -> np.ndarray:
        diff = rng.normal(0.0, self.sigma, size=T)
        intensity = float(self.base_intensity)
        jumps = np.zeros(T, dtype=float)
        for t in range(T):
            if rng.uniform() < intensity:
                jumps[t] = float(rng.normal(0.0, self.jump_scale))
                intensity = min(0.95, intensity + self.self_excitation)
            intensity = self.base_intensity + (intensity - self.base_intensity) * (1.0 - self.decay)
        return diff + jumps


NOISE_REGISTRY = {
    "N1": GaussianNoise,
    "N2": PinkFARIMANoise,
    "N3": StableNoise,
    "N4": MixedFARIMAStableNoise,
    "N5": RegimeSwitchNoise,
    "N6": JumpDiffusionNoise,
    "N7": HawkesClusteredJumpNoise,
}
