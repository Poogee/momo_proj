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


NOISE_REGISTRY = {
    "N1": GaussianNoise,
    "N2": PinkFARIMANoise,
    "N3": StableNoise,
    "N4": MixedFARIMAStableNoise,
}
