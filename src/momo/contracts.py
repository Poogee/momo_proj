from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


NoiseGenerator = Callable[[int, np.random.Generator], np.ndarray]
Filter = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class NoiseSpec:
    name: str
    params: dict


@dataclass(frozen=True)
class FilterSpec:
    name: str
    params: dict


@dataclass(frozen=True)
class RunConfig:
    task: str
    noise: NoiseSpec
    filter: FilterSpec
    optimizer: str
    lr: float
    steps: int
    seed: int
    batch_size: int = 32
    epsilon: float = 1e-3
    extra: dict | None = None


class TaskProtocol(Protocol):
    dim: int

    def loss(self, x: np.ndarray, batch: np.ndarray) -> float: ...

    def grad(self, x: np.ndarray, batch: np.ndarray) -> np.ndarray: ...

    def sample_batch(self, rng: np.random.Generator, n: int) -> np.ndarray: ...
