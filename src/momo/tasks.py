from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QuadraticTask:
    A: np.ndarray
    b: np.ndarray

    @property
    def dim(self) -> int:
        return int(self.A.shape[0])

    def loss(self, x: np.ndarray) -> float:
        return float(0.5 * x @ self.A @ x - self.b @ x)

    def grad(self, x: np.ndarray) -> np.ndarray:
        return self.A @ x - self.b

    def optimum(self) -> np.ndarray:
        return np.linalg.solve(self.A, self.b)


def make_quadratic(dim: int = 50, condition_number: float = 10.0,
                   seed: int = 0) -> QuadraticTask:
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    eigenvalues = np.geomspace(1.0, float(condition_number), dim)
    A = (Q * eigenvalues) @ Q.T
    A = 0.5 * (A + A.T)
    b = rng.normal(size=dim)
    norm = np.linalg.norm(b)
    if norm > 0:
        b = b / norm * rng.uniform(0.5, 1.0)
    return QuadraticTask(A=A, b=b)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


def _logistic_loss(x: np.ndarray, Z: np.ndarray, y: np.ndarray) -> float:
    z = Z @ x
    log1p = np.logaddexp(0.0, z)
    return float(np.mean(log1p - y * z))


def _logistic_grad(x: np.ndarray, Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = _sigmoid(Z @ x)
    return Z.T @ (p - y) / Z.shape[0]


@dataclass
class LogisticTask:
    Z_train: np.ndarray
    y_train: np.ndarray
    Z_test: np.ndarray
    y_test: np.ndarray
    x_star: np.ndarray

    @property
    def dim(self) -> int:
        return int(self.Z_train.shape[1])

    def loss(self, x: np.ndarray, Z=None, y=None) -> float:
        if Z is None:
            Z = self.Z_train
        if y is None:
            y = self.y_train
        return _logistic_loss(x, Z, y)

    def grad(self, x: np.ndarray, Z=None, y=None) -> np.ndarray:
        if Z is None:
            Z = self.Z_train
        if y is None:
            y = self.y_train
        return _logistic_grad(x, Z, y)

    def sample_batch(self, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
        idx = rng.integers(0, self.Z_train.shape[0], size=n)
        return self.Z_train[idx], self.y_train[idx]


def make_logistic(n: int = 5000, dim: int = 20, n_test: int = 2000,
                  noise_scale: float = 0.5, seed: int = 0) -> LogisticTask:
    rng = np.random.default_rng(seed)
    Z_train = rng.normal(size=(n, dim))
    Z_test = rng.normal(size=(n_test, dim))
    x_star = rng.normal(size=dim)
    x_star = x_star / np.linalg.norm(x_star)
    eps_train = rng.normal(scale=noise_scale, size=n)
    eps_test = rng.normal(scale=noise_scale, size=n_test)
    y_train = (Z_train @ x_star + eps_train > 0).astype(np.float64)
    y_test = (Z_test @ x_star + eps_test > 0).astype(np.float64)
    return LogisticTask(
        Z_train=Z_train,
        y_train=y_train,
        Z_test=Z_test,
        y_test=y_test,
        x_star=x_star,
    )


TASK_REGISTRY = {"quadratic": make_quadratic, "logistic": make_logistic}
