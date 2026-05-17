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
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    neg = ~pos
    with np.errstate(over="ignore"):
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[neg])
        out[neg] = ez / (1.0 + ez)
    return out


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


@dataclass
class MLPClassifierTask:
    """Tiny 2-layer (1 hidden) tanh MLP, binary cross-entropy with logits.

    The parameter vector ``x`` is the flat concatenation
    ``[vec(W1), b1, W2, b2]`` with ``W1 in R^{h x d}``, ``b1 in R^h``,
    ``W2 in R^h``, ``b2 in R``. Gradient is exact backprop (finite-diff
    checked in tests). Used so the convergence study covers a non-convex
    classifier, not only the convex logistic model.
    """

    Z_train: np.ndarray
    y_train: np.ndarray
    Z_test: np.ndarray
    y_test: np.ndarray
    d_in: int
    hidden: int

    @property
    def dim(self) -> int:
        h, d = self.hidden, self.d_in
        return h * d + h + h + 1

    def _unpack(self, x: np.ndarray):
        h, d = self.hidden, self.d_in
        i = 0
        W1 = x[i:i + h * d].reshape(h, d); i += h * d
        b1 = x[i:i + h]; i += h
        W2 = x[i:i + h]; i += h
        b2 = float(x[i])
        return W1, b1, W2, b2

    def _resolve(self, Z, y):
        Zr = self.Z_train if Z is None else np.asarray(Z, dtype=float)
        yr = self.y_train if y is None else np.asarray(y, dtype=float)
        return Zr, yr

    def _forward(self, x, Z):
        W1, b1, W2, b2 = self._unpack(np.asarray(x, dtype=float))
        a = Z @ W1.T + b1
        hsig = np.tanh(a)
        logit = hsig @ W2 + b2
        return logit, hsig, a

    def loss(self, x: np.ndarray, Z=None, y=None) -> float:
        Z, y = self._resolve(Z, y)
        logit, _, _ = self._forward(x, Z)
        return float(np.mean(np.logaddexp(0.0, logit) - y * logit))

    def grad(self, x: np.ndarray, Z=None, y=None) -> np.ndarray:
        Z, y = self._resolve(Z, y)
        W1, b1, W2, b2 = self._unpack(np.asarray(x, dtype=float))
        logit, hsig, a = self._forward(x, Z)
        n = Z.shape[0]
        dlogit = (_sigmoid(logit) - y) / n
        gW2 = hsig.T @ dlogit
        gb2 = float(dlogit.sum())
        da = np.outer(dlogit, W2) * (1.0 - hsig ** 2)
        gW1 = da.T @ Z
        gb1 = da.sum(axis=0)
        return np.concatenate([gW1.ravel(), gb1, gW2, [gb2]])

    def sample_batch(self, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
        m = self.Z_train.shape[0]
        idx = rng.integers(0, m, size=min(int(n), m))
        return self.Z_train[idx], self.y_train[idx]

    def accuracy(self, x: np.ndarray, Z=None, y=None) -> float:
        Z, y = self._resolve(Z, y)
        logit, _, _ = self._forward(x, Z)
        return float(np.mean((logit > 0).astype(float) == y))


def make_mlp_classifier(n: int = 4000, d_in: int = 10, hidden: int = 8,
                        n_test: int = 2000, noise_scale: float = 0.5,
                        seed: int = 0) -> MLPClassifierTask:
    """Labels from a fixed random *teacher* tanh-MLP plus a quadratic term,
    so the boundary is genuinely nonlinear (logistic underfits it)."""
    rng = np.random.default_rng(seed)
    Z_tr = rng.normal(size=(n, d_in))
    Z_te = rng.normal(size=(n_test, d_in))
    Wt = rng.normal(size=(hidden, d_in)) / np.sqrt(d_in)
    vt = rng.normal(size=hidden)
    q = rng.normal(size=d_in) / np.sqrt(d_in)

    def teacher(Z):
        return np.tanh(Z @ Wt.T) @ vt + 0.5 * (Z @ q) ** 2 - 0.5

    y_tr = (teacher(Z_tr) + rng.normal(scale=noise_scale, size=n) > 0).astype(np.float64)
    y_te = (teacher(Z_te) + rng.normal(scale=noise_scale, size=n_test) > 0).astype(np.float64)
    return MLPClassifierTask(Z_train=Z_tr, y_train=y_tr, Z_test=Z_te,
                             y_test=y_te, d_in=int(d_in), hidden=int(hidden))


TASK_REGISTRY = {
    "quadratic": make_quadratic,
    "logistic": make_logistic,
    "mlp": make_mlp_classifier,
}
