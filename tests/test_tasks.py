from __future__ import annotations

import numpy as np
import pytest

from momo.tasks import (
    LogisticTask,
    QuadraticTask,
    TASK_REGISTRY,
    make_logistic,
    make_quadratic,
)


def test_quadratic_grad_at_optimum_is_zero():
    task = make_quadratic(dim=20, condition_number=10.0, seed=0)
    x_star = task.optimum()
    g = task.grad(x_star)
    assert np.linalg.norm(g) < 1e-10


def test_quadratic_optimum_satisfies_normal_equation():
    task = make_quadratic(dim=15, condition_number=20.0, seed=1)
    x_star = task.optimum()
    assert np.allclose(task.A @ x_star, task.b, atol=1e-10)


def test_quadratic_condition_number_in_eigenvalues():
    cn = 50.0
    task = make_quadratic(dim=30, condition_number=cn, seed=2)
    eigs = np.linalg.eigvalsh(task.A)
    ratio = eigs.max() / eigs.min()
    assert abs(ratio - cn) / cn < 1e-6


def test_logistic_loss_decreases_toward_xstar():
    task = make_logistic(n=2000, dim=10, n_test=500, noise_scale=0.3, seed=3)
    x0 = np.zeros(task.dim)
    losses = [task.loss(x0 + alpha * task.x_star) for alpha in np.linspace(0.0, 1.0, 6)]
    assert losses[-1] < losses[0]
    assert losses[-1] < losses[1]


def test_logistic_sample_batch_shapes():
    task = make_logistic(n=1000, dim=8, n_test=200, seed=4)
    rng = np.random.default_rng(0)
    Z_b, y_b = task.sample_batch(rng, 32)
    assert Z_b.shape == (32, 8)
    assert y_b.shape == (32,)
    assert set(np.unique(y_b).tolist()).issubset({0.0, 1.0})


def test_registry_keys():
    assert TASK_REGISTRY["quadratic"] is make_quadratic
    assert TASK_REGISTRY["logistic"] is make_logistic
