from __future__ import annotations

import numpy as np
import pytest

from momo.tasks import (
    LogisticTask,
    MLPClassifierTask,
    QuadraticTask,
    TASK_REGISTRY,
    make_logistic,
    make_mlp_classifier,
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


def test_mlp_grad_finite_difference():
    task = make_mlp_classifier(n=300, d_in=6, hidden=5, n_test=100,
                               noise_scale=0.4, seed=3)
    rng = np.random.default_rng(0)
    x = rng.normal(scale=0.3, size=task.dim)
    g = task.grad(x)
    h = 1e-6
    g_num = np.zeros_like(g)
    for i in range(task.dim):
        e = np.zeros_like(x)
        e[i] = h
        g_num[i] = (task.loss(x + e) - task.loss(x - e)) / (2 * h)
    rel = np.linalg.norm(g - g_num) / max(np.linalg.norm(g_num), 1e-12)
    assert rel < 1e-5


def test_mlp_dim_matches_param_layout():
    task = make_mlp_classifier(n=100, d_in=7, hidden=4, n_test=50, seed=1)
    assert task.dim == 4 * 7 + 4 + 4 + 1
    W1, b1, W2, b2 = task._unpack(np.zeros(task.dim))
    assert W1.shape == (4, 7) and b1.shape == (4,) and W2.shape == (4,)
    assert isinstance(b2, float)


def test_mlp_trains_above_chance_and_beats_logistic():
    task = make_mlp_classifier(n=3000, d_in=8, hidden=10, n_test=1000,
                               noise_scale=0.3, seed=2)
    rng = np.random.default_rng(0)
    x = rng.normal(scale=0.1, size=task.dim)
    for _ in range(4000):
        x = x - 0.2 * task.grad(x)
    acc = task.accuracy(x, task.Z_test, task.y_test)
    assert acc > 0.65  # learns the nonlinear boundary
    # a linear logistic fit on the same features underfits this boundary
    w = np.zeros(task.Z_train.shape[1])
    for _ in range(4000):
        p = 1.0 / (1.0 + np.exp(-task.Z_train @ w))
        w = w - 0.2 * task.Z_train.T @ (p - task.y_train) / task.y_train.size
    lin_acc = np.mean((task.Z_test @ w > 0).astype(float) == task.y_test)
    assert acc >= lin_acc


def test_registry_keys():
    assert TASK_REGISTRY["quadratic"] is make_quadratic
    assert TASK_REGISTRY["logistic"] is make_logistic
    assert TASK_REGISTRY["mlp"] is make_mlp_classifier
