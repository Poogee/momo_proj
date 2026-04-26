from __future__ import annotations

import numpy as np
import pytest

from momo.filters import IdentityFilter, MovingAverageFilter
from momo.noise import GaussianNoise
from momo.optim import run_optimization
from momo.tasks import make_logistic, make_quadratic


def test_sgd_quadratic_converges():
    task = make_quadratic(dim=20, condition_number=5.0, seed=0)
    noise = GaussianNoise(sigma=0.05)
    filt = IdentityFilter()
    res = run_optimization(
        task=task, optimizer="sgd", noise=noise, filt=filt,
        steps=2000, lr=5e-2, seed=1, buffer_size=8, noise_scale=1.0,
    )
    tail = res.grad_norm_sq_history[-50:]
    assert np.median(tail) < 1e-2


def test_adam_quadratic_converges():
    task = make_quadratic(dim=20, condition_number=5.0, seed=0)
    noise = GaussianNoise(sigma=0.01)
    filt = IdentityFilter()
    res = run_optimization(
        task=task, optimizer="adam", noise=noise, filt=filt,
        steps=2000, lr=1e-2, seed=1, buffer_size=8, noise_scale=1.0,
    )
    tail = res.grad_norm_sq_history[-50:]
    assert np.median(tail) < 1e-2


def test_logistic_adam_test_accuracy():
    task = make_logistic(n=4000, dim=15, n_test=1500, noise_scale=0.3, seed=2)
    noise = GaussianNoise(sigma=0.01)
    filt = IdentityFilter()
    res = run_optimization(
        task=task, optimizer="adam", noise=noise, filt=filt,
        steps=2000, lr=5e-2, batch_size=64, seed=3, buffer_size=8, noise_scale=1.0,
    )
    logits = task.Z_test @ res.x_final
    pred = (logits > 0).astype(np.float64)
    acc = float(np.mean(pred == task.y_test))
    assert acc > 0.85


def test_determinism_same_seed():
    task = make_quadratic(dim=10, condition_number=3.0, seed=0)
    noise = GaussianNoise(sigma=0.1)
    filt = IdentityFilter()
    res1 = run_optimization(
        task=task, optimizer="adam", noise=noise, filt=filt,
        steps=200, lr=1e-2, seed=42, buffer_size=8,
    )
    res2 = run_optimization(
        task=task, optimizer="adam", noise=noise, filt=filt,
        steps=200, lr=1e-2, seed=42, buffer_size=8,
    )
    assert np.array_equal(res1.x_history, res2.x_history)
    assert np.array_equal(res1.grad_norm_sq_history, res2.grad_norm_sq_history)
    assert np.array_equal(res1.loss_history, res2.loss_history)
    assert np.array_equal(res1.x_final, res2.x_final)


def test_filter_changes_results():
    task = make_quadratic(dim=10, condition_number=3.0, seed=0)
    noise = GaussianNoise(sigma=0.5)
    res_f0 = run_optimization(
        task=task, optimizer="sgd", noise=noise, filt=IdentityFilter(),
        steps=200, lr=1e-2, seed=7, buffer_size=16,
    )
    res_f1 = run_optimization(
        task=task, optimizer="sgd", noise=noise, filt=MovingAverageFilter(window=8),
        steps=200, lr=1e-2, seed=7, buffer_size=16,
    )
    assert not np.array_equal(res_f0.x_history, res_f1.x_history)


def test_adamw_weight_decay_shrinks_x():
    task = make_quadratic(dim=20, condition_number=5.0, seed=0)
    noise = GaussianNoise(sigma=0.1)
    filt = IdentityFilter()
    res_adam = run_optimization(
        task=task, optimizer="adam", noise=noise, filt=filt,
        steps=500, lr=5e-2, seed=11, buffer_size=8, weight_decay=0.0,
    )
    res_adamw = run_optimization(
        task=task, optimizer="adamw", noise=noise, filt=filt,
        steps=500, lr=5e-2, seed=11, buffer_size=8, weight_decay=0.1,
    )
    assert np.linalg.norm(res_adamw.x_final) < np.linalg.norm(res_adam.x_final)
