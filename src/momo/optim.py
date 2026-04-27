from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from momo.tasks import LogisticTask, QuadraticTask


@dataclass
class OptimResult:
    x_history: np.ndarray
    grad_norm_sq_history: np.ndarray
    loss_history: np.ndarray
    x_final: np.ndarray
    config: dict


def _stochastic_grad(task: Any, x: np.ndarray, rng: np.random.Generator,
                     noise: Any, noise_scale: float, batch_size: int,
                     xi_override: np.ndarray | None = None) -> np.ndarray:
    dim = x.size
    xi = xi_override if xi_override is not None else noise.sample(dim, rng) * noise_scale
    if isinstance(task, QuadraticTask):
        return task.A @ x - task.b + xi
    if isinstance(task, LogisticTask):
        Z_b, y_b = task.sample_batch(rng, batch_size)
        return task.grad(x, Z_b, y_b) + xi
    return task.grad(x) + xi


def _build_noise_trajectory(noise: Any, steps: int, dim: int, rng: np.random.Generator,
                            noise_scale: float) -> np.ndarray:
    cols = [noise.sample(steps, rng) for _ in range(dim)]
    return np.column_stack(cols) * noise_scale


def _prefilter_columns(traj: np.ndarray, filt: Any) -> np.ndarray:
    out = np.empty_like(traj)
    for j in range(traj.shape[1]):
        out[:, j] = filt.apply(traj[:, j])
    return out


def _filter_buffer(buffer: np.ndarray, count: int, filt: Any) -> np.ndarray:
    capacity, dim = buffer.shape
    n = min(count, capacity)
    if n == 1:
        return buffer[0].copy() if count <= capacity else buffer[(count - 1) % capacity].copy()
    if count <= capacity:
        view = buffer[:n]
    else:
        start = count % capacity
        view = np.concatenate([buffer[start:], buffer[:start]], axis=0)
    out = np.empty(dim, dtype=float)
    for j in range(dim):
        filtered = filt.apply(view[:, j])
        out[j] = float(filtered[-1])
    return out


def _sgd_step(x, g, state, lr, weight_decay):
    return x - lr * g


def _adam_step(x, g, state, lr, weight_decay):
    state["t"] += 1
    t = state["t"]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    state["m"] = beta1 * state["m"] + (1 - beta1) * g
    state["v"] = beta2 * state["v"] + (1 - beta2) * (g * g)
    m_hat = state["m"] / (1 - beta1 ** t)
    v_hat = state["v"] / (1 - beta2 ** t)
    return x - lr * m_hat / (np.sqrt(v_hat) + eps)


def _adamw_step(x, g, state, lr, weight_decay):
    state["t"] += 1
    t = state["t"]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    state["m"] = beta1 * state["m"] + (1 - beta1) * g
    state["v"] = beta2 * state["v"] + (1 - beta2) * (g * g)
    m_hat = state["m"] / (1 - beta1 ** t)
    v_hat = state["v"] / (1 - beta2 ** t)
    return x - lr * (m_hat / (np.sqrt(v_hat) + eps) + weight_decay * x)


_OPTIMIZERS = {"sgd": _sgd_step, "adam": _adam_step, "adamw": _adamw_step}


def _make_state(name: str, dim: int) -> dict:
    if name == "sgd":
        return {}
    return {"m": np.zeros(dim), "v": np.zeros(dim), "t": 0}


def run_optimization(
    task: Any,
    optimizer: str,
    noise: Any,
    filt: Any,
    steps: int = 2000,
    lr: float = 1e-2,
    batch_size: int = 32,
    seed: int = 0,
    buffer_size: int = 32,
    weight_decay: float = 1e-4,
    log_every: int = 1,
    noise_scale: float = 1.0,
    preprocess_mode: str = "buffer",
) -> OptimResult:
    rng = np.random.default_rng(seed)
    dim = task.dim
    x = rng.normal(scale=0.01, size=dim)
    state = _make_state(optimizer, dim)
    step_fn = _OPTIMIZERS[optimizer]

    x_history = np.empty((steps + 1, dim), dtype=float)
    grad_norm_sq_history = np.empty(steps, dtype=float)
    loss_history = np.empty(steps, dtype=float)
    x_history[0] = x

    if preprocess_mode == "data":
        traj = _build_noise_trajectory(noise, steps, dim, rng, noise_scale)
        filt_traj = _prefilter_columns(traj, filt)
        for k in range(steps):
            g = _stochastic_grad(task, x, rng, noise, noise_scale, batch_size,
                                 xi_override=filt_traj[k])
            x = step_fn(x, g, state, lr, weight_decay)
            x_history[k + 1] = x
            true_grad = task.grad(x)
            grad_norm_sq_history[k] = float(true_grad @ true_grad)
            loss_history[k] = float(task.loss(x))
    elif preprocess_mode == "buffer":
        buffer = np.zeros((buffer_size, dim), dtype=float)
        count = 0
        for k in range(steps):
            g_raw = _stochastic_grad(task, x, rng, noise, noise_scale, batch_size)
            slot = count % buffer_size
            buffer[slot] = g_raw
            count += 1
            g_filtered = _filter_buffer(buffer, count, filt)
            x = step_fn(x, g_filtered, state, lr, weight_decay)
            x_history[k + 1] = x
            true_grad = task.grad(x)
            grad_norm_sq_history[k] = float(true_grad @ true_grad)
            loss_history[k] = float(task.loss(x))
    else:
        raise ValueError(f"unknown preprocess_mode: {preprocess_mode}")

    config = {
        "optimizer": optimizer,
        "steps": steps,
        "lr": lr,
        "batch_size": batch_size,
        "seed": seed,
        "buffer_size": buffer_size,
        "weight_decay": weight_decay,
        "noise_scale": noise_scale,
        "log_every": log_every,
        "dim": dim,
        "preprocess_mode": preprocess_mode,
    }
    return OptimResult(
        x_history=x_history,
        grad_norm_sq_history=grad_norm_sq_history,
        loss_history=loss_history,
        x_final=x,
        config=config,
    )
