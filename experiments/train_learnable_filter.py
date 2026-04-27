from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from momo.learnable import DenoiserCNN, count_parameters


def gen_clean_signals(batch: int, length: int, rng: np.random.Generator) -> np.ndarray:
    out = np.empty((batch, length), dtype=np.float32)
    families = rng.integers(0, 4, size=batch)
    t = np.arange(length, dtype=np.float32)
    for i in range(batch):
        fam = int(families[i])
        amp = float(rng.uniform(0.5, 2.0))
        if fam == 0:
            n_components = int(rng.integers(1, 4))
            sig = np.zeros(length, dtype=np.float32)
            for _ in range(n_components):
                period = float(rng.uniform(20.0, 200.0))
                phase = float(rng.uniform(0.0, 2.0 * math.pi))
                sig += np.sin(2.0 * math.pi * t / period + phase).astype(np.float32)
            out[i] = amp * sig / max(1, n_components)
        elif fam == 1:
            slope = float(rng.uniform(-2.0, 2.0)) / length
            intercept = float(rng.uniform(-1.0, 1.0))
            curv = float(rng.uniform(-1.0, 1.0)) / (length ** 2)
            out[i] = (intercept + slope * t + curv * t * t).astype(np.float32) * amp
        elif fam == 2:
            steps = rng.normal(0.0, 0.05, size=length).astype(np.float32)
            walk = np.cumsum(steps)
            kernel = np.ones(15, dtype=np.float32) / 15.0
            walk = np.convolve(walk, kernel, mode="same")
            out[i] = amp * walk
        else:
            n_seg = int(rng.integers(2, 7))
            edges = np.sort(rng.integers(1, length, size=n_seg - 1))
            edges = np.concatenate([[0], edges, [length]])
            sig = np.zeros(length, dtype=np.float32)
            for k in range(len(edges) - 1):
                sig[edges[k]:edges[k + 1]] = float(rng.uniform(-1.0, 1.0))
            out[i] = amp * sig
    return out


def _stable_noise_cms(alpha: float, size: tuple, rng: np.random.Generator) -> np.ndarray:
    U = rng.uniform(-math.pi / 2.0, math.pi / 2.0, size=size)
    W = rng.exponential(1.0, size=size)
    if abs(alpha - 1.0) < 1e-3:
        return np.tan(U).astype(np.float32)
    zeta = 0.0
    factor = (np.sin(alpha * U) / (np.cos(U) ** (1.0 / alpha)))
    bracket = ((np.cos(U - alpha * U)) / W) ** ((1.0 - alpha) / alpha)
    return (factor * bracket).astype(np.float32)


def _farima_pink(d: float, sigma: float, batch: int, length: int, rng: np.random.Generator, trunc: int = 256) -> np.ndarray:
    weights = np.empty(trunc, dtype=np.float32)
    weights[0] = 1.0
    for j in range(1, trunc):
        weights[j] = weights[j - 1] * (j - 1 + d) / j
    inn = rng.normal(0.0, sigma, size=(batch, length + trunc)).astype(np.float32)
    out = np.empty((batch, length), dtype=np.float32)
    for i in range(batch):
        f = np.convolve(inn[i], weights, mode="full")
        out[i] = f[trunc:trunc + length]
    return out


def gen_noise(batch: int, length: int, rng: np.random.Generator) -> np.ndarray:
    out = np.empty((batch, length), dtype=np.float32)
    types = rng.integers(0, 4, size=batch)
    log_lo, log_hi = math.log(0.05), math.log(1.0)
    for i in range(batch):
        sigma = float(math.exp(rng.uniform(log_lo, log_hi)))
        kind = int(types[i])
        if kind == 0:
            out[i] = rng.normal(0.0, sigma, size=length).astype(np.float32)
        elif kind == 1:
            d = float(rng.uniform(0.1, 0.45))
            n = _farima_pink(d, sigma, 1, length, rng).reshape(length)
            scale = sigma / max(1e-6, float(np.std(n)))
            out[i] = n * scale
        elif kind == 2:
            alpha = float(rng.uniform(1.4, 1.95))
            n = _stable_noise_cms(alpha, (length,), rng)
            n = np.clip(n, -50.0, 50.0)
            scale = sigma / max(1e-6, float(np.median(np.abs(n)) + 1e-6))
            out[i] = (n * scale).astype(np.float32)
        else:
            d = float(rng.uniform(0.1, 0.45))
            alpha = float(rng.uniform(1.5, 1.95))
            base = _stable_noise_cms(alpha, (length,), rng)
            base = np.clip(base, -50.0, 50.0)
            weights = np.empty(256, dtype=np.float32)
            weights[0] = 1.0
            for j in range(1, 256):
                weights[j] = weights[j - 1] * (j - 1 + d) / j
            base_pad = np.concatenate([np.zeros(256, dtype=np.float32), base])
            f = np.convolve(base_pad, weights, mode="full")[256:256 + length]
            scale = sigma / max(1e-6, float(np.median(np.abs(f)) + 1e-6))
            out[i] = (f * scale).astype(np.float32)
    return out


def make_batch(batch: int, length: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    clean = gen_clean_signals(batch, length, rng)
    noise = gen_noise(batch, length, rng)
    noisy = clean + noise
    return noisy, clean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--kernel", type=int, default=9)
    parser.add_argument("--out", type=str, default=str(ROOT / "models" / "learnable_filter.pt"))
    parser.add_argument("--log-every", type=int, default=500)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"[device] using {device}")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    model = DenoiserCNN(channels=args.channels, kernel_size=args.kernel, n_blocks=args.blocks).to(device)
    n_params = count_parameters(model)
    print(f"[model] params: {n_params}")

    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = CosineAnnealingLR(optim, T_max=args.steps, eta_min=args.lr * 0.05)
    loss_fn = nn.HuberLoss(delta=0.5)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model.train()
    t0 = time.time()
    running = 0.0
    running_mse = 0.0
    running_n = 0
    epoch_size = max(1, args.log_every)
    log_steps = []
    log_loss = []
    for step in range(1, args.steps + 1):
        noisy_np, clean_np = make_batch(args.batch, args.window, rng)
        noisy = torch.from_numpy(noisy_np).unsqueeze(1).to(device, non_blocking=True)
        clean = torch.from_numpy(clean_np).unsqueeze(1).to(device, non_blocking=True)
        pred = model(noisy)
        loss = loss_fn(pred, clean)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optim.step()
        sched.step()
        with torch.no_grad():
            mse = float(((pred - clean) ** 2).mean().item())
        running += float(loss.item())
        running_mse += mse
        running_n += 1
        if step % epoch_size == 0:
            elapsed = time.time() - t0
            avg_loss = running / running_n
            avg_mse = running_mse / running_n
            log_steps.append(step)
            log_loss.append(avg_mse)
            print(f"[step {step:6d}/{args.steps}] huber={avg_loss:.5f} mse={avg_mse:.5f} lr={sched.get_last_lr()[0]:.2e} elapsed={elapsed:.1f}s")
            running = 0.0
            running_mse = 0.0
            running_n = 0

    elapsed = time.time() - t0
    if device == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"[gpu] peak VRAM allocated: {peak_mb:.1f} MB")
    print(f"[done] total time: {elapsed:.1f}s ({elapsed/60.0:.2f} min)")

    torch.save(model.state_dict(), str(out_path))
    print(f"[save] weights -> {out_path}")

    model.eval()
    val_rng = np.random.default_rng(args.seed + 999)
    with torch.no_grad():
        noisy_np, clean_np = make_batch(256, args.window, val_rng)
        noisy = torch.from_numpy(noisy_np).unsqueeze(1).to(device)
        clean = torch.from_numpy(clean_np).unsqueeze(1).to(device)
        pred = model(noisy)
        val_mse = float(((pred - clean) ** 2).mean().item())
        baseline_mse = float(((noisy - clean) ** 2).mean().item())
    print(f"[val] mse_in={baseline_mse:.5f} mse_out={val_mse:.5f}")


if __name__ == "__main__":
    main()
