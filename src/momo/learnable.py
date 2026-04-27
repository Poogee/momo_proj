from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


DEFAULT_WEIGHTS_PATH = str(Path(__file__).resolve().parent.parent.parent / "models" / "learnable_filter.pt")


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 9):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        return self.act(x + h)


class DenoiserCNN(nn.Module):
    def __init__(self, channels: int = 48, kernel_size: int = 9, n_blocks: int = 2):
        super().__init__()
        pad = kernel_size // 2
        self.head = nn.Conv1d(1, channels, kernel_size=kernel_size, padding=pad)
        self.body = nn.Sequential(*[ResidualConvBlock(channels, kernel_size) for _ in range(n_blocks)])
        self.tail = nn.Conv1d(channels, 1, kernel_size=kernel_size, padding=pad)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        h = self.head(y)
        h = self.body(h)
        residual = self.tail(h)
        return y - residual


def _select_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class LearnableCNNFilter:
    weights_path: str = DEFAULT_WEIGHTS_PATH
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    window: int = 256
    channels: int = 48
    kernel_size: int = 9
    n_blocks: int = 2
    _model: Optional[nn.Module] = field(default=None, init=False, repr=False, compare=False)
    _device_resolved: Optional[str] = field(default=None, init=False, repr=False, compare=False)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        dev = _select_device(self.device)
        model = DenoiserCNN(channels=self.channels, kernel_size=self.kernel_size, n_blocks=self.n_blocks)
        path = Path(self.weights_path)
        if path.exists():
            state = torch.load(str(path), map_location=dev)
            model.load_state_dict(state)
        model.eval()
        model.to(dev)
        for p in model.parameters():
            p.requires_grad_(False)
        self._model = model
        self._device_resolved = dev

    @torch.no_grad()
    def apply(self, y: np.ndarray) -> np.ndarray:
        y_arr = np.asarray(y, dtype=np.float32).ravel()
        n = y_arr.size
        if n == 0:
            return y_arr.astype(float)
        self._ensure_loaded()
        w = int(self.window)
        if n <= w:
            pad_len = w - n
            x = np.concatenate([y_arr, np.zeros(pad_len, dtype=np.float32)]) if pad_len > 0 else y_arr
            t = torch.from_numpy(x).reshape(1, 1, w).to(self._device_resolved)
            out = self._model(t).reshape(-1).detach().cpu().numpy()
            return out[:n].astype(float)
        stride = w // 2
        starts = list(range(0, n - w + 1, stride))
        if starts[-1] != n - w:
            starts.append(n - w)
        accum = np.zeros(n, dtype=np.float64)
        weight = np.zeros(n, dtype=np.float64)
        win_w = np.hanning(w).astype(np.float64) + 1e-3
        chunks = []
        idxs = []
        batch_cap = 64
        for s in starts:
            chunks.append(y_arr[s:s + w])
            idxs.append(s)
            if len(chunks) >= batch_cap:
                self._run_batch(chunks, idxs, accum, weight, win_w)
                chunks = []
                idxs = []
        if chunks:
            self._run_batch(chunks, idxs, accum, weight, win_w)
        return (accum / np.maximum(weight, 1e-12)).astype(float)

    def _run_batch(self, chunks, idxs, accum, weight, win_w):
        batch = np.stack(chunks).astype(np.float32)
        t = torch.from_numpy(batch).unsqueeze(1).to(self._device_resolved)
        out = self._model(t).squeeze(1).detach().cpu().numpy()
        w = batch.shape[1]
        for k, s in enumerate(idxs):
            accum[s:s + w] += out[k].astype(np.float64) * win_w
            weight[s:s + w] += win_w


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
