from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from momo.metrics import mcculloch_alpha


@dataclass
class AlphaAwareClipper:
    window: int = 256
    refresh_every: int = 32
    base_scale: float = 5.0
    min_alpha: float = 1.05
    max_alpha: float = 2.0
    _buf: deque = field(default_factory=deque)
    _step: int = 0
    _alpha_hat: float = 2.0
    _scale: float = 0.0

    def update(self, g: np.ndarray) -> np.ndarray:
        self._buf.append(float(np.linalg.norm(g)))
        if len(self._buf) > self.window:
            self._buf.popleft()
        self._step += 1
        if self._step % self.refresh_every == 0 and len(self._buf) >= self.window // 2:
            arr = np.array(self._buf)
            est = mcculloch_alpha(arr - arr.mean())
            if not np.isnan(est):
                self._alpha_hat = float(np.clip(est, self.min_alpha, self.max_alpha))
            q = float(np.quantile(arr, 0.9))
            self._scale = self.base_scale * q * (2.0 / self._alpha_hat)
        if self._scale <= 0:
            return g
        norm = float(np.linalg.norm(g))
        if norm <= self._scale:
            return g
        return g * (self._scale / norm)

    @property
    def alpha_hat(self) -> float:
        return self._alpha_hat

    @property
    def clip_scale(self) -> float:
        return self._scale
