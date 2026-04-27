from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
import pywt
from scipy.signal import medfilt


@dataclass(frozen=True)
class IdentityFilter:
    def apply(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=float).copy()


@dataclass(frozen=True)
class MovingAverageFilter:
    window: int = 20

    def apply(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        w = int(self.window)
        if w <= 1:
            return y.copy()
        n = y.size
        pad = w - 1
        padded = np.pad(y, (pad, 0), mode="reflect")
        kernel = np.ones(w, dtype=float) / w
        out = np.convolve(padded, kernel, mode="valid")
        return out[:n]


@dataclass(frozen=True)
class KalmanLocalLevelFilter:
    process_var: float = 1e-4
    obs_var: float = 1.0

    def apply(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        n = y.size
        if n == 0:
            return y.copy()
        Q = float(self.process_var)
        R = float(self.obs_var)
        out = np.empty(n, dtype=float)
        x = float(y[0])
        P = R
        out[0] = x
        for t in range(1, n):
            P_pred = P + Q
            K = P_pred / (P_pred + R)
            x = x + K * (y[t] - x)
            P = (1.0 - K) * P_pred
            out[t] = x
        return out


@dataclass(frozen=True)
class WaveletThresholdFilter:
    wavelet: str = "db4"
    level: Union[int, None] = None
    mode: str = "soft"
    threshold: Union[str, float] = "universal"

    def apply(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        n = y.size
        if n < 2:
            return y.copy()
        wavelet = pywt.Wavelet(self.wavelet)
        max_level = pywt.dwt_max_level(n, wavelet.dec_len)
        level = max_level if self.level is None else min(int(self.level), max_level)
        if level < 1:
            return y.copy()
        coeffs = pywt.wavedec(y, wavelet, level=level, mode="symmetric")
        detail_finest = coeffs[-1]
        if isinstance(self.threshold, (int, float)):
            lam = float(self.threshold)
        else:
            sigma_hat = float(np.median(np.abs(detail_finest - np.median(detail_finest))) / 0.6745)
            lam = sigma_hat * float(np.sqrt(2.0 * np.log(max(n, 2))))
        new_coeffs = [coeffs[0]]
        for d in coeffs[1:]:
            with np.errstate(invalid="ignore", divide="ignore"):
                new_coeffs.append(np.nan_to_num(pywt.threshold(d, lam, mode=self.mode), nan=0.0))
        rec = pywt.waverec(new_coeffs, wavelet, mode="symmetric")
        return np.asarray(rec[:n], dtype=float)


@dataclass(frozen=True)
class MedianFilter:
    window: int = 21

    def apply(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        w = int(self.window)
        if w < 1:
            w = 1
        if w % 2 == 0:
            w += 1
        if w == 1:
            return y.copy()
        n = y.size
        pad = w // 2
        padded = np.pad(y, (pad, pad), mode="reflect")
        out = medfilt(padded, kernel_size=w)
        return out[pad:pad + n]


@dataclass(frozen=True)
class AdaptiveWaveletFilter:
    wavelet: str = "db4"
    level: Union[int, None] = None
    mode: str = "soft"
    base_factor: float = 1.0
    alpha_boost_exp: float = 2.0

    def apply(self, y: np.ndarray) -> np.ndarray:
        from momo.metrics import mcculloch_alpha
        y = np.asarray(y, dtype=float)
        n = y.size
        if n < 64:
            return y.copy()
        wavelet = pywt.Wavelet(self.wavelet)
        max_level = pywt.dwt_max_level(n, wavelet.dec_len)
        level = max_level if self.level is None else min(int(self.level), max_level)
        if level < 1:
            return y.copy()
        coeffs = pywt.wavedec(y, wavelet, level=level, mode="symmetric")
        detail_finest = coeffs[-1]
        q25, q75 = np.quantile(detail_finest, [0.25, 0.75])
        scale = float((q75 - q25) / 1.349)
        if scale <= 0:
            scale = float(np.median(np.abs(detail_finest - np.median(detail_finest))) / 0.6745)
        alpha_hat = mcculloch_alpha(y)
        if not np.isfinite(alpha_hat) or alpha_hat <= 0:
            alpha_hat = 2.0
        alpha_hat = float(np.clip(alpha_hat, 1.05, 2.0))
        boost = (2.0 / alpha_hat) ** float(self.alpha_boost_exp)
        lam = self.base_factor * scale * float(np.sqrt(2.0 * np.log(max(n, 2)))) * boost
        new_coeffs = [coeffs[0]]
        for d in coeffs[1:]:
            with np.errstate(invalid="ignore", divide="ignore"):
                new_coeffs.append(np.nan_to_num(pywt.threshold(d, lam, mode=self.mode), nan=0.0))
        rec = pywt.waverec(new_coeffs, wavelet, mode="symmetric")
        return np.asarray(rec[:n], dtype=float)


@dataclass(frozen=True)
class HybridMedianWaveletFilter:
    median_window: int = 5
    wavelet: str = "db4"
    mode: str = "soft"

    def apply(self, y: np.ndarray) -> np.ndarray:
        pre = MedianFilter(window=self.median_window).apply(y)
        return AdaptiveWaveletFilter(wavelet=self.wavelet, mode=self.mode).apply(pre)


@dataclass(frozen=True)
class AdaptiveMetaFilter:
    alpha_threshold: float = 1.9
    hurst_residual_threshold: float = 0.65
    min_length: int = 64

    def apply(self, y: np.ndarray) -> np.ndarray:
        from momo.metrics import hurst_dfa, mcculloch_alpha
        y = np.asarray(y, dtype=float)
        n = y.size
        if n < self.min_length:
            return MedianFilter(window=min(7, max(3, n // 4 * 2 + 1))).apply(y)
        diff = np.diff(y)
        alpha_hat = mcculloch_alpha(diff)
        if not np.isfinite(alpha_hat):
            alpha_hat = 2.0
        if alpha_hat < self.alpha_threshold:
            return HybridMedianWaveletFilter(median_window=5).apply(y)
        h_hat = hurst_dfa(diff)
        if not np.isfinite(h_hat):
            h_hat = 0.5
        if h_hat > self.hurst_residual_threshold:
            return AdaptiveWaveletFilter().apply(y)
        return KalmanLocalLevelFilter(process_var=1e-3, obs_var=1.0).apply(y)


FILTER_REGISTRY = {
    "F0": IdentityFilter,
    "F1": MovingAverageFilter,
    "F2": KalmanLocalLevelFilter,
    "F3": WaveletThresholdFilter,
    "F4": MedianFilter,
    "F6": AdaptiveWaveletFilter,
    "F7": HybridMedianWaveletFilter,
    "F8": AdaptiveMetaFilter,
}

from momo.learnable import LearnableCNNFilter  # noqa: E402

FILTER_REGISTRY["F5"] = LearnableCNNFilter
