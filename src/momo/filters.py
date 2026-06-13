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
class CausalMedianFilter:
    window: int = 5

    def apply(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        w = max(1, int(self.window))
        n = y.size
        out = np.empty(n, dtype=float)
        for i in range(n):
            lo = max(0, i - w + 1)
            out[i] = float(np.median(y[lo : i + 1]))
        return out


@dataclass(frozen=True)
class OnlineAdaptiveFilter:
    """Causal, online filter that adapts per step to local noise character.

    For each new point it inspects only the trailing window (no lookahead),
    estimates a robust scale (MAD) and the fraction of points beyond
    ``k`` MADs from the local median. If outliers are present (heavy-tailed
    / contaminated regime) it emits the trailing *median* (robust); if the
    window looks light-tailed it emits the trailing *mean* (statistically
    efficient). This automates the "diagnose then decide" rule online,
    without knowing the noise type in advance.
    """

    window: int = 9
    k: float = 3.0
    beta: float = 0.7
    warmup: int = 3

    def apply(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        n = y.size
        if n == 0:
            return y.copy()
        w = max(3, int(self.window))
        b = float(self.beta)
        out = np.empty(n, dtype=float)
        ema = float(y[0])
        for i in range(n):
            lo = max(0, i - w + 1)
            win = y[lo : i + 1]
            if win.size < self.warmup:
                ema = b * ema + (1.0 - b) * float(y[i])
                out[i] = ema
                continue
            med = float(np.median(win))
            mad = float(np.median(np.abs(win - med))) * 1.4826
            is_outlier = mad > 1e-12 and abs(float(y[i]) - med) > self.k * mad
            # robustify the EMA update against the current spike
            ema = b * ema + (1.0 - b) * (med if is_outlier else float(y[i]))
            frac_out = (float(np.mean(np.abs(win - med) > self.k * mad))
                        if mad > 1e-12 else 0.0)
            # heavy / contaminated window -> robust median; else tracking EMA
            out[i] = med if frac_out > 0.0 else ema
        return out


@dataclass(frozen=True)
class EnsembleAverageFilter:
    median_window: int = 9

    def apply(self, y: np.ndarray) -> np.ndarray:
        kalman = KalmanLocalLevelFilter(process_var=1e-3, obs_var=1.0).apply(y)
        median = MedianFilter(window=self.median_window).apply(y)
        hybrid = HybridMedianWaveletFilter(median_window=5).apply(y)
        return (kalman + median + hybrid) / 3.0


@dataclass(frozen=True)
class CausalCascadeFilter:
    """Strictly causal two-stage cascade designed for the mixed regime
    where the noise has both heavy tails (alpha<2) and long memory (H>1/2).

    Stage 1 — short causal median (window m): collapses isolated alpha-stable
    spikes into a finite-variance residual (Lemma in the paper, finite
    second moment of an order statistic).
    Stage 2 — Kalman local-level recursion: smooths the residual long-memory
    drift that the median alone cannot remove (the window is too short to
    capture LRD clusters).

    Both stages are O(T) and use only past samples, so applying it on a
    real series introduces no look-ahead bias.
    """

    median_window: int = 3
    process_var: float = 1e-3
    obs_var: float = 1.0

    def apply(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if y.size == 0:
            return y.copy()
        stage1 = CausalMedianFilter(window=max(1, int(self.median_window))).apply(y)
        stage2 = KalmanLocalLevelFilter(
            process_var=float(self.process_var),
            obs_var=float(self.obs_var),
        ).apply(stage1)
        return stage2


@dataclass(frozen=True)
class AdaptiveCascadeFilter:
    """Strictly causal adaptive cascade that selects its own stages from
    online noise diagnostics. Computed only on the *training* slice the
    user passes in (we never peek at future samples).

    Decision rule:
      alpha_hat < alpha_thr  -> include short causal median stage (tails)
      H_hat   > hurst_thr    -> append Kalman/Wavelet smoothing (LRD)
      both                    -> full cascade (median -> Kalman)
      neither                 -> identity (do nothing — F0)

    The thresholds match the practical criteria reported in the paper
    (alpha_thr=1.9, hurst_thr=0.6). If the series is too short for a stable
    estimate we fall back to a short causal median.
    """

    alpha_threshold: float = 1.9
    hurst_threshold: float = 0.6
    median_window: int = 3
    process_var: float = 1e-3
    obs_var: float = 1.0
    min_length: int = 128

    def apply(self, y: np.ndarray) -> np.ndarray:
        from momo.metrics import hurst_dfa, mcculloch_alpha

        y = np.asarray(y, dtype=float)
        n = y.size
        if n == 0:
            return y.copy()
        if n < self.min_length:
            return CausalMedianFilter(window=max(1, int(self.median_window))).apply(y)
        # diagnostics on the differenced series (so a non-stationary level
        # does not dominate the empirical quantiles)
        diff = np.diff(y)
        alpha_hat = mcculloch_alpha(diff)
        if not np.isfinite(alpha_hat):
            alpha_hat = 2.0
        h_hat = hurst_dfa(diff)
        if not np.isfinite(h_hat):
            h_hat = 0.5
        heavy = alpha_hat < float(self.alpha_threshold)
        long_mem = h_hat > float(self.hurst_threshold)
        out = y.copy()
        if heavy:
            out = CausalMedianFilter(window=max(1, int(self.median_window))).apply(out)
        if long_mem:
            out = KalmanLocalLevelFilter(
                process_var=float(self.process_var),
                obs_var=float(self.obs_var),
            ).apply(out)
        return out


@dataclass(frozen=True)
class CausalHybridMedianWavelet:
    """Causal variant of :class:`HybridMedianWaveletFilter` — uses the
    causal median in stage 1 so the cascade can be reported on real
    walk-forward data without look-ahead bias.
    """

    median_window: int = 5
    wavelet: str = "db4"
    mode: str = "soft"

    def apply(self, y: np.ndarray) -> np.ndarray:
        pre = CausalMedianFilter(window=max(1, int(self.median_window))).apply(y)
        return AdaptiveWaveletFilter(wavelet=self.wavelet, mode=self.mode).apply(pre)


class SequentialCascade:
    """Apply a list of causal filters in sequence: stage k runs on the output
    of stage k-1. Transparent composition of existing filters (no new filter
    math) — used to build the two-stage cascade for the mixed regime N4:
    a robust stage against alpha-stable spikes (causal median, F4) followed by
    a memory-oriented stage against the residual 1/f correlation (wavelet F3 or
    Kalman F2). Every stage is itself strictly causal, so the cascade is too."""

    def __init__(self, *stages):
        self.stages = stages

    def apply(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        for s in self.stages:
            y = s.apply(y)
        return y


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


# Canonical, contiguous numbering. Causal variants are used whenever a
# filter is reported on real-data walk-forward; non-causal variants are
# kept for synthetic-only ablations and clearly marked in the paper.
FILTER_REGISTRY = {
    "F0": IdentityFilter,                # identity (control)
    "F1": MovingAverageFilter,           # causal MA, linear
    "F2": KalmanLocalLevelFilter,        # causal Kalman, linear
    "F3": WaveletThresholdFilter,        # wavelet soft-threshold (synthetic)
    "F4": CausalMedianFilter,            # causal median (nonlinear)
    "F5": None,                          # learnable CNN (filled below)
    "F6": AdaptiveWaveletFilter,         # data-driven wavelet
    "F7": CausalHybridMedianWavelet,     # causal median -> wavelet cascade
    "F8": AdaptiveMetaFilter,            # diagnostic-routed single filter
    "F9": None,                          # learnable CNN (large, filled below)
    "F10": CausalCascadeFilter,          # NEW: causal median -> Kalman
    "F11": AdaptiveCascadeFilter,        # NEW: online-diagnostic cascade
    "FA": OnlineAdaptiveFilter,          # online causal med/EMA switch
    "FE": EnsembleAverageFilter,         # average ensemble
}

# learnable CNN filters live in their own module to keep torch optional: if
# torch is unavailable the classical filters (F0-F4, F6-F11, FA, FE) still load
# and F5/F9 stay None in the registry.
try:
    from momo.learnable import LearnableCNNFilter, LearnableCNNFilterV2  # noqa: E402

    FILTER_REGISTRY["F5"] = LearnableCNNFilter
    FILTER_REGISTRY["F9"] = LearnableCNNFilterV2
except ImportError:  # pragma: no cover - torch optional
    LearnableCNNFilter = None
    LearnableCNNFilterV2 = None
