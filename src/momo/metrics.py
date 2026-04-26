from __future__ import annotations

import numpy as np
from scipy import stats


def snr_db(signal: np.ndarray, observed: np.ndarray) -> float:
    noise = observed - signal
    s_var = float(np.var(signal, ddof=1))
    n_var = float(np.var(noise, ddof=1))
    if n_var <= 0:
        return float("inf")
    if s_var <= 0:
        return float("-inf")
    return 10.0 * float(np.log10(s_var / n_var))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def gaussian_log_likelihood(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    resid = y_true - y_pred
    sigma2 = float(np.var(resid, ddof=1))
    n = y_true.size
    return float(-0.5 * n * (np.log(2 * np.pi * sigma2) + 1.0))


def hurst_rs(x: np.ndarray, min_window: int = 10, max_window: int | None = None,
             num_points: int = 20) -> float:
    x = np.asarray(x, dtype=float)
    n = x.size
    if max_window is None:
        max_window = n // 4
    windows = np.unique(np.round(np.geomspace(min_window, max_window, num_points)).astype(int))
    rs_values = []
    log_w = []
    for w in windows:
        if w < 4 or w > n:
            continue
        chunks = n // w
        if chunks < 1:
            continue
        rs_chunk = []
        for i in range(chunks):
            seg = x[i * w:(i + 1) * w]
            mean = seg.mean()
            y = np.cumsum(seg - mean)
            r = y.max() - y.min()
            s = seg.std(ddof=1)
            if s > 0 and r > 0:
                rs_chunk.append(r / s)
        if rs_chunk:
            rs_values.append(np.mean(rs_chunk))
            log_w.append(w)
    if len(rs_values) < 4:
        return float("nan")
    slope, _, _, _, _ = stats.linregress(np.log(log_w), np.log(rs_values))
    return float(slope)


def hurst_dfa(x: np.ndarray, min_window: int = 8, max_window: int | None = None,
              num_points: int = 20, order: int = 1) -> float:
    x = np.asarray(x, dtype=float)
    n = x.size
    if max_window is None:
        max_window = n // 4
    y = np.cumsum(x - x.mean())
    windows = np.unique(np.round(np.geomspace(min_window, max_window, num_points)).astype(int))
    fluct = []
    log_w = []
    for w in windows:
        if w < order + 2 or w > n:
            continue
        chunks = n // w
        if chunks < 1:
            continue
        rms = []
        idx = np.arange(w)
        for i in range(chunks):
            seg = y[i * w:(i + 1) * w]
            coef = np.polyfit(idx, seg, order)
            trend = np.polyval(coef, idx)
            rms.append(np.sqrt(np.mean((seg - trend) ** 2)))
        if rms:
            fluct.append(np.mean(rms))
            log_w.append(w)
    if len(fluct) < 4:
        return float("nan")
    slope, _, _, _, _ = stats.linregress(np.log(log_w), np.log(fluct))
    return float(slope)


def hill_alpha(x: np.ndarray, k: int | None = None) -> float:
    x = np.asarray(x, dtype=float)
    abs_x = np.sort(np.abs(x[np.isfinite(x)]))[::-1]
    n = abs_x.size
    if k is None:
        k = max(int(0.05 * n), 20)
    k = min(k, n - 1)
    if k < 5 or abs_x[k] <= 0:
        return float("nan")
    log_ratio = np.log(abs_x[:k]) - np.log(abs_x[k])
    inv_alpha = log_ratio.mean()
    if inv_alpha <= 0:
        return float("nan")
    return float(1.0 / inv_alpha)


_MCCULLOCH_NU_ALPHA = np.array([
    [2.439, 2.500, 2.600, 2.700, 2.800, 3.000, 3.200, 3.500, 4.000, 5.000, 6.000, 8.000, 10.000, 15.000, 25.000],
])
_MCCULLOCH_ALPHA_TABLE = {
    2.439: 2.000, 2.500: 1.916, 2.600: 1.808, 2.700: 1.729, 2.800: 1.664,
    3.000: 1.563, 3.200: 1.484, 3.500: 1.391, 4.000: 1.279, 5.000: 1.128,
    6.000: 1.029, 8.000: 0.896, 10.000: 0.818, 15.000: 0.698, 25.000: 0.593,
}


def mcculloch_alpha(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 100:
        return float("nan")
    q05, q25, q50, q75, q95 = np.quantile(x, [0.05, 0.25, 0.50, 0.75, 0.95])
    iqr = q75 - q25
    if iqr <= 0:
        return float("nan")
    nu_alpha = (q95 - q05) / iqr
    keys = sorted(_MCCULLOCH_ALPHA_TABLE.keys())
    if nu_alpha <= keys[0]:
        return _MCCULLOCH_ALPHA_TABLE[keys[0]]
    if nu_alpha >= keys[-1]:
        return _MCCULLOCH_ALPHA_TABLE[keys[-1]]
    for i in range(len(keys) - 1):
        a, b = keys[i], keys[i + 1]
        if a <= nu_alpha <= b:
            t = (nu_alpha - a) / (b - a)
            return float((1 - t) * _MCCULLOCH_ALPHA_TABLE[a] + t * _MCCULLOCH_ALPHA_TABLE[b])
    return float("nan")


def time_to_eps(grad_norms_sq: np.ndarray, eps: float) -> int | None:
    arr = np.asarray(grad_norms_sq)
    hits = np.where(arr <= eps)[0]
    if hits.size == 0:
        return None
    return int(hits[0])
