from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse
import time

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from momo.data import fetch_returns, make_walk_forward_splits
from momo.filters import (
    AdaptiveWaveletFilter,
    HybridMedianWaveletFilter,
    IdentityFilter,
    KalmanLocalLevelFilter,
    MedianFilter,
    MovingAverageFilter,
    WaveletThresholdFilter,
)
from momo.metrics import gaussian_log_likelihood, hill_alpha, hurst_dfa, mcculloch_alpha, mse, snr_db
from momo.noise import GaussianNoise
from momo.optim import run_optimization
from momo.tasks import LogisticTask


FILTERS_BUILDERS = {
    "F0": lambda: IdentityFilter(),
    "F1": lambda: MovingAverageFilter(window=5),
    "F2": lambda: KalmanLocalLevelFilter(process_var=1e-4, obs_var=1.0),
    "F3": lambda: WaveletThresholdFilter(wavelet="db4", mode="soft", threshold="universal"),
    "F4": lambda: MedianFilter(window=5),
    "F6": lambda: AdaptiveWaveletFilter(),
    "F7": lambda: HybridMedianWaveletFilter(median_window=5),
}


class _LinearForecast:
    def __init__(self, p: int):
        self.p = p
        self.dim = p

    def attach(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def loss(self, x, Z=None, y=None):
        Z = self.X_train if Z is None else Z
        y = self.y_train if y is None else y
        r = Z @ x - y
        return float(np.mean(r * r))

    def grad(self, x, Z=None, y=None):
        Z = self.X_train if Z is None else Z
        y = self.y_train if y is None else y
        r = Z @ x - y
        return (2.0 / Z.shape[0]) * (Z.T @ r)

    def sample_batch(self, rng, n):
        idx = rng.integers(0, self.X_train.shape[0], size=n)
        return self.X_train[idx], self.y_train[idx]


def _ar_design(series: np.ndarray, p: int):
    n = series.size
    if n <= p:
        return np.empty((0, p)), np.empty(0)
    X = np.lib.stride_tricks.sliding_window_view(series[: n - 1], p)
    y = series[p:]
    return X.copy(), y.copy()


def run_split(ticker: str, returns_train: np.ndarray, returns_test: np.ndarray,
              p: int, filt_name: str, opt_name: str, opt_spec: dict,
              seed: int, steps: int, runs_dir: Path):
    filt = FILTERS_BUILDERS[filt_name]()
    train_smoothed = filt.apply(returns_train)
    full = np.concatenate([returns_train, returns_test])
    full_smoothed = filt.apply(full)
    test_smoothed = full_smoothed[returns_train.size:]
    test_raw = returns_test
    X_train, y_train = _ar_design(train_smoothed, p)
    X_test_smoothed, y_test_smoothed = _ar_design(np.concatenate([train_smoothed[-p:], test_smoothed]), p)
    X_test_smoothed = X_test_smoothed[: test_raw.size]
    y_test_smoothed = y_test_smoothed[: test_raw.size]
    X_test_raw, y_test_raw = _ar_design(np.concatenate([returns_train[-p:], returns_test]), p)
    X_test_raw = X_test_raw[: test_raw.size]
    y_test_raw = y_test_raw[: test_raw.size]
    if X_train.size == 0 or X_test_raw.size == 0:
        return None
    task = _LinearForecast(p)
    task.attach(X_train, y_train)
    res = run_optimization(
        task=task, optimizer=opt_name, noise=GaussianNoise(0.0),
        filt=IdentityFilter(), steps=steps, lr=opt_spec["lr"], batch_size=64,
        seed=seed, weight_decay=opt_spec.get("weight_decay", 0.0),
        preprocess_mode="data",
    )
    holdout_pred = X_test_raw @ res.x_final
    holdout_mse = mse(y_test_raw, holdout_pred)
    holdout_loglik = gaussian_log_likelihood(y_test_raw, holdout_pred)
    return dict(
        ticker=ticker, filter=filt_name, optimizer=opt_name, seed=seed,
        train_size=returns_train.size, test_size=returns_test.size,
        train_mse=float(res.loss_history[-1]),
        holdout_mse_raw=float(holdout_mse),
        holdout_loglik_raw=float(holdout_loglik),
        snr_filter_db=snr_db(train_smoothed, returns_train),
        hurst_train_raw=hurst_dfa(returns_train),
        hurst_train_smoothed=hurst_dfa(train_smoothed),
        alpha_train_raw=mcculloch_alpha(returns_train),
        alpha_train_smoothed=mcculloch_alpha(train_smoothed),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("experiments/configs/real.yaml"))
    parser.add_argument("--summary-csv", type=Path, default=Path("tables/real_walkforward_summary.csv"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/real_walkforward"))
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-seeds", type=int, default=3)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    args.runs_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching returns...")
    df = fetch_returns(cfg["tickers"], start=cfg["start"], end=cfg["end"])
    cells = []
    for ticker in df.columns:
        series = df[ticker].dropna().to_numpy()
        if series.size < cfg["p"] + 200:
            continue
        try:
            splits = make_walk_forward_splits(
                pd.Series(series), n_splits=args.n_splits,
                train_size=cfg["train_size"], test_size=cfg["test_size"],
            )
        except Exception:
            continue
        for split_idx, (tr, te) in enumerate(splits):
            for filt_name in FILTERS_BUILDERS:
                for opt_name in cfg["optimizers"]:
                    for seed in range(args.n_seeds):
                        cells.append((ticker, tr, te, cfg["p"], filt_name,
                                      opt_name, cfg["optimizers"][opt_name],
                                      seed, args.steps, args.runs_dir, split_idx))
    print(f"Running {len(cells)} cells")
    t0 = time.perf_counter()
    rows = Parallel(n_jobs=args.n_jobs, verbose=5, backend="loky")(
        delayed(run_split)(t, tr, te, p, fn, on, os_, sd, st, rd)
        for (t, tr, te, p, fn, on, os_, sd, st, rd, split_idx) in cells
    )
    rows = [r for r in rows if r is not None]
    elapsed = time.perf_counter() - t0
    out = pd.DataFrame(rows)
    out.to_csv(args.summary_csv, index=False)
    print(f"Done in {elapsed:.1f}s. Wrote {args.summary_csv}")
    print("\nMean holdout MSE x 1e4 by filter, optimizer:")
    print((out.groupby(["filter", "optimizer"])["holdout_mse_raw"].mean() * 1e4).unstack().round(3))


if __name__ == "__main__":
    main()
