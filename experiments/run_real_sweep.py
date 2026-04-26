from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse
import time
from itertools import product

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from momo.data import fetch_returns, make_ar_forecast_task, make_walk_forward_splits
from momo.filters import (
    IdentityFilter,
    KalmanLocalLevelFilter,
    MedianFilter,
    MovingAverageFilter,
    WaveletThresholdFilter,
)
from momo.metrics import (
    gaussian_log_likelihood,
    hill_alpha,
    hurst_dfa,
    mcculloch_alpha,
    mse,
    snr_db,
    time_to_eps,
)
from momo.noise import GaussianNoise
from momo.optim import run_optimization


def build_filter(spec: dict):
    kind = spec["kind"]
    if kind == "identity":
        return IdentityFilter()
    if kind == "ma":
        return MovingAverageFilter(window=spec["window"])
    if kind == "kalman":
        return KalmanLocalLevelFilter(process_var=spec["process_var"], obs_var=spec["obs_var"])
    if kind == "wavelet":
        return WaveletThresholdFilter(wavelet=spec["wavelet"], mode=spec["mode"], threshold=spec["threshold"])
    if kind == "median":
        return MedianFilter(window=spec["window"])
    raise ValueError(kind)


def filter_series(series: np.ndarray, filt) -> np.ndarray:
    return filt.apply(series)


def run_one(ticker: str, returns: np.ndarray, p: int, filt_name: str, filt_spec: dict,
            opt_name: str, opt_spec: dict, seed: int, steps: int, buffer_size: int,
            batch_size: int, noise_scale: float, runs_dir: Path) -> dict:
    filt = build_filter(filt_spec)
    smoothed = filter_series(returns, filt)
    task = make_ar_forecast_task(smoothed, p=p, train_frac=0.7)
    raw_task = make_ar_forecast_task(returns, p=p, train_frac=0.7)
    t0 = time.perf_counter()
    result = run_optimization(
        task=task,
        optimizer=opt_name,
        noise=GaussianNoise(sigma=noise_scale),
        filt=IdentityFilter(),
        steps=steps,
        lr=opt_spec["lr"],
        batch_size=batch_size,
        seed=seed,
        buffer_size=buffer_size,
        weight_decay=opt_spec.get("weight_decay", 0.0),
    )
    elapsed = time.perf_counter() - t0
    holdout_mse_smoothed = mse(task.test_y, task.test_x @ result.x_final)
    holdout_mse_raw = mse(raw_task.test_y, raw_task.test_x @ result.x_final)
    ll_raw = gaussian_log_likelihood(raw_task.test_y, raw_task.test_x @ result.x_final)
    snr_filter_db = snr_db(smoothed, returns)
    h_raw = hurst_dfa(returns)
    h_smoothed = hurst_dfa(smoothed)
    a_raw = mcculloch_alpha(returns)
    a_smoothed = mcculloch_alpha(smoothed)
    cell_dir = runs_dir / ticker / f"{filt_name}_{opt_name}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cell_dir / f"seed{seed:02d}.npz",
        grad_norm_sq=result.grad_norm_sq_history.astype(np.float32),
        loss=result.loss_history.astype(np.float32),
        x_final=result.x_final.astype(np.float32),
    )
    return dict(
        ticker=ticker, filter=filt_name, optimizer=opt_name, seed=seed,
        steps=steps,
        final_grad_norm_sq=float(result.grad_norm_sq_history[-1]),
        final_train_loss=float(result.loss_history[-1]),
        holdout_mse_smoothed=holdout_mse_smoothed,
        holdout_mse_raw=holdout_mse_raw,
        loglik_raw=ll_raw,
        snr_filter_db=snr_filter_db,
        hurst_raw=h_raw, hurst_smoothed=h_smoothed,
        alpha_raw=a_raw, alpha_smoothed=a_smoothed,
        elapsed_s=elapsed,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("experiments/configs/real.yaml"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/real"))
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--summary-csv", type=Path, default=Path("tables/real_summary.csv"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)

    print("Fetching returns...")
    df = fetch_returns(cfg["tickers"], start=cfg["start"], end=cfg["end"])
    print(f"Got {df.shape[0]} bars across {df.shape[1]} tickers")

    cells = []
    for ticker in df.columns:
        series = df[ticker].dropna().to_numpy()
        if series.size < cfg["p"] + 200:
            continue
        for filt_name, filt_spec in cfg["filters"].items():
            for opt_name, opt_spec in cfg["optimizers"].items():
                for seed in range(cfg["run"]["n_seeds"]):
                    cells.append((ticker, series, filt_name, filt_spec, opt_name, opt_spec, seed))

    print(f"Running {len(cells)} cells with n_jobs={args.n_jobs}")
    t0 = time.perf_counter()
    rows = Parallel(n_jobs=args.n_jobs, verbose=5, backend="loky")(
        delayed(run_one)(
            ticker, series, cfg["p"], fn, fs, on, os_, seed,
            cfg["run"]["steps"], cfg["run"]["buffer_size"],
            cfg["run"]["batch_size"], cfg["run"]["noise_scale"], args.runs_dir,
        )
        for (ticker, series, fn, fs, on, os_, seed) in cells
    )
    elapsed = time.perf_counter() - t0
    out = pd.DataFrame(rows)
    out.to_csv(args.summary_csv, index=False)
    print(f"Done in {elapsed:.1f}s. Wrote {args.summary_csv}")
    print(out.groupby(["filter", "optimizer"])["holdout_mse_raw"].mean())


if __name__ == "__main__":
    main()
