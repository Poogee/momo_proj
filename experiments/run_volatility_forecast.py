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
from momo.metrics import mse
from momo.noise import GaussianNoise
from momo.optim import run_optimization

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_real_walkforward import FILTERS_BUILDERS, _LinearForecast, _ar_design


def realized_vol(returns: np.ndarray, window: int = 20) -> np.ndarray:
    out = np.empty_like(returns)
    for i in range(returns.size):
        lo = max(0, i - window + 1)
        out[i] = float(np.std(returns[lo : i + 1], ddof=1)) if i > lo else 0.0
    return out


def run_one(ticker, returns_train, returns_test, p, filt_name, opt_name, seed, steps, lr):
    filt = FILTERS_BUILDERS[filt_name]()
    vol_train = realized_vol(returns_train, window=20)
    full_returns = np.concatenate([returns_train, returns_test])
    vol_full = realized_vol(full_returns, window=20)
    vol_test = vol_full[returns_train.size :]
    train_smoothed = filt.apply(vol_train)
    full_smoothed = filt.apply(vol_full)
    test_smoothed = full_smoothed[returns_train.size:]
    X_train, y_train = _ar_design(train_smoothed, p)
    X_test_raw, y_test_raw = _ar_design(np.concatenate([vol_train[-p:], vol_test]), p)
    X_test_raw = X_test_raw[: vol_test.size]
    y_test_raw = y_test_raw[: vol_test.size]
    if X_train.size == 0 or X_test_raw.size == 0:
        return None
    task = _LinearForecast(p)
    task.attach(X_train, y_train)
    res = run_optimization(
        task=task, optimizer=opt_name, noise=GaussianNoise(0.0),
        filt=type(filt)(),
        steps=steps, lr=lr, batch_size=64, seed=seed,
        preprocess_mode="data",
    )
    pred = X_test_raw @ res.x_final
    return dict(
        ticker=ticker, filter=filt_name, optimizer=opt_name, seed=seed,
        holdout_mse=float(mse(y_test_raw, pred)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("experiments/configs/real.yaml"))
    parser.add_argument("--out-csv", type=Path, default=Path("tables/real_vol_forecast.csv"))
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--n-seeds", type=int, default=2)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
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
        for tr, te in splits:
            for fn in FILTERS_BUILDERS:
                for on in ["adam", "sgd"]:
                    for sd in range(args.n_seeds):
                        cells.append((ticker, tr, te, cfg["p"], fn, on, sd, args.steps, args.lr))
    print(f"Running {len(cells)} cells")
    t0 = time.perf_counter()
    rows = Parallel(n_jobs=args.n_jobs, verbose=5, backend="loky")(
        delayed(run_one)(*c) for c in cells
    )
    rows = [r for r in rows if r is not None]
    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print(f"Done in {time.perf_counter()-t0:.1f}s")
    print("\n=== Mean holdout MSE × 1e6 by (filter, optimizer) ===")
    print((out.groupby(["filter", "optimizer"])["holdout_mse"].mean() * 1e6).unstack().round(3))


if __name__ == "__main__":
    main()
