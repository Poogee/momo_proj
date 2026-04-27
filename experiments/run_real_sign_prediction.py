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
from momo.noise import GaussianNoise
from momo.optim import run_optimization
from momo.tasks import LogisticTask, _sigmoid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_real_walkforward import FILTERS_BUILDERS, _ar_design


def run_one(ticker, returns_train, returns_test, p, filt_name, opt_name, seed, steps):
    filt = FILTERS_BUILDERS[filt_name]()
    train_smoothed = filt.apply(returns_train)
    full_smoothed = filt.apply(np.concatenate([returns_train, returns_test]))
    test_smoothed = full_smoothed[returns_train.size:]
    X_train, _ = _ar_design(train_smoothed, p)
    y_train = (returns_train[p:] > 0).astype(np.float64)
    X_test, _ = _ar_design(np.concatenate([train_smoothed[-p:], test_smoothed]), p)
    X_test = X_test[: returns_test.size]
    y_test = (returns_test > 0).astype(np.float64)[: X_test.shape[0]]
    if X_train.size == 0 or X_test.size == 0 or y_train.size != X_train.shape[0]:
        return None
    Z_train = np.column_stack([X_train, np.ones(X_train.shape[0])])
    Z_test = np.column_stack([X_test, np.ones(X_test.shape[0])])
    x_star = np.zeros(p + 1)
    task = LogisticTask(
        Z_train=Z_train, y_train=y_train,
        Z_test=Z_test, y_test=y_test, x_star=x_star,
    )
    res = run_optimization(
        task=task, optimizer=opt_name,
        noise=GaussianNoise(0.0), filt=type(filt)(),
        steps=steps, lr=1e-1, batch_size=64, seed=seed, weight_decay=1e-4,
        preprocess_mode="data",
    )
    preds = (_sigmoid(Z_test @ res.x_final) >= 0.5).astype(int)
    acc = float(np.mean(preds == y_test.astype(int)))
    return dict(ticker=ticker, filter=filt_name, optimizer=opt_name, seed=seed,
                test_acc=acc, baseline_majority=float(max(y_test.mean(), 1 - y_test.mean())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("experiments/configs/real.yaml"))
    parser.add_argument("--out-csv", type=Path, default=Path("tables/real_sign_pred.csv"))
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=8)
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
                for on in ["adam", "adamw"]:
                    for sd in range(args.n_seeds):
                        cells.append((ticker, tr, te, cfg["p"], fn, on, sd, args.steps))
    print(f"Running {len(cells)} cells")
    t0 = time.perf_counter()
    rows = Parallel(n_jobs=args.n_jobs, verbose=5, backend="loky")(
        delayed(run_one)(*c) for c in cells
    )
    rows = [r for r in rows if r is not None]
    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print(f"Done in {time.perf_counter()-t0:.1f}s")
    print("\n=== Mean test accuracy by (filter, optimizer) ===")
    print(out.groupby(["filter", "optimizer"])["test_acc"].mean().unstack().round(4))
    print(f"\nMajority baseline: mean={out['baseline_majority'].mean():.4f}")


if __name__ == "__main__":
    main()
