from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from joblib import Parallel, delayed

from momo.data import fetch_returns, make_walk_forward_splits
from momo.metrics import mse
from momo.noise import GaussianNoise
from momo.optim import run_optimization

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_real_walkforward import FILTERS_BUILDERS, _LinearForecast, _ar_design


def design_multistep(series: np.ndarray, p: int, h: int):
    n = series.size
    if n <= p + h:
        return np.empty((0, p)), np.empty(0)
    X = np.lib.stride_tricks.sliding_window_view(series[: n - h], p)
    y = series[p + h - 1 :]
    return X.copy(), y[: X.shape[0]].copy()


def run_one(ticker, returns_train, returns_test, p, h, filt_name, opt_name, seed, steps):
    filt = FILTERS_BUILDERS[filt_name]()
    train_smoothed = filt.apply(returns_train)
    full_smoothed = filt.apply(np.concatenate([returns_train, returns_test]))
    test_smoothed = full_smoothed[returns_train.size :]
    X_train, y_train = design_multistep(train_smoothed, p, h)
    X_test_raw, y_test_raw = design_multistep(np.concatenate([returns_train[-p:], returns_test]), p, h)
    X_test_raw = X_test_raw[: returns_test.size]
    y_test_raw = y_test_raw[: returns_test.size]
    if X_train.size == 0 or X_test_raw.size == 0:
        return None
    task = _LinearForecast(p)
    task.attach(X_train, y_train)
    res = run_optimization(
        task=task, optimizer=opt_name, noise=GaussianNoise(0.0),
        filt=type(filt)(), steps=steps, lr=1e-2, batch_size=64,
        seed=seed, preprocess_mode="data",
    )
    pred = X_test_raw @ res.x_final
    return dict(ticker=ticker, filter=filt_name, optimizer=opt_name,
                horizon=h, seed=seed,
                holdout_mse=float(mse(y_test_raw, pred)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("experiments/configs/real.yaml"))
    parser.add_argument("--out-csv", type=Path, default=Path("tables/multistep_horizon.csv"))
    parser.add_argument("--out-fig", type=Path, default=Path("figures/multistep_horizon.pdf"))
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--n-seeds", type=int, default=2)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    df = fetch_returns(cfg["tickers"], start=cfg["start"], end=cfg["end"])
    chosen_filters = ["F0", "F4", "F7", "F8", "F9"]
    horizons = [1, 5, 10, 20]

    cells = []
    for ticker in df.columns:
        series = df[ticker].dropna().to_numpy()
        if series.size < cfg["p"] + 200:
            continue
        try:
            splits = make_walk_forward_splits(
                pd.Series(series), n_splits=3,
                train_size=cfg["train_size"], test_size=cfg["test_size"],
            )
        except Exception:
            continue
        for tr, te in splits:
            for h in horizons:
                for fn in chosen_filters:
                    for sd in range(args.n_seeds):
                        cells.append((ticker, tr, te, cfg["p"], h, fn, "adam", sd, 1500))
    print(f"Running {len(cells)} cells")
    t0 = time.perf_counter()
    rows = Parallel(n_jobs=args.n_jobs, verbose=5, backend="loky")(
        delayed(run_one)(*c) for c in cells
    )
    rows = [r for r in rows if r is not None]
    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Done in {time.perf_counter()-t0:.1f}s")

    sns.set_theme(context="paper", style="whitegrid", font_scale=0.95)
    fig, ax = plt.subplots(figsize=(8, 5))
    agg = out.groupby(["filter", "horizon"])["holdout_mse"].mean().reset_index()
    for fn in chosen_filters:
        view = agg[agg["filter"] == fn].sort_values("horizon")
        ax.plot(view["horizon"], view["holdout_mse"] * 1e4, marker="o", lw=1.5, label=fn)
    ax.set_xlabel("forecast horizon h (days)")
    ax.set_ylabel(r"holdout MSE $\times 10^4$")
    ax.set_title("Влияние фильтра на multi-step ahead прогноз")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, dpi=150)
    plt.close(fig)
    print(f"Wrote {args.out_fig}")
    print()
    print(out.groupby(["horizon", "filter"])["holdout_mse"].mean().unstack().round(6))


if __name__ == "__main__":
    main()
