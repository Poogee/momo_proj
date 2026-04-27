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

from momo.data import fetch_returns
from momo.noise import GaussianNoise
from momo.optim import run_optimization
from momo.tasks import LogisticTask, _sigmoid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_real_walkforward import FILTERS_BUILDERS, _ar_design


def design(series: np.ndarray, p: int):
    X, _ = _ar_design(series, p)
    y = (series[p:] > 0).astype(np.float64)
    Z = np.column_stack([X, np.ones(X.shape[0])])
    return Z, y


def run_one(train_ticker, test_ticker, train_series, test_series, p, filt_name, opt_name, seed, steps):
    filt = FILTERS_BUILDERS[filt_name]()
    train_smoothed = filt.apply(train_series)
    test_smoothed = filt.apply(test_series)
    Z_train, y_train = design(train_smoothed, p)
    Z_test, y_test = design(test_series, p)
    if Z_train.size == 0 or Z_test.size == 0:
        return None
    task = LogisticTask(
        Z_train=Z_train, y_train=y_train,
        Z_test=Z_test, y_test=y_test, x_star=np.zeros(p + 1),
    )
    res = run_optimization(
        task=task, optimizer=opt_name,
        noise=GaussianNoise(0.0), filt=type(filt)(),
        steps=steps, lr=1e-1, batch_size=64, seed=seed, weight_decay=1e-4,
        preprocess_mode="data",
    )
    preds = (_sigmoid(Z_test @ res.x_final) >= 0.5).astype(int)
    acc = float(np.mean(preds == y_test.astype(int)))
    majority = float(max(y_test.mean(), 1 - y_test.mean()))
    return dict(
        train=train_ticker, test=test_ticker,
        filter=filt_name, optimizer=opt_name, seed=seed,
        test_acc=acc, majority=majority,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("experiments/configs/real.yaml"))
    parser.add_argument("--out-csv", type=Path, default=Path("tables/cross_asset_transfer.csv"))
    parser.add_argument("--out-fig", type=Path, default=Path("figures/cross_asset_transfer.pdf"))
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--n-seeds", type=int, default=3)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    df = fetch_returns(cfg["tickers"], start=cfg["start"], end=cfg["end"])

    chosen_filters = ["F0", "F4", "F7", "F8", "F9"]
    cells = []
    for train_ticker in df.columns:
        for test_ticker in df.columns:
            if train_ticker == test_ticker:
                continue
            train_series = df[train_ticker].dropna().to_numpy()
            test_series = df[test_ticker].dropna().to_numpy()
            for fn in chosen_filters:
                for sd in range(args.n_seeds):
                    cells.append((train_ticker, test_ticker, train_series, test_series,
                                  cfg["p"], fn, "adam", sd, args.steps))
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

    print("\n=== Mean test accuracy by (filter), pooled across all (train, test) pairs ===")
    print(out.groupby("filter")["test_acc"].agg(["mean", "std"]).round(4))
    print(f"\nMajority baseline mean: {out['majority'].mean():.4f}")

    sns.set_theme(context="paper", style="whitegrid", font_scale=0.9)
    fig, axes = plt.subplots(1, len(chosen_filters), figsize=(4 * len(chosen_filters), 5),
                             sharey=True)
    for ax, fn in zip(axes, chosen_filters):
        sub = out[out["filter"] == fn]
        piv = sub.groupby(["train", "test"])["test_acc"].mean().unstack()
        sns.heatmap(piv, annot=True, fmt=".2f", cmap="viridis_r", ax=ax,
                    vmin=0.45, vmax=0.65, cbar=ax is axes[-1])
        ax.set_title(f"{fn}")
        ax.set_xlabel("test")
        if ax is axes[0]:
            ax.set_ylabel("train")
        else:
            ax.set_ylabel("")
    fig.suptitle("Cross-asset sign-prediction accuracy")
    fig.tight_layout()
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, dpi=150)
    plt.close(fig)
    print(f"\nWrote {args.out_fig}")


if __name__ == "__main__":
    main()
