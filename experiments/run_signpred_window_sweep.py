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
from momo.filters import CausalMedianFilter, IdentityFilter, MedianFilter
from momo.noise import GaussianNoise
from momo.optim import run_optimization
from momo.tasks import LogisticTask, _sigmoid

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_real_walkforward import _ar_design


def design(series, p):
    X, _ = _ar_design(series, p)
    y = (series[p:] > 0).astype(np.float64)
    Z = np.column_stack([X, np.ones(X.shape[0])])
    return Z, y


def run_one(ticker, returns_train, returns_test, p, w, seed, steps):
    filt = MedianFilter(window=w) if w > 1 else IdentityFilter()
    train_smoothed = filt.apply(returns_train)
    full_smoothed = filt.apply(np.concatenate([returns_train, returns_test]))
    test_smoothed = full_smoothed[returns_train.size :]
    Z_train, y_train = design(train_smoothed, p)
    Z_test, _ = design(np.concatenate([train_smoothed[-p:], test_smoothed]), p)
    Z_test = Z_test[: returns_test.size]
    y_test = (returns_test > 0).astype(np.float64)[: Z_test.shape[0]]
    if Z_train.size == 0 or Z_test.size == 0:
        return None
    task = LogisticTask(
        Z_train=Z_train, y_train=y_train,
        Z_test=Z_test, y_test=y_test, x_star=np.zeros(p + 1),
    )
    res = run_optimization(
        task=task, optimizer="adam", noise=GaussianNoise(0.0),
        filt=type(filt)() if w == 1 else MedianFilter(window=w),
        steps=steps, lr=1e-1, batch_size=64, seed=seed, weight_decay=1e-4,
        preprocess_mode="data",
    )
    preds = (_sigmoid(Z_test @ res.x_final) >= 0.5).astype(int)
    acc = float(np.mean(preds == y_test.astype(int)))
    majority = float(max(y_test.mean(), 1 - y_test.mean()))
    return dict(ticker=ticker, window=w, seed=seed, acc=acc, majority=majority)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("experiments/configs/real.yaml"))
    parser.add_argument("--out-csv", type=Path, default=Path("tables/signpred_window.csv"))
    parser.add_argument("--out-fig", type=Path, default=Path("figures/signpred_window.pdf"))
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--n-seeds", type=int, default=3)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    df = fetch_returns(cfg["tickers"], start=cfg["start"], end=cfg["end"])
    windows = [1, 3, 5, 7, 9, 13, 17, 21, 31, 51]

    cells = []
    for ticker in df.columns:
        series = df[ticker].dropna().to_numpy()
        if series.size < cfg["p"] + 200:
            continue
        try:
            splits = make_walk_forward_splits(
                pd.Series(series), n_splits=4,
                train_size=cfg["train_size"], test_size=cfg["test_size"],
            )
        except Exception:
            continue
        for tr, te in splits:
            for w in windows:
                for sd in range(args.n_seeds):
                    cells.append((ticker, tr, te, cfg["p"], w, sd, 1500))
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
    agg = out.groupby("window").agg(mean_acc=("acc", "mean"), std_acc=("acc", "std")).reset_index()
    majority = out["majority"].mean()
    ax.errorbar(agg["window"], agg["mean_acc"], yerr=agg["std_acc"],
                marker="o", lw=1.5, capsize=3, color="C0")
    ax.axhline(majority, color="gray", lw=1, ls="--", label=f"majority baseline = {majority:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("median window w (1 = F0 identity)")
    ax.set_ylabel("test accuracy (mean ± std)")
    ax.set_title("Sign-prediction accuracy vs median window")
    ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, dpi=150)
    plt.close(fig)
    print(f"Wrote {args.out_fig}")
    print()
    print("=== mean accuracy by window ===")
    print(agg.round(4))


if __name__ == "__main__":
    main()
