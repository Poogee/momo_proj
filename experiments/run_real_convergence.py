"""Project factorial on real data with a convergence metric.

For each data type (daily candles, 5-min, 1-min) and each ticker we form
an AR(p) least-squares regression on the (filtered) return series and
train it with every optimizer x every filter. The reported metric is
*convergence speed*: the number of iterations until the gradient norm
drops by a fixed factor,

    T = min { k : ||grad f(x_k)||^2 <= eps_rel * ||grad f(x_0)||^2 },

capped at the step horizon. This is the declared factorial
(filter x optimizer x data) measured by convergence, not by accuracy.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse

import numpy as np
import pandas as pd

from momo.data import DEFAULT_TICKERS, fetch_intraday, fetch_returns
from momo.filters import (
    CausalMedianFilter,
    IdentityFilter,
    KalmanLocalLevelFilter,
    MovingAverageFilter,
    OnlineAdaptiveFilter,
    WaveletThresholdFilter,
)
from momo.noise import GaussianNoise
from momo.optim import run_optimization

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_real_walkforward import _ar_design, _LinearForecast

FILTERS = {
    "F0": lambda: IdentityFilter(),
    "F1": lambda: MovingAverageFilter(window=5),
    "F2": lambda: KalmanLocalLevelFilter(process_var=1e-3, obs_var=1.0),
    "F3": lambda: WaveletThresholdFilter(wavelet="db4", mode="soft",
                                         threshold="universal"),
    "F4": lambda: CausalMedianFilter(window=5),
    "FA": lambda: OnlineAdaptiveFilter(window=9, k=3.0),
}
OPTIMIZERS = ("sgd", "adam", "adamw")


def _intraday_returns(interval: str) -> pd.DataFrame:
    px = fetch_intraday(sum(DEFAULT_TICKERS.values(), []), interval=interval)
    return np.log(px / px.shift(1)).dropna(how="all")


def t_conv(g: np.ndarray, eps_rel: float) -> int:
    g0 = float(np.mean(g[:5]))
    if g0 <= 0:
        return len(g)
    hit = np.where(g <= eps_rel * g0)[0]
    return int(hit[0]) if hit.size else len(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", type=Path,
                    default=Path("tables/real_convergence_summary.csv"))
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--p", type=int, default=5)
    ap.add_argument("--eps-rel", type=float, default=1e-3)
    args = ap.parse_args()

    daily = fetch_returns(sum(DEFAULT_TICKERS.values(), []))
    datasets = {
        "daily": daily,
        "5min": _intraday_returns("5m"),
        "1min": _intraday_returns("1m"),
    }
    rows = []
    for dname, df in datasets.items():
        for tk in df.columns:
            r = df[tk].dropna().to_numpy()
            if r.size < 4 * args.p + 50:
                continue
            for fname, fbuild in FILTERS.items():
                rs = fbuild().apply(r)
                X, y = _ar_design(rs, args.p)
                if X.shape[0] < 50:
                    continue
                for opt in OPTIMIZERS:
                    task = _LinearForecast(args.p)
                    task.attach(X, y)
                    res = run_optimization(
                        task=task, optimizer=opt, noise=GaussianNoise(0.0),
                        filt=IdentityFilter(), steps=args.steps, lr=1e-2,
                        batch_size=64, seed=0, preprocess_mode="series",
                    )
                    g = res.grad_norm_sq_history
                    rows.append(dict(
                        data=dname, ticker=tk, filter=fname, optimizer=opt,
                        t_conv=t_conv(g, args.eps_rel),
                        final_grad=float(g[-1]),
                        steps=args.steps,
                    ))
    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv} ({len(out)} rows)")
    for dname in datasets:
        sub = out[out.data == dname]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="filter", columns="optimizer",
                              values="t_conv", aggfunc="median")
        piv = piv.reindex([f for f in FILTERS if f in piv.index])
        print(f"\n[{dname}] median iterations to converge "
              f"(eps_rel={args.eps_rel}, cap={args.steps}):")
        print(piv.round(0).to_string())


if __name__ == "__main__":
    main()
