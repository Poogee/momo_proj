"""Comprehensive real-data filter/optimizer study (June 2026).

Filter set -- clean, contiguous numbering F0..F7 (no gaps):

    F0  identity (control)
    F1  moving average            (linear, causal)
    F2  Kalman local level        (linear, causal)
    F3  wavelet soft-threshold    (NON-causal: oracle baseline only)
    F4  causal median             (nonlinear order statistic)
    F5  cascade  median -> Kalman (causal; "каскадный")
    F6  adaptive cascade          (causal; routes stages by alpha/H; "адаптивный")
    F7  online adaptive median/EMA switch (causal)

F0..F4 are the original five; F5 (cascade) and F6 (adaptive cascade) are
the contributions; F7 is the online switch. F3 is non-causal and is
reported only as an oracle upper bound -- it never enters the recommended
rule.

Datasets -- a WIDE basket spanning asset classes, frequencies and
non-financial domains so the (alpha-hat, H-hat) -> filter rule can be
derived with full coverage:

  financial returns:   extended daily basket, intraday 15m / 5m,
                       Binance BTC/ETH 1h
  financial volatility |r|: daily + crypto (long-memory regime)
  macro:               FRED
  sensors:             ETTh1/ETTh2/ETTm1/ETTm2, electricity, traffic
  prognostics:         NASA C-MAPSS turbofan
  operational:         daily ATM cash withdrawals
  scientific:          sunspots

Each cell: AR(5) regression + sign classification, SGD and Adam, 4 seeds,
strictly causal pre-filtering (F3 excepted, flagged). Holdout target is
the raw future, identical across filters.

Outputs:
  tables/new_datasets.csv            per-cell raw metrics
  tables/new_datasets_summary.csv    domain x model x optimizer x filter
  tables/new_datasets_criteria.csv   best causal filter per (domain,model,opt)
  tables/new_datasets_rules.csv      (alpha,H) bucket -> dominant filter
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from momo.data import (
    EXTENDED_TICKERS,
    DEFAULT_FRED_SERIES,
    fetch_atm,
    fetch_binance,
    fetch_cmapss,
    fetch_electricity,
    fetch_fred,
    fetch_intraday,
    fetch_nonfinancial,
    fetch_returns,
    fetch_sunspots,
    fetch_traffic,
    fetch_weather_noaa,
)
from momo.filters import (
    AdaptiveCascadeFilter,
    CausalCascadeFilter,
    CausalMedianFilter,
    IdentityFilter,
    KalmanLocalLevelFilter,
    MovingAverageFilter,
    OnlineAdaptiveFilter,
    WaveletThresholdFilter,
)
from momo.metrics import (
    convergence_auc,
    divergence_slope,
    hurst_dfa,
    mcculloch_alpha,
    noise_floor_quantiles,
    time_to_drop,
)
from momo.noise import GaussianNoise
from momo.optim import run_optimization

P = 5
STEPS = 4000
SEEDS = list(range(4))

# ---- contiguous filter set F0..F7 -----------------------------------
STUDY_FILTERS = {
    "F0": lambda: IdentityFilter(),
    "F1": lambda: MovingAverageFilter(window=15),
    "F2": lambda: KalmanLocalLevelFilter(process_var=1e-3, obs_var=1.0),
    "F3": lambda: WaveletThresholdFilter(),                 # non-causal oracle
    "F4": lambda: CausalMedianFilter(window=9),
    "F5": lambda: CausalCascadeFilter(median_window=3,      # cascade
                                      process_var=1e-3, obs_var=1.0),
    "F6": lambda: AdaptiveCascadeFilter(alpha_threshold=1.9,  # adaptive
                                        hurst_threshold=0.6,
                                        median_window=3,
                                        process_var=1e-3, obs_var=1.0),
    "F7": lambda: OnlineAdaptiveFilter(window=9, k=3.0),
}
# filters whose output at time t uses only s[<=t] (no look-ahead)
CAUSAL = {"F0", "F1", "F2", "F4", "F5", "F6", "F7"}
NONCAUSAL = {"F3"}
OPTIMIZERS = ["sgd", "adam"]


class _LinAR:
    def __init__(self, p, classification: bool = False):
        self.p = p
        self.dim = p
        self.cls = bool(classification)

    def attach(self, X, y):
        self.X, self.y = X, y

    def loss(self, x, Z=None, y=None):
        Z = self.X if Z is None else Z
        t = self.y if y is None else y
        if not self.cls:
            r = Z @ x - t
            return float(np.mean(r * r))
        z = Z @ x
        t01 = (t > 0).astype(float)
        return float(np.mean(np.logaddexp(0.0, z) - t01 * z))

    def grad(self, x, Z=None, y=None):
        Z = self.X if Z is None else Z
        t = self.y if y is None else y
        if not self.cls:
            r = Z @ x - t
            return (2.0 / Z.shape[0]) * (Z.T @ r)
        z = Z @ x
        s = 1.0 / (1.0 + np.exp(-z))
        t01 = (t > 0).astype(float)
        return (Z.T @ (s - t01)) / Z.shape[0]

    def sample_batch(self, rng, n):
        idx = rng.integers(0, self.X.shape[0], size=min(n, self.X.shape[0]))
        return self.X[idx], self.y[idx]


def _ar_design(series, p):
    n = series.size
    if n <= p:
        return np.empty((0, p)), np.empty(0)
    X = np.lib.stride_tricks.sliding_window_view(series[:n - 1], p)
    return X.copy(), series[p:].copy()


def _standardize(s):
    s = np.asarray(s, dtype=float)
    s = s[np.isfinite(s)]
    mu, sd = float(np.mean(s)), float(np.std(s))
    return (s - mu) / (sd if sd > 1e-12 else 1.0)


def _diagnostics(series: np.ndarray, on_level: bool) -> tuple[float, float]:
    x = series if on_level else np.diff(series)
    alpha_hat = mcculloch_alpha(x)
    if not np.isfinite(alpha_hat):
        alpha_hat = 2.0
    h_hat = hurst_dfa(x)
    if not np.isfinite(h_hat):
        h_hat = 0.5
    return float(alpha_hat), float(h_hat)


def _logret(s: pd.Series) -> np.ndarray:
    s = pd.to_numeric(s, errors="coerce").replace(0.0, np.nan).dropna()
    return np.log(s).diff().dropna().to_numpy()


def _domains() -> dict[str, tuple[dict[str, np.ndarray], bool]]:
    """Return {domain: ({series_name: array}, diag_on_level)}.

    diag_on_level=True diagnoses alpha/H on the series itself (for the
    volatility / long-memory level series); otherwise on its first
    difference (the default for returns and sensor levels)."""
    d: dict[str, tuple[dict[str, np.ndarray], bool]] = {}
    ext = sum(EXTENDED_TICKERS.values(), [])

    # ----- financial returns ----------------------------------------
    try:
        fin = fetch_returns(ext)
        d["financial_daily"] = ({c: fin[c].dropna().to_numpy()
                                 for c in list(fin.columns)[:6]}, False)
    except Exception:
        pass
    for itv, key in [("15m", "financial_15m"), ("5m", "financial_5m")]:
        try:
            intr = fetch_intraday(ext, interval=itv)
            r = np.log(intr / intr.shift(1)).dropna(how="all")
            d[key] = ({c: r[c].dropna().to_numpy()
                       for c in list(r.columns)[:3]}, False)
        except Exception:
            pass
    crypto = {}
    for sym in ("BTCUSDT", "ETHUSDT"):
        try:
            crypto[sym] = _logret(fetch_binance(sym)["close"])
        except Exception:
            pass
    if crypto:
        d["financial_crypto_1h"] = (crypto, False)

    # ----- financial volatility |r| (long-memory regime) ------------
    try:
        fin = fetch_returns(ext)
        d["financial_vol_daily"] = ({f"|{c}|": np.abs(fin[c].dropna().to_numpy())
                                     for c in list(fin.columns)[:3]}, True)
    except Exception:
        pass
    if crypto:
        d["financial_vol_crypto"] = ({f"|{k}|": np.abs(v)
                                      for k, v in crypto.items()}, True)

    # ----- macro -----------------------------------------------------
    try:
        fred = fetch_fred(DEFAULT_FRED_SERIES)
        d["macro_fred"] = ({c: fred[c].dropna().to_numpy()
                            for c in fred.columns
                            if fred[c].dropna().size > 300}, False)
    except Exception:
        pass

    # ----- sensors: ETT family + electricity + traffic --------------
    for v in ("ETTh1", "ETTh2", "ETTm1", "ETTm2"):
        try:
            ett = fetch_nonfinancial(variant=v)
            d[f"sensor_{v.lower()}"] = ({"OT": ett["OT"].to_numpy()[::4],
                                         "HUFL": ett["HUFL"].to_numpy()[::4]},
                                        False)
        except Exception:
            pass
    try:
        el = fetch_electricity(n_series=4)
        d["electricity_load"] = ({c: el[c].to_numpy() for c in el.columns}, True)
    except Exception:
        pass
    try:
        tr = fetch_traffic(n_series=4)
        d["traffic_pems"] = ({c: tr[c].to_numpy() for c in tr.columns}, True)
    except Exception:
        pass

    # ----- weather / prognostics / operational / scientific ---------
    try:
        wx = fetch_weather_noaa()
        d["noaa_weather"] = ({c: wx[c].dropna().to_numpy()
                              for c in wx.columns}, False)
    except Exception:
        pass
    try:
        cm = fetch_cmapss()
        d["cmapss_turbofan"] = ({c: cm[c].dropna().to_numpy()
                                 for c in list(cm.columns)[:6]}, False)
    except Exception:
        pass
    try:
        atm = fetch_atm()
        d["atm_withdrawals"] = ({"withdrawn": atm["withdrawn"].to_numpy()}, False)
    except Exception:
        pass
    try:
        sn = fetch_sunspots()
        d["scientific_sunspots"] = ({"sn": sn["sunspot"].to_numpy()[::5]}, True)
    except Exception:
        pass
    return d


def run_cell(domain, name, series, diag_level, filt_key, opt, classification, seed):
    s = _standardize(series)
    if s.size < 6 * P + 80:
        return None
    cut = int(0.7 * s.size)
    alpha_hat, h_hat = _diagnostics(s[:cut], diag_level)

    filt = STUDY_FILTERS[filt_key]()
    sf = filt.apply(s)
    Xtr, ytr = _ar_design(sf[:cut], P)
    Xte_f, _ = _ar_design(sf[cut - P:], P)
    _, yte_raw = _ar_design(s[cut - P:], P)
    m = min(Xte_f.shape[0], yte_raw.shape[0])
    if Xtr.shape[0] < 50 or m < 20:
        return None

    task = _LinAR(P, classification=classification)
    task.attach(Xtr, ytr)
    res = run_optimization(task=task, optimizer=opt,
                           noise=GaussianNoise(0.05), filt=IdentityFilter(),
                           steps=STEPS, lr=1e-2, batch_size=64, seed=seed,
                           noise_scale=1.0, preprocess_mode="series")
    g = res.grad_norm_sq_history
    td = time_to_drop(g, factor=1e2)
    fq = noise_floor_quantiles(g, tail_frac=0.2)

    pred = Xte_f[:m] @ res.x_final
    if classification:
        ho = float(np.mean((pred > 0).astype(float)
                           == (yte_raw[:m] > 0).astype(float)))
    else:
        ho = float(np.mean((pred - yte_raw[:m]) ** 2))

    return dict(domain=domain, series=name, filter=filt_key, optimizer=opt,
                model=("classification" if classification else "regression"),
                seed=seed, causal=int(filt_key in CAUSAL),
                t_conv=STEPS if td is None else td,
                conv=int(td is not None),
                slope=divergence_slope(g),
                floor_p50=fq[0.5], floor_p10=fq[0.1], floor_p90=fq[0.9],
                auc=convergence_auc(g), final_grad=float(g[-1]),
                holdout=ho, alpha_hat=alpha_hat, hurst_hat=h_hat)


def _bucket(ah, hh):
    if ah >= 1.9 and hh <= 0.6:
        return "light_short"
    if ah >= 1.9 and hh > 0.6:
        return "light_long"
    if ah < 1.9 and hh <= 0.6:
        return "heavy_short"
    return "heavy_long"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=9)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--csv", type=Path, default=Path("tables/new_datasets.csv"))
    ap.add_argument("--summary-csv", type=Path,
                    default=Path("tables/new_datasets_summary.csv"))
    ap.add_argument("--criteria-csv", type=Path,
                    default=Path("tables/new_datasets_criteria.csv"))
    ap.add_argument("--rules-csv", type=Path,
                    default=Path("tables/new_datasets_rules.csv"))
    args = ap.parse_args()

    dom = _domains()
    print("domains built:", {k: len(v[0]) for k, v in dom.items()})
    filters = list(STUDY_FILTERS)
    seeds = SEEDS
    models = [False, True]
    if args.smoke:
        dom = {k: ({n: s for n, s in list(v[0].items())[:1]}, v[1])
               for k, v in list(dom.items())}
        filters = ["F0", "F4", "F5", "F6"]
        seeds = [0]
        models = [False]

    cells = [(dn, nm, sv, lvl, fk, o, cls, s)
             for dn, (series, lvl) in dom.items()
             for nm, sv in series.items()
             for fk in filters for o in OPTIMIZERS
             for cls in models for s in seeds]
    print(f"running {len(cells)} cells on {args.n_jobs} jobs")
    t0 = time.perf_counter()
    out = [r for r in Parallel(n_jobs=args.n_jobs, verbose=4, backend="loky")(
        delayed(run_cell)(*c) for c in cells) if r is not None]
    print(f"done in {time.perf_counter() - t0:.1f}s, {len(out)} valid cells")

    df = pd.DataFrame(out)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)

    g = df.groupby(["domain", "model", "optimizer", "filter"])
    summ = g.agg(
        n_cells=("seed", "size"), causal=("causal", "max"),
        t_conv_med=("t_conv", "median"), conv_frac=("conv", "mean"),
        floor_p50_med=("floor_p50", "median"), slope_mean=("slope", "mean"),
        auc_mean=("auc", "mean"), holdout_med=("holdout", "median"),
        alpha_hat_med=("alpha_hat", "median"),
        hurst_hat_med=("hurst_hat", "median"),
    ).reset_index()
    summ["speedup_vs_F0"] = np.nan
    summ["floor_ratio_vs_F0"] = np.nan
    for _, sub in summ.groupby(["domain", "model", "optimizer"]):
        f0 = sub[sub["filter"] == "F0"]
        if f0.empty:
            continue
        f0t = float(f0["t_conv_med"].iloc[0])
        f0f = float(f0["floor_p50_med"].iloc[0])
        if f0t > 0:
            summ.loc[sub.index, "speedup_vs_F0"] = f0t / sub["t_conv_med"].clip(lower=1)
        if f0f > 0:
            summ.loc[sub.index, "floor_ratio_vs_F0"] = f0f / sub["floor_p50_med"]
    summ["bucket"] = [_bucket(a, h) for a, h in
                      zip(summ["alpha_hat_med"], summ["hurst_hat_med"])]
    summ.to_csv(args.summary_csv, index=False)

    # ---- best CAUSAL filter per (domain, model, optimizer) ----------
    rows = []
    for (dn, mdl, opt), sub in summ.groupby(["domain", "model", "optimizer"]):
        f0 = sub[sub["filter"] == "F0"]
        if f0.empty:
            continue
        f0h = float(f0["holdout_med"].iloc[0])
        f0t = float(f0["t_conv_med"].iloc[0])
        cau = sub[(sub["causal"] == 1) & (sub["filter"] != "F0")
                  & (sub["conv_frac"] >= 0.75)
                  & (sub["holdout_med"] <= 1.10 * f0h)]
        if cau.empty:
            best_name, sp, bh = "F0", 1.0, f0h
        else:
            b = cau.loc[cau["t_conv_med"].idxmin()]
            best_name = str(b["filter"])
            sp = f0t / max(float(b["t_conv_med"]), 1.0)
            bh = float(b["holdout_med"])
        ah = float(f0["alpha_hat_med"].iloc[0])
        hh = float(f0["hurst_hat_med"].iloc[0])
        rows.append(dict(domain=dn, model=mdl, optimizer=opt,
                         alpha_hat=ah, hurst_hat=hh, bucket=_bucket(ah, hh),
                         best_filter=best_name, speedup=sp,
                         best_holdout=bh, f0_holdout=f0h,
                         f0_conv_frac=float(f0["conv_frac"].iloc[0])))
    crit = pd.DataFrame(rows)
    crit.to_csv(args.criteria_csv, index=False)

    # ---- (alpha,H) bucket -> dominant recommended filter ------------
    rule_rows = []
    for bk, sub in crit.groupby("bucket"):
        vc = sub["best_filter"].value_counts()
        rule_rows.append(dict(
            bucket=bk, n=int(len(sub)),
            alpha_lo=float(sub["alpha_hat"].min()),
            alpha_hi=float(sub["alpha_hat"].max()),
            H_lo=float(sub["hurst_hat"].min()),
            H_hi=float(sub["hurst_hat"].max()),
            dominant_filter=str(vc.index[0]),
            dominant_share=float(vc.iloc[0] / len(sub)),
            median_speedup=float(sub["speedup"].median()),
            example_domains=", ".join(sorted(sub["domain"].unique())[:4])))
    rules = pd.DataFrame(rule_rows).sort_values("bucket")
    rules.to_csv(args.rules_csv, index=False)

    print("\n=== best causal filter per (domain, model, optimizer) ===")
    for _, r in crit.iterrows():
        print(f"  {r['domain']:22s} {r['model'][:5]:5s} {r['optimizer']:4s} "
              f"a={r['alpha_hat']:.2f} H={r['hurst_hat']:.2f} [{r['bucket']:11s}]"
              f" -> {r['best_filter']:3s} sp={r['speedup']:6.1f}x")
    print("\n=== derived rule: (alpha,H) bucket -> filter ===")
    for _, r in rules.iterrows():
        print(f"  {r['bucket']:11s} n={r['n']:2d}  -> {r['dominant_filter']:3s} "
              f"({r['dominant_share']*100:.0f}%), med speedup {r['median_speedup']:.1f}x"
              f"  [{r['example_domains']}]")
    print(f"\nwrote {args.csv}, {args.summary_csv}, {args.criteria_csv}, {args.rules_csv}")


if __name__ == "__main__":
    main()
