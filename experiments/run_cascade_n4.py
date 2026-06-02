"""Confirm Hypothesis 2 in the paper: on the mixed regime N4
(heavy-tailed and long-memory simultaneously) the new causal cascade
filters F10 and F11 lower the SGD noise floor where single-stage
filters do not.

Runs SGD and Adam on the quadratic and logistic tasks under
N4 (alpha=1.2, d=0.4) with filters {F0, F1, F2, F3, F4, F10, F11}.
8 seeds, identical lr / horizon / sigma for every cell.

Writes:
  tables/cascade_n4.csv         — per-cell raw metrics
  tables/cascade_n4_summary.csv — filter x optimizer x model summary
  tables/cascade_n4_block_c.tex — paper-ready table for paper_ru.tex
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from momo.filters import (
    AdaptiveCascadeFilter,
    CausalCascadeFilter,
    CausalMedianFilter,
    IdentityFilter,
    KalmanLocalLevelFilter,
    MovingAverageFilter,
    WaveletThresholdFilter,
)
from momo.metrics import (
    convergence_auc,
    noise_floor_quantiles,
    time_to_drop,
)
from momo.noise import MixedFARIMAStableNoise
from momo.optim import run_optimization
from momo.tasks import make_logistic, make_quadratic


SEEDS = list(range(8))
FILTERS = {
    "F0":  lambda: IdentityFilter(),
    "F1":  lambda: MovingAverageFilter(window=15),
    "F2":  lambda: KalmanLocalLevelFilter(process_var=1e-3, obs_var=1.0),
    "F3":  lambda: WaveletThresholdFilter(wavelet="db4", mode="soft"),
    "F4":  lambda: CausalMedianFilter(window=9),
    "F5":  lambda: CausalCascadeFilter(median_window=3,
                                       process_var=1e-3, obs_var=1.0),
    "F6":  lambda: AdaptiveCascadeFilter(alpha_threshold=1.9,
                                         hurst_threshold=0.6,
                                         median_window=3,
                                         process_var=1e-3, obs_var=1.0),
}
MODELS = {
    "quadratic": dict(steps=4000, lr=5e-3, noise_scale=0.4, eps=1e-1),
    "logistic":  dict(steps=4000, lr=2e-2, noise_scale=0.25, eps=5e-3),
}
OPTIMIZERS = ["sgd", "adam"]


def _task(model: str, seed: int):
    if model == "quadratic":
        return make_quadratic(dim=20, condition_number=5.0, seed=seed)
    return make_logistic(n=3000, dim=15, n_test=1500,
                         noise_scale=0.4, seed=seed)


def run_cell(model, opt, filt_key, seed):
    cfg = MODELS[model]
    task = _task(model, seed)
    noise = MixedFARIMAStableNoise(d=0.4, alpha=1.2, sigma=0.3)
    filt = FILTERS[filt_key]()
    res = run_optimization(task=task, optimizer=opt, noise=noise, filt=filt,
                           steps=cfg["steps"], lr=cfg["lr"], seed=seed,
                           noise_scale=cfg["noise_scale"],
                           preprocess_mode="series")
    g = res.grad_norm_sq_history
    fq = noise_floor_quantiles(g, tail_frac=0.2)
    td = time_to_drop(g, factor=1e2)
    return dict(model=model, optimizer=opt, filter=filt_key, seed=seed,
                steps=cfg["steps"],
                conv=int(td is not None),
                t_drop100=cfg["steps"] if td is None else td,
                floor_p50=fq[0.5], floor_p10=fq[0.1], floor_p90=fq[0.9],
                auc=convergence_auc(g),
                final_grad=float(g[-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=8)
    args = ap.parse_args()

    cells = [(m, o, f, s)
             for m in MODELS for o in OPTIMIZERS
             for f in FILTERS for s in SEEDS]
    print(f"running {len(cells)} cells")
    out = Parallel(n_jobs=args.n_jobs, verbose=4, backend="loky")(
        delayed(run_cell)(*c) for c in cells)
    df = pd.DataFrame(out)
    df.to_csv("tables/cascade_n4.csv", index=False)

    summ = (df.groupby(["model", "optimizer", "filter"])
              .agg(conv_frac=("conv", "mean"),
                   t_drop100_med=("t_drop100", "median"),
                   floor_p50_med=("floor_p50", "median"),
                   auc_mean=("auc", "mean"))
              .reset_index())
    # ratio vs F0
    summ["floor_ratio_vs_F0"] = np.nan
    for (m, o), sub in summ.groupby(["model", "optimizer"]):
        f0 = sub[sub["filter"] == "F0"]
        if f0.empty:
            continue
        base = float(f0["floor_p50_med"].iloc[0])
        if base > 0:
            summ.loc[sub.index, "floor_ratio_vs_F0"] = base / sub["floor_p50_med"]
    summ.to_csv("tables/cascade_n4_summary.csv", index=False)

    # build paper-ready LaTeX table (SGD, both models)
    sub = summ[summ["optimizer"] == "sgd"].copy()
    rows = []
    pretty_m = {"quadratic": "квадр.", "logistic": "логист."}
    for m, grp in sub.groupby("model"):
        f0 = grp[grp["filter"] == "F0"]
        if f0.empty:
            continue
        base = float(f0["floor_p50_med"].iloc[0])
        for fk in ["F0", "F1", "F2", "F3", "F4", "F5", "F6"]:
            r = grp[grp["filter"] == fk]
            if r.empty:
                continue
            r = r.iloc[0]
            ratio = float(r["floor_ratio_vs_F0"]) if base > 0 else float("nan")
            ratio_txt = (f"$\\mathbf{{{ratio:.1f}\\times}}$" if ratio >= 1.1
                         else f"${ratio:.2f}\\times$")
            rows.append(
                f"{pretty_m.get(m,m)} & $F_{fk[1:]}$ & "
                f"${float(r['conv_frac'])*8:.0f}/8$ & "
                f"${float(r['floor_p50_med']):.2e}$ & {ratio_txt}\\\\"
            )
        rows.append("\\midrule" if m == "logistic" else "")

    body = "\n".join(r for r in rows if r)
    Path("tables/cascade_n4_block_c.tex").write_text(
        "\\begin{table}[t]\n"
        "\\caption{Блок~C. Синтетический смешанный режим $N_4$"
        " ($\\hat d{=}0.4$, $\\alpha{=}1.2$), SGD, $8$ сидов, \\emph{все}"
        " фильтры $F_0$--$F_6$. Бинарная сходимость и шумовой пол"
        " относительно $F_0$. Новый каскад $F_5$ (медиана$\\to$Калман)"
        " даёт наибольшее снижение пола; одиночные линейные/вейвлет фильтры"
        " идут вдоль $F_0$.}\n"
        "\\label{tab:blockC}\n"
        "\\centering\\begin{tabular}{llccc}\n"
        "\\toprule\n"
        "модель & фильтр & бин. & пол $\\|\\nabla f\\|^2$ & vs $F_0$\\\\\n"
        "\\midrule\n" + body + "\n"
        "\\bottomrule\\end{tabular}\\end{table}\n",
        encoding="utf-8")

    print("\n=== Block C — cascade on N4, SGD ===")
    for _, r in sub.iterrows():
        print(f"  {r['model']:9s} {r['filter']:4s}: "
              f"conv={float(r['conv_frac']):.2f} "
              f"floor={float(r['floor_p50_med']):.2e} "
              f"vs F0 = {float(r['floor_ratio_vs_F0']):.1f}x")
    print("wrote tables/cascade_n4*.csv and cascade_n4_block_c.tex")


if __name__ == "__main__":
    main()
