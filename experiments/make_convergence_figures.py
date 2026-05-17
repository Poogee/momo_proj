"""Figures for the positive convergence story.

  figures/convergence_rescue.pdf   — (a) heavy-tailed SGD noise floor
      F0 vs filters per model (log scale, p10–p90 whiskers),
      (b) median ||g||^2 curves F0 vs F2 vs F4.
  figures/applied_convergence.pdf  — per real domain: iterations to
      converge (Adam) and causal holdout MSE, F0 vs causal filters.
  figures/calibrated_synthetic.pdf — floor F0 vs F4 at the
      data-calibrated tail index.

All inputs are the CSV/NPZ written by the run_* scripts; panels are
skipped gracefully if a file is missing.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FILT_ORDER = ["F0", "F1", "F2", "F3", "F4"]
MODEL_LBL = {"quadratic": "квадратичная регр.", "logistic": "логистич. класс.",
             "ar": "авторегрессия"}


def fig_rescue(summ_csv, curves_npz, out):
    summ = pd.read_csv(summ_csv)
    sns.set_theme(context="paper", style="whitegrid", font_scale=0.92)
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4.6))

    # (a) heavy-tailed N3, sgd: floor F0 vs filters, per model
    a = summ[(summ.block == "A") & (summ.noise == "N3")
             & (summ.optimizer == "sgd")]
    models = [m for m in ["quadratic", "logistic", "ar"]
              if m in a.model.unique()]
    filt = [f for f in FILT_ORDER if f in a["filter"].unique()]
    x = np.arange(len(models))
    w = 0.8 / max(len(filt), 1)
    for i, fk in enumerate(filt):
        ys, lo, hi = [], [], []
        for m in models:
            r = a[(a.model == m) & (a["filter"] == fk)]
            v = float(r["floor_p50_med"].iloc[0]) if not r.empty else np.nan
            ys.append(v)
            lo.append(v - float(r["floor_p10_med"].iloc[0]) if not r.empty else 0)
            hi.append(float(r["floor_p90_med"].iloc[0]) - v if not r.empty else 0)
        axa.bar(x + i * w, ys, w, yerr=[np.abs(lo), np.abs(hi)],
                capsize=2, label=fk, error_kw=dict(lw=0.6))
    axa.set_yscale("log")
    axa.set_xticks(x + 0.4 - w / 2)
    axa.set_xticklabels([MODEL_LBL.get(m, m) for m in models], rotation=15,
                        ha="right", fontsize=9)
    axa.set_ylabel(r"шумовой пол $\|\nabla f\|^2$ (медиана, p10–p90)")
    axa.set_title("(а) Тяжёлые хвосты N3, SGD: фильтр снижает пол")
    axa.legend(fontsize=8, ncol=2)

    # (b) median curves F0 vs F2 vs F4, quadratic & ar, N3 sgd
    try:
        cur = np.load(curves_npz)
        for m, ls in [("quadratic", "-"), ("ar", "--")]:
            for fk, col in [("F0", "C3"), ("F2", "C1"), ("F4", "C0")]:
                ks = [k for k in cur.files
                      if k.startswith(f"A|{m}|N3|sgd|{fk}|")]
                if not ks:
                    continue
                arr = np.stack([cur[k] for k in ks])
                med = np.median(arr, axis=0)
                xx = np.linspace(0, 1, med.size)
                axb.plot(xx, np.maximum(med, 1e-10), col, ls=ls, lw=1.4,
                         label=f"{m[:4]} {fk}")
        axb.set_yscale("log")
        axb.set_xlabel("доля горизонта")
        axb.set_ylabel(r"$\|\nabla f(x_k)\|^2$ (медиана по сидам)")
        axb.set_title("(б) F0/F2 застревают высоко; F4 — на 1–2 порядка ниже")
        axb.legend(fontsize=8, ncol=2)
    except Exception as e:  # pragma: no cover
        axb.text(0.5, 0.5, f"curves n/a\n{e}", ha="center")

    fig.suptitle("Предфильтрация спасает сходимость SGD при тяжёлохвостовом "
                 "градиентном шуме (8 сидов)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_applied(summ_csv, out):
    s = pd.read_csv(summ_csv)
    s = s[s.optimizer == "adam"]
    doms = list(s.domain.unique())
    filt = [f for f in ["F0", "F1", "F2", "F4"]
            if f in s["filter"].unique()]
    sns.set_theme(context="paper", style="whitegrid", font_scale=0.92)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    x = np.arange(len(doms))
    w = 0.8 / max(len(filt), 1)
    for i, fk in enumerate(filt):
        tc = []
        for d in doms:
            r = s[(s.domain == d) & (s["filter"] == fk)]
            if r.empty:
                tc.append(np.nan)
                continue
            t = float(r["t_conv_med"].iloc[0])
            cf = float(r["conv_frac"].iloc[0])
            # show iterations T; drop cells that did NOT converge
            # (majority of seeds never reached the 100x drop).
            tc.append(t if (np.isfinite(t) and cf >= 0.5) else np.nan)
        axes[0].bar(x + i * w, tc, w, label=fk)
    axes[0].set_xticks(x + 0.4 - w / 2)
    axes[0].set_xticklabels(doms, rotation=20, ha="right", fontsize=8)
    axes[0].set_ylabel(r"итераций $T$ до $100\times$ падения "
                       r"$\|\nabla f\|^2$ (Adam)")
    axes[0].set_title("(а) Время сходимости по доменам "
                      "(ниже — быстрее; не сошедшиеся опущены)")
    axes[0].legend(fontsize=7, ncol=2)
    for i, fk in enumerate(filt):
        hm = [float(s[(s.domain == d) & (s["filter"] == fk)]
                    ["holdout_mse_med"].iloc[0])
              if not s[(s.domain == d) & (s["filter"] == fk)].empty
              else np.nan for d in doms]
        axes[1].bar(x + i * w, hm, w, label=fk)
    axes[1].set_xticks(x + 0.4 - w / 2)
    axes[1].set_xticklabels(doms, rotation=20, ha="right", fontsize=8)
    axes[1].set_ylabel("causal holdout MSE (raw target)")
    axes[1].set_title("(б) Качество прогноза (causal walk-forward)")
    axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle("Применённый эксперимент: причинная фильтрация по доменам "
                 "(финансы / макро / нефинансовый сенсор)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_calibrated(csv, diag_csv, out):
    df = pd.read_csv(csv)
    sns.set_theme(context="paper", style="whitegrid", font_scale=0.92)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    a_used = float(df["alpha"].iloc[0])
    d_used = float(df["d"].iloc[0])
    for ax, nk in zip(axes, ["N3cal", "N4cal"]):
        sub = df[df.noise == nk]
        models = list(sub.model.unique())
        x = np.arange(len(models))
        for i, fk in enumerate(["F0", "F4"]):
            ys = [float(sub[(sub.model == m) & (sub["filter"] == fk)]
                        ["floor_p50"].median()) for m in models]
            ax.bar(x + i * 0.3, ys, 0.3, label=fk)
        ax.set_yscale("log")
        ax.set_xticks(x + 0.15)
        ax.set_xticklabels(models, fontsize=8)
        ax.set_ylabel(r"шумовой пол $\|\nabla f\|^2$ (медиана)")
        ax.set_title(f"{nk}  (α̂={a_used:.2f}"
                     + (f", d̂={d_used:.2f}" if nk == "N4cal" else "") + ")")
        ax.legend(fontsize=8)
    fig.suptitle("Калиброванная под реальные ряды синтетика: эффект "
                 "сохраняется при реалистичных α̂/d̂", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescue-summary", type=Path,
                    default=Path("tables/convergence_rescue_summary.csv"))
    ap.add_argument("--curves", type=Path,
                    default=Path("runs/convergence_rescue/curves.npz"))
    ap.add_argument("--applied-summary", type=Path,
                    default=Path("tables/applied_convergence_summary.csv"))
    ap.add_argument("--calibrated", type=Path,
                    default=Path("tables/calibrated_synthetic.csv"))
    ap.add_argument("--diag", type=Path,
                    default=Path("tables/real_noise_calibration.csv"))
    args = ap.parse_args()
    if args.rescue_summary.exists():
        fig_rescue(args.rescue_summary, args.curves,
                   Path("figures/convergence_rescue.pdf"))
    if args.applied_summary.exists():
        fig_applied(args.applied_summary,
                    Path("figures/applied_convergence.pdf"))
    if args.calibrated.exists():
        fig_calibrated(args.calibrated, args.diag,
                       Path("figures/calibrated_synthetic.pdf"))


if __name__ == "__main__":
    main()
