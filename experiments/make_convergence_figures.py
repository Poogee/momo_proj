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
    axa.set_ylabel(r"асимпт. уровень $\|\nabla f\|^2$ (медиана, p10–p90)")
    axa.set_title("(а) Тяжёлые хвосты N3, SGD: $F_4$ снижает $\\|\\nabla f\\|^2$")
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


def fig_gaussian_control(summ_csv, curves_npz, out):
    """N1 Gaussian control (H4): filtering is neutral; F4 slightly worse
    than F0/F1 on the distribution for which the mean is optimal."""
    summ = pd.read_csv(summ_csv)
    sns.set_theme(context="paper", style="whitegrid", font_scale=0.92)
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4.6))

    a = summ[(summ.block == "A") & (summ.noise == "N1")
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
    axa.set_ylabel(r"асимпт. уровень $\|\nabla f\|^2$ (медиана, p10–p90)")
    axa.set_title("(а) Гауссов N1, SGD: фильтры почти не различаются")
    axa.legend(fontsize=8, ncol=2)

    try:
        cur = np.load(curves_npz)
        for m, ls in [("quadratic", "-"), ("logistic", "--")]:
            for fk, col in [("F0", "C3"), ("F1", "C2"), ("F4", "C0")]:
                ks = [k for k in cur.files
                      if k.startswith(f"A|{m}|N1|sgd|{fk}|")]
                if not ks:
                    continue
                arr = np.stack([cur[k] for k in ks])
                med = np.median(arr, axis=0)
                xx = np.linspace(0, 1, med.size)
                axb.plot(xx, np.maximum(med, 1e-12), col, ls=ls, lw=1.4,
                         label=f"{m[:4]} {fk}")
        axb.set_yscale("log")
        axb.set_xlabel("доля горизонта")
        axb.set_ylabel(r"$\|\nabla f(x_k)\|^2$ (медиана по сидам)")
        axb.set_title("(б) F0/F1/F4 идут вместе; выигрыша от фильтрации нет")
        axb.legend(fontsize=8, ncol=2)
    except Exception as e:  # pragma: no cover
        axb.text(0.5, 0.5, f"curves n/a\n{e}", ha="center")

    fig.suptitle("Гауссов контроль N1: фильтрация нейтральна, медиана F4 "
                 "слегка проигрывает среднему (8 сидов)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_longmemory(summ_csv, curves_npz, out):
    """Long-memory N2 (H2): wavelet F3 accelerates Adam/AdamW; linear
    smoothers help little. Panel (a) median curves, (b) speed-up bars."""
    summ = pd.read_csv(summ_csv)
    sns.set_theme(context="paper", style="whitegrid", font_scale=0.92)
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4.6))

    # (a) median ||g||^2 curves for N2 quadratic adam, F0 vs F2 vs F3
    try:
        cur = np.load(curves_npz)
        for fk, col in [("F0", "C3"), ("F2", "C1"), ("F3", "C0")]:
            ks = [k for k in cur.files
                  if k.startswith(f"B|quadratic|N2|adam|{fk}|")]
            if not ks:
                continue
            arr = np.stack([cur[k] for k in ks])
            med = np.median(arr, axis=0)
            xx = np.linspace(0, 1, med.size)
            axa.plot(xx, np.maximum(med, 1e-10), col, lw=1.6, label=fk)
        axa.set_yscale("log")
        axa.set_xlabel("доля горизонта")
        axa.set_ylabel(r"$\|\nabla f(x_k)\|^2$ (медиана по сидам)")
        axa.set_title("(а) N2, квадратичная, Adam: F3 уходит вниз раньше")
        axa.legend(fontsize=8)
    except Exception as e:  # pragma: no cover
        axa.text(0.5, 0.5, f"curves n/a\n{e}", ha="center")

    # (b) speed-up vs F0 (t_eps) per filter, N2 quadratic, adam & adamw
    b = summ[(summ.block == "B") & (summ.noise == "N2")
             & (summ.model == "quadratic")]
    filt = [f for f in FILT_ORDER if f in b["filter"].unique()]
    x = np.arange(len(filt))
    for i, opt in enumerate(["adam", "adamw"]):
        ys = []
        for fk in filt:
            r = b[(b.optimizer == opt) & (b["filter"] == fk)]
            ys.append(float(r["speedup_vs_F0"].iloc[0]) if not r.empty else np.nan)
        axb.bar(x + i * 0.4, ys, 0.4, label=opt)
    axb.axhline(1.0, color="0.4", lw=0.8, ls=":")
    axb.set_xticks(x + 0.2)
    axb.set_xticklabels(filt)
    axb.set_ylabel(r"ускорение $T(\varepsilon)$ относительно F0")
    axb.set_title("(б) F3 даёт ~11× ускорение; линейные фильтры — нет")
    axb.legend(fontsize=8)

    fig.suptitle("Долгая память N2: вейвлет F3 ускоряет адаптивные "
                 "оптимизаторы (8 сидов)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def fig_longmemory_auc(summ_csv, curves_npz, out):
    """Mean-log-gradient visual for long-memory N2. The metric is the MEAN
    of log10||g||^2 over the trajectory (not an area under ||g||^2), hence
    negative since ||g||^2<1; lower = better.
    (a) the log10 curves with their mean drawn as a dashed line;
    (b) mean-log bars per filter for Adam/AdamW."""
    summ = pd.read_csv(summ_csv)
    sns.set_theme(context="paper", style="whitegrid", font_scale=0.92)
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4.6))

    try:
        cur = np.load(curves_npz)
        for fk, col in [("F0", "C3"), ("F2", "C1"), ("F3", "C0")]:
            ks = [k for k in cur.files
                  if k.startswith(f"B|quadratic|N2|adam|{fk}|")]
            if not ks:
                continue
            arr = np.stack([cur[k] for k in ks])
            med = np.median(arr, axis=0)
            y = np.log10(np.maximum(med, 1e-12))
            xx = np.linspace(0, 1, y.size)
            axa.plot(xx, y, col, lw=1.4, label=fk)
            axa.axhline(float(np.mean(y)), color=col, ls="--", lw=1.0)
        axa.set_xlabel("доля горизонта")
        axa.set_ylabel(r"$\log_{10}\|\nabla f(x_k)\|^2$")
        axa.set_title(r"(а) лог-кривые; пунктир — среднее $\overline{\log_{10}\|\nabla f\|^2}$")
        axa.legend(fontsize=8)
    except Exception as e:  # pragma: no cover
        axa.text(0.5, 0.5, f"curves n/a\n{e}", ha="center")

    b = summ[(summ.block == "B") & (summ.noise == "N2")
             & (summ.model == "quadratic")]
    filt = [f for f in FILT_ORDER if f in b["filter"].unique()]
    x = np.arange(len(filt))
    for i, opt in enumerate(["adam", "adamw"]):
        ys = [float(b[(b.optimizer == opt) & (b["filter"] == fk)]
                    ["auc_mean"].iloc[0]) for fk in filt]
        axb.bar(x + i * 0.4, ys, 0.4, label=opt)
    axb.set_xticks(x + 0.2)
    axb.set_xticklabels(filt)
    axb.set_ylabel(r"среднее $\log_{10}\|\nabla f\|^2$ (ниже — лучше)")
    axb.set_title("(б) по фильтрам: у F3 наименьшее (лучшее)")
    axb.legend(fontsize=8)

    fig.suptitle(r"Долгая память N2, Adam: средний $\log_{10}\|\nabla f\|^2$ "
                 r"по траектории (отрицателен, т.к. $\|\nabla f\|^2<1$)",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
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
        ax.set_ylabel(r"асимпт. уровень $\|\nabla f\|^2$ (медиана)")
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
        fig_gaussian_control(args.rescue_summary, args.curves,
                             Path("figures/gaussian_control.pdf"))
        fig_longmemory(args.rescue_summary, args.curves,
                       Path("figures/longmemory_wavelet.pdf"))
        fig_longmemory_auc(args.rescue_summary, args.curves,
                           Path("figures/longmemory_auc.pdf"))
    if args.applied_summary.exists():
        fig_applied(args.applied_summary,
                    Path("figures/applied_convergence.pdf"))
    if args.calibrated.exists():
        fig_calibrated(args.calibrated, args.diag,
                       Path("figures/calibrated_synthetic.pdf"))


if __name__ == "__main__":
    main()
