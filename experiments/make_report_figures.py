"""Generate the figures used in the main report from fresh series-filtering data.

Outputs (into figures/):
  conv_curves.pdf   convergence curves (median + p10-p90 band), quadratic, Adam
  snr_heatmap.pdf   SNR gain (dB) heatmap, filter x noise
  synth_bars.pdf    quadratic noise floor + logistic holdout accuracy
  real_mse.pdf      real causal walk-forward MSE by filter
"""
from __future__ import annotations

import glob
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

NOISE = ["N1", "N2", "N3", "N4"]
NOISE_LBL = {"N1": "N1 гаусс", "N2": "N2 розовый", "N3": "N3 тяжёлый", "N4": "N4 смеш."}
FILT = ["F0", "F1", "F2", "F3", "F4"]
FILT_LBL = {"F0": "F0 без фильтра", "F1": "F1 MA", "F2": "F2 Калман",
            "F3": "F3 вейвлет", "F4": "F4 медиана"}
COL = {"F0": "#888888", "F1": "#1f77b4", "F2": "#d62728",
       "F3": "#2ca02c", "F4": "#9467bd"}


def fig_convergence():
    """Median + p10-p90 band of ||grad||^2 vs iteration, quadratic / Adam."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, noise in zip(axes, ["N1", "N3"]):
        for f in ["F0", "F2", "F4"]:
            files = sorted(glob.glob(
                str(ROOT / f"runs/synthetic_series/quadratic/{f}_{noise}_adam/seed*.npz")))
            if not files:
                continue
            curves = np.stack([np.load(p)["grad_norm_sq"] for p in files])
            med = np.median(curves, axis=0)
            lo = np.percentile(curves, 10, axis=0)
            hi = np.percentile(curves, 90, axis=0)
            it = np.arange(med.size)
            ax.plot(it, med, color=COL[f], label=FILT_LBL[f], lw=1.5)
            ax.fill_between(it, lo, hi, color=COL[f], alpha=0.15)
        ax.set_yscale("log")
        ax.set_title(f"Квадратичная задача, Adam, {NOISE_LBL[noise]}")
        ax.set_xlabel("итерация $k$")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$\|\nabla f(x_k)\|^2$")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "conv_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_snr_heatmap():
    d = pd.read_csv(ROOT / "tables/filter_diagnostics.csv")
    d = d[d["filter"].isin(["F1", "F2", "F3", "F4"])]
    piv = d.pivot_table(index="filter", columns="noise",
                        values="delta_snr_db", aggfunc="mean")
    piv = piv.reindex(["F1", "F2", "F3", "F4"])[NOISE]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(NOISE)), [NOISE_LBL[n] for n in NOISE],
                  rotation=20, ha="right")
    ax.set_yticks(range(4), [FILT_LBL[f] for f in ["F1", "F2", "F3", "F4"]])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.1f}", ha="center",
                    va="center", color="w", fontsize=9)
    ax.set_title("Прирост SNR после фильтрации, дБ")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(FIG / "snr_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_synth_bars():
    s = pd.read_csv(ROOT / "tables/synthetic_series_summary.csv")
    q = (s[s.task == "quadratic"]
         .pivot_table(index="filter", columns="noise",
                      values="final_grad_norm_sq", aggfunc="median")
         .reindex(FILT)[NOISE])
    l = (s[s.task == "logistic"]
         .pivot_table(index="filter", columns="noise",
                      values="holdout_metric", aggfunc="mean")
         .reindex(FILT)[NOISE])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    x = np.arange(len(NOISE))
    w = 0.16
    for k, f in enumerate(FILT):
        axes[0].bar(x + (k - 2) * w, q.loc[f].values, w,
                    color=COL[f], label=FILT_LBL[f])
        axes[1].bar(x + (k - 2) * w, l.loc[f].values, w, color=COL[f])
    axes[0].set_yscale("log")
    axes[0].set_title("Квадратичная: шумовой пол (медиана $\\|\\nabla f\\|^2$)")
    axes[0].set_xticks(x, [NOISE_LBL[n] for n in NOISE])
    axes[0].set_ylabel("ниже — лучше")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_title("Логистическая: holdout-точность")
    axes[1].set_xticks(x, [NOISE_LBL[n] for n in NOISE])
    axes[1].set_ylim(0.5, 0.86)
    axes[1].set_ylabel("выше — лучше")
    for ax in axes:
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "synth_bars.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_real_mse():
    r = pd.read_csv(ROOT / "tables/real_walkforward_causal.csv")
    g = (r.groupby("filter").holdout_mse.mean() * 1e4).sort_values()
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    colors = ["#2ca02c" if "F0" in n else "#d62728" for n in g.index]
    ax.bar(range(len(g)), g.values, color=colors)
    ax.set_xticks(range(len(g)), g.index, rotation=15, ha="right")
    ax.set_ylabel("holdout MSE ×10⁴ (ниже — лучше)")
    ax.set_title("Реальные дневные доходности: causal walk-forward")
    for i, v in enumerate(g.values):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "real_mse.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_speed():
    """Primary metric: convergence speed T(eps) on the quadratic task."""
    s = pd.read_csv(ROOT / "tables/synthetic_speed_summary.csv")
    q = s[s.task == "quadratic"].copy()
    cap = int(s["steps"].iloc[0]) if "steps" in s.columns else 2000
    q["teps"] = q.t_eps.replace(-1, cap)
    order = ["F0", "F1", "F2", "F3", "F4", "FA"]
    lbl = {"F0": "F0 без фильтра", "F1": "F1 MA", "F2": "F2 Калман",
           "F3": "F3 вейвлет", "F4": "F4 медиана", "FA": "FA онлайн-адапт."}
    piv = q.pivot_table(index="filter", columns="noise",
                        values="teps", aggfunc="median").reindex(order)[NOISE]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    x = np.arange(len(NOISE))
    w = 0.14
    cols = ["#888888", "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#e6a000"]
    for k, f in enumerate(order):
        vals = np.clip(piv.loc[f].values, 1, None)
        ax.bar(x + (k - 2.5) * w, vals, w, color=cols[k], label=lbl[f])
    ax.axhline(cap, ls="--", lw=0.8, color="k")
    ax.text(0.02, cap * 0.78, "floor-limited\n(не сошёлся)", fontsize=7,
            va="top")
    ax.set_yscale("log")
    ax.set_ylim(1, cap * 1.4)
    ax.set_xticks(x, [NOISE_LBL[n] for n in NOISE])
    ax.set_ylabel(r"медиана $T(\varepsilon{=}0.1)$, итер. (лог, ниже—быстрее)")
    ax.set_title(r"Скорость сходимости (квадратичная, Adam, "
                 r"$\sigma=0.05$, до 15000 шагов)")
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    ax.grid(alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    fig.savefig(FIG / "speed.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_intraday():
    d = pd.read_csv(ROOT / "tables/intraday_summary.csv")
    cls = {"BTC-USD": "крипто", "ETH-USD": "крипто", "EURUSD=X": "FX",
           "GBPUSD=X": "FX", "JPY=X": "FX"}
    d["cls"] = d.ticker.map(lambda t: cls.get(t, "акции"))
    order = ["F0 raw", "F1 MA", "F2 Kalman", "F3 Wavelet",
             "F4 Median", "FA online"]
    piv = d.pivot_table(index="filter", columns="cls",
                        values="rel_err", aggfunc="median")
    piv = piv.reindex([f for f in order if f in piv.index])
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    cols = list(piv.columns)
    x = np.arange(len(piv.index))
    w = 0.8 / len(cols)
    for k, c in enumerate(cols):
        ax.bar(x + (k - (len(cols) - 1) / 2) * w, piv[c].values, w, label=c)
    ax.set_xticks(x, piv.index, rotation=12, ha="right")
    ax.set_ylabel(r"медиана $|RV_{1m}-RV_{5m}|/RV_{5m}$")
    ax.set_title("Реальные 1-мин данные: отклонение от 5-мин бенчмарка")
    ax.legend(fontsize=8, title="класс актива")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "intraday.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_real_conv():
    d = pd.read_csv(ROOT / "tables/real_convergence_summary.csv")
    order = ["F0", "F1", "F2", "F3", "F4", "FA"]
    cols = ["#888888", "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#e6a000"]
    dsets = ["daily", "5min", "1min"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, dn in zip(axes, dsets):
        sub = d[(d.data == dn) & (d.optimizer == "adam")]
        piv = sub.groupby("filter").t_conv.median().reindex(order)
        ax.bar(range(len(order)), piv.values, color=cols)
        ax.set_xticks(range(len(order)), order)
        ax.set_title(f"{dn} (Adam)")
        ax.set_yscale("log")
        ax.grid(alpha=0.3, axis="y", which="both")
    axes[0].set_ylabel("итераций до сходимости (лог, ниже — быстрее)")
    fig.suptitle("Реальные данные: сходимость обучения AR(5) "
                 "(SGD не сходится за 4000 — не показан)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "real_conv.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_semisynthetic():
    d = pd.read_csv(ROOT / "tables/semisynthetic_summary.csv")
    d = d[d.optimizer == "adam"]
    piv = d.pivot_table(index="filter", columns="snr_target",
                        values="holdout_mse", aggfunc="mean")
    order = ["F0", "F1 MA", "F2 Kalman", "F3 Wavelet", "F4 Median"]
    piv = piv.reindex([f for f in order if f in piv.index])
    snrs = sorted(piv.columns)
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    x = np.arange(len(piv.index))
    w = 0.25
    for k, snr in enumerate(snrs):
        ax.bar(x + (k - 1) * w, piv[snr].values, w,
               label=f"SNR {snr:+.0f} дБ")
    ax.set_xticks(x, piv.index, rotation=12, ha="right")
    ax.set_ylabel("holdout MSE восстановления $s$ (ниже — лучше)")
    ax.set_title("Полусинтетика: реальный шум доходностей + известный сигнал")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "semisynthetic.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_microstructure():
    d = pd.read_csv(ROOT / "tables/microstructure_summary.csv")
    d = d[d.regime == "gauss"]
    piv = d.pivot_table(index="filter", columns="gamma",
                        values="rel_err", aggfunc="median")
    order = ["F0 raw", "F1 MA", "F2 Kalman", "F3 Wavelet", "F4 Median",
             "FA online"]
    piv = piv.reindex([f for f in order if f in piv.index])
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for f in piv.index:
        ax.plot(piv.columns, piv.loc[f].values, marker="o", label=f)
    ax.set_xlabel(r"шум/сигнал $\gamma$")
    ax.set_ylabel(r"медиана $|RV-IV|/IV$ (ниже — лучше)")
    ax.set_title("Микроструктура: оценка интегрированной дисперсии")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIG / "microstructure.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_convergence()
    fig_snr_heatmap()
    fig_speed()
    fig_synth_bars()
    fig_real_mse()
    fig_real_conv()
    fig_semisynthetic()
    fig_microstructure()
    fig_intraday()
    print("wrote: conv_curves snr_heatmap speed real_conv synth_bars "
          "real_mse semisynthetic microstructure intraday (.pdf)")
