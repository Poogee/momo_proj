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

FILTER_ORDER = ["F0", "F1", "F2", "F3", "F4"]
NOISE_ORDER = ["N1", "N2", "N3", "N4"]
OPT_ORDER = ["sgd", "adam", "adamw"]

NOISE_LABELS = {
    "N1": "N1 Гауссов", "N2": "N2 Pink (1/f)",
    "N3": "N3 α-устойчивый", "N4": "N4 Смешанный",
}
FILTER_LABELS = {
    "F0": "F0", "F1": "F1 MA", "F2": "F2 Калман",
    "F3": "F3 Вейвлет", "F4": "F4 Медиана",
}
MODE_LABELS = {"buffer": "Буферный режим", "data": "Режим данных"}


def load_curves(runs_dir: Path, mode: str, task: str) -> dict:
    out = {}
    base = runs_dir / mode / task
    if not base.exists():
        return out
    for cell in base.iterdir():
        if not cell.is_dir():
            continue
        seeds = []
        for f in sorted(cell.glob("seed*.npz")):
            arr = np.load(f)
            seeds.append(arr["grad_norm_sq"])
        if seeds:
            out[cell.name] = np.stack(seeds)
    return out


def plot_mode_comparison(curves_b: dict, curves_d: dict, out_path: Path,
                         task: str, opt: str = "adam") -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey=True)
    for col, noise in enumerate(NOISE_ORDER):
        for row, (mode_name, curves) in enumerate(zip(["buffer", "data"], [curves_b, curves_d])):
            ax = axes[row, col]
            for filt in FILTER_ORDER:
                key = f"{filt}_{noise}_{opt}"
                if key not in curves:
                    continue
                arr = curves[key]
                med = np.median(arr, axis=0)
                p10 = np.quantile(arr, 0.10, axis=0)
                p90 = np.quantile(arr, 0.90, axis=0)
                x = np.arange(med.size)
                ax.plot(x, med, label=FILTER_LABELS[filt], lw=1.5)
                ax.fill_between(x, p10, p90, alpha=0.15)
            ax.set_yscale("log")
            if row == 0:
                ax.set_title(NOISE_LABELS[noise])
            if col == 0:
                ax.set_ylabel(f"{MODE_LABELS[mode_name]}\n" + r"$\|\nabla f\|^2$")
            if row == 1:
                ax.set_xlabel("итерация")
            ax.grid(alpha=0.3, which="both")
    axes[0, 0].legend(loc="lower left", fontsize=8, frameon=False)
    fig.suptitle(f"Сравнение режимов фильтрации ({task}, {opt}): буфер vs данные")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_final_grad_heatmap(summary: pd.DataFrame, out_path: Path, task: str) -> None:
    sub = summary[summary["task"] == task].copy()
    sub["log_final"] = np.log10(np.maximum(sub["final_grad_norm_sq"], 1e-12))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, mode in zip(axes, ["buffer", "data"]):
        view = sub[sub["mode"] == mode]
        pivot = (view.groupby(["filter", "noise"])["log_final"].median()
                 .unstack("noise").reindex(index=FILTER_ORDER, columns=NOISE_ORDER))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis_r", ax=ax,
                    cbar_kws={"label": r"$\log_{10}$ финальный $\|g\|^2$"})
        ax.set_title(MODE_LABELS[mode])
        ax.set_xlabel("шум")
        if ax is axes[0]:
            ax.set_ylabel("фильтр")
        else:
            ax.set_ylabel("")
    fig.suptitle(f"Финальная норма градиента (медиана по 5 запускам, Adam, {task})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_speedup(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    g = summary.groupby(["mode", "filter"])["elapsed_s"].mean().unstack("mode")
    g["speedup"] = g["buffer"] / g["data"]
    g.reindex(index=FILTER_ORDER)["speedup"].plot.bar(ax=ax, color="C2", rot=0)
    ax.set_ylabel("ускорение (буфер / данные)")
    ax.set_title("Время выполнения: буфер vs режим данных")
    for i, v in enumerate(g.reindex(index=FILTER_ORDER)["speedup"]):
        ax.text(i, v + 0.5, f"{v:.1f}×", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/mode_compare"))
    parser.add_argument("--summary-csv", type=Path, default=Path("tables/mode_compare_summary.csv"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    sns.set_theme(context="paper", style="whitegrid", font_scale=0.95)
    summary = pd.read_csv(args.summary_csv)
    args.figures_dir.mkdir(parents=True, exist_ok=True)

    for task in summary["task"].unique():
        cb = load_curves(args.runs_dir, "buffer", task)
        cd = load_curves(args.runs_dir, "data", task)
        plot_mode_comparison(cb, cd, args.figures_dir / f"mode_compare_{task}_adam.pdf", task, "adam")
        plot_final_grad_heatmap(summary, args.figures_dir / f"mode_final_grad_{task}.pdf", task)
    plot_speedup(summary, args.figures_dir / "mode_speedup.pdf")
    print(f"Wrote mode-comparison figures to {args.figures_dir}")


if __name__ == "__main__":
    main()
