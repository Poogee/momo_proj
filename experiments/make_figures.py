from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=RuntimeWarning)

NOISE_LABELS = {
    "N1": "N1 Гауссов",
    "N2": "N2 Pink (1/f)",
    "N3": "N3 α-устойчивый",
    "N4": "N4 Смешанный",
}
FILTER_LABELS = {
    "F0": "F0 Без фильтра",
    "F1": "F1 MA",
    "F2": "F2 Калман",
    "F3": "F3 Вейвлет",
    "F4": "F4 Медиана",
    "F5": "F5 CNN",
    "F6": "F6 Адапт.вейв.",
    "F7": "F7 Медиана+вейв.",
}
OPT_LABELS = {"sgd": "SGD", "adam": "Adam", "adamw": "AdamW"}

FILTER_ORDER_BASE = ["F0", "F1", "F2", "F3", "F4"]
FILTER_ORDER = ["F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7"]
NOISE_ORDER = ["N1", "N2", "N3", "N4"]
OPT_ORDER = ["sgd", "adam", "adamw"]


def load_curves(runs_dir: Path, task: str) -> dict:
    out: dict = {}
    base = runs_dir / task
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


def plot_convergence_grid(curves: dict, out_path: Path, title_prefix: str,
                          opt: str = "adam") -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, noise in zip(axes, NOISE_ORDER):
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
        ax.set_title(NOISE_LABELS[noise])
        ax.set_xlabel("итерация")
        ax.set_ylabel(r"$\|\nabla f\|^2$")
        ax.grid(alpha=0.3, which="both")
    axes[0].legend(loc="upper right", fontsize=9, frameon=False)
    fig.suptitle(f"{title_prefix} ({OPT_LABELS[opt]}): сходимость по фильтрам")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_t_eps_heatmap(summary: pd.DataFrame, out_path: Path, task: str) -> None:
    sub = summary[summary["task"] == task].copy()
    sub.loc[sub["t_eps"] == -1, "t_eps"] = sub["steps"].max()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, opt in zip(axes, OPT_ORDER):
        view = sub[sub["optimizer"] == opt]
        pivot = (
            view.groupby(["filter", "noise"])["t_eps"]
            .median()
            .unstack("noise")
            .reindex(index=FILTER_ORDER, columns=NOISE_ORDER)
        )
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="viridis_r", ax=ax,
                    cbar=ax is axes[-1])
        ax.set_title(f"{OPT_LABELS[opt]}")
        ax.set_xlabel("шум")
        if ax is axes[0]:
            ax.set_ylabel("фильтр")
        else:
            ax.set_ylabel("")
    fig.suptitle(f"Медианное T(ε) — задача: {task}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_filter_diagnostics(diag: pd.DataFrame, out_path: Path) -> None:
    summary = diag.groupby(["noise", "filter"]).agg(
        delta_snr=("delta_snr_db", "mean")
    ).reset_index()
    pivot = summary.pivot(index="filter", columns="noise", values="delta_snr").reindex(
        index=FILTER_ORDER, columns=NOISE_ORDER
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=ax)
    ax.set_title(r"Прирост SNR (дБ) после фильтрации")
    ax.set_xlabel("шум")
    ax.set_ylabel("фильтр")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_hurst_alpha_distortion(diag: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    g = diag.groupby(["noise", "filter"]).agg(
        h_in=("hurst_in", "mean"),
        h_out=("hurst_out", "mean"),
        a_in=("alpha_in", "mean"),
        a_out=("alpha_out", "mean"),
    ).reset_index()
    for metric_in, metric_out, label, fname in [
        ("h_in", "h_out", "Хёрст H", "hurst_distortion.pdf"),
        ("a_in", "a_out", "Хвостовой индекс α", "alpha_distortion.pdf"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5))
        for noise in NOISE_ORDER:
            sub = g[g["noise"] == noise]
            for _, row in sub.iterrows():
                ax.plot([0, 1], [row[metric_in], row[metric_out]],
                        marker="o", label=f"{noise} → {row['filter']}",
                        alpha=0.7, lw=1)
        ax.set_xticks([0, 1], ["до фильтрации", "после фильтрации"])
        ax.set_ylabel(label)
        ax.set_title(f"Искажение {label} фильтрами")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)


def make_recommendation_matrix(summary: pd.DataFrame, diag: pd.DataFrame,
                               out_path: Path) -> pd.DataFrame:
    rows = []
    for task in summary["task"].unique():
        for noise in NOISE_ORDER:
            sub = summary[(summary["task"] == task) & (summary["noise"] == noise)].copy()
            sub.loc[sub["t_eps"] == -1, "t_eps"] = sub["steps"].max()
            agg = sub.groupby(["filter", "optimizer"])["t_eps"].median().reset_index()
            best = agg.loc[agg["t_eps"].idxmin()]
            rows.append(dict(
                task=task, noise=noise,
                best_filter=best["filter"],
                best_optimizer=best["optimizer"],
                best_t_eps=int(best["t_eps"]),
            ))
    diag_summary = diag.groupby(["noise", "filter"])["delta_snr_db"].mean().reset_index()
    best_filter_by_snr = diag_summary.loc[diag_summary.groupby("noise")["delta_snr_db"].idxmax()]
    snr_recs = {row["noise"]: row["filter"] for _, row in best_filter_by_snr.iterrows()}
    for r in rows:
        r["best_filter_by_snr"] = snr_recs.get(r["noise"], "?")
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/synthetic"))
    parser.add_argument("--summary-csv", type=Path, default=Path("tables/synthetic_summary.csv"))
    parser.add_argument("--diag-csv", type=Path, default=Path("tables/filter_diagnostics.csv"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--rec-csv", type=Path, default=Path("tables/recommendations.csv"))
    args = parser.parse_args()

    sns.set_theme(context="paper", style="whitegrid", font_scale=0.95)

    diag = pd.read_csv(args.diag_csv)
    summary = pd.read_csv(args.summary_csv) if args.summary_csv.exists() else None

    plot_filter_diagnostics(diag, args.figures_dir / "filter_snr_heatmap.pdf")
    plot_hurst_alpha_distortion(diag, args.figures_dir)

    if summary is not None and not summary.empty:
        for task in summary["task"].unique():
            curves = load_curves(args.runs_dir, task)
            for opt in OPT_ORDER:
                plot_convergence_grid(
                    curves, args.figures_dir / f"convergence_{task}_{opt}.pdf",
                    title_prefix=f"Задача: {task}", opt=opt,
                )
            plot_t_eps_heatmap(summary, args.figures_dir / f"t_eps_heatmap_{task}.pdf", task)
        rec = make_recommendation_matrix(summary, diag, args.rec_csv)
        print(rec.to_string(index=False))
    else:
        print("Synthetic summary not found; skipping convergence/heatmap/recommendation plots.")


if __name__ == "__main__":
    main()
