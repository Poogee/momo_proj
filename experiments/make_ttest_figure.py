"""Forest plot of the paired-test effect sizes backing the filtering claims.

  figures/filter_ttests_forest.pdf
    (a) floor-ratio (F0/F4) with 95% bootstrap CI per task, heavy-tail N3
        and the real-calibrated N3cal — log x-axis, x=1 = "no effect" line.
    (b) the statistical-vs-practical contrast: floor ratio on N3 (~100x)
        vs N4 (~1x) on a log axis, so the reader sees the effect collapse
        exactly where H3 predicts no simple filter works.

Inputs: tables/filter_ttests.csv, tables/filter_ttests_calibrated.csv
(written by experiments/run_filter_ttests.py). Panels skip gracefully if a
file is missing.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TABLES = Path("tables")
OUT = Path("figures/filter_ttests_forest.pdf")
MODEL_LBL = {"quadratic": "квадратичная", "logistic": "логистическая", "ar": "авторегрессия"}


def _rows(df, noise, label):
    out = []
    for _, r in df[(df.noise == noise) & (df["filter"] == "F4")].iterrows():
        out.append((f"{MODEL_LBL.get(r.model, r.model)} ({label})",
                    r.floor_ratio, r.boot_ratio_lo, r.boot_ratio_hi))
    return out


def main() -> None:
    full = pd.read_csv(TABLES / "filter_ttests.csv")
    n3 = full[(full.noise == "N3") & (full.optimizer == "sgd") & (full.metric == "floor")]
    n4 = full[(full.noise == "N4") & (full.optimizer == "sgd") & (full.metric == "floor")]
    try:
        cal = pd.read_csv(TABLES / "filter_ttests_calibrated.csv")
    except FileNotFoundError:
        cal = pd.DataFrame(columns=full.columns)

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12, 4.4))

    # (a) forest plot: N3 synthetic + N3cal calibrated
    items = _rows(n3, "N3", "синт. α=1.2")
    items += _rows(cal, "N3cal", "калибр. α=1.21")
    items = items[::-1]
    ys = np.arange(len(items))
    for y, (lbl, ratio, lo, hi) in zip(ys, items):
        axa.plot([lo, hi], [y, y], color="#2b6cb0", lw=2, zorder=2)
        axa.plot(ratio, y, "o", color="#2b6cb0", ms=7, zorder=3)
    axa.axvline(1.0, color="crimson", ls="--", lw=1.2, label="нет эффекта (×1)")
    axa.set_yticks(ys)
    axa.set_yticklabels([it[0] for it in items], fontsize=8.5)
    axa.set_xscale("log")
    axa.set_xlabel(r"снижение пола $\|\nabla f\|^2$, F0/F4 (лог-шкала)")
    axa.set_title("(a) Тяжёлые хвосты: эффект и 95% бутстрэп-ДИ")
    axa.legend(loc="upper left", fontsize=8)
    axa.margins(y=0.08)

    # (b) statistical vs practical significance: N3 vs N4 floor ratio per task
    models = ["quadratic", "logistic", "ar"]
    x = np.arange(len(models))
    n3r = [float(n3[n3.model == m].floor_ratio.iloc[0]) for m in models]
    n4r = [float(n4[n4.model == m].floor_ratio.iloc[0]) for m in models]
    w = 0.36
    axb.bar(x - w / 2, n3r, w, color="#2b6cb0", label="N3 (тяжёлый хвост)")
    axb.bar(x + w / 2, n4r, w, color="#cbd5e0", label="N4 (смешанный)")
    axb.axhline(1.0, color="crimson", ls="--", lw=1.2)
    axb.set_yscale("log")
    axb.set_xticks(x)
    axb.set_xticklabels([MODEL_LBL[m] for m in models], fontsize=8.5)
    axb.set_ylabel("снижение пола F0/F4 (лог-шкала)")
    axb.set_title("(b) Статистич. vs практич.: 100× против 1×")
    axb.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
