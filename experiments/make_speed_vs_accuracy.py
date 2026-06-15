"""Representative real-data figure: on financial 15-min returns the filter
makes Adam converge far faster but does NOT improve out-of-sample accuracy.
Two panels for F0 (no filter) vs F2 (Kalman): (a) iterations to converge,
(b) holdout MSE — the second pair is practically equal."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

raw = pd.read_csv("tables/applied_convergence.csv")
g = raw[(raw.domain == "financial_15m") & (raw.optimizer == "adam")]

filt = ["F0", "F2"]
lbl = {"F0": "без фильтра (F0)", "F2": "фильтр Калмана (F2)"}
col = {"F0": "#C0504D", "F2": "#4472C4"}

t_med = {f: float(g[g["filter"] == f]["t_conv"].median()) for f in filt}
mse = {f: g[g["filter"] == f]["holdout_mse"].values for f in filt}
mse_med = {f: float(np.median(mse[f])) for f in filt}

sns.set_theme(context="paper", style="whitegrid", font_scale=1.0)
fig, (axa, axb) = plt.subplots(1, 2, figsize=(9, 3.6))

x = np.arange(len(filt))
axa.bar(x, [t_med[f] for f in filt], 0.55, color=[col[f] for f in filt])
axa.set_yscale("log")
axa.set_xticks(x); axa.set_xticklabels([lbl[f] for f in filt], fontsize=9)
axa.set_ylabel("итераций до сходимости (медиана)")
axa.set_title("(а) скорость: фильтр сходится в ~50 раз быстрее")
for xi, f in zip(x, filt):
    axa.text(xi, t_med[f] * 1.1, f"{t_med[f]:.0f}", ha="center", fontsize=10)

axb.boxplot([mse[f] for f in filt], positions=x, widths=0.55, showfliers=False,
            patch_artist=True,
            boxprops=dict(facecolor="#DDE5F0"), medianprops=dict(color="black", lw=1.5))
axb.set_xticks(x); axb.set_xticklabels([lbl[f] for f in filt], fontsize=9)
axb.set_ylabel("holdout MSE")
axb.set_title("(б) точность: holdout-MSE практически одинаков")
for xi, f in zip(x, filt):
    axb.text(xi, mse_med[f], f"  {mse_med[f]:.3f}", ha="left", va="center", fontsize=10)

fig.suptitle("Финансовые 15-мин: фильтр ускоряет обучение, но не улучшает прогноз",
             fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = Path("figures/real_speed_vs_accuracy.pdf")
fig.savefig(out, dpi=150)
print(f"wrote {out}")
