"""Comprehensive all-filters figure for the June 2026 study.

Two heatmaps over the full set F0..F7 x all domains (Adam, regression):
  (a) convergence speedup vs F0  (log-coloured, annotated)
  (b) gradient-floor reduction vs F0
so every filter and every domain is visible with the correct, contiguous
numbering. F3 is the non-causal oracle (marked *).

Output: figures/new_datasets_heatmap.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

SUMM = Path("tables/new_datasets_summary.csv")
OUT = Path("figures/new_datasets_heatmap.pdf")

FILTERS = ["F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7"]
FILT_LBL = {"F0": "$F_0$", "F1": "$F_1$", "F2": "$F_2$", "F3": "$F_3^{*}$",
            "F4": "$F_4$", "F5": "$F_5$", "F6": "$F_6$", "F7": "$F_7$"}
DOMAIN_PRETTY = {
    "financial_crypto_1h": "крипто 1ч",
    "financial_15m": "фин 15м", "financial_5m": "фин 5м",
    "financial_daily": "фин дн.",
    "financial_vol_crypto": "крипто |r|", "financial_vol_daily": "акции |r|",
    "macro_fred": "FRED",
    "sensor_etth1": "ETTh1", "sensor_etth2": "ETTh2",
    "sensor_ettm1": "ETTm1", "sensor_ettm2": "ETTm2",
    "electricity_load": "Electr.", "traffic_pems": "Traffic",
    "noaa_weather": "NOAA", "cmapss_turbofan": "CMAPSS",
    "atm_withdrawals": "ATM", "scientific_sunspots": "Sunspots",
}
DOMAIN_ORDER = list(DOMAIN_PRETTY)


def _pivot(df, value):
    g = df[(df["optimizer"] == "adam") & (df["model"] == "regression")]
    doms = [d for d in DOMAIN_ORDER if d in set(g["domain"])]
    M = np.full((len(FILTERS), len(doms)), np.nan)
    for i, fk in enumerate(FILTERS):
        for j, dn in enumerate(doms):
            r = g[(g["domain"] == dn) & (g["filter"] == fk)]
            if not r.empty:
                M[i, j] = float(r[value].iloc[0])
    return M, doms


def _panel(ax, M, doms, title, cbar_label):
    L = np.log10(np.clip(M, 1e-3, 1e3))
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=2.0)
    im = ax.imshow(L, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_xticks(range(len(doms)))
    ax.set_xticklabels([DOMAIN_PRETTY.get(d, d) for d in doms],
                       rotation=55, ha="right", fontsize=7)
    ax.set_yticks(range(len(FILTERS)))
    ax.set_yticklabels([FILT_LBL[f] for f in FILTERS], fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isfinite(v):
                continue
            txt = f"{v:.0f}" if v >= 9.5 else (f"{v:.1f}" if v >= 1.05
                                               else f"{v:.2f}")
            ax.text(j, i, txt, ha="center", va="center", fontsize=6,
                    color="white" if (L[i, j] > 1.0 or L[i, j] < -0.55)
                    else "black")
    ax.set_title(title, fontsize=10)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label(cbar_label, fontsize=8)
    cb.ax.tick_params(labelsize=7)


def main():
    if not SUMM.exists():
        print("no summary; skip")
        return
    df = pd.read_csv(SUMM)
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.6))
    M1, doms = _pivot(df, "speedup_vs_F0")
    _panel(axes[0], M1, doms,
           r"(а) Ускорение сходимости Adam относительно $F_0$ "
           r"(во сколько раз; $>1$ — быстрее)",
           r"$\log_{10}$ ускор.")
    M2, _ = _pivot(df, "floor_ratio_vs_F0")
    _panel(axes[1], M2, doms,
           r"(б) Снижение шумового пола $\|\nabla f\|^2$ относительно $F_0$ "
           r"($>1$ — ниже пол)",
           r"$\log_{10}$ сниж.")
    fig.text(0.012, 0.5, r"фильтр ($F_3^{*}$ — непричинный, оракул)",
             rotation=90, va="center", fontsize=8)
    fig.tight_layout(rect=(0.02, 0, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}  ({len(FILTERS)} filters x {len(doms)} domains)")


if __name__ == "__main__":
    sys.exit(main())
