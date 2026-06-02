"""Convert tables/extended_factorial_summary.csv into a compact LaTeX
table suitable for \\input{...} inside paper_ru.tex (Block D)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SUMM = Path("tables/extended_factorial_summary.csv")
CRIT = Path("tables/extended_factorial_criteria.csv")
OUT = Path("tables/extended_factorial_block_d.tex")


DOMAIN_PRETTY = {
    "financial_daily":     "фин.\\ дневные",
    "financial_15m":       "фин.\\ 15-мин",
    "financial_5m":        "фин.\\ 5-мин",
    "macro_fred":          "макро (FRED)",
    "sensor_etth1":        "сенсор (ETTh1)",
    "sensor_etth2":        "сенсор (ETTh2)",
    "sensor_ettm1":        "сенсор (ETTm1)",
    "scientific_sunspots": "scientific (Sunspots)",
}


def main():
    if not SUMM.exists():
        # placeholder — keep the paper compilable while the run is going
        OUT.write_text(
            "% extended_factorial_criteria.csv not built yet — placeholder\n"
            "\\begin{table}[t]\n"
            "\\caption{Блок D (заполняется после расширенного факториала).}\n"
            "\\label{tab:blockD}\n"
            "\\centering\\begin{tabular}{lccccc}\n"
            "\\toprule\n"
            "домен & $\\hat\\alpha$ & $\\hat H$ & лучш.\\ фильтр &"
            " ускор.\\ vs $F_0$ & holdout\\\\\n"
            "\\midrule\n"
            "\\multicolumn{6}{c}{[пересоберите после запуска"
            " run\\_extended\\_factorial.py]}\\\\\n"
            "\\bottomrule\\end{tabular}\\end{table}\n",
            encoding="utf-8")
        print(f"wrote placeholder {OUT}")
        return

    df = pd.read_csv(SUMM)
    # focus on Adam regression rows for the main table
    g = df[(df["optimizer"] == "adam") & (df["model"] == "regression")].copy()
    if g.empty:
        g = df.copy()

    rows = []
    for dn, sub in g.groupby("domain"):
        f0 = sub[sub["filter"] == "F0"]
        if f0.empty:
            continue
        f0t = float(f0["t_conv_med"].iloc[0])
        f0h = float(f0["holdout_med"].iloc[0])
        ah = float(f0["alpha_hat_med"].iloc[0])
        hh = float(f0["hurst_hat_med"].iloc[0])

        # candidates that converged for >=75% of seeds, with holdout
        # at most 10% worse than F0 (regression MSE); pick the one
        # with smallest t_conv_med
        ok = sub[(sub["conv_frac"] >= 0.75)
                 & (sub["holdout_med"] <= 1.10 * f0h)
                 & (sub["filter"] != "F0")]
        if ok.empty:
            best_name = "F0"
            sp = 1.0
            bh = f0h
        else:
            best = ok.loc[ok["t_conv_med"].idxmin()]
            best_name = str(best["filter"])
            sp = f0t / max(float(best["t_conv_med"]), 1.0)
            bh = float(best["holdout_med"])

        pretty = DOMAIN_PRETTY.get(str(dn), str(dn))
        speedup_txt = (f"$\\mathbf{{{sp:.1f}\\times}}$" if sp >= 1.5
                       else f"${sp:.2f}\\times$")
        holdout_txt = f"{bh:.3g}/{f0h:.3g}"
        rows.append(f"{pretty} & ${ah:.2f}$ & ${hh:.2f}$ & "
                    f"${best_name}$ & {speedup_txt} & {holdout_txt}\\\\")

    OUT.write_text(
        "\\begin{table}[t]\n"
        "\\caption{Блок~D. Расширенный факториал, Adam, регрессия AR(5),"
        " по 4 сида. Лучший фильтр --- такой, у которого доля сходимости"
        " $\\ge 0.75$ и holdout MSE не более чем на 10\\% хуже $F_0$;"
        " среди них минимальное число итераций до $100\\times$-снижения"
        " градиента. \\emph{ускор.\\ vs $F_0$} --- отношение медиан"
        " числа итераций; \\emph{holdout} --- лучший фильтр / $F_0$.}\n"
        "\\label{tab:blockD}\n"
        "\\centering\\begin{tabular}{lccccc}\n"
        "\\toprule\n"
        "домен & $\\hat\\alpha$ & $\\hat H$ & лучш. & ускор. & holdout\\\\\n"
        "\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}\\end{table}\n",
        encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
