"""Convert tables/new_datasets_summary.csv into a compact LaTeX table for
\\input{...} inside paper_ru.tex (Block D, June 2026 lean-filter sweep)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SUMM = Path("tables/new_datasets_summary.csv")
OUT = Path("tables/new_datasets_block_d.tex")

# Only the datasets that carry a reportable result are tabulated. The
# heavy-tailed Electricity and Traffic series were also run but F0 was
# optimal for them (no filter helped, no rescue), so they add nothing
# beyond the NOAA/ETTm2 negative controls and are omitted -- see the
# prose note in Block D and DECISIONS.md.
DOMAIN_PRETTY = {
    "binance_crypto_1h": "Binance BTC/ETH 1ч",
    "cmapss_turbofan":   "CMAPSS (турбина)",
    "atm_withdrawals":   "ATM (снятия)",
    "noaa_weather":      "NOAA Weather",
    "sensor_ettm2":      "ETTm2 (сенсор)",
}
DOMAIN_ORDER = list(DOMAIN_PRETTY)
KEEP = set(DOMAIN_PRETTY)


def _row_for(sub: pd.DataFrame) -> tuple[str, str, str, str, float, float] | None:
    """Pick the best filter for one domain at fixed (model,optimizer):
    converged for >=75% of seeds and holdout MSE no worse than 10% over
    F0; among those, smallest t_conv. Returns (alpha,H,best,speedup,bh,f0h)."""
    f0 = sub[sub["filter"] == "F0"]
    if f0.empty:
        return None
    f0t = float(f0["t_conv_med"].iloc[0])
    f0h = float(f0["holdout_med"].iloc[0])
    ah = float(f0["alpha_hat_med"].iloc[0])
    hh = float(f0["hurst_hat_med"].iloc[0])
    ok = sub[(sub["conv_frac"] >= 0.75)
             & (sub["holdout_med"] <= 1.10 * f0h)
             & (sub["filter"] != "F0")]
    if ok.empty:
        return (f"{ah:.2f}", f"{hh:.2f}", "F0", "$1.00\\times$", f0h, f0h)
    best = ok.loc[ok["t_conv_med"].idxmin()]
    sp = f0t / max(float(best["t_conv_med"]), 1.0)
    bh = float(best["holdout_med"])
    sp_txt = (f"$\\mathbf{{{sp:.1f}\\times}}$" if sp >= 1.5
              else f"${sp:.2f}\\times$")
    return (f"{ah:.2f}", f"{hh:.2f}", str(best["filter"]), sp_txt, bh, f0h)


def main():
    if not SUMM.exists():
        OUT.write_text(
            "% new_datasets_summary.csv not built yet -- placeholder\n"
            "\\begin{table}[t]\\caption{Блок D (заполняется).}\n"
            "\\label{tab:blockD}\\centering\\begin{tabular}{lccccc}\n"
            "\\toprule домен & $\\hat\\alpha$ & $\\hat H$ & лучш. & ускор. & "
            "holdout\\\\\\midrule\n\\multicolumn{6}{c}{[пересоберите]}\\\\\n"
            "\\bottomrule\\end{tabular}\\end{table}\n", encoding="utf-8")
        print(f"wrote placeholder {OUT}")
        return

    df = pd.read_csv(SUMM)
    g = df[(df["optimizer"] == "adam") & (df["model"] == "regression")
           & (df["domain"].isin(KEEP))].copy()
    present = [d for d in DOMAIN_ORDER if d in set(g["domain"])]

    rows = []
    for dn in present:
        sub = g[g["domain"] == dn]
        r = _row_for(sub)
        if r is None:
            continue
        ah, hh, best, sp_txt, bh, f0h = r
        pretty = DOMAIN_PRETTY.get(dn, dn)
        rows.append(f"{pretty} & ${ah}$ & ${hh}$ & ${best}$ & {sp_txt} & "
                    f"{bh:.3g}/{f0h:.3g}\\\\")

    OUT.write_text(
        "\\begin{table}[t]\n"
        "\\caption{Блок~D (июнь 2026). Новые датасеты, тощий набор фильтров"
        " $\\{F_0,F_4,F_{\\mathrm{A}},F_{10},F_{11}\\}$, Adam, регрессия"
        " AR(5), по 4 сида. Лучший фильтр --- доля сходимости $\\ge0.75$ и"
        " holdout MSE не более чем на 10\\% хуже $F_0$; среди них минимум"
        " итераций до $100\\times$-снижения градиента. \\emph{ускор.} ---"
        " отношение медиан числа итераций; \\emph{holdout} --- лучший/$F_0$.}\n"
        "\\label{tab:blockD}\n"
        "\\centering\\begin{tabular}{lccccc}\n\\toprule\n"
        "домен & $\\hat\\alpha$ & $\\hat H$ & лучш. & ускор. & holdout\\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n"
        "\\bottomrule\n\\end{tabular}\\end{table}\n", encoding="utf-8")
    print(f"wrote {OUT}\n" + "\n".join(rows))


if __name__ == "__main__":
    sys.exit(main())
