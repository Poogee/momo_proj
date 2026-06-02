"""Build the LaTeX tables for the comprehensive June 2026 study:
  tables/new_datasets_block_d.tex   per-domain best causal filter + speedup
  tables/new_datasets_rules.tex     (alpha,H) bucket -> recommended filter
Both are \\input{} into paper_ru.tex."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SUMM = Path("tables/new_datasets_summary.csv")
RULES = Path("tables/new_datasets_rules.csv")
OUT_D = Path("tables/new_datasets_block_d.tex")
OUT_R = Path("tables/new_datasets_rules.tex")

DOMAIN_PRETTY = {
    "financial_crypto_1h": "Binance BTC/ETH 1ч",
    "financial_daily":     "фин.\\ дневные",
    "financial_15m":       "фин.\\ 15-мин",
    "financial_5m":        "фин.\\ 5-мин",
    "financial_vol_crypto": "крипто $|r|$ (волат.)",
    "financial_vol_daily": "акции $|r|$ (волат.)",
    "macro_fred":          "макро (FRED)",
    "sensor_etth1":        "ETTh1",
    "sensor_etth2":        "ETTh2",
    "sensor_ettm1":        "ETTm1",
    "sensor_ettm2":        "ETTm2",
    "electricity_load":    "Electricity Load",
    "traffic_pems":        "Traffic PEMS",
    "noaa_weather":        "NOAA Weather",
    "cmapss_turbofan":     "CMAPSS (турбина)",
    "atm_withdrawals":     "ATM (снятия)",
    "scientific_sunspots": "Sunspots",
}
BUCKET_PRETTY = {
    "light_short": "$\\hat\\alpha{\\approx}2,\\ \\hat H{\\approx}0.5$",
    "light_long":  "$\\hat\\alpha{\\approx}2,\\ \\hat H{>}0.6$",
    "heavy_short": "$\\hat\\alpha{<}1.9,\\ \\hat H{\\approx}0.5$",
    "heavy_long":  "$\\hat\\alpha{<}1.9,\\ \\hat H{>}0.6$",
}
BUCKET_ORDER = ["light_short", "light_long", "heavy_short", "heavy_long"]


def _flabel(f: str) -> str:
    """'F2' -> 'F_2' for LaTeX subscript labels."""
    return f"F_{f[1:]}" if f.startswith("F") and f[1:].isdigit() else f


def _best_causal(sub: pd.DataFrame):
    f0 = sub[sub["filter"] == "F0"]
    if f0.empty:
        return None
    f0t = float(f0["t_conv_med"].iloc[0]); f0h = float(f0["holdout_med"].iloc[0])
    ah = float(f0["alpha_hat_med"].iloc[0]); hh = float(f0["hurst_hat_med"].iloc[0])
    ok = sub[(sub.get("causal", 1) == 1) & (sub["filter"] != "F0")
             & (sub["conv_frac"] >= 0.75) & (sub["holdout_med"] <= 1.10 * f0h)]
    if ok.empty:
        return ah, hh, "F0", "$1.00\\times$", f0h, f0h
    b = ok.loc[ok["t_conv_med"].idxmin()]
    sp = f0t / max(float(b["t_conv_med"]), 1.0)
    txt = f"$\\mathbf{{{sp:.1f}\\times}}$" if sp >= 1.5 else f"${sp:.2f}\\times$"
    return ah, hh, str(b["filter"]), txt, float(b["holdout_med"]), f0h


def block_d():
    df = pd.read_csv(SUMM)
    g = df[(df["optimizer"] == "adam") & (df["model"] == "regression")].copy()
    order = [d for d in DOMAIN_PRETTY if d in set(g["domain"])]
    order += [d for d in g["domain"].unique() if d not in DOMAIN_PRETTY]
    rows = []
    for dn in order:
        r = _best_causal(g[g["domain"] == dn])
        if r is None:
            continue
        ah, hh, best, sp_txt, bh, f0h = r
        rows.append(f"{DOMAIN_PRETTY.get(dn, dn)} & ${ah:.2f}$ & ${hh:.2f}$ & "
                    f"${_flabel(best)}$ & {sp_txt} & {bh:.3g}/{f0h:.3g}\\\\")
    OUT_D.write_text(
        "\\begin{table}[t]\n\\caption{Блок~D (июнь 2026). Широкая корзина,"
        " набор $F_0$--$F_7$, Adam, регрессия AR(5), 4 сида. Лучший"
        " \\emph{причинный} фильтр: доля сходимости $\\ge0.75$ и holdout MSE"
        " не более чем на 10\\% хуже $F_0$; среди них минимум итераций до"
        " $100\\times$-снижения градиента (оракульный $F_3$ в выбор не"
        " входит). \\emph{ускор.} --- отношение медиан числа итераций.}\n"
        "\\label{tab:blockD}\n\\centering\\footnotesize"
        "\\begin{tabular}{lccccc}\n\\toprule\n"
        "домен & $\\hat\\alpha$ & $\\hat H$ & лучш. & ускор. & holdout\\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n"
        "\\end{tabular}\\end{table}\n", encoding="utf-8")
    print(f"wrote {OUT_D} ({len(rows)} domains)")


def _bucket(ah, hh):
    return ("light" if ah >= 1.9 else "heavy") + "_" + ("long" if hh > 0.6 else "short")


def derive_rules() -> pd.DataFrame:
    """Recompute the (alpha,H) -> filter rule from the per-cell summary,
    reporting where filtering ACTUALLY helps (speedup>=1.5) rather than a
    raw mode over all cells (most of which never need a filter)."""
    s = pd.read_csv(SUMM)
    recs = []
    for (dn, mdl, opt), sub in s.groupby(["domain", "model", "optimizer"]):
        f0 = sub[sub["filter"] == "F0"]
        if f0.empty:
            continue
        f0t = float(f0["t_conv_med"].iloc[0]); f0h = float(f0["holdout_med"].iloc[0])
        ah = float(f0["alpha_hat_med"].iloc[0]); hh = float(f0["hurst_hat_med"].iloc[0])
        ok = sub[(sub.get("causal", 1) == 1) & (sub["filter"] != "F0")
                 & (sub["conv_frac"] >= 0.75) & (sub["holdout_med"] <= 1.10 * f0h)]
        if ok.empty:
            best, sp = "F0", 1.0
        else:
            b = ok.loc[ok["t_conv_med"].idxmin()]
            best, sp = str(b["filter"]), f0t / max(float(b["t_conv_med"]), 1.0)
        recs.append(dict(domain=dn, model=mdl, optimizer=opt,
                         bucket=_bucket(ah, hh), best=best, speedup=sp))
    r = pd.DataFrame(recs)
    rows = []
    for bk in BUCKET_ORDER:
        sub = r[r["bucket"] == bk]
        if sub.empty:
            continue
        helped = sub[sub["speedup"] >= 1.5]
        if len(helped):
            filt = helped["best"].value_counts().index[0]
            rows.append(dict(bucket=bk, n=len(sub), helped=len(helped),
                             filter=filt, share=len(helped) / len(sub),
                             med_speedup=float(helped["speedup"].median()),
                             max_speedup=float(helped["speedup"].max())))
        else:
            rows.append(dict(bucket=bk, n=len(sub), helped=0, filter="F0",
                             share=0.0, med_speedup=1.0, max_speedup=1.0))
    return pd.DataFrame(rows)


def rules():
    rl = derive_rules()
    rl.to_csv(RULES, index=False)  # overwrite crude version with the refined one
    by = rl.set_index("bucket")
    rows = []
    for bk in BUCKET_ORDER:
        if bk not in by.index:
            continue
        r = by.loc[bk]
        if r["helped"] == 0:
            rows.append(f"{BUCKET_PRETTY[bk]} & $F_0$ & --- & $1.0\\times$ & "
                        f"$1.0\\times$\\\\")
        else:
            sp = f"$\\mathbf{{{r['max_speedup']:.0f}\\times}}$" \
                if r['max_speedup'] >= 10 else f"${r['max_speedup']:.1f}\\times$"
            rows.append(f"{BUCKET_PRETTY[bk]} & ${_flabel(r['filter'])}$ & "
                        f"{int(r['helped'])}/{int(r['n'])} & "
                        f"${r['med_speedup']:.1f}\\times$ & {sp}\\\\")
    OUT_R.write_text(
        "\\begin{table}[t]\n\\caption{Эмпирическое правило, выведенное из"
        " блока~D. Для каждого сектора $(\\hat\\alpha,\\hat H)$: фильтр, дающий"
        " ускорение там, где фильтрация \\emph{вообще} помогает"
        " ($\\ge1.5\\times$), доля таких ячеек (домен$\\times$модель$\\times$опт.),"
        " медианное и максимальное ускорение среди них. Где доля нулевая ---"
        " рекомендуется $F_0$ (не фильтровать). $F_2$~--- Калман, $F_5$~---"
        " каскад медиана$\\to$Калман, $F_6$~--- адаптивный каскад.}\n"
        "\\label{tab:rules}\n\\centering\\begin{tabular}{lcccc}\n\\toprule\n"
        "сектор $(\\hat\\alpha,\\hat H)$ & фильтр & помог & медиан. & макс.\\\\\n"
        "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n"
        "\\end{tabular}\\end{table}\n", encoding="utf-8")
    print(f"wrote {OUT_R}\n" + rl.to_string(index=False))


def main():
    if not SUMM.exists():
        for p in (OUT_D, OUT_R):
            p.write_text("% pending run\n", encoding="utf-8")
        return
    block_d()
    if RULES.exists():
        rules()


if __name__ == "__main__":
    sys.exit(main())
