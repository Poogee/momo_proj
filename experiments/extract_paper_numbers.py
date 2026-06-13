#!/usr/bin/env python3
"""Dump every statistic the paper quotes, labelled by its prose location, from
the regenerated CSVs. Single source of truth for updating refactored_paper.tex
after the seed bump (8 -> 100 synthetic, 4 -> 50 real)."""
from __future__ import annotations
import importlib.util as _u
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
_spec = _u.spec_from_file_location("tt", ROOT / "experiments" / "run_filter_ttests.py")
tt = _u.module_from_spec(_spec); _spec.loader.exec_module(tt)


def rng(vals, fmt="{:.0f}"):
    vals = [v for v in vals if np.isfinite(v)]
    return f"{fmt.format(min(vals))}--{fmt.format(max(vals))}"


def sci(p):
    return f"{p:.1e}"


full = pd.read_csv(ROOT / "tables/filter_ttests.csv")
comb = pd.read_csv(ROOT / "tables/filter_ttests_combined.csv")
raw = pd.read_csv(ROOT / "tables/convergence_rescue.csv")
N = int(raw.seed.nunique())
print(f"#### SEEDS PER CELL n = {N}, df = {N-1}\n")

# ---- family size (Holm) ----
for metric, g in full.groupby("metric"):
    print(f"[L671/703 family size] metric={metric}: {len(g)} comparisons")
print(f"[total full rows] {len(full)}\n")

# ---- N3 floor F4 sgd (3 models) ----
n3 = full[(full.noise=="N3")&(full["filter"]=="F4")&(full.metric=="floor")&(full.optimizer=="sgd")]
print("=== [Result para 666-681 / table / abstract] N3 F4 floor, SGD ===")
for _, r in n3.iterrows():
    print(f"  {r.model:11s} ratio={r.floor_ratio:7.1f}x  t={r.t_stat:6.2f}  p={sci(r.p_one_sided)}  "
          f"d={r.cohen_d:6.2f}  bootCI=[{r.boot_ratio_lo:.0f},{r.boot_ratio_hi:.0f}]  "
          f"p_holm={sci(r.p_holm)}  conv_Fk={r.get('conv_Fk',np.nan):.2f}")
print(f"  RANGE ratio={rng(n3.floor_ratio)}  t={rng(n3.t_stat,'{:.1f}')}  "
      f"p={sci(n3.p_one_sided.max())}..{sci(n3.p_one_sided.min())}  d={rng(n3.cohen_d,'{:.1f}')}")
c = comb[(comb.noise=="N3")&(comb.metric=="floor")]
if len(c): print(f"  STOUFFER Z={c.iloc[0].stouffer_z:.1f}  p={sci(c.iloc[0].stouffer_p)}")
# convergence fraction for abstract "8/8 -> 100/100"
cf = raw[(raw.noise=="N3")&(raw["filter"]=="F4")&(raw.optimizer=="sgd")]
cf0 = raw[(raw.noise=="N3")&(raw["filter"]=="F0")&(raw.optimizer=="sgd")]
print(f"  [abstract conv] N3 F4 sgd conv100 mean={cf.conv100.mean():.3f} (n per model={N}); "
      f"F0 conv100 mean={cf0.conv100.mean():.3f}")
print()

# ---- N2 speed F3 adam (3 models) ----
n2 = full[(full.noise=="N2")&(full["filter"]=="F3")&(full.metric=="t_eps")&(full.optimizer=="adam")]
print("=== [Result para 683-690] N2 F3 t_eps, Adam ===")
for _, r in n2.iterrows():
    print(f"  {r.model:11s} speedup={r.speedup:6.2f}x  t={r.t_stat:6.2f}  p={sci(r.p_one_sided)}  "
          f"d={r.cohen_d:6.2f}  bootCI=[{r.boot_ratio_lo:.0f},{r.boot_ratio_hi:.0f}]  p_holm={sci(r.p_holm)}")
sig = n2[n2.p_one_sided < 0.05]
print(f"  SIG-RANGE speedup={rng(sig.speedup,'{:.1f}')}  t={rng(sig.t_stat,'{:.1f}')}  d_min={sig.cohen_d.min():.2f}")
c = comb[(comb.noise=="N2")&(comb.metric=="t_eps")]
if len(c): print(f"  STOUFFER Z={c.iloc[0].stouffer_z:.1f}  p={sci(c.iloc[0].stouffer_p)}")
print()

# ---- N1 control F4 sgd ----
n1 = full[(full.noise=="N1")&(full["filter"]=="F4")&(full.metric=="floor")&(full.optimizer=="sgd")]
print("=== [Control para 726] N1 F4 floor, SGD (expect no win) ===")
for _, r in n1.iterrows():
    print(f"  {r.model:11s} p_one={r.p_one_sided:.3f}  d={r.cohen_d:+.2f}  ratio={r.floor_ratio:.2f}")
print(f"  p_min={n1.p_one_sided.min():.3f}  all d<0: {(n1.cohen_d<0).all()}")
tost = pd.read_csv(ROOT / "tables/filter_tost_n1.csv")
print("  TOST:", [f"{r.model}:p={r.p_tost:.2e},eq={r.equivalent}" for _,r in tost.iterrows()])
print()

# ---- N4 control F4 sgd ----
n4 = full[(full.noise=="N4")&(full["filter"]=="F4")&(full.metric=="floor")&(full.optimizer=="sgd")]
print("=== [Control para 733-740] N4 F4 floor, SGD (tiny effect) ===")
for _, r in n4.iterrows():
    print(f"  {r.model:11s} p_one={r.p_one_sided:.4f}  p_holm={r.p_holm:.3f}  ratio={r.floor_ratio:.2f}  d={r.cohen_d:.2f}")
print(f"  nominal p range=[{n4.p_one_sided.min():.4f},{n4.p_one_sided.max():.4f}]  "
      f"p_holm range=[{n4.p_holm.min():.3f},{n4.p_holm.max():.3f}]  ratio={rng(n4.floor_ratio,'{:.2f}')}")
z4, p4 = tt.stouffer(n4.p_one_sided.values)
print(f"  STOUFFER Z={z4:.1f} p={sci(p4)}")
print()

# ---- power + permutation at n=N ----
print("=== [Power para 748-762] ===")
mde80 = tt.min_detectable_d(N, power=0.8)
mde50 = tt.min_detectable_d(N, power=0.5)
print(f"  MDE power=0.8: d>={mde80:.2f}   MDE power=0.5: d>={mde50:.2f}")
d_allpos = np.full(N, 0.5)  # all-positive differences
pperm = tt.exact_sign_perm_p(d_allpos)
print(f"  permutation all-positive n={N}: p={pperm:.2e}  (n<=12 exact, else MC 20000 draws)")
print(f"  -> n={N} uses {'EXACT 2^n' if N<=12 else 'MONTE-CARLO (20000 draws)'}")
print()

# ---- cross-optimizer N3 F4 ----
print("=== [Optimizer robustness 764-768] N3 F4 floor by optimizer ===")
for opt in ["clipped_sgd","normalized_sgd"]:
    g = full[(full.noise=="N3")&(full["filter"]=="F4")&(full.metric=="floor")&(full.optimizer==opt)]
    if len(g):
        print(f"  {opt:15s} ratio={rng(g.floor_ratio,'{:.1f}')}  p_max={sci(g.p_one_sided.max())}  d={rng(g.cohen_d,'{:.1f}')}")
print()

# ---- calibrated N3cal ----
calp = ROOT / "tables/filter_ttests_calibrated.csv"
if calp.exists():
    cal = pd.read_csv(calp)
    n3c = cal[cal.noise=="N3cal"]
    print("=== [forest fig / Exp2] calibrated N3cal F4 floor ===")
    for _, r in n3c.iterrows():
        print(f"  {r.model:11s} ratio={r.floor_ratio:.1f}x  t={r.t_stat:.1f}  p={sci(r.p_one_sided)}  d={r.cohen_d:.2f}")
    if len(n3c):
        z,p = tt.stouffer(n3c.p_one_sided.values)
        print(f"  ratio range={rng(n3c.floor_ratio,'{:.0f}')}  STOUFFER Z={z:.1f} p={sci(p)}")
    print()

# ---- applied real data ----
appp = ROOT / "tables/filter_ttests_applied.csv"
if appp.exists():
    ap = pd.read_csv(appp)
    print("=== [Exp3 stat para 922-931] applied real-data t-tests ===")
    for dom in ["financial_15m","financial_daily","macro_fred","nonfinancial_ett"]:
        for filt in ["F2","F1","F4"]:
            sel = ap[(ap.domain==dom)&(ap["filter"]==filt)&(ap.metric=="t_conv")]
            if len(sel):
                r = sel.iloc[0]
                print(f"  {dom:18s} {filt} speedup={r.speedup:6.2f}x  t={r.t_stat:6.2f}  n={int(r.n)}  "
                      f"p={sci(r.p_one_sided)}  conv {r.get('conv_F0',np.nan):.2f}->{r.get('conv_Fk',np.nan):.2f}")
    print()

# ---- applied descriptive (tab:real) ----
asp = ROOT / "tables/applied_convergence_summary.csv"
araw = ROOT / "tables/applied_convergence.csv"
if asp.exists():
    s = pd.read_csv(asp)
    rd = pd.read_csv(araw)
    print("=== [Exp3 tab:real 896-915] applied descriptive (Adam) ===")
    for dom in ["financial_15m","financial_daily","nonfinancial_ett","macro_fred"]:
        g = s[(s.domain==dom)&(s.optimizer=="adam")]
        if not len(g): continue
        f0 = g[g["filter"]=="F0"].iloc[0]
        ncell = len(rd[(rd.domain==dom)&(rd.optimizer=="adam")&(rd["filter"]=="F0")])
        best = g.loc[g.t_conv_med.idxmin()]
        speed = f0.t_conv_med / max(best.t_conv_med, 1)
        print(f"  {dom:16s} n_cells={ncell}  T_F0={f0.t_conv_med:.0f}  "
              f"conv_F0={f0.conv_frac:.3f} ({round(f0.conv_frac*ncell)}/{ncell})  "
              f"best={best['filter']} T={best.t_conv_med:.0f}  speedup={speed:.1f}x  "
              f"holdout best/F0={best.holdout_mse_med:.3f}/{f0.holdout_mse_med:.3f}")
    print()
# ---- descriptive tables from convergence_rescue_summary ----
summ = pd.read_csv(ROOT / "tables/convergence_rescue_summary.csv")
def srow(block, model, noise, opt, filt):
    g = summ[(summ.block==block)&(summ.model==model)&(summ.noise==noise)
             &(summ.optimizer==opt)&(summ["filter"]==filt)]
    return g.iloc[0] if len(g) else None

print("\n=== [tab:rescue 489-501 + L470-505 + abstract] N3 SGD descriptive ===")
for model in ["quadratic","logistic","ar"]:
    f0 = srow("A",model,"N3","sgd","F0"); f4 = srow("A",model,"N3","sgd","F4")
    if f0 is None or f4 is None: continue
    print(f"  {model:11s} bin F0={f0.conv100_frac:.3f}({round(f0.conv100_frac*N)}/{N}) "
          f"F4={f4.conv100_frac:.3f}({round(f4.conv100_frac*N)}/{N})  "
          f"T100 F0={f0.t_drop100_med:.0f} F4={f4.t_drop100_med:.0f}  "
          f"floor F0={f0.floor_p50_med:.4f} F4={f4.floor_p50_med:.5f}  "
          f"ratio F0/F4={f4.floor_ratio_vs_F0:.0f}x")

print("\n=== [tab:h2 568-578 + L538-549] N2 Adam quadratic descriptive ===")
for filt in ["F0","F1","F2","F3","F4"]:
    r = srow("B","quadratic","N2","adam",filt)
    if r is None: continue
    print(f"  {filt}: T_eps={r.t_eps_med:.0f}  speedup_vsF0={r.speedup_vs_F0:.1f}x  "
          f"floor={r.floor_p50_med:.3f}  meanlog10(auc)={r.auc_mean:+.2f}")

print("\n=== [L550-552] N2 SGD (block A) linear/median floor ratios F0/Fk ===")
ratios = []
for model in ["quadratic","logistic","ar"]:
    for filt in ["F1","F2","F3","F4"]:
        r = srow("A",model,"N2","sgd",filt)
        if r is not None and np.isfinite(r.floor_ratio_vs_F0): ratios.append(r.floor_ratio_vs_F0)
print(f"  N2 SGD F0/Fk ratio range=[{min(ratios):.2f},{max(ratios):.2f}]")

print("\n=== [tab:h3 617-625 + L604] N4 SGD floor ratios F0/Fk ===")
for model in ["quadratic","logistic","ar"]:
    cells = {f: srow("A",model,"N4","sgd",f) for f in ["F1","F2","F3","F4"]}
    vals = {f:(r.floor_ratio_vs_F0 if r is not None else float('nan')) for f,r in cells.items()}
    print(f"  {model:11s} " + "  ".join(f"{f}={v:.2f}" for f,v in vals.items()))

# ---- F5 cascade: N3 & N4 floor (SGD) ----
print("\n=== [CASCADE F5 = median(F4)->wavelet(F3)] N3 & N4 floor, SGD ===")
for noise in ["N3", "N4"]:
    g = full[(full.noise==noise)&(full["filter"]=="F5")&(full.metric=="floor")&(full.optimizer=="sgd")]
    print(f"  -- {noise} --")
    for _, r in g.iterrows():
        print(f"    {r.model:11s} ratio={r.floor_ratio:7.2f}x  t={r.t_stat:7.2f}  p={sci(r.p_one_sided)}  "
              f"d={r.cohen_d:6.2f}  p_holm={sci(r.p_holm)}  ci=[{r.boot_ratio_lo:.0f},{r.boot_ratio_hi:.0f}]  "
              f"conv_Fk={r.get('conv_Fk',float('nan')):.2f}")
    if len(g):
        z, p = tt.stouffer(g.p_one_sided.values)
        fm = '{:.2f}' if noise == "N4" else '{:.0f}'
        print(f"    RANGE ratio={rng(g.floor_ratio,fm)}  t={rng(g.t_stat,'{:.1f}')}  STOUFFER Z={z:.1f} p={sci(p)}")

print("\n=== [tab:h3 +F5] N4 SGD floor ratios F0/Fk (descriptive) ===")
for model in ["quadratic","logistic","ar"]:
    vals = {}
    for filt in ["F1","F2","F3","F4","F5"]:
        r = srow("A",model,"N4","sgd",filt)
        vals[filt] = r.floor_ratio_vs_F0 if r is not None else float('nan')
    print(f"  {model:11s} " + "  ".join(f"{f}={v:.2f}" for f,v in vals.items()))

print("\n=== [H1 note] N3 SGD descriptive: cascade F5 vs median F4 (floor F0/Fk) ===")
for model in ["quadratic","logistic","ar"]:
    f4 = srow("A",model,"N3","sgd","F4"); f5 = srow("A",model,"N3","sgd","F5")
    if f4 is not None and f5 is not None:
        print(f"  {model:11s} F4={f4.floor_ratio_vs_F0:.0f}x   F5(cascade)={f5.floor_ratio_vs_F0:.0f}x")

calp2 = ROOT / "tables/filter_ttests_calibrated.csv"
if calp2.exists():
    cal2 = pd.read_csv(calp2)
    print("\n=== [Exp2 cascade] calibrated F5 (N3cal/N4cal) ===")
    for noise in ["N3cal","N4cal"]:
        g = cal2[(cal2.noise==noise)&(cal2["filter"]=="F5")]
        for _, r in g.iterrows():
            print(f"  {noise} {r.model:11s} ratio={r.floor_ratio:.2f}x t={r.t_stat:.1f} "
                  f"p={sci(r.p_one_sided)} d={r.cohen_d:.2f}")

print("DONE")
