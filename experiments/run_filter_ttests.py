#!/usr/bin/env python3
"""Paired statistical tests for the filtering hypotheses.

For every experimental cell (block, model, noise, optimizer) the eight seeds
are *shared* across filters: seed ``s`` drives the same noise realisation for
F0 and for every Fk. That pairing lets us run the more powerful paired test
instead of a two-sample one, and it controls for the per-realisation difficulty
that dominates the between-seed variance.

We test the central claim of the paper — that the regime-matched filter lowers
the asymptotic gradient-noise floor ``||∇f||²`` relative to the unfiltered
baseline F0 — with a one-sided paired t-test on ``log10(floor)`` (a ratio on the
raw scale becomes a difference in log space, which is what the t-test models).
Each comparison is accompanied by

  * paired mean log10 difference  Δ = mean(log10 F0 − log10 Fk)  (Δ>0 ⇒ filter wins),
  * the implied floor ratio  10**Δ  (×-improvement),
  * Student t-statistic and one-sided p-value (H1: filter floor < F0 floor),
  * paired Cohen's d (Δ / sd of the per-seed differences),
  * a 95% CI for Δ on the log scale,
  * Wilcoxon signed-rank p (distribution-free backup),
  * Holm-corrected p across the family of comparisons.

For the long-memory block we additionally test the *speed* claim on T(ε).

Outputs
-------
tables/filter_ttests.csv      one row per (cell, filter) comparison
tables/filter_ttests_headline.csv   the regime-matched headline rows only
tables/filter_ttests.tex      LaTeX table for the paper (headline rows)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

SRC = "tables/convergence_rescue.csv"
OUT_FULL = "tables/filter_ttests.csv"
OUT_HEAD = "tables/filter_ttests_headline.csv"
OUT_TEX = "tables/filter_ttests.tex"

# Which filter is the regime-matched candidate for each noise class, and on
# which metric we expect it to win. floor = asymptotic ||∇f||² (lower better);
# t_eps = iterations to ε-accuracy (lower better).
REGIME = {
    "N3": ("F4", "floor"),   # heavy tails  -> causal median rescues convergence (H1)
    "N2": ("F3", "t_eps"),   # long memory  -> wavelet accelerates adaptive methods (H2)
    "N1": ("F4", "floor"),   # Gaussian control: expect NO win (H4)
    "N4": ("F4", "floor"),   # mixed: expect no simple-filter win (H3)
}

MODEL_RU = {"quadratic": "квадратичная", "logistic": "логистическая", "ar": "авторегрессия"}


def holm(pvals: np.ndarray) -> np.ndarray:
    """Holm–Bonferroni step-down adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def paired_log_floor_test(f0: np.ndarray, fk: np.ndarray) -> dict:
    """One-sided paired t-test on log10(floor): H1 filter floor < F0 floor."""
    eps = 1e-12
    l0 = np.log10(np.maximum(f0, eps))
    lk = np.log10(np.maximum(fk, eps))
    d = l0 - lk            # >0 means filter has the lower (better) floor
    n = d.size
    mean_d = float(np.mean(d))
    sd_d = float(np.std(d, ddof=1))
    se = sd_d / np.sqrt(n) if sd_d > 0 else 0.0
    if se > 0:
        t = mean_d / se
        p_one = float(stats.t.sf(t, df=n - 1))            # H1: d>0
    else:
        t = np.inf if mean_d > 0 else 0.0
        p_one = 0.0 if mean_d > 0 else 1.0
    cohen_d = mean_d / sd_d if sd_d > 0 else np.inf
    tcrit = stats.t.ppf(0.975, df=n - 1)
    ci = (mean_d - tcrit * se, mean_d + tcrit * se)
    # Wilcoxon signed-rank (one-sided, greater)
    try:
        w_p = float(stats.wilcoxon(d, alternative="greater", zero_method="wilcox").pvalue)
    except ValueError:
        w_p = np.nan
    return dict(
        n=n, mean_log_diff=mean_d, floor_ratio=float(10 ** mean_d),
        t_stat=float(t), p_one_sided=p_one, cohen_d=float(cohen_d),
        ci_lo=float(ci[0]), ci_hi=float(ci[1]), wilcoxon_p=w_p,
    )


def paired_teps_test(f0: np.ndarray, fk: np.ndarray) -> dict:
    """One-sided paired t-test on log10(T(ε)) speedup: H1 filter faster than F0.

    Non-converged seeds carry a censored T(ε); we cap them at the horizon so the
    comparison is conservative (a real never-converge is treated as merely
    'slow', which understates the filter's advantage)."""
    cap = float(np.nanmax(np.concatenate([f0, fk]))) * 2 + 1
    a = np.where(np.isfinite(f0) & (f0 > 0), f0, cap)
    b = np.where(np.isfinite(fk) & (fk > 0), fk, cap)
    l0 = np.log10(a)
    lk = np.log10(b)
    d = l0 - lk            # >0 means filter is faster
    n = d.size
    mean_d = float(np.mean(d))
    sd_d = float(np.std(d, ddof=1))
    se = sd_d / np.sqrt(n) if sd_d > 0 else 0.0
    if se > 0:
        t = mean_d / se
        p_one = float(stats.t.sf(t, df=n - 1))
    else:
        t = np.inf if mean_d > 0 else 0.0
        p_one = 0.0 if mean_d > 0 else 1.0
    cohen_d = mean_d / sd_d if sd_d > 0 else np.inf
    tcrit = stats.t.ppf(0.975, df=n - 1)
    ci = (mean_d - tcrit * se, mean_d + tcrit * se)
    return dict(
        n=n, mean_log_diff=mean_d, speedup=float(10 ** mean_d),
        t_stat=float(t), p_one_sided=p_one, cohen_d=float(cohen_d),
        ci_lo=float(ci[0]), ci_hi=float(ci[1]), wilcoxon_p=np.nan,
    )


def main() -> None:
    df = pd.read_csv(SRC)
    rows = []
    for (block, model, noise, opt), g in df.groupby(["block", "model", "noise", "optimizer"]):
        piv_floor = g.pivot_table(index="seed", columns="filter", values="floor_p50")
        piv_teps = g.pivot_table(index="seed", columns="filter", values="t_eps")
        piv_conv = g.pivot_table(index="seed", columns="filter", values="conv100")
        if "F0" not in piv_floor.columns:
            continue
        for filt in [c for c in piv_floor.columns if c != "F0"]:
            common = piv_floor[["F0", filt]].dropna()
            if len(common) < 3:
                continue
            res = paired_log_floor_test(common["F0"].values, common[filt].values)
            res.update(block=block, model=model, noise=noise, optimizer=opt,
                       filter=filt, metric="floor")
            # binary convergence fractions for context
            if filt in piv_conv.columns:
                res["conv_F0"] = float(np.nanmean(piv_conv["F0"].values))
                res["conv_Fk"] = float(np.nanmean(piv_conv[filt].values))
            rows.append(res)

            if filt in piv_teps.columns:
                ct = piv_teps[["F0", filt]].dropna(how="all")
                if len(ct) >= 3:
                    rt = paired_teps_test(ct["F0"].values, ct[filt].values)
                    rt.update(block=block, model=model, noise=noise, optimizer=opt,
                              filter=filt, metric="t_eps")
                    rows.append(rt)

    full = pd.DataFrame(rows)
    # Holm correction within each metric family
    full["p_holm"] = np.nan
    for metric, idx in full.groupby("metric").groups.items():
        full.loc[idx, "p_holm"] = holm(full.loc[idx, "p_one_sided"].values)
    full = full.sort_values(["metric", "p_one_sided"]).reset_index(drop=True)
    full.to_csv(OUT_FULL, index=False)
    print(f"wrote {OUT_FULL}: {len(full)} comparisons")

    # Headline: regime-matched filter/metric per noise class, SGD for floor /
    # Adam for speed, across the three models.
    head_rows = []
    for noise, (filt, metric) in REGIME.items():
        opt = "adam" if metric == "t_eps" else "sgd"
        sel = full[(full.noise == noise) & (full["filter"] == filt)
                   & (full.metric == metric) & (full.optimizer == opt)]
        head_rows.append(sel)
    head = pd.concat(head_rows).reset_index(drop=True)
    head.to_csv(OUT_HEAD, index=False)
    print(f"wrote {OUT_HEAD}: {len(head)} headline comparisons")

    # console summary
    print("\n=== Headline paired tests ===")
    for _, r in head.iterrows():
        eff = r.get("floor_ratio", r.get("speedup", np.nan))
        tag = "floor↓" if r.metric == "floor" else "T(ε)↓"
        print(f"{r.noise} {r['filter']} {r.optimizer:6s} {r.model:10s} {tag}: "
              f"{eff:6.1f}×  t={r.t_stat:7.2f}  p={r.p_one_sided:.2e}  "
              f"p_holm={r.p_holm:.2e}  d={r.cohen_d:5.2f}")

    write_tex(full)


def write_tex(full: pd.DataFrame) -> None:
    """Headline LaTeX table: N3 floor (F4) + N2 speed (F3) across models."""
    lines = []
    lines.append(r"\begin{tabular}{llcccc}")
    lines.append(r"\toprule")
    lines.append(r"Режим & Задача & эффект & $t$ & $p$ (одност.) & $d$ Коэна \\")
    lines.append(r"\midrule")

    def fmt_p(p):
        if p < 1e-4:
            return f"${p:.1e}".replace("e-0", r"\!\times\!10^{-").replace("e-", r"\!\times\!10^{-") + "}$"
        return f"${p:.4f}$"

    # N3 / F4 / floor (SGD), three models
    n3 = full[(full.noise == "N3") & (full["filter"] == "F4")
              & (full.metric == "floor") & (full.optimizer == "sgd")]
    for _, r in n3.iterrows():
        lines.append(
            f"N3 (хвосты), F4 & {MODEL_RU.get(r.model, r.model)} & "
            f"${r.floor_ratio:.0f}\\times$ ниже & ${r.t_stat:.1f}$ & "
            f"{fmt_p(r.p_one_sided)} & ${r.cohen_d:.2f}$ \\\\")
    lines.append(r"\midrule")
    # N2 / F3 / t_eps (Adam)
    n2 = full[(full.noise == "N2") & (full["filter"] == "F3")
              & (full.metric == "t_eps") & (full.optimizer == "adam")]
    for _, r in n2.iterrows():
        lines.append(
            f"N2 (память), F3 & {MODEL_RU.get(r.model, r.model)} & "
            f"${r.speedup:.1f}\\times$ быстрее & ${r.t_stat:.1f}$ & "
            f"{fmt_p(r.p_one_sided)} & ${r.cohen_d:.2f}$ \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    with open(OUT_TEX, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
