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
OUT_COMBINED = "tables/filter_ttests_combined.csv"

N_BOOT = 10000
BOOT_SEED = 20260604  # fixed: scripts must be reproducible (no wall-clock seed)

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


def bootstrap_ratio_ci(f0: np.ndarray, fk: np.ndarray, n_boot: int = N_BOOT,
                       seed: int = BOOT_SEED) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the floor ratio median(F0)/median(Fk).

    Resamples the paired seeds with replacement; the ratio is a robust,
    distribution-free effect size that does not lean on the log-normality the
    t-test assumes. Returns (ratio_point, ci_lo, ci_hi) at 95%."""
    rng = np.random.default_rng(seed)
    n = f0.size
    point = float(np.median(f0) / np.median(fk))
    idx = rng.integers(0, n, size=(n_boot, n))
    rb = np.median(f0[idx], axis=1) / np.median(fk[idx], axis=1)
    lo, hi = np.percentile(rb, [2.5, 97.5])
    return point, float(lo), float(hi)


def stouffer(pvals: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float]:
    """Combine independent one-sided p-values into one Z and p (Stouffer).

    The three models (quadratic, logistic, AR) are independent experiments of
    the same hypothesis; Stouffer's Z aggregates them into a single test of
    'filtering helps somewhere across tasks'."""
    p = np.clip(np.asarray(pvals, dtype=float), 1e-300, 1 - 1e-16)
    z = stats.norm.isf(p)  # one-sided -> z
    if weights is None:
        weights = np.ones_like(z)
    w = np.asarray(weights, dtype=float)
    z_comb = float(np.sum(w * z) / np.sqrt(np.sum(w ** 2)))
    p_comb = float(stats.norm.sf(z_comb))
    return z_comb, p_comb


def exact_sign_perm_p(d: np.ndarray) -> float:
    """Exact one-sided sign-flip permutation p for paired differences.

    Under H0 the sign of each paired difference is exchangeable, so the null
    is the 2**n equiprobable sign assignments. We count the fraction whose
    mean is >= the observed mean (one-sided, filter helps). For n<=8 this is
    enumerated exactly (256 assignments); larger n falls back to the
    closed-form normal-free bound via a fixed-seed Monte-Carlo draw."""
    d = np.asarray(d, dtype=float)
    n = d.size
    obs = float(np.mean(d))
    mag = np.abs(d)
    if n <= 12:
        ge = 0
        total = 1 << n
        for mask in range(total):
            signs = np.array([1.0 if (mask >> i) & 1 else -1.0 for i in range(n)])
            if np.mean(signs * mag) >= obs - 1e-15:
                ge += 1
        return ge / total
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.choice([-1.0, 1.0], size=(20000, n))
    means = (draws * mag).mean(axis=1)
    return float(np.mean(means >= obs - 1e-15))


def achieved_power(d_obs: float, n: int, alpha: float = 0.05) -> float:
    """Post-hoc power of the one-sided paired t-test at the observed effect."""
    from scipy.stats import nct, t as tdist
    ncp = d_obs * np.sqrt(n)               # noncentrality for paired t
    tcrit = tdist.ppf(1 - alpha, df=n - 1)
    return float(nct.sf(tcrit, df=n - 1, nc=ncp))


def min_detectable_d(n: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest paired Cohen's d the design can detect at given power."""
    from scipy.stats import nct, t as tdist
    tcrit = tdist.ppf(1 - alpha, df=n - 1)
    lo, hi = 0.0, 10.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        pw = nct.sf(tcrit, df=n - 1, nc=mid * np.sqrt(n))
        if pw < power:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def tost_equivalence(f0: np.ndarray, fk: np.ndarray, margin_log: float = 0.30) -> dict:
    """Two one-sided tests (TOST) for equivalence on log10(floor).

    H0 (non-equivalence): |E[d]| >= margin; H1 (equivalence): |E[d]| < margin,
    where d = log10(F0) - log10(Fk). margin_log=0.30 ~ a factor of 2 on the raw
    floor: filter and baseline are 'practically the same' if neither differs by
    more than ~2x. Returns the larger of the two one-sided p's (equivalence is
    claimed only if it is < alpha)."""
    d = np.log10(np.maximum(f0, 1e-12)) - np.log10(np.maximum(fk, 1e-12))
    n = d.size
    mean_d = float(np.mean(d))
    se = float(np.std(d, ddof=1)) / np.sqrt(n)
    if se == 0:
        eq = abs(mean_d) < margin_log
        return dict(mean_log_diff=mean_d, p_tost=0.0 if eq else 1.0, equivalent=eq)
    t_lo = (mean_d - (-margin_log)) / se     # test E[d] > -margin
    t_hi = (mean_d - margin_log) / se        # test E[d] <  margin
    p_lo = float(stats.t.sf(t_lo, df=n - 1))
    p_hi = float(stats.t.cdf(t_hi, df=n - 1))
    p_tost = max(p_lo, p_hi)
    return dict(mean_log_diff=mean_d, p_tost=p_tost, equivalent=p_tost < 0.05)


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
        perm_p=exact_sign_perm_p(d),
        power=achieved_power(abs(cohen_d) if np.isfinite(cohen_d) else 10.0, n),
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
    # bootstrap CI for the median speedup ratio
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    rb = np.median(a[idx], axis=1) / np.median(b[idx], axis=1)
    blo, bhi = np.percentile(rb, [2.5, 97.5])
    return dict(
        n=n, mean_log_diff=mean_d, speedup=float(10 ** mean_d),
        t_stat=float(t), p_one_sided=p_one, cohen_d=float(cohen_d),
        ci_lo=float(ci[0]), ci_hi=float(ci[1]), wilcoxon_p=np.nan,
        boot_ratio=float(np.median(a) / np.median(b)),
        boot_ratio_lo=float(blo), boot_ratio_hi=float(bhi),
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
            rp, rlo, rhi = bootstrap_ratio_ci(common["F0"].values, common[filt].values)
            res.update(boot_ratio=rp, boot_ratio_lo=rlo, boot_ratio_hi=rhi)
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

    # Combined (Stouffer) test across the three models for each headline regime
    comb_rows = []
    for noise, (filt, metric) in REGIME.items():
        opt = "adam" if metric == "t_eps" else "sgd"
        sel = full[(full.noise == noise) & (full["filter"] == filt)
                   & (full.metric == metric) & (full.optimizer == opt)]
        if len(sel) < 2:
            continue
        z, p = stouffer(sel["p_one_sided"].values)
        comb_rows.append(dict(
            noise=noise, filter=filt, metric=metric, optimizer=opt,
            n_models=len(sel), stouffer_z=z, stouffer_p=p,
            min_cohen_d=float(sel["cohen_d"].min()),
            max_cohen_d=float(sel["cohen_d"].max()),
        ))
    combined = pd.DataFrame(comb_rows)
    combined.to_csv(OUT_COMBINED, index=False)
    print(f"wrote {OUT_COMBINED}: {len(combined)} combined tests")
    print("\n=== Combined across models (Stouffer) ===")
    for _, r in combined.iterrows():
        print(f"{r.noise} {r['filter']} {r.metric:5s} ({r.n_models} tasks): "
              f"Z={r.stouffer_z:7.2f}  p={r.stouffer_p:.2e}  "
              f"|d|∈[{r.min_cohen_d:.2f},{r.max_cohen_d:.2f}]")

    # console summary
    print("\n=== Headline paired tests ===")
    for _, r in head.iterrows():
        eff = r.get("floor_ratio", r.get("speedup", np.nan))
        tag = "floor↓" if r.metric == "floor" else "T(ε)↓"
        print(f"{r.noise} {r['filter']} {r.optimizer:6s} {r.model:10s} {tag}: "
              f"{eff:6.1f}×  t={r.t_stat:7.2f}  p={r.p_one_sided:.2e}  "
              f"p_holm={r.p_holm:.2e}  d={r.cohen_d:5.2f}")

    calibrated_tests()
    control_tests(df, full)
    write_tex(full)


def control_tests(df: pd.DataFrame, full: pd.DataFrame) -> None:
    """Two honesty checks: (1) TOST equivalence of F4 and F0 on the Gaussian
    control N1; (2) robustness of the N3 rescue across SGD-family optimizers."""
    # (1) N1 equivalence
    print("\n=== N1 Gaussian control: TOST equivalence F4 vs F0 (margin=0.30 log) ===")
    eq_rows = []
    n1 = df[(df.noise == "N1") & (df.optimizer == "sgd")]
    for model, g in n1.groupby("model"):
        piv = g.pivot_table(index="seed", columns="filter", values="floor_p50")
        if "F0" not in piv.columns or "F4" not in piv.columns:
            continue
        r = tost_equivalence(piv["F0"].values, piv["F4"].values)
        r.update(model=model, noise="N1", filter="F4")
        eq_rows.append(r)
        print(f"  {model:10s} log-diff={r['mean_log_diff']:+.3f}  "
              f"p_TOST={r['p_tost']:.4f}  equivalent={r['equivalent']}")
    pd.DataFrame(eq_rows).to_csv("tables/filter_tost_n1.csv", index=False)

    # (2) N3 across optimizers
    print("\n=== N3 rescue robustness across SGD-family optimizers (F4 floor) ===")
    n3 = full[(full.noise == "N3") & (full["filter"] == "F4") & (full.metric == "floor")]
    for opt, g in n3.groupby("optimizer"):
        rmin, rmax = g.floor_ratio.min(), g.floor_ratio.max()
        print(f"  {opt:15s} ratio∈[{rmin:.1f},{rmax:.1f}]  "
              f"p_max={g.p_one_sided.max():.1e}  d∈[{g.cohen_d.min():.1f},{g.cohen_d.max():.1f}]")


def calibrated_tests() -> None:
    """Same paired floor test on the real-calibrated noise (Experiment 2,
    alpha_hat=1.21). Confirms the synthetic finding transfers to the tail
    index actually measured on financial data."""
    src = "tables/calibrated_synthetic.csv"
    out = "tables/filter_ttests_calibrated.csv"
    try:
        d = pd.read_csv(src)
    except FileNotFoundError:
        print(f"(skip calibrated: {src} not found)")
        return
    rows = []
    for (model, noise), g in d.groupby(["model", "noise"]):
        piv = g.pivot_table(index="seed", columns="filter", values="floor_p50")
        cv = g.pivot_table(index="seed", columns="filter", values="conv100")
        if "F0" not in piv.columns or "F4" not in piv.columns:
            continue
        c = piv[["F0", "F4"]].dropna()
        if len(c) < 3:
            continue
        r = paired_log_floor_test(c["F0"].values, c["F4"].values)
        rp, rlo, rhi = bootstrap_ratio_ci(c["F0"].values, c["F4"].values)
        r.update(boot_ratio=rp, boot_ratio_lo=rlo, boot_ratio_hi=rhi,
                 model=model, noise=noise, filter="F4", metric="floor",
                 conv_F0=float(np.nanmean(cv["F0"].values)) if "F0" in cv else np.nan,
                 conv_Fk=float(np.nanmean(cv["F4"].values)) if "F4" in cv else np.nan)
        rows.append(r)
    cal = pd.DataFrame(rows)
    cal["p_holm"] = np.nan
    for noise, idx in cal.groupby("noise").groups.items():
        cal.loc[idx, "p_holm"] = holm(cal.loc[idx, "p_one_sided"].values)
    cal.to_csv(out, index=False)
    print(f"\nwrote {out}: {len(cal)} calibrated comparisons")
    n3 = cal[cal.noise == "N3cal"]
    if len(n3):
        z, p = stouffer(n3["p_one_sided"].values)
        print(f"=== Calibrated N3cal (alpha=1.21) F4 vs F0: "
              f"Stouffer Z={z:.2f} p={p:.2e} ===")
        for _, r in n3.iterrows():
            print(f"  {r.model:10s} ratio={r.floor_ratio:6.1f}x  t={r.t_stat:6.2f}  "
                  f"p={r.p_one_sided:.2e}  d={r.cohen_d:5.2f}  "
                  f"conv {r.conv_F0:.2f}->{r.conv_Fk:.2f}")


def write_tex(full: pd.DataFrame) -> None:
    """Headline LaTeX table: N3 floor (F4) + N2 speed (F3) across models."""
    lines = []
    lines.append(r"\begin{tabular}{llccccc}")
    lines.append(r"\toprule")
    lines.append(r"Режим & Задача & эффект & 95\% ДИ$^{\dagger}$ & $t$ & $p$ (одност.) & $d$ Коэна \\")
    lines.append(r"\midrule")

    def fmt_p(p):
        if p < 1e-4:
            return f"${p:.1e}".replace("e-0", r"\!\times\!10^{-").replace("e-", r"\!\times\!10^{-") + "}$"
        return f"${p:.4f}$"

    def fmt_ci(lo, hi):
        return f"$[{lo:.0f},{hi:.0f}]$"

    # N3 / F4 / floor (SGD), three models
    n3 = full[(full.noise == "N3") & (full["filter"] == "F4")
              & (full.metric == "floor") & (full.optimizer == "sgd")]
    for _, r in n3.iterrows():
        lines.append(
            f"N3 (хвосты), F4 & {MODEL_RU.get(r.model, r.model)} & "
            f"${r.floor_ratio:.0f}\\times$ ниже & {fmt_ci(r.boot_ratio_lo, r.boot_ratio_hi)} & "
            f"${r.t_stat:.1f}$ & {fmt_p(r.p_one_sided)} & ${r.cohen_d:.2f}$ \\\\")
    lines.append(r"\midrule")
    # N2 / F3 / t_eps (Adam)
    n2 = full[(full.noise == "N2") & (full["filter"] == "F3")
              & (full.metric == "t_eps") & (full.optimizer == "adam")]
    for _, r in n2.iterrows():
        lines.append(
            f"N2 (память), F3 & {MODEL_RU.get(r.model, r.model)} & "
            f"${r.speedup:.1f}\\times$ быстрее & {fmt_ci(r.boot_ratio_lo, r.boot_ratio_hi)} & "
            f"${r.t_stat:.1f}$ & {fmt_p(r.p_one_sided)} & ${r.cohen_d:.2f}$ \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\multicolumn{7}{l}{\footnotesize $^{\dagger}$95\%-й бутстрэп-ДИ"
                 r" для отношения (10000 ресэмплов спаренных сидов).}\\")
    lines.append(r"\end{tabular}")
    with open(OUT_TEX, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT_TEX}")


if __name__ == "__main__":
    main()
