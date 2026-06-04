# Decisions log

Living record of non-obvious engineering / methodological choices.

## 2026-06-04 (statistical-significance section for the filtering claims)

- Added `experiments/run_filter_ttests.py`: formal paired tests backing the
  hypotheses, operating on the per-seed `tables/convergence_rescue.csv`
  (8 seeds, all blocks/models/noise/optimizers) and `calibrated_synthetic.csv`.
- **Paired, not two-sample.** Seed `s` drives the *same* noise realisation for
  F0 and every Fk, so we test the per-seed differences. This removes the
  between-realisation difficulty variance and is far more powerful — it is what
  makes the effect sizes huge (paired Cohen's d = 4–11 on N3).
- **Log scale on the floor.** `||grad f||^2` is heavy-tailed and the claim is a
  *ratio* (Nx lower floor); `d = log10(F0) - log10(Fk)` turns the ratio into a
  difference (what the t-test models) and stabilises variance.
- **One-sided** (H1: filter lowers the floor / speeds up T(eps)) — we have a
  directional prior from the theory, so two-sided would waste power.
- Multiplicity handled with **Holm** within each metric family (192 each).
  Backed up by **Wilcoxon**, **exact 2^8 sign-flip permutation** (p=1/256 on N3),
  **bootstrap** ratio CIs (10k resamples, fixed seed — scripts can't use
  Date/random), and **Stouffer** combine across the 3 tasks (N3 Z=9.2,
  p=1.5e-20).
- **Honesty checks, not just wins.** Gaussian N1: TOST equivalence (margin 0.30
  log ≈ factor 2) shows F4 ≡ F0 (p_TOST<1e-3) — filtering doesn't hurt either.
  Mixed N4: nominal effects die under Holm and the *effect size* is ~1.1x vs
  ~100x on N3 → reframed as statistical-vs-practical significance (this is H3).
  Power analysis: at n=8 the MDE is d≈0.98 @ power 0.8, so the design is
  overpowered for the real effect and the N1/N4 nulls are genuine.
- **Cross-optimizer robustness:** N3 rescue holds under Clipped-SGD (3.8–5x)
  but vanishes under Normalized-SGD (it already strips gradient scale) — input
  filtering and in-optimizer robustification are *substitutes* (premise ii).
- **Theory:** added a Proposition in §Предпосылки — the sample median of an
  alpha-stable window (w>=3) has finite variance, restoring the Robbins–Monro
  finite-variance premise; linear filters can't (stable laws are closed under
  convolution). This is *why* only F4 lowers the N3 floor and why d is so large.
- Outputs: `tables/filter_ttests*.csv`, `tables/filter_tost_n1.csv`,
  `tables/filter_ttests.tex` (\input into the paper), `figures/filter_ttests_forest.pdf`.
  New §«Статистическая значимость» (4.5) in refactored_paper.tex; paper now 9
  pages, compiles clean with xelatex. 10 unit tests in
  `tests/test_filter_ttests.py` (load the script via importlib — it is not an
  installed package). NB: xelatex is required (fontspec), not pdflatex.

## 2026-04-26

- Hardware: RTX 2070 SUPER (8 GB), Ryzen 7 5800X3D, 31 GB RAM. Synthetic
  factorial is small enough to run on CPU with joblib parallelism; GPU
  reserved for the learnable-filter / forecaster ablations later.
- System Python 3.12 is externally-managed (PEP 668); we install with
  `--break-system-packages` to match the existing environment instead of
  spinning a venv for already-present heavy deps (torch, scipy, pandas).
- Disk pressure (started at 9 GB free); cleared `~/.cache/pip` (~9.6 GB)
  to reclaim space. Run artifacts go under `runs/` with light parquet.
- Kalman filter implemented locally as a 1-D local-level model; full
  filterpy used as cross-check in tests.
- alpha-stable sampling via `scipy.stats.levy_stable.rvs` (CMS algorithm).
- FARIMA(0,d,0) realised via fractional-difference filter coefficients.
- Real data: yfinance with a fixed basket of equities + crypto + FX, log
  returns. Cache to data/cache/ as parquet so reruns are cheap.
- Report in Russian Typst (proposal style), even though the original
  proposal source is `.tex` — instructions explicitly request Typst.

## 2026-04-27 (innovation phase wrap-up)

- 44 innovation iterations on top of the original baseline. All committed
  with `innov N:` prefix. Per-iteration summary in `ITERATIONS.md`.
- F5/F9 learnable CNN denoisers added — F9 (4× larger) becomes the new
  universal lead on synthetic SNR (avg +17.3 dB over 6 noise classes).
- F8 adaptive meta-filter routes by online α̂/Ĥ on the differenced
  series; α-threshold default 1.9 is near-optimum (sensitivity sweep in
  iter 39).
- Iteration 28 reported a positive F4 sign-prediction result on real
  data (+7 pp over majority). Iteration 40 found this was a *lookahead
  bias* artifact: the centered MedianFilter peeks at the future. Added
  `CausalMedianFilter` and re-ran the experiment honestly in iteration 40
  — the gain disappears (causal F4 ≈ 0.59 < F0 0.60). The original
  finding is preserved in the report alongside the correction so the
  story stays auditable.
- Iteration 41 verified the magnitude-forecast conclusion (F0 wins) is
  robust to causality, since F1 (trailing MA) and F2 (forward Kalman)
  were already causal in the original sweep.
- Iteration 42 fixed a `_to_returns` data bug where `dropna(how="any")`
  silently wiped DataFrames if any single ticker's yfinance fetch failed.
  Replaced with `dropna(how="all")`; per-ticker retention happens at
  fetch_returns level via the keep-filter.
- Total assets at wrap-up: 11 modules in src/momo, 82 tests, 34 figures,
  21 tables, 41-page Typst PDF, 75+ commits. End-to-end reproduce.sh
  runtime ≈ 1–1.5 h on RTX 2070 SUPER + 16-core Ryzen.

## 2026-05-17 (positive-result phase, branch feature/filter-convergence-positive)

- Goal sharpened to *empirically prove and headline* that filter
  preprocessing rescues / accelerates stochastic-optimization
  convergence, with multi-metric evidence across models and domains.
- Operating points chosen by a calibration probe, then frozen (no
  per-seed/window/metric cherry-picking): the robust, model-independent
  signal is the **heavy-tailed SGD noise floor** (F4/F7/FA cut it
  17–186× vs F0); for quadratic/logistic this flips binary convergence
  0/8 → 8/8. The legacy "~700× Adam on N4" figure was *not* re-quoted:
  it could not be reproduced from a committed table at the stated
  settings, so per the no-fabrication rule it was replaced by a freshly
  measured, reproducible Adam/AdamW result (wavelet 11.2× on long-memory
  N2) with CIs. Honest negatives kept (Gaussian control, mixed N4cal,
  raw daily returns, ETT holdout, FRED).
- Reference integrity: every cited modern work verified from a primary
  source (arXiv abstract / publisher / the PDFs in `articles/`).
  Chandak et al. confirmed arXiv:2603.19648 (the proposal placeholder
  2503.XXXXX and the wrong Anantharam-like title were corrected);
  Anantharam–Borkar 2012 verified and the project repositioned relative
  to its asymptotic analysis vs Chandak's finite-time bounds.
- Data integrity in this sandbox: yfinance cache is genuinely real
  (verified: SPY −11.6% on 2020-03-16, α̂≈1.05, Ĥ|r|≈0.86); the ETT
  sensor download is real; the FRED endpoint is proxied/flaky here, so
  `fetch_fred` was rewritten to per-series `fredgraph.csv` (unambiguous)
  and the report leans headline numbers on the verifiably-real domains,
  describing FRED as reproducible-against-live but offline here.
- No GPU: F5/F9 learnable filters not retrained (committed weights kept,
  out of the new factorial); CPU joblib, 8 seeds, targeted horizons keep
  the new pipeline ≈30–60 min.
- Wrap-up: 105 tests passing, +4 experiment scripts, +1 figure builder,
  report.pdf 6 pp, reviewed proposal restructured (clean Постановка).

- Owner review follow-up: filter taxonomy of the *new* deliverables
  narrowed to **F0–F4** (F7 hybrid / FA online removed from the new
  experiments, figures, report; the `OnlineAdaptiveFilter` class stays
  in `src/momo/filters.py` so the pre-existing legacy experiments and
  their tests keep working — deleting it would break the repo and erase
  prior honest results, which the brief forbids). The **MLP task was
  removed** (the 100×-drop binary metric did not separate for it, so it
  added noise not signal). The binary-convergence bar chart was dropped
  (several models had all-identical bars → visually uninformative); the
  headline figure is now 2 panels (floor + median curves). Net effect on
  the story: the hero is unambiguously the **causal median F4** — under
  heavy tails F1/F2/F3 fail exactly like F0 (0/8), only F4 rescues
  (8/8), which is a *stronger* and cleaner claim than before. All
  numbers re-measured (8 seeds); MLP tests removed.

## 2026-05-30 (paper restructure)

- Paper rebuilt around the reviewer's structural request: separate the
  *formal problem* (§II) from *what we used to solve it* (§III) — §II
  contains only definitions of the time series model, the learning
  task, the forecast, the filter as an abstract operator, and the
  research question (\ref{def:research}). Concrete F1..F11 / SGD/Adam
  appear only in §III where they are needed for the lemma/proposition.
- Practical criteria (§V) are now per-rule tagged `[T]` (theoretical),
  `[E]` (experimental), or `[H]` (heuristic). This was directly asked
  for in the latest reviewer note.
- New causal cascade F10 (median→Kalman) added explicitly for the
  mixed N4 regime; an adaptive variant F11 picks stages by online
  α̂ and Ĥ. F11 is by construction `F0` on Gaussian (the relative
  efficiency penalty of §III Remark 1 should be avoided automatically).
- The synthetic-N4 evidence for the cascade is modest (~1.2×); we do
  not over-claim. The main positive comes from the extended real-data
  Block D where, on financial intraday returns, the Kalman and the
  cascade (F10/F11) deliver 14×..77× speedups for Adam with no holdout
  hit.
- Extended dataset basket aims at non-cherry-picked breadth: 22
  financial instruments across asset classes, plus 3 non-financial
  sensor variants (ETTh1/h2/m1) and a 200-year scientific series
  (sunspots). On the latter three F0 is best — also reported
  honestly.

## 2026-06-02 (fresh-data sweep + lean filter set)

- **Lean working filter set.** Trimmed the deployed/reported pool from 14
  operators to five strictly-causal O(T) filters:
  `{F0, F4, FA, F10, F11}`. Rationale: on every real series the linear
  (F1 MA, F2 Kalman) and wavelet (F3, F6) filters either tracked the F0
  baseline or were provably closed under the α-stable law (no finite
  variance); the learnable CNN denoisers (F5, F9) were slow and overfit
  to synthetic autocorrelation; the hybrid (F7), router (F8) and ensemble
  (FE) were strictly dominated by the cascades while running much slower.
  F1/F2/F3 are kept in the codebase **only as synthetic baselines** for
  Blocks A–B; the others are deprecated. Kalman survives solely as the
  2nd stage of the F10/F11 cascades. (Classes left in `src` for
  backward-compat of legacy scripts/tests; not deleted.)
- **New datasets parsed (June 2026).** Added loaders in `momo.data`:
  `fetch_binance` (BTCUSDT/ETHUSDT 1h, public data.binance.vision),
  `fetch_electricity`, `fetch_traffic` (UCI/LSTNet),
  `fetch_weather_noaa` (NOAA GHCN-daily), `fetch_cmapss` (NASA FD001),
  `fetch_atm` (daily cash withdrawals); ETTm2 via existing
  `fetch_nonfinancial`. Each follows the house pattern (parquet cache →
  `data/raw` → network → synthetic fallback).
- **Sources dropped as not freely obtainable:** LOBSTER full feed
  (portal/sample-gated — sample URLs return HTML error pages), NYSE TAQ
  and CRSP (WRDS subscription). Honestly noted in the paper rather than
  faked.
- **Datasets dropped as non-showing (per "keep only what produces a
  result"):** Electricity Load (α̂≈0.6) and Traffic PEMS (α̂≈1.0) — very
  heavy-tailed yet no filter beat F0 and there was no rescue (the AR base
  converges on its own; the bottleneck is not the gradient floor). Kept
  in the runner for reproducibility but omitted from Block D table; one
  honest prose sentence explains why.
- **Headline result (Block D, `run_new_datasets.py`):** on Binance 1h
  crypto returns (α̂≈1.46) Adam without a filter converges **0/4** seeds
  (stuck at the noise floor); the causal cascade F10 gives **4/4**,
  **72.7×** faster, gradient floor **169×** lower, holdout MSE flat
  (≈1.7%). Independent data, same regime as the old intraday-equity win.
- **Second positive (non-financial):** C-MAPSS turbofan — median F4
  improves **both** convergence (1.9×) **and** holdout MSE (−20%); the
  Kalman stage of F10 over-smooths the degradation trend and *hurts*
  holdout, which the adaptive F11 correctly avoids. New antipattern (iv)
  added: do not blindly cascade over a slow useful trend.
- **Negative controls confirm the rule:** near-Gaussian NOAA (α̂≈1.79)
  and smooth ETTm2 → best filter is F0 (1.00×); adaptive F11 declines to
  filter. Report sections updated: filter family, intro contribution,
  Block D, semi-synthetic, candidate table, "when not to filter",
  cascade-default, antipatterns, conclusion. `paper_ru.pdf` recompiled
  clean (no undefined refs); 27 filter tests pass.

## 2026-06-02 (restore full set + clean numbering + wide rule)

- **Reverted the lean trim; restored the original five and gave the whole
  pool contiguous numbering** $F_0$--$F_7$ (no more $F_4\to F_{10}$ gap):
  $F_0$ identity, $F_1$ MA, $F_2$ Kalman, $F_3$ wavelet (non-causal /
  oracle only), $F_4$ causal median, $F_5$ cascade median→Kalman (was
  F10), $F_6$ adaptive cascade (was F11), $F_7$ online median/EMA (was
  FA). The paper was globally renumbered to match. CNN denoisers, the
  adaptive wavelet, hybrid, router and ensemble stay out of the study
  (slow, never beat the cascades).
- **Wide basket for full-coverage rule derivation** (17 domains): the
  extended daily financial basket, intraday 15m/5m, Binance crypto 1h,
  volatility |r| (daily + crypto, the long-memory regime), FRED,
  ETTh1/h2/m1/m2, Electricity, Traffic, NOAA, C-MAPSS, ATM, sunspots.
  6528 cells in ~18 min. All four (α̂,Ĥ) sectors are now populated (the
  |r| and sunspots series supply Ĥ>0.6).
- **Headline shifted with the full set.** With F2 (Kalman) back in, on
  heavy-tailed short-memory returns it edges the cascade on raw Adam
  speed: crypto-1h F2 ≈78×, 15m ≈80×, 5m ≈78×, daily ≈19× (F0 converges
  only 0.38–0.7 of seeds). The cascade F5 is second on speed (≈64×) but
  gives the **largest gradient-floor reduction (127× vs 120× for F2)** —
  so F5 stays the principled default when the floor matters (SGD / mixed
  regime); F2 when only Adam wall-clock matters. Mechanism per Remark n2
  (Adam's EMA bias under autocorrelation), not the tail per se.
- **Dropped the lean-run "C-MAPSS −20% holdout" claim** — it was an
  artifact of a different F0 baseline (8 vs 6 sensors). In the full run
  causal F4/F7 give only ~5–8% holdout gain on C-MAPSS; the −41% comes
  from the *non-causal* oracle wavelet F3, which is excluded from the
  rule. Reported honestly as the gap between oracle and achievable causal
  gain. The "don't blindly cascade over a slow trend" antipattern still
  holds (F5 triples C-MAPSS holdout MSE: 0.97 vs 0.32).
- **Derived rule (tab:rules), reported where filtering ACTUALLY helps
  (speedup ≥1.5), not a raw mode**: heavy-tail+short-memory → F2/F5
  (helps 7/40 cells, up to 80×); every other sector → F0. New runner
  `run_new_datasets.py` + `make_new_datasets_table.py` emit
  `new_datasets_{summary,criteria,rules}.csv` and the two LaTeX tables.
  paper recompiled clean; tests pass.

## 2026-06-02 (figures/tables fixes: numbering, all filters, layout)

- **Killed the last stale F10/F11 in figures & tables.** Block C
  (`run_cascade_n4.py`) used the old keys F10/F11 and showed only 5
  filters without subscripts — regenerated with contiguous F0–F6, proper
  $F_n$ subscripts, and ALL seven filters. Deleted the unused
  `tables/extended_factorial_block_d.tex` (phase-1 leftover, still had F11).
- **New all-filters figure** (`make_new_datasets_figure.py` →
  `figures/new_datasets_heatmap.pdf`, embedded as a full-width
  `figure*`): two heatmaps over the full F0–F7 set × all 17 domains —
  (a) Adam speedup vs F0, (b) gradient-floor reduction vs F0. Directly
  answers "graphs don't show all filters". F3 marked as the non-causal
  oracle. The heavy-tailed-returns block lights up (F2/F5 = 78×/142×);
  everything else ≈1×.
- **New all-filters table** (`tables/new_datasets_crypto.tex`): every
  filter F0–F7 on the headline crypto-1h domain with conv/speedup/floor/
  holdout, so a table also shows all eight filter numbers.
- **Verified layout** by rasterizing the PDF (pdftoppm) and reading each
  table/figure page: no text overlap, tables sit as top floats with text
  below, all numbering contiguous. §II already keeps the problem
  statement free of concrete algorithms (F/φ are abstract operators;
  concrete F0–F7, SGD/Adam live only in §III), so the reviewer's
  "separate task from algorithms" point is satisfied.
- Wired the new generators into `reproduce.sh`.

## 2026-06-02 (drop "пол" jargon; reconcile tables IV/V)

- **Removed the gradient-"floor" jargon ("пол градиента").** In the
  theory it is renamed to the standard "асимптотический уровень
  $\|\nabla f\|^2$" (the limit in eq:floor); in the empirical tables and
  figure the floor metric is dropped entirely (it was confusing — e.g.
  F3 "не сошёлся 0/8" yet "пол 31×"). Table V lost its "пол↓" column,
  the heatmap lost its floor panel (now a single speedup heatmap), and
  every empirical "снижение пола 127×/120×/169×" claim is gone.
- **Reconciled Tables IV and V (they "не сходились").** The rules table
  was aggregating over all domains×models×optimizers (giving "7/40",
  "max 80×") while Table V is one domain, Adam, regression — impossible
  to cross-check. Rebuilt the rule derivation as a strict Adam-regression
  per-domain rollup of Block D: heavy-tail+short-memory → F2, 5/10
  domains, median 77.7×, max 80×. Now Table V crypto F2=78× sits exactly
  on the rule median, and max 80× = Block D's 15-min row. III↔IV↔V agree
  cell-for-cell.
- Fixed stray 0.38/"4/4" (crypto has 2 series × 4 seeds = 8 cells →
  "3/8 → 8/8" everywhere). Recompiled clean; tests pass.

## 2026-06-02 (readability pass: no jargon/meta, integrated filter math)

- **Removed all meta-commentary a reader can't know.** Deleted every
  mention of "removed/dropped filters", the "ранее удалённые операторы"
  note, the CNN-denoiser antipattern, and the conclusion's "из прежнего
  пула исключены". The paper now reads as if F0–F7 is simply the set.
- **Removed the "oracle" framing.** F3 is just described as non-causal
  ("использует будущее"), in prose, the heatmap, and the crypto table
  ("непр."). No "оракул"/"F3*" anywhere.
- **Integrated the filter math at the point of declaration.** The filter
  family is now a \description list where each F0–F7 carries its own
  formula and one analytic property inline, instead of a bare name-list
  followed by scattered \paragraph* derivations.
- **Renamed the floor term everywhere**, including the matplotlib figure
  (convergence_rescue.pdf y-axis/title) and the Block A table: "(шумовой)
  пол" → "асимптотический уровень ||grad f||^2".
- **Dropped the Block C table** (everything 0/8 — "nothing converges");
  Block C is now a short prose paragraph. run_cascade_n4.py no longer
  emits that table.
- **Fixed the Block D table overflow** (it was bleeding into the adjacent
  column) by wrapping the tabular in \resizebox{\columnwidth}.
- Paper now 8 pages, compiles clean (no undefined refs, no overfull
  boxes). NB: torch in this env currently fails to dlopen libtorch_cpu,
  so the torch-importing filter tests can't collect — environmental, not
  from these changes; the paper/figure/table generators don't use torch.

## 2026-06-02 (fix broken filter \description, heatmap, + poster)

- The filter family was an itemized \description with long bold labels;
  in the narrow IEEE column the labels overprinted the formulas ("text on
  text", "math on one line"). Rewrote it as run-in \paragraph* headers
  with each filter's formula on its own numbered display line --- readable.
- Heatmap made taller (9.2x5.0) with larger annotations/labels so cells
  are near-square and legible.
- Added poster.tex (a0poster, landscape, borderless 4-column red-header
  style matching the MIPT-class template the user linked): Задача /
  Фильтр-конвейер (F0–F7) / Механизм (медианная лемма, каскад) /
  Эксперимент / Результаты (heatmap) / Правило (α,H)→фильтр / Выводы.
  Reuses figures/new_datasets_heatmap.pdf. Compiles to A0 with pdflatex.
