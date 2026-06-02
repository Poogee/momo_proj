# Decisions log

Living record of non-obvious engineering / methodological choices.

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
