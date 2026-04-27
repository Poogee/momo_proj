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
