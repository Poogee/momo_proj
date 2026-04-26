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
