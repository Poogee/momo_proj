# Run index

Each cell directory holds one `.npz` per seed with arrays
`grad_norm_sq`, `loss`, `x_final`. Aggregated summaries live in `tables/`.

## Synthetic factorial

`experiments/run_synthetic_sweep.py`, config `experiments/configs/synthetic.yaml`.

- 2 tasks (quadratic, logistic)
- 5 filters × 4 noise classes × 3 optimizers × 5 seeds  ⇒  600 runs
- raw artifacts under `runs/synthetic/{task}/{filter}_{noise}_{optimizer}/seed{ss}.npz`
- summary CSV: `tables/synthetic_summary.csv` (1 row per run)
- aggregate clock time: ~5 min on 12-core CPU sweep

Headline (median ε-stop count, ε=0.01, missing means did not converge in `steps`):

| Task      | Best (filter, optimizer) | T(ε)  | Notes |
|-----------|--------------------------|-------|-------|
| logistic  | F4 median + SGD          |  559  | only filter to converge under N3/N4 |
| quadratic | F0 + SGD                 | n/c   | gradient-buffer filtering smears the signal trajectory |

## Real financial data

`experiments/run_real_sweep.py`, config `experiments/configs/real.yaml`.

- 9 tickers × 5 filters × 3 optimizers × 5 seeds  ⇒  675 runs
- raw artifacts under `runs/real/{ticker}/{filter}_{optimizer}/seed{ss}.npz`
- summary CSV: `tables/real_summary.csv` (1 row per run)
- aggregate clock time: 23 s

Headline:
- Filtering daily log-returns hurts holdout MSE (no smooth signal to recover).
- Linear filters (MA, Kalman, wavelet) Gaussianize the series (α: 1.18 → 1.6–1.8).
- Median filter preserves heavy tails (α: 1.18 → 0.93) while still smoothing.

## Filter diagnostics

`experiments/run_filter_diagnostics.py`. SNR and structural distortion of
F0–F4 on a clean smooth signal contaminated with each of N1–N4 at σ=0.5.
Output: `tables/filter_diagnostics.csv`. Recommendation matrix in
`tables/recommendations.csv`.
