# momo — denoising as preprocessing for stochastic optimization on financial time series

НИУ ВШЭ project · Russian Typst report (`report.pdf`, 41 pages) · Reproducible code · GitHub: [Poogee/momo_proj](https://github.com/Poogee/momo_proj)

The original brief was the factorial

  (filter F0..F4) × (noise N1..N4) × (optimizer ∈ {SGD, Adam, AdamW})

on synthetic tasks (quadratic + logistic) and a basket of real financial
series. Through 44 innovation iterations the project was extended to:

  10 filters (F0..F9 + ensemble FE, including learned CNN denoisers and an adaptive meta-filter)
  × 7 noise classes (N1..N7, adding regime-switch, jump-diffusion, Hawkes-clustered)
  × 5 optimizers (SGD, Clipped-SGD, Normalized-SGD, Adam, AdamW)

with walk-forward evaluation on real returns (magnitude, sign, volatility,
multi-step), σ scans, batch-size and convergence-rate ablations,
window-sensitivity studies, theoretical sketch, and a critical
lookahead-bias correction.

## Headline finding (positive, reproducible)

**Filtering the observed series rescues stochastic-optimization
convergence under heavy-tailed gradient noise.** At a tail index
calibrated to real series (α̂≈1.2; SPY α̂=1.05, Hurst|r|=0.86), plain
SGD on the quadratic and logistic tasks converges in **0 of 8 seeds**;
with a causal median/hybrid/online filter (F4/F7/FA) it converges in
**8 of 8**, and the noise floor is **17–186× lower** across regression,
convex & non-convex classification, and autoregression. The effect
persists at calibrated tails (91–158×). Adam/AdamW gain **11.2×** from
wavelet preprocessing on long-memory noise. On a genuinely real,
fully-causal task (15-min returns) Adam converges in 7/16 runs without
a filter vs **16/16, ≈78× faster** with a causal Kalman filter, holdout
unchanged. Honest negative zones (clean Gaussian noise, raw daily
returns, mixed long-memory, ETT holdout) are reported in full.
See `report.pdf`; experiment `experiments/run_convergence_rescue.py`,
tables `tables/convergence_rescue*.csv`,
figure `figures/convergence_rescue.pdf`.

## Other findings

| Question | Answer |
|---|---|
| Best universal filter on synthetic noise (avg over N1..N6) | **F9 (4×-larger learned CNN), +17.3 dB SNR** |
| Best classical hand-crafted filter | F2 Kalman, 13.9 dB avg |
| Best hand-crafted on heavy-tailed noise | F4 Median or F7 Hybrid (+17–18 dB on N3) |
| Best filter on real daily log-returns (causal walk-forward) | **F0 (no filtering)** — daily returns have no smooth signal to recover |
| Pure SGD under heavy-tailed (α=1.2) | DIVERGES (slope log‖g‖² vs log k = +0.25); rescued by clipping, normalization, or filtering |
| Filter type catastrophically failing on regime-switch (N5) | F2 Kalman (0% convergence on Adam) |
| Filter that reduces F0+SGD floor 5× on N3 | AlphaAwareClipper (no filter needed) |
| Sign prediction on real returns | F0 still wins after lookahead-bias correction (centered median was cheating) |

Full result table: `report.pdf` Conclusion section. Per-iteration log:
`ITERATIONS.md`. TL;DR figure: `figures/tldr_summary.pdf`.

## Project layout

```
src/momo/
  noise.py       N1 Gaussian, N2 FARIMA pink, N3 α-stable, N4 mixed,
                 N5 regime-switch, N6 jump-diffusion, N7 Hawkes-clustered
  filters.py     F0 Identity, F1 MA, F2 Kalman, F3 Wavelet, F4 Median,
                 F4c CausalMedian, F6 AdaptiveWavelet, F7 HybridMedianWavelet,
                 F8 AdaptiveMeta, FE EnsembleAverage
  learnable.py   F5 LearnableCNNFilter (small), F9 LearnableCNNFilterV2 (large)
  metrics.py     SNR, Hurst R/S + DFA, Hill α, McCulloch α, time-to-eps
  tasks.py       QuadraticTask, LogisticTask, MLPClassifierTask
  optim.py       SGD, Clipped-SGD, Normalized-SGD, Adam, AdamW;
                 buffer/data preprocess modes; AlphaAwareClipper hook
  clipping.py    AlphaAwareClipper (online α̂ → adaptive threshold)
  data.py        yfinance (daily + 1m/5m/15m/60m), FRED macro,
                 non-financial ETT, walk-forward splits, AR(p) ForecastTask
  contracts.py   shared dataclasses

experiments/
  configs/                          YAML for sweeps
  fetch_data.py                     one-off data pull
  run_filter_diagnostics.py         10-filter × 6-noise SNR table
  run_synthetic_sweep.py            full F×N×optimizer factorial on synthetic
  run_real_sweep.py                 simple F × optimizer × ticker on real
  run_real_walkforward.py           causal walk-forward (F1, F2 already causal)
  run_real_walkforward_causal.py    explicit-causal version (F4c trailing median)
  run_real_full_ho.py               + Clipped/Normalized SGD baselines
  run_real_sign_prediction.py       sign-prediction (centered F4 — has lookahead)
  run_signpred_causal.py            sign-prediction with causal F4c (corrects iter 28)
  run_signpred_window_sweep.py      F4 window scan for sign-pred
  run_volatility_forecast.py        F × optimizer × ticker on realized vol
  run_mode_comparison_sweep.py      buffer-vs-data preprocessing mode
  run_clipping_ablation.py          α-clip on/off on heavy-tailed
  run_sigma_scan.py                 holdout vs σ scan
  run_batch_size_ablation.py        F × batch_size
  run_filter_speed_benchmark.py     Pareto qual-vs-time
  run_window_sensitivity.py         F1/F4/F7 window scan
  run_meta_routing_sensitivity.py   F8 α-threshold scan
  run_stress_test.py                α ∈ [1.1, 1.9] stress
  run_multistep_horizon.py          h ∈ {1,5,10,20} forecast horizon
  run_cross_asset_transfer.py       train on one ticker, test on another
  verify_convergence_rate.py        empirical slopes log‖g‖² vs log k
  run_convergence_rescue.py         HEADLINE: filter rescues convergence
                                    (multi-metric, 4 models, 8 seeds)
  run_calibrated_synthetic.py       synthetic calibrated to real α̂/Ĥ
  run_applied_convergence.py        causal AR convergence across domains
                                    (financial / FRED macro / ETT sensor)
  train_learnable_filter.py         F5 / F9 training
  compare_learnable_v1_v2.py        F5 vs F9 head-to-head
  make_*.py                         figure builders (incl.
                                    make_convergence_figures.py)

tests/                              105 pytest tests (~12 s)
runs/                               raw .npz per run, indexed in INDEX.md
tables/                             21 summary CSVs
figures/                            34 PDF figures
data/cache/                         yfinance parquet cache
models/                             trained CNN weights (F5 0.34 MB, F9 3.3 MB)

report.typ / report.pdf             Russian, positive-result headline,
                                    clean math Постановка + subsections
refs.bib                            bibliography (verified, 1951–2026)
ITERATIONS.md                       one-liner per iteration
DECISIONS.md                        engineering / methodology log
NOTICE                              third-party licenses
reproduce.sh                        end-to-end reproduction script
```

## Setup

Tested on Python 3.12 with PyTorch 2.10 / CUDA 12.8 already installed.

```bash
pip install --break-system-packages \
  pywavelets pyyaml filterpy hurst joblib pyarrow yfinance
```

The `--break-system-packages` flag is needed only on system-Python installs
governed by PEP 668. For a clean venv use `pyproject.toml`.

## Reproduce all results

```bash
bash reproduce.sh
```

This runs the full 44-iteration pipeline: tests, data pull, synthetic
factorial, real-data walk-forward, all sensitivity scans, F5/F9 training
(if GPU available), all figure builders, and the Typst compile.

## Hardware notes

Reference workstation: **RTX 2070 SUPER (8 GB) + Ryzen 7 5800X3D (16 threads)
+ 32 GB RAM**. Most sweeps are CPU-bound on numpy; GPU is used only for
F5/F9 training and inference.

| Step | Time |
|---|---|
| Tests | ~10 s |
| Synthetic factorial (840 cells, 5 seeds, F0–F4 + F6–F8) | ~6 min |
| Real walk-forward (2835 cells) | ~50 s |
| Filter diagnostics (10 × 6 × 10 seeds) | ~30 s |
| F5 training (40k steps) | ~10 min on RTX 2070 |
| F9 training (60k steps, 4× larger) | ~33 min on RTX 2070 |
| Master SNR heatmap (10 × 6 × 10 seeds) | ~30 s |
| Stress test (500 cells) | ~2 min |
| Cross-asset transfer (1080 cells) | ~50 s |
| Convergence-rate verification (450 cells × 4000 steps) | ~4 min |
| Typst compile | ~1 s |

End-to-end reproduce.sh runtime ≈ 1–1.5 hours on the reference machine.

## What's in the report

`report.typ` mirrors the proposal sections (Введение, Постановка, Методы,
Эксперименты, Результаты, Обсуждение, Заключение) and adds a large
*Дополнительные эксперименты* section covering all 44 innovation
iterations. The conclusion holds the final scenario→recommendation
mapping and the TL;DR figure (`figures/tldr_summary.pdf`) sits in the
abstract for one-glance comprehension.

Important methodological note: iteration 40 corrected an earlier
positive sign-prediction result (iteration 28) — the original gain
turned out to be a lookahead-bias artifact from a centered median
filter. The causal `CausalMedianFilter` was added and the correction
documented openly in the report.

## License / attribution

Third-party libraries are listed in `NOTICE`. Methodology references are in
`refs.bib`. Code is the authors'; no Claude attribution requested or added.
