# momo — denoising as preprocessing for stochastic optimization

Empirical study of denoising-as-preprocessing for stochastic optimization on
financial time series. The original brief implements

  (filter F0..F4) × (noise N1..N4) × (optimizer ∈ {SGD, Adam, AdamW})

on synthetic tasks (quadratic + logistic) and a basket of real financial
series. Through 24 innovation iterations the project was extended to:

  10 filters (F0..F9, including learned CNN denoisers and an adaptive meta-filter)
  × 6 noise classes (N1..N6, adding regime-switch and jump-diffusion)
  × 5 optimizers (SGD, Clipped-SGD, Normalized-SGD, Adam, AdamW)

with walk-forward evaluation on real returns and realized-volatility, sigma
scans, batch-size and convergence-rate ablations, and a 30-page Typst report.

## Headline findings

| Question | Answer |
|---|---|
| Best universal filter on synthetic noise (avg over N1..N6) | **F9 (4×-larger learned CNN), +17.3 dB SNR** |
| Best classical hand-crafted filter | F2 Kalman, 13.9 dB avg |
| Best hand-crafted on heavy-tailed noise | F4 Median or F7 Hybrid (+17-18 dB on N3) |
| Best filter on real daily log-returns | **F0 (no filtering)** — daily returns have no smooth signal to recover |
| Pure SGD under heavy-tailed (α=1.2) | DIVERGES; rescued by clipping, normalization, or filtering |
| Filter type catastrophically failing on regime-switch | F2 Kalman (0% convergence on N5 with Adam) |
| Filter that reduces F0+SGD floor 5× on N3 | AlphaAwareClipper (no filter needed) |

Full result table: `report.pdf` Conclusion section. Per-iteration log:
`ITERATIONS.md`.

## Layout

```
src/momo/
  noise.py       N1 Gaussian, N2 FARIMA pink, N3 α-stable, N4 mixed,
                 N5 regime-switch, N6 jump-diffusion
  filters.py     F0 Identity, F1 MA, F2 Kalman, F3 Wavelet, F4 Median,
                 F6 AdaptiveWavelet, F7 HybridMedianWavelet, F8 AdaptiveMeta
  learnable.py   F5 LearnableCNNFilter (small), F9 LearnableCNNFilterV2 (large)
  metrics.py     SNR, Hurst R/S + DFA, Hill α, McCulloch α, time-to-eps
  tasks.py       QuadraticTask, LogisticTask
  optim.py       SGD, Clipped-SGD, Normalized-SGD, Adam, AdamW; buffer/data preprocess modes
  clipping.py    AlphaAwareClipper (online α̂ → adaptive threshold)
  data.py        yfinance pipeline, walk-forward splits, AR(p) ForecastTask
  contracts.py   shared dataclasses

experiments/
  configs/                   YAML for sweeps
  fetch_data.py              one-off data pull
  run_filter_diagnostics.py  10×6 SNR table on synthetic
  run_synthetic_sweep.py     full F×N×optimizer factorial on synthetic
  run_real_sweep.py          F×optimizer × ticker on real (no walk-forward)
  run_real_walkforward.py    F×optimizer × ticker × split (causal)
  run_real_full_ho.py        + Clipped/Normalized SGD baselines
  run_volatility_forecast.py F×optimizer × ticker on realized vol
  run_mode_comparison_sweep.py  buffer-vs-data preprocessing mode
  run_clipping_ablation.py   clip-on vs clip-off on heavy-tailed
  run_sigma_scan.py          holdout vs σ scan
  run_batch_size_ablation.py F×batch_size
  run_filter_speed_benchmark.py  Pareto qual-vs-time
  verify_convergence_rate.py empirical slopes log||g||² vs log k
  train_learnable_filter.py  F5 / F9 training (CNN denoiser)
  compare_learnable_v1_v2.py F5 vs F9 head-to-head
  make_*.py                  figure builders

tests/                       73 pytest tests
runs/                        raw .npz per run, indexed in INDEX.md
tables/                      summary CSVs
figures/                     30 PDF figures used in the report
data/cache/                  yfinance parquet cache
models/                      trained CNN weights (F5: 0.34 MB, F9: 3.3 MB)

report.typ / report.pdf      Russian, ~30 pages, IEEE/proposal style
refs.bib                     bibliography (24 entries incl. 2020-2026 arXiv)
ITERATIONS.md                one-liner per innovation
DECISIONS.md                 engineering choices log
NOTICE                       third-party licenses
```

## Setup

The system is built and tested on Python 3.12 with PyTorch 2.10 / CUDA 12.8
already installed. Install the remaining dependencies:

```bash
pip install --break-system-packages \
  pywavelets pyyaml filterpy hurst joblib pyarrow yfinance
```

(The `--break-system-packages` flag is needed only on system-Python installs
governed by PEP 668. Use a virtualenv if you prefer; `pyproject.toml` lists
the full dependency set.)

## Reproduce baseline (no innovations)

```bash
bash reproduce.sh
```

## Reproduce specific iterations

```bash
PYTHONPATH=src python3 -m pytest -q
PYTHONPATH=src python3 experiments/fetch_data.py
PYTHONPATH=src python3 experiments/run_filter_diagnostics.py
PYTHONPATH=src python3 experiments/run_synthetic_sweep.py --config experiments/configs/synthetic_full.yaml
PYTHONPATH=src python3 experiments/run_real_walkforward.py
PYTHONPATH=src python3 experiments/run_clipping_ablation.py
PYTHONPATH=src python3 experiments/run_sigma_scan.py
# F5 / F9 training (GPU recommended):
PYTHONPATH=src python3 experiments/train_learnable_filter.py --steps 40000
PYTHONPATH=src python3 experiments/train_learnable_filter.py \
    --steps 60000 --channels 96 --blocks 4 --kernel 11 --batch 96 \
    --out models/learnable_filter_v2.pt
PYTHONPATH=src python3 experiments/make_figures.py --runs-dir runs/synthetic_full --summary-csv tables/synthetic_full_summary.csv
PYTHONPATH=src python3 experiments/make_master_heatmap.py
typst compile report.typ report.pdf
```

## Hardware notes

Reference workstation: RTX 2070 SUPER (8 GB) + Ryzen 7 5800X3D (16 threads)
+ 32 GB RAM. Most sweeps are CPU-bound on numpy; GPU is used only for F5/F9
training and inference.

| Step | Time |
|---|---|
| Tests | ~10 s |
| Synthetic factorial (600 cells, 5 seeds, F0-F4) | ~5 min |
| Synthetic factorial (840 cells, 5 seeds, F0-F4 + F6-F8) | ~6 min |
| Real walk-forward (2835 cells) | ~50 s |
| Filter diagnostics (10 filters × 4 noises × 10 seeds) | ~30 s |
| F5 training (40k steps) | ~10 min on RTX 2070 |
| F9 training (60k steps, 4× larger model) | ~33 min on RTX 2070 |
| Master SNR heatmap (10 × 6 × 10 seeds) | ~30 s |
| Typst compile | ~1 s |

## What's in the report

`report.typ` mirrors the proposal sections (Введение, Постановка, Методы,
Эксперименты, Результаты, Обсуждение, Заключение) and adds a large
*Дополнительные эксперименты* section covering the 24 innovation iterations.
The conclusion holds the final scenario→recommendation mapping.

## License / attribution

Third-party libraries are listed in `NOTICE`. Methodology references are in
`refs.bib`.
