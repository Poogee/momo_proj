# momo — denoising as preprocessing for stochastic optimization

Empirical study of denoising-as-preprocessing for stochastic optimization on
financial time series. Implements the full factorial

  (filter F0..F4) × (noise N1..N4) × (optimizer ∈ {SGD, Adam, AdamW})

on synthetic tasks (quadratic + logistic) and a basket of real financial
series. Measures convergence, holdout quality, SNR, and structural distortion
(Hurst, tail index). Code, configs, raw run artifacts, figures, tables, and a
Russian Typst report are included.

## Layout

```
src/momo/        — package: noise, filters, metrics, tasks, optim, data
experiments/     — run_synthetic_sweep.py, run_real_sweep.py,
                   run_filter_diagnostics.py, fetch_data.py, make_*.py
experiments/configs/ — synthetic.yaml, real.yaml
tests/           — pytest suite (57 tests)
runs/            — raw .npz per run (synthetic/, real/)
tables/          — synthetic_summary.csv, real_summary.csv,
                   filter_diagnostics.csv, recommendations.csv
figures/         — every PDF figure used in the report
data/cache/      — yfinance cache (parquet)
report.typ       — Typst source
report.pdf       — compiled report
refs.bib         — bibliography
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

## Reproduce all results

```bash
PYTHONPATH=src python3 -m pytest -q                                    # 57 tests
PYTHONPATH=src python3 experiments/fetch_data.py                       # populate cache
PYTHONPATH=src python3 experiments/run_filter_diagnostics.py           # tables/filter_diagnostics.csv
PYTHONPATH=src python3 experiments/run_synthetic_sweep.py --n-jobs 12  # ~5 min
PYTHONPATH=src python3 experiments/run_real_sweep.py     --n-jobs 12   # ~30 s
PYTHONPATH=src python3 experiments/make_figures.py                     # figures/
PYTHONPATH=src python3 experiments/make_real_figures.py
typst compile report.typ report.pdf
```

Or simply: `bash reproduce.sh`.

## Hardware notes

Designed for a single workstation. Reference run on RTX 2070 SUPER + Ryzen 7
5800X3D + 32 GB RAM:
- Synthetic factorial (600 cells, 5 seeds each): ~5 min on 12 CPU cores via joblib.
- Real-data sweep (675 cells): ~25 s.
- Filter diagnostics (40 cells, 10 seeds): ~10 s.
- Tests: ~10 s.

The current pipeline is CPU-bound on numpy; GPU is reserved for the learnable-
filter and forecaster ablations in the innovation loop.

## Configuration

`experiments/configs/{synthetic,real}.yaml` enumerate filters, noise classes,
optimizers, hyperparameters, seeds. Edit + rerun to extend the design.

## What's in the report

`report.typ` mirrors the proposal sections (Введение, Постановка, Методы,
Эксперименты, Результаты, Обсуждение, Заключение, Дополнительные
эксперименты). Headline findings are the SNR diagnostic matrix
(`figures/filter_snr_heatmap.pdf`), per-task convergence grids, T(ε) heatmaps,
and the recommendation table in `tables/recommendations.csv`.

## License / attribution

Third-party libraries are listed in `NOTICE`. Methodology references are in
`refs.bib`.
