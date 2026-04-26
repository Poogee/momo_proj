#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src

python3 -m pytest -q
python3 experiments/fetch_data.py
python3 experiments/run_filter_diagnostics.py
python3 experiments/run_synthetic_sweep.py --n-jobs 12
python3 experiments/run_real_sweep.py     --n-jobs 12
python3 experiments/make_figures.py
python3 experiments/make_real_figures.py
typst compile report.typ report.pdf
echo "Reproduced. Output: report.pdf, figures/, tables/."
