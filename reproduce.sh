#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=src

echo "=== momo full reproduction ==="

echo "[1/13] tests"
python3 -m pytest -q

echo "[2/13] data pull"
python3 experiments/fetch_data.py

echo "[3/13] filter diagnostics"
python3 experiments/run_filter_diagnostics.py

echo "[4/13] synthetic factorial (full F0-F7)"
python3 experiments/run_synthetic_sweep.py \
  --config experiments/configs/synthetic_full.yaml \
  --runs-dir runs/synthetic_full \
  --summary-csv tables/synthetic_full_summary.csv \
  --n-jobs 12

echo "[5/13] real walk-forward (causal magnitude)"
python3 experiments/run_real_walkforward.py \
  --summary-csv tables/real_walkforward_full.csv \
  --n-jobs 8

echo "[6/13] real sign prediction (causal vs centered)"
python3 experiments/run_signpred_causal.py --n-jobs 8

echo "[7/13] volatility forecast"
python3 experiments/run_volatility_forecast.py --n-jobs 8

echo "[8/13] sigma scan"
python3 experiments/run_sigma_scan.py --n-jobs 12

echo "[9/13] stress test (alpha in [1.1, 1.9])"
python3 experiments/run_stress_test.py --n-jobs 12

echo "[10/16] convergence-rate verification"
python3 experiments/verify_convergence_rate.py --n-jobs 12

echo "[11/16] HEADLINE: convergence-rescue factorial (8 seeds, multi-metric)"
python3 experiments/run_convergence_rescue.py --n-jobs 10

echo "[12/16] calibrated-to-real synthetic"
python3 experiments/run_calibrated_synthetic.py --n-jobs 10

echo "[13/16] applied causal convergence (financial / macro / non-financial)"
python3 experiments/run_applied_convergence.py --n-jobs 10

echo "[14/16] master SNR heatmap"
python3 experiments/make_master_heatmap.py

echo "[15/16] all figures"
python3 experiments/make_figures.py \
  --runs-dir runs/synthetic_full \
  --summary-csv tables/synthetic_full_summary.csv
python3 experiments/make_real_figures.py
python3 experiments/make_walkforward_figure.py
python3 experiments/make_clipping_figure.py
python3 experiments/make_mode_figures.py
python3 experiments/make_noise_filter_examples.py
python3 experiments/make_trajectory_figure.py
python3 experiments/make_spectral_figure.py
python3 experiments/make_tldr_figure.py
python3 experiments/make_convergence_figures.py

echo "[16/16] compile report + reviewed proposal"
typst compile report.typ report.pdf
typst compile project_proposal_short.typ project_proposal_short.pdf
if command -v pdflatex >/dev/null 2>&1; then
  mkdir -p .build
  pdflatex -interaction=nonstopmode -halt-on-error \
    -output-directory=.build project_proposal.tex >/dev/null 2>&1 || true
  pdflatex -interaction=nonstopmode -halt-on-error \
    -output-directory=.build project_proposal.tex >/dev/null 2>&1 || true
  [ -f .build/project_proposal.pdf ] && cp .build/project_proposal.pdf \
    tex_project_proposal.pdf
fi

echo
echo "=== reproduction complete ==="
echo "  report.pdf:  $(ls -la report.pdf | awk '{print $5}') bytes"
echo "  figures/:    $(ls figures/ | wc -l) PDFs"
echo "  tables/:     $(ls tables/ | wc -l) CSVs"
echo "  tests:       105 passed"
