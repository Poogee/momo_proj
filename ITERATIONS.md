# Innovation log

What was added on top of the proposal's baseline (which itself is fully
delivered: synthetic factorial F0-F4 × N1-N4 × {SGD, Adam, AdamW},
real-data sweep on 9 tickers, Russian Typst report, 57 tests).

| # | Title | Tangible artifact | Headline result |
|---|-------|-------------------|-----------------|
| 1 | Data-mode preprocessing alternative | `optim.py:preprocess_mode`, `runs/mode_compare/`, fig `mode_compare_*` | Buffer-mode F2 final ‖g‖² = 49 → data-mode = 0.77 (Gaussian noise). But data-mode catastrophically fails on N2/N4 (correlated bias). |
| 2 | F5 learnable CNN denoiser | `learnable.py`, `models/learnable_filter.pt`, `train_learnable_filter.py` | ~80k-param residual 1-D CNN trained on uniform mixture of N1-N4. Solid generalist (always within 1 dB of best on each noise). |
| 3 | F6 adaptive wavelet, F7 hybrid median+wavelet | `filters.py:AdaptiveWaveletFilter, HybridMedianWaveletFilter` | F7 best on α-stable (+18.33 dB SNR vs F4's +17.95). |
| 4 | α-aware gradient clipping | `clipping.py:AlphaAwareClipper`, `runs/clipping/` | 5× floor reduction for SGD+F0 on N3 logistic (0.161 → 0.034). Adam already adapts so no benefit. |
| 5 | Full F0-F7 factorial sweep | `runs/synthetic_full/`, `tables/synthetic_full_summary.csv` | F7 8× lower floor than F0 on N3 logistic; +7pp holdout accuracy. |
| 6 | F8 adaptive meta-filter | `filters.py:AdaptiveMetaFilter` | Best average SNR across N1-N4 (11.89 dB) by routing on diff-series α̂ and Ĥ. |
| 7 | N5 regime-switch + N6 jump-diffusion noise | `noise.py:RegimeSwitchNoise, JumpDiffusionNoise` | F2 Kalman fails completely under N5 (0% convergence). F8 routes correctly. |
| 8 | Logistic sweep on N5/N6 | `runs/synthetic_extended/`, `tables/synthetic_extended_summary.csv` | F4/F7/F8 give +1pp holdout vs F0 on jump-diffusion. F2 catastrophic on regime switches. |
| 9 | Clipped-SGD + Normalized-SGD baselines | `optim.py:_clipped_sgd_step, _normalized_sgd_step` | Normalized-SGD F0 wins on heavy-tailed (0.013 final ‖g‖²) without any filter. |
| 10 | σ-scan: filter benefit vs noise scale | `figures/sigma_scan.pdf`, `tables/sigma_scan.csv` | F7/F8 give +6pp holdout at σ=2 on α-stable; filters hurt slightly on Gaussian. |
| 11 | Visual noise+filter showcase | `figures/noise_filter_examples.pdf` | One-page intuition for why median dominates on heavy-tailed. |
| 12 | Consolidated final recommendation table | `report.typ` Conclusion | Clear scenario → (filter, optimizer) → expected gain mapping. |
| 13 | Pareto qual-vs-time benchmark | `figures/filter_speed.pdf`, `tables/filter_speed.csv` | F4 at 0.20 ms gives 90% of F2's quality at 5× speed. F8 at 0.72 ms is mid-tier. |
| 14 | 2D Adam trajectories | `figures/2d_trajectory.pdf` | Visual: heavy-tailed noise causes Adam excursions; F4/F7 smooth them. |
| 15 | Spectral response analysis | `figures/spectral_response.pdf` | Empirical amplitude transfer of each filter to sinusoidal probes. |
| 16 | Convergence-rate verification | `figures/convergence_rate_alpha.pdf`, `tables/convergence_rates.csv` | SGD F0 DIVERGES at α=1.2 (slope +0.25). Norm-SGD/Clipped-SGD/F4 all stabilize. |
| 17 | Batch-size ablation | `figures/batch_size_ablation.pdf` | F7/F8 advantage stable +2-3pp across batch ∈ {4..256}. |
| 18 | Real-data 5-optimizer × 7-filter sweep | `tables/real_ho_full.csv`, `experiments/run_real_full_ho.py` | SGD/Clipped-SGD insensitive to filter choice; adaptive methods harmed by filtering on real returns. |
| 19 | ITERATIONS.md innovation log | `ITERATIONS.md` | One-line summary of each innovation with artifact path and headline. |
| 20 | F9 = 4× larger CNN denoiser | `models/learnable_filter_v2.pt`, `LearnableCNNFilterV2`, `tables/learnable_v1_vs_v2.csv` | F9 wins 5 of 6 noises; +5.4 dB on N4, +4.8 dB on N3, +2.8 dB on N1 over best hand-crafted. New universal recommendation when GPU is available. |

After iteration 18:
- 9 filters (F0–F8 with F5 learnable, F6 adaptive wavelet, F7 hybrid, F8 meta)
- 6 noise classes (N1–N6 with N5 regime-switch, N6 jump-diffusion)
- 5 optimizers (SGD, Clipped-SGD, Normalized-SGD, Adam, AdamW)
- ~73 tests passing
- ~26-page Typst report
- Full reproduction via `bash reproduce.sh`
