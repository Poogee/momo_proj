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
| 21 | Housekeeping: conclusion table + ITERATIONS.md updated for F9 | `report.typ`, `ITERATIONS.md` | F9 added as new top recommendation when GPU available. |
| 22 | F9 on real data — synthetic gains don't transfer | `tables/real_walkforward_with_learnable.csv` | F9 → MSE 6.71 (worse than F0 = 3.63). High synthetic SNR ≠ better real-data forecast. |
| 23 | Volatility forecast (smoother target) | `experiments/run_volatility_forecast.py`, `tables/real_vol_forecast.csv` | F4 within 1% of F0; CNN F5/F9 break (out-of-distribution positive series). |
| 24 | Master SNR heatmap + global ranking | `figures/master_snr_heatmap.pdf`, `tables/master_snr_table.csv` | F9: 17.28 dB > F2 Kalman: 13.92 dB > F8 Meta: 13.48 dB > F4 Median: 13.15 dB. |
| 25 | README rewrite reflecting all innovations | `README.md` | Master ranking, layout, headline findings table. |
| 26 | Window-size sensitivity (F1, F4, F7) | `tables/window_sensitivity.csv`, `figures/window_sensitivity.pdf` | F7 most robust to window choice (18-22 dB on N3 across all windows). |
| 27 | F8 added to walk-forward | `tables/real_walkforward_full.csv` | F8 routes to F7 on real returns (α̂ ≈ 1.18); same MSE — adaptive routing right per SNR but wrong objective for forecast. |
| 28 | Sign prediction on real data | `experiments/run_real_sign_prediction.py`, `tables/real_sign_pred.csv` | **F4 Median +7.1 pp accuracy vs majority baseline** — first positive real-data result! Task choice (sign vs magnitude) is decisive. |
| 29 | Conclusion + recommendation table updated for F4 sign-pred win | `report.typ` | F4 + Adam now recommended for sign prediction on real data. |
| 30 | ITERATIONS.md catch-up | `ITERATIONS.md` | Coverage of 21-29. |
| 31 | Cross-asset transfer (F4 sign-pred train→test) | `experiments/run_cross_asset_transfer.py`, `tables/cross_asset_transfer.csv` | F4 win is asset-specific; cross-asset transfer fails for ALL filters. |
| 32 | Theoretical sketch (Robbins-Monro / Cutkosky / Defossez) | `report.typ` §32 | Filter rescues SGD via variance reduction or order statistics; matches empirical slopes. |
| 33 | N7 Hawkes-clustered jump noise | `noise.py:HawkesClusteredJumpNoise` | F2 Kalman wins +22.3 dB on this regime; tests added. |
| 34 | Test suite expansion for F6/F7/F8 + Hawkes | `tests/test_extended_filters.py` | +9 tests, 82 total. |
| 35 | Stress test α ∈ [1.1, 1.9] | `experiments/run_stress_test.py`, `figures/stress_test.pdf` | F4+Norm-SGD universal robust pair; SGD F0 catastrophic at α=1.1. |
| 36 | Multi-step horizon scan (h ∈ {1,5,10,20}) | `figures/multistep_horizon.pdf` | F4 catches F0 at h≥10, slightly beats it. |
| 37 | F4 window sensitivity for sign prediction | `figures/signpred_window.pdf` | w=3 best (+6 pp); robust across w∈[3,51]. |
| 38 | Ensemble FE = (F2+F4+F7)/3 | `EnsembleAverageFilter` | 13.22 dB avg, between F2 and F4; no win over best single. |
| 39 | F8 α-threshold sensitivity | `figures/meta_routing_sensitivity.pdf` | Robust 12.97-13.87 dB across [1.5, 1.95]; default near optimum. |
| 40 | **CORRECTION** F4 sign-pred +7pp was lookahead artifact | `CausalMedianFilter`, `figures/signpred_causal.pdf` | Causal median (no future) is *worse* than F0 (0.59 vs 0.60). Sign-pred conclusion reversed. |
| 41 | Magnitude forecast with fully-causal filters | `experiments/run_real_walkforward_causal.py` | F0 still wins; magnitude conclusion robust (F1, F2 were already causal). |
| 42 | Bug fix: `_to_returns` partial-failure handling | `src/momo/data.py` | dropna(how="any") → dropna(how="all"); single-ticker yfinance failures no longer wipe entire DataFrame. |
| 43 | ITERATIONS.md catch-up to iter 42 | `ITERATIONS.md` | Reflects lookahead correction and bugfix. |
| 44 | TL;DR summary figure (5-panel) | `experiments/make_tldr_figure.py`, `figures/tldr_summary.pdf` | Single-page abstract: SNR heatmap, real-data MSE bar, σ-scan, divergence vs α, sign-pred causal correction. |

| 45 | Literature fix + Anantharam–Borkar 2012 | `refs.bib`, `project_proposal.tex` | Chandak placeholder arXiv:2503.XXXXX → verified arXiv:2603.19648 (correct title/authors, checked vs articles PDF + arxiv.org); Gorbunov 2020 title corrected ("Accelerated Gradient Clipping"); Anantharam & Borkar 2012 (Queueing Syst. 71, doi 10.1007/s11134-012-9283-0) verified & cited; project positioned vs both. |
| 46 | New code: MLP task, FRED/non-financial loaders, convergence metrics | `tasks.py:MLPClassifierTask`, `data.py:fetch_fred/fetch_nonfinancial`, `metrics.py` | Non-convex classifier (analytic backprop, finite-diff tested); FRED (per-series fredgraph.csv) + ETT sensor loaders + 15m/60m intraday, cached + synthetic fallback; binary-conv/divergence-slope/AUC/floor-quantile metrics. 105 tests. |
| 47 | **HEADLINE** convergence-rescue factorial | `experiments/run_convergence_rescue.py`, `tables/convergence_rescue*.csv` | Heavy-tailed N3 (α=1.2) SGD: quadratic & logistic converge **0/8 seeds without filter → 8/8 with F4/F7/FA**, noise floor **17–186× lower** across regression/convex+non-convex classification/AR; Gaussian-control neutral (honest). Adam/AdamW: wavelet **11.2× faster** on long-memory N2; honest negatives on N4/logistic. |
| 48 | Calibrated synthetic + applied causal by domain | `run_calibrated_synthetic.py`, `run_applied_convergence.py` | Real diagnostics (SPY α̂=1.05, Ĥ\|r\|=0.86; median α̂=1.21): floor rescue persists **91–158×** at calibrated tails; mixed N4cal honest no-gain. Applied causal: financial 15-min Adam converges 7/16 (F0) → **16/16, ≈78× faster** with causal Kalman F2, holdout unchanged; ETT/FRED/daily negatives reported honestly. |
| 49 | Report rewrite + proposal Постановка restructure + reproduce.sh | `report.typ`/`report.pdf`, `project_proposal.tex`/`tex_project_proposal.pdf`, `reproduce.sh` | Positive convergence result is now the report headline (multi-metric/model/domain tables, CIs); clean math Постановка subsection + separate Шумы/Фильтры/Оптимизаторы/Метрики in report **and** reviewed proposal (reviewer note 1); practical scenario (note 3); reproduce.sh + README updated. |

After iteration 18:
- 9 filters (F0–F8 with F5 learnable, F6 adaptive wavelet, F7 hybrid, F8 meta)
- 6 noise classes (N1–N6 with N5 regime-switch, N6 jump-diffusion)
- 5 optimizers (SGD, Clipped-SGD, Normalized-SGD, Adam, AdamW)
- ~73 tests passing
- ~26-page Typst report
- Full reproduction via `bash reproduce.sh`
