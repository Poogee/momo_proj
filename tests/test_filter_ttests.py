"""Unit tests for the paired statistical-test helpers used to back the
filtering hypotheses (experiments/run_filter_ttests.py)."""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest
from scipy import stats

# load the experiment script as a module (it is not an installed package)
_PATH = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "run_filter_ttests.py"
_spec = importlib.util.spec_from_file_location("run_filter_ttests", _PATH)
tt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tt)


def test_holm_monotone_and_matches_known_case():
    # Holm on a textbook vector; adjusted p must be nondecreasing in raw rank
    p = np.array([0.01, 0.04, 0.03, 0.005])
    adj = tt.holm(p)
    # smallest raw (0.005, m=4) -> 4*0.005 = 0.02
    assert adj[3] == pytest.approx(0.02, abs=1e-12)
    # adjusted p are valid probabilities and never below the raw values
    assert np.all(adj <= 1.0) and np.all(adj >= p)
    # step-down monotonicity when sorted by raw p
    order = np.argsort(p)
    assert np.all(np.diff(adj[order]) >= -1e-12)


def test_paired_floor_test_detects_large_gap():
    # F0 floor ~0.5, Fk floor ~0.005 -> ~100x, must be hugely significant
    f0 = np.array([0.28, 0.48, 0.29, 0.69, 0.40, 0.60, 0.76, 1.16])
    fk = np.array([0.0045, 0.0050, 0.0044, 0.0054, 0.0044, 0.0055, 0.0056, 0.0050])
    r = tt.paired_log_floor_test(f0, fk)
    assert r["floor_ratio"] > 50          # at least 50x lower floor
    assert r["p_one_sided"] < 1e-4        # strongly significant
    assert r["t_stat"] > 5
    assert r["cohen_d"] > 2               # very large effect
    assert r["ci_lo"] > 0                 # log-diff CI excludes zero


def test_paired_floor_test_null_on_equal_data():
    rng = np.random.default_rng(0)
    base = np.abs(rng.normal(0.5, 0.05, size=8)) + 0.1
    # identical floors -> no improvement -> not significant in the H1 direction
    r = tt.paired_log_floor_test(base, base.copy())
    assert r["floor_ratio"] == pytest.approx(1.0, abs=1e-9)
    assert r["p_one_sided"] > 0.4


def test_paired_floor_one_sided_agrees_with_scipy():
    f0 = np.array([0.28, 0.48, 0.29, 0.69, 0.40, 0.60, 0.76, 1.16])
    fk = np.array([0.05, 0.08, 0.04, 0.10, 0.06, 0.09, 0.11, 0.13])
    r = tt.paired_log_floor_test(f0, fk)
    d = np.log10(f0) - np.log10(fk)
    t_ref, p_two = stats.ttest_1samp(d, 0.0)
    assert r["t_stat"] == pytest.approx(t_ref, rel=1e-6)
    assert r["p_one_sided"] == pytest.approx(p_two / 2, rel=1e-6)


def test_bootstrap_ratio_ci_brackets_point_and_excludes_one():
    f0 = np.array([0.28, 0.48, 0.29, 0.69, 0.40, 0.60, 0.76, 1.16])
    fk = np.array([0.0045, 0.0050, 0.0044, 0.0054, 0.0044, 0.0055, 0.0056, 0.0050])
    point, lo, hi = tt.bootstrap_ratio_ci(f0, fk, n_boot=2000)
    assert lo < point < hi
    assert lo > 1.0                       # whole CI above 1 -> robust win
    # reproducible: same fixed seed -> identical CI
    again = tt.bootstrap_ratio_ci(f0, fk, n_boot=2000)
    assert again == (point, lo, hi)


def test_exact_sign_permutation_all_positive():
    # all-positive differences (n=8) -> minimum exact one-sided p = 1/256
    d = np.array([0.4, 0.6, 0.3, 0.7, 0.5, 0.6, 0.8, 1.2])
    assert tt.exact_sign_perm_p(d) == pytest.approx(1.0 / 256, abs=1e-12)
    # symmetric differences around zero -> p ~ 0.5
    d2 = np.array([0.5, -0.5, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1])
    assert tt.exact_sign_perm_p(d2) == pytest.approx(0.5, abs=0.06)


def test_power_and_mde_consistent():
    # huge observed effect -> power ~ 1
    assert tt.achieved_power(5.0, 8) > 0.999
    # the d that gives power 0.8 should round-trip through achieved_power
    mde = tt.min_detectable_d(8, power=0.8)
    assert 0.8 < mde < 1.2
    assert tt.achieved_power(mde, 8) == pytest.approx(0.8, abs=0.02)
    # more seeds -> smaller detectable effect
    assert tt.min_detectable_d(20, power=0.8) < tt.min_detectable_d(8, power=0.8)


def test_tost_equivalence_close_vs_far():
    rng = np.random.default_rng(1)
    # tiny, tight difference (|log diff| ~ 0.05 << margin 0.30) -> equivalent
    f0 = np.abs(rng.normal(0.5, 0.02, 8)) + 0.3
    fk = f0 * 10 ** rng.normal(0.05, 0.01, 8)
    r = tt.tost_equivalence(f0, fk, margin_log=0.30)
    assert r["equivalent"] is True
    assert r["p_tost"] < 0.05
    # a 100x gap is NOT equivalence (it's a huge real effect)
    far = tt.tost_equivalence(f0, f0 / 100.0, margin_log=0.30)
    assert far["equivalent"] is False
    assert far["p_tost"] > 0.05


def test_stouffer_combines_consistent_evidence():
    # three independent small one-sided p's should combine to a far smaller one
    z, p = tt.stouffer(np.array([0.01, 0.02, 0.03]))
    assert z > 0
    assert p < 0.001
    # symmetric null: p=0.5 each -> combined Z ~ 0, p ~ 0.5
    z0, p0 = tt.stouffer(np.array([0.5, 0.5, 0.5]))
    assert abs(z0) < 1e-9
    assert p0 == pytest.approx(0.5, abs=1e-9)
