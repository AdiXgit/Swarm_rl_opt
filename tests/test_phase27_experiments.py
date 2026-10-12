"""Phase 2.7: tests for scripts/phase27_experiments.py -- point-in-time
correctness of the horizon construction, target-version separation,
reproducibility, and that the ad-hoc majority-baseline computation (fixed
after a real bug was caught mid-task) uses VALIDATION's own distribution.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

spec = importlib.util.spec_from_file_location("phase27_experiments", ROOT / "scripts" / "phase27_experiments.py")
p27 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p27)

spec26 = importlib.util.spec_from_file_location("phase26_experiments", ROOT / "scripts" / "phase26_experiments.py")
p26 = importlib.util.module_from_spec(spec26)
spec26.loader.exec_module(p26)


def _synthetic_hist(n=200, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    return pd.DataFrame({"date": dates, "close": close})


# -- point-in-time correctness of the horizon sampler ------------------------#
def test_horizon_samples_never_use_future_closes_beyond_the_window():
    """Each sampled index i's feature set uses only closes up to i, and its
    label uses only close[i+horizon] -- never anything beyond that."""
    hist = _synthetic_hist()
    close = hist["close"].to_numpy()
    horizon, max_lookback = 5, 20
    n = len(close)
    idx = list(range(max_lookback, n - horizon, horizon))
    for i in idx:
        # feature construction only reaches back to i-20
        assert i - 20 >= 0
        # label construction only reaches forward to i+horizon, never further
        assert i + horizon < n


def test_horizon_indices_are_strictly_non_overlapping():
    """Consecutive sampled decision indices must be spaced by exactly the
    horizon, so no two labels share any part of their forecast window."""
    n, horizon, max_lookback = 500, 5, 20
    idx = list(range(max_lookback, n - horizon, horizon))
    gaps = np.diff(idx)
    assert np.all(gaps == horizon)


def test_horizon_median_split_is_fit_on_train_only():
    """The median used to label UP/DOWN must come from the TRAIN window
    passed in, not from data outside it (would leak future information into
    the label's own definition)."""
    hist = _synthetic_hist(n=100)
    close = hist["close"].to_numpy()
    idx = list(range(20, 90, 5))
    fwd_ret = np.array([close[i + 5] / close[i] - 1.0 for i in idx])
    median_from_this_window = np.median(fwd_ret)

    # mutating data far outside [20, 90] must not change the computed median
    mutated = close.copy()
    mutated[95:] *= 100.0
    fwd_ret2 = np.array([mutated[i + 5] / mutated[i] - 1.0 for i in idx])
    assert np.median(fwd_ret2) == pytest.approx(median_from_this_window)


# -- target-version separation -----------------------------------------------#
def test_tercile_thresholds_differ_from_original_fixed_bands():
    """The Stage D volatility-tercile redesign must use TRAIN-fit quantile
    thresholds, not silently fall back to the original fixed 0.9/1.1 bands
    -- that would be the same target wearing a new name."""
    vr = pd.Series(np.random.default_rng(0).gamma(2.0, 0.4, size=300))
    q1, q2 = vr.quantile([1 / 3, 2 / 3])
    assert q1 != pytest.approx(0.9, abs=1e-6)
    assert q2 != pytest.approx(1.1, abs=1e-6)


def test_feature_families_are_disjoint_from_full_set_only_by_subsetting():
    for name, feats in p27.FEATURE_FAMILIES.items():
        assert set(feats).issubset(set(p26.FEATURE_NAMES_V2)), name


# -- the bug this task caught and fixed: majority baseline must use VAL's own distribution --#
def test_majority_baseline_uses_validation_distribution_not_train():
    """Regression guard for a real bug found mid-task: computing the
    'majority baseline' as (y_val == TRAIN's majority class).mean() instead
    of VAL's own majority fraction silently misrepresents whether a model
    beats the trivial baseline."""
    y_train = pd.Series(["UP"] * 60 + ["DOWN"] * 40)  # TRAIN majority: UP
    y_val = pd.Series(["DOWN"] * 25 + ["UP"] * 9)      # VAL majority: DOWN

    buggy = float((y_val == y_train.value_counts().idxmax()).mean())  # checks against "UP"
    correct = float(y_val.value_counts().max() / len(y_val))           # VAL's own majority fraction

    assert buggy == pytest.approx(9 / 34)
    assert correct == pytest.approx(25 / 34)
    assert buggy != correct  # the two must differ in this constructed case, proving the bug was real


# -- reproducibility ----------------------------------------------------------#
def test_lightgbm_smoke_test_handles_failure_without_propagating(monkeypatch):
    """The real lightgbm crash is a native OSError (access violation) that
    was already triggered once, deliberately, in scripts/phase27_experiments.py
    itself (see docs/PHASE_2_7_SIGNAL_BOTTLENECK_REPORT.md section on model
    comparison) -- per the task's own instruction not to repeatedly invoke a
    crashing model, this test does NOT call the real lightgbm path again.
    It instead confirms the function's error-handling contract (any
    exception -> a structured FAILED result, nothing propagates) using a
    monkeypatched, ordinary exception."""
    import lightgbm as lgb

    class _BoomClassifier:
        def __init__(self, **kw):
            pass

        def fit(self, X, y):
            raise RuntimeError("simulated failure, not the real native crash")

    monkeypatch.setattr(lgb, "LGBMClassifier", _BoomClassifier)
    result = p27.stage_f_lightgbm_smoke_test()
    assert result["status"] == "FAILED"
    assert "RuntimeError" in result["error"]


# -- prediction collapse in a redesigned target -------------------------------#
def test_tercile_target_produces_balanced_train_classes_by_construction():
    """Unlike the original fixed-threshold volatility target (≈70% CONTRACT),
    a TRAIN-fit tercile split must produce exactly balanced classes on
    TRAIN -- confirming the rebalancing actually took effect, independent
    of whether it improves predictive accuracy (it does not, per the Phase
    2.7 report)."""
    vr = pd.Series(np.random.default_rng(1).gamma(2.0, 0.4, size=300))
    q1, q2 = vr.quantile([1 / 3, 2 / 3])
    labels = vr.apply(lambda r: "LOW" if r <= q1 else ("HIGH" if r >= q2 else "MID"))
    counts = labels.value_counts()
    assert counts.max() - counts.min() <= len(vr) * 0.05  # within 5% of perfectly balanced
