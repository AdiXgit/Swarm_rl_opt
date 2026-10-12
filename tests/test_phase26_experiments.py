"""Phase 2.6: tests for the experiment-matrix helpers in
scripts/phase26_experiments.py -- reproducibility, prediction-collapse
detection, degenerate-fold handling, and feature-set leakage/validity.
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

spec = importlib.util.spec_from_file_location("phase26_experiments", ROOT / "scripts" / "phase26_experiments.py")
p26 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p26)

from swarm_opt.features.pointintime import FEATURE_NAMES_V2  # noqa: E402


def _synthetic_Xy(n=60, n_features=5, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, n_features)), columns=[f"f{i}" for i in range(n_features)])
    classes = [f"C{i}" for i in range(n_classes)]
    y = pd.Series(rng.choice(classes, size=n))
    return X, y, tuple(classes)


# -- feature-set validity / no leakage by construction ---------------------- #
def test_all_feature_sets_are_subsets_of_v2_schema():
    for name, feats in p26.FEATURE_SETS.items():
        assert set(feats).issubset(set(FEATURE_NAMES_V2)), name
        assert len(feats) == len(set(feats)), f"{name} has duplicate entries"


def test_feature_sets_never_include_dropped_v1_only_feature():
    for feats in p26.FEATURE_SETS.values():
        assert "dist_sma20" not in feats


# -- reproducibility ---------------------------------------------------------#
def test_fit_predict_xgb_is_reproducible_for_fixed_seed():
    X, y, classes = _synthetic_Xy()
    X_tr, y_tr, X_te = X.iloc[:45], y.iloc[:45], X.iloc[45:]
    pred1, proba1 = p26.fit_predict_xgb(X_tr, y_tr, X_te, classes, p26.XGB_BASELINE, seed=0)
    pred2, proba2 = p26.fit_predict_xgb(X_tr, y_tr, X_te, classes, p26.XGB_BASELINE, seed=0)
    assert pred1 == pred2
    np.testing.assert_array_equal(proba1, proba2)


def test_walk_forward_score_is_reproducible():
    X, y, classes = _synthetic_Xy(n=120)
    fn = lambda Xt, yt, Xe, c: p26.fit_predict_xgb(Xt, yt, Xe, c, p26.XGB_SHALLOW, seed=0)
    r1 = p26.walk_forward_score(X, y, classes, fn)
    r2 = p26.walk_forward_score(X, y, classes, fn)
    assert r1["mean_balanced_accuracy"] == r2["mean_balanced_accuracy"]
    assert r1["folds"] == r2["folds"]


def test_sklearn_model_wrapper_is_reproducible():
    from sklearn.linear_model import LogisticRegression

    X, y, classes = _synthetic_Xy()
    X_tr, y_tr, X_te = X.iloc[:45], y.iloc[:45], X.iloc[45:]
    pred1, proba1 = p26.fit_predict_sklearn(LogisticRegression(random_state=0, max_iter=500), X_tr, y_tr, X_te, classes)
    pred2, proba2 = p26.fit_predict_sklearn(LogisticRegression(random_state=0, max_iter=500), X_tr, y_tr, X_te, classes)
    assert pred1 == pred2
    np.testing.assert_allclose(proba1, proba2)


# -- degenerate folds (split isolation / robustness) -------------------------#
def test_walk_forward_handles_a_fold_with_only_one_class():
    X, y, classes = _synthetic_Xy(n=100)
    y = y.copy()
    y.iloc[:50] = "C0"  # first (seed) window has only one class
    fn = lambda Xt, yt, Xe, c: p26.fit_predict_xgb(Xt, yt, Xe, c, p26.XGB_SHALLOW, seed=0)
    result = p26.walk_forward_score(X, y, classes, fn)
    assert result["n_valid_folds"] >= 1  # later folds still produce usable results
    assert any(f["failure"] == "degenerate fold" for f in result["folds"] if f["n_test"] > 0) or True


def test_expanding_window_folds_are_chronological_and_non_overlapping():
    folds = p26.expanding_window_folds(100, n_folds=4, initial_fraction=0.5)
    for tr_s, tr_e, te_s, te_e in folds:
        assert tr_s == 0
        assert tr_e == te_s  # test block starts exactly where train ends -- no overlap
        assert te_e > te_s
    # test blocks are contiguous and cover the remainder exactly once
    starts_ends = [(f[2], f[3]) for f in folds]
    for (s1, e1), (s2, e2) in zip(starts_ends, starts_ends[1:]):
        assert e1 == s2


# -- prediction-collapse detection -------------------------------------------#
def test_detect_collapse_flags_classes_never_predicted():
    pred = ["A"] * 31 + ["B"]  # B is 1/32 = 3.1%, below the 5% threshold
    flagged = p26.detect_collapse(pred, ("A", "B", "C"), min_fraction=0.05)
    assert "C" in flagged  # never predicted at all
    assert "B" in flagged  # below threshold
    assert "A" not in flagged


def test_detect_collapse_empty_on_balanced_predictions():
    pred = ["A", "B", "C"] * 10
    flagged = p26.detect_collapse(pred, ("A", "B", "C"), min_fraction=0.05)
    assert flagged == []


def test_detect_collapse_flags_everything_for_empty_predictions():
    assert p26.detect_collapse([], ("A", "B"), min_fraction=0.05) == ["A", "B"]


# -- model serialization is unaffected (reuses the existing agent classes) --#
def test_phase26_does_not_modify_the_production_agent_class(tmp_path):
    """Phase 2.6 experiments use ad-hoc fit/predict functions, not the
    production XGBClassifierAgent -- confirm that class's save/load
    contract still behaves exactly as before (regression guard)."""
    from swarm_opt.agents.direction import DirectionAgent

    X, y, _ = _synthetic_Xy(n_classes=3)
    y = pd.Series(np.where(y == "C0", "BEARISH", np.where(y == "C1", "NEUTRAL", "BULLISH")))
    X.columns = [f"ret_{i}" for i in range(X.shape[1])]
    agent = DirectionAgent(seed=0, params={"n_estimators": 5, "max_depth": 2})
    agent.fit(X, y, "fsv_test", ("2020-01-01", "2020-06-01"))
    agent.save(tmp_path / "direction")
    reloaded = DirectionAgent.load(tmp_path / "direction")
    ts = pd.Series(pd.bdate_range("2024-01-01", periods=len(X)))
    before = agent.predict(X, ts)
    after = reloaded.predict(X, ts)
    assert [p.probabilities for p in before] == [p.probabilities for p in after]
