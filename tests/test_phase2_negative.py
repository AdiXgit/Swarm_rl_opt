"""Explicit negative-path checks from the Phase 2 spec section 12.F."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from swarm_opt.agents.base import AgentError, AgentPrediction
from swarm_opt.agents.direction import DirectionAgent
from swarm_opt.agents.greeks_agent import compute_greeks
from swarm_opt.agents.volatility import VolatilityAgent
from swarm_opt.features.audit import LeakageError, assert_no_cross_split_labels, assert_no_future_features
from swarm_opt.features.pointintime import FEATURE_NAMES, build_market_features
from swarm_opt.state import SwarmStateError, build_swarm_state, validate_swarm_state


def _preds_fixture():
    def pred(name, classes):
        probs = {c: 0.0 for c in classes}
        probs[classes[0]] = 1.0
        return AgentPrediction(
            agent_name=name, model_version="xgb_v1", observation_timestamp="2024-06-03T00:00:00",
            predicted_class=classes[0], probabilities=probs, confidence=1.0, data_quality={},
        )
    return {
        "direction": pred("direction", ["BEARISH", "NEUTRAL", "BULLISH"]),
        "volatility": pred("volatility", ["CONTRACT", "NEUTRAL", "EXPAND"]),
        "regime": pred("regime", ["TRENDING_DOWN", "TRENDING_UP", "RANGE_BOUND", "HIGH_VOLATILITY", "LOW_VOLATILITY"]),
    }


def test_future_derived_feature_injection_is_rejected():
    hist = pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=30),
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
    })
    feats, _ = build_market_features(hist)
    feats["forward_looking_leak"] = 1.0
    with pytest.raises(LeakageError):
        assert_no_future_features(feats)


def test_label_horizon_crossing_split_boundary_is_rejected():
    targets = pd.DataFrame({
        "decision_date": [date(2025, 8, 29)],  # last VALIDATION trading date
        "exit_date": [date(2025, 9, 2)],       # resolves into TEST
    })
    with pytest.raises(LeakageError):
        assert_no_cross_split_labels(targets)


def test_required_columns_absent_at_predict_time_raises():
    X = pd.DataFrame(
        np.random.default_rng(0).normal(size=(10, len(FEATURE_NAMES))), columns=FEATURE_NAMES
    )
    y = pd.Series(["BULLISH", "NEUTRAL"] * 5)
    ts = pd.Series(pd.bdate_range("2024-01-01", periods=10))
    agent = DirectionAgent(seed=0, params={"n_estimators": 5, "max_depth": 2})
    agent.fit(X, y, "fsv1", ("2024-01-01", "2024-03-01"))
    with pytest.raises(AgentError):
        agent.predict(pd.DataFrame({"only_one_col": [1, 2, 3]}), ts.iloc[:3])


def test_agent_returning_invalid_probabilities_is_rejected():
    bad = AgentPrediction(
        agent_name="direction", model_version="xgb_v1", observation_timestamp="2024-01-01T00:00:00",
        predicted_class="BULLISH", probabilities={"BULLISH": 1.5, "NEUTRAL": -0.5}, confidence=0.1,
        data_quality={},
    )
    with pytest.raises(AgentError):
        bad.validate()


def test_agent_timestamps_disagreeing_is_caught_by_state_validation():
    preds = _preds_fixture()
    probs = preds["regime"].probabilities
    preds["regime"] = AgentPrediction(
        agent_name="regime", model_version="xgb_v1", observation_timestamp="1999-01-01T00:00:00",
        predicted_class=preds["regime"].predicted_class, probabilities=probs, confidence=1.0,
        data_quality={},
    )
    with pytest.raises(SwarmStateError):
        build_swarm_state("2024-06-03", "NIFTY", {}, preds, None, "pointintime_v1")


def test_model_artifacts_missing_is_a_clear_failure(tmp_path):
    with pytest.raises(AgentError):
        VolatilityAgent.load(tmp_path / "nothing_here")


def test_unsupported_schema_version_is_an_explicit_invalid_state():
    state = build_swarm_state("2024-06-03", "NIFTY", {}, _preds_fixture(), None, "pointintime_v1")
    broken = state.to_dict()
    broken["schema_version"] = "SWARM_STATE_V0"
    errors = validate_swarm_state(type(state).from_dict(broken))
    assert errors  # explicit invalid state, not a silently-accepted one


def test_greeks_cannot_be_computed_is_an_explicit_absence_not_an_exception():
    empty_chain = pd.DataFrame(columns=["days_to_expiry", "expiry", "strike", "spot", "option_type", "close"])
    result = compute_greeks(empty_chain, pd.Timestamp("2024-06-03"), "NIFTY")
    assert result.atm_call is None and result.atm_put is None
    assert result.portfolio is None and result.portfolio_valid is False


def test_training_split_lacking_a_required_class_degrades_explicitly():
    X = pd.DataFrame(
        np.random.default_rng(0).normal(size=(10, len(FEATURE_NAMES))), columns=FEATURE_NAMES
    )
    y = pd.Series(["NEUTRAL"] * 10)  # BULLISH and BEARISH entirely absent
    ts = pd.Series(pd.bdate_range("2024-01-01", periods=10))
    agent = DirectionAgent(seed=0, params={"n_estimators": 5, "max_depth": 2})
    agent.fit(X, y, "fsv1", ("2024-01-01", "2024-03-01"))
    assert set(agent._missing_classes_in_fit) == {"BULLISH", "BEARISH"}
    preds = agent.predict(X, ts)
    for p in preds:
        p.validate()
