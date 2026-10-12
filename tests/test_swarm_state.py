from __future__ import annotations

import json

import pytest

from swarm_opt.agents.base import AgentPrediction
from swarm_opt.agents.greeks_agent import GreeksAgentResult
from swarm_opt.state import SwarmStateError, build_swarm_state, validate_swarm_state


def _pred(name, classes, ts="2024-06-03T00:00:00"):
    probs = {c: (1.0 if i == 0 else 0.0) for i, c in enumerate(classes)}
    probs[classes[0]] = 1.0
    return AgentPrediction(
        agent_name=name, model_version="xgb_v1", observation_timestamp=ts,
        predicted_class=classes[0], probabilities=probs, confidence=1.0,
        data_quality={"had_missing_feature": False},
    )


def _preds(ts="2024-06-03T00:00:00"):
    return {
        "direction": _pred("direction", ["BEARISH", "NEUTRAL", "BULLISH"], ts),
        "volatility": _pred("volatility", ["CONTRACT", "NEUTRAL", "EXPAND"], ts),
        "regime": _pred("regime", ["TRENDING_DOWN", "TRENDING_UP", "RANGE_BOUND", "HIGH_VOLATILITY", "LOW_VOLATILITY"], ts),
    }


NO_GREEKS = GreeksAgentResult(
    agent_name="greeks", model_version="black76_v1", observation_timestamp="2024-06-03T00:00:00",
    underlying="NIFTY", atm_call=None, atm_put=None, portfolio=None,
    portfolio_valid=False, portfolio_reason="no portfolio constructed in Phase 2",
)

MARKET_FEATURES = {"ret_1": 0.01, "has_full_lookback": True, "atm_iv_available": False}


def test_build_swarm_state_valid():
    state = build_swarm_state("2024-06-03", "NIFTY", MARKET_FEATURES, _preds(), NO_GREEKS, "pointintime_v1")
    assert state.schema_version == "SWARM_STATE_V1"
    assert validate_swarm_state(state) == []


def test_build_swarm_state_missing_agent_raises():
    preds = _preds()
    del preds["regime"]
    with pytest.raises(SwarmStateError):
        build_swarm_state("2024-06-03", "NIFTY", MARKET_FEATURES, preds, NO_GREEKS, "pointintime_v1")


def test_build_swarm_state_rejects_mismatched_agent_timestamps():
    preds = _preds()
    preds["regime"] = _pred("regime", ["TRENDING_DOWN", "TRENDING_UP", "RANGE_BOUND", "HIGH_VOLATILITY", "LOW_VOLATILITY"], ts="2024-06-04T00:00:00")
    with pytest.raises(SwarmStateError):
        build_swarm_state("2024-06-03", "NIFTY", MARKET_FEATURES, preds, NO_GREEKS, "pointintime_v1")


def test_validate_catches_invalid_probabilities_without_raising():
    state = build_swarm_state("2024-06-03", "NIFTY", MARKET_FEATURES, _preds(), NO_GREEKS, "pointintime_v1")
    broken = state.to_dict()
    broken["direction"]["probabilities"] = {"BEARISH": 0.9, "NEUTRAL": 0.9, "BULLISH": 0.9}
    errors = validate_swarm_state(type(state).from_dict(broken))
    assert any("sum to" in e for e in errors)


def test_validate_catches_missing_required_field():
    state = build_swarm_state("2024-06-03", "NIFTY", MARKET_FEATURES, _preds(), NO_GREEKS, "pointintime_v1")
    broken = state.to_dict()
    del broken["direction"]["predicted_class"]
    errors = validate_swarm_state(type(state).from_dict(broken))
    assert any("missing field" in e for e in errors)


def test_validate_catches_incompatible_schema_version():
    state = build_swarm_state("2024-06-03", "NIFTY", MARKET_FEATURES, _preds(), NO_GREEKS, "pointintime_v1")
    broken = state.to_dict()
    broken["schema_version"] = "SWARM_STATE_V2"
    errors = validate_swarm_state(type(state).from_dict(broken))
    assert any("schema_version" in e for e in errors)


def test_validate_catches_nonnull_portfolio():
    state = build_swarm_state("2024-06-03", "NIFTY", MARKET_FEATURES, _preds(), NO_GREEKS, "pointintime_v1")
    broken = state.to_dict()
    broken["portfolio"] = {"capital": 100000}
    errors = validate_swarm_state(type(state).from_dict(broken))
    assert any("portfolio" in e for e in errors)


def test_validate_catches_infinite_market_feature():
    state = build_swarm_state("2024-06-03", "NIFTY", MARKET_FEATURES, _preds(), NO_GREEKS, "pointintime_v1")
    broken = state.to_dict()
    broken["market_features"]["ret_1"] = float("inf")
    errors = validate_swarm_state(type(state).from_dict(broken))
    assert any("infinite" in e for e in errors)


def test_serialization_round_trip_preserves_meaning():
    state = build_swarm_state("2024-06-03", "NIFTY", MARKET_FEATURES, _preds(), NO_GREEKS, "pointintime_v1")
    text = json.dumps(state.to_dict(), default=str)
    restored = type(state).from_dict(json.loads(text))
    assert restored == state


def test_greeks_unavailable_is_reflected_in_data_quality():
    state = build_swarm_state("2024-06-03", "NIFTY", MARKET_FEATURES, _preds(), None, "pointintime_v1")
    assert state.greeks is None
    assert state.data_quality["greeks_available"] is False
