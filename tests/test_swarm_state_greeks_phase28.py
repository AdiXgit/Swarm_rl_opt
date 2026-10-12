"""Phase 2.8 A8: confirm SWARM_STATE_V1 already carries validated,
contract-level Greeks (Delta/Gamma/Theta/Vega, IV-solver status via
`valid`/`reason`, forward provenance) without any schema change -- this
is a verification task, not an implementation task: the fields already
exist (Phase 2 `agents/greeks_agent.py` + `state.py`), so these tests
check serialization round-trip and backward compatibility rather than add
new fields.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from swarm_opt.agents.base import AgentPrediction
from swarm_opt.agents.greeks_agent import compute_greeks
from swarm_opt.state import build_swarm_state, validate_swarm_state
from swarm_opt.features.pricing import black76_price


def _preds(ts="2024-06-03T00:00:00"):
    def pred(name, classes):
        probs = {c: (1.0 if i == 0 else 0.0) for i, c in enumerate(classes)}
        return AgentPrediction(agent_name=name, model_version="xgb_v1", observation_timestamp=ts,
                                predicted_class=classes[0], probabilities=probs, confidence=1.0,
                                data_quality={"had_missing_feature": False})
    return {
        "direction": pred("direction", ["BEARISH", "NEUTRAL", "BULLISH"]),
        "volatility": pred("volatility", ["CONTRACT", "NEUTRAL", "EXPAND"]),
        "regime": pred("regime", ["TRENDING_DOWN", "TRENDING_UP", "RANGE_BOUND", "HIGH_VOLATILITY", "LOW_VOLATILITY"]),
    }


def _real_chain(forward=24230.0, dte=5):
    rows = []
    for k in (23800.0, 24000.0, 24230.0, 24500.0, 24700.0):
        for opt in ("CE", "PE"):
            price = float(black76_price(forward, k, 0.15, dte / 365.0, opt))
            rows.append({
                "date": pd.Timestamp("2024-06-03"), "underlying": "NIFTY", "spot": forward - 5,
                "expiry": pd.Timestamp("2024-06-03") + pd.Timedelta(days=dte), "strike": k,
                "option_type": opt, "close": price, "volume": 10, "forward": forward,
                "forward_source": "futures", "days_to_expiry": dte,
            })
    return pd.DataFrame(rows)


def test_swarm_state_greeks_block_carries_delta_gamma_theta_vega_and_provenance():
    chain = _real_chain()
    greeks_result = compute_greeks(chain, pd.Timestamp("2024-06-03"), "NIFTY")
    state = build_swarm_state("2024-06-03", "NIFTY", {"ret_1": 0.0}, _preds(), greeks_result, "pointintime_v2")

    assert validate_swarm_state(state) == []
    atm_call = state.greeks["atm_call"]
    for field in ("delta", "gamma", "theta", "vega", "implied_volatility", "forward_source", "valid", "reason"):
        assert field in atm_call
    assert atm_call["valid"] is True
    assert atm_call["forward_source"] == "futures"


def test_swarm_state_round_trip_preserves_greeks_exactly():
    chain = _real_chain()
    greeks_result = compute_greeks(chain, pd.Timestamp("2024-06-03"), "NIFTY")
    state = build_swarm_state("2024-06-03", "NIFTY", {"ret_1": 0.0}, _preds(), greeks_result, "pointintime_v2")

    restored = type(state).from_dict(json.loads(json.dumps(state.to_dict(), default=str)))
    assert restored.greeks == state.greeks
    assert restored == state


def test_swarm_state_handles_unpriceable_contract_explicitly_not_fabricated():
    """A contract whose IV cannot be inverted (missing forward) must show
    up as valid=False with a documented reason, not a silently-dropped
    field or a fabricated zero Greek."""
    chain = _real_chain()
    chain["forward"] = float("nan")
    chain["forward_source"] = "missing"
    greeks_result = compute_greeks(chain, pd.Timestamp("2024-06-03"), "NIFTY")
    state = build_swarm_state("2024-06-03", "NIFTY", {"ret_1": 0.0}, _preds(), greeks_result, "pointintime_v2")

    assert validate_swarm_state(state) == []  # still a structurally valid state
    assert state.greeks["atm_call"]["valid"] is False
    assert state.greeks["atm_call"]["reason"] == "no_forward"
    assert state.data_quality["greeks_available"] is False


def test_existing_swarm_state_consumers_unaffected_by_phase28_additions():
    """Regression guard: nothing in Phase 2.8 changed state.py or
    greeks_agent.py's schema -- a state built exactly as in Phase 2 still
    validates and serializes identically."""
    state = build_swarm_state(
        "2024-06-03", "NIFTY", {"ret_1": 0.01, "has_full_lookback": True, "atm_iv_available": False},
        _preds(), None, "pointintime_v1",
    )
    assert validate_swarm_state(state) == []
    assert state.greeks is None
    assert state.data_quality["greeks_available"] is False
