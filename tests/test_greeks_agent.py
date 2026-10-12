from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swarm_opt.agents.greeks_agent import compute_greeks
from swarm_opt.features.pricing import black76_price


def _synthetic_chain(forward=100.0, dte=5, strikes=None, forward_source="futures"):
    strikes = strikes or [90, 95, 100, 105, 110]
    rows = []
    for k in strikes:
        for opt_type in ("CE", "PE"):
            price = float(black76_price(forward, k, 0.20, dte / 365.0, opt_type))
            rows.append({
                "date": pd.Timestamp("2024-06-03"),
                "underlying": "NIFTY",
                "spot": forward - 2.0,
                "expiry": pd.Timestamp("2024-06-03") + pd.Timedelta(days=dte),
                "strike": float(k),
                "option_type": opt_type,
                "close": price,
                "volume": 10,
                "forward": forward,
                "forward_source": forward_source,
                "days_to_expiry": dte,
            })
    return pd.DataFrame(rows)


def test_atm_greeks_are_finite_with_correct_signs():
    chain = _synthetic_chain()
    result = compute_greeks(chain, pd.Timestamp("2024-06-03"), "NIFTY")
    assert result.atm_call is not None and result.atm_call.valid
    assert result.atm_put is not None and result.atm_put.valid
    assert 0.0 < result.atm_call.delta < 1.0
    assert -1.0 < result.atm_put.delta < 0.0
    assert result.atm_call.gamma > 0 and result.atm_put.gamma > 0
    assert result.atm_call.vega > 0 and result.atm_put.vega > 0


def test_no_near_expiry_returns_none_not_fabricated():
    chain = _synthetic_chain(dte=60)  # outside MAX_DAYS_TO_EXPIRY
    result = compute_greeks(chain, pd.Timestamp("2024-06-03"), "NIFTY")
    assert result.atm_call is None
    assert result.atm_put is None


def test_portfolio_greeks_are_always_unavailable_in_phase2():
    chain = _synthetic_chain()
    result = compute_greeks(chain, pd.Timestamp("2024-06-03"), "NIFTY")
    assert result.portfolio is None
    assert result.portfolio_valid is False
    assert "Decision Engine" in result.portfolio_reason or "Trade Constructor" in result.portfolio_reason


def test_missing_forward_is_not_fabricated():
    chain = _synthetic_chain()
    chain["forward"] = np.nan
    chain["forward_source"] = "missing"
    result = compute_greeks(chain, pd.Timestamp("2024-06-03"), "NIFTY")
    assert result.atm_call.valid is False
    assert result.atm_call.reason == "no_forward"
    assert np.isnan(result.atm_call.implied_volatility)
    assert np.isnan(result.atm_call.delta)


def test_observation_timestamp_propagates():
    chain = _synthetic_chain()
    ts = pd.Timestamp("2024-06-03")
    result = compute_greeks(chain, ts, "NIFTY")
    assert result.observation_timestamp == ts.isoformat()
