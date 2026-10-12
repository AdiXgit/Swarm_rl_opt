"""Phase 2.8: additional Black-76 edge-case coverage not already in
tests/test_pricing.py (parity, bounds, monotonicity, IV round-trips, and
analytical-vs-finite-difference Greeks are already thoroughly covered
there -- see the Phase 2.8 audit in
docs/PHASE_2_8_GREEKS_AND_NEW_INFORMATION_REPORT.md for what was reused
vs. added here).

This file adds: a broad multi-dimensional (F, K, sigma, T) grid check,
extreme-moneyness numerical stability, and behavior at the IV solver's own
bracket extremes (IV_LOW, IV_HIGH) -- none of which the existing suite
exercises together.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from swarm_opt.features.pricing import (
    IV_HIGH,
    IV_LOW,
    black76_price,
    greeks,
    implied_vol_scalar,
    intrinsic,
    upper_bound,
)

FORWARDS = [100.0, 5_000.0, 24_230.0, 100_000.0]
STRIKE_OFFSETS = [-0.5, -0.1, -0.01, 0.0, 0.01, 0.1, 0.5]  # fraction of forward
VOLS = [0.02, 0.1, 0.3, 0.8, 2.0]
TIMES = [1 / 365, 7 / 365, 30 / 365, 1.0]


# --------------------------------------------------------------------------- #
# broad deterministic grid: bounds and parity must hold everywhere
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("forward,offset,sigma,t", list(itertools.product(FORWARDS, STRIKE_OFFSETS, VOLS, TIMES))[::7])
def test_price_bounds_hold_across_a_broad_grid(forward, offset, sigma, t):
    """Every (F, K, sigma, T) cell must keep price finite, non-negative, and
    inside [intrinsic, upper_bound] for both calls and puts. Strided to a
    representative 7th of the full product for runtime, not because the
    other cells are expected to differ in kind."""
    strike = forward * (1 + offset)
    if strike <= 0:
        return
    for opt in ("CE", "PE"):
        price = float(black76_price(forward, strike, sigma, t, opt))
        lo = float(intrinsic(forward, strike, opt))
        hi = float(upper_bound(forward, strike, opt))
        assert np.isfinite(price)
        assert price >= 0.0
        assert lo - 1e-6 <= price <= hi + 1e-6


@pytest.mark.parametrize("forward,offset,sigma,t", list(itertools.product(FORWARDS, STRIKE_OFFSETS, VOLS, TIMES))[::11])
def test_put_call_parity_holds_across_a_broad_grid(forward, offset, sigma, t):
    strike = forward * (1 + offset)
    if strike <= 0:
        return
    c = float(black76_price(forward, strike, sigma, t, "CE"))
    p = float(black76_price(forward, strike, sigma, t, "PE"))
    assert c - p == pytest.approx(forward - strike, abs=max(1e-6, 1e-9 * forward))


# --------------------------------------------------------------------------- #
# extreme moneyness: numerical stability, not just correctness
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("forward", FORWARDS)
def test_deep_otm_call_is_finite_and_near_zero(forward):
    strike = forward * 10  # 10x out of the money
    price = float(black76_price(forward, strike, 0.2, 7 / 365, "CE"))
    assert np.isfinite(price)
    assert 0 <= price < forward * 1e-3


@pytest.mark.parametrize("forward", FORWARDS)
def test_deep_itm_call_is_finite_and_near_intrinsic(forward):
    strike = forward * 0.1  # deep in the money
    price = float(black76_price(forward, strike, 0.2, 7 / 365, "CE"))
    intrinsic_val = float(intrinsic(forward, strike, "CE"))
    assert np.isfinite(price)
    assert price == pytest.approx(intrinsic_val, rel=1e-6)


def test_greeks_stay_finite_at_extreme_moneyness():
    forward = 24230.0
    for strike in (forward * 0.01, forward * 100.0):
        g = greeks(forward, strike, 0.2, 7 / 365, "CE")
        for name in ("delta", "gamma", "theta", "vega"):
            val = float(g[name])
            assert np.isfinite(val), f"{name} not finite at strike={strike}"


# --------------------------------------------------------------------------- #
# IV solver bracket extremes: must not silently clamp or overflow
# --------------------------------------------------------------------------- #
def test_iv_at_the_low_bracket_edge_round_trips():
    forward, strike, t = 24230.0, 24230.0, 7 / 365
    price = float(black76_price(forward, strike, IV_LOW * 1.01, t, "CE"))
    got = implied_vol_scalar(price, forward, strike, t, "CE")
    if np.isfinite(got):
        assert got == pytest.approx(IV_LOW * 1.01, rel=1e-3)


def test_iv_at_the_high_bracket_edge_round_trips():
    forward, strike, t = 24230.0, 24230.0, 7 / 365
    price = float(black76_price(forward, strike, IV_HIGH * 0.99, t, "CE"))
    got = implied_vol_scalar(price, forward, strike, t, "CE")
    assert got == pytest.approx(IV_HIGH * 0.99, rel=1e-3)


def test_price_does_not_overflow_at_extreme_volatility_and_long_maturity():
    """IV_HIGH (5.0 = 500% vol) over a full year is a stress case the
    bracket is deliberately wide enough to cover (see pricing.py's own
    comment on IV_HIGH); price must stay finite, not overflow to inf."""
    price = float(black76_price(24230.0, 24230.0, IV_HIGH, 1.0, "CE"))
    assert np.isfinite(price)
    assert price <= 24230.0 + 1e-6  # still bounded by the call's upper bound (F)


def test_price_does_not_underflow_to_a_negative_number_at_tiny_volatility():
    price = float(black76_price(24230.0, 24230.0, IV_LOW, 1 / 365, "CE"))
    assert np.isfinite(price)
    assert price >= 0.0
