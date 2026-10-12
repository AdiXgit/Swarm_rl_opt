"""Phase 2.8: portfolio-level Greek aggregation tests.

Unit tests of the aggregation arithmetic are kept separate from the
finite-difference tests (which validate that the aggregated Greeks are a
locally correct linear approximation of a real multi-leg Black-76
portfolio's value, not just that the sums are arithmetically consistent
with themselves).
"""
from __future__ import annotations

import numpy as np
import pytest

from swarm_opt.agents.portfolio_greeks import (
    OptionPosition,
    PortfolioGreeksError,
    aggregate_greeks,
)
from swarm_opt.features.pricing import black76_price, greeks

F, T, SIG = 24230.0, 7 / 365, 0.14
LOT = 75


def _leg(option_type, strike, side, qty=1, lot=LOT, sigma=SIG, valid=True, forward=F, t=T):
    g = greeks(forward, strike, sigma, t, option_type)
    return OptionPosition(
        option_type=option_type, strike=strike, expiry="2024-06-20", side=side,
        quantity_contracts=qty, lot_size=lot,
        delta=float(g["delta"]), gamma=float(g["gamma"]), theta=float(g["theta"]), vega=float(g["vega"]),
        valid=valid,
    )


# --------------------------------------------------------------------------- #
# position-level construction and validation
# --------------------------------------------------------------------------- #
def test_invalid_side_raises():
    with pytest.raises(PortfolioGreeksError):
        _leg("CE", F, side="LEFT")


def test_nonpositive_quantity_raises():
    with pytest.raises(PortfolioGreeksError):
        _leg("CE", F, side="LONG", qty=0)


def test_nonpositive_lot_size_raises():
    with pytest.raises(PortfolioGreeksError):
        _leg("CE", F, side="LONG", lot=0)


def test_invalid_option_type_raises():
    with pytest.raises(PortfolioGreeksError):
        _leg("XX", F, side="LONG")


def test_signed_contracts_and_multiplier():
    long_leg = _leg("CE", F, side="LONG", qty=3)
    short_leg = _leg("CE", F, side="SHORT", qty=3)
    assert long_leg.signed_contracts == 3
    assert short_leg.signed_contracts == -3
    assert long_leg.multiplier == 3 * LOT
    assert short_leg.multiplier == -3 * LOT


def test_lot_size_applied_exactly_once():
    leg = _leg("CE", F, side="LONG", qty=2, lot=LOT)
    per_unit_delta = leg.delta
    assert leg.position_greek("delta") == pytest.approx(2 * LOT * per_unit_delta)
    # not lot*lot, not quantity*quantity
    assert leg.position_greek("delta") != pytest.approx(2 * LOT * LOT * per_unit_delta)


def test_reversing_side_reverses_signed_exposure_exactly():
    long_leg = _leg("CE", F + 100, side="LONG", qty=2)
    short_leg = _leg("CE", F + 100, side="SHORT", qty=2)
    for name in ("delta", "gamma", "theta", "vega"):
        assert long_leg.position_greek(name) == pytest.approx(-short_leg.position_greek(name))


def test_invalid_leg_excluded_not_zeroed(caplog=None):
    valid_leg = _leg("CE", F, side="LONG")
    invalid_leg = OptionPosition(
        option_type="PE", strike=F, expiry="2024-06-20", side="LONG", quantity_contracts=1,
        lot_size=LOT, delta=float("nan"), gamma=float("nan"), theta=float("nan"), vega=float("nan"),
        valid=False,
    )
    port = aggregate_greeks([valid_leg, invalid_leg])
    assert port.n_legs == 2 and port.n_valid_legs == 1 and port.n_invalid_legs == 1
    # the aggregate must equal the valid leg alone, not be NaN-poisoned or zero-padded
    assert port.delta == pytest.approx(valid_leg.position_greek("delta"))


def test_aggregate_greeks_requires_at_least_one_position():
    with pytest.raises(PortfolioGreeksError):
        aggregate_greeks([])


def test_all_invalid_legs_yields_nan_aggregate_not_a_fabricated_zero():
    invalid_leg = OptionPosition(
        option_type="CE", strike=F, expiry="2024-06-20", side="LONG", quantity_contracts=1,
        lot_size=LOT, delta=float("nan"), gamma=float("nan"), theta=float("nan"), vega=float("nan"),
        valid=False,
    )
    port = aggregate_greeks([invalid_leg])
    assert np.isnan(port.delta)
    assert port.n_valid_legs == 0


# --------------------------------------------------------------------------- #
# standard strategies: sign and magnitude sanity
# --------------------------------------------------------------------------- #
def test_long_call_has_positive_delta_and_gamma_negative_theta():
    leg = _leg("CE", F, side="LONG")
    p = aggregate_greeks([leg])
    assert p.delta > 0 and p.gamma > 0 and p.theta < 0 and p.vega > 0


def test_long_put_has_negative_delta_positive_gamma_negative_theta():
    leg = _leg("PE", F, side="LONG")
    p = aggregate_greeks([leg])
    assert p.delta < 0 and p.gamma > 0 and p.theta < 0 and p.vega > 0


def test_short_call_has_negative_delta_negative_gamma_positive_theta():
    leg = _leg("CE", F, side="SHORT")
    p = aggregate_greeks([leg])
    assert p.delta < 0 and p.gamma < 0 and p.theta > 0 and p.vega < 0


def test_short_put_has_positive_delta_negative_gamma_positive_theta():
    leg = _leg("PE", F, side="SHORT")
    p = aggregate_greeks([leg])
    assert p.delta > 0 and p.gamma < 0 and p.theta > 0 and p.vega < 0


def test_bull_call_spread_has_positive_but_bounded_delta():
    long_leg = _leg("CE", F - 200, side="LONG")
    short_leg = _leg("CE", F + 200, side="SHORT")
    p = aggregate_greeks([long_leg, short_leg])
    # long the near strike, short the far strike -> net long delta, but
    # smaller than either leg alone (the short leg partially offsets it)
    assert 0 < p.delta < long_leg.position_greek("delta")


def test_bear_put_spread_has_negative_but_bounded_delta():
    long_leg = _leg("PE", F + 200, side="LONG")
    short_leg = _leg("PE", F - 200, side="SHORT")
    p = aggregate_greeks([long_leg, short_leg])
    assert long_leg.position_greek("delta") < p.delta < 0


def test_long_straddle_delta_near_zero_atm_and_gamma_vega_doubled():
    call = _leg("CE", F, side="LONG")
    put = _leg("PE", F, side="LONG")
    p = aggregate_greeks([call, put])
    assert abs(p.delta) < 0.1 * LOT  # near-zero net delta ATM (call ~+0.5, put ~-0.5)
    assert p.gamma == pytest.approx(call.position_greek("gamma") + put.position_greek("gamma"))
    assert p.theta < 0 and p.vega > 0


def test_iron_condor_is_short_gamma_and_long_theta_net():
    # short the near wings, long the far wings (classic iron condor), all same expiry
    short_put = _leg("PE", F - 200, side="SHORT")
    long_put = _leg("PE", F - 400, side="LONG")
    short_call = _leg("CE", F + 200, side="SHORT")
    long_call = _leg("CE", F + 400, side="LONG")
    p = aggregate_greeks([short_put, long_put, short_call, long_call])
    assert p.theta > 0  # net short premium decays in the position's favor
    assert p.gamma < 0  # net short gamma


# --------------------------------------------------------------------------- #
# finite-difference validation of the PORTFOLIO-level local approximation
# --------------------------------------------------------------------------- #
def _portfolio_value(positions, forward=F, sigma=SIG, t=T):
    total = 0.0
    for p in positions:
        price = float(black76_price(forward, p.strike, sigma, t, p.option_type))
        total += p.multiplier * price
    return total


def test_portfolio_delta_matches_finite_difference_in_forward():
    legs = [_leg("CE", F - 200, "LONG"), _leg("CE", F + 200, "SHORT")]
    port = aggregate_greeks(legs)
    h = 1.0
    up = _portfolio_value(legs, forward=F + h)
    dn = _portfolio_value(legs, forward=F - h)
    fd_delta = (up - dn) / (2 * h)
    assert port.delta == pytest.approx(fd_delta, rel=1e-3)


def test_portfolio_gamma_matches_finite_difference_second_derivative():
    legs = [_leg("CE", F, "LONG"), _leg("PE", F, "LONG")]
    port = aggregate_greeks(legs)
    h = 5.0
    up = _portfolio_value(legs, forward=F + h)
    mid = _portfolio_value(legs, forward=F)
    dn = _portfolio_value(legs, forward=F - h)
    fd_gamma = (up - 2 * mid + dn) / h**2
    assert port.gamma == pytest.approx(fd_gamma, rel=1e-2)


def test_portfolio_vega_matches_finite_difference_in_volatility():
    legs = [_leg("PE", F - 200, "SHORT"), _leg("PE", F - 400, "LONG")]
    port = aggregate_greeks(legs)
    h = 1e-4
    up = _portfolio_value(legs, sigma=SIG + h)
    dn = _portfolio_value(legs, sigma=SIG - h)
    fd_vega = (up - dn) / (2 * h)
    assert port.vega == pytest.approx(fd_vega, rel=1e-3)


def test_portfolio_theta_matches_one_day_decay_approximately():
    """Theta is per calendar day; advancing time by one day and repricing
    (all else equal) should land close to the reported theta -- an
    approximation, not an exact match, since theta is a local derivative
    and one full day is not infinitesimal."""
    legs = [_leg("CE", F, "LONG")]
    port = aggregate_greeks(legs)
    t_now = T
    t_tomorrow = T - 1 / 365
    value_now = _portfolio_value(legs, t=t_now)
    value_tomorrow = _portfolio_value(legs, t=t_tomorrow)
    one_day_change = value_tomorrow - value_now
    assert one_day_change == pytest.approx(port.theta, rel=0.15)  # loose: theta is a local derivative


def test_multi_leg_reprice_is_consistent_before_finite_differencing():
    """All legs must be repriced at the SAME perturbed forward/vol/time --
    pricing one leg at the bumped value and another at the original would
    silently corrupt every finite-difference check above."""
    legs = [_leg("CE", F - 100, "LONG"), _leg("CE", F + 100, "SHORT")]
    v1 = _portfolio_value(legs, forward=F + 10)
    v2 = sum(
        p.multiplier * float(black76_price(F + 10, p.strike, SIG, T, p.option_type))
        for p in legs
    )
    assert v1 == pytest.approx(v2)
