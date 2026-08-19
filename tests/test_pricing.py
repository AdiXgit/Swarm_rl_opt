"""Black-76 pricing, IV inversion and Greeks."""
from __future__ import annotations

import numpy as np
import pytest

from swarm_opt.features.pricing import (
    ARB_TOL,
    IV_HIGH,
    black76_price,
    greeks,
    implied_vol,
    implied_vol_scalar,
    intrinsic,
    year_fraction,
)

F, K, T, SIG = 24230.0, 24250.0, 7 / 365, 0.12


# --------------------------------------------------------------------------- #
# pricing identities
# --------------------------------------------------------------------------- #
def test_put_call_parity_holds_undiscounted():
    """Under Black-76 with no discounting, C - P == F - K exactly."""
    c = float(black76_price(F, K, SIG, T, "CE"))
    p = float(black76_price(F, K, SIG, T, "PE"))
    assert c - p == pytest.approx(F - K, abs=1e-6)


def test_atm_price_matches_the_closed_form():
    """At F == K, Black-76 collapses to 2F(N(s/2) - 1/2), s = sigma*sqrt(T)."""
    from scipy.stats import norm

    s = SIG * np.sqrt(T)
    expected = 2 * F * (norm.cdf(s / 2) - 0.5)
    assert float(black76_price(F, F, SIG, T, "CE")) == pytest.approx(expected, rel=1e-9)


def test_price_sits_inside_the_no_arbitrage_band():
    for opt, bound in (("CE", F), ("PE", K)):
        px = float(black76_price(F, K, SIG, T, opt))
        assert float(intrinsic(F, K, opt)) < px < bound


def test_price_is_monotone_in_volatility():
    vols = np.linspace(0.05, 0.9, 25)
    for opt in ("CE", "PE"):
        px = black76_price(F, K, vols, T, opt)
        assert np.all(np.diff(px) > 0)


def test_call_falls_and_put_rises_in_strike():
    strikes = np.linspace(22000, 26000, 30)
    assert np.all(np.diff(black76_price(F, strikes, SIG, T, "CE")) < 0)
    assert np.all(np.diff(black76_price(F, strikes, SIG, T, "PE")) > 0)


# --------------------------------------------------------------------------- #
# implied volatility
# --------------------------------------------------------------------------- #
#: NSE quotes index option premia in 0.05 steps, so on real data a premium is
#: either zero or at least a tick. The sub-tick cases below are synthetic.
TICK = 0.05


@pytest.mark.parametrize("opt", ["CE", "PE"])
@pytest.mark.parametrize("sigma", [0.06, 0.12, 0.35, 0.9])
@pytest.mark.parametrize("strike", [23000.0, 24230.0, 25500.0])
def test_inversion_round_trips(opt, sigma, strike):
    """Round-trip everywhere the quote carries any volatility information.

    The contract is the no-arbitrage band, not the tick: some cells of this
    grid (5% out, 7 days, 6% vol) carry time value of order 1e-9 and collapse
    onto intrinsic in float, and those must come back NaN rather than a
    fabricated number.
    """
    px = float(black76_price(F, strike, sigma, T, opt))
    time_value = px - float(intrinsic(F, strike, opt))
    got = implied_vol_scalar(px, F, strike, T, opt)

    if time_value <= ARB_TOL:
        assert np.isnan(got)
    else:
        assert got == pytest.approx(sigma, rel=1e-5)


def test_inversion_gives_up_when_the_quote_collapses_onto_intrinsic():
    """A quote worth intrinsic plus nano-rupees pins no volatility.

    NaN is the honest answer; the danger would be returning the bracket edge
    and letting that be read as a smile.
    """
    px = float(black76_price(F, 20000.0, 0.06, T, "CE"))
    assert px - float(intrinsic(F, 20000.0, "CE")) <= ARB_TOL
    assert np.isnan(implied_vol_scalar(px, F, 20000.0, T, "CE"))


def test_inversion_survives_a_thin_but_real_quote():
    """One tick of time value is still information and must still invert."""
    px = float(black76_price(F, 25500.0, 0.06, T, "CE"))
    assert ARB_TOL < px - float(intrinsic(F, 25500.0, "CE")) < TICK
    assert implied_vol_scalar(px, F, 25500.0, T, "CE") == pytest.approx(0.06, rel=1e-4)


def test_inversion_works_at_a_realistic_deep_wing_quote():
    """Guard the other side: a genuinely quotable wing must still invert.

    This is what would break if the arbitrage-bound tolerance were widened to
    make the sub-tick cases pass.
    """
    px = float(black76_price(F, 21000.0, 0.35, T, "PE"))
    assert px > TICK
    assert implied_vol_scalar(px, F, 21000.0, T, "PE") == pytest.approx(0.35, rel=1e-5)


def test_quote_at_intrinsic_is_nan_not_zero():
    """Deep-ITM contracts print at intrinsic and carry no vol information.

    Returning 0.0 (or clamping to the solver bracket) would pin a false floor
    into the smile that every IV-rank and skew feature would then inherit.
    """
    px = float(intrinsic(F, 18000.0, "CE"))
    assert np.isnan(implied_vol_scalar(px, F, 18000.0, T, "CE"))


def test_quote_outside_the_arbitrage_band_is_nan():
    assert np.isnan(implied_vol_scalar(F + 1.0, F, K, T, "CE"))   # above upper bound
    assert np.isnan(implied_vol_scalar(-1.0, F, K, T, "CE"))      # negative price
    assert np.isnan(implied_vol_scalar(K + 1.0, F, K, T, "PE"))


def test_unsolvable_inputs_are_nan_not_exceptions():
    assert np.isnan(implied_vol_scalar(np.nan, F, K, T, "CE"))
    assert np.isnan(implied_vol_scalar(100.0, np.nan, K, T, "CE"))
    assert np.isnan(implied_vol_scalar(100.0, F, K, 0.0, "CE"))
    assert np.isnan(implied_vol_scalar(100.0, -F, K, T, "CE"))


def test_vol_beyond_the_bracket_is_reported_missing():
    px = float(black76_price(F, K, IV_HIGH * 1.5, T, "CE"))
    assert np.isnan(implied_vol_scalar(px, F, K, T, "CE"))


def test_vectorised_inversion_matches_scalar():
    strikes = np.array([22000.0, 24000.0, 24230.0, 25000.0, 26500.0])
    opts = np.array(["CE", "PE", "CE", "PE", "CE"])
    px = black76_price(F, strikes, SIG, T, opts)
    got = implied_vol(px, F, strikes, T, opts)
    assert np.allclose(got, SIG, rtol=1e-5)


# --------------------------------------------------------------------------- #
# greeks
# --------------------------------------------------------------------------- #
def test_delta_bounds_and_call_put_relation():
    strikes = np.linspace(20000, 28000, 40)
    call = greeks(F, strikes, SIG, T, "CE")["delta"]
    put = greeks(F, strikes, SIG, T, "PE")["delta"]
    assert np.all((call >= 0) & (call <= 1))
    assert np.all((put >= -1) & (put <= 0))
    assert np.allclose(call - put, 1.0, atol=1e-9)
    # and it must actually vary across the ladder, not sit pinned at the bounds
    assert call.min() < 0.05 and call.max() > 0.95


def test_atm_delta_is_about_a_half():
    assert float(greeks(F, F, SIG, T, "CE")["delta"]) == pytest.approx(0.5, abs=0.01)


def test_gamma_and_vega_are_side_independent_and_positive():
    call = greeks(F, K, SIG, T, "CE")
    put = greeks(F, K, SIG, T, "PE")
    assert float(call["gamma"]) == pytest.approx(float(put["gamma"]), rel=1e-12)
    assert float(call["vega"]) == pytest.approx(float(put["vega"]), rel=1e-12)
    assert float(call["gamma"]) > 0 and float(call["vega"]) > 0


def test_long_options_decay():
    assert float(greeks(F, K, SIG, T, "CE")["theta"]) < 0
    assert float(greeks(F, K, SIG, T, "PE")["theta"]) < 0


def test_delta_matches_a_numerical_derivative():
    h = 1e-3
    up = float(black76_price(F + h, K, SIG, T, "CE"))
    dn = float(black76_price(F - h, K, SIG, T, "CE"))
    assert float(greeks(F, K, SIG, T, "CE")["delta"]) == pytest.approx(
        (up - dn) / (2 * h), rel=1e-5
    )


def test_vega_matches_a_numerical_derivative():
    h = 1e-6
    up = float(black76_price(F, K, SIG + h, T, "CE"))
    dn = float(black76_price(F, K, SIG - h, T, "CE"))
    assert float(greeks(F, K, SIG, T, "CE")["vega"]) == pytest.approx(
        (up - dn) / (2 * h), rel=1e-5
    )


def test_gamma_matches_a_numerical_second_derivative():
    h = 1.0
    up = float(black76_price(F + h, K, SIG, T, "CE"))
    mid = float(black76_price(F, K, SIG, T, "CE"))
    dn = float(black76_price(F - h, K, SIG, T, "CE"))
    assert float(greeks(F, K, SIG, T, "CE")["gamma"]) == pytest.approx(
        (up - 2 * mid + dn) / h**2, rel=1e-4
    )


def test_greeks_are_nan_where_vol_is_missing():
    g = greeks(F, K, np.array([np.nan, 0.0, 0.2]), T, "CE")
    for name in ("delta", "gamma", "theta", "vega"):
        assert np.isnan(g[name][0]) and np.isnan(g[name][1])
        assert np.isfinite(g[name][2])


# --------------------------------------------------------------------------- #
# time
# --------------------------------------------------------------------------- #
def test_expiry_day_keeps_finite_greeks():
    """A 0-DTE contract still trades; zero time would make every Greek blow up."""
    t = float(year_fraction(0))
    assert t > 0
    g = greeks(F, K, SIG, t, "CE")
    assert all(np.isfinite(float(g[k])) for k in ("delta", "gamma", "theta", "vega"))


def test_year_fraction_is_act_365():
    assert float(year_fraction(365)) == pytest.approx(1.0)
    assert float(year_fraction(7)) == pytest.approx(7 / 365)
