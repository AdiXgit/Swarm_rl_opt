"""Black-76 pricing, implied volatility and Greeks (approved decisions 1-3).

NIFTY options are priced off the expiry-matched future, not the index. On a
sample date the parity-implied forward sat ~70 points above spot; inverting
against spot would push that entire offset into the volatility smile and
manufacture a skew that is not in the market. Everything here is therefore
Black-76 on the forward supplied by the data layer.

Conventions
-----------
* Prices are undiscounted. NSE marks options on the futures-style convention
  and the simulator books cash on the same basis, so applying a discount factor
  here and not there would leak a small bias into every P&L.
* Time is in calendar years, ACT/365. Expiry day itself is floored at a
  fraction of a day so a same-day contract has finite (not zero) vega.
* Vega is per 1.00 of vol (not per vol point); Theta is per calendar day.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

#: Floor on time-to-expiry, in years. An expiry-day contract still has a few
#: hours of life; zero would make every Greek singular.
MIN_T = 1.0 / (365.0 * 24.0)

#: Bracket for the IV solver. Weekly index options have printed above 150% vol
#: in stress, so the upper bound is deliberately generous.
IV_LOW, IV_HIGH = 1e-4, 5.0

#: A quote is only invertible strictly inside the no-arbitrage band. Quotes on
#: or outside it carry no volatility information at all.
ARB_TOL = 1e-8


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    theta: float
    vega: float


def year_fraction(days_to_expiry) -> np.ndarray:
    """Calendar days -> years, floored so expiry day stays well defined."""
    return np.maximum(np.asarray(days_to_expiry, dtype=float) / 365.0, MIN_T)


def _d1_d2(forward, strike, sigma, t):
    forward = np.asarray(forward, dtype=float)
    strike = np.asarray(strike, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    t = np.asarray(t, dtype=float)

    vol_t = sigma * np.sqrt(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(forward / strike) + 0.5 * sigma**2 * t) / vol_t
    return d1, d1 - vol_t


def black76_price(forward, strike, sigma, t, option_type) -> np.ndarray:
    """Undiscounted Black-76 price. `option_type` is "CE" or "PE"."""
    d1, d2 = _d1_d2(forward, strike, sigma, t)
    forward = np.asarray(forward, dtype=float)
    strike = np.asarray(strike, dtype=float)
    is_call = np.asarray(option_type) == "CE"

    call = forward * norm.cdf(d1) - strike * norm.cdf(d2)
    put = strike * norm.cdf(-d2) - forward * norm.cdf(-d1)
    return np.where(is_call, call, put)


def intrinsic(forward, strike, option_type) -> np.ndarray:
    """Undiscounted intrinsic on the forward: the lower no-arbitrage bound."""
    forward = np.asarray(forward, dtype=float)
    strike = np.asarray(strike, dtype=float)
    is_call = np.asarray(option_type) == "CE"
    return np.where(is_call, np.maximum(forward - strike, 0.0),
                    np.maximum(strike - forward, 0.0))


def upper_bound(forward, strike, option_type) -> np.ndarray:
    """Undiscounted upper no-arbitrage bound: F for a call, K for a put."""
    forward = np.asarray(forward, dtype=float)
    strike = np.asarray(strike, dtype=float)
    return np.where(np.asarray(option_type) == "CE", forward, strike)


# --------------------------------------------------------------------------- #
# implied volatility
# --------------------------------------------------------------------------- #
def implied_vol_scalar(price, forward, strike, t, option_type) -> float:
    """Invert one Black-76 quote. Returns NaN when the quote carries no vol.

    A quote at or beyond either no-arbitrage bound is not a failed solve, it is
    a quote with no volatility content -- deep-ITM contracts that print at
    intrinsic are the common case. Those are reported as NaN rather than
    clamped to the bracket, which would otherwise pin a hard floor or ceiling
    into the smile and corrupt every IV-rank and skew feature built on it.
    """
    if not all(np.isfinite([price, forward, strike, t])):
        return float("nan")
    if forward <= 0 or strike <= 0 or t <= 0:
        return float("nan")

    lo = float(intrinsic(forward, strike, option_type))
    hi = float(upper_bound(forward, strike, option_type))
    if price <= lo + ARB_TOL or price >= hi - ARB_TOL:
        return float("nan")

    def objective(sigma: float) -> float:
        return float(black76_price(forward, strike, sigma, t, option_type)) - price

    try:
        if objective(IV_LOW) > 0 or objective(IV_HIGH) < 0:
            return float("nan")
        return float(brentq(objective, IV_LOW, IV_HIGH, xtol=1e-8, maxiter=200))
    except (ValueError, RuntimeError):
        return float("nan")


def implied_vol(price, forward, strike, t, option_type) -> np.ndarray:
    """Vectorised wrapper over `implied_vol_scalar`."""
    price, forward, strike, t, option_type = np.broadcast_arrays(
        np.asarray(price, dtype=float),
        np.asarray(forward, dtype=float),
        np.asarray(strike, dtype=float),
        np.asarray(t, dtype=float),
        np.asarray(option_type),
    )
    out = np.empty(price.shape, dtype=float)
    for i in np.ndindex(price.shape):
        out[i] = implied_vol_scalar(
            price[i], forward[i], strike[i], t[i], option_type[i]
        )
    return out


# --------------------------------------------------------------------------- #
# greeks
# --------------------------------------------------------------------------- #
def greeks(forward, strike, sigma, t, option_type) -> dict[str, np.ndarray]:
    """Black-76 Greeks, per unit of the underlying, undiscounted.

    Delta is with respect to the *forward*, which is what a futures-settled
    index option actually hedges against. Theta is per calendar day and Vega
    per 1.00 of volatility.
    """
    d1, d2 = _d1_d2(forward, strike, sigma, t)
    forward = np.asarray(forward, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    t = np.asarray(t, dtype=float)
    is_call = np.asarray(option_type) == "CE"

    pdf = norm.pdf(d1)
    sqrt_t = np.sqrt(t)

    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(is_call, norm.cdf(d1), norm.cdf(d1) - 1.0)
        gamma = pdf / (forward * sigma * sqrt_t)
        vega = forward * pdf * sqrt_t
        theta = -(forward * pdf * sigma) / (2.0 * sqrt_t) / 365.0

    # np.where rather than in-place masking: these are 0-d arrays for scalar
    # inputs, and item assignment on those raises.
    valid = np.isfinite(sigma) & (sigma > 0)
    return {
        name: np.where(valid, arr, np.nan)
        for name, arr in (
            ("delta", delta), ("gamma", gamma), ("theta", theta), ("vega", vega)
        )
    }
