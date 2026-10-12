"""Phase 2.8: position-level and portfolio-level Greek aggregation.

New, additive module. Does not change `features.pricing` (Black-76,
already validated by `tests/test_pricing.py`) or `agents.greeks_agent`
(contract-level ATM Greeks already in `SWARM_STATE_V1`). This is the
missing layer between "one contract's per-underlying-unit Greek" and "a
multi-leg position's signed exposure" -- needed for Workstream A's
Greek-aware risk research, not for any Phase 3 Decision Engine or Trade
Constructor (neither of which exists and neither of which this module
builds).

Convention, stated once and used everywhere below
---------------------------------------------------
`features.pricing.greeks()` returns Greeks **per underlying unit of one
option** (delta in [-1, 1], gamma/vega positive, theta negative for long
options) -- confirmed against `tests/test_pricing.py`'s existing
finite-difference checks. Lot size is a separate column on the normalized
chain (`contracts.py`) and is applied **exactly once**, here, when going
from "one option's per-unit Greek" to "one position's signed exposure":

    position_greek = signed_contracts * lot_size * per_unit_greek

where `signed_contracts` is positive for LONG, negative for SHORT. Nothing
upstream of this module (enrich_chain, greeks_agent) multiplies by lot size,
so this is the only place that does -- multiplying twice would silently
double every portfolio exposure number.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VALID_SIDES = ("LONG", "SHORT")
GREEK_NAMES = ("delta", "gamma", "theta", "vega")


class PortfolioGreeksError(ValueError):
    pass


@dataclass(frozen=True)
class OptionPosition:
    """One leg of a (possibly multi-leg) options position.

    `delta`/`gamma`/`theta`/`vega` are the PER-UNDERLYING-UNIT Greeks for
    this contract (i.e. exactly what `features.pricing.greeks()` or
    `agents.greeks_agent.ContractGreeks` already report) -- not yet scaled
    by lot size or quantity. `valid` mirrors `ContractGreeks.valid`: a leg
    whose IV could not be inverted (missing forward, quote outside the
    no-arbitrage band) carries NaN Greeks and `valid=False`, and is excluded
    from aggregation rather than treated as a zero exposure.
    """

    option_type: str  # "CE" | "PE"
    strike: float
    expiry: str
    side: str  # "LONG" | "SHORT"
    quantity_contracts: int  # positive count of contracts
    lot_size: int
    delta: float
    gamma: float
    theta: float
    vega: float
    valid: bool
    label: str = ""

    def __post_init__(self):
        if self.side not in VALID_SIDES:
            raise PortfolioGreeksError(f"side must be one of {VALID_SIDES}, got {self.side!r}")
        if self.quantity_contracts <= 0:
            raise PortfolioGreeksError(f"quantity_contracts must be positive, got {self.quantity_contracts}")
        if self.lot_size <= 0:
            raise PortfolioGreeksError(f"lot_size must be positive, got {self.lot_size}")
        if self.option_type not in ("CE", "PE"):
            raise PortfolioGreeksError(f"option_type must be CE or PE, got {self.option_type!r}")

    @property
    def signed_contracts(self) -> int:
        return self.quantity_contracts if self.side == "LONG" else -self.quantity_contracts

    @property
    def multiplier(self) -> int:
        """`signed_contracts * lot_size` -- applied once, to the per-unit Greek."""
        return self.signed_contracts * self.lot_size

    def position_greek(self, name: str) -> float:
        if name not in GREEK_NAMES:
            raise PortfolioGreeksError(f"unknown greek {name!r}; expected one of {GREEK_NAMES}")
        per_unit = getattr(self, name)
        if not self.valid or not np.isfinite(per_unit):
            return float("nan")
        return float(self.multiplier * per_unit)

    def to_dict(self) -> dict:
        d = {"option_type": self.option_type, "strike": self.strike, "expiry": self.expiry,
             "side": self.side, "quantity_contracts": self.quantity_contracts,
             "lot_size": self.lot_size, "valid": self.valid, "label": self.label}
        d.update({name: self.position_greek(name) for name in GREEK_NAMES})
        return d


@dataclass(frozen=True)
class PortfolioGreeks:
    delta: float
    gamma: float
    theta: float
    vega: float
    n_legs: int
    n_valid_legs: int
    n_invalid_legs: int
    scaling_convention: str = "signed_contracts * lot_size * per_underlying_unit_greek"

    def to_dict(self) -> dict:
        return {
            "delta": self.delta, "gamma": self.gamma, "theta": self.theta, "vega": self.vega,
            "n_legs": self.n_legs, "n_valid_legs": self.n_valid_legs, "n_invalid_legs": self.n_invalid_legs,
            "scaling_convention": self.scaling_convention,
        }


def aggregate_greeks(positions: list[OptionPosition]) -> PortfolioGreeks:
    """Sum signed, lot-scaled Greeks across all legs with `valid=True`.

    A leg with invalid Greeks (unsolvable IV, missing forward) is excluded
    from the sum -- not treated as a zero exposure, which would understate
    risk. `n_invalid_legs` reports how many were dropped so the caller can
    see that the aggregate is partial, not pretend the portfolio has no
    unpriced legs.
    """
    if not positions:
        raise PortfolioGreeksError("aggregate_greeks needs at least one position")

    valid_legs = [p for p in positions if p.valid]
    sums = {}
    for name in GREEK_NAMES:
        values = [p.position_greek(name) for p in valid_legs]
        sums[name] = float(np.sum(values)) if values else float("nan")

    return PortfolioGreeks(
        delta=sums["delta"], gamma=sums["gamma"], theta=sums["theta"], vega=sums["vega"],
        n_legs=len(positions), n_valid_legs=len(valid_legs), n_invalid_legs=len(positions) - len(valid_legs),
    )
