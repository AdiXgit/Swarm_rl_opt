"""Deterministic Greeks Agent.

Reuses the validated Black-76 pricing (`features.pricing`) and the
futures / parity / carry-curve forward hierarchy already attached by
`data.normalize.attach_forward` (`forward_source` is carried through
unchanged). This module computes no prices and solves no IV itself beyond
calling `features.enrich.enrich_chain`; it does not modify that validated
pipeline.

Market-level vs. contract-level
--------------------------------
This agent reports **contract-level** Greeks for the at-the-money call and
put of the nearest eligible expiry on the observation date -- the two
contracts a weekly decision is actually most likely to act on. It does not
report every strike's Greeks (that is chain-level data, already available
from `enrich_chain` directly) and it does not report **portfolio** Greeks,
because no portfolio exists yet in Phase 2 (no Decision Engine, no Trade
Constructor): `portfolio` is always `None` with `valid=False` and an
explicit reason, never a fabricated zero or an invented exposure.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features.enrich import enrich_chain

AGENT_NAME = "greeks"
MODEL_VERSION = "black76_v1"

#: Mirrors scripts/build_market_features.py so the agent and the cached
#: training-time feature both look at the same slice of the chain.
MAX_DAYS_TO_EXPIRY = 10


@dataclass(frozen=True)
class ContractGreeks:
    option_type: str
    strike: float
    expiry: str
    days_to_expiry: int
    forward: float
    forward_source: str
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    valid: bool
    reason: str

    def to_dict(self) -> dict:
        return {
            "option_type": self.option_type,
            "strike": self.strike,
            "expiry": self.expiry,
            "days_to_expiry": self.days_to_expiry,
            "forward": self.forward,
            "forward_source": self.forward_source,
            "implied_volatility": self.implied_volatility,
            "delta": self.delta,
            "gamma": self.gamma,
            "theta": self.theta,
            "vega": self.vega,
            "valid": self.valid,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class GreeksAgentResult:
    agent_name: str
    model_version: str
    observation_timestamp: str
    underlying: str
    atm_call: ContractGreeks | None
    atm_put: ContractGreeks | None
    portfolio: None
    portfolio_valid: bool
    portfolio_reason: str

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "model_version": self.model_version,
            "observation_timestamp": self.observation_timestamp,
            "underlying": self.underlying,
            "atm_call": self.atm_call.to_dict() if self.atm_call else None,
            "atm_put": self.atm_put.to_dict() if self.atm_put else None,
            "portfolio": self.portfolio,
            "portfolio_valid": self.portfolio_valid,
            "portfolio_reason": self.portfolio_reason,
        }


def _contract_greeks(row: pd.Series) -> ContractGreeks:
    iv = row["implied_volatility"]
    valid = bool(np.isfinite(iv)) and all(
        np.isfinite(row[g]) for g in ("delta", "gamma", "theta", "vega")
    )
    reason = "ok" if valid else (
        "no_forward" if not np.isfinite(row["forward"]) else "iv_not_invertible"
    )
    return ContractGreeks(
        option_type=str(row["option_type"]),
        strike=float(row["strike"]),
        expiry=pd.Timestamp(row["expiry"]).date().isoformat(),
        days_to_expiry=int(row["days_to_expiry"]),
        forward=float(row["forward"]) if np.isfinite(row["forward"]) else float("nan"),
        forward_source=str(row["forward_source"]),
        implied_volatility=float(iv) if np.isfinite(iv) else float("nan"),
        delta=float(row["delta"]) if np.isfinite(row["delta"]) else float("nan"),
        gamma=float(row["gamma"]) if np.isfinite(row["gamma"]) else float("nan"),
        theta=float(row["theta"]) if np.isfinite(row["theta"]) else float("nan"),
        vega=float(row["vega"]) if np.isfinite(row["vega"]) else float("nan"),
        valid=valid,
        reason=reason,
    )


def compute_greeks(
    chain: pd.DataFrame, observation_timestamp: pd.Timestamp, underlying: str
) -> GreeksAgentResult:
    """ATM call/put Greeks for the nearest eligible expiry in `chain`.

    `chain` must already be restricted to the observation date (the caller's
    responsibility, same as every other point-in-time read in this
    project). Returns `atm_call = atm_put = None` when no expiry within
    `MAX_DAYS_TO_EXPIRY` is listed at all -- an explicit absence, not a
    fabricated zero.
    """
    near = chain[chain["days_to_expiry"] <= MAX_DAYS_TO_EXPIRY]
    if near.empty:
        return GreeksAgentResult(
            agent_name=AGENT_NAME,
            model_version=MODEL_VERSION,
            observation_timestamp=pd.Timestamp(observation_timestamp).isoformat(),
            underlying=underlying,
            atm_call=None,
            atm_put=None,
            portfolio=None,
            portfolio_valid=False,
            portfolio_reason="no portfolio constructed in Phase 2 (Decision Engine/Trade "
            "Constructor are out of scope)",
        )

    nearest_expiry = near["expiry"].min()
    leg = near[near["expiry"] == nearest_expiry]
    spot = float(leg["spot"].iloc[0])
    enriched = enrich_chain(leg)

    atm_strike = enriched.loc[(enriched["strike"] - spot).abs().idxmin(), "strike"]
    at_strike = enriched[enriched["strike"] == atm_strike]
    call_rows = at_strike[at_strike["option_type"] == "CE"]
    put_rows = at_strike[at_strike["option_type"] == "PE"]

    return GreeksAgentResult(
        agent_name=AGENT_NAME,
        model_version=MODEL_VERSION,
        observation_timestamp=pd.Timestamp(observation_timestamp).isoformat(),
        underlying=underlying,
        atm_call=_contract_greeks(call_rows.iloc[0]) if len(call_rows) else None,
        atm_put=_contract_greeks(put_rows.iloc[0]) if len(put_rows) else None,
        portfolio=None,
        portfolio_valid=False,
        portfolio_reason="no portfolio constructed in Phase 2 (Decision Engine/Trade "
        "Constructor are out of scope)",
    )
