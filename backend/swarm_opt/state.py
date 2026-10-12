"""SWARM_STATE_V1: the single authoritative, versioned swarm-state schema.

Combines the three specialist agent predictions (Direction, Volatility,
Regime) and the deterministic Greeks agent output with market features into
one object a future Decision Engine can consume without knowing any agent's
internal implementation. Carries no trading action, strategy recommendation,
or portfolio state (capital, positions, P&L, drawdown, margin) -- those
belong to the Decision Engine / Trade Constructor, which is out of scope for
Phase 2 (see CLAUDE.md "First Task" and the Phase 2 prompt, section 10).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .agents.base import AgentPrediction
from .agents.greeks_agent import GreeksAgentResult

SCHEMA_VERSION = "SWARM_STATE_V1"
_COMPATIBLE_SCHEMA_VERSIONS = {SCHEMA_VERSION}

REQUIRED_AGENT_NAMES = ("direction", "volatility", "regime")


class SwarmStateError(ValueError):
    pass


@dataclass(frozen=True)
class SwarmState:
    schema_version: str
    observation_timestamp: str
    underlying: str
    market_features: dict
    direction: dict
    volatility: dict
    regime: dict
    greeks: dict | None
    data_quality: dict
    model_versions: dict
    feature_schema_version: str
    portfolio: None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SwarmState":
        return cls(**d)


def build_swarm_state(
    observation_timestamp,
    underlying: str,
    market_features: dict,
    agent_predictions: dict[str, AgentPrediction],
    greeks_result: GreeksAgentResult | None,
    feature_schema_version: str,
) -> SwarmState:
    """Deterministic for the same valid inputs. Raises `SwarmStateError`
    rather than emitting a valid-looking state built from incomplete or
    mismatched inputs -- see `validate_swarm_state` for the full check set,
    which this calls before returning.
    """
    missing = [n for n in REQUIRED_AGENT_NAMES if n not in agent_predictions]
    if missing:
        raise SwarmStateError(f"missing required agent prediction(s): {missing}")

    obs_ts = pd.Timestamp(observation_timestamp).isoformat()

    data_quality = {
        "market_features_complete": bool(
            not any(
                v is None or (isinstance(v, float) and not np.isfinite(v))
                for k, v in market_features.items()
                if k not in ("atm_iv_available", "has_full_lookback")
            )
        ),
        "has_full_lookback": bool(market_features.get("has_full_lookback", False)),
        "atm_iv_available": bool(market_features.get("atm_iv_available", False)),
        "greeks_available": greeks_result is not None and (
            (greeks_result.atm_call is not None and greeks_result.atm_call.valid)
            or (greeks_result.atm_put is not None and greeks_result.atm_put.valid)
        ),
    }
    for name in REQUIRED_AGENT_NAMES:
        data_quality[f"{name}_had_missing_feature"] = bool(
            agent_predictions[name].data_quality.get("had_missing_feature", False)
        )

    model_versions = {name: agent_predictions[name].model_version for name in REQUIRED_AGENT_NAMES}
    if greeks_result is not None:
        model_versions["greeks"] = greeks_result.model_version

    state = SwarmState(
        schema_version=SCHEMA_VERSION,
        observation_timestamp=obs_ts,
        underlying=underlying,
        market_features=dict(market_features),
        direction=agent_predictions["direction"].to_dict(),
        volatility=agent_predictions["volatility"].to_dict(),
        regime=agent_predictions["regime"].to_dict(),
        greeks=greeks_result.to_dict() if greeks_result is not None else None,
        data_quality=data_quality,
        model_versions=model_versions,
        feature_schema_version=feature_schema_version,
        portfolio=None,
    )
    errors = validate_swarm_state(state)
    if errors:
        raise SwarmStateError("built an invalid swarm state: " + "; ".join(errors))
    return state


def validate_swarm_state(state: SwarmState) -> list[str]:
    """Returns a list of validation errors; empty means valid.

    Never raises itself (callers that want a hard failure should raise on a
    non-empty result, as `build_swarm_state` does) so it can also be used to
    inspect a state that is known to be invalid, e.g. in negative tests.
    """
    errors: list[str] = []

    if state.schema_version not in _COMPATIBLE_SCHEMA_VERSIONS:
        errors.append(f"unsupported schema_version {state.schema_version!r}")

    try:
        obs_ts = pd.Timestamp(state.observation_timestamp)
        if pd.isna(obs_ts):
            errors.append("observation_timestamp is not a valid timestamp")
    except (ValueError, TypeError):
        errors.append("observation_timestamp is not a valid timestamp")
        obs_ts = None

    if not state.underlying:
        errors.append("underlying is empty")

    agent_blocks = {
        "direction": state.direction,
        "volatility": state.volatility,
        "regime": state.regime,
    }
    agent_timestamps = set()
    for name, block in agent_blocks.items():
        if block is None:
            errors.append(f"{name} prediction is missing")
            continue
        required_keys = {"predicted_class", "probabilities", "agent_name", "observation_timestamp"}
        missing_keys = required_keys - set(block)
        if missing_keys:
            errors.append(f"{name} prediction missing field(s): {sorted(missing_keys)}")
            continue
        probs = block["probabilities"]
        if not probs:
            errors.append(f"{name} prediction has no probabilities")
        else:
            total = sum(probs.values())
            if not all(np.isfinite(p) for p in probs.values()):
                errors.append(f"{name} prediction has non-finite probabilities")
            elif any(p < -1e-6 or p > 1 + 1e-6 for p in probs.values()):
                errors.append(f"{name} prediction has probabilities outside [0,1]")
            elif abs(total - 1.0) > 1e-3:
                errors.append(f"{name} prediction probabilities sum to {total}, not 1")
            if block["predicted_class"] not in probs:
                errors.append(f"{name} predicted_class not among its own probabilities")
        try:
            agent_timestamps.add(pd.Timestamp(block["observation_timestamp"]))
        except (ValueError, TypeError, KeyError):
            errors.append(f"{name} observation_timestamp is not a valid timestamp")

    if obs_ts is not None and agent_timestamps and agent_timestamps != {obs_ts}:
        errors.append(
            f"agent observation timestamps disagree with state timestamp: {agent_timestamps}"
        )
    elif len(agent_timestamps) > 1:
        errors.append(f"agents disagree on observation timestamp: {agent_timestamps}")

    expected_versions = {"direction", "volatility", "regime"}
    if not expected_versions.issubset(state.model_versions):
        errors.append(
            f"model_versions missing agent(s): {expected_versions - set(state.model_versions)}"
        )

    if not isinstance(state.feature_schema_version, str) or not state.feature_schema_version:
        errors.append("feature_schema_version is missing or empty")

    for k, v in (state.market_features or {}).items():
        if isinstance(v, float) and np.isinf(v):
            errors.append(f"market_features['{k}'] is infinite")

    if state.portfolio is not None:
        errors.append("portfolio must be None in Phase 2 (no Decision Engine yet)")

    return errors
