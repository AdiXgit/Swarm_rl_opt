"""Phase 2.5: integration tests for the extended-dataset pipeline, using the
real artifacts produced by scripts/train_agents_extended.py if present.
Skips cleanly if the extension hasn't been built in this environment (it is
a large, separately-run collection step, not something a unit test should
ever trigger). Never touches TEST and never retrains.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from swarm_opt.agents.direction import DirectionAgent
from swarm_opt.agents.greeks_agent import compute_greeks
from swarm_opt.agents.regime import RegimeAgent
from swarm_opt.agents.volatility import VolatilityAgent
from swarm_opt.data.providers.combined_provider import CombinedParquetProvider
from swarm_opt.data.providers.local_parquet_provider import LocalParquetProvider
from swarm_opt.features.manifest import FeatureManifest
from swarm_opt.features.pointintime import FEATURE_NAMES_V2, FEATURE_SCHEMA_VERSION_V2, build_market_features
from swarm_opt.state import build_swarm_state, validate_swarm_state

ROOT = Path(__file__).resolve().parents[1]
EXTENSION_ROOT = ROOT / "data" / "processed" / "nifty_eod_ext_2019_2021"
ORIGINAL_ROOT = ROOT / "data" / "processed" / "nifty_eod"
EXTENSION_ATM_IV = ROOT / "data" / "features" / "atm_iv_history_ext_2019_2021.parquet"
ORIGINAL_ATM_IV = ROOT / "data" / "features" / "atm_iv_history.parquet"
ARTIFACTS = ROOT / "data" / "models" / "extended_2019_2021"

AGENT_CLASSES = {"direction": DirectionAgent, "volatility": VolatilityAgent, "regime": RegimeAgent}

pytestmark = pytest.mark.skipif(
    not ARTIFACTS.exists(), reason="extended artifacts not built in this environment"
)


@pytest.fixture(scope="module")
def combined_provider():
    return CombinedParquetProvider([LocalParquetProvider(EXTENSION_ROOT), LocalParquetProvider(ORIGINAL_ROOT)])


def test_extended_feature_manifest_matches_v2_schema():
    manifest = FeatureManifest.load(ARTIFACTS / "feature_manifest.json")
    assert manifest.feature_schema_version == FEATURE_SCHEMA_VERSION_V2
    assert manifest.feature_names == FEATURE_NAMES_V2
    assert "dist_sma20" not in manifest.feature_names


def test_all_three_extended_agents_share_the_v2_schema():
    for name, cls in AGENT_CLASSES.items():
        agent = cls.load(ARTIFACTS / name)
        assert agent.feature_schema_version == FEATURE_SCHEMA_VERSION_V2
        assert agent.feature_names == FEATURE_NAMES_V2


def test_extended_inference_builds_a_valid_state_inside_extension_period(combined_provider):
    from datetime import date
    on = date(2019, 6, 10)
    hist = combined_provider.underlying_history("NIFTY", date(2019, 2, 18), on)
    atm_iv = pd.concat(
        [pd.read_parquet(EXTENSION_ATM_IV), pd.read_parquet(ORIGINAL_ATM_IV)], ignore_index=True
    )
    atm_iv = atm_iv[atm_iv["date"] <= pd.Timestamp(on)]
    features, _ = build_market_features(hist, atm_iv_hist=atm_iv, feature_names=FEATURE_NAMES_V2)
    row = features[features["date"] == pd.Timestamp(on)]
    assert not row.empty
    X = row[FEATURE_NAMES_V2]
    ts = pd.Series([pd.Timestamp(on)])

    preds = {}
    for name, cls in AGENT_CLASSES.items():
        agent = cls.load(ARTIFACTS / name)
        preds[name] = agent.predict(X, ts)[0]

    chain = combined_provider.option_chain("NIFTY", on)
    greeks = compute_greeks(chain, pd.Timestamp(on), "NIFTY")

    mf = {c: float(row.iloc[0][c]) for c in FEATURE_NAMES_V2 if pd.notna(row.iloc[0][c])}
    mf["has_full_lookback"] = bool(row.iloc[0]["has_full_lookback"])
    mf["atm_iv_available"] = bool(row.iloc[0]["atm_iv_available"])

    state = build_swarm_state(on, "NIFTY", mf, preds, greeks, FEATURE_SCHEMA_VERSION_V2)
    assert validate_swarm_state(state) == []
    assert "dist_sma20" not in state.market_features


def test_extended_agent_rejects_v1_feature_frame():
    """A frame still carrying the dropped `dist_sma20` column (or any other
    v1-shaped input) must not silently work; the agent's own stored
    feature_names (v2, 17 columns) governs what it accepts."""
    from swarm_opt.agents.base import AgentError

    agent = DirectionAgent.load(ARTIFACTS / "direction")
    import numpy as np
    bad = pd.DataFrame(
        np.zeros((2, len(FEATURE_NAMES_V2) + 1)),
        columns=FEATURE_NAMES_V2 + ["dist_sma20"],
    )
    # Extra column is harmless (agent selects only what it needs)...
    ts = pd.Series(pd.bdate_range("2024-01-01", periods=2))
    agent.predict(bad, ts)  # must not raise: selecting known columns ignores the extra one

    missing = bad.drop(columns=[FEATURE_NAMES_V2[0]])
    with pytest.raises(AgentError):
        agent.predict(missing, ts)  # but a genuinely missing required column must raise
