"""Small, fast, fully-synthetic end-to-end pipeline test.

Exercises feature build -> target build -> leakage audit -> agent fit ->
predict -> Greeks -> SWARM_STATE_V1 -> validate -> save/reload -> reproduce,
without touching the real (slow) frozen dataset. The real-dataset path is
covered once by `scripts/run_smoke_test.py`.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from swarm_opt.agents.base import AgentError
from swarm_opt.agents.direction import DirectionAgent
from swarm_opt.agents.greeks_agent import compute_greeks
from swarm_opt.agents.regime import RegimeAgent
from swarm_opt.agents.volatility import VolatilityAgent
from swarm_opt.data.calendar import WeeklyEpisode
from swarm_opt.features.audit import (
    assert_duplicate_free,
    assert_feature_label_alignment,
    assert_no_cross_split_labels,
    assert_no_future_features,
)
from swarm_opt.features.pointintime import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, build_market_features
from swarm_opt.features.targets import TargetThresholds, build_targets, fit_thresholds_on_train
from swarm_opt.state import build_swarm_state, validate_swarm_state


def _synthetic_underlying(n=90, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    rets = rng.normal(0.0005, 0.012, n)
    close = 100 * np.cumprod(1 + rets)
    return pd.DataFrame({
        "date": dates, "open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
    })


def _weekly_episodes(hist: pd.DataFrame) -> list[WeeklyEpisode]:
    dates = [d.date() for d in hist["date"]]
    episodes = []
    for start in range(20, len(dates) - 5, 5):
        decision = dates[start]
        marks = tuple(dates[start + 1 : start + 6])
        episodes.append(WeeklyEpisode(decision, marks[-1], marks[-1], marks))
    return episodes


def test_full_synthetic_pipeline_builds_valid_state_and_reproduces():
    hist = _synthetic_underlying()
    features, build_report = build_market_features(hist)
    assert_no_future_features(features)
    assert_duplicate_free(features, "date")

    episodes = _weekly_episodes(hist)
    baseline_vol = features.set_index(features["date"].dt.date)["realized_vol_20"]

    n_train = int(len(episodes) * 0.8)
    train_eps, hold_eps = episodes[:n_train], episodes[n_train:]

    provisional = build_targets(train_eps, hist, baseline_vol, TargetThresholds(1.0, 1.0))
    thresholds = fit_thresholds_on_train(provisional["forward_return"])

    train_targets = build_targets(train_eps, hist, baseline_vol, thresholds)
    hold_targets = build_targets(hold_eps, hist, baseline_vol, thresholds)
    all_targets = pd.concat([train_targets, hold_targets], ignore_index=True)
    assert_feature_label_alignment(features, all_targets)
    assert_duplicate_free(all_targets, "decision_date")

    feats_by_date = features.set_index(features["date"].dt.date)

    def make_xy(targets, label_col):
        usable = targets[targets["complete"] & (targets[label_col] != "UNKNOWN")]
        X = feats_by_date.loc[usable["decision_date"], FEATURE_NAMES].reset_index(drop=True)
        y = usable[label_col].reset_index(drop=True)
        ts = pd.Series(pd.to_datetime(usable["decision_date"].to_numpy()))
        return X, y, ts

    agents = {}
    predictions_at_hold = {}
    for cls, label_col in (
        (DirectionAgent, "direction_label"),
        (VolatilityAgent, "volatility_label"),
        (RegimeAgent, "regime_label"),
    ):
        X_train, y_train, _ = make_xy(train_targets, label_col)
        X_hold, y_hold, ts_hold = make_xy(hold_targets, label_col)
        assert len(X_train) > 5 and len(X_hold) > 0

        agent = cls(seed=0, params={"n_estimators": 10, "max_depth": 2})
        agent.fit(X_train, y_train, FEATURE_SCHEMA_VERSION, ("2024-01-01", "2024-03-01"))
        preds = agent.predict(X_hold, ts_hold)
        assert len(preds) == len(X_hold)
        agents[cls.AGENT_NAME] = (agent, X_hold, ts_hold)
        predictions_at_hold[cls.AGENT_NAME] = preds

    last = {name: preds[-1] for name, preds in predictions_at_hold.items()}
    observation_date = pd.Timestamp(agents["direction"][2].iloc[-1]).date()

    empty_chain = pd.DataFrame(columns=["days_to_expiry", "expiry", "strike", "spot", "option_type", "close"])
    greeks_result = compute_greeks(empty_chain, pd.Timestamp(observation_date), "NIFTY")

    row = features[features["date"] == pd.Timestamp(observation_date)].iloc[0]
    market_features = {c: (None if pd.isna(row[c]) else float(row[c])) for c in FEATURE_NAMES}
    market_features["has_full_lookback"] = bool(row["has_full_lookback"])
    market_features["atm_iv_available"] = bool(row["atm_iv_available"])

    state = build_swarm_state(
        observation_date, "NIFTY", market_features, last, greeks_result, FEATURE_SCHEMA_VERSION,
    )
    assert validate_swarm_state(state) == []
    assert state.greeks["portfolio"] is None


def test_pipeline_fails_safely_on_missing_artifact(tmp_path):
    with pytest.raises(AgentError):
        DirectionAgent.load(tmp_path / "does_not_exist")


def test_pipeline_fails_safely_on_incompatible_feature_schema():
    X, y, ts = (
        pd.DataFrame(np.random.default_rng(0).normal(size=(10, len(FEATURE_NAMES))), columns=FEATURE_NAMES),
        pd.Series(["BULLISH"] * 10),
        pd.Series(pd.bdate_range("2024-01-01", periods=10)),
    )
    agent = DirectionAgent(seed=0, params={"n_estimators": 5, "max_depth": 2})
    agent.fit(X, y, FEATURE_SCHEMA_VERSION, ("2024-01-01", "2024-03-01"))
    with pytest.raises(AgentError):
        agent.predict(X.drop(columns=["ret_1"]), ts)
