from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swarm_opt.agents.base import AgentError, AgentPrediction
from swarm_opt.agents.direction import DirectionAgent
from swarm_opt.agents.volatility import VolatilityAgent


def _synthetic_Xy(n=60, seed=0):
    rng = np.random.default_rng(seed)
    from swarm_opt.features.pointintime import FEATURE_NAMES
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    y = pd.Series(rng.choice(["BEARISH", "NEUTRAL", "BULLISH"], size=n))
    ts = pd.Series(pd.bdate_range("2024-01-01", periods=n))
    return X, y, ts


def _tiny_agent(**kw):
    return DirectionAgent(seed=0, params={"n_estimators": 5, "max_depth": 2, **kw})


def test_fit_predict_round_trip_valid_predictions():
    X, y, ts = _synthetic_Xy()
    agent = _tiny_agent()
    agent.fit(X, y, "fsv1", ("2024-01-01", "2024-03-01"))
    preds = agent.predict(X, ts)
    assert len(preds) == len(X)
    for p in preds:
        assert set(p.probabilities) == set(DirectionAgent.CLASSES)
        assert abs(sum(p.probabilities.values()) - 1.0) < 1e-6
        assert p.predicted_class in DirectionAgent.CLASSES
        assert np.isfinite(p.confidence)


def test_predictions_are_deterministic_for_fixed_artifact_and_input():
    X, y, ts = _synthetic_Xy()
    agent = _tiny_agent()
    agent.fit(X, y, "fsv1", ("2024-01-01", "2024-03-01"))
    p1 = agent.predict(X, ts)
    p2 = agent.predict(X, ts)
    assert [p.probabilities for p in p1] == [p.probabilities for p in p2]


def test_timestamp_propagates_to_prediction():
    X, y, ts = _synthetic_Xy(n=10)
    agent = _tiny_agent()
    agent.fit(X, y, "fsv1", ("2024-01-01", "2024-03-01"))
    preds = agent.predict(X, ts)
    for pred, t in zip(preds, ts):
        assert pred.observation_timestamp == pd.Timestamp(t).isoformat()


def test_fit_rejects_label_outside_classes():
    X, y, ts = _synthetic_Xy(n=10)
    y.iloc[0] = "SIDEWAYS"
    agent = _tiny_agent()
    with pytest.raises(AgentError):
        agent.fit(X, y, "fsv1", ("2024-01-01", "2024-03-01"))


def test_predict_rejects_missing_feature_column():
    X, y, ts = _synthetic_Xy()
    agent = _tiny_agent()
    agent.fit(X, y, "fsv1", ("2024-01-01", "2024-03-01"))
    with pytest.raises(AgentError):
        agent.predict(X.drop(columns=[X.columns[0]]), ts)


def test_predict_before_fit_or_load_raises():
    with pytest.raises(AgentError):
        DirectionAgent().predict(pd.DataFrame(), pd.Series(dtype="datetime64[ns]"))


def test_training_fold_missing_a_class_does_not_crash():
    """Real occurrence with ~190 weekly TRAIN rows: a fold can lack a class
    entirely. Must degrade gracefully, not raise, and must be recorded."""
    X, y, ts = _synthetic_Xy(n=20)
    y[:] = "NEUTRAL"
    y.iloc[0] = "BULLISH"  # BEARISH entirely absent
    agent = _tiny_agent()
    agent.fit(X, y, "fsv1", ("2024-01-01", "2024-03-01"))
    assert agent._missing_classes_in_fit == ["BEARISH"]
    preds = agent.predict(X, ts)
    for p in preds:
        p.validate()  # still produces valid, finite, sum-to-one probabilities


def test_save_load_round_trip_reproduces_predictions(tmp_path):
    X, y, ts = _synthetic_Xy()
    agent = _tiny_agent()
    agent.fit(X, y, "fsv1", ("2024-01-01", "2024-03-01"))
    before = agent.predict(X, ts)

    agent.save(tmp_path / "direction")
    loaded = DirectionAgent.load(tmp_path / "direction")
    after = loaded.predict(X, ts)

    assert [p.probabilities for p in before] == [p.probabilities for p in after]
    assert [p.predicted_class for p in before] == [p.predicted_class for p in after]


def test_load_rejects_incompatible_classes(tmp_path):
    X, y, ts = _synthetic_Xy()
    agent = _tiny_agent()
    agent.fit(X, y, "fsv1", ("2024-01-01", "2024-03-01"))
    agent.save(tmp_path / "direction")
    with pytest.raises(AgentError):
        VolatilityAgent.load(tmp_path / "direction")  # wrong CLASSES


def test_load_missing_artifact_raises(tmp_path):
    with pytest.raises(AgentError):
        DirectionAgent.load(tmp_path / "nonexistent")


def test_agent_prediction_validate_rejects_bad_probabilities():
    bad = AgentPrediction(
        agent_name="direction", model_version="xgb_v1", observation_timestamp="2024-01-01T00:00:00",
        predicted_class="BULLISH", probabilities={"BULLISH": 0.9, "NEUTRAL": 0.3}, confidence=0.1,
        data_quality={},
    )
    with pytest.raises(AgentError):
        bad.validate()


def test_agent_prediction_validate_rejects_nonfinite_probability():
    bad = AgentPrediction(
        agent_name="direction", model_version="xgb_v1", observation_timestamp="2024-01-01T00:00:00",
        predicted_class="BULLISH", probabilities={"BULLISH": float("nan"), "NEUTRAL": 0.5, "BEARISH": 0.5},
        confidence=0.1, data_quality={},
    )
    with pytest.raises(AgentError):
        bad.validate()


def test_agent_prediction_validate_rejects_predicted_class_not_in_probabilities():
    bad = AgentPrediction(
        agent_name="direction", model_version="xgb_v1", observation_timestamp="2024-01-01T00:00:00",
        predicted_class="SIDEWAYS", probabilities={"BULLISH": 0.5, "NEUTRAL": 0.3, "BEARISH": 0.2},
        confidence=0.1, data_quality={},
    )
    with pytest.raises(AgentError):
        bad.validate()


def test_predict_flags_rows_with_missing_feature_values():
    X, y, ts = _synthetic_Xy(n=10)
    agent = _tiny_agent()
    agent.fit(X, y, "fsv1", ("2024-01-01", "2024-03-01"))
    X_with_nan = X.copy()
    X_with_nan.iloc[0, 0] = np.nan
    preds = agent.predict(X_with_nan, ts)
    assert preds[0].data_quality["had_missing_feature"] is True
    assert preds[1].data_quality["had_missing_feature"] is False
