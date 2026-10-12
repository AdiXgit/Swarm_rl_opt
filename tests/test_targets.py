from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from swarm_opt.data.calendar import WeeklyEpisode
from swarm_opt.features.targets import (
    TargetThresholds,
    build_targets,
    direction_label,
    fit_thresholds_on_train,
    regime_label,
    volatility_label,
)

THR = TargetThresholds(bullish_return=0.02, bearish_return=0.02)


def test_direction_label_boundaries():
    assert direction_label(0.021, THR) == "BULLISH"
    assert direction_label(0.02, THR) == "NEUTRAL"  # strictly greater required
    assert direction_label(-0.021, THR) == "BEARISH"
    assert direction_label(0.0, THR) == "NEUTRAL"
    assert direction_label(float("nan"), THR) == "UNKNOWN"


def test_volatility_label_bands():
    assert volatility_label(1.11, THR) == "EXPAND"
    assert volatility_label(0.89, THR) == "CONTRACT"
    assert volatility_label(1.0, THR) == "NEUTRAL"
    assert volatility_label(float("nan"), THR) == "UNKNOWN"


def test_regime_label_priority_vol_extremes_before_trend():
    # Strongly trending AND high-vol: high-vol must win by the documented
    # priority rule, never double-counted as a trend regime.
    assert regime_label(forward_return=0.05, vol_ratio=1.5, thresholds=THR) == "HIGH_VOLATILITY"
    assert regime_label(forward_return=-0.05, vol_ratio=0.5, thresholds=THR) == "LOW_VOLATILITY"
    assert regime_label(forward_return=0.03, vol_ratio=1.0, thresholds=THR) == "TRENDING_UP"
    assert regime_label(forward_return=-0.03, vol_ratio=1.0, thresholds=THR) == "TRENDING_DOWN"
    assert regime_label(forward_return=0.0, vol_ratio=1.0, thresholds=THR) == "RANGE_BOUND"


def test_fit_thresholds_on_train_is_half_std():
    returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.0])
    thr = fit_thresholds_on_train(returns)
    expected = 0.5 * returns.std(ddof=0)
    assert thr.bullish_return == pytest.approx(expected)
    assert thr.bearish_return == pytest.approx(expected)


def test_fit_thresholds_requires_at_least_two_observations():
    with pytest.raises(ValueError):
        fit_thresholds_on_train(pd.Series([0.01]))


def _hist(dates: list[date], closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(dates), "open": closes, "high": closes, "low": closes, "close": closes,
    })


def test_build_targets_manual_forward_return_and_vol():
    d0 = date(2024, 1, 1)
    days = [d0 + timedelta(days=i) for i in range(6)]
    closes = [100.0, 101.0, 102.0, 101.0, 103.0, 104.0]
    hist = _hist(days, closes)

    episode = WeeklyEpisode(
        decision_date=days[0], expiry=days[5], exit_date=days[5], mark_dates=tuple(days[1:])
    )
    baseline_vol = pd.Series({days[0]: 0.20})
    thresholds = TargetThresholds(bullish_return=0.01, bearish_return=0.01)

    targets = build_targets([episode], hist, baseline_vol, thresholds)
    row = targets.iloc[0]

    assert row["complete"]
    assert row["forward_return"] == pytest.approx(104.0 / 100.0 - 1.0)
    expected_log_ret = np.diff(np.log(closes))
    expected_vol = np.std(expected_log_ret, ddof=0) * np.sqrt(252)
    assert row["forward_realized_vol"] == pytest.approx(expected_vol)
    assert row["vol_ratio"] == pytest.approx(expected_vol / 0.20)


def test_build_targets_incomplete_window_is_unknown_not_dropped():
    d0 = date(2024, 1, 1)
    days = [d0 + timedelta(days=i) for i in range(4)]
    hist = _hist(days[:3], [100.0, 101.0, 102.0])  # day 4 missing from history

    episode = WeeklyEpisode(decision_date=days[0], expiry=days[3], exit_date=days[3], mark_dates=tuple(days[1:]))
    baseline_vol = pd.Series({days[0]: 0.2})
    targets = build_targets([episode], hist, baseline_vol, TargetThresholds(0.01, 0.01))

    row = targets.iloc[0]
    assert not row["complete"]
    assert row["direction_label"] == "UNKNOWN"
    assert row["volatility_label"] == "UNKNOWN"
    assert row["regime_label"] == "UNKNOWN"
    assert len(targets) == 1  # kept, not dropped


def test_build_targets_missing_baseline_vol_is_unknown():
    d0 = date(2024, 1, 1)
    days = [d0 + timedelta(days=i) for i in range(3)]
    hist = _hist(days, [100.0, 101.0, 102.0])
    episode = WeeklyEpisode(decision_date=days[0], expiry=days[2], exit_date=days[2], mark_dates=tuple(days[1:]))
    targets = build_targets([episode], hist, pd.Series(dtype=float), TargetThresholds(0.01, 0.01))
    assert targets.iloc[0]["volatility_label"] == "UNKNOWN"
    assert targets.iloc[0]["regime_label"] == "UNKNOWN"
