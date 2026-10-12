from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from swarm_opt.data.calendar import WeeklyEpisode
from swarm_opt.features.audit import (
    LeakageError,
    assert_duplicate_free,
    assert_feature_label_alignment,
    assert_no_cross_split_labels,
    assert_no_future_features,
    build_temporal_audit,
)
from swarm_opt.features.pointintime import build_market_features
from swarm_opt.features.targets import TargetThresholds, build_targets


def _hist(n=30):
    dates = pd.bdate_range("2024-01-01", periods=n)
    closes = [100.0 + i for i in range(n)]
    return pd.DataFrame({"date": dates, "open": closes, "high": closes, "low": closes, "close": closes})


def test_assert_no_future_features_catches_injected_column():
    feats, _ = build_market_features(_hist())
    injected = feats.copy()
    injected["future_return_5"] = 0.0
    with pytest.raises(LeakageError):
        assert_no_future_features(injected)


def test_assert_no_future_features_passes_clean_frame():
    feats, _ = build_market_features(_hist())
    assert_no_future_features(feats)  # must not raise


def test_assert_feature_label_alignment_catches_missing_decision_date():
    feats, _ = build_market_features(_hist())
    targets = pd.DataFrame({
        "decision_date": [date(2099, 1, 1)],
        "exit_date": [date(2099, 1, 5)],
    })
    with pytest.raises(LeakageError):
        assert_feature_label_alignment(feats, targets)


def test_assert_feature_label_alignment_catches_inverted_horizon():
    feats, _ = build_market_features(_hist())
    d = feats["date"].iloc[25].date()
    targets = pd.DataFrame({"decision_date": [d], "exit_date": [d - timedelta(days=1)]})
    with pytest.raises(LeakageError):
        assert_feature_label_alignment(feats, targets)


def test_assert_no_cross_split_labels_catches_boundary_straddle():
    # TRAIN ends 2024-12-31, VALIDATION starts 2025-01-01: an episode decided
    # in TRAIN but exiting in VALIDATION must be caught.
    targets = pd.DataFrame({
        "decision_date": [date(2024, 12, 30)],
        "exit_date": [date(2025, 1, 2)],
    })
    with pytest.raises(LeakageError):
        assert_no_cross_split_labels(targets)


def test_assert_no_cross_split_labels_passes_within_one_split():
    targets = pd.DataFrame({
        "decision_date": [date(2024, 6, 1)],
        "exit_date": [date(2024, 6, 5)],
    })
    assert_no_cross_split_labels(targets)  # must not raise


def test_assert_duplicate_free_catches_duplicate_dates():
    frame = pd.DataFrame({"date": [date(2024, 1, 1), date(2024, 1, 1)]})
    with pytest.raises(LeakageError):
        assert_duplicate_free(frame, "date")


def test_temporal_audit_report_records_exclusions_and_class_counts():
    frame = pd.DataFrame({"direction_label": ["BULLISH", "BULLISH", "NEUTRAL"]})
    audit = build_temporal_audit(
        agent_name="direction",
        feature_cutoff="decision_date",
        label_horizon="weekly",
        train_range=(date(2021, 4, 1), date(2024, 12, 31)),
        rows_removed_lookback=20,
        rows_removed_incomplete_target=5,
        rows_removed_purge=2,
        split_frames={"train": frame},
        label_column="direction_label",
    )
    assert audit.final_counts == {"train": 3}
    assert audit.class_distribution == {"train": {"BULLISH": 2, "NEUTRAL": 1}}
    assert len(audit.exclusions) == 3


def test_real_calendar_episode_split_purging_leaves_no_boundary_crossers():
    """Regression: the frozen `swarm_opt.data.splits.purge_straddling` must
    already guarantee what `assert_no_cross_split_labels` checks, for actual
    weekly episodes, not just the synthetic cases above."""
    from swarm_opt.data.splits import purge_straddling

    episodes = [
        WeeklyEpisode(
            decision_date=date(2024, 12, 27), expiry=date(2025, 1, 2),
            exit_date=date(2025, 1, 2), mark_dates=(date(2025, 1, 2),),
        ),
        WeeklyEpisode(
            decision_date=date(2024, 12, 20), expiry=date(2024, 12, 26),
            exit_date=date(2024, 12, 26), mark_dates=(date(2024, 12, 26),),
        ),
    ]
    kept, purged = purge_straddling(episodes)
    assert len(purged) == 1 and purged[0].decision_date == date(2024, 12, 27)

    targets = pd.DataFrame({
        "decision_date": [e.decision_date for e in kept],
        "exit_date": [e.exit_date for e in kept],
    })
    assert_no_cross_split_labels(targets)  # must not raise
