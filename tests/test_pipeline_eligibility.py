"""Phase 2.2 Task A2: explicit `has_full_lookback` enforcement in
`pipeline.make_xy` / `pipeline.eligibility_report`.

Lookback eligibility and target-label availability are two independent
conditions. These tests construct a scenario where they actually diverge
(unlike the real frozen dataset, where they happen to coincide -- see
`tests/test_pipeline_eligibility.py::test_real_dataset_eligible_counts_unchanged_from_baseline`
below) to prove the lookback filter does something, not just that it is a
no-op that happens to match the old behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import pytest

from swarm_opt.data.providers.local_parquet_provider import LocalParquetProvider
from swarm_opt.data.splits import DATASET_END, DATASET_START
from swarm_opt.features.pointintime import FEATURE_NAMES, MAX_LOOKBACK_DAYS, build_market_features
from swarm_opt.pipeline import Dataset, build_dataset, eligibility_report, make_xy, split_targets


def _underlying_hist(n=30, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    return pd.DataFrame({"date": dates, "open": close, "high": close, "low": close, "close": close})


def _minimal_dataset(features: pd.DataFrame) -> Dataset:
    """A `Dataset` stub with just enough populated to exercise `make_xy` /
    `eligibility_report`, which only read `.features`."""
    return Dataset(
        underlying="NIFTY", underlying_hist=pd.DataFrame(), features=features,
        calendar=None, episodes_kept=[], episodes_purged=[], thresholds=None,
        targets=pd.DataFrame(), feature_build_report=None,
    )


def _targets_for(dates: list[date], labels: list[str], complete: list[bool]) -> pd.DataFrame:
    return pd.DataFrame({
        "decision_date": dates,
        "exit_date": dates,
        "direction_label": labels,
        "complete": complete,
    })


def test_insufficient_lookback_row_excluded_even_with_a_complete_target():
    """The actual divergence case: construct a target row that is
    `complete=True` (target-side) but whose decision date falls inside the
    first MAX_LOOKBACK_DAYS rows (feature-side NOT ok). Pre-A2, make_xy
    would have used this row; post-A2 it must not."""
    hist = _underlying_hist(n=30)
    features, _ = build_market_features(hist)
    dataset = _minimal_dataset(features)

    early_date = features["date"].iloc[5].date()   # inside the 20-day lookback window
    late_date = features["date"].iloc[25].date()    # outside it
    targets = _targets_for(
        [early_date, late_date], ["BULLISH", "BEARISH"], [True, True]
    )

    X, y, ts = make_xy(dataset, targets, "direction_label")
    assert len(X) == 1
    assert pd.Timestamp(late_date) in pd.to_datetime(ts).tolist()
    assert pd.Timestamp(early_date) not in pd.to_datetime(ts).tolist()

    report = eligibility_report(dataset, targets, "direction_label", "direction")
    assert report.excluded_insufficient_lookback_among_target_complete == 1
    assert report.used == 1
    assert "real count change" in report.exclusion_overlap_note


def test_lookback_eligibility_is_independent_of_target_completeness():
    """A row with insufficient lookback AND an incomplete target must be
    counted once by `excluded_target_incomplete`, not double-counted into
    the lookback bucket -- the two reasons must partition, not overlap."""
    hist = _underlying_hist(n=30)
    features, _ = build_market_features(hist)
    dataset = _minimal_dataset(features)

    early_date = features["date"].iloc[3].date()
    targets = _targets_for([early_date], ["UNKNOWN"], [False])

    report = eligibility_report(dataset, targets, "direction_label", "direction")
    assert report.excluded_target_incomplete == 1
    assert report.excluded_insufficient_lookback_among_target_complete == 0
    assert report.used == 0


def test_boundary_row_at_exactly_max_lookback_days_is_eligible():
    hist = _underlying_hist(n=30)
    features, _ = build_market_features(hist)
    dataset = _minimal_dataset(features)

    boundary_date = features["date"].iloc[MAX_LOOKBACK_DAYS].date()  # first eligible row
    just_before_date = features["date"].iloc[MAX_LOOKBACK_DAYS - 1].date()  # last ineligible row

    targets = _targets_for(
        [just_before_date, boundary_date], ["BULLISH", "BULLISH"], [True, True]
    )
    X, y, ts = make_xy(dataset, targets, "direction_label")
    assert len(X) == 1
    assert pd.Timestamp(boundary_date) in pd.to_datetime(ts).tolist()


def test_decision_date_absent_from_features_treated_as_ineligible_not_a_crash():
    hist = _underlying_hist(n=30)
    features, _ = build_market_features(hist)
    dataset = _minimal_dataset(features)

    missing_date = date(2099, 1, 1)
    targets = _targets_for([missing_date], ["BULLISH"], [True])
    X, y, ts = make_xy(dataset, targets, "direction_label")
    assert len(X) == 0

    report = eligibility_report(dataset, targets, "direction_label", "direction")
    assert report.used == 0


def test_make_xy_and_eligibility_report_agree_on_used_count():
    """The two independently-computed paths (selection vs. reporting) must
    never disagree -- that is the entire point of computing them
    separately."""
    hist = _underlying_hist(n=40, seed=2)
    features, _ = build_market_features(hist)
    dataset = _minimal_dataset(features)

    dates = [features["date"].iloc[i].date() for i in (5, 10, 19, 20, 25, 35)]
    targets = _targets_for(dates, ["BULLISH"] * len(dates), [True] * len(dates))

    X, y, ts = make_xy(dataset, targets, "direction_label")
    report = eligibility_report(dataset, targets, "direction_label")
    assert len(X) == report.used


def test_real_dataset_eligible_counts_unchanged_from_baseline():
    """Integration: on the real frozen dataset, the A2 hardening must not
    change the 189 TRAIN / 34 VALIDATION counts baked into
    data/models/phase2_training_report.json for any of the three agents --
    confirmed here by direct computation, not assumed."""
    from pathlib import Path

    data_root = Path(__file__).resolve().parents[1] / "data" / "processed" / "nifty_eod"
    atm_path = Path(__file__).resolve().parents[1] / "data" / "features" / "atm_iv_history.parquet"
    if not data_root.exists() or not atm_path.exists():
        pytest.skip("frozen dataset / ATM IV cache not present in this environment")

    provider = LocalParquetProvider(data_root)
    atm_iv_hist = pd.read_parquet(atm_path)
    dataset = build_dataset(provider, "NIFTY", DATASET_START, DATASET_END, atm_iv_hist)
    by_split = split_targets(dataset)

    expected = {"train": 189, "validation": 34}
    for agent, col in (
        ("direction", "direction_label"),
        ("volatility", "volatility_label"),
        ("regime", "regime_label"),
    ):
        for split_name, expected_n in expected.items():
            X, y, ts = make_xy(dataset, by_split[split_name], col)
            report = eligibility_report(dataset, by_split[split_name], col, agent)
            assert len(X) == expected_n, f"{agent}/{split_name}: expected {expected_n}, got {len(X)}"
            assert report.excluded_insufficient_lookback_among_target_complete == 0
