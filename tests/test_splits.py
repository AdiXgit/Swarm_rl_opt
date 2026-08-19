"""Chronological split definitions for the locked MVP dataset.

These are cheap assertions about dates, but they guard the thing that would
invalidate every Week-4 number: a split boundary that overlaps, leaves a gap,
or drifts away from what the audit and the reports assume.
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from swarm_opt.data.splits import (
    DATASET_END,
    DATASET_START,
    EXPIRY_REGIME_CHANGE,
    SCHEMA_CHANGE,
    SPLITS,
    TEST,
    TRAIN,
    VALIDATION,
    assert_well_formed,
    expiry_regime,
    split_for,
)


def test_locked_dataset_range():
    assert DATASET_START == date(2021, 4, 1)
    assert DATASET_END == date(2026, 8, 18)


def test_approved_boundaries_are_unchanged():
    assert (TRAIN.start, TRAIN.end) == (date(2021, 4, 1), date(2024, 12, 31))
    assert (VALIDATION.start, VALIDATION.end) == (date(2025, 1, 1), date(2025, 8, 31))
    assert (TEST.start, TEST.end) == (date(2025, 9, 1), date(2026, 8, 18))


def test_splits_are_well_formed():
    assert_well_formed()


def test_splits_are_chronological_contiguous_and_exhaustive():
    assert [s.name for s in SPLITS] == ["train", "validation", "test"]
    for earlier, later in zip(SPLITS, SPLITS[1:]):
        assert earlier.end < later.start
        assert later.start - earlier.end == timedelta(days=1)
    assert SPLITS[0].start == DATASET_START
    assert SPLITS[-1].end == DATASET_END


def test_every_day_in_range_lands_in_exactly_one_split():
    day = DATASET_START
    while day <= DATASET_END:
        hits = [s for s in SPLITS if s.contains(day)]
        assert len(hits) == 1, (day, hits)
        assert split_for(day) == hits[0].name
        day += timedelta(days=1)


def test_days_outside_the_range_belong_to_no_split():
    assert split_for(DATASET_START - timedelta(days=1)) is None
    assert split_for(DATASET_END + timedelta(days=1)) is None


# --------------------------------------------------------------------------- #
# structural boundaries
# --------------------------------------------------------------------------- #
def test_schema_change_sits_inside_train():
    """Deliberate: TRAIN must span both bhavcopy provenances.

    If the 2024-01-02 UDiFF switch were a split edge, a model could learn a
    legacy-only artifact and appear to fail at test time for the wrong reason.
    """
    assert TRAIN.contains(SCHEMA_CHANGE)
    assert SCHEMA_CHANGE not in (TRAIN.start, TRAIN.end)
    assert not VALIDATION.contains(SCHEMA_CHANGE)
    assert not TEST.contains(SCHEMA_CHANGE)


def test_expiry_regime_change_opens_the_test_split():
    """The Thursday->Tuesday move lands exactly on the validation/test edge.

    That is a known confound, not an oversight; docs/EXPERIMENTS.md makes the
    pre-2025-09 re-run mandatory because of it. This pins the boundary so the
    confound cannot silently move.
    """
    assert TEST.start == EXPIRY_REGIME_CHANGE
    assert VALIDATION.end == EXPIRY_REGIME_CHANGE - timedelta(days=1)


def test_test_split_is_entirely_one_expiry_regime():
    assert expiry_regime(TEST.start) == "tuesday"
    assert expiry_regime(TEST.end) == "tuesday"


def test_train_and_validation_are_entirely_the_thursday_regime():
    for split in (TRAIN, VALIDATION):
        assert expiry_regime(split.start) == "thursday"
        assert expiry_regime(split.end) == "thursday"


def test_a_thursday_regime_subperiod_exists_for_the_mandatory_rerun():
    """Experiment 4 needs enough pre-transition history to re-cut a full
    train/validation/test inside the Thursday regime."""
    span = (EXPIRY_REGIME_CHANGE - DATASET_START).days
    assert span > 3 * 365


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def test_filter_selects_only_in_range_dates():
    dates = [date(2021, 1, 1), date(2022, 6, 1), date(2025, 3, 1), date(2026, 8, 18)]
    assert TRAIN.filter(dates) == [date(2022, 6, 1)]
    assert VALIDATION.filter(dates) == [date(2025, 3, 1)]
    assert TEST.filter(dates) == [date(2026, 8, 18)]


@pytest.mark.parametrize(
    "day,expected",
    [
        (date(2021, 4, 1), "thursday"),
        (date(2025, 8, 31), "thursday"),
        (date(2025, 9, 1), "tuesday"),
        (date(2026, 8, 18), "tuesday"),
    ],
)
def test_expiry_regime_boundary(day, expected):
    assert expiry_regime(day) == expected
