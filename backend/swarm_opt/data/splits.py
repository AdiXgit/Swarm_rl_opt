"""Chronological split definitions for the locked MVP dataset.

Single source of truth. Every model, evaluation and report resolves splits
through here rather than restating dates, so a boundary cannot drift between
modules and quietly invalidate a comparison.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

DATASET_START = date(2021, 4, 1)
DATASET_END = date(2026, 8, 18)


@dataclass(frozen=True)
class Split:
    name: str
    start: date
    end: date

    def contains(self, day: date) -> bool:
        return self.start <= day <= self.end

    def filter(self, dates):
        return [d for d in dates if self.contains(d)]


TRAIN = Split("train", date(2021, 4, 1), date(2024, 12, 31))
VALIDATION = Split("validation", date(2025, 1, 1), date(2025, 8, 31))
TEST = Split("test", date(2025, 9, 1), date(2026, 8, 18))

SPLITS: tuple[Split, ...] = (TRAIN, VALIDATION, TEST)

#: NSE moved the NIFTY weekly expiry from Thursday to Tuesday here. The primary
#: TEST lies wholly inside the Tuesday regime; docs/EXPERIMENTS.md Experiment 4
#: re-runs the comparison inside the Thursday regime.
EXPIRY_REGIME_CHANGE = date(2025, 9, 1)

#: NSE switched the F&O bhavcopy to the UDiFF format here. Deliberately inside
#: TRAIN so neither provenance coincides with a split boundary.
SCHEMA_CHANGE = date(2024, 1, 2)


def split_for(day: date) -> str | None:
    for split in SPLITS:
        if split.contains(day):
            return split.name
    return None


def episode_split(episode) -> str | None:
    """Split an episode belongs to, or None if it must be purged.

    A weekly episode decided in the last days of one split resolves in the
    next: its reward is computed from EOD marks on the far side of the wall.
    Assigning it by decision date alone would let a handful of validation and
    test observations bleed into training. Such episodes are purged instead.

    The cost is small and known -- see `PURGED_EPISODE_NOTE` -- and is reported
    in the dataset manifest rather than absorbed silently.
    """
    start = split_for(episode.decision_date)
    end = split_for(episode.exit_date)
    if start is None or end is None or start != end:
        return None
    return start


def purge_straddling(episodes):
    """Partition episodes into (kept, purged) at the split boundaries."""
    kept, purged = [], []
    for ep in episodes:
        (kept if episode_split(ep) is not None else purged).append(ep)
    return kept, purged


PURGED_EPISODE_NOTE = (
    "Episodes whose EOD mark window crosses a split boundary are dropped so no "
    "reward is computed from data belonging to a later split."
)


def expiry_regime(day: date) -> str:
    return "tuesday" if day >= EXPIRY_REGIME_CHANGE else "thursday"


def assert_well_formed() -> None:
    """Splits must be chronological, contiguous and cover the dataset exactly.

    A silent gap or overlap here would leak data between splits or drop
    episodes without anything downstream noticing.
    """
    assert SPLITS[0].start == DATASET_START, "train must start at the dataset start"
    assert SPLITS[-1].end == DATASET_END, "test must end at the dataset end"
    for split in SPLITS:
        assert split.start <= split.end, f"{split.name}: inverted range"
    for earlier, later in zip(SPLITS, SPLITS[1:]):
        assert earlier.end < later.start, f"{earlier.name}/{later.name}: overlap"
        assert (later.start - earlier.end).days == 1, (
            f"{earlier.name}/{later.name}: gap between splits"
        )
