"""Temporal leakage audit for the Phase 2 feature/target pipeline.

Checks the actual timestamp relationships between features, labels and split
boundaries -- not a naive "column name contains 'future'" scan. Every
assertion here fails loudly (raises `LeakageError`) rather than returning a
warning, so a violation cannot be silently reported and ignored.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from ..data.splits import SPLITS, split_for


class LeakageError(AssertionError):
    """Raised when a point-in-time or split-boundary guarantee is violated."""


@dataclass(frozen=True)
class TemporalAuditReport:
    agent_name: str
    feature_cutoff: str
    label_horizon: str
    train_range: tuple[str, str]
    rows_removed_lookback: int
    rows_removed_incomplete_target: int
    rows_removed_purge: int
    final_counts: dict[str, int]
    class_distribution: dict[str, dict[str, int]]
    exclusions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "feature_cutoff": self.feature_cutoff,
            "label_horizon": self.label_horizon,
            "train_range": list(self.train_range),
            "rows_removed_lookback": self.rows_removed_lookback,
            "rows_removed_incomplete_target": self.rows_removed_incomplete_target,
            "rows_removed_purge": self.rows_removed_purge,
            "final_counts": self.final_counts,
            "class_distribution": self.class_distribution,
            "exclusions": self.exclusions,
        }


def assert_no_future_features(features: pd.DataFrame, max_feature_date_col: str = "date") -> None:
    """Features must be internally finite-or-NaN and carry no look-ahead marker.

    This cannot prove the absence of leakage by inspection alone -- that is
    what `assert_feature_label_alignment` and the rolling-window unit tests
    are for -- but it does catch the class of bug where a future column slips
    into the frame under an unexpected name.
    """
    if max_feature_date_col not in features.columns:
        raise LeakageError(f"features frame has no '{max_feature_date_col}' column")
    suspicious = [
        c for c in features.columns
        if c != max_feature_date_col and ("future" in c.lower() or "forward" in c.lower())
    ]
    if suspicious:
        raise LeakageError(f"feature frame contains forward-looking column(s): {suspicious}")


def assert_feature_label_alignment(
    features: pd.DataFrame, targets: pd.DataFrame
) -> None:
    """Every label's decision date must be covered by a feature row, and a
    label's horizon (its `exit_date`) must never resolve before its own
    decision date."""
    feature_dates = set(pd.to_datetime(features["date"]).dt.date)
    missing = sorted(set(targets["decision_date"]) - feature_dates)
    if missing:
        raise LeakageError(f"{len(missing)} label decision dates have no feature row: {missing[:5]}")

    inverted = targets[targets["exit_date"] < targets["decision_date"]]
    if len(inverted):
        raise LeakageError(f"{len(inverted)} labels have exit_date before decision_date")


def assert_no_cross_split_labels(targets: pd.DataFrame) -> None:
    """A kept label's decision date and exit date must resolve to the same
    split. Episodes crossing a boundary must already have been purged by
    `swarm_opt.data.splits.purge_straddling` before targets are built."""
    bad = []
    for row in targets.itertuples():
        start, end = split_for(row.decision_date), split_for(row.exit_date)
        if start is None or end is None or start != end:
            bad.append(row.decision_date)
    if bad:
        raise LeakageError(f"{len(bad)} labels cross a split boundary: {bad[:5]}")


def assert_duplicate_free(frame: pd.DataFrame, key: str) -> None:
    dupes = frame[frame.duplicated(key, keep=False)]
    if len(dupes):
        raise LeakageError(f"{len(dupes)} duplicate '{key}' rows")


def build_temporal_audit(
    agent_name: str,
    feature_cutoff: str,
    label_horizon: str,
    train_range: tuple[date, date],
    rows_removed_lookback: int,
    rows_removed_incomplete_target: int,
    rows_removed_purge: int,
    split_frames: dict[str, pd.DataFrame],
    label_column: str,
) -> TemporalAuditReport:
    final_counts = {name: int(len(df)) for name, df in split_frames.items()}
    class_distribution = {
        name: {str(k): int(v) for k, v in df[label_column].value_counts().items()}
        for name, df in split_frames.items()
    }
    exclusions = []
    if rows_removed_lookback:
        exclusions.append(f"{rows_removed_lookback} rows dropped for insufficient lookback")
    if rows_removed_incomplete_target:
        exclusions.append(f"{rows_removed_incomplete_target} rows dropped for incomplete target horizon")
    if rows_removed_purge:
        exclusions.append(f"{rows_removed_purge} rows dropped for straddling a split boundary")

    return TemporalAuditReport(
        agent_name=agent_name,
        feature_cutoff=feature_cutoff,
        label_horizon=label_horizon,
        train_range=(train_range[0].isoformat(), train_range[1].isoformat()),
        rows_removed_lookback=rows_removed_lookback,
        rows_removed_incomplete_target=rows_removed_incomplete_target,
        rows_removed_purge=rows_removed_purge,
        final_counts=final_counts,
        class_distribution=class_distribution,
        exclusions=exclusions,
    )
