"""End-to-end glue: dataset -> point-in-time features/targets -> agent-ready
tables, shared by `scripts/train_agents.py`, `scripts/run_smoke_test.py` and
the Phase 2 test suite so the three do not drift apart.

Nothing here fits anything on VALIDATION or TEST: `fit_thresholds_on_train`
is called with TRAIN targets only, by construction (see `build_dataset`).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from .data.calendar import ExpiryCalendar, WeeklyEpisode
from .data.providers.base import DataProvider
from .data.splits import SPLITS, purge_straddling, split_for
from .features.pointintime import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, build_market_features
from .features.targets import TARGET_SCHEMA_VERSION, TargetThresholds, build_targets, fit_thresholds_on_train


@dataclass
class Dataset:
    underlying: str
    underlying_hist: pd.DataFrame
    features: pd.DataFrame
    calendar: ExpiryCalendar
    episodes_kept: list[WeeklyEpisode]
    episodes_purged: list[WeeklyEpisode]
    thresholds: TargetThresholds
    targets: pd.DataFrame  # one row per kept episode, all splits
    feature_build_report: object


def build_dataset(
    provider: DataProvider,
    underlying: str,
    start: date,
    end: date,
    atm_iv_hist: pd.DataFrame | None = None,
) -> Dataset:
    hist = provider.underlying_history(underlying, start, end)
    if hist.empty:
        raise ValueError(f"no underlying history for {underlying} in [{start}, {end}]")

    features, report = build_market_features(hist, atm_iv_hist=atm_iv_hist)

    dates = provider.trading_dates(start, end)
    calendar = ExpiryCalendar.from_provider(provider, underlying, dates)
    all_episodes = calendar.weekly_episodes()
    kept, purged = purge_straddling(all_episodes)

    baseline_vol = features.set_index(features["date"].dt.date)["realized_vol_20"]

    train_episodes = [e for e in kept if split_for(e.decision_date) == "train"]
    if not train_episodes:
        raise ValueError("no TRAIN episodes available to fit target thresholds")
    provisional = build_targets(
        train_episodes, hist, baseline_vol, TargetThresholds(bullish_return=1.0, bearish_return=1.0)
    )
    thresholds = fit_thresholds_on_train(provisional["forward_return"])

    targets = build_targets(kept, hist, baseline_vol, thresholds)

    return Dataset(
        underlying=underlying,
        underlying_hist=hist,
        features=features,
        calendar=calendar,
        episodes_kept=kept,
        episodes_purged=purged,
        thresholds=thresholds,
        targets=targets,
        feature_build_report=report,
    )


def split_targets(dataset: Dataset) -> dict[str, pd.DataFrame]:
    out = {}
    for split in SPLITS:
        mask = dataset.targets["decision_date"].map(lambda d: split_for(d) == split.name)
        out[split.name] = dataset.targets[mask].reset_index(drop=True)
    return out


@dataclass(frozen=True)
class EligibilityReport:
    """Why a row did or did not make it into `make_xy`'s output.

    Phase 2.2 Task A2: lookback eligibility is tracked and reported
    separately from target-label availability, even though in the current
    frozen dataset every insufficient-lookback decision date also happens to
    have an incomplete target (see `exclusion_overlap_note`). Two different
    mechanisms produce that overlap today; neither is allowed to stand in
    for the other going forward.
    """

    agent_name: str
    total_candidate_rows: int
    excluded_target_incomplete: int
    excluded_insufficient_lookback_among_target_complete: int
    used: int
    exclusion_overlap_note: str

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "total_candidate_rows": self.total_candidate_rows,
            "excluded_target_incomplete": self.excluded_target_incomplete,
            "excluded_insufficient_lookback_among_target_complete": (
                self.excluded_insufficient_lookback_among_target_complete
            ),
            "used": self.used,
            "exclusion_overlap_note": self.exclusion_overlap_note,
        }


def _lookback_ok(dataset: Dataset, decision_dates: pd.Series) -> pd.Series:
    feats_by_date = dataset.features.set_index(dataset.features["date"].dt.date)
    reindexed = feats_by_date["has_full_lookback"].reindex(decision_dates)
    return reindexed.where(reindexed.notna(), False).astype(bool)


def eligibility_report(
    dataset: Dataset, targets: pd.DataFrame, label_column: str, agent_name: str = ""
) -> EligibilityReport:
    """Independent accounting of every exclusion `make_xy` applies, computed
    the same way `make_xy` applies them but never used to select rows
    itself -- so a bug in one cannot hide behind the other reporting the
    same (wrong) number.

    `excluded_insufficient_lookback_among_target_complete` is the count that
    this task's hardening actually adds: rows whose target was already
    complete (so the old, pre-A2 `make_xy` would have used them) but whose
    feature row lacks the full 20-day lookback. It is 0 for the current
    frozen TRAIN/VALIDATION data -- the two exclusion mechanisms happen to
    coincide there, see `exclusion_overlap_note` -- but is computed
    independently rather than assumed to always be 0.
    """
    target_ok = targets["complete"] & (targets[label_column] != "UNKNOWN")
    excluded_target_incomplete = int((~target_ok).sum())

    candidates = targets[target_ok]  # every row here already has a complete target
    lookback_ok = _lookback_ok(dataset, candidates["decision_date"])
    excluded_insufficient_lookback = int((~lookback_ok.to_numpy()).sum())

    overlap_note = (
        "0 rows with a complete target failed the lookback check: in the "
        "current frozen dataset every insufficient-lookback decision date "
        "also has an incomplete target (both stem from the same first-20-"
        "trading-day window), so this filter changes no counts today. It is "
        "enforced independently so that coincidence is never relied upon."
        if excluded_insufficient_lookback == 0 else
        f"{excluded_insufficient_lookback} row(s) with an otherwise-complete "
        "target failed the lookback check and are now excluded by make_xy "
        "that previously were not -- this is a real count change, not a "
        "coincidence; investigate before using this eligibility set."
    )

    return EligibilityReport(
        agent_name=agent_name,
        total_candidate_rows=int(len(targets)),
        excluded_target_incomplete=excluded_target_incomplete,
        excluded_insufficient_lookback_among_target_complete=excluded_insufficient_lookback,
        used=int(lookback_ok.to_numpy().sum()),
        exclusion_overlap_note=overlap_note,
    )


def make_xy(
    dataset: Dataset, targets: pd.DataFrame, label_column: str
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Feature matrix / label vector / observation timestamps for rows that
    are eligible on BOTH of two independent conditions:

    1. The label is complete and not `UNKNOWN` (target-side).
    2. The feature row has `has_full_lookback = True` (feature-side; Phase
       2.2 Task A2). Previously only (1) was enforced, which happened to be
       equivalent to enforcing both in the current frozen dataset -- see
       `eligibility_report` for why -- but that equivalence was never
       guaranteed by anything other than coincidence.

    Both exclusion counts are visible via `eligibility_report`, computed
    independently of this function so the two cannot drift silently."""
    target_ok = targets["complete"] & (targets[label_column] != "UNKNOWN")
    candidates = targets[target_ok]

    lookback_ok = _lookback_ok(dataset, candidates["decision_date"])
    usable = candidates[lookback_ok.to_numpy()]

    feats_by_date = dataset.features.set_index(dataset.features["date"].dt.date)
    X = feats_by_date.loc[usable["decision_date"], FEATURE_NAMES].reset_index(drop=True)
    y = usable[label_column].reset_index(drop=True)
    ts = pd.Series(pd.to_datetime(usable["decision_date"].to_numpy()))
    return X, y, ts
