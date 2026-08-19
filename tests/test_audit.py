"""Tests for the audit logic itself.

The audits are the evidence that the dataset is sound, so they need their own
evidence: an audit that cannot fail is not an audit. Each leakage check here is
fed a deliberately corrupted calendar or frame and must report the violation.
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "audit_dataset", ROOT / "scripts" / "audit_dataset.py"
)
audit_dataset = importlib.util.module_from_spec(_spec)
sys.modules["audit_dataset"] = audit_dataset
_spec.loader.exec_module(audit_dataset)

from swarm_opt.data.calendar import ExpiryCalendar  # noqa: E402
from swarm_opt.data.splits import split_for  # noqa: E402


def _clean_calendar() -> ExpiryCalendar:
    """Four consecutive Thursday-expiry weeks inside TRAIN."""
    start = date(2022, 5, 2)  # a Monday
    trading = [
        start + timedelta(days=i)
        for i in range(28)
        if (start + timedelta(days=i)).weekday() < 5
    ]
    expiries = [d for d in trading if d.weekday() == 3]
    return ExpiryCalendar(
        {d: [e for e in expiries if e >= d][:3] for d in trading}
    )


def _chain(dates, expiry_offset=3) -> pd.DataFrame:
    rows = []
    for d in dates:
        expiry = d + timedelta(days=expiry_offset)
        for strike in (17000.0, 17050.0):
            for side in ("CE", "PE"):
                rows.append({
                    "date": pd.Timestamp(d),
                    "underlying": "NIFTY",
                    "expiry": pd.Timestamp(expiry),
                    "strike": strike,
                    "option_type": side,
                    "days_to_expiry": expiry_offset,
                    "open_interest": 5_000_000,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def calendar() -> ExpiryCalendar:
    return _clean_calendar()


# --------------------------------------------------------------------------- #
# the clean case
# --------------------------------------------------------------------------- #
INFORMATIONAL = {
    "clean",
    "episodes_purged_at_split_boundaries",
    "purged_episode_decision_dates",
    "material_reappearance_detail",
    "immaterial_expiry_flicker",
    "immaterial_flicker_detail",
}


def test_clean_dataset_passes_every_leakage_check(calendar):
    chains = _chain(calendar.trading_dates)
    result = audit_dataset.audit_leakage(chains, calendar)
    assert result["clean"] is True
    assert all(v == 0 for k, v in result.items() if k not in INFORMATIONAL)


# --------------------------------------------------------------------------- #
# each check must actually catch its violation
# --------------------------------------------------------------------------- #
def test_catches_an_expired_contract_in_the_chain(calendar):
    chains = _chain(calendar.trading_dates)
    chains.loc[0, "expiry"] = chains.loc[0, "date"] - pd.Timedelta(days=1)
    result = audit_dataset.audit_leakage(chains, calendar)
    assert result["expired_contracts_in_chain"] == 1
    assert result["clean"] is False


def test_catches_negative_days_to_expiry(calendar):
    chains = _chain(calendar.trading_dates)
    chains.loc[0, "days_to_expiry"] = -2
    result = audit_dataset.audit_leakage(chains, calendar)
    assert result["negative_days_to_expiry"] == 1
    assert result["clean"] is False


def test_catches_duplicate_primary_keys(calendar):
    chains = _chain(calendar.trading_dates)
    chains = pd.concat([chains, chains.iloc[[0]]], ignore_index=True)
    result = audit_dataset.audit_leakage(chains, calendar)
    assert result["duplicate_primary_keys"] == 1
    assert result["clean"] is False


def test_a_delisted_expiry_alone_is_not_leakage():
    """An expiry legitimately stops being listed when a holiday moves it, NSE
    migrates the ladder, or its open interest reaches zero under the approved
    filter. None of that is lookahead, so it must not fail the audit."""
    clean = _clean_calendar()
    listed = {d: clean.expiries_on(d) for d in clean.trading_dates}
    dates = sorted(listed)
    # delist it from this date onward, the way a real delisting behaves
    victim = listed[dates[5]][-1]
    for d in dates[5:]:
        listed[d] = [e for e in listed[d] if e != victim]

    result = audit_dataset.audit_leakage(_chain(dates), ExpiryCalendar(listed))
    assert result["material_expiries_reappearing_after_delisting"] == 0
    assert result["clean"] is True


def _chain_from_listing(listed: dict, open_interest: dict | None = None) -> pd.DataFrame:
    """Chain frame whose (date, expiry) pairs match the calendar exactly.

    The leakage audit reads open interest per (date, expiry), so a synthetic
    frame that does not carry the calendar's expiries silently reports zero.
    """
    rows = []
    for day, expiries in listed.items():
        for expiry in expiries:
            for strike in (17000.0, 17050.0):
                for side in ("CE", "PE"):
                    rows.append({
                        "date": pd.Timestamp(day),
                        "underlying": "NIFTY",
                        "expiry": pd.Timestamp(expiry),
                        "strike": strike,
                        "option_type": side,
                        "days_to_expiry": (expiry - day).days,
                        "open_interest": (open_interest or {}).get(expiry, 5_000_000),
                    })
    return pd.DataFrame(rows)


def _reappearing(open_interest: int):
    """Calendar where one expiry vanishes for a day and comes back, carrying
    the given open interest on the last date it was listed."""
    clean = _clean_calendar()
    listed = {d: list(clean.expiries_on(d)) for d in clean.trading_dates}
    dates = sorted(listed)

    target = listed[dates[2]][1]
    listed[dates[3]] = [e for e in listed[dates[3]] if e != target]
    assert target in listed[dates[4]], "fixture must let it come back"

    chains = _chain_from_listing(listed, {target: open_interest})
    return chains, ExpiryCalendar(listed)


def test_catches_a_material_expiry_reappearing_after_delisting():
    """Disappearing and coming back in a contract people actually hold means
    the chain was assembled out of order or a file was corrupt."""
    chains, cal = _reappearing(open_interest=5_000_000)
    result = audit_dataset.audit_leakage(chains, cal)
    assert result["material_expiries_reappearing_after_delisting"] > 0
    assert result["clean"] is False


def test_flicker_in_a_dead_expiry_is_not_a_failure():
    """The approved open-interest filter removes an expiry whose strikes all
    reach zero OI, and a near-dead contract can cross back over. The rows are
    present in the raw bhavcopy throughout, so this is the filter working."""
    chains, cal = _reappearing(open_interest=1)
    result = audit_dataset.audit_leakage(chains, cal)
    assert result["material_expiries_reappearing_after_delisting"] == 0
    assert result["immaterial_expiry_flicker"] > 0
    assert result["clean"] is True


def _straddling_calendar() -> ExpiryCalendar:
    """An episode decided in TRAIN that resolves after the TRAIN/VALIDATION wall."""
    boundary = date(2024, 12, 31)
    trading = [boundary + timedelta(days=i) for i in range(-3, 6)]
    trading = [d for d in trading if d.weekday() < 5]
    expiry = next(d for d in trading if d > boundary)
    return ExpiryCalendar({d: [expiry] for d in trading if d <= expiry})


def test_an_episode_straddling_a_split_boundary_is_purged():
    """Its reward would be computed from data on the far side of the wall, so
    it is dropped rather than assigned to either split."""
    cal = _straddling_calendar()
    result = audit_dataset.audit_leakage(_chain(sorted(cal.trading_dates)), cal)

    assert result["episodes_purged_at_split_boundaries"] > 0
    assert result["episodes_straddling_a_split_boundary_after_purge"] == 0
    # purging is the fix, so the audit stays clean
    assert result["clean"] is True


def test_purging_removes_exactly_the_straddling_episodes():
    from swarm_opt.data.splits import episode_split, purge_straddling

    cal = _straddling_calendar()
    kept, purged = purge_straddling(cal.weekly_episodes())
    assert purged
    for ep in purged:
        assert split_for(ep.decision_date) != split_for(ep.exit_date)
        assert episode_split(ep) is None
    for ep in kept:
        assert episode_split(ep) == split_for(ep.decision_date)


def test_catches_dates_outside_every_split():
    outside = date(2019, 6, 3)
    trading = [outside + timedelta(days=i) for i in range(5)]
    expiry = trading[-1]
    cal = ExpiryCalendar({d: [expiry] for d in trading})
    result = audit_dataset.audit_leakage(_chain(trading), cal)
    assert result["dates_outside_every_split"] == len(trading)
    assert result["clean"] is False


# --------------------------------------------------------------------------- #
# run-length helper
# --------------------------------------------------------------------------- #
def test_runs_collapse_consecutive_values():
    pairs = [
        (date(2021, 1, 1), 50), (date(2021, 1, 2), 50),
        (date(2021, 1, 3), 25), (date(2021, 1, 4), 50),
    ]
    runs = audit_dataset._runs(pairs)
    assert [r["value"] for r in runs] == [50, 25, 50]
    assert runs[0]["dates"] == 2
    assert runs[0]["start"] == "2021-01-01" and runs[0]["end"] == "2021-01-02"


def test_runs_do_not_merge_a_value_that_recurs_later():
    """NIFTY's lot went 50 -> 25 -> 75; a naive grouping would merge the two
    75 periods and hide a regime change."""
    pairs = [
        (date(2021, 1, 1), 75), (date(2022, 1, 1), 50), (date(2025, 1, 1), 75),
    ]
    runs = audit_dataset._runs(pairs)
    assert len(runs) == 3
    assert [r["value"] for r in runs] == [75, 50, 75]
