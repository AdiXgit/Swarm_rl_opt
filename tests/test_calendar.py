"""Expiry calendar and weekly episode construction (approved decision 5).

The recurring hazard these guard against is a hard-coded expiry weekday. NSE
has moved NIFTY's weekly expiry more than once, and holidays shift individual
expiries off the usual day, so every boundary must come from observed data.
"""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from swarm_opt.data.calendar import ExpiryCalendar
from swarm_opt.data.providers.local_parquet_provider import LocalParquetProvider

SPAN = (date(2023, 1, 1), date(2027, 1, 1))


@pytest.fixture(scope="module")
def calendar(fixture_root) -> ExpiryCalendar:
    provider = LocalParquetProvider(fixture_root)
    return ExpiryCalendar.from_provider(
        provider, "NIFTY", provider.trading_dates(*SPAN)
    )


def test_built_from_observed_dates_only(calendar):
    assert calendar.trading_dates == sorted(calendar.trading_dates)
    assert len(calendar.trading_dates) == 10


def test_expiries_are_point_in_time(calendar):
    for d in calendar.trading_dates:
        assert all(e >= d for e in calendar.expiries_on(d))


def test_next_expiry_respects_the_minimum_horizon(calendar):
    on = date(2026, 8, 18)  # this date is itself an expiry
    assert calendar.next_expiry(on, min_days=0) == on
    # min_days=1 must skip the same-day expiry rather than hand back a dead contract
    assert calendar.next_expiry(on, min_days=1) > on


def test_nth_expiry_walks_the_ladder(calendar):
    on = date(2026, 8, 12)
    ladder = [calendar.nth_expiry(on, n, min_days=1) for n in range(3)]
    assert ladder == sorted(ladder)
    assert len(set(ladder)) == 3
    assert all(e is not None for e in ladder)


def test_nth_expiry_past_the_ladder_is_none(calendar):
    assert calendar.nth_expiry(date(2026, 8, 12), 999) is None


def test_no_weekday_is_assumed(calendar):
    """The fixture straddles two different NSE weekly-expiry regimes.

    2023-07-20 was a Thursday expiry; 2026-08-18 is a Tuesday one. Any code
    that hard-codes a weekday would break across this boundary.
    """
    weekdays = calendar.observed_expiry_weekdays()
    assert set(weekdays) == {"Thursday", "Tuesday"}


# --------------------------------------------------------------------------- #
# weekly episodes
# --------------------------------------------------------------------------- #
def test_episodes_are_anchored_to_real_expiries(calendar):
    episodes = calendar.weekly_episodes()
    assert episodes
    for ep in episodes:
        assert ep.expiry in calendar.expiries_on(ep.decision_date)
        assert ep.decision_date < ep.exit_date <= ep.expiry


def test_a_gap_in_the_data_cannot_fabricate_an_episode(calendar):
    """The fixture jumps from Jul-2023 to Aug-2026.

    Without the listing and span guards the partition bridges that gap and
    emits one 3-year "week" whose reward spans the hole.
    """
    for ep in calendar.weekly_episodes():
        assert ep.days_to_expiry <= 10
        assert (ep.exit_date - ep.decision_date).days <= 10


def test_both_eras_yield_an_episode(calendar):
    years = {ep.decision_date.year for ep in calendar.weekly_episodes()}
    assert years == {2023, 2026}


def test_episode_marks_are_strictly_after_the_decision(calendar):
    """The decision date's own EOD is the last thing the agent sees; every
    mark must come after it, or the reward would leak into the state."""
    for ep in calendar.weekly_episodes():
        assert all(m > ep.decision_date for m in ep.mark_dates)
        assert ep.mark_dates == tuple(sorted(ep.mark_dates))
        assert ep.exit_date == ep.mark_dates[-1]


def test_episodes_do_not_overlap(calendar):
    episodes = calendar.weekly_episodes()
    for a, b in zip(episodes, episodes[1:]):
        assert a.expiry < b.expiry
        assert a.exit_date <= b.decision_date


def test_days_to_expiry_is_consistent(calendar):
    for ep in calendar.weekly_episodes():
        assert ep.days_to_expiry == (ep.expiry - ep.decision_date).days
        assert ep.days_to_expiry >= 1


def test_expiry_landing_inside_the_episode_is_flagged(calendar):
    # Every weekly episode here runs to its own expiry, so settlement applies.
    assert all(ep.expires_within_episode for ep in calendar.weekly_episodes())


# --------------------------------------------------------------------------- #
# synthetic regimes: the weekday actually moving
# --------------------------------------------------------------------------- #
def _weekly(start: date, weeks: int, weekday: int) -> list[date]:
    days = [start + timedelta(days=i) for i in range((weeks + 2) * 7)]
    return [d for d in days if d.weekday() < 5]


def test_handles_a_weekday_change_mid_sample():
    """Thursday expiries switching to Tuesday must not confuse the partition."""
    trading = _weekly(date(2025, 1, 1), 8, 0)
    thursdays = [d for d in trading if d.weekday() == 3 and d < date(2025, 2, 1)]
    tuesdays = [d for d in trading if d.weekday() == 1 and d >= date(2025, 2, 1)]
    expiries = sorted(thursdays + tuesdays)

    listed = {
        d: [e for e in expiries if e >= d][:3] for d in trading
    }
    cal = ExpiryCalendar(listed)
    episodes = cal.weekly_episodes()

    assert len(episodes) >= 4
    assert {e.expiry for e in episodes} <= set(expiries)
    for a, b in zip(episodes, episodes[1:]):
        assert a.exit_date <= b.decision_date


def test_holiday_shifted_expiry_is_respected():
    """When the usual expiry day is a holiday the expiry moves; the episode
    must follow the listed expiry, not the weekday."""
    trading = [d for d in _weekly(date(2025, 3, 3), 4, 0)]
    normal = [d for d in trading if d.weekday() == 3]
    shifted = normal[1] - timedelta(days=1)  # that Thursday was a holiday
    trading = [d for d in trading if d != normal[1]]
    expiries = sorted([normal[0], shifted] + normal[2:])

    cal = ExpiryCalendar({d: [e for e in expiries if e >= d][:3] for d in trading})
    episodes = cal.weekly_episodes()

    assert shifted in {e.expiry for e in episodes}
    assert normal[1] not in {e.expiry for e in episodes}
