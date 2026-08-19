"""Expiry and weekly-episode calendar, derived from observed data.

NIFTY's weekly expiry weekday is not stable: NSE has moved it several times
(Thursday for most of the sample, Wednesday, and Tuesday as of 2026). Nothing
here assumes a weekday. Expiries come from the expiry dates actually listed in
the chain on each date, and a week is delimited by consecutive expiries rather
than by a calendar rule.

All lookups are point-in-time: `expiries_on` only ever sees the chain for that
date, so a decision can never learn about a listing that did not exist yet.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class WeeklyEpisode:
    """One PPO decision and the EOD marks that resolve it.

    `decision_date` is the last observation before the position is opened;
    `mark_dates` are the EOD observations the simulator marks through, ending
    at `exit_date`. `expiry` is the expiry the episode is anchored to.
    """

    decision_date: date
    expiry: date
    exit_date: date
    mark_dates: tuple[date, ...]

    @property
    def days_to_expiry(self) -> int:
        return (self.expiry - self.decision_date).days

    @property
    def expires_within_episode(self) -> bool:
        return self.expiry <= self.exit_date


class ExpiryCalendar:
    """Observed expiry structure for one underlying over a date range.

    Built once from the provider and then queried offline. `listed` maps each
    trading date to the expiries visible on that date.
    """

    def __init__(self, listed: dict[date, list[date]]):
        self._listed = {d: sorted(e) for d, e in listed.items()}
        self._dates = sorted(self._listed)

    # -- construction ------------------------------------------------------ #
    @classmethod
    def from_provider(cls, provider, underlying: str, dates) -> "ExpiryCalendar":
        return cls({d: provider.expiries(underlying, d) for d in dates})

    @classmethod
    def from_frame(cls, chains: pd.DataFrame) -> "ExpiryCalendar":
        """Build from a concatenated normalized frame (`date`, `expiry`)."""
        grouped = chains.groupby(chains["date"].dt.date)["expiry"]
        return cls({d: sorted({x.date() for x in g.unique()}) for d, g in grouped})

    # -- queries ----------------------------------------------------------- #
    @property
    def trading_dates(self) -> list[date]:
        return list(self._dates)

    def expiries_on(self, on: date) -> list[date]:
        return list(self._listed.get(on, []))

    def next_expiry(self, on: date, min_days: int = 0) -> date | None:
        """Nearest expiry listed on `on` that is at least `min_days` away.

        `min_days=1` skips an expiry landing on the decision date itself, which
        would otherwise hand trade construction a contract with no life left.
        """
        for e in self.expiries_on(on):
            if (e - on).days >= min_days:
                return e
        return None

    def nth_expiry(self, on: date, n: int, min_days: int = 0) -> date | None:
        """`n`-th eligible expiry (0-based) for the expiry-bucket action."""
        eligible = [e for e in self.expiries_on(on) if (e - on).days >= min_days]
        return eligible[n] if n < len(eligible) else None

    def observed_expiry_weekdays(self) -> dict[str, int]:
        """Diagnostic: how the weekly expiry weekday actually moved.

        Counts the weekday of each expiry that fell on an observed trading date,
        so a shift shows up in the dataset audit instead of silently breaking a
        hard-coded assumption.
        """
        seen = {e for exp in self._listed.values() for e in exp}
        landed = sorted(seen & set(self._dates))
        out: dict[str, int] = {}
        for e in landed:
            key = e.strftime("%A")
            out[key] = out.get(key, 0) + 1
        return out

    # -- episodes ---------------------------------------------------------- #
    def weekly_episodes(
        self,
        min_days_to_expiry: int = 1,
        max_days_to_expiry: int = 10,
    ) -> list[WeeklyEpisode]:
        """Partition the trading dates into consecutive weekly episodes.

        A decision is taken on the first trading date after one expiry has
        passed on which the next expiry is actually listed; the episode runs to
        the trading date on or before that expiry. This tracks whatever expiry
        cycle NSE ran, including weekday changes and holiday-shifted expiries.

        Two guards keep a gap in the data from fabricating an episode. The
        expiry must be listed on the decision date, so an episode can never be
        anchored to a contract that was not quoted yet; and the span is bounded
        by `max_days_to_expiry`, so a stretch of missing dates cannot silently
        produce a single enormous "week" whose reward spans months.
        """
        expiry_dates = sorted(
            {e for exp in self._listed.values() for e in exp} & set(self._dates)
        )
        episodes: list[WeeklyEpisode] = []
        previous: date | None = None

        for expiry in expiry_dates:
            window = [
                d for d in self._dates
                if (previous is None or d > previous) and d < expiry
            ]
            previous = expiry

            start = next(
                (
                    d for d in window
                    if expiry in self._listed.get(d, [])
                    and min_days_to_expiry <= (expiry - d).days <= max_days_to_expiry
                ),
                None,
            )
            if start is None:
                continue

            marks = tuple(d for d in self._dates if start < d <= expiry)
            if not marks:
                continue
            episodes.append(
                WeeklyEpisode(
                    decision_date=start,
                    expiry=expiry,
                    exit_date=marks[-1],
                    mark_dates=marks,
                )
            )
        return episodes
