"""Per-date ATM IV / skew cache: the one chain-derived input the point-in-time
feature pipeline needs beyond the underlying OHLC series.

Extracted from `scripts/build_market_features.py` (Phase 2.2, Task A1) so the
point-in-time property can be unit-tested directly against a `DataProvider`
rather than only by code review of the CLI script. Behavior is unchanged:
the script now imports this module instead of defining its own copy.

Point-in-time by construction: `build_atm_iv_row` calls
`provider.option_chain(underlying, on)` exactly once, for exactly `on`. Every
`DataProvider` implementation in this project (see `data/providers/base.py`)
guarantees that call returns only that date's observations -- there is no
parameter and no code path here that could pull in a later date's chain, a
later date's spot, or a later date's expiry listing.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from ..data.providers.base import DataProvider
from .enrich import atm_summary, enrich_chain

#: Moneyness band around spot kept before solving IV. ATM/25-delta skew only
#: ever needs strikes near the money; everything outside this band is
#: discarded before the (expensive) IV solve, not after.
MONEYNESS_BAND = 0.10
MAX_DAYS_TO_EXPIRY = 10

ATM_CACHE_COLUMNS = ["date", "expiry", "days_to_expiry", "atm_iv", "skew_25d", "priced_strikes"]


def nearest_expiry_band(
    chain: pd.DataFrame,
    max_days_to_expiry: int = MAX_DAYS_TO_EXPIRY,
    moneyness_band: float = MONEYNESS_BAND,
) -> pd.DataFrame:
    """Nearest eligible expiry's strikes within `moneyness_band` of spot.

    Operates only on rows already present in `chain` -- it filters and
    selects, it does not fetch or join anything. `chain` carries its own
    `date` column throughout, so a caller that accidentally passes a
    multi-date frame would get a nearest expiry computed within whatever
    `chain` contains, not across dates; `build_atm_iv_row` below only ever
    passes a single date's chain.
    """
    near = chain[chain["days_to_expiry"] <= max_days_to_expiry]
    if near.empty:
        return near
    nearest_expiry = near["expiry"].min()
    leg = near[near["expiry"] == nearest_expiry]
    spot = float(leg["spot"].iloc[0])
    lo, hi = spot * (1 - moneyness_band), spot * (1 + moneyness_band)
    return leg[(leg["strike"] >= lo) & (leg["strike"] <= hi)]


def build_atm_iv_row(chain: pd.DataFrame, observation_date: date) -> dict | None:
    """One cache row for `observation_date`, or `None` if nothing is
    priceable (empty chain, no expiry within `MAX_DAYS_TO_EXPIRY`, or no
    strike in the band that solves to a finite IV). `None` is an explicit
    absence -- the caller must not fabricate a row for it.
    """
    if chain.empty:
        return None
    band = nearest_expiry_band(chain)
    if band.empty:
        return None
    enriched = enrich_chain(band)
    summary = atm_summary(enriched)
    if summary.empty:
        return None
    row = summary.iloc[0]
    return {
        "date": pd.Timestamp(observation_date),
        "expiry": row["expiry"],
        "days_to_expiry": int(row["days_to_expiry"]),
        "atm_iv": float(row["atm_iv"]),
        "skew_25d": float(row["skew_25d"]),
        "priced_strikes": int(row["priced_strikes"]),
    }


def build_atm_iv_history(
    provider: DataProvider, underlying: str, dates: list[date]
) -> pd.DataFrame:
    """One row per date in `dates` that has a priceable near-term chain.

    Each date's row is built from exactly that date's own
    `provider.option_chain(underlying, on)` call -- dates are independent of
    each other and processed in the order given, not re-ordered or batched
    in a way that could let one date's computation see another's data.
    """
    rows = []
    for on in dates:
        chain = provider.option_chain(underlying, on)
        row = build_atm_iv_row(chain, on)
        if row is not None:
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=ATM_CACHE_COLUMNS)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
