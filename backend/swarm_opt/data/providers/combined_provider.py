"""Read-only union of two or more non-overlapping `LocalParquetProvider`s.

Phase 2.5 need: the historical extension (2019-02-18..2021-03-31) lives in
its own directory, separate from the frozen 2021-04-01..2026-08-18 dataset
(see docs/HISTORICAL_PROBE_AND_EXPIRY_AUDIT.md). Rather than merging the
extension into the frozen dataset's own files -- forbidden by this project's
"never overwrite the original" rule -- this presents both as one logical
`DataProvider` so the existing, unmodified calendar/feature/target pipeline
can run across the combined range without copying any file.

Additive only: does not change `LocalParquetProvider` or any other existing
provider.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from ..contracts import DTYPES, NORMALIZED_COLUMNS
from .base import DataProvider
from .local_parquet_provider import LocalParquetProvider


class CombinedParquetProvider(DataProvider):
    """Routes each date to whichever member provider actually has it.

    Member providers are expected to cover disjoint date ranges (verified by
    the caller, e.g. the extension ends 2021-03-31 and the frozen dataset
    starts 2021-04-01 -- no overlap). If two members both had data for the
    same date, the first one listed wins; this is a routing union, not a
    merge, and never fabricates or reconciles conflicting rows.
    """

    provider_id = "combined_local_parquet"

    def __init__(self, providers: list[LocalParquetProvider]):
        if not providers:
            raise ValueError("CombinedParquetProvider needs at least one member provider")
        self._providers = providers

    def trading_dates(self, start: date, end: date) -> list[date]:
        dates: set[date] = set()
        for p in self._providers:
            dates.update(p.trading_dates(start, end))
        return sorted(dates)

    def underlying_history(self, underlying: str, start: date, end: date) -> pd.DataFrame:
        frames = [p.underlying_history(underlying, start, end) for p in self._providers]
        non_empty = [f for f in frames if not f.empty]
        if not non_empty:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close"])
        combined = pd.concat(non_empty, ignore_index=True)
        duplicated_dates = combined[combined.duplicated("date", keep=False)]
        if len(duplicated_dates):
            raise ValueError(
                f"member providers disagree or overlap on {duplicated_dates['date'].nunique()} "
                "date(s) of underlying history; CombinedParquetProvider requires disjoint ranges"
            )
        return combined.sort_values("date").reset_index(drop=True)

    def option_chain(self, underlying: str, on: date) -> pd.DataFrame:
        for p in self._providers:
            chain = p.option_chain(underlying, on)
            if not chain.empty:
                return chain
        return pd.DataFrame(columns=NORMALIZED_COLUMNS).astype(DTYPES)

    def expiries(self, underlying: str, on: date) -> list[date]:
        chain = self.option_chain(underlying, on)
        return sorted({d.date() for d in chain["expiry"].unique()})
