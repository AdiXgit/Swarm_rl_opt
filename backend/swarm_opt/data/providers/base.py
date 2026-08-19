"""DataProvider contract.

ML/RL code must depend only on this interface (see docs/ARCHITECTURE.md).
Implementations return frames conforming to `swarm_opt.data.contracts`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

from ..contracts import MissingnessReport


class DataProvider(ABC):
    """Read-only source of normalized EOD option chains and underlying marks.

    All methods are point-in-time: a call for `on` must never consult data
    stamped later than `on` (docs/DATA_SPEC.md, Leakage Rules).
    """

    #: stable provider id written into the `source` column
    provider_id: str

    @abstractmethod
    def trading_dates(self, start: date, end: date) -> list[date]:
        """Actual NSE trading dates in ``[start, end]``, ascending."""

    @abstractmethod
    def option_chain(self, underlying: str, on: date) -> pd.DataFrame:
        """Full normalized EOD chain for `underlying` as of `on`.

        Contains every expiry and strike the provider reports for that date.
        Columns: `contracts.NORMALIZED_COLUMNS`.
        """

    @abstractmethod
    def expiries(self, underlying: str, on: date) -> list[date]:
        """Expiries listed on `on`, ascending. Never includes future listings."""

    @abstractmethod
    def underlying_history(
        self, underlying: str, start: date, end: date
    ) -> pd.DataFrame:
        """EOD OHLC for the underlying index.

        Columns: ``date, open, high, low, close``.
        """

    def missingness(self, frame: pd.DataFrame) -> MissingnessReport:
        """Report unsupplied fields rather than filling them."""
        null_frac = {c: float(frame[c].isna().mean()) for c in frame.columns}
        return MissingnessReport(
            rows=int(len(frame)),
            fully_missing=sorted(c for c, f in null_frac.items() if f == 1.0),
            null_fraction={c: round(f, 6) for c, f in null_frac.items() if f > 0},
        )
