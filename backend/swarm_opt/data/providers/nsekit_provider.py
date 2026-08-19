"""NseKit-backed DataProvider.

Uses only NseKit's own published client, at a request rate below its default,
against NSE's public archive endpoints. No scraping, no captcha or rate-limit
circumvention (docs/DATA_SPEC.md, Provider).

The trading calendar is taken from the index series rather than a holiday list,
so a date is a trading date exactly when the index actually printed a close.
"""
from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from ..contracts import DTYPES, NORMALIZED_COLUMNS
from ..normalize import normalize_bhavcopy
from .base import DataProvider

log = logging.getLogger(__name__)

_NSE_DATE = "%d-%m-%Y"

#: NSE index name used for the underlying series, keyed by F&O ticker symbol.
INDEX_NAME = {"NIFTY": "NIFTY 50", "BANKNIFTY": "NIFTY BANK"}


class NseKitProvider(DataProvider):
    provider_id = "nsekit"

    def __init__(self, max_rps: float = 1.0, retries: int = 2, retry_delay: float = 2.0):
        from NseKit import NseKit as _nk  # imported lazily: only this class needs it

        self._nse = _nk.Nse(max_rps=max_rps, retries=retries, retry_delay=retry_delay)
        # One frame per underlying covering the widest span fetched so far.
        # Keying the cache per (start, end) instead would defeat it entirely on
        # a bulk build: the legacy schema needs a spot lookup for every single
        # date, which would double the request count for three years of dates.
        self._history: dict[str, pd.DataFrame] = {}
        self._history_span: dict[str, tuple[date, date]] = {}

    # -- calendar ---------------------------------------------------------- #
    def trading_dates(self, start: date, end: date) -> list[date]:
        hist = self.underlying_history("NIFTY", start, end)
        return [d.date() for d in hist["date"]]

    # -- underlying -------------------------------------------------------- #
    def _fetch_history(self, underlying: str, start: date, end: date) -> pd.DataFrame:
        raw = self._nse.index_historical_data(
            INDEX_NAME.get(underlying, underlying),
            start.strftime(_NSE_DATE),
            end.strftime(_NSE_DATE),
        )
        if raw is None or len(raw) == 0:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close"])
        return pd.DataFrame({
            "date": pd.to_datetime(raw["Date"], format="%d-%b-%Y"),
            "open": raw["Open"].astype(float),
            "high": raw["High"].astype(float),
            "low": raw["Low"].astype(float),
            "close": raw["Close"].astype(float),
        }).sort_values("date").reset_index(drop=True)

    def underlying_history(self, underlying: str, start: date, end: date) -> pd.DataFrame:
        span = self._history_span.get(underlying)
        if span is None or start < span[0] or end > span[1]:
            lo = min(start, span[0]) if span else start
            hi = max(end, span[1]) if span else end
            fetched = self._fetch_history(underlying, lo, hi)
            if not fetched.empty:
                self._history[underlying] = fetched
                self._history_span[underlying] = (lo, hi)

        cached = self._history.get(underlying)
        if cached is None or cached.empty:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close"])

        mask = (cached["date"] >= pd.Timestamp(start)) & (cached["date"] <= pd.Timestamp(end))
        return cached.loc[mask].reset_index(drop=True)

    # -- chains ------------------------------------------------------------ #
    def option_chain(self, underlying: str, on: date) -> pd.DataFrame:
        raw = self._nse.fno_eod_bhav_copy(on.strftime(_NSE_DATE))
        if raw is None or raw.empty:
            # NSE trading holiday, or the archive has no file for that date.
            log.info("no bhavcopy for %s (holiday or missing archive)", on)
            return pd.DataFrame(columns=NORMALIZED_COLUMNS).astype(DTYPES)

        spot = None
        if "INSTRUMENT" in raw.columns:  # legacy schema carries no underlying price
            hist = self.underlying_history(underlying, on, on)
            if hist.empty:
                raise ValueError(f"no index close for {on}; cannot normalize legacy chain")
            spot = float(hist["close"].iloc[0])

        return normalize_bhavcopy(
            raw, underlying=underlying, spot=spot, source=self.provider_id
        )

    def expiries(self, underlying: str, on: date) -> list[date]:
        chain = self.option_chain(underlying, on)
        return sorted({d.date() for d in chain["expiry"].unique()})
