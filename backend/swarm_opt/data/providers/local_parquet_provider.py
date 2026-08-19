"""Offline DataProvider over the normalized Parquet fixture.

This is the provider every test, notebook, and training run should use: it is
deterministic, needs no network, and cannot be affected by NSE archive changes.
Layout produced by `scripts/build_fixture.py`:

    <root>/chains/underlying=NIFTY/date=YYYY-MM-DD/part.parquet
    <root>/underlying/NIFTY.parquet
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from ..contracts import DTYPES, NORMALIZED_COLUMNS
from .base import DataProvider


class LocalParquetProvider(DataProvider):
    provider_id = "local_parquet"

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"fixture root not found: {self.root}")

    # -- paths ------------------------------------------------------------- #
    def _chain_path(self, underlying: str, on: date) -> Path:
        return (
            self.root
            / "chains"
            / f"underlying={underlying}"
            / f"date={on.isoformat()}"
            / "part.parquet"
        )

    # -- calendar ---------------------------------------------------------- #
    def trading_dates(self, start: date, end: date) -> list[date]:
        base = self.root / "chains" / "underlying=NIFTY"
        if not base.exists():
            return []
        found = sorted(
            date.fromisoformat(p.name.split("=", 1)[1])
            for p in base.iterdir()
            if p.is_dir() and p.name.startswith("date=")
        )
        return [d for d in found if start <= d <= end]

    # -- underlying -------------------------------------------------------- #
    def underlying_history(self, underlying: str, start: date, end: date) -> pd.DataFrame:
        path = self.root / "underlying" / f"{underlying}.parquet"
        if not path.exists():
            return pd.DataFrame(columns=["date", "open", "high", "low", "close"])
        df = pd.read_parquet(path)
        mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
        return df.loc[mask].sort_values("date").reset_index(drop=True)

    # -- chains ------------------------------------------------------------ #
    def option_chain(self, underlying: str, on: date) -> pd.DataFrame:
        path = self._chain_path(underlying, on)
        if not path.exists():
            return pd.DataFrame(columns=NORMALIZED_COLUMNS).astype(DTYPES)
        return pd.read_parquet(path).astype(DTYPES)

    def expiries(self, underlying: str, on: date) -> list[date]:
        chain = self.option_chain(underlying, on)
        return sorted({d.date() for d in chain["expiry"].unique()})
