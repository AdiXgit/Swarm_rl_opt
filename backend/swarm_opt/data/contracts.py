"""Normalized EOD option-chain data contract for SWARM-OPT.

The canonical columns follow docs/DATA_SPEC.md. Provider-specific columns are
mapped into this contract by `swarm_opt.data.normalize`; anything the provider
does not supply is emitted as NaN and recorded in a missingness report. Nothing
is fabricated.
"""
from __future__ import annotations

from dataclasses import dataclass

# --- docs/DATA_SPEC.md "Normalized Option Record" -----------------------------
SPEC_COLUMNS: list[str] = [
    "date",
    "underlying",
    "spot",
    "expiry",
    "strike",
    "option_type",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "open_interest",
    "change_in_open_interest",
    "implied_volatility",
]

# --- Extensions required to price/mark trades without fabricating data --------
# Each one is derived from a field the provider actually returns, or left NaN.
EXTENSION_COLUMNS: list[str] = [
    "settle_price",        # NSE daily settlement price (marking + untraded legs)
    "forward",             # expiry-matched futures / parity-implied forward
    "forward_source",      # futures | put_call_parity | interpolated | missing
    "lot_size",            # contract multiplier on that date
    "turnover_notional",   # INR notional turnover
    "is_traded",           # volume > 0 on that date
    "days_to_expiry",      # calendar days, decision date -> expiry
    "source",              # provider id
    "schema_version",      # raw provider schema the row came from
]

NORMALIZED_COLUMNS: list[str] = SPEC_COLUMNS + EXTENSION_COLUMNS

DTYPES: dict[str, str] = {
    "date": "datetime64[ns]",
    "underlying": "string",
    "spot": "float64",
    "expiry": "datetime64[ns]",
    "strike": "float64",
    "option_type": "string",   # "CE" | "PE"
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "int64",         # contracts (lots), NOT units
    "open_interest": "int64",  # contracts (lots)
    "change_in_open_interest": "int64",
    "implied_volatility": "float64",
    "settle_price": "float64",
    "forward": "float64",
    "forward_source": "string",
    "lot_size": "int64",
    "turnover_notional": "float64",
    "is_traded": "bool",
    "days_to_expiry": "int64",
    "source": "string",
    "schema_version": "string",
}

PRIMARY_KEY: list[str] = ["date", "underlying", "expiry", "strike", "option_type"]


@dataclass(frozen=True)
class MissingnessReport:
    """What the provider did not supply, per DATA_SPEC 'log, do not fabricate'."""

    rows: int
    fully_missing: list[str]
    null_fraction: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "rows": self.rows,
            "fully_missing": self.fully_missing,
            "null_fraction": self.null_fraction,
        }
