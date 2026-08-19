"""Attach IV and Greeks to a normalized chain.

This is the layer that fills the `implied_volatility` column the data contract
leaves NaN. It is deliberately separate from `data.normalize`: normalization
reports what NSE published, enrichment derives what NSE did not, and keeping
them apart means the raw layer stays auditable against the source files.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .pricing import greeks, implied_vol, year_fraction

GREEK_COLUMNS = ["delta", "gamma", "theta", "vega"]
ENRICHED_COLUMNS = GREEK_COLUMNS + ["moneyness", "log_moneyness", "iv_source"]


def enrich_chain(chain: pd.DataFrame, price_column: str = "close") -> pd.DataFrame:
    """Solve IV against the forward and attach Greeks.

    Rows without a forward, or whose quote sits outside the no-arbitrage band,
    keep NaN IV and NaN Greeks and are labelled in `iv_source`. They are not
    dropped: trade construction still needs to see the contract exists, and the
    simulator can still mark it on its published price.
    """
    out = chain.copy()
    if out.empty:
        for col in ENRICHED_COLUMNS:
            out[col] = pd.Series(dtype="float64" if col != "iv_source" else "string")
        return out

    t = year_fraction(out["days_to_expiry"].to_numpy())
    forward = out["forward"].to_numpy(dtype=float)
    strike = out["strike"].to_numpy(dtype=float)
    opt_type = out["option_type"].astype(str).to_numpy()
    price = out[price_column].to_numpy(dtype=float)

    iv = implied_vol(price, forward, strike, t, opt_type)
    out["implied_volatility"] = iv

    g = greeks(forward, strike, iv, t, opt_type)
    for name in GREEK_COLUMNS:
        out[name] = g[name]

    with np.errstate(divide="ignore", invalid="ignore"):
        out["moneyness"] = strike / forward
        out["log_moneyness"] = np.log(strike / forward)

    out["iv_source"] = np.where(
        np.isfinite(iv),
        f"black76_{price_column}",
        np.where(~np.isfinite(forward), "no_forward", "outside_arb_bounds"),
    )
    out["iv_source"] = out["iv_source"].astype("string")
    return out


def atm_summary(chain: pd.DataFrame) -> pd.DataFrame:
    """Per-expiry ATM IV and 25-delta skew, for the volatility specialist.

    ATM IV is the average of the call and put IV at the strike closest to the
    forward, which is far more stable than either leg alone. Skew is the
    25-delta put IV less the 25-delta call IV.
    """
    rows = []
    for expiry, grp in chain.groupby("expiry"):
        priced = grp[grp["implied_volatility"].notna()]
        if priced.empty:
            continue
        forward = float(priced["forward"].iloc[0])

        atm_strike = priced.loc[
            (priced["strike"] - forward).abs().idxmin(), "strike"
        ]
        at_strike = priced[priced["strike"] == atm_strike]
        atm_iv = float(at_strike["implied_volatility"].mean())

        def nearest_delta(side: str, target: float) -> float:
            leg = priced[priced["option_type"] == side]
            leg = leg[leg["delta"].notna()]
            if leg.empty:
                return float("nan")
            idx = (leg["delta"].abs() - target).abs().idxmin()
            return float(leg.loc[idx, "implied_volatility"])

        put_25 = nearest_delta("PE", 0.25)
        call_25 = nearest_delta("CE", 0.25)

        rows.append({
            "date": grp["date"].iloc[0],
            "expiry": expiry,
            "days_to_expiry": int(grp["days_to_expiry"].iloc[0]),
            "forward": forward,
            "atm_strike": float(atm_strike),
            "atm_iv": atm_iv,
            "skew_25d": put_25 - call_25,
            "priced_strikes": int(priced["strike"].nunique()),
        })
    return pd.DataFrame(rows).sort_values("days_to_expiry").reset_index(drop=True)
