"""Quantify what the open_interest > 0 tradability filter removes.

Approved decision 4 requires the filter be applied consistently across both
schema eras and its impact documented. This reports, per era, what the filter
drops overall and -- the part that actually matters for trade construction --
what it drops from the strikes a weekly strategy would reach for.

    python scripts/measure_filter_impact.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from swarm_opt.data.normalize import normalize_bhavcopy  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "fixture_nifty" / "filter_impact.json"

# One representative date per era, with the index close for the legacy one.
CASES = [
    ("legacy", RAW / "fo_bhavcopy_legacy_20230717.parquet", 19711.45),
    ("udiff", RAW / "fo_bhavcopy_18082026.parquet", None),
]

#: Strikes within this fraction of the forward are the ones a weekly spread,
#: condor or butterfly can actually reach.
TRADEABLE_BAND = 0.10
TRADEABLE_DTE = 45


def summarise(df: pd.DataFrame, label: str) -> dict:
    near = df[df["days_to_expiry"] <= TRADEABLE_DTE]
    band = near[abs(near["strike"] - near["forward"]) <= TRADEABLE_BAND * near["forward"]]
    return {
        "label": label,
        "rows": int(len(df)),
        "strikes": int(df["strike"].nunique()),
        "expiries": int(df["expiry"].nunique()),
        "traded_rows": int((df["volume"] > 0).sum()),
        "near_expiry_rows": int(len(near)),
        "tradeable_band_rows": int(len(band)),
        "tradeable_band_strikes": int(band["strike"].nunique()),
        "forward_missing_expiries": int(
            (df.groupby("expiry")["forward_source"].first() == "missing").sum()
        ),
    }


def main() -> None:
    results = []
    for era, path, spot in CASES:
        raw = pd.read_parquet(path)
        unfiltered = normalize_bhavcopy(raw, "NIFTY", spot=spot, min_open_interest=0)
        filtered = normalize_bhavcopy(raw, "NIFTY", spot=spot, min_open_interest=1)

        before = summarise(unfiltered, "min_oi=0")
        after = summarise(filtered, "min_oi=1")
        results.append({
            "era": era,
            "date": str(unfiltered["date"].iloc[0].date()),
            "before": before,
            "after": after,
            "dropped_rows": before["rows"] - after["rows"],
            "dropped_row_pct": round(
                100 * (1 - after["rows"] / before["rows"]), 2
            ),
            "dropped_tradeable_band_rows": (
                before["tradeable_band_rows"] - after["tradeable_band_rows"]
            ),
            "dropped_tradeable_band_pct": round(
                100 * (1 - after["tradeable_band_rows"] / before["tradeable_band_rows"]), 2
            ),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")

    for r in results:
        print(f"\n=== {r['era']} ({r['date']}) ===")
        print(f"  all rows        {r['before']['rows']:>6} -> {r['after']['rows']:>6} "
              f"({r['dropped_row_pct']}% dropped)")
        print(f"  tradeable band  {r['before']['tradeable_band_rows']:>6} -> "
              f"{r['after']['tradeable_band_rows']:>6} "
              f"({r['dropped_tradeable_band_pct']}% dropped)")
        print(f"  band strikes    {r['before']['tradeable_band_strikes']:>6} -> "
              f"{r['after']['tradeable_band_strikes']:>6}")
        print(f"  traded rows     {r['before']['traded_rows']:>6} -> "
              f"{r['after']['traded_rows']:>6}")
        print(f"  expiries w/o fwd{r['before']['forward_missing_expiries']:>6} -> "
              f"{r['after']['forward_missing_expiries']:>6}")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
