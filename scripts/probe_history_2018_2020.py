"""Phase 2.4 bounded historical probe, 2018-01-01 .. 2020-12-31.

A NEW, standalone investigation script -- it does not modify
`scripts/probe_history.py` and does not touch its output
(`data/raw/history_probe.json`). Writes to a separate, uniquely-named file
(`data/raw/history_probe_2018_2020.json`) so the existing probe record is
preserved exactly as-is.

Same access mechanism as the existing, already-approved `probe_history.py`:
`NseKitProvider` at `max_rps=1.0` (below NseKit's own default of 3.0), one
representative trading date per month, no bulk collection, no retries beyond
NseKitProvider's own built-in retry/backoff. Bounded to at most 36 requested
dates (Jan 2018 - Dec 2020 is exactly 36 months).

    python scripts/probe_history_2018_2020.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from swarm_opt.data.providers.nsekit_provider import NseKitProvider  # noqa: E402

OUT = ROOT / "data" / "raw" / "history_probe_2018_2020.json"
START = date(2018, 1, 1)
END = date(2020, 12, 31)
MAX_REQUESTED_DATES = 36


def classify(row: dict) -> str:
    if not row.get("is_trading_session", True):
        return "NON_TRADING_DAY"
    if row.get("access_error"):
        return "ACCESS_FAILED"
    if not row.get("chain_available"):
        return "DATA_AVAILABLE_BUT_INSUFFICIENT" if row.get("index_available") else "ACCESS_FAILED"
    if row.get("rows", 0) == 0:
        return "DATA_AVAILABLE_BUT_INSUFFICIENT"
    spacing = row.get("front_two_expiry_gap_days")
    if spacing is None:
        return "UNRESOLVED"
    # A gap of <= 10 days between the front and second-listed expiry is the
    # direct signature of a weekly cadence; anything close to ~28-35 days (or
    # a front expiry sitting ~28-35 days out itself) is a monthly cadence.
    # This checks the SAMPLED date only -- it is not inferred onto any date
    # that was not actually requested.
    if spacing <= 10:
        return "WEEKLY_COMPATIBLE"
    return "MONTHLY_ONLY_OR_NO_WEEKLY_EVIDENCE"


def main() -> int:
    provider = NseKitProvider(max_rps=1.0)

    hist = provider.underlying_history("NIFTY", START, END)
    if hist.empty:
        print("no index history at all for 2018-2020; stopping, nothing else requested", file=sys.stderr)
        OUT.write_text(json.dumps({"error": "no index history for 2018-2020", "rows": []}, indent=2))
        return 1

    hist = hist.assign(ym=hist["date"].dt.to_period("M"))
    picks = [g.iloc[len(g) // 2]["date"].date() for _, g in hist.groupby("ym")]
    picks = picks[:MAX_REQUESTED_DATES]
    print(f"requesting {len(picks)} monthly samples: {picks[0]} .. {picks[-1]} (bound: {MAX_REQUESTED_DATES})")

    rows = []
    for i, on in enumerate(picks, 1):
        row = {
            "requested_date": on.isoformat(),
            "is_trading_session": True,  # drawn from the index series itself, so true by construction
            "index_available": True,     # same reason
            "access_error": None,
            "chain_available": False,
            "rows": 0,
            "distinct_expiries": [],
            "front_expiry": None,
            "front_days_to_expiry": None,
            "front_two_expiry_gap_days": None,
            "near_strikes": None,
            "schema_version": None,
            "lot_size": None,
            "evidence": None,
        }
        try:
            chain = provider.option_chain("NIFTY", on)
        except Exception as exc:
            row["access_error"] = f"{type(exc).__name__}: {exc}"
            row["evidence"] = "exception raised by NseKitProvider.option_chain"
            row["classification"] = classify(row)
            rows.append(row)
            print(f"  [{i}/{len(picks)}] {on}: ACCESS_FAILED ({row['access_error']})", flush=True)
            continue

        if chain.empty:
            row["evidence"] = "option_chain returned an empty, correctly-typed frame (no archive file for this date)"
            row["classification"] = classify(row)
            rows.append(row)
            print(f"  [{i}/{len(picks)}] {on}: no chain (empty)", flush=True)
            continue

        row["chain_available"] = True
        row["rows"] = int(len(chain))
        expiries = sorted(chain["expiry"].dt.date.unique())
        row["distinct_expiries"] = [e.isoformat() for e in expiries]
        row["schema_version"] = str(chain["schema_version"].iloc[0])
        row["lot_size"] = int(chain["lot_size"].iloc[0])

        if expiries:
            front = expiries[0]
            row["front_expiry"] = front.isoformat()
            row["front_days_to_expiry"] = (front - on).days
            near = chain[chain["expiry"].dt.date == front]
            row["near_strikes"] = int(near["strike"].nunique())
            if len(expiries) >= 2:
                row["front_two_expiry_gap_days"] = (expiries[1] - front).days
            row["evidence"] = (
                f"{len(expiries)} distinct expiries listed; front expiry {front} is "
                f"{row['front_days_to_expiry']} days out; gap to next listed expiry is "
                f"{row['front_two_expiry_gap_days']} days"
            )
        else:
            row["evidence"] = "chain had rows but zero distinct expiry values (unexpected)"

        row["classification"] = classify(row)
        rows.append(row)
        print(
            f"  [{i}/{len(picks)}] {on}: {row['classification']} "
            f"front={row['front_expiry']} gap_to_next={row['front_two_expiry_gap_days']}d "
            f"schema={row['schema_version']} lot={row['lot_size']}",
            flush=True,
        )

    out_doc = {
        "probe_window": [START.isoformat(), END.isoformat()],
        "max_requested_dates_bound": MAX_REQUESTED_DATES,
        "actual_requested_dates": len(picks),
        "access_mechanism": "NseKitProvider (unmodified), max_rps=1.0, same as scripts/probe_history.py",
        "rows": rows,
    }
    OUT.write_text(json.dumps(out_doc, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
