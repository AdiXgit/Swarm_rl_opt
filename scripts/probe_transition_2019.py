"""Phase 2.5: pin the NIFTY weekly-options transition date.

Phase 2.4's monthly probe found MONTHLY_ONLY on 2019-01-16 and
WEEKLY_COMPATIBLE on 2019-02-15. This makes exactly 4 additional, evenly
spaced, targeted requests inside that 30-day window to narrow it, using the
same unmodified NseKitProvider mechanism and the same max_rps=1.0. Writes to
a new file; never touches data/raw/history_probe.json or
data/raw/history_probe_2018_2020.json.

    python scripts/probe_transition_2019.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from swarm_opt.data.providers.nsekit_provider import NseKitProvider  # noqa: E402

OUT = ROOT / "data" / "raw" / "history_probe_transition_2019.json"
CANDIDATE_DATES = [date(2019, 1, 24), date(2019, 1, 31), date(2019, 2, 7), date(2019, 2, 14)]
assert len(CANDIDATE_DATES) <= 4


def main() -> int:
    provider = NseKitProvider(max_rps=1.0)
    rows = []
    for i, on in enumerate(CANDIDATE_DATES, 1):
        row = {"requested_date": on.isoformat()}
        try:
            chain = provider.option_chain("NIFTY", on)
        except Exception as exc:
            row["access_error"] = f"{type(exc).__name__}: {exc}"
            row["classification"] = "ACCESS_FAILED"
            rows.append(row)
            print(f"  [{i}/4] {on}: ACCESS_FAILED {row['access_error']}", flush=True)
            continue

        if chain.empty:
            row["classification"] = "NON_TRADING_DAY_OR_NO_ARCHIVE"
            rows.append(row)
            print(f"  [{i}/4] {on}: empty (non-trading day or no archive)", flush=True)
            continue

        expiries = sorted(chain["expiry"].dt.date.unique())
        front = expiries[0]
        gap = (expiries[1] - front).days if len(expiries) >= 2 else None
        row.update({
            "chain_available": True,
            "rows": int(len(chain)),
            "distinct_expiries": [e.isoformat() for e in expiries],
            "front_expiry": front.isoformat(),
            "front_days_to_expiry": (front - on).days,
            "front_two_expiry_gap_days": gap,
            "classification": "WEEKLY_COMPATIBLE" if gap is not None and gap <= 10 else "MONTHLY_ONLY_OR_NO_WEEKLY_EVIDENCE",
            "evidence": f"front expiry {front}, gap to next listed expiry {gap} days",
        })
        rows.append(row)
        print(f"  [{i}/4] {on}: {row['classification']} front={front} gap={gap}d", flush=True)

    OUT.write_text(json.dumps({
        "purpose": "pin the NIFTY weekly-options transition between 2019-01-16 (confirmed monthly) and 2019-02-15 (confirmed weekly)",
        "request_bound": 4,
        "requests_made": len(rows),
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
