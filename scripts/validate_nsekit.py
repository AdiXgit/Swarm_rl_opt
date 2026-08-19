"""Week-1 validation probe for NseKit as the SWARM-OPT EOD options provider.

Read-only. Small sample. Rate limited well below the library default.
Writes a JSON report to data/raw/nsekit_validation_report.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
from NseKit import NseKit as NK

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw"
OUT.mkdir(parents=True, exist_ok=True)

# 5 consecutive NSE trading dates (DD-MM-YYYY). 2026-08-15 is a Saturday holiday.
SAMPLE_DATES = ["12-08-2026", "13-08-2026", "14-08-2026", "17-08-2026", "18-08-2026"]

nse = NK.Nse(max_rps=1.0, retries=2, retry_delay=2.0)
report: dict = {"sample_dates": SAMPLE_DATES, "bhavcopy": {}}


def probe_bhavcopy(date: str) -> dict:
    t0 = time.time()
    df = nse.fno_eod_bhav_copy(date)
    dt = round(time.time() - t0, 2)
    if df is None or df.empty:
        return {"ok": False, "seconds": dt, "rows": 0}
    raw_path = OUT / f"fo_bhavcopy_{date.replace('-', '')}.parquet"
    df.to_parquet(raw_path, index=False)
    return {
        "ok": True,
        "seconds": dt,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "saved": str(raw_path.relative_to(ROOT)),
    }


for d in SAMPLE_DATES:
    r = probe_bhavcopy(d)
    report["bhavcopy"][d] = r
    print(f"[bhavcopy] {d}: ok={r['ok']} rows={r['rows']} in {r['seconds']}s", flush=True)

(OUT / "nsekit_validation_report.json").write_text(
    json.dumps(report, indent=2, default=str), encoding="utf-8"
)
print("\nwrote", OUT / "nsekit_validation_report.json")
