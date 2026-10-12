"""Phase 2.8 Workstream B: cache India VIX EOD history.

Uses the EXISTING, unmodified `NseKitProvider.underlying_history` -- "INDIA
VIX" is not in its `INDEX_NAME` map, so the lookup falls through to the raw
string unchanged, which NSE's own `index_historical_data` already accepts.
No new provider, no new endpoint, no code change to the provider.

One bulk range request (same pattern as the project's own NIFTY index
history, not a per-date loop), covering the full combined
EXTENDED-TRAIN..TEST-end range so the cache is reusable -- but per this
phase's protocol, only TRAIN/VALIDATION rows are ever used in any feature
comparison; TEST dates are cached here only because the range is
contiguous, never read for selection, tuning, or evaluation.

Known data-quality note: 2021-03-30 prints open=high=low=0.0 with a
nonzero close in NSE's own feed (the same date the F&O bhavcopy extension
build separately found missing, Phase 2.5) -- treated as a documented,
excluded observation, not silently used.

    python scripts/build_india_vix_history.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from swarm_opt.data.providers.nsekit_provider import NseKitProvider  # noqa: E402

START, END = date(2019, 2, 18), date(2026, 8, 18)
OUT = ROOT / "data" / "features" / "india_vix_history.parquet"


def main() -> int:
    provider = NseKitProvider(max_rps=1.0)
    hist = provider.underlying_history("INDIA VIX", START, END)
    if hist.empty:
        print("no India VIX history returned", file=sys.stderr)
        return 1

    bad = (hist["open"] == 0) & (hist["high"] == 0) & (hist["low"] == 0)
    n_bad = int(bad.sum())
    hist = hist.rename(columns={"close": "india_vix_close"})[["date", "india_vix_close"]].copy()
    hist.loc[bad.to_numpy(), "india_vix_close"] = np.nan  # documented exclusion, not silently kept

    OUT.parent.mkdir(parents=True, exist_ok=True)
    hist.to_parquet(OUT, index=False)
    print(f"wrote {len(hist)} rows ({n_bad} flagged as zero-OHL data-quality exclusions) -> {OUT}")
    print(f"range: {hist['date'].min()} .. {hist['date'].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
