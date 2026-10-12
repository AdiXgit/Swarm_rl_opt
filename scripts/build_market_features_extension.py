"""ATM IV / skew cache builder for the Phase 2.5 historical extension.

`scripts/build_market_features.py` hardcodes `DATASET_START`/`DATASET_END`
from `swarm_opt.data.splits` (the frozen 2021-2026 range) rather than
deriving the date range from whatever `--root` it is pointed at -- fine for
the original dataset (they coincide), wrong for the extension directory
(2019-02-18..2021-03-31), where it silently resolved to zero trading dates.
Rather than changing that validated script's behavior for the original
dataset, this is a small, separate script that calls the same underlying
`swarm_opt.features.atm_cache` function with the extension's own date range.

    python scripts/build_market_features_extension.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from swarm_opt.data.providers.local_parquet_provider import LocalParquetProvider  # noqa: E402
from swarm_opt.features.atm_cache import build_atm_iv_history  # noqa: E402

EXT_ROOT = ROOT / "data" / "processed" / "nifty_eod_ext_2019_2021"
EXT_START, EXT_END = date(2019, 2, 18), date(2021, 3, 31)
OUT_PATH = ROOT / "data" / "features" / "atm_iv_history_ext_2019_2021.parquet"


def main() -> int:
    provider = LocalParquetProvider(EXT_ROOT)
    dates = provider.trading_dates(EXT_START, EXT_END)
    print(f"{len(dates)} extension trading dates {dates[0]} .. {dates[-1]}", flush=True)

    history = build_atm_iv_history(provider, "NIFTY", dates)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    history.to_parquet(OUT_PATH, index=False)
    print(f"wrote {len(history)} rows -> {OUT_PATH} ({len(history)}/{len(dates)} dates priced)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
