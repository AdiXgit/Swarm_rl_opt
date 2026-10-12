"""Precompute per-date ATM IV / skew for the nearest tradeable expiry.

This is the one external-chain-derived input the Phase 2 feature pipeline
needs beyond the underlying OHLC series. It is expensive (one Black-76 IV
solve per near-ATM strike per date) so it is cached to Parquet once here
rather than recomputed on every feature build.

Point-in-time: for date `d` only the chain observed on `d` is read, and only
the nearest expiry listed on `d` is used. Nothing from a later date enters
the row for `d`. The per-date logic lives in
`swarm_opt.features.atm_cache` (Phase 2.2, Task A1) so it can be
unit-tested directly; this script is now just the CLI wrapper.

    python scripts/build_market_features.py
    python scripts/build_market_features.py --root data/processed/nifty_eod
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from swarm_opt.data.providers.local_parquet_provider import LocalParquetProvider  # noqa: E402
from swarm_opt.data.splits import DATASET_END, DATASET_START  # noqa: E402
from swarm_opt.features.atm_cache import build_atm_iv_row  # noqa: E402

DEFAULT_ROOT = ROOT / "data" / "processed" / "nifty_eod"
OUT_PATH = ROOT / "data" / "features" / "atm_iv_history.parquet"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--underlying", default="NIFTY")
    p.add_argument("--out", type=Path, default=OUT_PATH)
    return p.parse_args(argv)


def main(argv=None) -> int:
    import pandas as pd

    args = parse_args(argv)
    provider = LocalParquetProvider(args.root)
    dates = provider.trading_dates(DATASET_START, DATASET_END)
    print(f"{len(dates)} trading dates", flush=True)

    rows = []
    started = time.time()
    for i, on in enumerate(dates, 1):
        chain = provider.option_chain(args.underlying, on)
        row = build_atm_iv_row(chain, on)
        if row is not None:
            rows.append(row)
        if i % 100 == 0 or i == len(dates):
            rate = i / max(time.time() - started, 1e-9)
            eta = (len(dates) - i) / max(rate, 1e-9)
            print(f"  [{i}/{len(dates)}] {on} ({rate:.2f} dates/s, eta {eta/60:.1f} min)", flush=True)

    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"wrote {len(out)} rows -> {args.out} ({time.time() - started:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
