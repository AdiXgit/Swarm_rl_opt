"""Bulk EOD dataset build. Resumable, manifested, rate-limited.

Pulls one F&O bhavcopy per trading date, normalizes it to the data contract and
writes it in the LocalParquetProvider layout. Safe to interrupt and re-run: a
date already present on disk is skipped unless --overwrite is passed.

The trading calendar comes from the index series, so holidays are never
requested and a genuinely empty archive day is distinguishable from a holiday.

    python scripts/build_dataset.py --start 2021-04-01 --end 2026-08-18
    python scripts/build_dataset.py --start 2021-04-01 --end 2026-08-18 --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from swarm_opt.data.normalize import MIN_OPEN_INTEREST  # noqa: E402
from swarm_opt.data.providers.nsekit_provider import NseKitProvider  # noqa: E402

DEFAULT_ROOT = ROOT / "data" / "processed" / "nifty_eod"


@dataclass
class BuildStats:
    requested: int = 0
    written: int = 0
    skipped_existing: int = 0
    empty: int = 0
    failed: int = 0
    rows: int = 0
    failures: list[str] = field(default_factory=list)
    empties: list[str] = field(default_factory=list)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", required=True, type=date.fromisoformat)
    p.add_argument("--end", required=True, type=date.fromisoformat)
    p.add_argument("--underlying", default="NIFTY")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--max-rps", type=float, default=1.0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve the calendar and report the plan without fetching chains",
    )
    return p.parse_args(argv)


def chain_path(root: Path, underlying: str, on: date) -> Path:
    return root / "chains" / f"underlying={underlying}" / f"date={on.isoformat()}" / "part.parquet"


def main(argv=None) -> int:
    args = parse_args(argv)
    provider = NseKitProvider(max_rps=args.max_rps)

    # The index series defines the trading calendar, so holidays are never
    # requested and cannot be mistaken for missing data.
    hist = provider.underlying_history(args.underlying, args.start, args.end)
    if hist.empty:
        print("no index history for that range; nothing to do", file=sys.stderr)
        return 1
    dates = [d.date() for d in hist["date"]]

    (args.root / "underlying").mkdir(parents=True, exist_ok=True)
    hist.to_parquet(args.root / "underlying" / f"{args.underlying}.parquet", index=False)

    pending = [
        d for d in dates
        if args.overwrite or not chain_path(args.root, args.underlying, d).exists()
    ]
    print(
        f"{len(dates)} trading dates {dates[0]} .. {dates[-1]}; "
        f"{len(pending)} to fetch, {len(dates) - len(pending)} already on disk"
    )
    if args.dry_run:
        print("dry run: no chains fetched")
        return 0

    stats = BuildStats(skipped_existing=len(dates) - len(pending))
    started = time.time()

    for i, on in enumerate(pending, 1):
        stats.requested += 1
        try:
            chain = provider.option_chain(args.underlying, on)
        except Exception as exc:  # keep going; record and move on
            stats.failed += 1
            stats.failures.append(f"{on}: {type(exc).__name__}: {exc}")
            print(f"  [{i}/{len(pending)}] {on}: FAILED {exc}", flush=True)
            continue

        if chain.empty:
            # A trading date with no archive file. Recorded, not silently lost.
            stats.empty += 1
            stats.empties.append(on.isoformat())
            print(f"  [{i}/{len(pending)}] {on}: EMPTY", flush=True)
            continue

        out = chain_path(args.root, args.underlying, on)
        out.parent.mkdir(parents=True, exist_ok=True)
        chain.to_parquet(out, index=False)
        stats.written += 1
        stats.rows += len(chain)

        if i % 25 == 0 or i == len(pending):
            rate = i / max(time.time() - started, 1e-9)
            eta = (len(pending) - i) / max(rate, 1e-9)
            print(
                f"  [{i}/{len(pending)}] {on}: {len(chain)} rows "
                f"({rate:.2f} dates/s, eta {eta/60:.1f} min)",
                flush=True,
            )

    manifest = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "underlying": args.underlying,
        "requested_range": [args.start.isoformat(), args.end.isoformat()],
        "trading_dates": len(dates),
        "first_date": dates[0].isoformat(),
        "last_date": dates[-1].isoformat(),
        "provider": provider.provider_id,
        "min_open_interest": MIN_OPEN_INTEREST,
        "stats": asdict(stats),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (args.root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"\nwrote {stats.written} dates ({stats.rows} rows), "
        f"{stats.empty} empty, {stats.failed} failed -> {args.root / 'manifest.json'}"
    )
    return 1 if stats.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
