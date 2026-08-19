"""Map the structural regimes across the candidate dataset window.

Samples one trading date per month and records the bhavcopy schema, the NIFTY
board lot, the expiry ladder and the weekly-expiry weekday. This is what the
train/validation/test boundaries have to be placed against: a split that lands
mid-regime hands the model a distribution shift the metrics cannot see.

Cheap by design -- roughly one request per month, not per day.

    python scripts/probe_history.py --start 2021-01-01 --end 2026-08-18
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from swarm_opt.data.providers.nsekit_provider import NseKitProvider  # noqa: E402

OUT = ROOT / "data" / "raw" / "history_probe.json"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=date.fromisoformat, default=date(2021, 1, 1))
    p.add_argument("--end", type=date.fromisoformat, default=date(2026, 8, 18))
    p.add_argument("--underlying", default="NIFTY")
    p.add_argument("--max-rps", type=float, default=1.0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    provider = NseKitProvider(max_rps=args.max_rps)

    hist = provider.underlying_history(args.underlying, args.start, args.end)
    if hist.empty:
        print("no index history", file=sys.stderr)
        return 1

    # mid-month trading date, one per month
    hist = hist.assign(ym=hist["date"].dt.to_period("M"))
    picks = [g.iloc[len(g) // 2]["date"].date() for _, g in hist.groupby("ym")]
    print(f"probing {len(picks)} monthly samples {picks[0]} .. {picks[-1]}")

    rows = []
    for i, on in enumerate(picks, 1):
        try:
            chain = provider.option_chain(args.underlying, on)
        except Exception as exc:
            print(f"  {on}: FAILED {exc}", flush=True)
            rows.append({"date": on.isoformat(), "ok": False, "error": str(exc)})
            continue
        if chain.empty:
            print(f"  {on}: EMPTY", flush=True)
            rows.append({"date": on.isoformat(), "ok": False, "error": "empty"})
            continue

        near = chain[chain["days_to_expiry"] <= 45]
        front_expiry = chain["expiry"].min()
        rows.append({
            "date": on.isoformat(),
            "ok": True,
            "schema": chain["schema_version"].iloc[0],
            "lot_size": int(chain["lot_size"].iloc[0]),
            "spot": round(float(chain["spot"].iloc[0]), 2),
            "rows": int(len(chain)),
            "expiries": int(chain["expiry"].nunique()),
            "near_expiries": int(near["expiry"].nunique()),
            "strikes": int(chain["strike"].nunique()),
            "front_expiry": str(front_expiry.date()),
            "front_expiry_weekday": front_expiry.day_name(),
            "strike_step": float(
                pd.Series(sorted(near["strike"].unique())).diff().mode().iloc[0]
            ) if len(near) > 2 else None,
            "forward_missing_near": int(
                (near.groupby("expiry")["forward_source"].first() == "missing").sum()
            ),
        })
        print(
            f"  [{i}/{len(picks)}] {on}: schema={rows[-1]['schema']} "
            f"lot={rows[-1]['lot_size']} rows={rows[-1]['rows']} "
            f"front={rows[-1]['front_expiry_weekday']}",
            flush=True,
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    ok = [r for r in rows if r.get("ok")]
    print("\n=== regimes ===")
    for key in ("schema", "lot_size", "front_expiry_weekday"):
        seen: dict = {}
        for r in ok:
            seen.setdefault(r[key], []).append(r["date"])
        print(f"\n{key}:")
        for val, ds in seen.items():
            print(f"  {val}: {ds[0]} .. {ds[-1]}  ({len(ds)} months)")
    print("\nwrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
