"""Build the local Parquet fixture and the provider validation report.

Pulls a small sample through NseKitProvider, normalizes it, writes it in the
LocalParquetProvider layout, and records exactly what NSE did and did not
supply. Deliberately small: this is the Week-1 validation sample, not the
research dataset.

    python scripts/build_fixture.py
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

UNDERLYING = "NIFTY"
FIXTURE = ROOT / "data" / "processed" / "fixture_nifty"

# Five consecutive trading dates on each bhavcopy schema, so the fixture
# exercises both ETL paths. 2026-08-15 is an NSE holiday.
UDIFF_DATES = [date(2026, 8, d) for d in (12, 13, 14, 17, 18)]
LEGACY_DATES = [date(2023, 7, d) for d in (17, 18, 19, 20, 21)]
SAMPLE_DATES = LEGACY_DATES + UDIFF_DATES


def main() -> None:
    provider = NseKitProvider(max_rps=1.0)
    (FIXTURE / "underlying").mkdir(parents=True, exist_ok=True)

    hist = pd.concat(
        [
            provider.underlying_history(UNDERLYING, min(LEGACY_DATES), max(LEGACY_DATES)),
            provider.underlying_history(UNDERLYING, min(UDIFF_DATES), max(UDIFF_DATES)),
        ]
    ).sort_values("date").reset_index(drop=True)
    hist.to_parquet(FIXTURE / "underlying" / f"{UNDERLYING}.parquet", index=False)

    per_date: list[dict] = []
    for on in SAMPLE_DATES:
        chain = provider.option_chain(UNDERLYING, on)
        if chain.empty:
            per_date.append({"date": on.isoformat(), "rows": 0, "note": "empty"})
            print(f"{on}: EMPTY", flush=True)
            continue

        out_dir = FIXTURE / "chains" / f"underlying={UNDERLYING}" / f"date={on.isoformat()}"
        out_dir.mkdir(parents=True, exist_ok=True)
        chain.to_parquet(out_dir / "part.parquet", index=False)

        near = chain[chain["expiry"] == chain["expiry"].min()]
        report = provider.missingness(chain)
        per_date.append({
            "date": on.isoformat(),
            "schema": chain["schema_version"].iloc[0],
            "rows": int(len(chain)),
            "spot": float(chain["spot"].iloc[0]),
            "lot_size": int(chain["lot_size"].iloc[0]),
            "expiries": int(chain["expiry"].nunique()),
            "strikes": int(chain["strike"].nunique()),
            "traded_rows": int(chain["is_traded"].sum()),
            "ce_rows": int((chain["option_type"] == "CE").sum()),
            "pe_rows": int((chain["option_type"] == "PE").sum()),
            "nearest_expiry": str(chain["expiry"].min().date()),
            "nearest_expiry_strikes": int(near["strike"].nunique()),
            "forward_sources": chain.groupby("expiry")["forward_source"].first()
                                    .value_counts().to_dict(),
            "missingness": report.to_dict(),
            "duplicate_keys": int(
                chain.duplicated(["date", "expiry", "strike", "option_type"]).sum()
            ),
        })
        print(
            f"{on}: schema={per_date[-1]['schema']} rows={len(chain)} "
            f"expiries={per_date[-1]['expiries']} strikes={per_date[-1]['strikes']} "
            f"lot={per_date[-1]['lot_size']}",
            flush=True,
        )

    out = {
        "underlying": UNDERLYING,
        "fixture_root": str(FIXTURE.relative_to(ROOT)),
        "underlying_rows": int(len(hist)),
        "per_date": per_date,
    }
    (FIXTURE / "validation_report.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8"
    )
    print("\nwrote", FIXTURE / "validation_report.json")


if __name__ == "__main__":
    main()
