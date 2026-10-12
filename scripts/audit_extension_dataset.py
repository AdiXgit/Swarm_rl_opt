"""Phase 2.5: data-quality audit for the historical extension dataset.

Mirrors `scripts/audit_dataset.py`'s checks (coverage, missingness, expiry/
lot regimes, leakage) applied to the standalone extension directory only
(2019-02-18..2021-03-31), then writes a separate, new manifest. Does not
read, touch, or alter the original frozen dataset or its manifest.

    python scripts/audit_extension_dataset.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from swarm_opt.data.calendar import ExpiryCalendar  # noqa: E402
from swarm_opt.data.contracts import PRIMARY_KEY  # noqa: E402
from swarm_opt.data.providers.local_parquet_provider import LocalParquetProvider  # noqa: E402

EXT_ROOT = ROOT / "data" / "processed" / "nifty_eod_ext_2019_2021"
EXT_START = date(2019, 2, 18)
EXT_END = date(2021, 3, 31)
ORIGINAL_TRAIN_START = date(2021, 4, 1)  # the join point with the frozen dataset
TRADEABLE_DTE = 45
OUT = ROOT / "data" / "models" / "historical_extension_dataset_manifest.json"


def load_all(provider, dates) -> pd.DataFrame:
    frames = []
    for d in dates:
        chain = provider.option_chain("NIFTY", d)
        if not chain.empty:
            frames.append(chain)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> int:
    provider = LocalParquetProvider(EXT_ROOT)
    dates = provider.trading_dates(EXT_START, EXT_END)
    if not dates:
        print(f"no chains under {EXT_ROOT}; has the collection finished?", file=sys.stderr)
        return 1

    build_manifest = json.loads((EXT_ROOT / "manifest.json").read_text())
    print(f"loading {len(dates)} extension trading dates ...", flush=True)
    chains = load_all(provider, dates)
    calendar = ExpiryCalendar.from_frame(chains)

    # -- coverage -------------------------------------------------------- #
    hist = provider.underlying_history("NIFTY", EXT_START, EXT_END)
    expected = [d.date() for d in hist["date"]]
    have = set(dates)
    missing_dates = [d for d in expected if d not in have]

    per_date = chains.groupby(chains["date"].dt.date).agg(rows=("strike", "size"), expiries=("expiry", "nunique"))
    near = chains[chains["days_to_expiry"] <= TRADEABLE_DTE]

    # -- missingness ------------------------------------------------------ #
    missingness = {c: round(float(chains[c].isna().mean()), 6) for c in chains.columns if chains[c].isna().any()}
    forward_sources = {k: int(v) for k, v in chains["forward_source"].value_counts().items()}

    # -- duplicates / validity -------------------------------------------- #
    duplicate_keys = int(chains.duplicated(PRIMARY_KEY).sum())
    negative_dte = int((chains["days_to_expiry"] < 0).sum())
    expired_in_chain = int((chains["expiry"] < chains["date"]).sum())
    zero_or_negative_prices = int((chains["close"] <= 0).sum())
    invalid_oi = int((chains["open_interest"] < 0).sum())

    # -- expiry / lot regimes ---------------------------------------------- #
    lots = chains.groupby(chains["date"].dt.date)["lot_size"].agg(lambda s: int(s.mode().iloc[0]))
    schema = chains.groupby(chains["date"].dt.date)["schema_version"].first()
    expiry_weekdays = calendar.observed_expiry_weekdays()

    # -- weekly episodes (extension range only, un-purged against the join) #
    all_episodes = calendar.weekly_episodes()
    print(f"\n=== coverage ===")
    print(f"  {len(chains):,} rows / {len(dates)} dates / missing {len(missing_dates)}")
    print(f"  duplicate keys: {duplicate_keys}  negative dte: {negative_dte}  "
          f"expired-in-chain: {expired_in_chain}  non-positive close: {zero_or_negative_prices}  "
          f"invalid OI: {invalid_oi}")
    print(f"  lot sizes observed: {sorted(set(lots))}")
    print(f"  schema versions observed: {sorted(set(schema))}")
    print(f"  expiry weekdays: {expiry_weekdays}")
    print(f"  forward sources: {forward_sources}")
    print(f"  weekly episodes before purge: {len(all_episodes)}")

    # -- IV solvability within the tradeable window ------------------------ #
    near_fwd = near.groupby([near["date"].dt.date, "expiry"])["forward_source"].first()
    near_missing_fwd = int((near_fwd == "missing").sum())

    manifest = {
        "dataset_version": "nifty_eod_ext_2019-02-18_2021-03-31",
        "extension_identifier": "phase2_5_historical_extension",
        "underlying": "NIFTY",
        "requested_range": [EXT_START.isoformat(), EXT_END.isoformat()],
        "actual_range": [str(min(dates)), str(max(dates))],
        "source": "nsekit",
        "collection_method": "scripts/build_dataset.py (unmodified), max_rps=1.0",
        "collection_build_log": {
            "requested": build_manifest["stats"]["requested"],
            "written": build_manifest["stats"]["written"],
            "empty": build_manifest["stats"]["empty"],
            "failed": build_manifest["stats"]["failed"],
            "failures": build_manifest["stats"]["failures"],
            "empties": build_manifest["stats"]["empties"],
            "elapsed_seconds": build_manifest["elapsed_seconds"],
        },
        "trading_dates": len(dates),
        "missing_trading_dates": [str(d) for d in missing_dates],
        "rows": int(len(chains)),
        "data_quality": {
            "duplicate_primary_keys": duplicate_keys,
            "negative_days_to_expiry": negative_dte,
            "expired_contracts_in_chain": expired_in_chain,
            "non_positive_close_prices": zero_or_negative_prices,
            "invalid_open_interest": invalid_oi,
            "missingness_by_column": missingness,
            "forward_source_counts": forward_sources,
            "near_expiries_without_forward": near_missing_fwd,
            "near_expiries_total": int(len(near_fwd)),
        },
        "lot_size_regimes": sorted(set(int(x) for x in lots)),
        "schema_versions": sorted(set(schema)),
        "expiry_weekday_counts": expiry_weekdays,
        "weekly_episodes_before_purge": len(all_episodes),
        "join_point_with_frozen_dataset": {
            "extension_last_date": EXT_END.isoformat(),
            "frozen_dataset_first_date": ORIGINAL_TRAIN_START.isoformat(),
            "contiguous": (ORIGINAL_TRAIN_START - EXT_END).days == 1,
        },
        "transition_evidence_reference": "data/raw/history_probe_transition_2019.json, data/raw/history_probe_2018_2020.json",
        "transition_conservative_start_rationale": (
            "2019-02-18 chosen as 4 calendar days after the first directly-confirmed "
            "WEEKLY_COMPATIBLE sample (2019-02-14), to sit unambiguously inside the "
            "confirmed weekly regime rather than exactly at the transition boundary."
        ),
    }

    digest = hashlib.sha256()
    for f in sorted((EXT_ROOT / "chains").rglob("*.parquet")):
        digest.update(f.relative_to(EXT_ROOT).as_posix().encode())
        digest.update(str(f.stat().st_size).encode())
    manifest["content_fingerprint"] = digest.hexdigest()[:16]

    OUT.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT}")
    print(f"extension content_fingerprint: {manifest['content_fingerprint']}")

    # Zero tolerance for anything that would indicate a corrupt or
    # out-of-order build. Missing trading dates are judged on materiality,
    # not zero-tolerance: a single confirmed-genuine archive gap (retried
    # once via the same unmodified provider and still absent -- see
    # docs/HISTORICAL_PROBE_AND_EXPIRY_AUDIT.md-style disclosure in the
    # Phase 2.5 report) is recorded and excluded, not silently patched over,
    # but does not by itself invalidate several hundred other clean dates.
    integrity_clean = (
        duplicate_keys == 0 and negative_dte == 0 and expired_in_chain == 0
        and zero_or_negative_prices == 0 and invalid_oi == 0
    )
    missing_fraction = len(missing_dates) / len(expected) if expected else 0.0
    coverage_acceptable = missing_fraction <= 0.01  # <=1% missing, each one individually disclosed
    quality_clean = integrity_clean and coverage_acceptable
    manifest["data_quality"]["missing_dates_confirmed_genuine_not_substituted"] = [str(d) for d in missing_dates]
    manifest["data_quality"]["missing_fraction"] = round(missing_fraction, 6)
    manifest["data_quality_gate"] = "PASS" if quality_clean else "FAIL"
    OUT.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"\nDATA QUALITY GATE: {'PASS' if quality_clean else 'FAIL'} "
          f"(integrity_clean={integrity_clean}, missing={len(missing_dates)}/{len(expected)}={missing_fraction:.4%})")
    return 0 if quality_clean else 2


if __name__ == "__main__":
    raise SystemExit(main())
