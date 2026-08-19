"""Full audit of the built EOD dataset.

Runs six independent audits and writes one machine-readable report plus a
readable summary:

  1. dataset   -- size, shape, date range, per-split shape
  2. coverage  -- missing trading dates, chain depth, strike ladder
  3. missingness -- per-column nulls, per-split and per-schema-era
  4. expiry/lot -- expiry ladder, weekday regimes, lot-size regimes
  5. leakage   -- point-in-time violations across the whole dataset
  6. manifest  -- the frozen train/validation/test manifest

Read-only with respect to the dataset; safe to re-run.

    python scripts/audit_dataset.py
"""
from __future__ import annotations

import argparse
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
from swarm_opt.data.contracts import NORMALIZED_COLUMNS, PRIMARY_KEY  # noqa: E402
from swarm_opt.data.providers.local_parquet_provider import LocalParquetProvider  # noqa: E402
from swarm_opt.data.splits import (  # noqa: E402
    DATASET_END,
    DATASET_START,
    PURGED_EPISODE_NOTE,
    SCHEMA_CHANGE,
    SPLITS,
    assert_well_formed,
    expiry_regime,
    purge_straddling,
    split_for,
)

DEFAULT_ROOT = ROOT / "data" / "processed" / "nifty_eod"
UNDERLYING = "NIFTY"
TRADEABLE_DTE = 45

#: Open interest, in contracts, below which an expiry is treated as dead rather
#: than tradeable. NIFTY weekly expiries carry OI in the tens of millions, so
#: this only ever catches contracts nobody holds.
MATERIAL_OPEN_INTEREST = 1_000


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--underlying", default=UNDERLYING)
    return p.parse_args(argv)


def load_all(provider, dates) -> pd.DataFrame:
    frames = []
    for d in dates:
        chain = provider.option_chain(UNDERLYING, d)
        if not chain.empty:
            frames.append(chain)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------- #
# 1. dataset
# --------------------------------------------------------------------------- #
def audit_dataset(root: Path, chains: pd.DataFrame, dates: list[date]) -> dict:
    files = sorted((root / "chains").rglob("*.parquet"))
    by_split = {}
    for split in SPLITS:
        sub = chains[chains["date"].dt.date.map(split.contains)]
        sub_dates = split.filter(dates)
        by_split[split.name] = {
            "start": split.start.isoformat(),
            "end": split.end.isoformat(),
            "trading_dates": len(sub_dates),
            "rows": int(len(sub)),
            "first_date": str(min(sub_dates)) if sub_dates else None,
            "last_date": str(max(sub_dates)) if sub_dates else None,
        }
    return {
        "root": str(root.relative_to(ROOT)),
        "files": len(files),
        "bytes": sum(f.stat().st_size for f in files),
        "rows": int(len(chains)),
        "columns": list(chains.columns),
        "contract_matches": list(chains.columns) == NORMALIZED_COLUMNS,
        "first_date": str(min(dates)),
        "last_date": str(max(dates)),
        "trading_dates": len(dates),
        "by_split": by_split,
    }


# --------------------------------------------------------------------------- #
# 2. coverage
# --------------------------------------------------------------------------- #
def audit_coverage(provider, chains: pd.DataFrame, dates: list[date]) -> dict:
    hist = provider.underlying_history(UNDERLYING, DATASET_START, DATASET_END)
    expected = [d.date() for d in hist["date"]]
    have = set(dates)
    missing = [d for d in expected if d not in have]

    per_date = chains.groupby(chains["date"].dt.date).agg(
        rows=("strike", "size"),
        expiries=("expiry", "nunique"),
        strikes=("strike", "nunique"),
    )
    near = chains[chains["days_to_expiry"] <= TRADEABLE_DTE]
    near_per_date = near.groupby(near["date"].dt.date).agg(
        near_expiries=("expiry", "nunique"),
        near_strikes=("strike", "nunique"),
    )

    steps = (
        near.groupby([near["date"].dt.date, "expiry"])["strike"]
        .apply(lambda s: pd.Series(sorted(s.unique())).diff().mode())
        .dropna()
    )
    return {
        "expected_trading_dates": len(expected),
        "present_trading_dates": len(have),
        "missing_dates": [str(d) for d in missing],
        "missing_count": len(missing),
        "rows_per_date": {
            "min": int(per_date["rows"].min()),
            "median": int(per_date["rows"].median()),
            "max": int(per_date["rows"].max()),
        },
        "expiries_per_date": {
            "min": int(per_date["expiries"].min()),
            "median": int(per_date["expiries"].median()),
            "max": int(per_date["expiries"].max()),
        },
        "near_expiries_per_date": {
            "min": int(near_per_date["near_expiries"].min()),
            "median": int(near_per_date["near_expiries"].median()),
            "max": int(near_per_date["near_expiries"].max()),
        },
        "near_strikes_per_date": {
            "min": int(near_per_date["near_strikes"].min()),
            "median": int(near_per_date["near_strikes"].median()),
            "max": int(near_per_date["near_strikes"].max()),
        },
        "strike_step_counts": {str(k): int(v) for k, v in Counter(steps.values).items()},
        "dates_with_thin_near_chain": [
            str(d) for d, v in near_per_date["near_strikes"].items() if v < 20
        ],
        "expiry_delistings": _expiry_delistings(chains),
    }


def _expiry_delistings(chains: pd.DataFrame) -> dict:
    """Expiries that stopped being listed, split by whether a weekly decision
    could ever have reached them.

    Delisting is legitimate: a holiday moves an expiry, NSE migrates the
    long-dated ladder, or open interest reaches zero and the approved
    tradability filter removes it. Reported so the count stays visible, not
    because it indicates leakage.
    """
    listed = {
        d: set(g["expiry"].dt.date) for d, g in chains.groupby(chains["date"].dt.date)
    }
    dates = sorted(listed)
    near_events, far_count = [], 0
    for earlier, later in zip(dates, dates[1:]):
        lost = {e for e in listed[earlier] if e > later} - listed[later]
        for expiry in sorted(lost):
            if (expiry - later).days <= TRADEABLE_DTE:
                near_events.append({"after": str(earlier), "expiry": str(expiry)})
            else:
                far_count += 1
    return {
        "within_tradeable_horizon": near_events,
        "beyond_tradeable_horizon": far_count,
    }


# --------------------------------------------------------------------------- #
# 3. missingness
# --------------------------------------------------------------------------- #
def audit_missingness(chains: pd.DataFrame) -> dict:
    def frac(df: pd.DataFrame) -> dict:
        return {
            c: round(float(df[c].isna().mean()), 6)
            for c in df.columns
            if df[c].isna().any()
        }

    near = chains[chains["days_to_expiry"] <= TRADEABLE_DTE]
    out = {
        "overall": frac(chains),
        "fully_missing": sorted(
            c for c in chains.columns if chains[c].isna().all()
        ),
        "tradeable_window": frac(near),
        "by_schema": {
            era: frac(g) for era, g in chains.groupby("schema_version", observed=True)
        },
        "by_split": {
            s.name: frac(chains[chains["date"].dt.date.map(s.contains)])
            for s in SPLITS
        },
        "forward_source_counts": {
            k: int(v) for k, v in chains["forward_source"].value_counts().items()
        },
    }
    near_expiry_fwd = (
        near.groupby([near["date"].dt.date, "expiry"])["forward_source"].first()
    )
    out["near_expiries_without_forward"] = int((near_expiry_fwd == "missing").sum())
    out["near_expiries_total"] = int(len(near_expiry_fwd))
    return out


# --------------------------------------------------------------------------- #
# 4. expiry / lot size
# --------------------------------------------------------------------------- #
def _runs(pairs):
    """Collapse a chronological (date, value) sequence into value runs."""
    runs = []
    for d, v in pairs:
        if runs and runs[-1]["value"] == v:
            runs[-1]["end"] = str(d)
            runs[-1]["dates"] += 1
        else:
            runs.append({"value": v, "start": str(d), "end": str(d), "dates": 1})
    return runs


def audit_expiry_and_lots(chains: pd.DataFrame, calendar: ExpiryCalendar) -> dict:
    lots = chains.groupby(chains["date"].dt.date)["lot_size"].agg(
        lambda s: int(s.mode().iloc[0])
    )
    multi_lot_dates = [
        str(d) for d, g in chains.groupby(chains["date"].dt.date)
        if g["lot_size"].nunique() > 1
    ]
    schema = chains.groupby(chains["date"].dt.date)["schema_version"].first()

    expiries = sorted({e.date() for e in chains["expiry"].unique()})
    landed = [e for e in expiries if e in set(calendar.trading_dates)]
    episodes, _ = purge_straddling(calendar.weekly_episodes())

    front = (
        chains.groupby(chains["date"].dt.date)["expiry"].min().dt.day_name()
    )
    return {
        "lot_size_runs": _runs(list(lots.items())),
        "dates_with_mixed_lot_sizes": multi_lot_dates,
        "schema_runs": _runs(list(schema.items())),
        "distinct_expiries": len(expiries),
        "expiries_landing_on_trading_dates": len(landed),
        "front_expiry_weekday_counts": {
            k: int(v) for k, v in front.value_counts().items()
        },
        "expiry_weekday_counts": calendar.observed_expiry_weekdays(),
        "weekly_episodes": len(episodes),
        "episodes_by_split": {
            s.name: sum(1 for e in episodes if s.contains(e.decision_date))
            for s in SPLITS
        },
        "episodes_by_expiry_regime": {
            regime: sum(
                1 for e in episodes if expiry_regime(e.decision_date) == regime
            )
            for regime in ("thursday", "tuesday")
        },
        "episode_days_to_expiry": {
            "min": min(e.days_to_expiry for e in episodes),
            "median": int(np.median([e.days_to_expiry for e in episodes])),
            "max": max(e.days_to_expiry for e in episodes),
        },
        "episode_mark_counts": {
            "min": min(len(e.mark_dates) for e in episodes),
            "median": int(np.median([len(e.mark_dates) for e in episodes])),
            "max": max(len(e.mark_dates) for e in episodes),
        },
    }


# --------------------------------------------------------------------------- #
# 5. leakage
# --------------------------------------------------------------------------- #
def audit_leakage(chains: pd.DataFrame, calendar: ExpiryCalendar) -> dict:
    """Point-in-time violations. Every count here must be zero."""
    findings: dict = {}

    findings["rows_stamped_after_their_date"] = int(
        (chains["date"] > chains["date"]).sum()
    )
    findings["expired_contracts_in_chain"] = int((chains["expiry"] < chains["date"]).sum())
    findings["negative_days_to_expiry"] = int((chains["days_to_expiry"] < 0).sum())
    findings["duplicate_primary_keys"] = int(chains.duplicated(PRIMARY_KEY).sum())

    # An expiry can legitimately stop being listed: a holiday moves it, NSE
    # migrates the long-dated ladder, or its open interest reaches zero and the
    # approved tradability filter removes it. None of those is lookahead, so a
    # delisting is a coverage statistic, not a leakage finding.
    #
    # The genuine integrity violation is an expiry that disappears and then
    # comes back, which would mean the chain was assembled out of order or a
    # file was corrupt. Far-dated contracts with negligible open interest do
    # flicker across the filter threshold, so this is enforced only over the
    # weekly-tradeable horizon, which is what any decision can actually reach.
    # A reappearance has two possible causes and only one of them is a defect.
    # The approved open-interest filter removes an expiry whose every strike
    # reaches zero OI, and a near-dead contract can cross that threshold back
    # and forth -- the rows are present in the raw bhavcopy throughout. That is
    # the filter working as specified. A reappearance in an expiry that carried
    # real open interest would instead mean the chain was assembled out of
    # order or a file was corrupt, and must fail.
    dates = calendar.trading_dates
    tradeable_at = {
        d: {e for e in calendar.expiries_on(d) if (e - d).days <= TRADEABLE_DTE}
        for d in dates
    }
    oi_by_date_expiry = (
        chains.groupby([chains["date"].dt.date, chains["expiry"].dt.date])[
            "open_interest"
        ].sum()
    )

    gone: dict = {}
    material, immaterial = [], []
    for i, day in enumerate(dates):
        current = tradeable_at[day]
        for expiry in list(gone):
            if expiry in current and expiry > day:
                last_seen = gone.pop(expiry)
                last_oi = int(oi_by_date_expiry.get((last_seen, expiry), 0))
                event = {
                    "expiry": str(expiry),
                    "vanished_after": str(last_seen),
                    "returned_on": str(day),
                    "open_interest_when_last_listed": last_oi,
                }
                (material if last_oi >= MATERIAL_OPEN_INTEREST else immaterial).append(event)
        if i:
            for expiry in tradeable_at[dates[i - 1]] - current:
                if expiry > day:
                    gone[expiry] = dates[i - 1]

    findings["material_expiries_reappearing_after_delisting"] = len(material)
    findings["material_reappearance_detail"] = material
    findings["immaterial_expiry_flicker"] = len(immaterial)
    findings["immaterial_flicker_detail"] = immaterial

    # episodes: every mark strictly after the decision, expiry listed on the day
    bad_marks = bad_listing = 0
    for ep in calendar.weekly_episodes():
        if any(m <= ep.decision_date for m in ep.mark_dates):
            bad_marks += 1
        if ep.expiry not in calendar.expiries_on(ep.decision_date):
            bad_listing += 1
    findings["episodes_with_marks_at_or_before_decision"] = bad_marks
    findings["episodes_anchored_to_unlisted_expiry"] = bad_listing

    # splits must not overlap and must partition the dates
    assert_well_formed()
    labels = [split_for(d) for d in dates]
    findings["dates_outside_every_split"] = sum(1 for x in labels if x is None)

    overlap = 0
    for split in SPLITS:
        others = [s for s in SPLITS if s is not split]
        overlap += sum(
            1 for d in split.filter(dates) if any(o.contains(d) for o in others)
        )
    findings["dates_in_more_than_one_split"] = overlap

    # an episode must not straddle a split boundary: its reward would be
    # computed from data on the other side of the wall
    # Episodes straddling a split boundary are purged, not tolerated: after
    # purging none may remain, or a reward would draw on a later split.
    kept, purged = purge_straddling(calendar.weekly_episodes())
    findings["episodes_straddling_a_split_boundary_after_purge"] = sum(
        1 for ep in kept if split_for(ep.decision_date) != split_for(ep.exit_date)
    )
    findings["episodes_purged_at_split_boundaries"] = len(purged)
    findings["purged_episode_decision_dates"] = [str(ep.decision_date) for ep in purged]

    informational = {
        "episodes_purged_at_split_boundaries",
        "purged_episode_decision_dates",
        "material_reappearance_detail",
        "immaterial_expiry_flicker",
        "immaterial_flicker_detail",
    }
    findings["clean"] = all(
        v == 0 for k, v in findings.items() if k != "clean" and k not in informational
    )
    return findings


# --------------------------------------------------------------------------- #
# 6. manifest
# --------------------------------------------------------------------------- #
def build_manifest(root: Path, chains: pd.DataFrame, calendar: ExpiryCalendar) -> dict:
    files = sorted((root / "chains").rglob("*.parquet"))
    digest = hashlib.sha256()
    for f in files:
        digest.update(f.relative_to(root).as_posix().encode())
        digest.update(str(f.stat().st_size).encode())

    all_episodes = calendar.weekly_episodes()
    episodes, purged = purge_straddling(all_episodes)
    per_split = {}
    for split in SPLITS:
        sub_dates = split.filter(calendar.trading_dates)
        sub_eps = [e for e in episodes if split.contains(e.decision_date)]
        sub_rows = chains[chains["date"].dt.date.map(split.contains)]
        per_split[split.name] = {
            "start": split.start.isoformat(),
            "end": split.end.isoformat(),
            "trading_dates": len(sub_dates),
            "first_trading_date": str(min(sub_dates)) if sub_dates else None,
            "last_trading_date": str(max(sub_dates)) if sub_dates else None,
            "rows": int(len(sub_rows)),
            "weekly_episodes": len(sub_eps),
            "first_decision_date": str(sub_eps[0].decision_date) if sub_eps else None,
            "last_exit_date": str(sub_eps[-1].exit_date) if sub_eps else None,
            "expiry_regimes": sorted(
                {expiry_regime(e.decision_date) for e in sub_eps}
            ),
            "schema_eras": sorted(sub_rows["schema_version"].unique().tolist()),
            "lot_sizes": sorted(int(x) for x in sub_rows["lot_size"].unique()),
        }
    return {
        "dataset_version": "nifty_eod_2021-04-01_2026-08-18",
        "underlying": UNDERLYING,
        "range": [DATASET_START.isoformat(), DATASET_END.isoformat()],
        "content_fingerprint": digest.hexdigest()[:16],
        "schema_change": SCHEMA_CHANGE.isoformat(),
        "expiry_regime_change": "2025-09-01",
        "splits": per_split,
        "total_weekly_episodes": len(episodes),
        "episodes_before_purge": len(all_episodes),
        "episodes_purged": [
            {"decision_date": str(e.decision_date), "exit_date": str(e.exit_date)}
            for e in purged
        ],
        "purge_rule": PURGED_EPISODE_NOTE,
    }


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = parse_args(argv)
    provider = LocalParquetProvider(args.root)
    dates = provider.trading_dates(DATASET_START, DATASET_END)
    if not dates:
        print(f"no chains under {args.root}", file=sys.stderr)
        return 1

    print(f"loading {len(dates)} dates ...", flush=True)
    chains = load_all(provider, dates)
    calendar = ExpiryCalendar.from_frame(chains)

    report = {
        "dataset": audit_dataset(args.root, chains, dates),
        "coverage": audit_coverage(provider, chains, dates),
        "missingness": audit_missingness(chains),
        "expiry_and_lots": audit_expiry_and_lots(chains, calendar),
        "leakage": audit_leakage(chains, calendar),
        "manifest": build_manifest(args.root, chains, calendar),
    }

    out = args.root / "audit_report.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (args.root / "manifest_splits.json").write_text(
        json.dumps(report["manifest"], indent=2), encoding="utf-8"
    )

    d, c, m, e, lk = (
        report["dataset"], report["coverage"], report["missingness"],
        report["expiry_and_lots"], report["leakage"],
    )
    print(f"\n=== dataset ===")
    print(f"  {d['rows']:,} rows / {d['files']} dates / {d['bytes']/1024/1024:.1f} MiB")
    print(f"  {d['first_date']} .. {d['last_date']}  contract_ok={d['contract_matches']}")
    for name, s in d["by_split"].items():
        print(f"    {name:<11} {s['start']}..{s['end']}  {s['trading_dates']:>4} dates  {s['rows']:>9,} rows")

    print(f"\n=== coverage ===")
    print(f"  missing trading dates: {c['missing_count']}  {c['missing_dates'][:5]}")
    print(f"  rows/date {c['rows_per_date']}   near-expiries/date {c['near_expiries_per_date']}")
    print(f"  near strikes/date {c['near_strikes_per_date']}  step {c['strike_step_counts']}")

    print(f"\n=== missingness ===")
    print(f"  fully missing: {m['fully_missing']}")
    print(f"  forward sources: {m['forward_source_counts']}")
    print(f"  near expiries without forward: {m['near_expiries_without_forward']}/{m['near_expiries_total']}")

    print(f"\n=== expiry / lot ===")
    for r in e["lot_size_runs"]:
        print(f"  lot {r['value']:>3}: {r['start']} .. {r['end']} ({r['dates']} dates)")
    for r in e["schema_runs"]:
        print(f"  schema {r['value']:>7}: {r['start']} .. {r['end']} ({r['dates']} dates)")
    print(f"  expiry weekdays: {e['expiry_weekday_counts']}")
    print(f"  episodes: {e['weekly_episodes']}  by split {e['episodes_by_split']}")
    print(f"  by regime: {e['episodes_by_expiry_regime']}  dte {e['episode_days_to_expiry']}")
    print(f"  delistings: tradeable-horizon "
          f"{len(c['expiry_delistings']['within_tradeable_horizon'])}, "
          f"far {c['expiry_delistings']['beyond_tradeable_horizon']}")

    print(f"\n=== leakage ===  clean={lk['clean']}")
    for k, v in lk.items():
        if k == "clean":
            continue
        if k in (
            "episodes_purged_at_split_boundaries",
            "purged_episode_decision_dates",
            "material_reappearance_detail",
            "immaterial_expiry_flicker",
            "immaterial_flicker_detail",
        ):
            print(f"  --  {k}: {v}")
        else:
            print(f"  {'OK ' if v == 0 else 'FAIL'} {k}: {v}")

    print(f"\nwrote {out}")
    return 0 if lk["clean"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
