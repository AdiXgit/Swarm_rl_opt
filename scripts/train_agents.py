"""Train the Direction, Volatility and Regime agents on the frozen dataset.

Verifies the frozen manifest fingerprint before touching any data, builds
point-in-time features/targets, runs the temporal leakage audit, fits all
three agents on TRAIN only, evaluates on TRAIN and VALIDATION, and writes
artifacts + a machine-readable report. TEST is never read by this script;
use `scripts/evaluate_on_test.py` only after every modeling choice is
frozen.

    python scripts/train_agents.py
    python scripts/train_agents.py --root data/processed/nifty_eod --seed 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from swarm_opt.agents.direction import DirectionAgent  # noqa: E402
from swarm_opt.agents.regime import RegimeAgent  # noqa: E402
from swarm_opt.agents.volatility import VolatilityAgent  # noqa: E402
from swarm_opt.data.providers.local_parquet_provider import LocalParquetProvider  # noqa: E402
from swarm_opt.data.splits import DATASET_END, DATASET_START, TRAIN, split_for  # noqa: E402
from swarm_opt.evaluation import evaluate_predictions  # noqa: E402
from swarm_opt.features.audit import (  # noqa: E402
    assert_duplicate_free,
    assert_feature_label_alignment,
    assert_no_cross_split_labels,
    assert_no_future_features,
    build_temporal_audit,
)
from swarm_opt.features.manifest import FeatureManifest  # noqa: E402
from swarm_opt.features.pointintime import FEATURE_NAMES, FEATURE_SCHEMA_VERSION  # noqa: E402
from swarm_opt.features.targets import TARGET_SCHEMA_VERSION  # noqa: E402
from swarm_opt.pipeline import build_dataset, make_xy, split_targets  # noqa: E402

DEFAULT_ROOT = ROOT / "data" / "processed" / "nifty_eod"
DEFAULT_ATM_IV = ROOT / "data" / "features" / "atm_iv_history.parquet"
DEFAULT_ARTIFACTS = ROOT / "data" / "models"
DEFAULT_REPORT = ROOT / "data" / "models" / "phase2_training_report.json"

AGENTS = {
    "direction": (DirectionAgent, "direction_label"),
    "volatility": (VolatilityAgent, "volatility_label"),
    "regime": (RegimeAgent, "regime_label"),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--atm-iv", type=Path, default=DEFAULT_ATM_IV)
    p.add_argument("--underlying", default="NIFTY")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return p.parse_args(argv)


def verify_manifest(root: Path) -> dict:
    manifest_path = root / "manifest_splits.json"
    if not manifest_path.exists():
        raise SystemExit(f"frozen manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    expected_fp = "653ccb70c795d86f"
    if manifest["content_fingerprint"] != expected_fp:
        raise SystemExit(
            f"dataset fingerprint mismatch: manifest has {manifest['content_fingerprint']!r}, "
            f"Phase 2 was frozen against {expected_fp!r}. Stopping before any training."
        )
    expected_counts = {"train": 929, "validation": 165, "test": 239}
    for name, expected in expected_counts.items():
        actual = manifest["splits"][name]["trading_dates"]
        if actual != expected:
            raise SystemExit(
                f"{name} split has {actual} trading dates, frozen manifest expects {expected}"
            )
    print(f"manifest verified: fingerprint={manifest['content_fingerprint']} OK")
    return manifest


def main(argv=None) -> int:
    args = parse_args(argv)
    verify_manifest(args.root)

    provider = LocalParquetProvider(args.root)
    atm_iv_hist = pd.read_parquet(args.atm_iv) if args.atm_iv.exists() else None
    if atm_iv_hist is None:
        print(f"no ATM IV cache at {args.atm_iv}; IV features will be NaN", file=sys.stderr)

    print("building point-in-time features and targets ...", flush=True)
    dataset = build_dataset(provider, args.underlying, DATASET_START, DATASET_END, atm_iv_hist)
    report = dataset.feature_build_report
    print(f"  features: {report}")
    print(f"  episodes kept={len(dataset.episodes_kept)} purged={len(dataset.episodes_purged)}")
    print(f"  thresholds: {dataset.thresholds}")

    by_split = split_targets(dataset)
    for name, df in by_split.items():
        print(f"  {name}: {len(df)} episodes, complete={df['complete'].sum()}")

    # -- leakage audit, run before any model sees the data ------------------ #
    assert_no_future_features(dataset.features)
    assert_feature_label_alignment(dataset.features, dataset.targets)
    assert_no_cross_split_labels(dataset.targets)
    assert_duplicate_free(dataset.features, "date")
    assert_duplicate_free(dataset.targets, "decision_date")
    print("leakage audit: PASS")

    feature_manifest = FeatureManifest(
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_names=FEATURE_NAMES,
        max_lookback_days=20,
        target_schema_version=TARGET_SCHEMA_VERSION,
    )

    args.artifacts.mkdir(parents=True, exist_ok=True)
    feature_manifest.save(args.artifacts / "feature_manifest.json")

    train_range = (TRAIN.start, TRAIN.end)
    full_report: dict = {
        "dataset_fingerprint_verified": True,
        "thresholds": {
            "bullish_return": dataset.thresholds.bullish_return,
            "bearish_return": dataset.thresholds.bearish_return,
            "vol_expand_ratio": dataset.thresholds.vol_expand_ratio,
            "vol_contract_ratio": dataset.thresholds.vol_contract_ratio,
            "regime_high_vol_ratio": dataset.thresholds.regime_high_vol_ratio,
            "regime_low_vol_ratio": dataset.thresholds.regime_low_vol_ratio,
        },
        "feature_build_report": {
            "rows_in": report.rows_in,
            "rows_with_full_lookback": report.rows_with_full_lookback,
            "rows_dropped_insufficient_lookback": report.rows_dropped_insufficient_lookback,
            "atm_iv_coverage": report.atm_iv_coverage,
        },
        "episodes_purged": [
            {"decision_date": str(e.decision_date), "exit_date": str(e.exit_date)}
            for e in dataset.episodes_purged
        ],
        "agents": {},
    }

    for name, (agent_cls, label_col) in AGENTS.items():
        print(f"\n=== {name} ===", flush=True)
        train_df = by_split["train"]
        val_df = by_split["validation"]

        X_train, y_train, ts_train = make_xy(dataset, train_df, label_col)
        X_val, y_val, ts_val = make_xy(dataset, val_df, label_col)

        rows_removed_incomplete_train = len(train_df) - len(X_train)
        rows_removed_incomplete_val = len(val_df) - len(X_val)

        agent = agent_cls(seed=args.seed)
        agent.fit(X_train, y_train, FEATURE_SCHEMA_VERSION, (str(train_range[0]), str(train_range[1])))
        agent.save(args.artifacts / name)
        print(f"  trained on {len(X_train)} rows, saved to {args.artifacts / name}")
        if agent._missing_classes_in_fit:
            print(f"  WARNING: TRAIN fold missing class(es): {agent._missing_classes_in_fit}")

        train_preds = agent.predict(X_train, ts_train)
        val_preds = agent.predict(X_val, ts_val)

        train_proba = pd.DataFrame([p.probabilities for p in train_preds])[list(agent.CLASSES)].to_numpy()
        val_proba = pd.DataFrame([p.probabilities for p in val_preds])[list(agent.CLASSES)].to_numpy()
        train_pred_cls = [p.predicted_class for p in train_preds]
        val_pred_cls = [p.predicted_class for p in val_preds]

        train_eval = evaluate_predictions("train", agent.CLASSES, y_train, train_pred_cls, train_proba)
        val_eval = evaluate_predictions("validation", agent.CLASSES, y_val, val_pred_cls, val_proba)
        print(f"  TRAIN acc={train_eval.accuracy:.3f} (majority={train_eval.majority_baseline_accuracy:.3f})")
        print(f"  VAL   acc={val_eval.accuracy:.3f} (majority={val_eval.majority_baseline_accuracy:.3f})")

        audit = build_temporal_audit(
            agent_name=name,
            feature_cutoff="decision_date (EOD close inclusive)",
            label_horizon="episode.decision_date -> episode.exit_date (weekly, calendar-derived)",
            train_range=train_range,
            rows_removed_lookback=int((~dataset.features["has_full_lookback"]).sum()),
            rows_removed_incomplete_target=rows_removed_incomplete_train + rows_removed_incomplete_val,
            rows_removed_purge=len(dataset.episodes_purged),
            split_frames=by_split,
            label_column=label_col,
        )

        full_report["agents"][name] = {
            "target_definition": agent.TARGET_DEFINITION,
            "train_eval": train_eval.to_dict(),
            "validation_eval": val_eval.to_dict(),
            "temporal_audit": audit.to_dict(),
        }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(full_report, indent=2, default=str))
    print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
