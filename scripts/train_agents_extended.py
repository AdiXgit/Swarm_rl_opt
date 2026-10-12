"""Phase 2.5: train Direction/Volatility/Regime on the EXTENDED TRAIN set.

EXTENDED TRAIN = historical extension (2019-02-18..2021-03-31, its own
verified, separately-versioned dataset) + the original frozen TRAIN
(2021-04-01..2024-12-31). VALIDATION (2025-01-01..2025-08-31) and TEST
(2025-09-01..2026-08-18) are the ORIGINAL, unchanged dates from the frozen
dataset -- never rebuilt, never touched beyond being read for evaluation
(VALIDATION) or not read at all (TEST, never accessed by this script).

Uses the Phase 2.5 `pointintime_v2` feature schema (17 features, `dist_sma20`
dropped -- see backend/swarm_opt/features/pointintime.py). Target rule and
class order are unchanged from Phase 2 (`targets_v1`); only the threshold
VALUE is refit, because it is defined as "0.5*std(TRAIN forward returns)"
and TRAIN itself is now larger -- refitting it is the existing rule applied
to new data, not a new rule, and it is fit and frozen before VALIDATION is
evaluated, exactly as the original training script already does.

Writes ONLY to new paths: data/models/extended_2019_2021/{direction,
volatility,regime}/, data/models/extended_2019_2021/feature_manifest.json,
data/models/extended_2019_2021/training_report.json. Never touches
data/models/{direction,volatility,regime}/ or phase2_training_report.json.

    python scripts/train_agents_extended.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from swarm_opt.agents.direction import DirectionAgent  # noqa: E402
from swarm_opt.agents.regime import RegimeAgent  # noqa: E402
from swarm_opt.agents.volatility import VolatilityAgent  # noqa: E402
from swarm_opt.data.calendar import ExpiryCalendar  # noqa: E402
from swarm_opt.data.providers.combined_provider import CombinedParquetProvider  # noqa: E402
from swarm_opt.data.providers.local_parquet_provider import LocalParquetProvider  # noqa: E402
from swarm_opt.data.splits import DATASET_END, TEST, TRAIN, VALIDATION  # noqa: E402
from swarm_opt.evaluation import evaluate_predictions  # noqa: E402
from swarm_opt.features.audit import (  # noqa: E402
    assert_duplicate_free,
    assert_feature_label_alignment,
    assert_no_future_features,
)
from swarm_opt.features.manifest import FeatureManifest  # noqa: E402
from swarm_opt.features.pointintime import FEATURE_NAMES_V2, FEATURE_SCHEMA_VERSION_V2, build_market_features  # noqa: E402
from swarm_opt.features.targets import (  # noqa: E402
    TARGET_SCHEMA_VERSION,
    TargetThresholds,
    build_targets,
    fit_thresholds_on_train,
)

ORIGINAL_ROOT = ROOT / "data" / "processed" / "nifty_eod"
EXTENSION_ROOT = ROOT / "data" / "processed" / "nifty_eod_ext_2019_2021"
ORIGINAL_ATM_IV = ROOT / "data" / "features" / "atm_iv_history.parquet"
EXTENSION_ATM_IV = ROOT / "data" / "features" / "atm_iv_history_ext_2019_2021.parquet"
OUT_DIR = ROOT / "data" / "models" / "extended_2019_2021"

EXTENDED_TRAIN_START = date(2019, 2, 18)
EXTENDED_TRAIN_END = TRAIN.end  # 2024-12-31, unchanged

# swarm_opt.data.splits.split_for/purge_straddling are frozen to the
# ORIGINAL TRAIN start (2021-04-01) and must stay that way for the original
# dataset's own purposes. Reusing them here would resolve every extension
# decision date (before 2021-04-01) to split_for() = None and purge the
# entire extension -- a real bug caught before this script was ever run
# live. This is a separate, parallel split definition for the EXTENDED
# experiment only; it does not alter swarm_opt.data.splits in any way.
EXTENDED_TRAIN_SPLIT = (EXTENDED_TRAIN_START, EXTENDED_TRAIN_END)
EXTENDED_SPLITS = [("extended_train", EXTENDED_TRAIN_START, EXTENDED_TRAIN_END),
                    ("validation", VALIDATION.start, VALIDATION.end),
                    ("test", TEST.start, TEST.end)]


def extended_split_for(day: date) -> str | None:
    for name, start, end in EXTENDED_SPLITS:
        if start <= day <= end:
            return name
    return None


def extended_purge_straddling(episodes):
    kept, purged = [], []
    for ep in episodes:
        start, end = extended_split_for(ep.decision_date), extended_split_for(ep.exit_date)
        (kept if (start is not None and start == end) else purged).append(ep)
    return kept, purged


def persistence_predictions(train_targets: pd.DataFrame, val_targets: pd.DataFrame, label_col: str):
    """Predict each VALIDATION row's label as the immediately preceding
    chronological episode's true label (the first VALIDATION row is
    predicted from EXTENDED TRAIN's last label -- strictly historical, no
    future access). Mirrors scripts/run_phase2_2_experiments.py's
    persistence_predictions, recomputed here since EXTENDED TRAIN's
    labels differ from the original TRAIN's (refit thresholds)."""
    def usable(df):
        return df[df["complete"] & (df[label_col] != "UNKNOWN")][["decision_date", label_col]]

    combined = pd.concat([usable(train_targets), usable(val_targets)], ignore_index=True)
    combined = combined.sort_values("decision_date").reset_index(drop=True)
    pred = combined[label_col].shift(1)
    is_val = combined["decision_date"].isin(set(usable(val_targets)["decision_date"]))
    mask = is_val & pred.notna()
    return combined.loc[mask, label_col].reset_index(drop=True), pred.loc[mask].reset_index(drop=True)


def extended_assert_no_cross_split_labels(targets: pd.DataFrame) -> None:
    """Same check as `features.audit.assert_no_cross_split_labels`, against
    the EXTENDED split boundaries instead of the frozen original ones (which
    would otherwise flag every pre-2021-04-01 extension label as crossing a
    boundary, since split_for() returns None for dates before the original
    TRAIN start)."""
    from swarm_opt.features.audit import LeakageError

    bad = [
        row.decision_date for row in targets.itertuples()
        if extended_split_for(row.decision_date) is None
        or extended_split_for(row.decision_date) != extended_split_for(row.exit_date)
    ]
    if bad:
        raise LeakageError(f"{len(bad)} labels cross an EXTENDED split boundary: {bad[:5]}")

AGENTS = {
    "direction": (DirectionAgent, "direction_label"),
    "volatility": (VolatilityAgent, "volatility_label"),
    "regime": (RegimeAgent, "regime_label"),
}


def sha256_dir(root: Path) -> str:
    digest = hashlib.sha256()
    for f in sorted(root.rglob("*.parquet")):
        digest.update(f.relative_to(root).as_posix().encode())
        digest.update(str(f.stat().st_size).encode())
    return digest.hexdigest()[:16]


def verify_original_untouched() -> None:
    manifest = json.loads((ORIGINAL_ROOT / "manifest_splits.json").read_text())
    expected_fp = "653ccb70c795d86f"
    if manifest["content_fingerprint"] != expected_fp:
        raise SystemExit(f"ORIGINAL dataset fingerprint changed: {manifest['content_fingerprint']} != {expected_fp}")
    print(f"original frozen fingerprint verified unchanged: {expected_fp}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    verify_original_untouched()

    if not EXTENSION_ROOT.exists():
        raise SystemExit(f"extension dataset not found at {EXTENSION_ROOT}; run scripts/build_dataset.py first")
    if not EXTENSION_ATM_IV.exists():
        raise SystemExit(f"extension ATM IV cache not found at {EXTENSION_ATM_IV}; run scripts/build_market_features.py first")

    ext_provider = LocalParquetProvider(EXTENSION_ROOT)
    orig_provider = LocalParquetProvider(ORIGINAL_ROOT)
    combined = CombinedParquetProvider([ext_provider, orig_provider])

    print("building combined underlying history and point-in-time features ...", flush=True)
    hist = combined.underlying_history("NIFTY", EXTENDED_TRAIN_START, DATASET_END)
    atm_iv = pd.concat(
        [pd.read_parquet(EXTENSION_ATM_IV), pd.read_parquet(ORIGINAL_ATM_IV)], ignore_index=True
    ).sort_values("date").reset_index(drop=True)
    features, report = build_market_features(hist, atm_iv_hist=atm_iv, feature_names=FEATURE_NAMES_V2)
    print(f"  {report}")

    dates = combined.trading_dates(EXTENDED_TRAIN_START, DATASET_END)
    calendar = ExpiryCalendar.from_provider(combined, "NIFTY", dates)
    all_episodes = calendar.weekly_episodes()
    kept, purged = extended_purge_straddling(all_episodes)
    print(f"  weekly episodes: {len(all_episodes)} before purge, {len(kept)} kept, {len(purged)} purged at split walls")

    baseline_vol = features.set_index(features["date"].dt.date)["realized_vol_20"]
    extended_train_episodes = [
        e for e in kept if EXTENDED_TRAIN_START <= e.decision_date <= EXTENDED_TRAIN_END
    ]
    if not extended_train_episodes:
        raise SystemExit("no EXTENDED TRAIN episodes found")

    provisional = build_targets(
        extended_train_episodes, hist, baseline_vol, TargetThresholds(bullish_return=1.0, bearish_return=1.0),
    )
    thresholds = fit_thresholds_on_train(provisional["forward_return"])
    print(f"  EXTENDED TRAIN thresholds (refit on {len(extended_train_episodes)} episodes): {thresholds}")

    val_episodes = [e for e in kept if VALIDATION.contains(e.decision_date)]
    targets = build_targets(extended_train_episodes + val_episodes, hist, baseline_vol, thresholds)
    extended_train_targets = targets[targets["decision_date"].isin({e.decision_date for e in extended_train_episodes})]
    val_targets = targets[targets["decision_date"].isin({e.decision_date for e in val_episodes})]

    assert_no_future_features(features)
    assert_feature_label_alignment(features, targets)
    extended_assert_no_cross_split_labels(targets)
    assert_duplicate_free(features, "date")
    assert_duplicate_free(targets, "decision_date")
    print("leakage audit: PASS")

    feats_by_date = features.set_index(features["date"].dt.date)
    lookback_ok = feats_by_date["has_full_lookback"]

    def make_xy(target_df: pd.DataFrame, label_col: str):
        target_ok = target_df["complete"] & (target_df[label_col] != "UNKNOWN")
        candidates = target_df[target_ok]
        lb = lookback_ok.reindex(candidates["decision_date"])
        usable = candidates[lb.fillna(False).to_numpy().astype(bool)]
        X = feats_by_date.loc[usable["decision_date"], FEATURE_NAMES_V2].reset_index(drop=True)
        y = usable[label_col].reset_index(drop=True)
        ts = pd.Series(pd.to_datetime(usable["decision_date"].to_numpy()))
        excluded_target = int((~target_ok).sum())
        excluded_lookback = int(len(candidates) - len(usable))
        return X, y, ts, excluded_target, excluded_lookback

    feature_manifest = FeatureManifest(
        feature_schema_version=FEATURE_SCHEMA_VERSION_V2,
        feature_names=FEATURE_NAMES_V2,
        max_lookback_days=20,
        target_schema_version=TARGET_SCHEMA_VERSION,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feature_manifest.save(OUT_DIR / "feature_manifest.json")

    training_report: dict = {
        "extended_train_range": [EXTENDED_TRAIN_START.isoformat(), EXTENDED_TRAIN_END.isoformat()],
        "validation_range_unchanged": [VALIDATION.start.isoformat(), VALIDATION.end.isoformat()],
        "test_range_untouched": "2025-09-01..2026-08-18 (never read by this script)",
        "feature_schema_version": FEATURE_SCHEMA_VERSION_V2,
        "feature_names": FEATURE_NAMES_V2,
        "target_schema_version": TARGET_SCHEMA_VERSION,
        "thresholds_refit_on_extended_train": {
            "bullish_return": thresholds.bullish_return,
            "bearish_return": thresholds.bearish_return,
            "vol_expand_ratio": thresholds.vol_expand_ratio,
            "vol_contract_ratio": thresholds.vol_contract_ratio,
            "regime_high_vol_ratio": thresholds.regime_high_vol_ratio,
            "regime_low_vol_ratio": thresholds.regime_low_vol_ratio,
        },
        "weekly_episodes_before_purge": len(all_episodes),
        "weekly_episodes_kept_after_purge": len(kept),
        "weekly_episodes_purged": len(purged),
        "extended_train_episodes_total": len(extended_train_episodes),
        "validation_episodes_total": len(val_episodes),
        "extension_dataset_fingerprint": sha256_dir(EXTENSION_ROOT),
        "original_dataset_fingerprint": "653ccb70c795d86f",
        "seed": args.seed,
        "agents": {},
    }

    for name, (agent_cls, label_col) in AGENTS.items():
        print(f"\n=== {name} (extended) ===", flush=True)
        X_train, y_train, ts_train, excl_target_tr, excl_lb_tr = make_xy(extended_train_targets, label_col)
        X_val, y_val, ts_val, excl_target_val, excl_lb_val = make_xy(val_targets, label_col)

        agent = agent_cls(seed=args.seed)
        agent.fit(X_train, y_train, FEATURE_SCHEMA_VERSION_V2, (EXTENDED_TRAIN_START.isoformat(), EXTENDED_TRAIN_END.isoformat()))
        agent.save(OUT_DIR / name)
        print(f"  trained on {len(X_train)} rows (excl target={excl_target_tr} lookback={excl_lb_tr}), saved to {OUT_DIR / name}")
        if agent._missing_classes_in_fit:
            print(f"  WARNING: missing class(es) in EXTENDED TRAIN: {agent._missing_classes_in_fit}")

        train_preds = agent.predict(X_train, ts_train)
        val_preds = agent.predict(X_val, ts_val)
        train_proba = pd.DataFrame([p.probabilities for p in train_preds])[list(agent.CLASSES)].to_numpy()
        val_proba = pd.DataFrame([p.probabilities for p in val_preds])[list(agent.CLASSES)].to_numpy()

        train_eval = evaluate_predictions("train", agent.CLASSES, y_train, [p.predicted_class for p in train_preds], train_proba)
        val_eval = evaluate_predictions("validation", agent.CLASSES, y_val, [p.predicted_class for p in val_preds], val_proba)

        y_true_persist, y_pred_persist = persistence_predictions(extended_train_targets, val_targets, label_col)
        persistence_acc = float((y_true_persist == y_pred_persist).mean()) if len(y_true_persist) else float("nan")

        print(f"  TRAIN n={len(X_train)} acc={train_eval.accuracy:.3f} (majority={train_eval.majority_baseline_accuracy:.3f})")
        print(f"  VAL   n={len(X_val)} acc={val_eval.accuracy:.3f} (majority={val_eval.majority_baseline_accuracy:.3f}) "
              f"(persistence={persistence_acc:.3f})")

        training_report["agents"][name] = {
            "target_definition": agent.TARGET_DEFINITION,
            "train_n": len(X_train), "val_n": len(X_val),
            "train_excluded_target_incomplete": excl_target_tr, "train_excluded_insufficient_lookback": excl_lb_tr,
            "val_excluded_target_incomplete": excl_target_val, "val_excluded_insufficient_lookback": excl_lb_val,
            "val_persistence_baseline_accuracy": persistence_acc,
            "val_persistence_n_scored": int(len(y_true_persist)),
            "missing_classes_in_fit": agent._missing_classes_in_fit,
            "train_eval": train_eval.to_dict(),
            "val_eval": val_eval.to_dict(),
        }

    (OUT_DIR / "training_report.json").write_text(json.dumps(training_report, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_DIR / 'training_report.json'}")

    verify_original_untouched()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
