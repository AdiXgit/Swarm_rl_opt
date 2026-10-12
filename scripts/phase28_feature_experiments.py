"""Phase 2.8 Workstream B, Stage E: controlled India VIX feature ablation.

Compares the existing pointintime_v2 17-feature baseline against baseline +
India VIX (level, 5-day change) -- a single new family, measured on its
own, not combined with any other untested change. Same walk-forward
harness as Phase 2.6/2.7 (imported, not duplicated). TRAIN-only fitting;
VALIDATION read once per agent for the final comparison. TEST never read.

    python scripts/phase28_feature_experiments.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

spec26 = importlib.util.spec_from_file_location("phase26_experiments", ROOT / "scripts" / "phase26_experiments.py")
p26 = importlib.util.module_from_spec(spec26)
spec26.loader.exec_module(p26)

from swarm_opt.data.providers.combined_provider import CombinedParquetProvider  # noqa: E402
from swarm_opt.data.providers.local_parquet_provider import LocalParquetProvider  # noqa: E402
from swarm_opt.evaluation import evaluate_predictions  # noqa: E402
from swarm_opt.features.pointintime import (  # noqa: E402
    FEATURE_NAMES_V2,
    FEATURE_NAMES_V3,
    FEATURE_SCHEMA_VERSION_V3,
    build_market_features,
)
from swarm_opt.features.targets import TargetThresholds, build_targets, fit_thresholds_on_train  # noqa: E402

OUT_DIR = ROOT / "data" / "models" / "phase2_8"
OUT_DIR.mkdir(parents=True, exist_ok=True)
INDIA_VIX_PATH = ROOT / "data" / "features" / "india_vix_history.parquet"


def build_dataset_v3():
    """Mirrors p26.build_extended_dataset but additionally merges India VIX,
    so both v2 (17-feature) and v3 (19-feature) columns are present in one
    frame -- the ablation then just selects different column subsets of
    the SAME feature table, not two separately-built pipelines."""
    ext_provider = LocalParquetProvider(p26.EXTENSION_ROOT)
    orig_provider = LocalParquetProvider(p26.ORIGINAL_ROOT)
    combined = CombinedParquetProvider([ext_provider, orig_provider])

    hist = combined.underlying_history("NIFTY", p26.EXTENDED_TRAIN_START, p26.TEST.end)
    atm_iv = pd.concat(
        [pd.read_parquet(p26.EXTENSION_ATM_IV), pd.read_parquet(p26.ORIGINAL_ATM_IV)], ignore_index=True
    ).sort_values("date").reset_index(drop=True)
    india_vix = pd.read_parquet(INDIA_VIX_PATH)

    features, report = build_market_features(
        hist, atm_iv_hist=atm_iv, feature_names=FEATURE_NAMES_V3, india_vix_hist=india_vix,
    )

    from swarm_opt.data.calendar import ExpiryCalendar

    dates = combined.trading_dates(p26.EXTENDED_TRAIN_START, p26.TEST.end)
    calendar = ExpiryCalendar.from_provider(combined, "NIFTY", dates)
    all_episodes = calendar.weekly_episodes()

    def extended_split_for(day):
        if p26.EXTENDED_TRAIN_START <= day <= p26.EXTENDED_TRAIN_END:
            return "extended_train"
        if p26.VALIDATION.start <= day <= p26.VALIDATION.end:
            return "validation"
        if p26.TEST.start <= day <= p26.TEST.end:
            return "test"
        return None

    kept = [ep for ep in all_episodes if extended_split_for(ep.decision_date) is not None
            and extended_split_for(ep.decision_date) == extended_split_for(ep.exit_date)]

    baseline_vol = features.set_index(features["date"].dt.date)["realized_vol_20"]
    train_episodes = [e for e in kept if extended_split_for(e.decision_date) == "extended_train"]
    val_episodes = [e for e in kept if extended_split_for(e.decision_date) == "validation"]

    provisional = build_targets(train_episodes, hist, baseline_vol, TargetThresholds(1.0, 1.0))
    thresholds = fit_thresholds_on_train(provisional["forward_return"])

    train_targets = build_targets(train_episodes, hist, baseline_vol, thresholds)
    val_targets = build_targets(val_episodes, hist, baseline_vol, thresholds)
    return features, train_targets, val_targets, thresholds, report


def main() -> int:
    print("building v3 (v2 + India VIX) feature table ...", flush=True)
    features, train_targets, val_targets, thresholds, report = build_dataset_v3()
    print(f"  {report}")
    print(f"  india_vix coverage: {report.india_vix_coverage:.4f}")

    registry: dict = {
        "feature_report": {"india_vix_coverage": report.india_vix_coverage, "atm_iv_coverage": report.atm_iv_coverage},
        "feature_sets_compared": {"baseline_v2_17": FEATURE_NAMES_V2, "v2_plus_india_vix_19": FEATURE_NAMES_V3},
        "agents": {},
    }

    fn = lambda Xt, yt, Xe, c: p26.fit_predict_xgb(Xt, yt, Xe, c, p26.XGB_BASELINE)

    for agent, (label_col, classes) in p26.AGENTS.items():
        print(f"\n=== {agent} ===")
        agent_results = {}
        for fs_name, fs_list in (("baseline_v2_17", FEATURE_NAMES_V2), ("v2_plus_india_vix_19", FEATURE_NAMES_V3)):
            X, y, _ = p26.make_xy(features, train_targets, label_col, fs_list)
            X_val, y_val, _ = p26.make_xy(features, val_targets, label_col, fs_list)
            wf = p26.walk_forward_score(X, y, classes, fn)

            pred_val, proba_val = fn(X, y, X_val, classes)
            maj_val = float(y_val.value_counts().max() / len(y_val))
            ev = evaluate_predictions("validation", classes, y_val, pred_val, proba_val)

            agent_results[fs_name] = {
                "n_features": len(fs_list), "train_n": len(X), "val_n": len(X_val),
                "walk_forward_mean_accuracy": wf["mean_accuracy"],
                "walk_forward_mean_balanced_accuracy": wf["mean_balanced_accuracy"],
                "walk_forward_per_fold": wf["folds"],
                "validation_accuracy": ev.accuracy, "validation_balanced_accuracy": ev.balanced_accuracy,
                "validation_majority_baseline": maj_val,
            }
            print(f"  {fs_name} ({len(fs_list)}f): walk_fwd_bal_acc={wf['mean_balanced_accuracy']:.3f} "
                  f"VAL_acc={ev.accuracy:.3f} VAL_bal_acc={ev.balanced_accuracy:.3f} majority={maj_val:.3f}")

        base = agent_results["baseline_v2_17"]
        plus = agent_results["v2_plus_india_vix_19"]
        delta_wf = plus["walk_forward_mean_balanced_accuracy"] - base["walk_forward_mean_balanced_accuracy"]
        delta_val = plus["validation_balanced_accuracy"] - base["validation_balanced_accuracy"]
        agent_results["incremental_contribution"] = {
            "walk_forward_balanced_accuracy_delta": delta_wf,
            "validation_balanced_accuracy_delta": delta_val,
            "persists_across_walk_forward_and_validation": (delta_wf > 0) and (delta_val > 0),
        }
        print(f"  india_vix incremental delta: walk_fwd={delta_wf:+.3f} val={delta_val:+.3f} "
              f"persists={agent_results['incremental_contribution']['persists_across_walk_forward_and_validation']}")
        registry["agents"][agent] = agent_results

    (OUT_DIR / "phase28_feature_experiment_results.json").write_text(
        json.dumps(registry, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nwrote {OUT_DIR / 'phase28_feature_experiment_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
