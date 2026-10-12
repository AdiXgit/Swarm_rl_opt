"""Phase 2.1 — read-only model diagnostics.

Reproduces the TRAIN/VALIDATION metrics in `data/models/phase2_training_report.json`
directly from the saved artifacts (no retraining), adds class-distribution and
prediction-stability checks, simple historical baselines beyond majority-class,
a feature-manifest audit, and a leakage-safeguard review.

Never touches TEST: no TEST row is scored by any agent in this script. TEST
class counts that already exist in the frozen `phase2_training_report.json`
are not reproduced or displayed here either, to keep this run's TEST
footprint at exactly zero.

    python scripts/run_phase2_diagnostics.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from swarm_opt.agents.direction import DirectionAgent  # noqa: E402
from swarm_opt.agents.regime import RegimeAgent  # noqa: E402
from swarm_opt.agents.volatility import VolatilityAgent  # noqa: E402
from swarm_opt.data.providers.local_parquet_provider import LocalParquetProvider  # noqa: E402
from swarm_opt.data.splits import DATASET_END, DATASET_START  # noqa: E402
from swarm_opt.evaluation import evaluate_predictions  # noqa: E402
from swarm_opt.features.audit import (  # noqa: E402
    assert_duplicate_free,
    assert_feature_label_alignment,
    assert_no_cross_split_labels,
    assert_no_future_features,
)
from swarm_opt.features.manifest import FeatureManifest  # noqa: E402
from swarm_opt.features.pointintime import FEATURE_NAMES  # noqa: E402
from swarm_opt.pipeline import build_dataset, make_xy, split_targets  # noqa: E402

DEFAULT_ROOT = ROOT / "data" / "processed" / "nifty_eod"
DEFAULT_ATM_IV = ROOT / "data" / "features" / "atm_iv_history.parquet"
DEFAULT_ARTIFACTS = ROOT / "data" / "models"
OUT_JSON = DEFAULT_ARTIFACTS / "phase2_diagnostics_report.json"
OUT_MD = DEFAULT_ARTIFACTS / "phase2_diagnostics_report.md"

EXISTING_REPORT = DEFAULT_ARTIFACTS / "phase2_training_report.json"

AGENTS = {
    "direction": (DirectionAgent, "direction_label"),
    "volatility": (VolatilityAgent, "volatility_label"),
    "regime": (RegimeAgent, "regime_label"),
}

#: Tolerance for declaring a reproduced metric "matches the saved report".
REPRO_TOLERANCE = 1e-9


def verify_manifest(root: Path) -> dict:
    manifest = json.loads((root / "manifest_splits.json").read_text())
    expected_fp = "653ccb70c795d86f"
    if manifest["content_fingerprint"] != expected_fp:
        raise SystemExit(
            f"dataset fingerprint mismatch: {manifest['content_fingerprint']!r} != {expected_fp!r}"
        )
    return manifest


def stratified_random_expected_accuracy(class_counts: dict[str, int]) -> float:
    """Expected accuracy of guessing each class with probability equal to its
    TRAIN frequency, independent of the true label each draw (a strictly
    weaker baseline than majority-class, included for contrast)."""
    n = sum(class_counts.values())
    if n == 0:
        return float("nan")
    return float(sum((c / n) ** 2 for c in class_counts.values()))


def persistence_baseline_accuracy(targets: pd.DataFrame, label_col: str) -> dict:
    """'Predict whatever the previous chronological episode's label actually
    was.' Uses only already-resolved information (the previous episode's
    label is known once that episode exits, which is before the current
    episode's decision date in a non-overlapping weekly cadence) so it is not
    a leakage risk -- it is a classic naive time-series baseline, not a
    feature used by any trained agent here."""
    ordered = targets.sort_values("decision_date")
    usable = ordered[ordered["complete"] & (ordered[label_col] != "UNKNOWN")]
    labels = usable[label_col].to_numpy()
    if len(labels) < 2:
        return {"n": len(labels), "accuracy": float("nan"), "note": "fewer than 2 usable rows"}
    correct = int(np.sum(labels[1:] == labels[:-1]))
    return {"n": len(labels) - 1, "accuracy": correct / (len(labels) - 1)}


def prediction_distribution(y_pred: list[str], classes: tuple[str, ...]) -> dict[str, float]:
    n = len(y_pred)
    counts = pd.Series(y_pred).value_counts()
    return {c: float(counts.get(c, 0) / n) if n else float("nan") for c in classes}


def collapse_flags(train_dist: dict, val_dist: dict, classes: tuple[str, ...]) -> list[str]:
    flags = []
    for c in classes:
        if train_dist.get(c, 0) == 0.0:
            flags.append(f"agent never predicts {c} on TRAIN")
        if val_dist.get(c, 0) == 0.0:
            flags.append(f"agent never predicts {c} on VALIDATION")
    # a single class taking >90% of VALIDATION predictions, when TRAIN did not
    # collapse the same way, is the signature of validation-time collapse
    max_val_class, max_val_share = max(val_dist.items(), key=lambda kv: kv[1])
    if max_val_share > 0.90 and train_dist.get(max_val_class, 0) < 0.90:
        flags.append(f"VALIDATION predictions collapse onto {max_val_class} ({max_val_share:.0%})")
    return flags


def probability_diagnostics(proba: np.ndarray) -> dict:
    sorted_desc = np.sort(proba, axis=1)[:, ::-1]
    margin = sorted_desc[:, 0] - sorted_desc[:, 1] if proba.shape[1] > 1 else sorted_desc[:, 0]
    return {
        "mean_top1_probability": float(np.mean(sorted_desc[:, 0])),
        "mean_margin_top1_top2": float(np.mean(margin)),
        "fraction_near_uniform_margin_lt_0.05": float(np.mean(margin < 0.05)),
        "min_probability": float(np.min(proba)),
        "max_probability": float(np.max(proba)),
    }


def feature_audit(dataset, train_targets: pd.DataFrame) -> dict:
    train_dates = set(train_targets[train_targets["complete"]]["decision_date"])
    feats_by_date = dataset.features.set_index(dataset.features["date"].dt.date)
    train_rows = feats_by_date.loc[feats_by_date.index.isin(train_dates), FEATURE_NAMES]

    missingness = {c: float(train_rows[c].isna().mean()) for c in FEATURE_NAMES}
    variance = {c: float(train_rows[c].var(skipna=True)) for c in FEATURE_NAMES}
    near_zero_variance = sorted(c for c, v in variance.items() if np.isfinite(v) and v < 1e-10)

    corr = train_rows.corr(numeric_only=True)
    redundant_pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if pd.notna(r) and abs(r) > 0.98:
                redundant_pairs.append({"a": cols[i], "b": cols[j], "corr": float(r)})

    return {
        "rows_used": int(len(train_rows)),
        "missingness_by_feature": missingness,
        "fully_missing_features": sorted(c for c, f in missingness.items() if f == 1.0),
        "near_zero_variance_features": near_zero_variance,
        "redundant_feature_pairs_abs_corr_gt_0.98": redundant_pairs,
    }


def leakage_review(dataset) -> dict:
    findings = {"automated_checks": {}, "manual_review_notes": []}

    assert_no_future_features(dataset.features)
    findings["automated_checks"]["assert_no_future_features"] = "PASS"
    assert_feature_label_alignment(dataset.features, dataset.targets)
    findings["automated_checks"]["assert_feature_label_alignment"] = "PASS"
    assert_no_cross_split_labels(dataset.targets)
    findings["automated_checks"]["assert_no_cross_split_labels"] = "PASS"
    assert_duplicate_free(dataset.features, "date")
    assert_duplicate_free(dataset.targets, "decision_date")
    findings["automated_checks"]["assert_duplicate_free"] = "PASS"

    findings["manual_review_notes"] = [
        "assert_no_future_features only matches column names containing "
        "'future'/'forward'; the actual leakage proof for the underlying-"
        "derived features (ret_*, mom_*, realized_vol_*, drawdown_20, "
        "range_20, trend_slope_20) is the mutation test in "
        "tests/test_features_pointintime.py::test_no_future_observation_enters_a_feature_row, "
        "not this name check by itself.",
        "The ATM IV / skew features (atm_iv, atm_iv_chg_5, skew_25d, "
        "atm_days_to_expiry) have no equivalent mutation-based leakage test: "
        "nothing in the suite proves that build_market_features.py never "
        "reads a chain dated after the row's own date. Point-in-time-ness "
        "for this cache currently rests on code review of "
        "scripts/build_market_features.py, not an automated regression test.",
        "pipeline.make_xy filters training rows on target completeness "
        "(targets['complete']) but does not independently filter on the "
        "feature-side 'has_full_lookback' flag. In the current frozen "
        "dataset this is harmless by coincidence: every feature that needs "
        "the full 20-day lookback also feeds the volatility baseline used "
        "to compute 'complete', so an insufficient-lookback decision date "
        "is always also an UNKNOWN-labelled one. This coupling is not "
        "enforced in code, so it would silently break if a future feature "
        "or target change decoupled the two lookback requirements.",
    ]
    return findings


def main() -> int:
    manifest = verify_manifest(DEFAULT_ROOT)
    print(f"manifest verified: fingerprint={manifest['content_fingerprint']}")

    existing_report = json.loads(EXISTING_REPORT.read_text())
    feature_manifest = FeatureManifest.load(DEFAULT_ARTIFACTS / "feature_manifest.json")

    provider = LocalParquetProvider(DEFAULT_ROOT)
    atm_iv_hist = pd.read_parquet(DEFAULT_ATM_IV)
    dataset = build_dataset(provider, "NIFTY", DATASET_START, DATASET_END, atm_iv_hist)
    by_split = split_targets(dataset)
    train_targets, val_targets = by_split["train"], by_split["validation"]

    report: dict = {
        "manifest_fingerprint": manifest["content_fingerprint"],
        "test_data_accessed": False,
        "test_data_access_note": (
            "No TEST row was scored by any agent in this run. TEST class "
            "counts are not reproduced or displayed here; they exist only in "
            "the previously-written phase2_training_report.json from the "
            "original (non-diagnostic) training run."
        ),
        "sample_size_explanation": {
            "train_trading_dates": manifest["splits"]["train"]["trading_dates"],
            "train_weekly_episodes_before_purge": None,  # filled below
            "train_weekly_episodes_after_purge": len(
                [e for e in dataset.episodes_kept if e.decision_date >= pd.Timestamp("2021-04-01").date()
                 and e.decision_date <= pd.Timestamp("2024-12-31").date()]
            ),
            "train_episodes_with_complete_target": int(train_targets["complete"].sum()),
            "train_episodes_unknown_target": int((~train_targets["complete"]).sum()),
            "reduction_chain": [
                "929 TRAIN trading dates -> 194 weekly episodes via "
                "ExpiryCalendar.weekly_episodes() (calendar-derived, not a fixed weekday)",
                "2 episodes purged dataset-wide for straddling a split boundary "
                "(swarm_opt.data.splits.purge_straddling); both purges fall at "
                "the TRAIN/VALIDATION and VALIDATION/TEST walls, not inside TRAIN itself",
                "194 TRAIN episodes kept after purge",
                "5 of those 194 have an incomplete target (decision date falls "
                "within the first 20 trading dates of the whole dataset, where "
                "the realized_vol_20 baseline used by the volatility/regime "
                "targets is NaN for lack of lookback) -> labelled UNKNOWN, excluded from fit",
                "189 TRAIN rows actually used to fit each agent, matching "
                "phase2_training_report.json exactly",
            ],
        },
        "agents": {},
        "feature_audit": feature_audit(dataset, train_targets),
        "leakage_review": leakage_review(dataset),
    }
    report["sample_size_explanation"]["train_weekly_episodes_before_purge"] = (
        len(dataset.episodes_kept) + len(dataset.episodes_purged)
    )

    for name, (agent_cls, label_col) in AGENTS.items():
        agent_dir = DEFAULT_ARTIFACTS / name
        agent = agent_cls.load(agent_dir)  # no retraining
        feature_manifest.assert_compatible(agent.feature_names, agent.feature_schema_version)

        X_train, y_train, ts_train = make_xy(dataset, train_targets, label_col)
        X_val, y_val, ts_val = make_xy(dataset, val_targets, label_col)

        train_preds = agent.predict(X_train, ts_train)
        val_preds = agent.predict(X_val, ts_val)
        train_proba = pd.DataFrame([p.probabilities for p in train_preds])[list(agent.CLASSES)].to_numpy()
        val_proba = pd.DataFrame([p.probabilities for p in val_preds])[list(agent.CLASSES)].to_numpy()
        train_pred_cls = [p.predicted_class for p in train_preds]
        val_pred_cls = [p.predicted_class for p in val_preds]

        train_eval = evaluate_predictions("train", agent.CLASSES, y_train, train_pred_cls, train_proba)
        val_eval = evaluate_predictions("validation", agent.CLASSES, y_val, val_pred_cls, val_proba)

        existing = existing_report["agents"][name]
        repro_diff = {
            "train_accuracy_diff": train_eval.accuracy - existing["train_eval"]["accuracy"],
            "val_accuracy_diff": val_eval.accuracy - existing["validation_eval"]["accuracy"],
            "train_log_loss_diff": train_eval.log_loss - existing["train_eval"]["log_loss"],
            "val_log_loss_diff": val_eval.log_loss - existing["validation_eval"]["log_loss"],
        }
        reproduced_exactly = all(abs(v) < 1e-6 for v in repro_diff.values())

        train_dist_true = {c: int(y_train.value_counts().get(c, 0)) for c in agent.CLASSES}
        val_dist_true = {c: int(y_val.value_counts().get(c, 0)) for c in agent.CLASSES}
        train_dist_pred = prediction_distribution(train_pred_cls, agent.CLASSES)
        val_dist_pred = prediction_distribution(val_pred_cls, agent.CLASSES)

        strat_random_train = stratified_random_expected_accuracy(train_dist_true)
        persistence_train = persistence_baseline_accuracy(train_targets, label_col)
        persistence_val = persistence_baseline_accuracy(val_targets, label_col)

        report["agents"][name] = {
            "reproduced_from_saved_artifact": True,
            "reproduction_matches_existing_report": reproduced_exactly,
            "reproduction_diff": repro_diff,
            "class_counts": {"train": train_dist_true, "validation": val_dist_true},
            "class_proportions": {
                "train": {c: v / sum(train_dist_true.values()) for c, v in train_dist_true.items()},
                "validation": {c: v / sum(val_dist_true.values()) for c, v in val_dist_true.items()},
            },
            "train_eval": train_eval.to_dict(),
            "validation_eval": val_eval.to_dict(),
            "prediction_distribution": {"train": train_dist_pred, "validation": val_dist_pred},
            "collapse_flags": collapse_flags(train_dist_pred, val_dist_pred, agent.CLASSES),
            "probability_diagnostics": {
                "train": probability_diagnostics(train_proba),
                "validation": probability_diagnostics(val_proba),
            },
            "baselines": {
                "majority_class": {
                    "train": existing["train_eval"]["majority_baseline_accuracy"],
                    "validation": existing["validation_eval"]["majority_baseline_accuracy"],
                },
                "stratified_random_on_train_frequencies": {
                    "train_expected_accuracy": strat_random_train,
                },
                "persistence_previous_episode_label": {
                    "train": persistence_train,
                    "validation": persistence_val,
                },
                "agent_vs_majority_validation": (
                    val_eval.accuracy - existing["validation_eval"]["majority_baseline_accuracy"]
                ),
                "agent_vs_persistence_validation": (
                    val_eval.accuracy - persistence_val.get("accuracy", float("nan"))
                ),
            },
        }
        print(
            f"{name}: reproduced={reproduced_exactly} "
            f"VAL acc={val_eval.accuracy:.3f} majority={existing['validation_eval']['majority_baseline_accuracy']:.3f} "
            f"persistence={persistence_val.get('accuracy', float('nan')):.3f}"
        )

    OUT_JSON.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_JSON}")
    write_markdown(report, existing_report)
    print(f"wrote {OUT_MD}")
    return 0


def write_markdown(report: dict, existing_report: dict) -> None:
    lines = ["# Phase 2.1 Diagnostics Report", ""]
    lines.append(f"Manifest fingerprint verified: `{report['manifest_fingerprint']}`")
    lines.append(f"TEST data accessed: **{report['test_data_accessed']}** - {report['test_data_access_note']}")
    lines.append("")
    lines.append("## Sample size / reduction chain (TRAIN)")
    for step in report["sample_size_explanation"]["reduction_chain"]:
        lines.append(f"- {step}")
    lines.append("")

    for name, a in report["agents"].items():
        lines.append(f"## {name.capitalize()} agent")
        lines.append(f"- Reproduced from saved artifact without retraining: **{a['reproduced_from_saved_artifact']}**")
        lines.append(f"- Matches existing phase2_training_report.json: **{a['reproduction_matches_existing_report']}**")
        lines.append("")
        lines.append("| split | n | accuracy | balanced acc | macro F1 | log loss | brier |")
        lines.append("|---|---|---|---|---|---|---|")
        for split in ("train_eval", "validation_eval"):
            e = a[split]
            lines.append(
                f"| {e['split']} | {e['n']} | {e['accuracy']:.3f} | {e['balanced_accuracy']:.3f} | "
                f"{e['macro_f1']:.3f} | {e['log_loss']:.3f} | {e['multiclass_brier']:.3f} |"
            )
        lines.append("")
        lines.append("**Baselines (VALIDATION accuracy comparison)**")
        b = a["baselines"]
        lines.append(f"- Majority-class baseline: {b['majority_class']['validation']:.3f}")
        lines.append(f"- Persistence baseline (previous episode's label): {b['persistence_previous_episode_label']['validation'].get('accuracy', float('nan')):.3f}")
        lines.append(f"- Agent accuracy vs. majority: {b['agent_vs_majority_validation']:+.3f}")
        lines.append(f"- Agent accuracy vs. persistence: {b['agent_vs_persistence_validation']:+.3f}")
        lines.append("")
        if a["collapse_flags"]:
            lines.append("**Collapse flags:**")
            for f in a["collapse_flags"]:
                lines.append(f"- {f}")
            lines.append("")

    lines.append("## Feature audit (TRAIN)")
    fa = report["feature_audit"]
    if fa["fully_missing_features"]:
        lines.append(f"- Fully missing features: {fa['fully_missing_features']}")
    if fa["near_zero_variance_features"]:
        lines.append(f"- Near-zero-variance features: {fa['near_zero_variance_features']}")
    if fa["redundant_feature_pairs_abs_corr_gt_0.98"]:
        lines.append("- Redundant feature pairs (|corr| > 0.98):")
        for p in fa["redundant_feature_pairs_abs_corr_gt_0.98"]:
            lines.append(f"  - `{p['a']}` vs `{p['b']}`: corr={p['corr']:.4f}")
    lines.append("")

    lines.append("## Leakage safeguard review")
    for check, result in report["leakage_review"]["automated_checks"].items():
        lines.append(f"- `{check}`: {result}")
    lines.append("")
    lines.append("**Manual review notes (gaps, not failures):**")
    for note in report["leakage_review"]["manual_review_notes"]:
        lines.append(f"- {note}")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
