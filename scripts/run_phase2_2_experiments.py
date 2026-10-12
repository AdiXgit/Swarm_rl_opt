"""Phase 2.2 Task C — pre-registered controlled experiment matrix.

Reads `data/models/phase2_2_protocol.json` (frozen before this script was
written to run anything) and executes exactly the 4 experiments it declares.
Never scores, reads, or reports a TEST row. Never overwrites the original
Phase 2 baseline artifacts (`data/models/{direction,volatility,regime}/`,
`data/models/phase2_training_report.json`,
`data/models/phase2_diagnostics_report.json`); all new output goes to
`data/models/phase2_2_experiment_registry.json`,
`data/models/phase2_2_experiments_report.md`, and (for the small number of
illustrative saved boosters) `data/models/experiments/`.

    python scripts/run_phase2_2_experiments.py
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from swarm_opt.data.providers.local_parquet_provider import LocalParquetProvider  # noqa: E402
from swarm_opt.data.splits import DATASET_END, DATASET_START, TRAIN  # noqa: E402
from swarm_opt.evaluation import evaluate_predictions  # noqa: E402
from swarm_opt.pipeline import build_dataset, eligibility_report, make_xy, split_targets  # noqa: E402

DEFAULT_ROOT = ROOT / "data" / "processed" / "nifty_eod"
DEFAULT_ATM_IV = ROOT / "data" / "features" / "atm_iv_history.parquet"
MODELS_DIR = ROOT / "data" / "models"
PROTOCOL_PATH = MODELS_DIR / "phase2_2_protocol.json"
OUT_REGISTRY = MODELS_DIR / "phase2_2_experiment_registry.json"
OUT_REPORT = MODELS_DIR / "phase2_2_experiments_report.md"
EXPERIMENTS_MODELS_DIR = MODELS_DIR / "experiments"

AGENT_CLASSES = {
    "direction": ("direction_label", ("BEARISH", "NEUTRAL", "BULLISH")),
    "volatility": ("volatility_label", ("CONTRACT", "NEUTRAL", "EXPAND")),
    "regime": ("regime_label", ("TRENDING_DOWN", "TRENDING_UP", "RANGE_BOUND", "HIGH_VOLATILITY", "LOW_VOLATILITY")),
}


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_baseline_hashes(protocol: dict) -> None:
    for rel, expected in protocol["baseline_artifacts"].items():
        if rel in ("note", "hash_algorithm"):
            continue
        actual = sha256_of(MODELS_DIR / rel)
        if actual != expected:
            raise SystemExit(
                f"baseline artifact changed since protocol freeze: {rel}\n"
                f"  expected {expected}\n  actual   {actual}"
            )
    print("baseline artifact hashes: verified unchanged since protocol freeze")


def verify_manifest(root: Path, expected_fp: str) -> None:
    manifest = json.loads((root / "manifest_splits.json").read_text())
    if manifest["content_fingerprint"] != expected_fp:
        raise SystemExit(f"dataset fingerprint changed: {manifest['content_fingerprint']} != {expected_fp}")


def git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# low-level fit/predict, independent of the production XGBClassifierAgent
# (Task A found no defect in agents/base.py, so it is left untouched; these
# experiments need configurations -- custom feature sets, sample weights --
# that class was not designed to vary, so a separate, explicitly
# experiment-only function is used instead of bending the production class).
# --------------------------------------------------------------------------- #
def fit_predict_xgb(X_train, y_train, X_val, classes, params, seed, sample_weight=None):
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = y_train.map(class_to_idx).to_numpy()
    dtrain = xgb.DMatrix(X_train.to_numpy(dtype=float), label=y_idx, weight=sample_weight)
    booster_params = {
        "objective": "multi:softprob", "num_class": len(classes), "seed": seed,
        "max_depth": params["max_depth"], "eta": params["learning_rate"],
        "subsample": params["subsample"], "colsample_bytree": params["colsample_bytree"],
        "lambda": params["reg_lambda"], "nthread": params["n_jobs"],
    }
    booster = xgb.train(booster_params, dtrain, num_boost_round=params["n_estimators"])

    def predict(X):
        proba = booster.predict(xgb.DMatrix(X.to_numpy(dtype=float))).reshape(len(X), len(classes))
        pred = [classes[i] for i in proba.argmax(axis=1)]
        return proba, pred

    train_proba, train_pred = predict(X_train)
    val_proba, val_pred = predict(X_val)
    return {
        "booster": booster,
        "train_proba": train_proba, "train_pred": train_pred,
        "val_proba": val_proba, "val_pred": val_pred,
        "missing_classes_in_fit": sorted(set(classes) - set(y_train.unique())),
    }


def make_xy_custom_features(dataset, targets, label_col, feature_names):
    """Same eligibility rule as `pipeline.make_xy` (target-complete AND
    full-lookback); only the feature *columns* returned differ."""
    _, y, ts = make_xy(dataset, targets, label_col)
    feats_by_date = dataset.features.set_index(dataset.features["date"].dt.date)
    dates = [pd.Timestamp(t).date() for t in ts]
    X = feats_by_date.loc[dates, feature_names].reset_index(drop=True)
    return X, y, ts


def persistence_predictions(train_targets, val_targets, label_col):
    """Predict each row's label as the immediately preceding chronological
    episode's true label, across the TRAIN->VALIDATION boundary (the first
    VALIDATION row is predicted from TRAIN's last label -- strictly
    historical, no future access). Returns (y_true, y_pred) for VALIDATION
    rows only; the very first row of the whole sequence has no predecessor
    and is dropped if it happens to be scored (it never is here, since the
    first eligible row is always in TRAIN)."""
    def _usable(df):
        return df[df["complete"] & (df[label_col] != "UNKNOWN")][["decision_date", label_col]]

    train_u, val_u = _usable(train_targets), _usable(val_targets)
    combined = pd.concat([train_u, val_u], ignore_index=True).sort_values("decision_date").reset_index(drop=True)
    pred = combined[label_col].shift(1)
    is_val = combined["decision_date"].isin(set(val_u["decision_date"]))
    mask = is_val & pred.notna()
    return combined.loc[mask, label_col].reset_index(drop=True), pred.loc[mask].reset_index(drop=True)


def full_metrics(split, classes, y_true, y_pred, proba, majority_baseline_acc, persistence_acc=None):
    ev = evaluate_predictions(split, classes, y_true, y_pred, proba)
    d = ev.to_dict()
    d["majority_baseline_accuracy_reference"] = majority_baseline_acc
    if persistence_acc is not None:
        d["persistence_baseline_accuracy_reference"] = persistence_acc
    d["predicted_class_counts"] = {c: int(pd.Series(y_pred).value_counts().get(c, 0)) for c in classes}
    return d


def mean_std(values: list[float]) -> dict:
    arr = np.array(values, dtype=float)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=0)), "values": [float(v) for v in arr]}


# --------------------------------------------------------------------------- #
def run_experiment_1(dataset, by_split, protocol, registry) -> None:
    """Feature redundancy ablation: baseline 18-feature set vs. dropping
    `dist_sma20` (byte-identical to `mom_20`). Same params, same eligible
    rows, same seeds; only the feature columns differ."""
    print("\n=== Experiment 1: feature redundancy ablation ===")
    feature_sets = protocol["feature_sets"]
    params = protocol["model_configurations"]["baseline_xgb"]
    seeds = protocol["seeds"]["experiment_1_feature_ablation"]
    out = {}

    for agent, (label_col, classes) in AGENT_CLASSES.items():
        train_df, val_df = by_split["train"], by_split["validation"]
        agent_out = {}
        for set_name, feat_list in (("baseline_18", feature_sets["baseline_18"]),
                                     ("ablation_17_no_dist_sma20", feature_sets["ablation_17_no_dist_sma20"])):
            X_train, y_train, _ = make_xy_custom_features(dataset, train_df, label_col, feat_list)
            X_val, y_val, _ = make_xy_custom_features(dataset, val_df, label_col, feat_list)
            maj_train = y_train.value_counts(normalize=True).max()
            maj_val = y_val.value_counts().reindex(y_train.value_counts().idxmax()) if False else None
            majority_class = y_train.value_counts().idxmax()
            maj_val_acc = float((y_val == majority_class).mean())

            val_accs = []
            per_seed = []
            for seed in seeds:
                res = fit_predict_xgb(X_train, y_train, X_val, classes, params, seed)
                val_eval = full_metrics("validation", classes, y_val, res["val_pred"], res["val_proba"], maj_val_acc)
                val_accs.append(val_eval["accuracy"])
                per_seed.append({"seed": seed, "val_eval": val_eval,
                                  "missing_classes_in_fit": res["missing_classes_in_fit"]})
            agent_out[set_name] = {
                "n_features": len(feat_list),
                "train_n": len(X_train), "val_n": len(X_val),
                "val_accuracy_across_seeds": mean_std(val_accs),
                "per_seed": per_seed,
            }
        a, b = agent_out["baseline_18"]["val_accuracy_across_seeds"], agent_out["ablation_17_no_dist_sma20"]["val_accuracy_across_seeds"]
        diff = b["mean"] - a["mean"]
        noise_floor = max(a["std"], b["std"])
        agent_out["comparison"] = {
            "val_accuracy_diff_ablation_minus_baseline": diff,
            "cross_seed_noise_floor": noise_floor,
            "verdict": "within cross-seed noise (inconclusive)" if abs(diff) <= noise_floor else "exceeds cross-seed noise",
        }
        out[agent] = agent_out
        print(f"  {agent}: baseline={a['mean']:.3f}+/-{a['std']:.3f} ablation={b['mean']:.3f}+/-{b['std']:.3f} "
              f"diff={diff:+.3f} ({agent_out['comparison']['verdict']})")

    registry["experiment_1_feature_ablation"] = out


def run_experiment_2(dataset, by_split, protocol, registry) -> None:
    """Model-capacity comparison: baseline XGB vs. a deliberately shallow
    XGB vs. a majority-class DummyClassifier vs. the persistence baseline."""
    print("\n=== Experiment 2: model-capacity comparison ===")
    from sklearn.dummy import DummyClassifier

    baseline_params = protocol["model_configurations"]["baseline_xgb"]
    shallow_params = protocol["model_configurations"]["shallow_xgb"]
    seeds = protocol["seeds"]["experiment_2_model_capacity"]
    feature_names = protocol["feature_sets"]["baseline_18"]
    out = {}

    for agent, (label_col, classes) in AGENT_CLASSES.items():
        train_df, val_df = by_split["train"], by_split["validation"]
        X_train, y_train, _ = make_xy_custom_features(dataset, train_df, label_col, feature_names)
        X_val, y_val, _ = make_xy_custom_features(dataset, val_df, label_col, feature_names)
        majority_class = y_train.value_counts().idxmax()
        maj_val_acc = float((y_val == majority_class).mean())

        y_true_persist, y_pred_persist = persistence_predictions(train_df, val_df, label_col)
        persistence_acc = float((y_true_persist == y_pred_persist).mean()) if len(y_true_persist) else float("nan")

        configs_out = {}
        for cfg_name, params in (("baseline_xgb", baseline_params), ("shallow_xgb", shallow_params)):
            val_accs, per_seed = [], []
            for seed in seeds:
                res = fit_predict_xgb(X_train, y_train, X_val, classes, params, seed)
                val_eval = full_metrics("validation", classes, y_val, res["val_pred"], res["val_proba"],
                                         maj_val_acc, persistence_acc)
                val_accs.append(val_eval["accuracy"])
                per_seed.append({"seed": seed, "val_eval": val_eval,
                                  "missing_classes_in_fit": res["missing_classes_in_fit"]})
            configs_out[cfg_name] = {"val_accuracy_across_seeds": mean_std(val_accs), "per_seed": per_seed}

        # Seed 0 of baseline_xgb must reproduce the frozen phase2_training_report.json
        # VALIDATION accuracy exactly -- an automated check, not a one-off manual spot check.
        frozen_val_acc = {"direction": 0.38235294117647056, "volatility": 0.5294117647058824,
                           "regime": 0.35294117647058826}[agent]
        seed0_acc = next(r["val_eval"]["accuracy"] for r in configs_out["baseline_xgb"]["per_seed"] if r["seed"] == 0)
        reproduces_frozen = abs(seed0_acc - frozen_val_acc) < 1e-9
        configs_out["baseline_xgb"]["seed0_reproduces_frozen_baseline"] = reproduces_frozen
        if not reproduces_frozen:
            raise SystemExit(
                f"{agent}: seed=0 baseline_xgb VAL accuracy {seed0_acc} does not reproduce the "
                f"frozen baseline {frozen_val_acc} -- stopping before trusting any other result."
            )

        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(np.zeros((len(y_train), 1)), y_train)  # fit on TRAIN labels only; X is a placeholder
        dummy_pred = dummy.predict(np.zeros((len(y_val), 1))).tolist()
        dummy_proba = dummy.predict_proba(np.zeros((len(y_val), 1)))
        dummy_classes = list(dummy.classes_)
        # re-order/pad dummy_proba onto the full class list for a consistent confusion matrix shape
        proba_full = np.zeros((len(y_val), len(classes)))
        for j, c in enumerate(classes):
            if c in dummy_classes:
                proba_full[:, j] = dummy_proba[:, dummy_classes.index(c)]
        dummy_eval = full_metrics("validation", classes, y_val, dummy_pred, proba_full, maj_val_acc, persistence_acc)
        configs_out["dummy_majority"] = {"val_eval": dummy_eval}
        configs_out["persistence"] = {
            "n_scored": int(len(y_true_persist)),
            "accuracy": persistence_acc,
            "predicted_class_counts": {c: int((y_pred_persist == c).sum()) for c in classes},
        }

        out[agent] = configs_out
        b = configs_out["baseline_xgb"]["val_accuracy_across_seeds"]
        s = configs_out["shallow_xgb"]["val_accuracy_across_seeds"]
        print(f"  {agent}: baseline={b['mean']:.3f}+/-{b['std']:.3f} shallow={s['mean']:.3f}+/-{s['std']:.3f} "
              f"dummy={dummy_eval['accuracy']:.3f} persistence={persistence_acc:.3f}")

    registry["experiment_2_model_capacity"] = out


def balanced_capped_weights(y_train: pd.Series, classes: tuple[str, ...], cap: float) -> dict:
    n = len(y_train)
    counts = y_train.value_counts()
    weights = {}
    for c in classes:
        count_c = counts.get(c, 0)
        weights[c] = min(cap, n / (len(classes) * count_c)) if count_c > 0 else cap
    return weights


def run_experiment_3(dataset, by_split, protocol, registry) -> None:
    """Targeted class weighting for Volatility (NEUTRAL) and Regime
    (TRENDING_UP), via a bounded balanced weighting rule computed from TRAIN
    only and frozen before any VALIDATION evaluation."""
    print("\n=== Experiment 3: targeted class weighting ===")
    params = protocol["model_configurations"]["baseline_xgb"]
    weighting_cfg = protocol["model_configurations"]["class_weighted_xgb"]
    cap = weighting_cfg["cap_value"]
    seeds = protocol["seeds"]["experiment_3_class_weighting"]
    feature_names = protocol["feature_sets"]["baseline_18"]
    out = {}

    for agent in ("volatility", "regime"):
        label_col, classes = AGENT_CLASSES[agent]
        train_df, val_df = by_split["train"], by_split["validation"]
        X_train, y_train, _ = make_xy_custom_features(dataset, train_df, label_col, feature_names)
        X_val, y_val, _ = make_xy_custom_features(dataset, val_df, label_col, feature_names)
        majority_class = y_train.value_counts().idxmax()
        maj_val_acc = float((y_val == majority_class).mean())

        weights = balanced_capped_weights(y_train, classes, cap)  # frozen from TRAIN before any VAL look
        sample_weight = y_train.map(weights).to_numpy(dtype=float)

        results = {"unweighted": {"per_seed": []}, "weighted": {"per_seed": []}}
        for seed in seeds:
            res_u = fit_predict_xgb(X_train, y_train, X_val, classes, params, seed)
            results["unweighted"]["per_seed"].append({
                "seed": seed,
                "val_eval": full_metrics("validation", classes, y_val, res_u["val_pred"], res_u["val_proba"], maj_val_acc),
            })
            res_w = fit_predict_xgb(X_train, y_train, X_val, classes, params, seed, sample_weight=sample_weight)
            results["weighted"]["per_seed"].append({
                "seed": seed,
                "val_eval": full_metrics("validation", classes, y_val, res_w["val_pred"], res_w["val_proba"], maj_val_acc),
            })

        for key in ("unweighted", "weighted"):
            accs = [r["val_eval"]["accuracy"] for r in results[key]["per_seed"]]
            bal_accs = [r["val_eval"]["balanced_accuracy"] for r in results[key]["per_seed"]]
            results[key]["val_accuracy_across_seeds"] = mean_std(accs)
            results[key]["val_balanced_accuracy_across_seeds"] = mean_std(
                [v for v in bal_accs if np.isfinite(v)]
            ) if any(np.isfinite(v) for v in bal_accs) else {"mean": float("nan"), "std": float("nan"), "values": bal_accs}

        target_class = "NEUTRAL" if agent == "volatility" else "TRENDING_UP"
        results["frozen_weights"] = weights
        results["target_class_recall_across_seeds"] = {
            key: mean_std([r["val_eval"]["per_class_recall"].get(target_class, float("nan"))
                            for r in results[key]["per_seed"]])
            for key in ("unweighted", "weighted")
        }
        out[agent] = results
        u = results["unweighted"]["val_accuracy_across_seeds"]
        w = results["weighted"]["val_accuracy_across_seeds"]
        tr = results["target_class_recall_across_seeds"]
        print(f"  {agent}: unweighted acc={u['mean']:.3f}+/-{u['std']:.3f} weighted acc={w['mean']:.3f}+/-{w['std']:.3f} "
              f"| {target_class} recall unweighted={tr['unweighted']['mean']:.3f} weighted={tr['weighted']['mean']:.3f}")

    registry["experiment_3_class_weighting"] = out


def build_expanding_window_folds(n: int, n_folds: int = 4, initial_fraction: float = 0.4):
    initial = int(n * initial_fraction)
    remainder = n - initial
    fold_size = remainder // n_folds
    folds = []
    start = initial
    for i in range(n_folds):
        end = start + fold_size if i < n_folds - 1 else n
        folds.append((0, start, start, end))  # (train_start, train_end, test_start, test_end)
        start = end
    return folds


def run_experiment_4(dataset, by_split, protocol, registry) -> None:
    """Expanding-window evaluation within TRAIN only (model-selection
    diagnostic, not a VALIDATION/TEST substitute)."""
    print("\n=== Experiment 4: expanding-window evaluation within TRAIN ===")
    params = protocol["model_configurations"]["baseline_xgb"]
    seed = protocol["seeds"]["experiment_4_expanding_window"][0]
    feature_names = protocol["feature_sets"]["baseline_18"]
    train_df = by_split["train"]

    # one eligible row-set, shared by all 3 agents (confirmed: 'complete' is
    # per-episode, not per-label -- see targets.build_targets)
    _, y_direction, ts = make_xy_custom_features(dataset, train_df, "direction_label", feature_names)
    ordered = train_df[train_df["complete"] & (train_df["direction_label"] != "UNKNOWN")].sort_values("decision_date").reset_index(drop=True)
    n = len(ordered)
    assert n == len(y_direction), "eligible row count mismatch between targets table and make_xy"

    # episodes are already non-overlapping, consecutive weekly windows
    # (ExpiryCalendar.weekly_episodes partitions trading dates by expiry); a
    # direct check that no two eligible episodes' mark windows overlap:
    exit_dates = ordered["exit_date"].tolist()
    decision_dates = ordered["decision_date"].tolist()
    for i in range(1, n):
        assert decision_dates[i] > exit_dates[i - 1], (
            f"overlapping episode horizons at index {i}: decision {decision_dates[i]} <= prior exit {exit_dates[i-1]}"
        )
    print(f"  confirmed {n} eligible TRAIN episodes are non-overlapping and chronologically ordered")

    folds = build_expanding_window_folds(n, n_folds=4, initial_fraction=0.4)
    out = {"n_eligible_train_episodes": n, "n_folds": len(folds), "folds": {}}

    feats_by_date = dataset.features.set_index(dataset.features["date"].dt.date)

    for agent, (label_col, classes) in AGENT_CLASSES.items():
        _, y_full, _ = make_xy_custom_features(dataset, train_df, label_col, feature_names)
        X_full = feats_by_date.loc[ordered["decision_date"], feature_names].reset_index(drop=True)
        fold_results = []
        for i, (tr_start, tr_end, te_start, te_end) in enumerate(folds):
            X_tr, y_tr = X_full.iloc[tr_start:tr_end], y_full.iloc[tr_start:tr_end]
            X_te, y_te = X_full.iloc[te_start:te_end], y_full.iloc[te_start:te_end]
            fold_info = {
                "fold": i,
                "train_date_range": [str(decision_dates[tr_start]), str(decision_dates[tr_end - 1])],
                "test_date_range": [str(decision_dates[te_start]), str(decision_dates[te_end - 1])],
                "train_n": len(X_tr), "test_n": len(X_te),
                "train_class_distribution": {c: int((y_tr == c).sum()) for c in classes},
                "test_class_distribution": {c: int((y_te == c).sum()) for c in classes},
            }
            try:
                res = fit_predict_xgb(X_tr, y_tr, X_te, classes, params, seed)
                majority_class = y_tr.value_counts().idxmax()
                maj_acc = float((y_te == majority_class).mean())
                fold_info["eval"] = full_metrics("test_block", classes, y_te, res["val_pred"], res["val_proba"], maj_acc)
                fold_info["missing_classes_in_fit"] = res["missing_classes_in_fit"]
                fold_info["failure"] = None
            except Exception as exc:  # a fold is allowed to fail; it must not crash the run
                fold_info["eval"] = None
                fold_info["failure"] = f"{type(exc).__name__}: {exc}"
            fold_results.append(fold_info)
        out["folds"][agent] = fold_results
        accs = [f["eval"]["accuracy"] for f in fold_results if f["eval"]]
        print(f"  {agent}: fold accuracies = {[round(a, 3) for a in accs]} "
              f"(failures: {sum(1 for f in fold_results if f['failure'])})")

    out["limitations"] = (
        f"Each fold's test block has only ~{(n - int(n*0.4)) // 4} episodes and the initial training "
        f"window only ~{int(n*0.4)} -- both far too small for fold-to-fold metric differences to be "
        "treated as anything beyond a rough, high-variance sanity check. This is consistent with "
        "docs/EXPERIMENTS.md's documented sample-size limitation (~186-194 TRAIN episodes total) and "
        "is not a substitute for the VALIDATION-based comparisons in Experiments 1-3."
    )
    registry["experiment_4_expanding_window"] = out


# --------------------------------------------------------------------------- #
def main() -> int:
    protocol = json.loads(PROTOCOL_PATH.read_text())
    verify_manifest(DEFAULT_ROOT, protocol["repository_state"]["dataset_manifest_fingerprint"])
    verify_baseline_hashes(protocol)

    provider = LocalParquetProvider(DEFAULT_ROOT)
    atm_iv_hist = pd.read_parquet(DEFAULT_ATM_IV)
    dataset = build_dataset(provider, "NIFTY", DATASET_START, DATASET_END, atm_iv_hist)
    by_split = split_targets(dataset)  # TEST rows exist in by_split['test'] but are NEVER read below

    registry: dict = {
        "protocol_id": protocol["protocol_id"],
        "git_head_at_run": git_head(),
        "dataset_manifest_fingerprint": protocol["repository_state"]["dataset_manifest_fingerprint"],
        "test_data_accessed": False,
        "dependency_versions": {
            "xgboost": xgb.__version__, "numpy": np.__version__, "pandas": pd.__version__,
            "python": platform.python_version(),
        },
        "commands_executed": ["python scripts/run_phase2_2_experiments.py"],
        "eligible_sample_counts": {
            split: {
                agent: eligibility_report(dataset, by_split[split], label_col, agent).to_dict()
                for agent, (label_col, _) in AGENT_CLASSES.items()
            }
            for split in ("train", "validation")
        },
    }

    run_experiment_1(dataset, by_split, protocol, registry)
    run_experiment_2(dataset, by_split, protocol, registry)
    run_experiment_3(dataset, by_split, protocol, registry)
    run_experiment_4(dataset, by_split, protocol, registry)

    OUT_REGISTRY.write_text(json.dumps(registry, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_REGISTRY}")
    write_markdown_report(registry, protocol)
    print(f"wrote {OUT_REPORT}")
    return 0


def write_markdown_report(registry: dict, protocol: dict) -> None:
    lines = ["# Phase 2.2 Experiment Report", ""]
    lines.append(f"Protocol: `{registry['protocol_id']}` (data/models/phase2_2_protocol.json)")
    lines.append(f"Git HEAD at run: `{registry['git_head_at_run']}`")
    lines.append(f"Dataset manifest fingerprint: `{registry['dataset_manifest_fingerprint']}`")
    lines.append(f"TEST data accessed: **{registry['test_data_accessed']}**")
    lines.append("")

    lines.append("## Experiment 1: feature redundancy ablation (`dist_sma20` removed)")
    for agent, a in registry["experiment_1_feature_ablation"].items():
        c = a["comparison"]
        lines.append(f"- **{agent}**: diff={c['val_accuracy_diff_ablation_minus_baseline']:+.3f}, "
                      f"noise floor={c['cross_seed_noise_floor']:.3f} -> {c['verdict']}")
    lines.append("")

    lines.append("## Experiment 2: model-capacity comparison")
    lines.append("| agent | baseline VAL acc | shallow VAL acc | dummy-majority VAL acc | persistence VAL acc |")
    lines.append("|---|---|---|---|---|")
    for agent, a in registry["experiment_2_model_capacity"].items():
        b = a["baseline_xgb"]["val_accuracy_across_seeds"]
        s = a["shallow_xgb"]["val_accuracy_across_seeds"]
        d = a["dummy_majority"]["val_eval"]["accuracy"]
        p = a["persistence"]["accuracy"]
        lines.append(f"| {agent} | {b['mean']:.3f}±{b['std']:.3f} | {s['mean']:.3f}±{s['std']:.3f} | {d:.3f} | {p:.3f} |")
    lines.append("")

    lines.append("## Experiment 3: targeted class weighting")
    for agent, r in registry["experiment_3_class_weighting"].items():
        u, w = r["val_accuracy_across_seeds"] if False else (r["unweighted"]["val_accuracy_across_seeds"], r["weighted"]["val_accuracy_across_seeds"])
        tr = r["target_class_recall_across_seeds"]
        target = "NEUTRAL" if agent == "volatility" else "TRENDING_UP"
        lines.append(f"- **{agent}** (target class `{target}`): accuracy unweighted={u['mean']:.3f}±{u['std']:.3f} "
                      f"weighted={w['mean']:.3f}±{w['std']:.3f}; `{target}` recall unweighted={tr['unweighted']['mean']:.3f} "
                      f"weighted={tr['weighted']['mean']:.3f}")
        lines.append(f"  - frozen weights: {r['frozen_weights']}")
    lines.append("")

    lines.append("## Experiment 4: expanding-window evaluation within TRAIN")
    exp4 = registry["experiment_4_expanding_window"]
    lines.append(f"- {exp4['n_eligible_train_episodes']} eligible TRAIN episodes, {exp4['n_folds']} folds")
    for agent, folds in exp4["folds"].items():
        accs = [f["eval"]["accuracy"] if f["eval"] else None for f in folds]
        lines.append(f"  - **{agent}** fold accuracies: {accs}")
    lines.append(f"- Limitations: {exp4['limitations']}")
    lines.append("")

    lines.append("## Reproducibility")
    lines.append(f"- Dependency versions: {registry['dependency_versions']}")
    lines.append(f"- Commands executed: {registry['commands_executed']}")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
