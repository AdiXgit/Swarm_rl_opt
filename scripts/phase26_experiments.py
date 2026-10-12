"""Phase 2.6 — bounded, pre-registered feature/model experiment matrix.

Loads the EXTENDED dataset (pointintime_v2, 17 features; see
docs/PHASE_2_5_EXTENDED_DATASET_AND_RETRAINING.md) exactly as
scripts/train_agents_extended.py builds it. For each of the three agents:

1. Runs a label/leakage audit (re-asserts the existing point-in-time and
   cross-split guarantees; reports class balance, boundary-margin stability).
2. Evaluates a small, predeclared grid of (feature set x model config)
   combinations via 5-fold expanding-window walk-forward CV *within
   EXTENDED TRAIN only*.
3. Selects the single best configuration per agent by walk-forward mean
   balanced accuracy (chosen once, not re-chosen after looking at
   VALIDATION), then fits it on the full EXTENDED TRAIN and evaluates it
   on VALIDATION exactly once.

Never touches TEST. Never overwrites the original or Phase 2.5 artifacts;
everything is written under data/models/phase2_6/.

    python scripts/phase26_experiments.py
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from swarm_opt.data.calendar import ExpiryCalendar  # noqa: E402
from swarm_opt.data.providers.combined_provider import CombinedParquetProvider  # noqa: E402
from swarm_opt.data.providers.local_parquet_provider import LocalParquetProvider  # noqa: E402
from swarm_opt.data.splits import TEST, VALIDATION  # noqa: E402
from swarm_opt.evaluation import evaluate_predictions  # noqa: E402
from swarm_opt.features.audit import (  # noqa: E402
    assert_duplicate_free,
    assert_feature_label_alignment,
    assert_no_future_features,
)
from swarm_opt.features.pointintime import FEATURE_NAMES_V2, FEATURE_SCHEMA_VERSION_V2, build_market_features  # noqa: E402
from swarm_opt.features.targets import TargetThresholds, build_targets, fit_thresholds_on_train  # noqa: E402

ORIGINAL_ROOT = ROOT / "data" / "processed" / "nifty_eod"
EXTENSION_ROOT = ROOT / "data" / "processed" / "nifty_eod_ext_2019_2021"
ORIGINAL_ATM_IV = ROOT / "data" / "features" / "atm_iv_history.parquet"
EXTENSION_ATM_IV = ROOT / "data" / "features" / "atm_iv_history_ext_2019_2021.parquet"
OUT_DIR = ROOT / "data" / "models" / "phase2_6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXTENDED_TRAIN_START = date(2019, 2, 18)
EXTENDED_TRAIN_END = date(2024, 12, 31)
SEED = 0
N_FOLDS = 5
INITIAL_FRACTION = 0.5

AGENTS = {
    "direction": ("direction_label", ("BEARISH", "NEUTRAL", "BULLISH")),
    "volatility": ("volatility_label", ("CONTRACT", "NEUTRAL", "EXPAND")),
    "regime": ("regime_label", ("TRENDING_DOWN", "TRENDING_UP", "RANGE_BOUND", "HIGH_VOLATILITY", "LOW_VOLATILITY")),
}

# ---- predeclared feature-set grid ----------------------------------------- #
IV_FEATURES = ["atm_iv", "atm_iv_chg_5", "skew_25d", "atm_days_to_expiry"]
RETURN_MOM = ["ret_1", "ret_5", "ret_10", "ret_20", "mom_10", "mom_20", "sma5_20_ratio", "trend_slope_20"]
VOL_FEATURES = ["realized_vol_10", "realized_vol_20", "vol_ratio_10_20"]
REGIME_FEATURES = ["drawdown_20", "range_20"]

FEATURE_SETS = {
    "all_17": FEATURE_NAMES_V2,
    "no_iv_13": [f for f in FEATURE_NAMES_V2 if f not in IV_FEATURES],
    "returns_momentum_vol_only_11": RETURN_MOM + VOL_FEATURES,
}
for name, feats in FEATURE_SETS.items():
    assert all(f in FEATURE_NAMES_V2 for f in feats), name

# ---- predeclared model-config grid ----------------------------------------- #
XGB_BASELINE = {"n_estimators": 50, "max_depth": 2, "learning_rate": 0.1, "subsample": 0.8,
                "colsample_bytree": 0.8, "reg_lambda": 2.0, "n_jobs": 1}
XGB_SHALLOW = {"n_estimators": 20, "max_depth": 1, "learning_rate": 0.1, "subsample": 0.8,
               "colsample_bytree": 0.8, "reg_lambda": 2.0, "n_jobs": 1}
XGB_DEEPER_REGULARIZED = {"n_estimators": 80, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.7,
                           "colsample_bytree": 0.7, "reg_lambda": 5.0, "n_jobs": 1,
                           "min_child_weight": 5}

try:
    import lightgbm as lgb
    HAVE_LGBM = True
except ImportError:
    HAVE_LGBM = False


def fit_predict_xgb(X_tr, y_tr, X_te, classes, params, seed=SEED, sample_weight=None):
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = y_tr.map(class_to_idx).to_numpy()
    dtrain = xgb.DMatrix(X_tr.to_numpy(dtype=float), label=y_idx, weight=sample_weight)
    bp = {"objective": "multi:softprob", "num_class": len(classes), "seed": seed,
          "max_depth": params["max_depth"], "eta": params["learning_rate"],
          "subsample": params["subsample"], "colsample_bytree": params["colsample_bytree"],
          "lambda": params["reg_lambda"], "nthread": params["n_jobs"]}
    if "min_child_weight" in params:
        bp["min_child_weight"] = params["min_child_weight"]
    booster = xgb.train(bp, dtrain, num_boost_round=params["n_estimators"])
    proba = booster.predict(xgb.DMatrix(X_te.to_numpy(dtype=float))).reshape(len(X_te), len(classes))
    pred = [classes[i] for i in proba.argmax(axis=1)]
    return pred, proba


def fit_predict_sklearn(model, X_tr, y_tr, X_te, classes):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr.fillna(0.0).to_numpy(dtype=float))
    X_te_s = scaler.transform(X_te.fillna(0.0).to_numpy(dtype=float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_tr_s, y_tr)
    proba_raw = model.predict_proba(X_te_s)
    full = np.zeros((len(X_te), len(classes)))
    for j, c in enumerate(model.classes_):
        full[:, list(classes).index(c)] = proba_raw[:, j]
    pred = [classes[i] for i in full.argmax(axis=1)]
    return pred, full


def fit_predict_lgbm(X_tr, y_tr, X_te, classes, seed=SEED):
    class_to_idx = {c: i for i, c in enumerate(classes)}
    model = lgb.LGBMClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.1, num_leaves=7,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, random_state=seed,
        verbosity=-1, n_jobs=1,
    )
    y_idx = y_tr.map(class_to_idx)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_tr.fillna(0.0), y_idx)
    proba_raw = model.predict_proba(X_te.fillna(0.0))
    full = np.zeros((len(X_te), len(classes)))
    for j, c in enumerate(model.classes_):
        full[:, c] = proba_raw[:, j]
    pred = [classes[i] for i in full.argmax(axis=1)]
    return pred, full


MODEL_CONFIGS = {
    "xgb_baseline": lambda X_tr, y_tr, X_te, classes: fit_predict_xgb(X_tr, y_tr, X_te, classes, XGB_BASELINE),
    "xgb_shallow": lambda X_tr, y_tr, X_te, classes: fit_predict_xgb(X_tr, y_tr, X_te, classes, XGB_SHALLOW),
    "xgb_deeper_regularized": lambda X_tr, y_tr, X_te, classes: fit_predict_xgb(X_tr, y_tr, X_te, classes, XGB_DEEPER_REGULARIZED),
    "logistic_l2": lambda X_tr, y_tr, X_te, classes: fit_predict_sklearn(
        LogisticRegression(penalty="l2", C=1.0, max_iter=2000, random_state=SEED), X_tr, y_tr, X_te, classes),
    "random_forest": lambda X_tr, y_tr, X_te, classes: fit_predict_sklearn(
        RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=5, random_state=SEED, n_jobs=1),
        X_tr, y_tr, X_te, classes),
}
if HAVE_LGBM:
    MODEL_CONFIGS["lightgbm"] = lambda X_tr, y_tr, X_te, classes: fit_predict_lgbm(X_tr, y_tr, X_te, classes)


def build_extended_dataset():
    ext_provider = LocalParquetProvider(EXTENSION_ROOT)
    orig_provider = LocalParquetProvider(ORIGINAL_ROOT)
    combined = CombinedParquetProvider([ext_provider, orig_provider])

    hist = combined.underlying_history("NIFTY", EXTENDED_TRAIN_START, TEST.end)
    atm_iv = pd.concat(
        [pd.read_parquet(EXTENSION_ATM_IV), pd.read_parquet(ORIGINAL_ATM_IV)], ignore_index=True
    ).sort_values("date").reset_index(drop=True)
    features, report = build_market_features(hist, atm_iv_hist=atm_iv, feature_names=FEATURE_NAMES_V2)

    dates = combined.trading_dates(EXTENDED_TRAIN_START, TEST.end)
    calendar = ExpiryCalendar.from_provider(combined, "NIFTY", dates)
    all_episodes = calendar.weekly_episodes()

    def extended_split_for(day):
        if EXTENDED_TRAIN_START <= day <= EXTENDED_TRAIN_END:
            return "extended_train"
        if VALIDATION.start <= day <= VALIDATION.end:
            return "validation"
        if TEST.start <= day <= TEST.end:
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

    assert_no_future_features(features)
    all_targets = pd.concat([train_targets, val_targets], ignore_index=True)
    assert_feature_label_alignment(features, all_targets)
    assert_duplicate_free(features, "date")
    assert_duplicate_free(all_targets, "decision_date")

    return features, train_targets, val_targets, thresholds, provisional


def make_xy(features, targets, label_col, feature_list):
    feats_by_date = features.set_index(features["date"].dt.date)
    target_ok = targets["complete"] & (targets[label_col] != "UNKNOWN")
    candidates = targets[target_ok]
    lookback = feats_by_date["has_full_lookback"].reindex(candidates["decision_date"])
    usable = candidates[lookback.fillna(False).to_numpy().astype(bool)]
    X = feats_by_date.loc[usable["decision_date"], feature_list].reset_index(drop=True)
    y = usable[label_col].reset_index(drop=True)
    ts = pd.Series(pd.to_datetime(usable["decision_date"].to_numpy()))
    return X, y, ts


def detect_collapse(pred: list[str], classes: tuple[str, ...], min_fraction: float = 0.05) -> list[str]:
    """Classes predicted less than `min_fraction` of the time -- the
    signature of a model that has collapsed onto one or two dominant
    classes rather than learning genuine multi-class structure. A high
    accuracy score alongside a non-empty result here is the "suspiciously
    high score" the task asks to investigate before trusting it."""
    n = len(pred)
    if n == 0:
        return list(classes)
    counts = pd.Series(pred).value_counts()
    return [c for c in classes if counts.get(c, 0) / n < min_fraction]


def expanding_window_folds(n, n_folds=N_FOLDS, initial_fraction=INITIAL_FRACTION):
    initial = int(n * initial_fraction)
    remainder = n - initial
    fold_size = remainder // n_folds
    folds, start = [], initial
    for i in range(n_folds):
        end = start + fold_size if i < n_folds - 1 else n
        folds.append((0, start, start, end))
        start = end
    return folds


def walk_forward_score(X, y, classes, fit_predict_fn):
    """Returns per-fold (balanced_accuracy, accuracy, n_test, failure) plus
    the mean/std balanced accuracy across folds that didn't fail."""
    folds = expanding_window_folds(len(X))
    results = []
    for tr_s, tr_e, te_s, te_e in folds:
        X_tr, y_tr = X.iloc[tr_s:tr_e], y.iloc[tr_s:tr_e]
        X_te, y_te = X.iloc[te_s:te_e], y.iloc[te_s:te_e]
        if y_tr.nunique() < 2 or len(X_te) == 0:
            results.append({"balanced_accuracy": None, "accuracy": None, "n_test": len(X_te), "failure": "degenerate fold"})
            continue
        try:
            pred, proba = fit_predict_fn(X_tr, y_tr, X_te, classes)
            ev = evaluate_predictions("fold", classes, y_te, pred, proba)
            results.append({"balanced_accuracy": ev.balanced_accuracy, "accuracy": ev.accuracy,
                             "n_test": len(X_te), "failure": None})
        except Exception as exc:
            results.append({"balanced_accuracy": None, "accuracy": None, "n_test": len(X_te),
                             "failure": f"{type(exc).__name__}: {exc}"})
    valid_bal = [r["balanced_accuracy"] for r in results if r["balanced_accuracy"] is not None and np.isfinite(r["balanced_accuracy"])]
    valid_acc = [r["accuracy"] for r in results if r["accuracy"] is not None]
    return {
        "folds": results,
        "mean_balanced_accuracy": float(np.mean(valid_bal)) if valid_bal else float("nan"),
        "std_balanced_accuracy": float(np.std(valid_bal)) if valid_bal else float("nan"),
        "mean_accuracy": float(np.mean(valid_acc)) if valid_acc else float("nan"),
        "n_valid_folds": len(valid_bal),
    }


def main() -> int:
    print("building extended dataset (pointintime_v2) ...", flush=True)
    features, train_targets, val_targets, thresholds, provisional = build_extended_dataset()
    print(f"  thresholds refit: {thresholds}")
    print("leakage audit: PASS (assert_no_future_features / alignment / duplicate-free)")

    registry: dict = {
        "thresholds": vars(thresholds) if hasattr(thresholds, "__dict__") else thresholds.__dict__,
        "feature_sets": {k: v for k, v in FEATURE_SETS.items()},
        "model_configs_grid": list(MODEL_CONFIGS.keys()),
        "lightgbm_available": HAVE_LGBM,
        "walk_forward": {"n_folds": N_FOLDS, "initial_fraction": INITIAL_FRACTION},
        "agents": {},
    }

    label_audit: dict = {}

    for agent, (label_col, classes) in AGENTS.items():
        print(f"\n{'='*20} {agent} {'='*20}", flush=True)
        X_full, y_full, ts_full = make_xy(features, train_targets, label_col, FEATURE_NAMES_V2)
        X_val_full, y_val_full, ts_val_full = make_xy(features, val_targets, label_col, FEATURE_NAMES_V2)

        # -- label quality: boundary-margin stability (direction/volatility only) -- #
        margin_stats = None
        if agent == "direction":
            fr = provisional["forward_return"].dropna()
            thr = thresholds.bullish_return
            near_boundary = ((fr - thr).abs() < 0.2 * thr) | ((fr + thr).abs() < 0.2 * thr)
            margin_stats = {"fraction_within_20pct_of_threshold": float(near_boundary.mean())}
        elif agent == "volatility":
            vr = train_targets["vol_ratio"].dropna()
            near_boundary = ((vr - thresholds.vol_expand_ratio).abs() < 0.05) | ((vr - thresholds.vol_contract_ratio).abs() < 0.05)
            margin_stats = {"fraction_within_0.05_of_ratio_threshold": float(near_boundary.mean())}

        label_audit[agent] = {
            "train_class_counts": {c: int((y_full == c).sum()) for c in classes},
            "val_class_counts": {c: int((y_val_full == c).sum()) for c in classes},
            "val_min_class_count": int(min((y_val_full == c).sum() for c in classes)),
            "boundary_margin_stats": margin_stats,
        }

        grid_results = {}
        for fs_name, fs_list in FEATURE_SETS.items():
            X, _, _ = make_xy(features, train_targets, label_col, fs_list)
            for model_name, fn in MODEL_CONFIGS.items():
                key = f"{fs_name}__{model_name}"
                wf = walk_forward_score(X, y_full, classes, fn)
                grid_results[key] = wf
                print(f"  {key}: mean_bal_acc={wf['mean_balanced_accuracy']:.3f} "
                      f"mean_acc={wf['mean_accuracy']:.3f} valid_folds={wf['n_valid_folds']}/{N_FOLDS}")

        best_key = max(grid_results, key=lambda k: grid_results[k]["mean_balanced_accuracy"]
                        if np.isfinite(grid_results[k]["mean_balanced_accuracy"]) else -1)
        best_fs_name, best_model_name = best_key.split("__")
        print(f"  >>> selected by walk-forward mean balanced accuracy: {best_key}")

        X_best, y_best, _ = make_xy(features, train_targets, label_col, FEATURE_SETS[best_fs_name])
        X_val_best, y_val_best, _ = make_xy(features, val_targets, label_col, FEATURE_SETS[best_fs_name])
        pred_val, proba_val = MODEL_CONFIGS[best_model_name](X_best, y_best, X_val_best, classes)
        majority_class = y_best.value_counts().idxmax()
        maj_val_acc = float((y_val_best == majority_class).mean())
        val_eval = evaluate_predictions("validation", classes, y_val_best, pred_val, proba_val)
        print(f"  FINAL (single VALIDATION look): acc={val_eval.accuracy:.3f} "
              f"bal_acc={val_eval.balanced_accuracy:.3f} majority={maj_val_acc:.3f}")

        registry["agents"][agent] = {
            "grid_results": grid_results,
            "selected_feature_set": best_fs_name,
            "selected_model": best_model_name,
            "selected_feature_list": FEATURE_SETS[best_fs_name],
            "validation_eval": val_eval.to_dict(),
            "validation_majority_baseline": maj_val_acc,
            "predicted_class_counts": {c: int(pd.Series(pred_val).value_counts().get(c, 0)) for c in classes},
            "collapsed_classes_on_validation": detect_collapse(pred_val, classes),
        }

    registry["label_audit"] = label_audit

    # Compare against the Phase 2.5 baseline (data/models/extended_2019_2021),
    # same VALIDATION dates, so a Phase 2.6 candidate is only ever described
    # as an improvement if it actually beats that reference, not just a
    # majority/persistence baseline.
    phase25_report_path = ROOT / "data" / "models" / "extended_2019_2021" / "training_report.json"
    if phase25_report_path.exists():
        phase25 = json.loads(phase25_report_path.read_text())
        comparison = {}
        for agent in AGENTS:
            p25 = phase25["agents"][agent]["val_eval"]
            p26 = registry["agents"][agent]["validation_eval"]
            comparison[agent] = {
                "phase25_baseline": {"accuracy": p25["accuracy"], "balanced_accuracy": p25["balanced_accuracy"],
                                      "log_loss": p25["log_loss"], "brier": p25["multiclass_brier"]},
                "phase26_selected": {"accuracy": p26["accuracy"], "balanced_accuracy": p26["balanced_accuracy"],
                                      "log_loss": p26["log_loss"], "brier": p26["multiclass_brier"]},
                "phase26_beats_phase25_on_balanced_accuracy": p26["balanced_accuracy"] > p25["balanced_accuracy"],
            }
        registry["comparison_vs_phase25_baseline"] = comparison
        print("\n=== Phase 2.6 vs Phase 2.5 baseline (same VALIDATION dates) ===")
        for agent, c in comparison.items():
            print(f"  {agent}: phase25 bal_acc={c['phase25_baseline']['balanced_accuracy']:.3f} "
                  f"phase26 bal_acc={c['phase26_selected']['balanced_accuracy']:.3f} "
                  f"beats_baseline={c['phase26_beats_phase25_on_balanced_accuracy']}")
    (OUT_DIR / "phase26_experiment_registry.json").write_text(
        json.dumps(registry, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nwrote {OUT_DIR / 'phase26_experiment_registry.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
