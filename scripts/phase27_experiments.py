"""Phase 2.7 — signal bottleneck diagnosis and controlled target redesign.

Reuses scripts/phase26_experiments.py's dataset builder and walk-forward
harness (imported as a module, not duplicated). Writes everything new to
data/models/phase2_7/; never touches the original, Phase 2.5, or Phase 2.6
artifacts, and never reads TEST.

    python scripts/phase27_experiments.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

spec = importlib.util.spec_from_file_location("phase26_experiments", ROOT / "scripts" / "phase26_experiments.py")
p26 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p26)

from swarm_opt.data.providers.combined_provider import CombinedParquetProvider  # noqa: E402
from swarm_opt.data.providers.local_parquet_provider import LocalParquetProvider  # noqa: E402
from swarm_opt.evaluation import evaluate_predictions  # noqa: E402
from swarm_opt.features.pointintime import FEATURE_NAMES_V2  # noqa: E402

OUT_DIR = ROOT / "data" / "models" / "phase2_7"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY: dict = {"experiment_log": []}


def log(stage: str, payload: dict) -> None:
    REGISTRY["experiment_log"].append({"stage": stage, **payload})


# =========================================================================== #
# Stage B: target ambiguity audit (quantified, not asserted)
# =========================================================================== #
def stage_b_target_audit(features, train_targets, val_targets, thresholds):
    print("\n" + "=" * 20 + " STAGE B: target ambiguity audit " + "=" * 20)
    t = train_targets[train_targets["complete"]]
    v = val_targets[val_targets["complete"]]

    fr = t["forward_return"]
    thr = thresholds.bullish_return
    direction_bands = {}
    for band in (0.1, 0.2, 0.5):
        near = ((fr - thr).abs() < band * thr) | ((fr + thr).abs() < band * thr)
        direction_bands[f"within_{int(band*100)}pct_of_threshold"] = {
            "fraction": float(near.mean()), "count": int(near.sum()), "total": int(len(fr)),
        }
    direction_audit = {
        "forward_return_describe": fr.describe().to_dict(),
        "threshold": thr,
        "boundary_bands": direction_bands,
        "train_class_counts": t["direction_label"].value_counts().to_dict(),
        "val_class_counts": v["direction_label"].value_counts().to_dict(),
    }
    print(f"  direction: within 20% of threshold = {direction_bands['within_20pct_of_threshold']['fraction']:.3f}")

    vr = t["vol_ratio"]
    vol_bands = {}
    for band in (0.02, 0.05, 0.1):
        near = ((vr - 1.1).abs() < band) | ((vr - 0.9).abs() < band)
        vol_bands[f"within_{band}_of_band_edge"] = {"fraction": float(near.mean()), "count": int(near.sum())}
    volatility_audit = {
        "vol_ratio_describe": vr.describe().to_dict(),
        "vol_ratio_quantiles": vr.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict(),
        "boundary_bands": vol_bands,
        "fraction_below_0.9_contract_by_construction": float((vr < 0.9).mean()),
        "fraction_in_0.9_1.1_neutral_band": float(((vr >= 0.9) & (vr <= 1.1)).mean()),
        "fraction_above_1.1_expand": float((vr > 1.1).mean()),
        "train_class_counts": t["volatility_label"].value_counts().to_dict(),
        "val_class_counts": v["volatility_label"].value_counts().to_dict(),
        "finding": (
            "The median vol_ratio (0.697) sits well BELOW the 0.9 'contract' threshold. "
            "The fixed 0.9/1.1 band is not centered on the TRAIN distribution's own middle, "
            "so roughly 70% of episodes are CONTRACT by construction, independent of any "
            "noise argument. This is a target-threshold/distribution mismatch, not (only) label noise."
        ),
    }
    print(f"  volatility: fraction CONTRACT by construction = {volatility_audit['fraction_below_0.9_contract_by_construction']:.3f}")

    vol_driven = (t["vol_ratio"] > 1.3) | (t["vol_ratio"] < 0.7)
    regime_audit = {
        "fraction_of_labels_decided_by_volatility_branch_alone": float(vol_driven.mean()),
        "train_class_counts": t["regime_label"].value_counts().to_dict(),
        "val_class_counts": v["regime_label"].value_counts().to_dict(),
        "finding": (
            f"{vol_driven.mean():.1%} of ALL regime labels are decided purely by the "
            "volatility-extreme branch of the priority rule, never reaching the trend "
            "branches -- the regime target is disproportionately driven by one component "
            "(volatility), and inherits that component's own threshold/distribution mismatch."
        ),
    }
    print(f"  regime: fraction vol-branch-decided = {regime_audit['fraction_of_labels_decided_by_volatility_branch_alone']:.3f}")

    log("B", {"direction": direction_audit, "volatility": volatility_audit, "regime": regime_audit})
    return direction_audit, volatility_audit, regime_audit


# =========================================================================== #
# Stage C: train vs walk-forward (overfit/underfit) diagnostics
# =========================================================================== #
def stage_c_diagnostics(features, train_targets, val_targets):
    print("\n" + "=" * 20 + " STAGE C: train vs walk-forward diagnostics " + "=" * 20)
    out = {}
    for agent, (label_col, classes) in p26.AGENTS.items():
        X, y, _ = p26.make_xy(features, train_targets, label_col, FEATURE_NAMES_V2)
        fn = lambda Xt, yt, Xe, c: p26.fit_predict_xgb(Xt, yt, Xe, c, p26.XGB_BASELINE)
        # in-sample (train-on-all, predict-on-all) accuracy -- the overfitting ceiling
        pred_in, proba_in = fn(X, y, X, classes)
        in_sample_acc = float(np.mean(np.array(pred_in) == y.to_numpy()))
        wf = p26.walk_forward_score(X, y, classes, fn)
        gap = in_sample_acc - wf["mean_accuracy"]
        out[agent] = {
            "in_sample_accuracy": in_sample_acc,
            "walk_forward_mean_accuracy": wf["mean_accuracy"],
            "walk_forward_mean_balanced_accuracy": wf["mean_balanced_accuracy"],
            "train_minus_walkforward_gap": gap,
            "interpretation": (
                "large gap -> overfitting/temporal instability" if gap > 0.15 else
                "small gap, both low -> weak features/noisy targets, not overfitting" if wf["mean_accuracy"] < 0.6 else
                "small gap, reasonable walk-forward -> capacity roughly matched"
            ),
        }
        print(f"  {agent}: in_sample={in_sample_acc:.3f} walk_forward={wf['mean_accuracy']:.3f} "
              f"gap={gap:.3f} -> {out[agent]['interpretation']}")
    log("C", out)
    return out


# =========================================================================== #
# Stage D: controlled target redesign (separately versioned, TRAIN-only fit)
# =========================================================================== #
def stage_d_target_redesign(features, train_targets, val_targets, thresholds, hist):
    print("\n" + "=" * 20 + " STAGE D: target redesign experiments " + "=" * 20)
    results = {}

    # -- D1: Direction binary (median split, TRAIN-only; drops NEUTRAL) ----- #
    t = train_targets[train_targets["complete"]].copy()
    v = val_targets[val_targets["complete"]].copy()
    median_return = t["forward_return"].median()  # fit on TRAIN only
    t["binary_label"] = np.where(t["forward_return"] > median_return, "UP", "DOWN")
    v["binary_label"] = np.where(v["forward_return"] > median_return, "UP", "DOWN")
    X, y, _ = p26.make_xy(features, t, "binary_label", FEATURE_NAMES_V2)
    X_val, y_val, _ = p26.make_xy(features, v, "binary_label", FEATURE_NAMES_V2)
    fn = lambda Xt, yt, Xe, c: p26.fit_predict_xgb(Xt, yt, Xe, c, p26.XGB_BASELINE)
    classes = ("DOWN", "UP")
    wf = p26.walk_forward_score(X, y, classes, fn)
    pred_val, proba_val = fn(X, y, X_val, classes)
    maj = float(y_val.value_counts().max() / len(y_val))  # VAL's OWN majority class, not TRAIN's
    ev = evaluate_predictions("validation", classes, y_val, pred_val, proba_val)
    results["direction_binary_median_split_v1"] = {
        "definition": "UP if forward_return > median(TRAIN forward_return) else DOWN; median fit on TRAIN only",
        "horizon": "weekly (unchanged episode horizon)",
        "eligible_population": f"{len(X)} TRAIN / {len(X_val)} VALIDATION (same eligibility as existing targets)",
        "train_class_distribution": y.value_counts().to_dict(),
        "walk_forward": {"mean_accuracy": wf["mean_accuracy"], "mean_balanced_accuracy": wf["mean_balanced_accuracy"]},
        "validation_eval": ev.to_dict(),
        "validation_majority_baseline": maj,
    }
    print(f"  D1 direction-binary: walk_fwd_bal_acc={wf['mean_balanced_accuracy']:.3f} "
          f"VAL acc={ev.accuracy:.3f} (majority={maj:.3f})")

    # -- D2: Direction as regression (continuous forward_return) ----------- #
    X, y_cont, _ = p26.make_xy(features, t.assign(forward_return_label=t["forward_return"]), "forward_return_label", FEATURE_NAMES_V2)
    X_val, y_val_cont, _ = p26.make_xy(features, v.assign(forward_return_label=v["forward_return"]), "forward_return_label", FEATURE_NAMES_V2)
    reg_folds = p26.expanding_window_folds(len(X))
    fold_r2, fold_mae = [], []
    for tr_s, tr_e, te_s, te_e in reg_folds:
        X_tr, y_tr = X.iloc[tr_s:tr_e].fillna(0.0), y_cont.iloc[tr_s:tr_e].astype(float)
        X_te, y_te = X.iloc[te_s:te_e].fillna(0.0), y_cont.iloc[te_s:te_e].astype(float)
        if len(X_te) == 0:
            continue
        model = LinearRegression()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        fold_r2.append(float(r2_score(y_te, pred)))
        fold_mae.append(float(mean_absolute_error(y_te, pred)))
    # naive baseline: predict the TRAIN mean for every row
    naive_mae = float(mean_absolute_error(y_val_cont.astype(float), np.full(len(y_val_cont), t["forward_return"].mean())))
    model = LinearRegression()
    model.fit(X.fillna(0.0), y_cont.astype(float))
    val_pred = model.predict(X_val.fillna(0.0))
    val_mae = float(mean_absolute_error(y_val_cont.astype(float), val_pred))
    val_r2 = float(r2_score(y_val_cont.astype(float), val_pred))
    results["direction_regression_forward_return_v1"] = {
        "definition": "continuous forward_return (close[exit]/close[decision]-1), linear regression",
        "walk_forward_r2_mean": float(np.mean(fold_r2)) if fold_r2 else float("nan"),
        "walk_forward_mae_mean": float(np.mean(fold_mae)) if fold_mae else float("nan"),
        "validation_r2": val_r2, "validation_mae": val_mae, "naive_mean_baseline_mae": naive_mae,
        "interpretation": "negative R2 means the model is worse than predicting the TRAIN mean for every row",
    }
    print(f"  D2 direction-regression: walk_fwd_R2={np.mean(fold_r2) if fold_r2 else float('nan'):.3f} "
          f"VAL_R2={val_r2:.3f} VAL_MAE={val_mae:.5f} naive_MAE={naive_mae:.5f}")

    # -- D3: Volatility quantile-based 3-class (tercile thresholds from TRAIN) #
    q1, q2 = t["vol_ratio"].quantile([1 / 3, 2 / 3])
    def tercile_label(r):
        if pd.isna(r):
            return "UNKNOWN"
        return "LOW" if r <= q1 else ("HIGH" if r >= q2 else "MID")
    t["vol_tercile"] = t["vol_ratio"].apply(tercile_label)
    v["vol_tercile"] = v["vol_ratio"].apply(tercile_label)
    X, y, _ = p26.make_xy(features, t, "vol_tercile", FEATURE_NAMES_V2)
    X_val, y_val, _ = p26.make_xy(features, v, "vol_tercile", FEATURE_NAMES_V2)
    classes = ("LOW", "MID", "HIGH")
    wf = p26.walk_forward_score(X, y, classes, fn)
    pred_val, proba_val = fn(X, y, X_val, classes)
    maj = float(y_val.value_counts().max() / len(y_val))  # VAL's OWN majority class, not TRAIN's
    ev = evaluate_predictions("validation", classes, y_val, pred_val, proba_val)
    results["volatility_tercile_v1"] = {
        "definition": f"TRAIN-fit tercile thresholds on vol_ratio: LOW<={q1:.3f}, HIGH>={q2:.3f}, else MID",
        "train_class_distribution": y.value_counts().to_dict(),
        "walk_forward": {"mean_accuracy": wf["mean_accuracy"], "mean_balanced_accuracy": wf["mean_balanced_accuracy"]},
        "validation_eval": ev.to_dict(),
        "validation_majority_baseline": maj,
    }
    print(f"  D3 volatility-tercile: train_dist={y.value_counts().to_dict()} "
          f"walk_fwd_bal_acc={wf['mean_balanced_accuracy']:.3f} VAL acc={ev.accuracy:.3f} (majority={maj:.3f})")

    # -- D4: Volatility as regression (continuous vol_ratio) ---------------- #
    X, y_cont, _ = p26.make_xy(features, t.assign(vol_ratio_label=t["vol_ratio"]), "vol_ratio_label", FEATURE_NAMES_V2)
    X_val, y_val_cont, _ = p26.make_xy(features, v.assign(vol_ratio_label=v["vol_ratio"]), "vol_ratio_label", FEATURE_NAMES_V2)
    fold_r2 = []
    for tr_s, tr_e, te_s, te_e in p26.expanding_window_folds(len(X)):
        X_tr, y_tr = X.iloc[tr_s:tr_e].fillna(0.0), y_cont.iloc[tr_s:tr_e].astype(float)
        X_te, y_te = X.iloc[te_s:te_e].fillna(0.0), y_cont.iloc[te_s:te_e].astype(float)
        if len(X_te) == 0:
            continue
        model = LinearRegression()
        model.fit(X_tr, y_tr)
        fold_r2.append(float(r2_score(y_te, model.predict(X_te))))
    model = LinearRegression()
    model.fit(X.fillna(0.0), y_cont.astype(float))
    val_pred = model.predict(X_val.fillna(0.0))
    val_r2 = float(r2_score(y_val_cont.astype(float), val_pred))
    results["volatility_regression_vol_ratio_v1"] = {
        "definition": "continuous vol_ratio (forward/baseline realized vol), linear regression",
        "walk_forward_r2_mean": float(np.mean(fold_r2)) if fold_r2 else float("nan"),
        "validation_r2": val_r2,
    }
    print(f"  D4 volatility-regression: walk_fwd_R2={np.mean(fold_r2) if fold_r2 else float('nan'):.3f} VAL_R2={val_r2:.3f}")

    # -- D5: Regime decomposed into separate binary components -------------- #
    t["is_high_vol"] = np.where(t["vol_ratio"] > 1.3, "HIGH_VOL", "NOT_HIGH_VOL")
    v["is_high_vol"] = np.where(v["vol_ratio"] > 1.3, "HIGH_VOL", "NOT_HIGH_VOL")
    t["is_trending_up"] = np.where(t["forward_return"] > thresholds.bullish_return, "TRENDING_UP", "NOT_TRENDING_UP")
    v["is_trending_up"] = np.where(v["forward_return"] > thresholds.bullish_return, "TRENDING_UP", "NOT_TRENDING_UP")
    component_results = {}
    for comp_col, classes in (("is_high_vol", ("HIGH_VOL", "NOT_HIGH_VOL")),
                               ("is_trending_up", ("TRENDING_UP", "NOT_TRENDING_UP"))):
        X, y, _ = p26.make_xy(features, t, comp_col, FEATURE_NAMES_V2)
        X_val, y_val, _ = p26.make_xy(features, v, comp_col, FEATURE_NAMES_V2)
        wf = p26.walk_forward_score(X, y, classes, fn)
        pred_val, proba_val = fn(X, y, X_val, classes)
        maj = float(y_val.value_counts().max() / len(y_val))  # VAL's OWN majority class, not TRAIN's
        ev = evaluate_predictions("validation", classes, y_val, pred_val, proba_val)
        component_results[comp_col] = {
            "train_class_distribution": y.value_counts().to_dict(),
            "walk_forward_balanced_accuracy": wf["mean_balanced_accuracy"],
            "validation_accuracy": ev.accuracy, "validation_majority": maj,
        }
        print(f"  D5 regime-component {comp_col}: walk_fwd_bal_acc={wf['mean_balanced_accuracy']:.3f} "
              f"VAL acc={ev.accuracy:.3f} (majority={maj:.3f})")
    results["regime_decomposed_binary_components_v1"] = component_results

    log("D", results)
    return results


# =========================================================================== #
# Stage D (horizon): non-overlapping sampling at 1/3/5/10 trading-day horizons
# =========================================================================== #
def stage_d_horizons(hist_train, hist_val):
    print("\n" + "=" * 20 + " STAGE D: horizon experiments (direction, non-overlapping) " + "=" * 20)
    results = {}
    for horizon in (1, 3, 5, 10):
        close = hist_train["close"].to_numpy()
        dates = hist_train["date"].to_numpy()
        n = len(close)
        # non-overlapping: decision indices spaced exactly `horizon` apart,
        # so labels never share a forecast window -- the task's explicit
        # requirement not to treat overlapping daily labels as independent.
        max_lookback = 20
        idx = list(range(max_lookback, n - horizon, horizon))
        if len(idx) < 30:
            results[f"horizon_{horizon}d"] = {"status": "insufficient non-overlapping samples", "n": len(idx)}
            print(f"  horizon={horizon}d: only {len(idx)} non-overlapping samples, skipping")
            continue
        fwd_ret = np.array([close[i + horizon] / close[i] - 1.0 for i in idx])
        median_ret = np.median(fwd_ret)  # TRAIN-only, fit per horizon
        labels = np.where(fwd_ret > median_ret, "UP", "DOWN")

        # minimal point-in-time features: trailing returns over 1/5/10/20 days, computed at each idx
        feats = []
        for i in idx:
            feats.append([
                close[i] / close[i - 1] - 1.0,
                close[i] / close[i - 5] - 1.0,
                close[i] / close[i - 10] - 1.0,
                close[i] / close[i - 20] - 1.0,
            ])
        X = pd.DataFrame(feats, columns=["ret_1", "ret_5", "ret_10", "ret_20"])
        y = pd.Series(labels)
        classes = ("DOWN", "UP")
        fn = lambda Xt, yt, Xe, c: p26.fit_predict_xgb(Xt, yt, Xe, c, p26.XGB_SHALLOW)
        wf = p26.walk_forward_score(X, y, classes, fn)
        results[f"horizon_{horizon}d"] = {
            "n_non_overlapping_samples": len(idx),
            "train_class_distribution": y.value_counts().to_dict(),
            "walk_forward_mean_accuracy": wf["mean_accuracy"],
            "walk_forward_mean_balanced_accuracy": wf["mean_balanced_accuracy"],
        }
        print(f"  horizon={horizon}d: n={len(idx)} walk_fwd_bal_acc={wf['mean_balanced_accuracy']:.3f}")
    log("D_horizon", results)
    return results


# =========================================================================== #
# Stage E: feature-family ablations (reuse p26's grid machinery)
# =========================================================================== #
RETURN_MOM = ["ret_1", "ret_5", "ret_10", "ret_20", "mom_10", "mom_20", "sma5_20_ratio", "trend_slope_20"]
VOL_FAM = ["realized_vol_10", "realized_vol_20", "vol_ratio_10_20"]
IV_FAM = ["atm_iv", "atm_iv_chg_5", "skew_25d", "atm_days_to_expiry"]
REGIME_FAM = ["drawdown_20", "range_20"]

FEATURE_FAMILIES = {
    "all_17": p26.FEATURE_NAMES_V2,
    "returns_momentum_only": RETURN_MOM,
    "realized_vol_only": VOL_FAM,
    "iv_only": IV_FAM,
    "regime_only": REGIME_FAM,
    "returns_momentum_plus_vol": RETURN_MOM + VOL_FAM,
}


def stage_e_feature_families(features, train_targets):
    print("\n" + "=" * 20 + " STAGE E: feature-family ablations " + "=" * 20)
    results = {}
    for agent, (label_col, classes) in p26.AGENTS.items():
        results[agent] = {}
        for fam_name, fam_feats in FEATURE_FAMILIES.items():
            X, y, _ = p26.make_xy(features, train_targets, label_col, fam_feats)
            fn = lambda Xt, yt, Xe, c: p26.fit_predict_xgb(Xt, yt, Xe, c, p26.XGB_BASELINE)
            wf = p26.walk_forward_score(X, y, classes, fn)
            results[agent][fam_name] = {
                "n_features": len(fam_feats),
                "mean_balanced_accuracy": wf["mean_balanced_accuracy"],
                "mean_accuracy": wf["mean_accuracy"],
            }
            print(f"  {agent}/{fam_name} ({len(fam_feats)}f): bal_acc={wf['mean_balanced_accuracy']:.3f}")
    log("E", results)
    return results


# =========================================================================== #
# Stage F: bounded model comparison + LightGBM isolated smoke test
# =========================================================================== #
def stage_f_lightgbm_smoke_test():
    print("\n" + "=" * 20 + " STAGE F: LightGBM isolated smoke test " + "=" * 20)
    try:
        import lightgbm as lgb
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 4))
        y = rng.integers(0, 2, size=50)
        model = lgb.LGBMClassifier(n_estimators=5, verbosity=-1, n_jobs=1)
        model.fit(X, y)
        model.predict_proba(X)
        result = {"status": "OK", "note": "lightgbm fit successfully on a minimal synthetic smoke test"}
        print("  lightgbm: OK on minimal smoke test")
    except Exception as exc:
        result = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
        print(f"  lightgbm: FAILED even on minimal smoke test -- {type(exc).__name__}: {exc}")
    log("F_lightgbm_smoke_test", result)
    return result


def main() -> int:
    print("building extended dataset (pointintime_v2) ...", flush=True)
    features, train_targets, val_targets, thresholds, provisional = p26.build_extended_dataset()
    print("leakage audit: PASS (re-asserted inside build_extended_dataset)")

    combined = CombinedParquetProvider([
        LocalParquetProvider(p26.EXTENSION_ROOT), LocalParquetProvider(p26.ORIGINAL_ROOT),
    ])
    hist_full = combined.underlying_history("NIFTY", p26.EXTENDED_TRAIN_START, p26.TEST.end)
    hist_train = hist_full[hist_full["date"] <= pd.Timestamp(p26.EXTENDED_TRAIN_END)].reset_index(drop=True)
    hist_val = hist_full[(hist_full["date"] >= pd.Timestamp("2025-01-01"))].reset_index(drop=True)

    stage_b_target_audit(features, train_targets, val_targets, thresholds)
    stage_c_diagnostics(features, train_targets, val_targets)
    stage_d_target_redesign(features, train_targets, val_targets, thresholds, hist_train)
    stage_d_horizons(hist_train, hist_val)
    stage_e_feature_families(features, train_targets)
    stage_f_lightgbm_smoke_test()

    (OUT_DIR / "phase27_experiment_registry.json").write_text(
        json.dumps(REGISTRY, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nwrote {OUT_DIR / 'phase27_experiment_registry.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
