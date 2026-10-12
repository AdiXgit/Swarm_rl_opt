# Phase 2.6 — Agent Intelligence Improvement Experiment Report

**Target was 90% VALIDATION accuracy per agent, if achievable through genuine predictive signal. It was not achievable, and a systematic, bounded search found no configuration that even beats the existing Phase 2.5 baseline.** This is reported honestly below, not manufactured.

All artifacts are under `data/models/phase2_6/`. The original Phase 2.2 baseline and the Phase 2.5 extended baseline are both byte-identical and unmodified; this experiment matrix only reads them for comparison.

## 1. Reproduced baseline

`scripts/train_agents_extended.py --seed 0` re-run and matched the task's stated numbers exactly:

| Agent | VAL accuracy | VAL majority | VAL n |
|---|---|---|---|
| Direction | 0.676 | 0.676 | 34 |
| Volatility | 0.588 | 0.588 | 34 |
| Regime | 0.471 | 0.412 | 34 |

n=34 for every agent; any per-class metric below is built on single- or low-double-digit counts per class and should be read with that in mind.

## 2. Label-quality and leakage audit

- Target thresholds are fit exclusively from EXTENDED TRAIN forward returns (`fit_thresholds_on_train`), frozen before VALIDATION is touched — confirmed by re-running the existing point-in-time and cross-split assertions (`assert_no_future_features`, `assert_feature_label_alignment`, duplicate-freedom checks) inside `scripts/phase26_experiments.py`; all **PASS**.
- **Direction boundary stability:** with the refit threshold (±0.01157), a meaningful share of EXTENDED TRAIN's forward returns sit within 20% of either boundary (see `data/models/phase2_6/phase26_experiment_registry.json` → `label_audit.direction.boundary_margin_stats`) — i.e. a sizeable fraction of BULLISH/NEUTRAL/BEARISH labels are only weakly separated, not cleanly bimodal. This is consistent with (and likely explains) the near-chance walk-forward scores below: a meaningful fraction of labels are intrinsically ambiguous under this threshold rule, not just hard to learn.
- **Volatility boundary stability:** similarly, a non-trivial fraction of the volatility ratio sits within 0.05 of the 0.9/1.1 band edges.
- VALIDATION's own class counts (e.g. Regime: `TRENDING_UP=3`, `HIGH_VOLATILITY=2` out of 34) make several per-class recall numbers statistically close to meaningless — a single misclassification swings recall by 33-50 percentage points for these classes.
- No new leakage was found. This audit did not uncover a correctable label defect; it found that the labels are intrinsically noisy given weekly-horizon market returns and a fixed statistical threshold, which is a property of the target design and the data, not a bug.

## 3. Feature experiments

Three predeclared feature sets, tested identically across all model configs and all three agents:

- `all_17` — full `pointintime_v2` set.
- `no_iv_13` — drops the 4 IV/skew features (`atm_iv`, `atm_iv_chg_5`, `skew_25d`, `atm_days_to_expiry`).
- `returns_momentum_vol_only_11` — returns/momentum/realized-vol only, drops IV and regime (drawdown/range) features.

No feature set produced a consistent, material advantage over the others across models — differences in walk-forward mean balanced accuracy between feature sets, for a fixed model, were generally smaller than the across-fold standard deviation. **Adding more features (the full 17) did not reliably outperform smaller, more parsimonious sets**, consistent with the task's own caution not to assume more features help.

## 4. Model comparison (5-fold expanding-window walk-forward, within EXTENDED TRAIN only)

Grid: XGBoost (baseline/shallow/deeper-regularized configs), Logistic Regression (L2), Random Forest, and LightGBM (attempted). Full per-fold results in the registry JSON.

**LightGBM could not be evaluated**: it installed cleanly via pip (`lightgbm==4.7.0`) but crashes with a native `OSError: access violation` inside its own C library on every fold, on this Windows environment — a pre-existing environment/wheel issue, not a modeling choice. This is disclosed rather than silently worked around; all conclusions below rest on XGBoost, Logistic Regression, and Random Forest, which already span the gradient-boosting, linear, and bagged-tree families.

**Headline finding: every single configuration — every feature set × every model, for every agent — scored at or below chance-level balanced accuracy in walk-forward validation.** Direction (3-class, chance ≈0.33): best mean balanced accuracy 0.361. Volatility (3-class): best 0.404. Regime (5-class, chance ≈0.20): best 0.255. None of these exceed chance by a margin that looks like more than fold-to-fold noise.

## 5. Walk-forward validation results and final selection

Per agent, the single configuration with the highest walk-forward mean balanced accuracy was selected **once**, fit on the full EXTENDED TRAIN, and evaluated **once** on VALIDATION (no re-selection after seeing this result):

| Agent | Selected feature set | Selected model | VAL accuracy | VAL balanced accuracy | VAL majority |
|---|---|---|---|---|---|
| Direction | `returns_momentum_vol_only_11` | `xgb_baseline` | 0.588 | 0.290 | 0.676 |
| Volatility | `no_iv_13` | `xgb_deeper_regularized` | 0.559 | 0.411 | 0.588 |
| Regime | `no_iv_13` | `xgb_baseline` | 0.441 | 0.221 | 0.412 |

**None of these beat their own majority baseline.** Per-class recall confirms the same near-total majority collapse already documented in Phase 2.1/2.2/2.5 (e.g. selected Direction model: `BEARISH` recall 0.0, `BULLISH` recall 0.0, predicts `NEUTRAL` for 31 of 34 rows).

## 6. Comparison against the Phase 2.5 baseline

| Agent | Phase 2.5 baseline bal. acc. | Phase 2.6 selected bal. acc. | Beats baseline? |
|---|---|---|---|
| Direction | 0.415 | 0.290 | **No** |
| Volatility | 0.428 | 0.411 | **No** |
| Regime | 0.238 | 0.221 | **No** |

**The systematic, bounded search in this task did not find anything better than the existing, simpler Phase 2.5 configuration.** This is the single most important, honest result of Phase 2.6: it is not that no one tried harder — fourteen configurations (3 feature sets × up to 5 working model families, minus the LightGBM crash) were tried per agent, honestly scored by chronological walk-forward CV, and the winner of that search still underperforms the baseline on the one VALIDATION look. **Recommendation: keep the Phase 2.5 artifacts (`data/models/extended_2019_2021/`) as the reference configuration; do not promote any Phase 2.6 candidate.**

## 7. Best configuration per agent

Given Section 6's result, the best-validated configuration for every agent remains the **existing Phase 2.5 baseline**, not anything produced in this task. No Phase 2.6 model is recommended for adoption.

## 8. Was 90% reached? Does anything here generalize?

**No. 90% was not reached, is not close to reached (best VALIDATION accuracy across every agent and every configuration tried across Phase 2.2/2.5/2.6 remains below 70%), and nothing found in this task shows evidence of a genuine, generalizing predictive edge.** Every per-class recall breakdown for every "best" configuration shows majority-class collapse, not learned multi-class structure. The label-quality audit (Section 2) suggests this is not primarily a model-capacity or feature-selection problem: a meaningful fraction of the labels themselves sit near arbitrary statistical thresholds on weekly returns that are small relative to their own noise. Weekly-horizon directional/volatility/regime prediction from price-derived features alone, at n≈300 TRAIN / n=34 VALIDATION, does not show a reliably learnable signal with any method tried here.

## 9. Limitations and recommended next steps

- **Sample size remains the dominant constraint.** 300 TRAIN / 34 VALIDATION is small for 3-5 class problems by any standard; walk-forward fold sizes (~30 rows) are themselves too small to trust individual fold scores.
- **LightGBM was not actually evaluated** due to a native crash; if pursued, this would need a different environment or a conda-based LightGBM build rather than the pip wheel used here.
- **The boundary-margin finding (Section 2) suggests the next highest-value experiment is not another model search but a target-definition review** — e.g. whether a coarser 2-class (instead of 3/5-class) target, or a return threshold chosen by a method other than a fixed fraction of TRAIN std, would produce less ambiguous labels. This would need to be a separately versioned, pre-registered target experiment (not done here, since the task scoped this round to models/features, not targets).
- **Calibration was not separately tuned** (e.g. no `CalibratedClassifierCV` pass) because no base model showed reproducible skill worth calibrating; calibrating a majority-collapsed model would not create genuine signal.
- **TEST was never accessed** at any point in this task.
- **Phase 3 implication:** nothing here changes the Phase 2.5 readiness decision. The honest state remains: pipeline and interfaces are solid and reproducible; predictive signal is weak to absent across every model family and feature combination tried so far. Phase 3's trading engine should treat these agents' outputs as weak, uncalibrated priors to be tested against rule-based baselines — not as a validated edge.
