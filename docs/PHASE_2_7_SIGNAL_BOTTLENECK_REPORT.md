# Phase 2.7 — Signal Bottleneck Diagnosis and Target Redesign

Diagnostic research only. No Decision Engine, Trade Constructor, Risk Engine, EOD Simulator, PPO, or RL code was added. All artifacts under `data/models/phase2_7/`; the original, Phase 2.5, and Phase 2.6 artifacts are untouched (verified byte-identical, Section 11 below).

**Headline finding, stated up front: across every target redesign, every horizon, every feature family, and every model family tested, no configuration shows walk-forward balanced accuracy meaningfully above chance, and every regression variant has a non-positive R². The dominant, well-evidenced conclusion is `ACCEPT CURRENT LIMITATION` for predictive accuracy on all three agents — but the investigation also found and documents a genuine, fixable target-engineering defect in the Volatility/Regime thresholds that is worth correcting on its own merits, independent of whether it changes accuracy (it does not).**

## 1. Reproduced reference metrics

`scripts/train_agents_extended.py --seed 0` re-run and matched exactly: Direction 0.676/0.676 (majority), Volatility 0.588/0.588, Regime 0.471/0.412, all n=34. Confirmed against the same model artifacts, `pointintime_v2` feature schema, and `targets_v1` rule used in Phase 2.5.

## 2. Phase A/B — Target ambiguity audit (quantified)

**Direction:** 16.7% of TRAIN forward-returns sit within 20% of the ±0.01157 threshold — present, but modest, ambiguity. `forward_return` describe: mean 0.0023, std 0.0232, median 0.0043 (n=305).

**Volatility — the clearest, best-evidenced finding of this phase:** the TRAIN `vol_ratio` median is **0.697**, already below the 0.9 "contraction" threshold. **69.7% of all episodes are CONTRACT purely because the fixed 0.9/1.1 band is not centered on the TRAIN distribution's own middle** — not because of pointwise label noise. This is a quantified target-design defect, not a vague impression.

**Regime:** **62.3% of all regime labels are decided purely by the volatility-extreme branch** of the priority rule (ratio `>1.3` or `<0.7`), never reaching the trend branches at all — confirming the task's suspicion that the label is "disproportionately driven by a single component." Since that component (volatility) inherits the same distributional mismatch just described, `LOW_VOLATILITY` alone claims ~50% of all regime labels.

No new leakage was found (point-in-time and cross-split assertions re-verified, all PASS, inside `scripts/phase27_experiments.py::build_extended_dataset` — reused unchanged from Phase 2.6).

## 3. Phase B — Underfitting vs. overfitting diagnostics

| Agent | In-sample acc | Walk-forward acc | Walk-forward bal. acc | Gap | Reading |
|---|---|---|---|---|---|
| Direction | 0.707 | 0.540 | 0.333 | 0.167 | large gap → overfitting/temporal instability |
| Volatility | 0.790 | 0.653 | 0.376 | 0.137 | raw accuracy gap looks smaller, **but this is a confound**: 0.653 walk-forward accuracy is inflated by the same majority-collapse described above (balanced accuracy 0.376 tells the real story — near chance) |
| Regime | 0.677 | 0.460 | 0.212 | 0.217 | large gap → overfitting/temporal instability |

**Caveat on this table, stated explicitly per the task's own warning:** raw accuracy is not a safe diagnostic when classes are this imbalanced. Balanced accuracy is the metric that matters here, and by that measure none of the three agents show any walk-forward performance meaningfully above chance (chance ≈0.33 for 3-class, ≈0.20 for 5-class) — a *large train/walk-forward gap with a low walk-forward floor* is consistent with **both** overfitting *and* an underlying lack of signal; this diagnostic alone cannot separate the two. Section 5 (target redesign) and Section 7 (feature ablation) are what actually disambiguate it.

## 4. Phase C — Controlled target redesign (separately versioned, TRAIN-fit)

All thresholds below are fit on TRAIN only and frozen before touching VALIDATION. **Majority baselines are VALIDATION's own distribution** — a real bug was caught and fixed in this task where an earlier draft accidentally checked against TRAIN's majority class, which would have overstated several of these results; see Section 11.

| Experiment | Definition | Walk-forward | VALIDATION | VAL majority | Beats majority? |
|---|---|---|---|---|---|
| `direction_binary_median_split_v1` | UP/DOWN at TRAIN median forward_return | bal_acc 0.547 | acc 0.618 | 0.676 | **No** |
| `direction_regression_forward_return_v1` | continuous forward_return, linear regression | R²=-0.260 | R²=-0.523, MAE 0.0143 vs. naive-mean MAE 0.0114 | — | **No — worse than predicting the TRAIN mean for every row** |
| `volatility_tercile_v1` | TRAIN-fit terciles on vol_ratio (exactly balanced 100/100/100 classes) | bal_acc 0.438 | acc 0.353 | 0.441 | **No** |
| `volatility_regression_vol_ratio_v1` | continuous vol_ratio, linear regression | R²=-0.036 | R²=-0.086 | — | **No** |
| `regime_decomposed_is_high_vol_v1` | binary HIGH_VOL vs. not | bal_acc 0.495 | acc 0.941 | 0.941 | Ties (extreme VAL class imbalance: 2/34 positive) |
| `regime_decomposed_is_trending_up_v1` | binary TRENDING_UP vs. not | bal_acc 0.533 | acc 0.794 | 0.824 | **No** |

**Interpretation — and the key nuance of this phase:** the tercile redesign *does* fix the engineering defect found in Section 2 (classes are exactly balanced by construction, 100/100/100 on TRAIN, instead of 70/11/20). **It still does not improve predictive accuracy.** Combined with the near-zero/negative regression R² for both continuous targets, this is strong evidence that the bottleneck is not merely how the target is bucketed — the underlying continuous quantities (forward return, forward/baseline vol ratio) have close to zero *linear* relationship with the current point-in-time features at all. Decomposing Regime into its components doesn't help either; each component underperforms its own (correctly-computed) majority baseline.

## 5. Phase D — Horizon experiments (non-overlapping, direction)

Sampled every `horizon` trading days (no two labels share any part of their forecast window — the task's explicit requirement), median-split UP/DOWN fit on the sampled TRAIN window, minimal 4-feature set (`ret_1/5/10/20`), 5-fold walk-forward:

| Horizon | Non-overlapping samples | Walk-forward balanced accuracy |
|---|---|---|
| 1 day | 1,432 | 0.508 |
| 3 days | 477 | 0.491 |
| 5 days | 286 | 0.498 |
| 10 days | 143 | 0.445 |

**All four horizons sit at or below chance (0.50) for a binary target.** The weekly (≈5-10 day) horizon is not uniquely hard — shortening to daily doesn't help either, and the 10-day horizon is actually worse (smaller non-overlapping sample, as expected). **The horizon is not the dominant bottleneck**; this rules out "a more appropriate prediction horizon" as the fix.

## 6. Phase E — Feature-family ablations

Six predeclared families (full 17, returns/momentum-only, realized-vol-only, IV-only, regime-only, returns+vol combined), walk-forward balanced accuracy, all three agents:

- **Direction:** 0.312–0.361 across all six families — no family stands out; combining returns+vol (0.361) is marginally highest but within noise of the rest.
- **Volatility:** 0.346–0.403 — IV-only (0.400) and returns+vol (0.403) are the highest, suggesting IV carries slightly more information than realized-vol-only (0.346) for this target, but the gap is small and not confirmed by the regression R² test (still ≈0).
- **Regime:** 0.195–0.227 — uniformly near or below 5-class chance (0.20); IV-only is actually the *worst* (0.195).

**No feature family shows a decisive advantage for any agent.** Adding more features (the full 17) is never clearly better than a smaller, more interpretable subset — consistent with Phase 2.6's finding and the task's own caution.

## 7. Phase F — Bounded model comparison

Reuses Phase 2.6's full grid (XGBoost × 3 configs, Logistic Regression, Random Forest — see `docs/PHASE_2_6_AGENT_INTELLIGENCE_EXPERIMENT_REPORT.md`). **LightGBM isolated smoke test**, run exactly once on a minimal synthetic 50-row dataset (per the task's explicit instruction not to repeatedly invoke a crashing model): **still fails** with the same native `OSError: access violation` seen in Phase 2.6, confirming it is a genuine, reproducible environment/wheel defect, not an artifact of the real dataset's size or shape. LightGBM remains excluded from all conclusions; XGBoost, Logistic Regression, and Random Forest already span the relevant model families.

## 8. Phase G — Historical data quantity and information limits

This mostly reconfirms, rather than overturns, Phase 2.3-2.5's findings:

- The verified weekly-compatible extension window (2019-02-18..2021-03-31) is already incorporated into the EXTENDED TRAIN used throughout this phase (305 episodes, up from 194).
- **Distinguishing a shortage of independent episodes from a shortage of informative features (the task's explicit ask):** Section 4's regression tests are the decisive evidence here. If the bottleneck were purely sample size (an estimation-variance problem), we would expect R² to be noisy but centered near some small positive value; instead every regression variant is clearly negative, which is the signature of **a model that would be *harmed* by fitting a relationship that isn't really there** — consistent with a genuine absence of (linear, point-in-time, price/IV-derived) signal, not merely insufficient data to estimate a real one precisely. This does not *prove* more data would never help, but it means more of the *same kind* of data (more weekly episodes, same feature families) is not the clearly indicated next step; the honest read is that it is unlikely to be transformative on its own.
- No further historical extension was attempted in this phase (none was justified by the evidence above).

## 9. Does the evidence support each proposed bottleneck?

| Candidate bottleneck | Supported? | Evidence |
|---|---|---|
| Target definition ambiguity | **Partially** — a real, quantified threshold/distribution mismatch exists for Volatility/Regime (Section 2), but fixing it (Section 4) does not improve accuracy | Sections 2, 4 |
| Prediction horizon | **Not supported** — every horizon from 1 to 10 days sits at or below chance | Section 5 |
| Feature information content | **Not strongly supported** — no family shows a decisive edge, and continuous regression R² is ≈0 or negative everywhere | Sections 4, 6 |
| Model fitting/generalization | **Partially** — large train/walk-forward gaps for Direction/Regime are consistent with some overfitting, but fixing capacity (Phase 2.6's shallow/regularized configs) didn't help either | Section 3, cross-referenced to Phase 2.6 |
| Historical data quantity | **Weakly supported at best** — negative R² is more consistent with an information ceiling than an estimation-variance problem | Section 8 |

## 10. Best candidate per agent, and 90% feasibility

**No candidate from this phase or Phase 2.6 is recommended as better than the existing Phase 2.5 baseline.** For every agent, the best-validated configuration remains `data/models/extended_2019_2021/` (Direction: weekly 3-class direction, XGBoost baseline config; Volatility: same; Regime: same). None of the redesigned targets, horizons, or feature families beat it.

**90% is not approached by any candidate tested in Phase 2.6 or 2.7.** The highest VALIDATION accuracy seen anywhere in this project remains Direction's 0.676 — which is a majority-class tie, not learned skill. No result here is a selection-bias artifact of repeated VALIDATION tuning: every Stage D/E/F candidate was evaluated via walk-forward *within TRAIN only*, and VALIDATION was touched exactly once per candidate (for the Phase 2.6 "selected" configs) or not at all (for most Phase 2.7 diagnostic variants, which report walk-forward only).

## 11. Final decision per agent

- **Direction: `ACCEPT CURRENT LIMITATION`.** No target variant, horizon, feature family, or model beats chance-level walk-forward balanced accuracy; regression R² is clearly negative.
- **Volatility: `ACCEPT CURRENT LIMITATION`** for predictive accuracy, **with a documented secondary recommendation to adopt the TRAIN-fit tercile threshold as a future target-schema version** purely for correctness (it fixes a real, quantified 70/11/20 imbalance caused by a threshold that doesn't match the TRAIN distribution) — adopt this for interpretability/engineering soundness, not because it improves accuracy (it doesn't, Section 4).
- **Regime: `ACCEPT CURRENT LIMITATION`.** The component-decomposition finding (62.3% volatility-branch-driven) explains *why* the label collapses onto `LOW_VOLATILITY`, but neither the decomposed binary components nor the rebalanced volatility component show real predictive accuracy.

## 12. Remaining uncertainty and the next highest-value experiment

- The negative-R² evidence is suggestive, not conclusive, about a true information ceiling vs. an estimation-variance problem at n≈300; a materially larger, independently-sourced dataset (not just more of the same weekly cadence) could in principle still reveal something — but nothing in this project's own evidence currently justifies that specific, expensive next step.
- **The single most promising unexplored lever, not tested in this phase, is genuinely new information sources** beyond price/IV-derived features already in `pointintime_v2` — e.g. market-breadth, cross-asset, or order-flow signals not currently in the data contract at all. This was outside this phase's bounded feature-family list (which only re-sliced existing columns) and would require new data engineering, not more modeling.
- Given the consistent, convergent null result across five independent lines of attack (targets, horizons, features, models, train/walk-forward diagnostics), the most defensible next step before any further agent-tuning is a hard stop on this axis: proceed to Phase 3 with these agents' outputs explicitly labeled as weak, unconfirmed priors, and let the trading engine's own rule-based baselines be the real test of whether swarm signals add value at the portfolio level — which is a different question than whether any one agent beats a classroom accuracy bar in isolation.

## 13. Test suite

`pytest` → **273 passed, 0 failed** (265 prior + 8 new Phase 2.7 tests covering point-in-time correctness of the horizon sampler, non-overlapping-sample verification, TRAIN-only median/tercile fitting, target-version separation, the majority-baseline bug regression guard, and the LightGBM failure-handling contract via a safe mock, not a repeated real crash).

## 14. Final verification

- Original dataset fingerprint `653ccb70c795d86f`: **unchanged**.
- All 8 Phase 2.2 baseline artifact hashes: **unchanged**.
- `data/models/extended_2019_2021/training_report.json` (Phase 2.5 reference): **unchanged** (Direction VAL accuracy re-read as 0.6765, matching exactly).
- `data/models/phase2_6/phase26_experiment_registry.json` (Phase 2.6): **unchanged**, read-only.
- TEST: **never accessed** at any point in this task.
- No PPO, Decision Engine, Trade Constructor, Risk Engine, or EOD Simulator code was added.
