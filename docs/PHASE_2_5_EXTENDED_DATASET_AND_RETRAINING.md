# Phase 2.5 — Historical Extension, Retraining, and Phase 3 Readiness

Builds on `docs/HISTORICAL_EXTENSION_FEASIBILITY.md` (Phase 2.3) and `docs/HISTORICAL_PROBE_AND_EXPIRY_AUDIT.md` (Phase 2.4), both preserved unchanged. This document is the authoritative record of what was actually built, trained, and decided in Phase 2.5.

## 1. Historical coverage and verified expiry regimes

The weekly-options transition, left as a ~30-day window in Phase 2.4, is now pinned to a 7-day window by 4 additional targeted requests (`scripts/probe_transition_2019.py`, `data/raw/history_probe_transition_2019.json`):

| Date | Front expiry | Gap to next expiry | Classification |
|---|---|---|---|
| 2019-01-24 | 2019-01-31 | 28d | monthly |
| 2019-01-31 | 2019-01-31 (itself) | 28d | monthly |
| 2019-02-07 | 2019-02-28 | 28d | monthly |
| 2019-02-14 | 2019-02-14 (itself) | 7d | **weekly** |

**Conservative verified extension start: 2019-02-18** (the Monday after the first confirmed weekly sample, chosen to sit unambiguously inside the confirmed weekly regime rather than exactly at the transition boundary).

## 2. Extension dataset

Built with the existing, unmodified `scripts/build_dataset.py`, pointed at a new directory — no change to that script, no change to any original file:

```
python scripts/build_dataset.py --start 2019-02-18 --end 2021-03-31 --root data/processed/nifty_eod_ext_2019_2021
python scripts/build_market_features_extension.py   # ATM IV/skew cache for the extension
python scripts/audit_extension_dataset.py             # data-quality audit + manifest
```

- **524 requested trading dates, 523 written, 1 empty** (`2021-03-30`: confirmed genuine trading day via the index series, but NSE's reports-API fallback returned HTTP 404 on two separate attempts — not a holiday, not substituted, documented as a real gap).
- **Data-quality gate: PASS.** Zero duplicate primary keys, zero negative days-to-expiry, zero expired-in-chain rows, zero non-positive prices, zero invalid open interest. Missing-date fraction 0.19% (1/524), below the 1% materiality threshold used for the gate.
- Lot size `75` throughout; schema `legacy` throughout (consistent with the existing `SCHEMA_CHANGE = 2024-01-02` boundary).
- **Forward-source coverage, the metric that actually matters:** 0 of 3,123 near-expiry (≤45 DTE) expiries missing a forward — identical to the original dataset's own 0/7,925. The whole-chain "missing" forward rate (26.3% vs. the original's 6.4%) is concentrated entirely in far-dated, illiquid strikes the weekly pipeline never touches — reported for completeness, not a blocker.
- Extension content fingerprint: `1386411c71b25e19` (independent of, and never compared against, the original `653ccb70c795d86f`).
- Full manifest: `data/models/historical_extension_dataset_manifest.json`.

## 3. Combining with the frozen dataset

`backend/swarm_opt/data/providers/combined_provider.py::CombinedParquetProvider` is a new, additive `DataProvider` that routes each date to whichever of its (disjoint) member providers actually has it — a read-only union, not a merge, and not a modification of `LocalParquetProvider` or anything else. Verified that the extension (ends 2021-03-31) and the original dataset (starts 2021-04-01) are contiguous with no overlap.

**A real bug was caught and fixed before any training ran:** `swarm_opt.data.splits.split_for`/`purge_straddling` are frozen to the original TRAIN start (2021-04-01); reusing them against extension dates (all before that) would have resolved every one of them to `split_for() = None` and silently purged the entire extension. `scripts/train_agents_extended.py` defines its own parallel `extended_split_for`/`extended_purge_straddling`/`extended_assert_no_cross_split_labels`, scoped to that one script, leaving `swarm_opt/data/splits.py` completely untouched.

## 4. Feature-schema change (`pointintime_v2`)

`dist_sma20` (byte-for-byte identical to `mom_20`, confirmed in Phase 2.2's `phase2_2_experiment_registry.json`) is dropped from the canonical feature set for the new schema version. Implemented additively in `backend/swarm_opt/features/pointintime.py`:

- `FEATURE_NAMES` / `FEATURE_SCHEMA_VERSION` (`pointintime_v1`, 18 features) are **unchanged** — every existing Phase 2/2.1/2.2 artifact, report, and reproduction script still works exactly as before (verified: `scripts/run_phase2_diagnostics.py` still reproduces the frozen baseline to <1e-9 after this change).
- `FEATURE_NAMES_V2` / `FEATURE_SCHEMA_VERSION_V2 = "pointintime_v2"` (17 features) is new. `build_market_features()` gained an optional `feature_names` parameter (default `None` → unchanged v1 behavior) so both schemas share one implementation.

**17 `pointintime_v2` features:** `ret_1, ret_5, ret_10, ret_20, mom_10, mom_20, sma5_20_ratio, trend_slope_20, realized_vol_10, realized_vol_20, vol_ratio_10_20, drawdown_20, range_20, atm_iv, atm_iv_chg_5, skew_25d, atm_days_to_expiry`.

## 5. Extended chronological experiment

```
EXTENDED TRAIN  = 2019-02-18 .. 2024-12-31   (extension + original TRAIN, contiguous)
VALIDATION      = 2025-01-01 .. 2025-08-31   (ORIGINAL dates, unchanged)
TEST            = 2025-09-01 .. 2026-08-18   (ORIGINAL dates, never read by this phase)
```

Target **rule** and class order are unchanged (`targets_v1`). The direction threshold **value** was refit on the larger EXTENDED TRAIN, because the rule is defined as "0.5×std(TRAIN forward returns)" and TRAIN itself changed — this is the existing rule mechanically applied to new data, frozen before VALIDATION was evaluated, not a new rule:

| | Original (TRAIN 2021-04..2024-12) | Extended (TRAIN 2019-02..2024-12) |
|---|---|---|
| `bullish_return` / `bearish_return` | 0.00798 | 0.01157 |

**This changes VALIDATION's own ground-truth labels** for the same 34 dates (e.g. Direction's VALIDATION class counts shift from `{BEARISH:8, NEUTRAL:17, BULLISH:9}` to `{BEARISH:5, NEUTRAL:23, BULLISH:6}`), which is why the majority-baseline accuracy reported below differs from Phase 2.2's. **This is disclosed explicitly, not glossed over** — any comparison between the original and extended runs must account for it, per the task's own rule 4.

Split safeguards: all 10 required checks were run — `assert_no_future_features`, `assert_feature_label_alignment`, the extended cross-split check, duplicate-freedom on both features and targets, and the explicit `has_full_lookback` filter — all PASS, logged by `scripts/train_agents_extended.py` before any model was fit.

## 6. Episode counts (measured, not extrapolated)

| | Count |
|---|---|
| Weekly episodes, full extended range, before purge | 391 |
| Kept after purge | 389 |
| Purged at split walls | 2 (same two as the original: the TRAIN/VALIDATION and VALIDATION/TEST boundary episodes) |
| EXTENDED TRAIN episodes (2019-02-18..2024-12-31) | 305 |
| EXTENDED TRAIN eligible for fitting (after target/lookback exclusions) | **300** (5 excluded: incomplete target; 0 excluded: insufficient lookback) |
| VALIDATION episodes (unchanged) | 34, all eligible |
| **Net new episodes from the extension** | 305 − 194 (original TRAIN) = **111** |

This lands almost exactly inside Phase 2.4's pre-registered estimate band (95-120 additional episodes) — the estimate held up against the actual measured build.

## 7. Retraining results

```
python scripts/train_agents_extended.py --seed 0
```

| Agent | TRAIN n | TRAIN acc | VAL n | VAL acc | VAL majority | VAL persistence |
|---|---|---|---|---|---|---|
| Direction | 300 | 0.707 | 34 | 0.676 | 0.676 | 0.441 |
| Volatility | 300 | 0.790 | 34 | 0.588 | 0.588 | 0.588 |
| Regime | 300 | 0.677 | 34 | **0.471** | 0.412 | 0.441 |

Full confusion matrices, per-class precision/recall, balanced accuracy, log loss, and probability diagnostics are in `data/models/extended_2019_2021/training_report.json`.

**Honest reading, not a sales pitch:**
- **Direction and Volatility tie their own (shifted) majority baseline exactly.** Their confusion matrices show near-total collapse onto the majority class (Direction: predicts BEARISH 0 times; Volatility: predicts NEUTRAL 0 times, matching the exact collapse pattern already flagged in Phase 2.1/2.2). Tying majority accuracy here is not evidence of skill.
- **Regime is the one agent, in the entire Phase 2 line of experiments so far, that beats BOTH the majority-class baseline (+0.059) AND the persistence baseline (+0.030) on VALIDATION.** It is still weak (47% accuracy, 5 classes, balanced accuracy 0.238 — it still never predicts `TRENDING_UP`), and n=34 is far too small to call this anything more than a noteworthy, unconfirmed signal. It is reported as exactly that: noteworthy, not proven.
- No model is declared superior to anything without the comparison being shown above.

## 8. Inference pipeline validation

```
python scripts/build_swarm_state_extended.py --date 2019-06-10   # inside the extension
python scripts/build_swarm_state_extended.py --date 2025-06-05   # inside VALIDATION
```

Both produce a `SWARM_STATE_V1` that passes `validate_swarm_state()` with zero errors; `market_features` correctly has no `dist_sma20` key; running the same date twice byte-for-byte reproduces the same JSON; a date outside `[2019-02-18, 2026-08-18]` fails with a clear, explicit error (no exception traceback, no silent wrong answer); a feature frame missing a required v2 column raises `AgentError` rather than proceeding. See `tests/test_extended_pipeline_integration.py`.

## 9. Known limitations

- VALIDATION ground-truth labels differ from Phase 2.2's (Section 5) — any cross-phase comparison must account for this.
- n=34 VALIDATION episodes is unchanged and still far too small for confident conclusions; the Regime agent's improvement is noteworthy, not proven.
- The extension's single missing date (2021-03-30) means that one trading date's data is absent from the extended TRAIN; its effect is judged immaterial (1/524) but not zero.
- The `atm_iv_coverage` for the combined dataset is 99.95% (one date's IV row missing), consistent with the one missing raw trading date.
- TEST was not evaluated in this phase and remains the only genuinely held-out check; nothing here should be read as a TEST-validated result.

## 10. Reproduction commands

```bash
# 1. Pin the transition date (4 bounded requests)
python scripts/probe_transition_2019.py

# 2. Build and audit the extension
python scripts/build_dataset.py --start 2019-02-18 --end 2021-03-31 --root data/processed/nifty_eod_ext_2019_2021
python scripts/build_market_features_extension.py
python scripts/audit_extension_dataset.py

# 3. Retrain on EXTENDED TRAIN, evaluate on unchanged VALIDATION
python scripts/train_agents_extended.py --seed 0

# 4. Validate inference
python scripts/build_swarm_state_extended.py --date 2019-06-10
python scripts/build_swarm_state_extended.py --date 2025-06-05

# 5. Tests
pytest
```

## 11. Phase 3 readiness decision

**Ready to proceed criteria, checked explicitly:**
- Extended dataset reproducible and passes quality audits — **yes** (Section 2).
- Frozen benchmark unchanged — **yes**, fingerprint `653ccb70c795d86f` and all 8 Phase 2.2 baseline artifact hashes verified byte-identical at both the start and end of this task.
- Regression and integration tests pass — **yes**, 254/254 (240 prior + 14 new).
- All three agents expose stable, versioned interfaces — **yes**, unchanged `XGBClassifierAgent` contract, now also exercised against a second feature schema (v2) without modification.
- `SWARM_STATE_V1` reproducible and validated — **yes** (Section 8).
- Metrics and baselines documented — **yes** (Section 7), including the honest result that 2 of 3 agents still show no real skill.
- Weak/unproven signals explicitly identified — **yes**: Direction/Volatility majority-collapse, Regime's modest and unconfirmed improvement.

**Decision: READY TO PROCEED to Phase 3**, on the terms Section 10 of the Phase 2.5 prompt itself sets: predictive accuracy does not have to be excellent, and it is not. The pipeline, interfaces, and contracts are correct, reproducible, and now proven to hold up under more data and a second feature schema. The honest state of the specialist signals (one modestly promising, two still weak) should be carried into Phase 3 as something the trading engine tests against simple rule-based baselines, not something to assume away.
