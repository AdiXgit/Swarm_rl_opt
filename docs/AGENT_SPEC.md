# Agent Specification

## Status (Phase 2.5, complete)
An extended-dataset retraining exists alongside the original Phase 2 baseline
(never overwritten): `pointintime_v2` feature schema (17 features, `dist_sma20`
dropped) and `data/models/extended_2019_2021/` artifacts trained on an
EXTENDED TRAIN (2019-02-18..2024-12-31) that combines a separately-versioned
historical extension with the original TRAIN. See
`docs/PHASE_2_5_EXTENDED_DATASET_AND_RETRAINING.md` for full detail,
metrics, and the Phase 3 readiness decision.

## Status (Phase 2, complete)
Direction, Volatility and Regime are trained XGBoost classifiers. Greeks is
deterministic. No PPO/RL, Decision Engine, or Trade Constructor exists yet —
those remain out of scope (see `docs/RL_SPEC.md`).

## Decision frequency and feature cutoff
One row per trading date. A decision for date `d` may use the underlying
close on `d` and any date before it; nothing stamped later than `d` ever
enters the row for `d`. See `backend/swarm_opt/features/pointintime.py`.

## Point-in-time features (`FEATURE_SCHEMA_VERSION = "pointintime_v1"`)
18 features, frozen order (`pointintime.FEATURE_NAMES`):
- Returns: `ret_1, ret_5, ret_10, ret_20`
- Momentum/trend: `mom_10, mom_20, sma5_20_ratio, dist_sma20, trend_slope_20`
- Volatility: `realized_vol_10, realized_vol_20, vol_ratio_10_20`
- Regime: `drawdown_20, range_20`
- IV (optional, chain-derived, cached by `scripts/build_market_features.py`):
  `atm_iv, atm_iv_chg_5, skew_25d, atm_days_to_expiry`

`has_full_lookback` (needs 20 trailing trading dates) and
`atm_iv_available` (the IV cache covers that date) are explicit flags, not
silent NaNs. The IV cache uses only the nearest eligible expiry, strikes
within ±10% of spot, and `features.enrich.enrich_chain` (Black-76) — no new
pricing logic.

## Targets (`TARGET_SCHEMA_VERSION = "targets_v1"`, frozen in `features/targets.py`)
Horizon: one `WeeklyEpisode` (`data.calendar`) — `decision_date` to
`exit_date`, i.e. the same calendar-derived weekly cycle the simulator and
eventual PPO episode use. Not a fixed weekday or day count.

- **Direction** — `BEARISH | NEUTRAL | BULLISH` on the forward return
  `close[exit]/close[decision] - 1` vs. `±0.5·std(TRAIN forward returns)`,
  fit once on TRAIN (`fit_thresholds_on_train`) and frozen for VALIDATION/TEST.
- **Volatility** — `CONTRACT | NEUTRAL | EXPAND` on the ratio of forward
  realized vol (annualised stdev of log returns over the episode's mark
  dates) to the strictly historical `realized_vol_20` baseline as of the
  decision date, fixed bands 0.90 / 1.10.
- **Regime** — `TRENDING_DOWN | TRENDING_UP | RANGE_BOUND | HIGH_VOLATILITY |
  LOW_VOLATILITY`, mutually exclusive via an explicit priority rule: vol
  extremes (ratio > 1.30 / < 0.70) are checked before trend, so a violent
  trending week is never also counted as a vol regime.

Rows whose forward window is incomplete get `UNKNOWN`/`complete=False`
rather than being silently dropped.

## Agent interface (`backend/swarm_opt/agents/base.py`)
`XGBClassifierAgent.fit/predict/save/load`. Serialization uses XGBoost's own
JSON Booster format plus a metadata JSON sidecar (classes, feature names,
feature-schema version, target definition, train range, seed, package
versions) — never pickle. `predict` returns `AgentPrediction`: agent name,
model version, observation timestamp, predicted class, per-class
probabilities (validated finite, in `[0,1]`, summing to 1), `confidence`
(the uncalibrated top1-top2 probability margin — explicitly not a
calibrated probability of correctness), and `data_quality`.

Uses the low-level `xgb.train`/`Booster` API rather than the sklearn
wrapper so a TRAIN fold missing a class (a real occurrence at ~190 weekly
rows) degrades to near-zero probability for that class instead of raising.

## Greeks Agent (`agents/greeks_agent.py`)
Deterministic. Reuses `features.pricing` (Black-76) and
`features.enrich.enrich_chain` unchanged. Reports **contract-level** Greeks
(delta/gamma/theta/vega + IV + `forward_source`) for the ATM call and put of
the nearest eligible expiry — not every strike, and not a portfolio.
`portfolio` is always `None` with `portfolio_valid=False` and an explicit
reason: no Decision Engine or Trade Constructor exists yet to own one.

## SWARM_STATE_V1 (`backend/swarm_opt/state.py`)
`build_swarm_state(observation_timestamp, underlying, market_features,
agent_predictions, greeks_result, feature_schema_version)` is deterministic
for the same inputs and raises `SwarmStateError` if `validate_swarm_state`
finds anything wrong (missing agent, mismatched timestamps, bad
probabilities, wrong schema version, non-`None` portfolio, ...). Carries no
trading action or portfolio state — those are Decision Engine / Trade
Constructor concerns, out of scope here.

## Cadence
Unchanged from Week 1: specialists run once per weekly decision; future EOD
marks are never part of the pre-decision state.
