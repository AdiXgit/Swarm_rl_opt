# Experiments

## Goal
Test whether the specialist swarm improves PPO's weekly options strategy selection.

## Experiment 0 — Baseline
Simple deterministic strategy. Used to validate the simulator and establish context.

## Experiment 1 — Rule-Based Swarm
Use specialist outputs in frozen deterministic rules.

## Experiment 2 — PPO Without Swarm
PPO receives market/portfolio features but no specialist predictions.

## Experiment 3 — Swarm + PPO
PPO receives market state, direction, volatility, regime, Greeks, and portfolio state. This is the main architecture.

## Ablations
Remove Direction, Volatility, or Regime one at a time.

## Experiment 4 — Expiry-Regime Robustness (MANDATORY)
Not optional and not contingent on time remaining. Week 4 is incomplete without it.

NSE moved the NIFTY weekly expiry from Thursday to Tuesday in September 2025. A chronological split forces TEST to be the most recent data, so the primary TEST period (2025-09-01 → 2026-08-18) lies entirely inside the Tuesday regime and the transition coincides with the VALIDATION/TEST boundary. Any Swarm + PPO improvement measured on the primary TEST alone is therefore confounded with the regime change: it cannot distinguish "the swarm helps" from "the market structure changed".

Run the full Experiment 0-3 comparison a second time on a historical sub-period held entirely inside the Thursday regime, using the same code, seeds, costs and reward.

- Primary TEST stays 2025-09-01 → 2026-08-18 (Tuesday regime). This remains the headline result.
- Robustness TEST is a pre-2025-09 sub-period (Thursday regime), with TRAIN and VALIDATION re-cut chronologically before it so no future data enters either.
- Report both side by side. Never substitute the robustness result for the primary one.

Interpretation:
- Improvement in both regimes: evidence the swarm architecture, not the regime, is doing the work.
- Improvement only post-transition: treat the primary result as regime-specific and say so explicitly.
- Improvement only pre-transition: the architecture does not carry into current market structure.

## Robustness if time permits
Vary transaction costs, slippage, allocation limits, reward weights, and strategy universe. (Historical sub-period is no longer in this list -- it is mandatory, above.)

## Primary Metrics
Sharpe, maximum drawdown, total return. Secondary: win rate, profit factor, turnover, costs, strategy frequency.

## Sample Size — Known Limitation
The locked dataset (2021-04-01 → 2026-08-18) contains roughly 267 weekly episodes, of which about 186 fall in TRAIN. This is a small sample for PPO and is a property of the weekly cadence, not of the data source; it is accepted for the MVP rather than worked around.

Consequences, which every experiment must follow:
- Train every configuration under multiple random seeds. Report mean and standard deviation across seeds, never a single run.
- A difference between configurations that is smaller than the across-seed spread is not a result. State it as inconclusive.
- Report the seed count and the per-seed metrics alongside every aggregate, so the spread is auditable.
- Expect high variance. Prefer conclusions that hold across seeds, ablations, and both expiry regimes.

Extending the history (data exists back to at least 2018) is the lever if training proves too noisy. It is deliberately out of scope for the MVP; the dataset range is locked.

## Interpretation
Do not claim success from one high-return run. Look for risk-adjusted improvement, robustness, cost sensitivity, drawdown behavior, and ablation evidence.

A result counts only if it survives: the across-seed spread, the ablations, and the mandatory expiry-regime robustness check.

## Reproducibility
Save dataset version, feature/model versions, seed list, configuration, training metadata, and test metrics.
