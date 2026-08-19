# Project Specification

## Objective
Build a research prototype for weekly NIFTY options strategy selection using four specialist components and PPO. The output must be a complete simulated trade, not only a strategy label.

## Research Question
Does the specialist swarm improve PPO's strategy selection and risk-adjusted performance compared with PPO without swarm signals?

## Weekly Design
One primary PPO decision per weekly episode. Use EOD data only. Mark the position on available EOD observations through the week and use the weekly result as the main outcome/reward.

## Data Scope
Target approximately 3–5 years of EOD options data if a reliable source provides it; validate with a small sample first. Preserve multiple upcoming expiries and all relevant strikes. NIFTY is the initial underlying.

## Strategies
HOLD, LONG_CALL, LONG_PUT, BULL_CALL_SPREAD, BEAR_PUT_SPREAD, LONG_STRADDLE, SHORT_STRADDLE, IRON_CONDOR, BUTTERFLY.

## Four-Week Roadmap
### Week 1
Validate NseKit/provider feasibility; create provider interface and local fixture; normalize EOD chains; features; Greeks; weekly EOD simulator; tests.
### Week 2
Direction XGBoost, volatility XGBoost, regime XGBoost, Greeks component, swarm state, PPO environment integration.
### Week 3
PPO training, strategy/action representation, position sizing, expiry/strike selection, actual leg construction, hard risk validation, end-to-end weekly episodes.
### Week 4
Chronological train/validation/test, baseline, rule swarm, PPO-only, swarm+PPO, ablations, metrics, result persistence, optional Next.js dashboard.

## Non-Goals
Intraday trading, real money, broker APIs, Rust, Go, LLMs, LangGraph, Kafka, HFT, distributed RL, unnecessary microservices.
