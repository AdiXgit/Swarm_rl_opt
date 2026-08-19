# PPO / RL Specification

## Objective
PPO learns a weekly policy for selecting and sizing NIFTY options strategies.

## Episode
```text
Start-of-week state → PPO action → trade construction → risk validation → EOD marks through week → weekly P&L → reward
```

## Observation
Market features, direction/volatility/regime signals, Greeks, capital, positions, margin, P&L, drawdown. No future EOD information.

## Actions
Strategy + strategy parameters + allocation. Initial strategies: HOLD, LONG_CALL, LONG_PUT, BULL_CALL_SPREAD, BEAR_PUT_SPREAD, LONG_STRADDLE, SHORT_STRADDLE, IRON_CONDOR, BUTTERFLY.

Possible parameters: expiry bucket, target delta/moneyness, wing width, center, allocation.

## Reward
Initial concept: `weekly_pnl - transaction_cost_penalty - drawdown_penalty - excessive_risk_penalty`. Freeze the exact formula before final experiments.

## Hard Constraints
Maximum allocation, margin, loss, Delta, Gamma, Vega, and drawdown. Deterministic risk validation is authoritative.

## Training
Stable-Baselines3 PPO. Save seed, dataset/feature/environment versions, hyperparameters, steps, and checkpoint metadata.

## Leakage
Future outcomes may determine reward, but future observations cannot enter the pre-action state.
