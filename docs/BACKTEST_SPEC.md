# Backtest Specification

## Weekly Protocol
```text
Decision date → state → specialists → PPO → trade → EOD marks through week → exit/settlement → weekly result
```

## Coverage
Target 3–5 years of reliable EOD options data if available. Start with a small sample for development.

## Split
TRAIN → VALIDATION → TEST, chronological.

## Model Selection
Validation is for hyperparameters/checkpoint selection. Test remains untouched until final evaluation.

## Metrics
Total return, weekly return, Sharpe, maximum drawdown, volatility, win rate, profit factor, turnover, transaction costs. Optional Sortino/Calmar.

## Benchmarks
At minimum: deterministic baseline, rule-based swarm, PPO-only, swarm+PPO.

## Integrity
Record dataset/feature/model versions, seed, date range, strategy universe, cost and slippage assumptions. Prevent future EOD prices/IV/OI/expiry availability from entering current state.

## Reporting
Equity curve, drawdown curve, weekly return distribution, strategy frequency, performance and risk tables.
