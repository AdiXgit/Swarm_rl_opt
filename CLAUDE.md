# SWARM-OPT — Claude Code Instructions

SWARM-OPT is a four-week research prototype for weekly NIFTY options strategy selection and position management using EOD historical options data. Four specialist components create a swarm state; PPO selects an options strategy and position size; trade construction creates actual option legs; a deterministic risk layer validates them; a Python EOD simulator produces weekly P&L/reward.

## Locked Architecture
```text
Historical NIFTY EOD Options
        ↓
ETL + Features
        ↓
Direction XGBoost | Volatility XGBoost | Greeks Quant | Regime XGBoost
        ↓
Swarm State → PPO → Strategy + Parameters + Size
        ↓
Trade Construction → Hard Risk Validation → EOD Simulator
        ↓
EOD marks through week → Weekly P&L → Reward
```

## Locked MVP
- Weekly NIFTY options strategy selection using EOD historical data.
- One primary PPO decision per weekly episode; simulator marks positions on available EOD observations through the week.
- Python only for MVP.
- XGBoost, PyTorch, Gymnasium, Stable-Baselines3, Pandas, NumPy, SciPy.
- Parquet for large datasets; PostgreSQL/Supabase for application/results data.
- Python API + Next.js/TypeScript + Vercel.
- No Rust, Go, LLMs, LangGraph, Kafka, broker APIs, real-money/live autonomous trading, HFT, or intraday execution in MVP.
- No NSE scraping or access-control/rate-limit bypassing. Use a provider abstraction; validate NseKit as the primary candidate and use local Parquet fixtures for development.

## Rules
1. Read the relevant docs before changing a subsystem.
2. Do not silently change state/action/reward definitions, strategy universe, weekly cadence, or data contract.
3. Never introduce lookahead bias.
4. Time-series splits are chronological.
5. Future EOD prices, IV, OI, or option availability cannot enter the current decision state.
6. If bid/ask is unavailable, use a documented EOD price/slippage assumption; never fabricate bid/ask.
7. Keep data, ML, RL, trading, simulation, evaluation, API, and frontend modular.
8. Add tests for numerical/financial logic and end-to-end integration.
9. Make stochastic experiments reproducible.
10. Do not optimize or add infrastructure prematurely.

## Four-Week Milestones
- Week 1: data-source validation, normalized EOD data, features, Greeks, weekly EOD simulator.
- Week 2: direction, volatility, regime, Greeks components and swarm state.
- Week 3: PPO, strategy/action space, position sizing, trade construction, strike/expiry selection, hard risk validation.
- Week 4: chronological evaluation, baselines, ablations, metrics, persistence, optional Next.js dashboard.

## First Task
Do not build the whole system. Inspect the repository/docs/environment, validate NseKit with a small sample, create a DataProvider interface and local Parquet fixture, report exact fields/coverage/limitations, then stop for approval.
