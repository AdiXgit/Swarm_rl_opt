# SWARM-OPT

**Swarm-Based Reinforcement Learning for Weekly NIFTY Options Strategy Selection and Position Management**

SWARM-OPT uses EOD historical NIFTY options data. Direction, volatility, Greeks, and market-regime specialists form a swarm state. PPO chooses a strategy, parameters, and allocation. A trade-construction engine maps that decision to actual option legs, a hard risk layer validates it, and an EOD simulator marks it through the week.

## Weekly Flow
```text
Weekly decision → Specialists → Swarm → PPO → Trade construction → Risk validation → EOD marks → Weekly P&L → Reward
```

## Data
Preferred fields: date, underlying/spot, expiry, strike, CE/PE, OHLC/close, volume, OI, change in OI, IV where available. Intraday data is not required.

## Stack
Python, XGBoost, PyTorch, Gymnasium, Stable-Baselines3, Parquet, PostgreSQL/Supabase, Python API, Next.js/TypeScript, Vercel.

## MVP Exclusions
Real-money/live trading, broker integration, intraday execution, Rust, Go, LLMs, LangGraph, Kafka, HFT, distributed RL.

## Four Weeks
1. Data + EOD simulator
2. Specialist swarm
3. PPO + complete trade construction
4. Research evaluation + optional dashboard
