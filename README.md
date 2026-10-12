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
1. Data + EOD simulator — done
2. Specialist swarm — done (Direction/Volatility/Regime XGBoost agents, deterministic Greeks agent, `SWARM_STATE_V1`)
3. PPO + complete trade construction — not started
4. Research evaluation + optional dashboard — not started

## Phase 2 commands
```bash
# one-time: cache per-date ATM IV/skew from the option chain (~9 min)
python scripts/build_market_features.py

# verify the frozen manifest, build features/targets, run the leakage audit,
# train Direction/Volatility/Regime on TRAIN, evaluate on TRAIN+VALIDATION
python scripts/train_agents.py

# fast, fully end-to-end smoke test on a small TRAIN-only subset
python scripts/run_smoke_test.py

# build one SWARM_STATE_V1 from saved artifacts for a historical date
python scripts/build_swarm_state.py --date 2025-06-05

# full suite / just Phase 2
pytest
pytest tests/test_features_pointintime.py tests/test_targets.py tests/test_leakage_phase2.py \
       tests/test_agents.py tests/test_greeks_agent.py tests/test_swarm_state.py \
       tests/test_phase2_integration.py tests/test_phase2_negative.py
```
See `docs/AGENT_SPEC.md` for feature/target definitions and `data/models/phase2_training_report.json`
for the TRAIN/VALIDATION metrics the last `train_agents.py` run produced.

## Phase 2.5: historical extension + retraining
```bash
# build + audit the separately-versioned extension (2019-02-18..2021-03-31)
python scripts/build_dataset.py --start 2019-02-18 --end 2021-03-31 --root data/processed/nifty_eod_ext_2019_2021
python scripts/build_market_features_extension.py
python scripts/audit_extension_dataset.py

# retrain on EXTENDED TRAIN (extension + original TRAIN), evaluate on the unchanged VALIDATION
python scripts/train_agents_extended.py --seed 0

# build one SWARM_STATE_V1 from the extended artifacts
python scripts/build_swarm_state_extended.py --date 2025-06-05
```
See `docs/PHASE_2_5_EXTENDED_DATASET_AND_RETRAINING.md` for the full report, metrics, and the Phase 3 readiness decision.
