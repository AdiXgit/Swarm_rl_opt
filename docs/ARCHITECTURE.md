# Architecture

```text
EOD Options → DataProvider → Normalized Data → Features
                                      ↓
              Direction / Volatility / Regime / Greeks
                                      ↓
                                 Swarm State
                                      ↓
                                     PPO
                                      ↓
                         Strategy + Parameters + Size
                                      ↓
                              Trade Construction
                                      ↓
                              Risk Validation
                                      ↓
                               EOD Simulator
                                      ↓
                                Weekly P&L
```

## Modules
```text
backend/swarm_opt/
├── data/providers         # Week 1
├── features               # Week 1 pricing/enrich + Phase 2 pointintime/targets/audit/manifest
├── agents                 # Phase 2: direction, volatility, regime (XGBoost), greeks_agent (deterministic)
├── state.py                # Phase 2: SWARM_STATE_V1 builder + validator
├── pipeline.py              # Phase 2: dataset -> features/targets glue, shared by scripts and tests
├── evaluation.py            # Phase 2: classification metrics vs. baselines
├── rl                      # not yet implemented (deferred, see docs/RL_SPEC.md)
├── trading                 # not yet implemented
├── simulation              # not yet implemented
└── api                      # not yet implemented
```

Phase 2 status: Direction/Volatility/Regime agents, the Greeks agent and
`SWARM_STATE_V1` are implemented and tested (see `docs/AGENT_SPEC.md`). PPO,
the Decision Engine and the Trade Constructor are out of scope and not
implemented.

## Provider Abstraction
`DataProvider` is the contract. Implement `NseKitProvider` after validation and `LocalParquetProvider` for deterministic/offline development. ML/RL code must not depend directly on NseKit.

## Storage
Parquet: raw/processed/features. PostgreSQL/Supabase: experiments, trades, legs, signals, PPO decisions, weekly results, metrics.

## Frontend
Next.js presents persisted results. It does not train PPO.

## Future Paper Trading
Only after the MVP: PPO → Trade Construction → Risk Validation → Paper Executor.
