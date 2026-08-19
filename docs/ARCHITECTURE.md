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
backend/
├── data/providers
├── features
├── agents
├── rl
├── trading
├── simulation
├── evaluation
└── api
```

## Provider Abstraction
`DataProvider` is the contract. Implement `NseKitProvider` after validation and `LocalParquetProvider` for deterministic/offline development. ML/RL code must not depend directly on NseKit.

## Storage
Parquet: raw/processed/features. PostgreSQL/Supabase: experiments, trades, legs, signals, PPO decisions, weekly results, metrics.

## Frontend
Next.js presents persisted results. It does not train PPO.

## Future Paper Trading
Only after the MVP: PPO → Trade Construction → Risk Validation → Paper Executor.
