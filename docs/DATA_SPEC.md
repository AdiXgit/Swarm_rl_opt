# Data Specification

## Granularity
EOD only. No intraday requirement.

## Normalized Option Record
```text
date
underlying
spot
expiry
strike
option_type
open
high
low
close
volume
open_interest
change_in_open_interest
implied_volatility
```

Some fields may be unavailable from the provider; missing fields must be logged, not fabricated.

## Provider
Primary candidate: NseKit historical options data. Validate actual schema and coverage before committing. Local CSV/Parquet is the fallback/development provider. Do not scrape or bypass NSE controls.

## Coverage
Target 3–5 years if available. Start with ~5 dates and 1–2 expiries/date for validation. Final dataset should document first/last date, dates, expiries, strikes, rows, and missingness.

## Features
Returns, momentum, realized volatility, IV level/rank/percentile, skew, term structure, volume/OI, moneyness, time to expiry, Greeks.

## Leakage Rules
A decision date can use only data available then. Never use future EOD prices, IV, OI, expiry availability, labels, or test-set-fitted transforms.

## Pricing
If bid/ask is absent, use a documented EOD close/LTP + slippage/cost assumption. Never fabricate historical bid/ask.

## Historical extension (Phase 2.5)
A separately-versioned historical extension (2019-02-18..2021-03-31,
content fingerprint `1386411c71b25e19`) lives in
`data/processed/nifty_eod_ext_2019_2021/`, built with the unmodified
`scripts/build_dataset.py`. It is combined with the frozen dataset only
through the new, additive `CombinedParquetProvider` (read-only union) and
never merged into or overwriting the original files. See
`docs/PHASE_2_5_EXTENDED_DATASET_AND_RETRAINING.md` and
`docs/HISTORICAL_PROBE_AND_EXPIRY_AUDIT.md` for the verified weekly-options
transition date (confirmed between 2019-02-07 and 2019-02-14) and the
data-quality audit.

## Splits
TRAIN → VALIDATION → TEST, chronological. Frozen manifest fingerprint
`653ccb70c795d86f` (`data/processed/nifty_eod/manifest_splits.json`); Phase 2
training verifies this before touching any data (`scripts/train_agents.py`).

## Phase 2 point-in-time features and targets (status: implemented)
See `docs/AGENT_SPEC.md` for the full feature/target definitions. Summary:
- Features come from the underlying OHLC series (always available) plus an
  optional, separately cached ATM IV/skew table
  (`scripts/build_market_features.py` → `data/features/atm_iv_history.parquet`)
  built from the nearest eligible expiry only, within ±10% of spot.
- Targets are defined over `WeeklyEpisode.decision_date -> exit_date`
  (`backend/swarm_opt/data/calendar.py`), not a fixed day count.
- Direction/volatility/regime thresholds are fit on TRAIN only and frozen
  before being applied to VALIDATION/TEST.
