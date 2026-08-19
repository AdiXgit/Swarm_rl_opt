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

## Splits
TRAIN → VALIDATION → TEST, chronological.
