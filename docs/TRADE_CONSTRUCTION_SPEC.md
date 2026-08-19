# Trade Construction Specification

## Purpose
Map PPO's abstract weekly action to actual option contracts available on the decision date.

Example PPO output:
```text
IRON_CONDOR
allocation=4%
short_delta=0.25
wing_width=100
expiry_bucket=next_valid_expiry
```

Output must contain: decision date, underlying, strategy, expiry, legs, quantity, entry price, capital required, credit/debit, max profit/loss, breakevens, Delta/Gamma/Theta/Vega.

## Strike Selection
Use target delta, moneyness, spot distance, and wing width. Target delta is preferred when reliable IV/Greeks exist.

## Expiry
Only expiries available on the decision date. Never choose an expiry using future information.

## Strategies
- Long Call: long CE
- Long Put: long PE
- Bull Call Spread: long lower CE + short higher CE
- Bear Put Spread: long higher PE + short lower PE
- Long Straddle: long CE + long PE at center
- Short Straddle: short CE + short PE at center
- Iron Condor: long lower PE + short higher PE + short lower CE + long higher CE
- Butterfly: documented symmetric convention

## Pricing
Use actual EOD fields if available. Otherwise use the configured EOD price/slippage model. Never invent bid/ask.

## Validation
Contracts exist, expiry matches, legs form strategy, quantity is valid, payoff is valid, and hard risk limits pass.
