# EOD Simulator Specification

## Purpose
Simulate weekly options trades using EOD historical data. No intraday execution is modeled.

## Lifecycle
```text
Start-of-week entry → EOD mark → next EOD marks → week end → exit/settlement → weekly P&L
```

## Entry
Use an explicit EOD assumption: close/LTP or another documented proxy plus configurable slippage.

## Marking
At each available EOD observation calculate leg values, portfolio value, unrealized/realized P&L, Greeks where possible, and drawdown.

## Exit
Default: hold through the weekly horizon and exit at the final available EOD; handle expiry explicitly if it occurs during the episode.

## Accounting
Track cash, margin, positions, realized P&L, unrealized P&L, equity, Greeks, drawdown.

## Costs
Transaction costs and slippage are configurable and saved with each run.

## Tests
Test long calls/puts, spreads, iron condor, butterfly, multi-leg accounting, expiry, costs/slippage, risk rejection, and deterministic repeated simulation.
