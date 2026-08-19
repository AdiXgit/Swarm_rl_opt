# Agent Specification

## Direction
XGBoost. Inputs: returns, momentum, volume, price/volatility features. Outputs: bullish probability, bearish probability, confidence.

## Volatility
XGBoost. Inputs: IV, realized volatility, IV rank/percentile, skew, term structure, OI/volume. Outputs: expansion/contraction probabilities and confidence.

## Greeks
Quantitative calculations: Delta, Gamma, Theta, Vega for options/candidate portfolios.

## Market Regime
XGBoost. Initial classes: TRENDING_UP, TRENDING_DOWN, RANGE_BOUND, HIGH_VOLATILITY, LOW_VOLATILITY.

## Swarm State
Combine specialist outputs with market and portfolio features. Include decision date, model versions, and confidence where applicable.

## PPO
Consumes swarm + market + portfolio state and chooses strategy, strategy parameters, and allocation. It cannot bypass hard risk constraints.

## Cadence
Specialists run for the weekly decision. Future EOD marks are not part of the pre-action state.
