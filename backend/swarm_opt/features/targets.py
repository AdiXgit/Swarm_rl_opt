"""Frozen target definitions for the Direction, Volatility and Regime agents.

Horizon
-------
Every target is defined over one `WeeklyEpisode`
(`swarm_opt.data.calendar.WeeklyEpisode`): the label for `episode.decision_date`
is computed from the underlying close on `episode.decision_date` and the
closes on `episode.mark_dates` (which run strictly after the decision date
through `episode.exit_date`). This is the same weekly horizon the eventual
PPO episode uses, and it is why labels are built from `WeeklyEpisode` objects
rather than a fixed number of calendar days: the actual NSE expiry cycle, not
a hard-coded weekday, delimits the horizon (see `data/calendar.py`).

These forward-looking quantities are **labels only**. They are computed in
this module and must never be joined back into the feature frame built by
`pointintime.build_market_features`; `tests/test_leakage_phase2.py` checks
that directly.

Direction target
-----------------
Three classes over the forward return `close[exit] / close[decision] - 1`:
BULLISH above `+bullish_return`, BEARISH below `-bearish_return` (frozen as
one symmetric threshold, see `fit_thresholds_on_train`), NEUTRAL otherwise.

Volatility target
-------------------
Three classes comparing forward realized volatility (annualised stdev of the
daily log returns on `episode.mark_dates`) against the trailing
`realized_vol_20` feature already computed as of the decision date (the
"strictly historical baseline"): EXPAND above `vol_expand_ratio`, CONTRACT
below `vol_contract_ratio`, NEUTRAL in the near-equal band between them.

Regime target
--------------
Five mutually exclusive classes via an explicit priority rule (volatility
extremes checked before trend, so a violently trending week is never also
double-counted as a trend regime):
1. `HIGH_VOLATILITY` if the forward/baseline vol ratio exceeds
   `regime_high_vol_ratio`.
2. `LOW_VOLATILITY` if it is below `regime_low_vol_ratio`.
3. `TRENDING_UP` if the forward return exceeds `+bullish_return`.
4. `TRENDING_DOWN` if it is below `-bullish_return`.
5. `RANGE_BOUND` otherwise.

Threshold freezing
-------------------
`bullish_return` / `bearish_return` are the only data-dependent threshold:
half a standard deviation of the TRAIN-only forward-return distribution,
fit once by `fit_thresholds_on_train` and then frozen into a
`TargetThresholds` that both VALIDATION and TEST labels reuse unchanged. The
volatility/regime ratio bands are fixed constants, chosen before looking at
any split's results, and never retuned against VALIDATION or TEST metrics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.calendar import WeeklyEpisode
from .pointintime import TRADING_DAYS_PER_YEAR

DIRECTION_CLASSES = ("BEARISH", "NEUTRAL", "BULLISH")
VOLATILITY_CLASSES = ("CONTRACT", "NEUTRAL", "EXPAND")
REGIME_CLASSES = (
    "TRENDING_DOWN", "TRENDING_UP", "RANGE_BOUND", "HIGH_VOLATILITY", "LOW_VOLATILITY",
)

TARGET_SCHEMA_VERSION = "targets_v1"


@dataclass(frozen=True)
class TargetThresholds:
    bullish_return: float
    bearish_return: float
    vol_expand_ratio: float = 1.10
    vol_contract_ratio: float = 0.90
    regime_high_vol_ratio: float = 1.30
    regime_low_vol_ratio: float = 0.70


def fit_thresholds_on_train(train_forward_returns: pd.Series) -> TargetThresholds:
    """Freeze the direction threshold from TRAIN-only forward returns.

    Half a standard deviation of the TRAIN weekly forward-return distribution,
    symmetric around zero. This is fit once on TRAIN and reused unchanged for
    VALIDATION and TEST labelling -- it is never refit or re-tuned against
    either split's metrics.
    """
    clean = train_forward_returns.dropna()
    if len(clean) < 2:
        raise ValueError("need at least 2 TRAIN forward returns to fit thresholds")
    std = float(clean.std(ddof=0))
    return TargetThresholds(bullish_return=0.5 * std, bearish_return=0.5 * std)


def direction_label(forward_return: float, thresholds: TargetThresholds) -> str:
    if not np.isfinite(forward_return):
        return "UNKNOWN"
    if forward_return > thresholds.bullish_return:
        return "BULLISH"
    if forward_return < -thresholds.bearish_return:
        return "BEARISH"
    return "NEUTRAL"


def volatility_label(vol_ratio: float, thresholds: TargetThresholds) -> str:
    if not np.isfinite(vol_ratio):
        return "UNKNOWN"
    if vol_ratio > thresholds.vol_expand_ratio:
        return "EXPAND"
    if vol_ratio < thresholds.vol_contract_ratio:
        return "CONTRACT"
    return "NEUTRAL"


def regime_label(forward_return: float, vol_ratio: float, thresholds: TargetThresholds) -> str:
    if not np.isfinite(forward_return) or not np.isfinite(vol_ratio):
        return "UNKNOWN"
    if vol_ratio > thresholds.regime_high_vol_ratio:
        return "HIGH_VOLATILITY"
    if vol_ratio < thresholds.regime_low_vol_ratio:
        return "LOW_VOLATILITY"
    if forward_return > thresholds.bullish_return:
        return "TRENDING_UP"
    if forward_return < -thresholds.bullish_return:
        return "TRENDING_DOWN"
    return "RANGE_BOUND"


def _forward_realized_vol(closes: np.ndarray) -> float:
    """Annualised stdev of log returns across one episode's mark window."""
    if len(closes) < 2:
        return float("nan")
    log_ret = np.diff(np.log(closes))
    if len(log_ret) < 1 or not np.all(np.isfinite(log_ret)):
        return float("nan")
    return float(np.std(log_ret, ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))


def build_targets(
    episodes: list[WeeklyEpisode],
    underlying_hist: pd.DataFrame,
    baseline_vol: pd.Series,
    thresholds: TargetThresholds,
) -> pd.DataFrame:
    """One label row per episode, keyed by `decision_date`.

    `baseline_vol` must be a `pandas.Series` of the `realized_vol_20` column
    from `pointintime.build_market_features` on the SAME `underlying_hist`,
    indexed by plain `datetime.date` (not `Timestamp`) -- i.e. a strictly
    historical quantity computed only from data up to and including the
    decision date. Passing anything else here would silently defeat the
    leakage guarantee this function is built on.

    Rows whose forward window is incomplete (decision date or any mark date
    missing from `underlying_hist`) get `UNKNOWN` labels and
    `complete = False` rather than being silently dropped, so a caller can
    report the exclusion count instead of losing it.
    """
    close_by_date = underlying_hist.set_index(
        pd.to_datetime(underlying_hist["date"]).dt.date
    )["close"].astype(float)

    rows = []
    for ep in episodes:
        decision_close = close_by_date.get(ep.decision_date, np.nan)
        exit_close = close_by_date.get(ep.exit_date, np.nan)
        forward_return = (
            float(exit_close / decision_close - 1.0)
            if np.isfinite(decision_close) and np.isfinite(exit_close) and decision_close
            else np.nan
        )

        window_dates = [ep.decision_date, *ep.mark_dates]
        window_closes = np.array(
            [close_by_date.get(d, np.nan) for d in window_dates], dtype=float
        )
        complete = bool(np.all(np.isfinite(window_closes)))
        forward_vol = _forward_realized_vol(window_closes) if complete else np.nan

        base_vol = baseline_vol.get(ep.decision_date, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            vol_ratio = (
                float(forward_vol / base_vol)
                if np.isfinite(forward_vol) and np.isfinite(base_vol) and base_vol
                else np.nan
            )

        complete = complete and np.isfinite(forward_return) and np.isfinite(vol_ratio)
        rows.append({
            "decision_date": ep.decision_date,
            "expiry": ep.expiry,
            "exit_date": ep.exit_date,
            "forward_return": forward_return,
            "forward_realized_vol": forward_vol,
            "baseline_realized_vol": base_vol,
            "vol_ratio": vol_ratio,
            "complete": complete,
            "direction_label": direction_label(forward_return, thresholds) if complete else "UNKNOWN",
            "volatility_label": volatility_label(vol_ratio, thresholds) if complete else "UNKNOWN",
            "regime_label": (
                regime_label(forward_return, vol_ratio, thresholds) if complete else "UNKNOWN"
            ),
        })
    return pd.DataFrame(rows)
