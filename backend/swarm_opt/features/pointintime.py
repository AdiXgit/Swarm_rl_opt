"""Point-in-time market features for the Direction / Volatility / Regime agents.

Decision frequency and cutoff
------------------------------
One feature row per trading date `d`. The row for `d` may use the underlying
close on `d` and any date before it; it must never use a close, IV, or any
other observation stamped after `d`. `d` is meant to be used as a weekly
`WeeklyEpisode.decision_date` (see `swarm_opt.data.calendar`), but the
function itself is calendar-agnostic -- it just emits one point-in-time row
per input date.

All rolling/lag computations below operate on a frame already sorted
ascending by date and use only `pandas` operations that look backward
(`.shift(k)` with k >= 0, `.rolling(...)` ending at the current row). There is
no `.shift(-k)` and no join against a later date anywhere in this module.

ATM IV / skew are optional: they come from a separately cached, chain-derived
table (`scripts/build_market_features.py`) and are left NaN with
`atm_iv_available = False` when that cache does not cover a date, rather than
fabricated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_SCHEMA_VERSION = "pointintime_v1"

#: Longest trailing window used below. A row needs this many prior trading
#: dates on top of itself before every feature is defined.
MAX_LOOKBACK_DAYS = 20

RETURN_FEATURES = ["ret_1", "ret_5", "ret_10", "ret_20"]
MOMENTUM_FEATURES = ["mom_10", "mom_20", "sma5_20_ratio", "dist_sma20", "trend_slope_20"]
VOLATILITY_FEATURES = ["realized_vol_10", "realized_vol_20", "vol_ratio_10_20"]
REGIME_FEATURES = ["drawdown_20", "range_20"]
IV_FEATURES = ["atm_iv", "atm_iv_chg_5", "skew_25d", "atm_days_to_expiry"]

#: Deterministic, frozen column order. Persisted in the feature manifest so a
#: saved model's input order can be reproduced exactly at inference time.
#: FROZEN for `pointintime_v1` -- every Phase 2/2.1/2.2 artifact and report
#: was built against this exact 18-name list. Do not edit in place; a schema
#: change gets a new version and a new constant (see FEATURE_NAMES_V2 below),
#: so `build_market_features`'s default behavior and every existing saved
#: model stay reproducible.
FEATURE_NAMES: list[str] = (
    RETURN_FEATURES + MOMENTUM_FEATURES + VOLATILITY_FEATURES + REGIME_FEATURES + IV_FEATURES
)

#: Phase 2.5 feature-schema change: `dist_sma20` is byte-for-byte identical to
#: `mom_20` (both `close/sma20 - 1.0`; confirmed in
#: data/models/phase2_2_experiment_registry.json's feature-redundancy
#: ablation). Dropped here, under a new version, rather than removed from
#: FEATURE_NAMES itself -- any code still asking for `pointintime_v1`
#: (e.g. scripts/run_phase2_diagnostics.py reproducing the original Phase 2.2
#: baseline) keeps seeing the original 18-feature list unchanged.
FEATURE_SCHEMA_VERSION_V2 = "pointintime_v2"
FEATURE_NAMES_V2: list[str] = [f for f in FEATURE_NAMES if f != "dist_sma20"]

#: Phase 2.8 Workstream B: India VIX (NSE's own options-implied volatility
#: index) is a genuinely new, economically-justified cross-asset source,
#: confirmed fetchable through the EXISTING, unmodified NseKitProvider
#: (see scripts/build_india_vix_history.py). Same EOD timing convention as
#: every other feature here -- VIX_close[d] is published at the same market
#: close as NIFTY's own close[d], so using it for a decision made as of EOD
#: `d` introduces no lookahead, by the same reasoning already applied to
#: every other close-derived feature in this module.
CROSS_ASSET_FEATURES = ["india_vix_level", "india_vix_chg_5"]
FEATURE_SCHEMA_VERSION_V3 = "pointintime_v3"
FEATURE_NAMES_V3: list[str] = FEATURE_NAMES_V2 + CROSS_ASSET_FEATURES

#: Trading-day annualisation factor for realized volatility (NSE EOD data).
TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class FeatureBuildReport:
    """What happened when turning a date range into feature rows."""

    rows_in: int
    rows_with_full_lookback: int
    rows_dropped_insufficient_lookback: int
    atm_iv_coverage: float
    india_vix_coverage: float = 0.0


def _log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def build_market_features(
    underlying_hist: pd.DataFrame,
    atm_iv_hist: pd.DataFrame | None = None,
    feature_names: list[str] | None = None,
    india_vix_hist: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, FeatureBuildReport]:
    """One point-in-time feature row per date in `underlying_hist`.

    Every candidate feature (including `dist_sma20`) is always computed
    internally; `feature_names` only selects which columns are kept in the
    output, so `pointintime_v1` (the default, `feature_names=None` ->
    `FEATURE_NAMES`) and `pointintime_v2` (`feature_names=FEATURE_NAMES_V2`)
    share one implementation with zero duplicated logic. Passing nothing
    reproduces today's behavior exactly -- no existing caller is affected.

    Parameters
    ----------
    underlying_hist : columns `date, open, high, low, close`, ascending,
        one row per trading date, no duplicates. Future dates relative to
        the row being computed are never touched.
    atm_iv_hist : optional columns `date, atm_iv, skew_25d, days_to_expiry`,
        one row per date the chain-derived cache covers (see
        `scripts/build_market_features.py`). Dates missing from this table
        get NaN IV features and `atm_iv_available = False`.

    Returns
    -------
    (features, report) where `features` has columns
    `["date"] + FEATURE_NAMES + ["has_full_lookback", "atm_iv_available"]`,
    one row per input date, in the same order as `underlying_hist`.
    Rows without `MAX_LOOKBACK_DAYS` of prior history are kept (not dropped)
    but flagged `has_full_lookback = False`; a caller doing supervised
    training should filter on that flag rather than have rows silently
    vanish.
    """
    if underlying_hist["date"].duplicated().any():
        raise ValueError("duplicate timestamps in underlying_hist")
    hist = underlying_hist.sort_values("date").reset_index(drop=True)

    close = hist["close"].astype(float)
    n = len(hist)

    out = pd.DataFrame({"date": hist["date"].values})

    for k in (1, 5, 10, 20):
        out[f"ret_{k}"] = (close / close.shift(k) - 1.0).to_numpy()

    sma10 = close.rolling(10).mean()
    sma20 = close.rolling(20).mean()
    out["mom_10"] = (close / sma10 - 1.0).to_numpy()
    out["mom_20"] = (close / sma20 - 1.0).to_numpy()
    out["sma5_20_ratio"] = (close.rolling(5).mean() / sma20 - 1.0).to_numpy()
    out["dist_sma20"] = (close / sma20 - 1.0).to_numpy()

    # Trend slope: OLS slope of close over the trailing 20 observations,
    # normalised by the current close so it is scale-free across the sample.
    def _slope(window: np.ndarray) -> float:
        y = window
        x = np.arange(len(y), dtype=float)
        if not np.all(np.isfinite(y)):
            return np.nan
        beta = np.polyfit(x, y, 1)[0]
        return beta / y[-1] if y[-1] != 0 else np.nan

    out["trend_slope_20"] = (
        close.rolling(20).apply(_slope, raw=True).to_numpy()
    )

    log_ret = _log_returns(close)
    vol10 = log_ret.rolling(10).std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)
    vol20 = log_ret.rolling(20).std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)
    out["realized_vol_10"] = vol10.to_numpy()
    out["realized_vol_20"] = vol20.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out["vol_ratio_10_20"] = (vol10 / vol20 - 1.0).to_numpy()

    roll_max20 = close.rolling(20).max()
    out["drawdown_20"] = (close / roll_max20 - 1.0).to_numpy()
    roll_high20 = hist["high"].rolling(20).max()
    roll_low20 = hist["low"].rolling(20).min()
    out["range_20"] = ((roll_high20 - roll_low20) / close).to_numpy()

    if atm_iv_hist is not None and len(atm_iv_hist):
        iv = atm_iv_hist.sort_values("date").drop_duplicates("date")[
            ["date", "atm_iv", "skew_25d", "days_to_expiry"]
        ].rename(columns={"days_to_expiry": "atm_days_to_expiry"})
        out = out.merge(iv, on="date", how="left")
        out["atm_iv_chg_5"] = out["atm_iv"] - out["atm_iv"].shift(5)
        out["atm_iv_available"] = out["atm_iv"].notna()
    else:
        for col in IV_FEATURES:
            out[col] = np.nan
        out["atm_iv_available"] = False

    if india_vix_hist is not None and len(india_vix_hist):
        vix = india_vix_hist.sort_values("date").drop_duplicates("date")[["date", "india_vix_close"]]
        out = out.merge(vix, on="date", how="left")
        out["india_vix_level"] = out["india_vix_close"]
        out["india_vix_chg_5"] = out["india_vix_close"] - out["india_vix_close"].shift(5)
        out["india_vix_available"] = out["india_vix_close"].notna()
        out = out.drop(columns=["india_vix_close"])
    else:
        for col in CROSS_ASSET_FEATURES:
            out[col] = np.nan
        out["india_vix_available"] = False

    lookback_ok = np.arange(n) >= MAX_LOOKBACK_DAYS
    out["has_full_lookback"] = lookback_ok

    report = FeatureBuildReport(
        rows_in=n,
        rows_with_full_lookback=int(lookback_ok.sum()),
        rows_dropped_insufficient_lookback=int((~lookback_ok).sum()),
        atm_iv_coverage=float(out["atm_iv_available"].mean()) if n else 0.0,
        india_vix_coverage=float(out["india_vix_available"].mean()) if n else 0.0,
    )

    cols = feature_names if feature_names is not None else FEATURE_NAMES
    extra_flags = ["has_full_lookback", "atm_iv_available"]
    if any(c in CROSS_ASSET_FEATURES for c in cols):
        extra_flags.append("india_vix_available")
    out = out[["date"] + cols + extra_flags]
    return out, report
