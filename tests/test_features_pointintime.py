from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swarm_opt.features.pointintime import (
    FEATURE_NAMES,
    MAX_LOOKBACK_DAYS,
    build_market_features,
)


def _synthetic_hist(n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    rets = rng.normal(0, 0.01, n)
    close = 100 * np.cumprod(1 + rets)
    high = close * 1.01
    low = close * 0.99
    open_ = close * 1.0
    return pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low, "close": close})


def test_feature_columns_and_order_are_stable():
    hist = _synthetic_hist()
    feats, report = build_market_features(hist)
    assert list(feats.columns) == ["date"] + FEATURE_NAMES + ["has_full_lookback", "atm_iv_available"]
    assert len(feats) == len(hist)
    assert report.rows_in == len(hist)


def test_lookback_flag_marks_insufficient_history():
    hist = _synthetic_hist(n=30)
    feats, report = build_market_features(hist)
    assert feats["has_full_lookback"].iloc[:MAX_LOOKBACK_DAYS].eq(False).all()
    assert feats["has_full_lookback"].iloc[MAX_LOOKBACK_DAYS:].eq(True).all()
    assert report.rows_dropped_insufficient_lookback == MAX_LOOKBACK_DAYS


def test_ret_1_matches_manual_calculation():
    hist = _synthetic_hist()
    feats, _ = build_market_features(hist)
    expected = hist["close"].iloc[25] / hist["close"].iloc[24] - 1.0
    assert feats["ret_1"].iloc[25] == pytest.approx(expected)


def test_no_future_observation_enters_a_feature_row():
    """The row for date d must be identical whether or not the data after d
    changes. This is the actual leakage proof, not a column-name heuristic."""
    hist = _synthetic_hist(n=40, seed=1)
    feats_full, _ = build_market_features(hist)

    mutated = hist.copy()
    cutoff = 25
    mutated.loc[cutoff + 1 :, "close"] *= 5.0  # violently alter everything after the cutoff
    mutated.loc[cutoff + 1 :, "high"] *= 5.0
    mutated.loc[cutoff + 1 :, "low"] *= 5.0
    feats_mutated, _ = build_market_features(mutated)

    row_cols = [c for c in FEATURE_NAMES if c not in ("atm_iv", "atm_iv_chg_5", "skew_25d", "atm_days_to_expiry")]
    pd.testing.assert_frame_equal(
        feats_full.loc[:cutoff, row_cols].reset_index(drop=True),
        feats_mutated.loc[:cutoff, row_cols].reset_index(drop=True),
    )


def test_duplicate_timestamp_raises():
    hist = _synthetic_hist(n=10)
    dup = pd.concat([hist, hist.iloc[[3]]], ignore_index=True)
    with pytest.raises(ValueError):
        build_market_features(dup)


def test_atm_iv_missing_dates_are_flagged_not_fabricated():
    hist = _synthetic_hist(n=30)
    atm = pd.DataFrame({
        "date": hist["date"].iloc[[25, 26]],
        "atm_iv": [0.15, 0.16],
        "skew_25d": [0.01, 0.01],
        "days_to_expiry": [3, 2],
    })
    feats, report = build_market_features(hist, atm_iv_hist=atm)
    assert feats["atm_iv_available"].sum() == 2
    assert feats.loc[~feats["atm_iv_available"], "atm_iv"].isna().all()
    assert report.atm_iv_coverage == pytest.approx(2 / 30)


def test_beginning_of_dataset_has_nan_rolling_features_not_errors():
    hist = _synthetic_hist(n=5)
    feats, _ = build_market_features(hist)
    assert feats["ret_20"].isna().all()
    assert not feats["has_full_lookback"].any()
