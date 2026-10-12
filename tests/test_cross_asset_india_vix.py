"""Phase 2.8 Workstream B: India VIX cross-asset feature tests --
point-in-time correctness, timestamp alignment, missing-data handling,
and feature-schema (v3) separation from v1/v2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swarm_opt.features.pointintime import (
    CROSS_ASSET_FEATURES,
    FEATURE_NAMES,
    FEATURE_NAMES_V2,
    FEATURE_NAMES_V3,
    FEATURE_SCHEMA_VERSION_V3,
    build_market_features,
)


def _hist(n=40, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    return pd.DataFrame({"date": dates, "open": close, "high": close, "low": close, "close": close})


def _vix_hist(dates, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"date": dates, "india_vix_close": rng.uniform(10, 25, size=len(dates))})


# -- feature-schema v3 separation --------------------------------------------#
def test_v3_adds_only_cross_asset_features_to_v2():
    assert set(FEATURE_NAMES_V3) - set(FEATURE_NAMES_V2) == set(CROSS_ASSET_FEATURES)
    assert len(FEATURE_NAMES_V3) == len(FEATURE_NAMES_V2) + len(CROSS_ASSET_FEATURES)


def test_default_call_unaffected_by_v3_existing():
    hist = _hist()
    feats, report = build_market_features(hist)
    assert "india_vix_level" not in feats.columns
    assert list(feats.columns) == ["date"] + FEATURE_NAMES + ["has_full_lookback", "atm_iv_available"]
    assert report.india_vix_coverage == 0.0


def test_v2_call_unaffected_by_india_vix_hist_when_not_requested():
    """Passing india_vix_hist without also selecting v3 feature names must
    not change v1/v2 output shape -- the column set is driven by
    `feature_names`, not by which optional data happens to be supplied."""
    hist = _hist()
    dates = hist["date"]
    vix = _vix_hist(dates)
    feats, _ = build_market_features(hist, feature_names=FEATURE_NAMES_V2, india_vix_hist=vix)
    assert "india_vix_level" not in feats.columns
    assert list(feats.columns) == ["date"] + FEATURE_NAMES_V2 + ["has_full_lookback", "atm_iv_available"]


# -- point-in-time correctness: mutation test, same pattern as the existing
#    underlying-derived feature leakage proof -------------------------------#
def test_india_vix_feature_unaffected_by_mutating_future_vix_values():
    hist = _hist(n=40, seed=2)
    dates = hist["date"]
    vix = _vix_hist(dates, seed=3)

    feats_before, _ = build_market_features(hist, feature_names=FEATURE_NAMES_V3, india_vix_hist=vix)

    cutoff = 25
    mutated_vix = vix.copy()
    mutated_vix.loc[mutated_vix["date"] > dates.iloc[cutoff], "india_vix_close"] = 999.0
    feats_after, _ = build_market_features(hist, feature_names=FEATURE_NAMES_V3, india_vix_hist=mutated_vix)

    pd.testing.assert_series_equal(
        feats_before.loc[:cutoff, "india_vix_level"], feats_after.loc[:cutoff, "india_vix_level"],
    )
    pd.testing.assert_series_equal(
        feats_before.loc[:cutoff, "india_vix_chg_5"], feats_after.loc[:cutoff, "india_vix_chg_5"],
    )


def test_india_vix_chg_5_uses_only_trailing_values():
    hist = _hist(n=20)
    dates = hist["date"]
    levels = np.arange(10.0, 30.0)  # deterministic, strictly increasing
    vix = pd.DataFrame({"date": dates, "india_vix_close": levels})
    feats, _ = build_market_features(hist, feature_names=FEATURE_NAMES_V3, india_vix_hist=vix)
    # chg_5 at row i should equal levels[i] - levels[i-5], not involve i+5
    for i in range(5, 20):
        assert feats["india_vix_chg_5"].iloc[i] == pytest.approx(levels[i] - levels[i - 5])


# -- missing-data / timestamp-alignment handling -----------------------------#
def test_india_vix_available_flag_reflects_actual_coverage():
    hist = _hist(n=30)
    dates = hist["date"]
    partial_vix = pd.DataFrame({"date": dates.iloc[10:20], "india_vix_close": np.linspace(12, 18, 10)})
    feats, report = build_market_features(hist, feature_names=FEATURE_NAMES_V3, india_vix_hist=partial_vix)
    assert "india_vix_available" in feats.columns
    assert feats["india_vix_available"].sum() == 10
    assert report.india_vix_coverage == pytest.approx(10 / 30)
    # outside the covered window, the level is NaN, not fabricated
    assert feats.loc[~feats["india_vix_available"], "india_vix_level"].isna().all()


def test_missing_india_vix_hist_entirely_gives_nan_not_an_error():
    hist = _hist(n=20)
    feats, report = build_market_features(hist, feature_names=FEATURE_NAMES_V3, india_vix_hist=None)
    assert feats["india_vix_level"].isna().all()
    assert not feats["india_vix_available"].any()
    assert report.india_vix_coverage == 0.0


def test_duplicate_india_vix_dates_are_deduplicated_not_summed():
    hist = _hist(n=15)
    dates = hist["date"]
    vix = pd.DataFrame({
        "date": list(dates) + [dates.iloc[5]],  # one duplicate date
        "india_vix_close": list(np.linspace(10, 20, 15)) + [999.0],
    })
    feats, _ = build_market_features(hist, feature_names=FEATURE_NAMES_V3, india_vix_hist=vix)
    # drop_duplicates keeps the FIRST occurrence -- the real value, not the injected duplicate
    assert feats["india_vix_level"].iloc[5] != pytest.approx(999.0)
