"""Phase 2.5 feature-schema change: FEATURE_NAMES_V2 drops `dist_sma20`
without disturbing the default (v1) behavior anything else relies on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from swarm_opt.features.pointintime import (
    FEATURE_NAMES,
    FEATURE_NAMES_V2,
    FEATURE_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION_V2,
    build_market_features,
)


def _hist(n=30, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    return pd.DataFrame({"date": dates, "open": close, "high": close, "low": close, "close": close})


def test_v2_feature_list_drops_only_dist_sma20():
    assert set(FEATURE_NAMES) - set(FEATURE_NAMES_V2) == {"dist_sma20"}
    assert len(FEATURE_NAMES_V2) == len(FEATURE_NAMES) - 1
    assert FEATURE_SCHEMA_VERSION_V2 != FEATURE_SCHEMA_VERSION


def test_default_call_is_unaffected_by_v2_existing():
    hist = _hist()
    feats, _ = build_market_features(hist)
    assert "dist_sma20" in feats.columns
    assert list(feats.columns) == ["date"] + FEATURE_NAMES + ["has_full_lookback", "atm_iv_available"]


def test_v2_call_excludes_dist_sma20_but_matches_v1_elsewhere():
    hist = _hist()
    feats_v1, _ = build_market_features(hist)
    feats_v2, _ = build_market_features(hist, feature_names=FEATURE_NAMES_V2)

    assert "dist_sma20" not in feats_v2.columns
    assert list(feats_v2.columns) == ["date"] + FEATURE_NAMES_V2 + ["has_full_lookback", "atm_iv_available"]

    shared = [c for c in FEATURE_NAMES_V2 if c != "mom_20"]  # mom_20 == dist_sma20 by construction; check the rest
    pd.testing.assert_frame_equal(feats_v1[shared], feats_v2[shared])
    # and mom_20 itself (kept in both) is identical across v1/v2 since it's the same computation
    pd.testing.assert_series_equal(feats_v1["mom_20"], feats_v2["mom_20"])


def test_v1_and_v2_mom_20_and_dropped_dist_sma20_are_identical_values():
    hist = _hist()
    feats_v1, _ = build_market_features(hist)
    valid = feats_v1["mom_20"].notna()
    pd.testing.assert_series_equal(
        feats_v1.loc[valid, "mom_20"], feats_v1.loc[valid, "dist_sma20"], check_names=False,
    )
