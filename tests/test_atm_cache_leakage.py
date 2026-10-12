"""Phase 2.2 Task A1: mutation-based leakage tests for the ATM IV/skew cache.

These tests build a tiny synthetic chain fixture on disk (real
`LocalParquetProvider` file layout, not an in-memory stub) so the cache-key,
expiry-selection and join logic actually exercises file-based point-in-time
reads, not just the in-memory DataFrame filtering.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from swarm_opt.data.providers.local_parquet_provider import LocalParquetProvider
from swarm_opt.features.atm_cache import build_atm_iv_history, build_atm_iv_row, nearest_expiry_band
from swarm_opt.features.pointintime import build_market_features
from swarm_opt.features.pricing import black76_price, year_fraction

UNDERLYING = "NIFTY"


def _chain_for_date(on: date, forward: float, expiry: date, strikes=None, sigma=0.20) -> pd.DataFrame:
    strikes = strikes or [forward * m for m in (0.92, 0.96, 1.0, 1.04, 1.08)]
    dte = (expiry - on).days
    t = float(year_fraction(max(dte, 0)))
    rows = []
    for k in strikes:
        for opt_type in ("CE", "PE"):
            price = float(black76_price(forward, k, sigma, t, opt_type))
            rows.append({
                "date": pd.Timestamp(on), "underlying": UNDERLYING, "spot": forward - 1.5,
                "expiry": pd.Timestamp(expiry), "strike": float(k), "option_type": opt_type,
                "open": price, "high": price, "low": price, "close": price,
                "volume": 10, "open_interest": 100, "change_in_open_interest": 0,
                "implied_volatility": np.nan, "settle_price": price,
                "forward": forward, "forward_source": "futures", "lot_size": 25,
                "turnover_notional": 0.0, "is_traded": True, "days_to_expiry": dte,
                "source": "synthetic", "schema_version": "synthetic",
            })
    return pd.DataFrame(rows)


def _write_fixture(root, dates_forwards_sigmas, expiry: date):
    for on, forward, sigma in dates_forwards_sigmas:
        chain = _chain_for_date(on, forward, expiry, sigma=sigma)
        path = root / "chains" / f"underlying={UNDERLYING}" / f"date={on.isoformat()}" / "part.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        chain.to_parquet(path, index=False)
    hist = pd.DataFrame({
        "date": [pd.Timestamp(on) for on, _, _ in dates_forwards_sigmas],
        "open": [f for _, f, _ in dates_forwards_sigmas],
        "high": [f for _, f, _ in dates_forwards_sigmas],
        "low": [f for _, f, _ in dates_forwards_sigmas],
        "close": [f for _, f, _ in dates_forwards_sigmas],
    })
    (root / "underlying").mkdir(parents=True, exist_ok=True)
    hist.to_parquet(root / "underlying" / f"{UNDERLYING}.parquet", index=False)


def test_mutating_a_later_date_chain_does_not_change_an_earlier_row(tmp_path):
    expiry = date(2024, 1, 11)
    days = [expiry - timedelta(days=d) for d in (8, 6, 4, 2)][::-1]  # ascending
    provider_root = tmp_path / "fixture"
    _write_fixture(provider_root, [(d, 100.0 + i, 0.20) for i, d in enumerate(days)], expiry)
    provider = LocalParquetProvider(provider_root)

    cutoff = days[1]
    history_before = build_atm_iv_history(provider, UNDERLYING, days)
    row_at_cutoff_before = history_before[history_before["date"] == pd.Timestamp(cutoff)].iloc[0]

    # Violently mutate every chain dated strictly after the cutoff: forward,
    # strikes, prices and sigma all change, not just a feature column.
    for on in days:
        if on <= cutoff:
            continue
        mutated = _chain_for_date(on, forward=9999.0, expiry=expiry, sigma=0.95)
        path = provider_root / "chains" / f"underlying={UNDERLYING}" / f"date={on.isoformat()}" / "part.parquet"
        mutated.to_parquet(path, index=False)

    history_after = build_atm_iv_history(provider, UNDERLYING, days)
    row_at_cutoff_after = history_after[history_after["date"] == pd.Timestamp(cutoff)].iloc[0]

    pd.testing.assert_series_equal(row_at_cutoff_before, row_at_cutoff_after)


def test_mutating_the_future_underlying_series_does_not_change_an_earlier_feature_row(tmp_path):
    """End-to-end: mutate the future at both the chain layer and the
    underlying-OHLC layer, then prove the merged point-in-time feature row
    for the cutoff date is unchanged -- the actual leakage proof the Phase
    2.1 diagnostics flagged as missing for the IV/skew columns."""
    expiry = date(2024, 2, 15)
    days = [expiry - timedelta(days=d) for d in range(40, 0, -2)]  # 20 dates, every other day
    provider_root = tmp_path / "fixture"
    _write_fixture(provider_root, [(d, 100.0 + 0.1 * i, 0.20) for i, d in enumerate(days)], expiry)
    provider = LocalParquetProvider(provider_root)

    cutoff_idx = 10
    cutoff = days[cutoff_idx]

    hist = provider.underlying_history(UNDERLYING, days[0], days[-1])
    atm_hist = build_atm_iv_history(provider, UNDERLYING, days)
    feats_before, _ = build_market_features(hist, atm_iv_hist=atm_hist)
    row_before = feats_before[feats_before["date"] == pd.Timestamp(cutoff)].iloc[0]

    # Mutate the underlying close AND every chain strictly after the cutoff.
    mutated_hist = hist.copy()
    mutated_hist.loc[mutated_hist["date"] > pd.Timestamp(cutoff), ["open", "high", "low", "close"]] *= 50.0
    for on in days[cutoff_idx + 1 :]:
        mutated_chain = _chain_for_date(on, forward=50000.0, expiry=expiry, sigma=0.99)
        path = provider_root / "chains" / f"underlying={UNDERLYING}" / f"date={on.isoformat()}" / "part.parquet"
        mutated_chain.to_parquet(path, index=False)

    atm_hist_after = build_atm_iv_history(provider, UNDERLYING, days)
    feats_after, _ = build_market_features(mutated_hist, atm_iv_hist=atm_hist_after)
    row_after = feats_after[feats_after["date"] == pd.Timestamp(cutoff)].iloc[0]

    pd.testing.assert_series_equal(row_before, row_after)


def test_missing_chain_for_a_date_produces_no_row_not_a_fabricated_one(tmp_path):
    provider_root = tmp_path / "fixture"
    (provider_root / "chains" / f"underlying={UNDERLYING}" / "date=2024-01-01").mkdir(parents=True)
    # no part.parquet written -> option_chain returns an empty frame for this date
    (provider_root / "underlying").mkdir()
    pd.DataFrame({"date": [pd.Timestamp("2024-01-01")], "open": [100.0], "high": [100.0],
                  "low": [100.0], "close": [100.0]}).to_parquet(provider_root / "underlying" / f"{UNDERLYING}.parquet")
    provider = LocalParquetProvider(provider_root)

    history = build_atm_iv_history(provider, UNDERLYING, [date(2024, 1, 1)])
    assert history.empty


def test_chain_with_no_iv_invertible_strikes_produces_no_row():
    """Every quote printed at intrinsic (zero vol content): `atm_summary`
    finds no priced strikes, so the row is skipped, not fabricated with a
    placeholder IV."""
    expiry = date(2024, 1, 10)
    on = date(2024, 1, 8)
    forward = 100.0
    rows = []
    for k, opt_type, price in [(100.0, "CE", 0.0), (100.0, "PE", 0.0)]:  # printed at intrinsic
        rows.append({
            "date": pd.Timestamp(on), "underlying": UNDERLYING, "spot": forward, "expiry": pd.Timestamp(expiry),
            "strike": k, "option_type": opt_type, "close": price, "volume": 5,
            "forward": forward, "forward_source": "futures", "days_to_expiry": 2,
        })
    chain = pd.DataFrame(rows)
    assert build_atm_iv_row(chain, on) is None


def test_duplicate_records_do_not_crash_and_are_documented():
    """Duplicate (date, strike, option_type) rows on one date's chain: the
    cache builder must not crash. `atm_summary` averages duplicate IVs at
    the ATM strike, so the result is deterministic, just not deduplicated
    upstream -- documented here rather than silently assumed."""
    expiry = date(2024, 1, 10)
    on = date(2024, 1, 5)
    forward = 100.0
    chain = _chain_for_date(on, forward, expiry)
    duplicated = pd.concat([chain, chain.iloc[[2, 3]]], ignore_index=True)  # duplicate one strike/side pair
    row = build_atm_iv_row(duplicated, on)
    row_clean = build_atm_iv_row(chain, on)
    assert row is not None and row_clean is not None
    assert row["atm_iv"] == pytest.approx(row_clean["atm_iv"])  # duplicate is an exact copy, mean is unchanged


def test_expiry_day_itself_is_included_not_silently_dropped():
    """days_to_expiry == 0 (decision date is the expiry date) must still be
    eligible: MAX_DAYS_TO_EXPIRY uses <=, not <."""
    expiry = date(2024, 1, 10)
    chain = _chain_for_date(expiry, forward=100.0, expiry=expiry)
    row = build_atm_iv_row(chain, expiry)
    assert row is not None
    assert row["days_to_expiry"] == 0


def test_nearest_expiry_band_never_crosses_into_a_further_expiry():
    """When two expiries are both within the lookahead window, only the
    nearer one's strikes are ever selected -- a join bug here would blend
    two different expiries' smiles into one 'ATM' row."""
    near_expiry = date(2024, 1, 10)
    far_expiry = date(2024, 1, 17)
    on = date(2024, 1, 5)
    near_leg = _chain_for_date(on, forward=100.0, expiry=near_expiry)
    far_leg = _chain_for_date(on, forward=110.0, expiry=far_expiry)  # different forward entirely
    combined = pd.concat([near_leg, far_leg], ignore_index=True)

    band = nearest_expiry_band(combined)
    assert (band["expiry"] == pd.Timestamp(near_expiry)).all()
    assert not (band["expiry"] == pd.Timestamp(far_expiry)).any()
