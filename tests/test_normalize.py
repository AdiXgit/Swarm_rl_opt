"""Numerical and contract tests for the bhavcopy normalizer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swarm_opt.data.contracts import DTYPES, NORMALIZED_COLUMNS, PRIMARY_KEY
from swarm_opt.data.normalize import (
    SchemaError,
    _parity_forward,
    detect_schema,
    infer_lot_size,
    normalize_bhavcopy,
)


# --------------------------------------------------------------------------- #
# schema detection
# --------------------------------------------------------------------------- #
def test_detects_both_nse_schemas(udiff_raw, legacy_raw):
    assert detect_schema(udiff_raw) == "udiff"
    assert detect_schema(legacy_raw) == "legacy"


def test_unknown_schema_raises():
    with pytest.raises(SchemaError):
        detect_schema(pd.DataFrame({"foo": [1]}))


def test_legacy_without_spot_raises(legacy_raw):
    # The legacy file carries no underlying price; silently defaulting one
    # would corrupt every moneyness and Greek downstream.
    with pytest.raises(SchemaError):
        normalize_bhavcopy(legacy_raw, "NIFTY", spot=None)


# --------------------------------------------------------------------------- #
# contract conformance
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def udiff_norm(udiff_raw):
    return normalize_bhavcopy(udiff_raw, "NIFTY")


@pytest.fixture(scope="module")
def legacy_norm(legacy_raw, legacy_spot):
    return normalize_bhavcopy(legacy_raw, "NIFTY", spot=legacy_spot)


@pytest.mark.parametrize("frame", ["udiff_norm", "legacy_norm"])
def test_matches_data_contract(frame, request):
    df = request.getfixturevalue(frame)
    assert list(df.columns) == NORMALIZED_COLUMNS
    for col, dtype in DTYPES.items():
        assert str(df[col].dtype) == dtype, f"{col}: {df[col].dtype} != {dtype}"
    assert not df.empty


@pytest.mark.parametrize("frame", ["udiff_norm", "legacy_norm"])
def test_primary_key_is_unique(frame, request):
    df = request.getfixturevalue(frame)
    assert not df.duplicated(PRIMARY_KEY).any()


@pytest.mark.parametrize("frame", ["udiff_norm", "legacy_norm"])
def test_only_option_rows_survive(frame, request):
    df = request.getfixturevalue(frame)
    assert set(df["option_type"].unique()) <= {"CE", "PE"}
    assert (df["strike"] > 0).all()
    assert (df["underlying"] == "NIFTY").all()


@pytest.mark.parametrize("frame", ["udiff_norm", "legacy_norm"])
def test_single_trade_date_and_no_expired_contracts(frame, request):
    df = request.getfixturevalue(frame)
    assert df["date"].nunique() == 1
    # An EOD chain may include the contract expiring today, never one already gone.
    assert (df["days_to_expiry"] >= 0).all()
    assert (df["expiry"] >= df["date"]).all()


@pytest.mark.parametrize("frame", ["udiff_norm", "legacy_norm"])
def test_iv_is_absent_not_invented(frame, request):
    # NSE publishes no IV in either schema. DATA_SPEC: log, never fabricate.
    df = request.getfixturevalue(frame)
    assert df["implied_volatility"].isna().all()


@pytest.mark.parametrize("frame", ["udiff_norm", "legacy_norm"])
def test_ohlc_ordering_on_traded_rows(frame, request):
    df = request.getfixturevalue(frame)
    t = df[df["is_traded"]]
    assert (t["high"] >= t["low"]).all()
    assert (t["high"] >= t["open"]).all()
    assert (t["low"] <= t["open"]).all()


@pytest.mark.parametrize("frame", ["udiff_norm", "legacy_norm"])
def test_is_traded_tracks_volume(frame, request):
    df = request.getfixturevalue(frame)
    assert (df["is_traded"] == (df["volume"] > 0)).all()


# --------------------------------------------------------------------------- #
# tradability filter (approved decision 4)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("frame", ["udiff_norm", "legacy_norm"])
def test_tradability_filter_is_applied_to_both_eras(frame, request):
    df = request.getfixturevalue(frame)
    assert (df["open_interest"] > 0).all()


def test_filter_equalises_the_two_eras(legacy_raw, legacy_spot, udiff_raw):
    """The filter exists to stop the eras exposing different universes.

    Unfiltered, the legacy era carries dead zero-OI strikes that NseKit already
    strips from the modern era. Filtered, both eras keep essentially only rows
    that carry a position.
    """
    leg_open = normalize_bhavcopy(legacy_raw, "NIFTY", spot=legacy_spot, min_open_interest=0)
    leg_filt = normalize_bhavcopy(legacy_raw, "NIFTY", spot=legacy_spot, min_open_interest=1)
    ud_open = normalize_bhavcopy(udiff_raw, "NIFTY", min_open_interest=0)
    ud_filt = normalize_bhavcopy(udiff_raw, "NIFTY", min_open_interest=1)

    leg_drop = 1 - len(leg_filt) / len(leg_open)
    ud_drop = 1 - len(ud_filt) / len(ud_open)
    # the legacy era is where the dead strikes live
    assert leg_drop > 0.2
    assert ud_drop < 0.02


def test_filter_keeps_every_tradeable_strike(legacy_raw, legacy_spot):
    """It must remove dead contracts without narrowing the strike ladder a
    spread or condor selects from."""
    unfiltered = normalize_bhavcopy(legacy_raw, "NIFTY", spot=legacy_spot, min_open_interest=0)
    filtered = normalize_bhavcopy(legacy_raw, "NIFTY", spot=legacy_spot, min_open_interest=1)

    def band(df):
        near = df[df["days_to_expiry"] <= 45]
        near = near[near["forward"].notna()]
        inside = abs(near["strike"] - near["forward"]) <= 0.10 * near["forward"]
        return set(near.loc[inside, "strike"])

    assert band(filtered) == band(unfiltered)


# --------------------------------------------------------------------------- #
# lot size
# --------------------------------------------------------------------------- #
def test_lot_size_reconciles_from_turnover(legacy_norm, udiff_norm):
    # NIFTY traded a lot of 50 in Jul-2023 and 65 in Aug-2026; the legacy value
    # is recovered from turnover alone, the UDiFF one is published per row.
    assert legacy_norm["lot_size"].unique().tolist() == [50]
    assert udiff_norm["lot_size"].unique().tolist() == [65]


@pytest.mark.parametrize("frame", ["udiff_norm", "legacy_norm"])
def test_turnover_identity_holds(frame, request):
    """NSE options turnover is notional on (strike + premium), not on spot.

    Getting this wrong silently rescales every volume/turnover feature away
    from the money, so it is pinned on both schemas.
    """
    df = request.getfixturevalue(frame)
    t = df[(df["volume"] > 500) & (df["turnover_notional"] > 0)]
    unit = t["turnover_notional"] / (t["volume"] * t["lot_size"])
    assert np.allclose(unit, t["strike"] + t["close"], rtol=0.01)
    # the spot-based reading is wrong away from the money; make sure we notice
    assert not np.allclose(unit, t["spot"], rtol=0.01)


def test_lot_size_refuses_to_guess():
    n = 60
    contracts = pd.Series([100.0] * n)
    strike = pd.Series([20000.0] * n)
    premium = pd.Series([0.0] * n)
    notional = pd.Series([100.0 * 37.5 * 20000.0] * n)  # lot 37.5: not admissible
    with pytest.raises(SchemaError):
        infer_lot_size(contracts, notional, strike, premium)


def test_lot_size_is_exact_away_from_the_money():
    n = 60
    contracts = pd.Series(np.linspace(1000, 5000, n))
    strike = pd.Series(np.linspace(15000.0, 25000.0, n))
    premium = pd.Series(np.linspace(0.5, 4000.0, n))
    notional = contracts * 50 * (strike + premium)
    assert infer_lot_size(contracts, notional, strike, premium) == 50


# --------------------------------------------------------------------------- #
# forward / put-call parity
# --------------------------------------------------------------------------- #
def test_parity_forward_recovers_listed_future(udiff_norm):
    """The strongest available consistency check on the option marks.

    For an expiry that has a listed future, the forward implied by put-call
    parity must land on the futures close. It also lands ~70pts above the index
    spot, which is why IV must be solved against the forward, not the spot.
    """
    near = udiff_norm[udiff_norm["forward_source"] == "futures"]
    expiry = near["expiry"].min()
    chain = udiff_norm[udiff_norm["expiry"] == expiry]

    listed = float(chain["forward"].iloc[0])
    spot = float(chain["spot"].iloc[0])
    parity = _parity_forward(chain, spot)

    assert np.isfinite(parity)
    assert abs(parity - listed) / listed < 0.002, (parity, listed)
    # and the spot is materially different, so the distinction is not cosmetic
    assert abs(listed - spot) > 20


@pytest.mark.parametrize("frame", ["udiff_norm", "legacy_norm"])
def test_forward_is_labelled_never_silently_filled(frame, request):
    df = request.getfixturevalue(frame)
    labels = set(df["forward_source"].unique())
    assert labels <= {"futures", "put_call_parity", "interpolated", "missing"}
    assert df.loc[df["forward_source"] == "missing", "forward"].isna().all()
    assert df.loc[df["forward_source"] != "missing", "forward"].notna().all()


@pytest.mark.parametrize("frame", ["udiff_norm", "legacy_norm"])
def test_weekly_tradeable_expiries_all_have_a_forward(frame, request):
    """Every expiry a weekly episode could reach must be priceable.

    Far-dated LEAPS may stay unpriced; the nearest expiries may not, in either
    era, or trade construction would have to skip a decision date.
    """
    df = request.getfixturevalue(frame)
    near = df[df["days_to_expiry"] <= 45]
    assert near["forward"].notna().all()
    assert (near["forward_source"] != "missing").all()
    assert near["expiry"].nunique() >= 3


@pytest.mark.parametrize("frame", ["udiff_norm", "legacy_norm"])
def test_forward_curve_is_monotone_in_expiry(frame, request):
    """Carry is positive on NIFTY, so the forward curve must slope up.

    This is what catches a bad parity fit or a mis-interpolated expiry: a
    stale wing strike would kink the curve.
    """
    df = request.getfixturevalue(frame)
    curve = (
        df[df["forward_source"] != "missing"]
        .groupby("expiry")["forward"].first().sort_index()
    )
    assert curve.is_monotonic_increasing, curve.to_dict()
    assert curve.iloc[0] >= df["spot"].iloc[0] - 50


def test_interpolated_forward_lands_between_its_neighbours(legacy_norm):
    """The 2023-08-17 expiry is too thin to price directly once dead strikes
    are filtered out; it must be recovered from the carry curve, not dropped."""
    curve = legacy_norm.groupby("expiry").agg(
        src=("forward_source", "first"),
        fwd=("forward", "first"),
        dte=("days_to_expiry", "first"),
    ).sort_values("dte")
    interp = curve[(curve["src"] == "interpolated") & (curve["dte"] <= 45)]
    assert len(interp) >= 1

    for expiry, row in interp.iterrows():
        before = curve[curve["dte"] < row["dte"]]["fwd"].iloc[-1]
        after = curve[curve["dte"] > row["dte"]]["fwd"].iloc[0]
        assert before < row["fwd"] < after, (expiry, before, row["fwd"], after)


def test_forward_is_never_extrapolated_past_observed_expiries(udiff_norm):
    # Far-dated LEAPS beyond the last priced expiry stay missing on purpose.
    priced = udiff_norm[udiff_norm["forward_source"] != "missing"]
    missing = udiff_norm[udiff_norm["forward_source"] == "missing"]
    if len(missing):
        assert missing["days_to_expiry"].min() > priced["days_to_expiry"].max() - 1


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #
def test_normalization_is_deterministic(udiff_raw):
    a = normalize_bhavcopy(udiff_raw, "NIFTY")
    b = normalize_bhavcopy(udiff_raw, "NIFTY")
    pd.testing.assert_frame_equal(a, b)
