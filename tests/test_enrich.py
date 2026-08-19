"""IV/Greeks enrichment against real chains from both schema eras."""
from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from swarm_opt.data.providers.local_parquet_provider import LocalParquetProvider
from swarm_opt.features.enrich import GREEK_COLUMNS, atm_summary, enrich_chain

UDIFF_DATE = date(2026, 8, 18)
LEGACY_DATE = date(2023, 7, 17)


@pytest.fixture(scope="module")
def provider(fixture_root):
    return LocalParquetProvider(fixture_root)


@pytest.fixture(scope="module", params=[UDIFF_DATE, LEGACY_DATE], ids=["udiff", "legacy"])
def enriched(provider, request):
    return enrich_chain(provider.option_chain("NIFTY", request.param))


def test_enrichment_preserves_every_row(provider, enriched):
    """Unsolvable rows are labelled, never dropped: trade construction still
    needs to know the contract exists and the simulator can still mark it."""
    assert len(enriched) == len(provider.option_chain("NIFTY", enriched["date"].iloc[0].date()))
    for col in GREEK_COLUMNS + ["moneyness", "log_moneyness", "iv_source"]:
        assert col in enriched.columns


def test_most_of_the_tradeable_chain_prices(enriched):
    near = enriched[enriched["days_to_expiry"] <= 45]
    assert near["implied_volatility"].notna().mean() > 0.6


def test_solved_vols_are_plausible_for_nifty(enriched):
    iv = enriched.loc[enriched["implied_volatility"].notna(), "implied_volatility"]
    assert (iv > 0).all()
    assert iv.median() < 1.0


def test_unsolved_rows_are_labelled_and_have_no_greeks(enriched):
    assert set(enriched["iv_source"].dropna().unique()) <= {
        "black76_close", "no_forward", "outside_arb_bounds"
    }
    unsolved = enriched[enriched["implied_volatility"].isna()]
    if len(unsolved):
        assert (unsolved["iv_source"] != "black76_close").all()
        for col in GREEK_COLUMNS:
            assert unsolved[col].isna().all()


def test_greeks_present_wherever_iv_solved(enriched):
    solved = enriched[enriched["implied_volatility"].notna()]
    for col in GREEK_COLUMNS:
        assert solved[col].notna().all()


def test_greek_signs_follow_the_option_side(enriched):
    solved = enriched[enriched["implied_volatility"].notna()]
    calls = solved[solved["option_type"] == "CE"]
    puts = solved[solved["option_type"] == "PE"]

    assert (calls["delta"] >= 0).all() and (calls["delta"] <= 1).all()
    assert (puts["delta"] <= 0).all() and (puts["delta"] >= -1).all()
    assert (solved["gamma"] >= 0).all()
    assert (solved["vega"] >= 0).all()
    assert (solved["theta"] <= 0).all()


def test_moneyness_is_measured_against_the_forward(enriched):
    """Not against spot: that is the whole point of the Black-76 decision."""
    solved = enriched[enriched["forward"].notna()]
    assert np.allclose(solved["moneyness"], solved["strike"] / solved["forward"])
    atm = solved.iloc[(solved["moneyness"] - 1.0).abs().argsort()[:1]]
    assert abs(float(atm["log_moneyness"].iloc[0])) < 0.01


def test_atm_delta_is_near_half(enriched):
    """The ATM strike of the front expiry must come out near 0.5 delta.

    If IV had been solved against spot instead of the forward, the ~70-point
    offset would push this measurably away from a half.
    """
    near = enriched[enriched["implied_volatility"].notna()]
    expiry = near.loc[near["days_to_expiry"] > 0, "days_to_expiry"].min()
    front = near[near["days_to_expiry"] == expiry]
    forward = float(front["forward"].iloc[0])

    calls = front[front["option_type"] == "CE"]
    atm = calls.iloc[(calls["strike"] - forward).abs().argsort()[:1]]
    assert float(atm["delta"].iloc[0]) == pytest.approx(0.5, abs=0.06)


# --------------------------------------------------------------------------- #
# surface summary
# --------------------------------------------------------------------------- #
def test_atm_summary_covers_the_near_expiries(enriched):
    summary = atm_summary(enriched)
    assert not summary.empty
    assert summary["days_to_expiry"].is_monotonic_increasing
    near = summary[summary["days_to_expiry"].between(1, 45)]
    assert len(near) >= 3
    assert near["atm_iv"].notna().all()


def test_term_structure_is_coherent(enriched):
    """ATM IV across the near expiries must sit in a tight band.

    A spot-based inversion produced a spurious 17.5% hump at one expiry where
    the neighbours were near 10%; this pins that class of error.
    """
    summary = atm_summary(enriched)
    near = summary[summary["days_to_expiry"].between(1, 45)]
    assert near["atm_iv"].between(0.03, 0.60).all()
    assert near["atm_iv"].max() / near["atm_iv"].min() < 2.0


def test_skew_is_finite_and_small(enriched):
    summary = atm_summary(enriched)
    near = summary[summary["days_to_expiry"].between(1, 45)]
    skew = near["skew_25d"].dropna()
    assert len(skew) >= 2
    assert skew.abs().max() < 0.25


def test_empty_chain_enriches_to_an_empty_frame(provider):
    out = enrich_chain(provider.option_chain("NIFTY", date(2026, 8, 15)))
    assert out.empty
    for col in GREEK_COLUMNS:
        assert col in out.columns
