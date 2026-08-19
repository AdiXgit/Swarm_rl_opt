"""Provider-contract and no-lookahead tests.

These run entirely against the local fixture: no network, so they are
deterministic and cannot regress because NSE changed an archive.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from swarm_opt.data.contracts import NORMALIZED_COLUMNS
from swarm_opt.data.providers.base import DataProvider
from swarm_opt.data.providers.local_parquet_provider import LocalParquetProvider

SAMPLE = date(2026, 8, 17)
LEGACY_SAMPLE = date(2023, 7, 17)
SPAN = (date(2023, 1, 1), date(2027, 1, 1))


@pytest.fixture(scope="module")
def provider(fixture_root) -> LocalParquetProvider:
    return LocalParquetProvider(fixture_root)


def test_implements_the_provider_contract(provider):
    assert isinstance(provider, DataProvider)
    assert provider.provider_id == "local_parquet"


def test_trading_dates_are_sorted_and_bounded(provider):
    dates = provider.trading_dates(*SPAN)
    assert dates == sorted(dates)
    assert len(dates) == 10  # 5 legacy-era + 5 udiff-era sample dates
    assert all(SPAN[0] <= d <= SPAN[1] for d in dates)


def test_trading_dates_excludes_nse_holidays(provider):
    # 2026-08-15 is Independence Day; it must not appear as a trading date.
    assert date(2026, 8, 15) not in provider.trading_dates(*SPAN)


@pytest.mark.parametrize("on", [SAMPLE, LEGACY_SAMPLE])
def test_chain_conforms_and_is_stamped_with_its_own_date(provider, on):
    chain = provider.option_chain("NIFTY", on)
    assert list(chain.columns) == NORMALIZED_COLUMNS
    assert not chain.empty
    assert (chain["date"] == pd.Timestamp(on)).all()


def test_missing_date_yields_empty_typed_frame(provider):
    chain = provider.option_chain("NIFTY", date(2026, 8, 15))
    assert chain.empty
    assert list(chain.columns) == NORMALIZED_COLUMNS


# --------------------------------------------------------------------------- #
# leakage
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("on", [SAMPLE, LEGACY_SAMPLE])
def test_no_expiry_predates_the_decision_date(provider, on):
    """docs/DATA_SPEC.md: expiry availability must be point-in-time."""
    assert all(e >= on for e in provider.expiries("NIFTY", on))


def test_chain_carries_no_future_observations(provider):
    """A decision date must not see any row stamped after it."""
    for on in provider.trading_dates(*SPAN):
        chain = provider.option_chain("NIFTY", on)
        assert (chain["date"] <= pd.Timestamp(on)).all()


def test_expiry_universe_only_grows_with_time(provider):
    """Weekly listings roll forward; a later date must not lose a live expiry."""
    dates = [d for d in provider.trading_dates(*SPAN) if d.year == 2026]
    for earlier, later in zip(dates, dates[1:]):
        live = {e for e in provider.expiries("NIFTY", earlier) if e > later}
        assert live <= set(provider.expiries("NIFTY", later))


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #
def test_repeated_reads_are_identical(provider):
    a = provider.option_chain("NIFTY", SAMPLE)
    b = provider.option_chain("NIFTY", SAMPLE)
    pd.testing.assert_frame_equal(a, b)


def test_underlying_history_is_sorted_and_windowed(provider):
    hist = provider.underlying_history("NIFTY", *SPAN)
    assert list(hist.columns) == ["date", "open", "high", "low", "close"]
    assert hist["date"].is_monotonic_increasing
    assert (hist["high"] >= hist["low"]).all()

    narrow = provider.underlying_history("NIFTY", date(2026, 8, 1), date(2026, 8, 31))
    assert len(narrow) == 5


def test_spot_matches_the_index_series(provider):
    """The chain's spot must agree with the standalone index close."""
    hist = provider.underlying_history("NIFTY", *SPAN).set_index("date")["close"]
    for on in provider.trading_dates(*SPAN):
        chain = provider.option_chain("NIFTY", on)
        assert abs(float(chain["spot"].iloc[0]) - float(hist[pd.Timestamp(on)])) < 1.0
