from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from swarm_opt.data.providers.combined_provider import CombinedParquetProvider
from swarm_opt.data.providers.local_parquet_provider import LocalParquetProvider


def _write_fixture(root, dates_closes):
    for on, close in dates_closes:
        path = root / "chains" / "underlying=NIFTY" / f"date={on.isoformat()}" / "part.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "date": [pd.Timestamp(on)], "underlying": ["NIFTY"], "spot": [close], "expiry": [pd.Timestamp(on)],
            "strike": [close], "option_type": ["CE"], "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0],
            "volume": [1], "open_interest": [1], "change_in_open_interest": [0], "implied_volatility": [0.2],
            "settle_price": [1.0], "forward": [close], "forward_source": ["futures"], "lot_size": [25],
            "turnover_notional": [1.0], "is_traded": [True], "days_to_expiry": [0], "source": ["test"],
            "schema_version": ["test"],
        }).to_parquet(path, index=False)
    (root / "underlying").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "date": [pd.Timestamp(d) for d, _ in dates_closes],
        "open": [c for _, c in dates_closes], "high": [c for _, c in dates_closes],
        "low": [c for _, c in dates_closes], "close": [c for _, c in dates_closes],
    }).to_parquet(root / "underlying" / "NIFTY.parquet", index=False)


@pytest.fixture
def early_provider(tmp_path):
    root = tmp_path / "early"
    _write_fixture(root, [(date(2020, 1, 1), 100.0), (date(2020, 1, 2), 101.0)])
    return LocalParquetProvider(root)


@pytest.fixture
def late_provider(tmp_path):
    root = tmp_path / "late"
    _write_fixture(root, [(date(2020, 1, 5), 110.0), (date(2020, 1, 6), 111.0)])
    return LocalParquetProvider(root)


def test_trading_dates_unions_both_members(early_provider, late_provider):
    combined = CombinedParquetProvider([early_provider, late_provider])
    dates = combined.trading_dates(date(2020, 1, 1), date(2020, 1, 10))
    assert dates == [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 5), date(2020, 1, 6)]


def test_underlying_history_unions_and_sorts(early_provider, late_provider):
    combined = CombinedParquetProvider([early_provider, late_provider])
    hist = combined.underlying_history("NIFTY", date(2020, 1, 1), date(2020, 1, 10))
    assert list(hist["date"].dt.date) == [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 5), date(2020, 1, 6)]
    assert hist["close"].tolist() == [100.0, 101.0, 110.0, 111.0]


def test_option_chain_routes_to_the_member_that_has_the_date(early_provider, late_provider):
    combined = CombinedParquetProvider([early_provider, late_provider])
    chain = combined.option_chain("NIFTY", date(2020, 1, 5))
    assert not chain.empty
    assert float(chain["spot"].iloc[0]) == 110.0


def test_option_chain_returns_empty_for_a_date_neither_member_has(early_provider, late_provider):
    combined = CombinedParquetProvider([early_provider, late_provider])
    chain = combined.option_chain("NIFTY", date(2020, 1, 3))
    assert chain.empty


def test_overlapping_members_raise_rather_than_silently_pick_one(tmp_path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    _write_fixture(root_a, [(date(2020, 1, 1), 100.0)])
    _write_fixture(root_b, [(date(2020, 1, 1), 999.0)])  # same date, different provider -- a real conflict
    combined = CombinedParquetProvider([LocalParquetProvider(root_a), LocalParquetProvider(root_b)])
    with pytest.raises(ValueError):
        combined.underlying_history("NIFTY", date(2020, 1, 1), date(2020, 1, 1))


def test_requires_at_least_one_provider():
    with pytest.raises(ValueError):
        CombinedParquetProvider([])
