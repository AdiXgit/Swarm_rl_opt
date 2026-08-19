from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

FIXTURE_ROOT = ROOT / "data" / "processed" / "fixture_nifty"
RAW_DIR = ROOT / "data" / "raw"

UDIFF_RAW = RAW_DIR / "fo_bhavcopy_18082026.parquet"
LEGACY_RAW = RAW_DIR / "fo_bhavcopy_legacy_20230717.parquet"

#: NIFTY 50 close on 2023-07-17, from the index series (legacy bhavcopy has no spot).
LEGACY_SPOT = 19711.45


@pytest.fixture(scope="session")
def udiff_raw() -> pd.DataFrame:
    if not UDIFF_RAW.exists():
        pytest.skip(f"missing raw sample {UDIFF_RAW}; run scripts/validate_nsekit.py")
    return pd.read_parquet(UDIFF_RAW)


@pytest.fixture(scope="session")
def legacy_raw() -> pd.DataFrame:
    if not LEGACY_RAW.exists():
        pytest.skip(f"missing raw sample {LEGACY_RAW}")
    return pd.read_parquet(LEGACY_RAW)


@pytest.fixture(scope="session")
def legacy_spot() -> float:
    return LEGACY_SPOT


@pytest.fixture(scope="session")
def fixture_root() -> Path:
    if not (FIXTURE_ROOT / "chains").exists():
        pytest.skip("fixture not built; run scripts/build_fixture.py")
    return FIXTURE_ROOT
