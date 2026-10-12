"""Phase 2.5: build one SWARM_STATE_V1 from the EXTENDED model artifacts.

Same contract as scripts/build_swarm_state.py, pointed at
data/models/extended_2019_2021/ and the pointintime_v2 feature schema, and
able to resolve any observation date across the combined (extension +
original) provider. Never retrains. Fails clearly on missing/incompatible
artifacts or an unsupported date.

    python scripts/build_swarm_state_extended.py --date 2025-06-05
    python scripts/build_swarm_state_extended.py --date 2019-06-10
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from swarm_opt.agents.direction import DirectionAgent  # noqa: E402
from swarm_opt.agents.greeks_agent import compute_greeks  # noqa: E402
from swarm_opt.agents.regime import RegimeAgent  # noqa: E402
from swarm_opt.agents.volatility import VolatilityAgent  # noqa: E402
from swarm_opt.data.providers.combined_provider import CombinedParquetProvider  # noqa: E402
from swarm_opt.data.providers.local_parquet_provider import LocalParquetProvider  # noqa: E402
from swarm_opt.data.splits import DATASET_END  # noqa: E402
from swarm_opt.features.manifest import FeatureManifest  # noqa: E402
from swarm_opt.features.pointintime import FEATURE_NAMES_V2, build_market_features  # noqa: E402
from swarm_opt.state import build_swarm_state  # noqa: E402

EXTENDED_TRAIN_START = date(2019, 2, 18)
EXTENSION_ROOT = ROOT / "data" / "processed" / "nifty_eod_ext_2019_2021"
ORIGINAL_ROOT = ROOT / "data" / "processed" / "nifty_eod"
EXTENSION_ATM_IV = ROOT / "data" / "features" / "atm_iv_history_ext_2019_2021.parquet"
ORIGINAL_ATM_IV = ROOT / "data" / "features" / "atm_iv_history.parquet"
ARTIFACTS = ROOT / "data" / "models" / "extended_2019_2021"

AGENT_CLASSES = {"direction": DirectionAgent, "volatility": VolatilityAgent, "regime": RegimeAgent}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date", required=True, type=date.fromisoformat)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not (EXTENDED_TRAIN_START <= args.date <= DATASET_END):
        raise SystemExit(f"{args.date} is outside the extended range [{EXTENDED_TRAIN_START}, {DATASET_END}]")

    manifest_path = ARTIFACTS / "feature_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"extended feature manifest not found: {manifest_path}; run scripts/train_agents_extended.py first")
    feature_manifest = FeatureManifest.load(manifest_path)

    provider = CombinedParquetProvider([LocalParquetProvider(EXTENSION_ROOT), LocalParquetProvider(ORIGINAL_ROOT)])
    hist = provider.underlying_history("NIFTY", EXTENDED_TRAIN_START, args.date)
    if hist.empty or pd.Timestamp(args.date) not in set(hist["date"]):
        raise SystemExit(f"{args.date} is not a known trading date")

    atm_iv = pd.concat(
        [pd.read_parquet(EXTENSION_ATM_IV), pd.read_parquet(ORIGINAL_ATM_IV)], ignore_index=True
    )
    atm_iv = atm_iv[atm_iv["date"] <= pd.Timestamp(args.date)].sort_values("date").reset_index(drop=True)

    features, _ = build_market_features(hist, atm_iv_hist=atm_iv, feature_names=FEATURE_NAMES_V2)
    feature_manifest.assert_compatible(FEATURE_NAMES_V2, feature_manifest.feature_schema_version)
    row = features[features["date"] == pd.Timestamp(args.date)]
    if row.empty:
        raise SystemExit(f"no feature row for {args.date}")
    X = row[FEATURE_NAMES_V2]
    ts = pd.Series([pd.Timestamp(args.date)])

    agent_predictions = {}
    for name, cls in AGENT_CLASSES.items():
        agent_dir = ARTIFACTS / name
        if not agent_dir.exists():
            raise SystemExit(f"missing extended agent artifact: {agent_dir}")
        agent = cls.load(agent_dir)
        if agent.feature_schema_version != feature_manifest.feature_schema_version:
            raise SystemExit(
                f"{name}: agent feature_schema_version {agent.feature_schema_version!r} "
                f"incompatible with {feature_manifest.feature_schema_version!r}"
            )
        agent_predictions[name] = agent.predict(X, ts)[0]

    chain = provider.option_chain("NIFTY", args.date)
    greeks_result = compute_greeks(chain, pd.Timestamp(args.date), "NIFTY")

    market_features = {c: (None if pd.isna(row.iloc[0][c]) else float(row.iloc[0][c])) for c in FEATURE_NAMES_V2}
    market_features["has_full_lookback"] = bool(row.iloc[0]["has_full_lookback"])
    market_features["atm_iv_available"] = bool(row.iloc[0]["atm_iv_available"])

    state = build_swarm_state(
        observation_timestamp=args.date, underlying="NIFTY", market_features=market_features,
        agent_predictions=agent_predictions, greeks_result=greeks_result,
        feature_schema_version=feature_manifest.feature_schema_version,
    )
    out_text = json.dumps(state.to_dict(), indent=2, default=str)
    if args.out:
        args.out.write_text(out_text)
        print(f"wrote {args.out}")
    else:
        print(out_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
