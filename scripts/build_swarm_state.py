"""Load saved Phase 2 artifacts and build one SWARM_STATE_V1 for a historical
observation date. Never retrains; fails clearly on missing/incompatible
artifacts or an unsupported input date.

    python scripts/build_swarm_state.py --date 2025-06-05
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
from swarm_opt.data.providers.local_parquet_provider import LocalParquetProvider  # noqa: E402
from swarm_opt.data.splits import DATASET_END, DATASET_START  # noqa: E402
from swarm_opt.features.manifest import FeatureManifest  # noqa: E402
from swarm_opt.features.pointintime import FEATURE_NAMES, build_market_features  # noqa: E402
from swarm_opt.state import build_swarm_state  # noqa: E402

DEFAULT_ROOT = ROOT / "data" / "processed" / "nifty_eod"
DEFAULT_ARTIFACTS = ROOT / "data" / "models"
DEFAULT_ATM_IV = ROOT / "data" / "features" / "atm_iv_history.parquet"

AGENT_CLASSES = {
    "direction": DirectionAgent,
    "volatility": VolatilityAgent,
    "regime": RegimeAgent,
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date", required=True, type=date.fromisoformat)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    p.add_argument("--atm-iv", type=Path, default=DEFAULT_ATM_IV)
    p.add_argument("--underlying", default="NIFTY")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not (DATASET_START <= args.date <= DATASET_END):
        raise SystemExit(f"{args.date} is outside the frozen dataset range [{DATASET_START}, {DATASET_END}]")

    manifest_path = args.artifacts / "feature_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"feature manifest not found: {manifest_path}; run scripts/train_agents.py first")
    feature_manifest = FeatureManifest.load(manifest_path)

    provider = LocalParquetProvider(args.root)
    hist = provider.underlying_history(args.underlying, DATASET_START, args.date)
    if hist.empty or pd.Timestamp(args.date) not in set(hist["date"]):
        raise SystemExit(f"{args.date} is not a known trading date for {args.underlying}")

    atm_iv_hist = pd.read_parquet(args.atm_iv) if args.atm_iv.exists() else None
    if atm_iv_hist is not None:
        atm_iv_hist = atm_iv_hist[atm_iv_hist["date"] <= pd.Timestamp(args.date)]

    features, _ = build_market_features(hist, atm_iv_hist=atm_iv_hist)
    feature_manifest.assert_compatible(FEATURE_NAMES, feature_manifest.feature_schema_version)
    row = features[features["date"] == pd.Timestamp(args.date)]
    if row.empty:
        raise SystemExit(f"no feature row for {args.date}")
    X = row[FEATURE_NAMES]
    ts = pd.Series([pd.Timestamp(args.date)])

    agent_predictions = {}
    for name, cls in AGENT_CLASSES.items():
        agent_dir = args.artifacts / name
        if not agent_dir.exists():
            raise SystemExit(f"missing agent artifact: {agent_dir}; run scripts/train_agents.py first")
        agent = cls.load(agent_dir)
        if agent.feature_schema_version != feature_manifest.feature_schema_version:
            raise SystemExit(
                f"{name}: agent feature_schema_version {agent.feature_schema_version!r} "
                f"incompatible with current {feature_manifest.feature_schema_version!r}"
            )
        agent_predictions[name] = agent.predict(X, ts)[0]

    chain = provider.option_chain(args.underlying, args.date)
    greeks_result = compute_greeks(chain, pd.Timestamp(args.date), args.underlying)

    market_features = {c: (None if pd.isna(row.iloc[0][c]) else float(row.iloc[0][c])) for c in FEATURE_NAMES}
    market_features["has_full_lookback"] = bool(row.iloc[0]["has_full_lookback"])
    market_features["atm_iv_available"] = bool(row.iloc[0]["atm_iv_available"])

    state = build_swarm_state(
        observation_timestamp=args.date,
        underlying=args.underlying,
        market_features=market_features,
        agent_predictions=agent_predictions,
        greeks_result=greeks_result,
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
