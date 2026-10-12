"""Phase 2 end-to-end smoke test on a small chronological TRAIN-only subset.

Exercises the full pipeline fast: manifest check -> point-in-time features
and targets -> leakage audit -> train all three agents -> evaluate ->
save/reload artifacts -> predict on a later permitted date -> Greeks ->
SWARM_STATE_V1 -> validate -> serialize. Exits nonzero if any mandatory
assertion fails. Does not touch VALIDATION or TEST and is not a substitute
for `scripts/train_agents.py` (the full frozen-data run).

    python scripts/run_smoke_test.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from swarm_opt.agents.direction import DirectionAgent  # noqa: E402
from swarm_opt.agents.greeks_agent import compute_greeks  # noqa: E402
from swarm_opt.agents.regime import RegimeAgent  # noqa: E402
from swarm_opt.agents.volatility import VolatilityAgent  # noqa: E402
from swarm_opt.data.providers.local_parquet_provider import LocalParquetProvider  # noqa: E402
from swarm_opt.data.splits import TRAIN  # noqa: E402
from swarm_opt.features.audit import (  # noqa: E402
    assert_duplicate_free,
    assert_feature_label_alignment,
    assert_no_cross_split_labels,
    assert_no_future_features,
)
from swarm_opt.features.pointintime import FEATURE_NAMES, FEATURE_SCHEMA_VERSION  # noqa: E402
from swarm_opt.pipeline import build_dataset, make_xy  # noqa: E402
from swarm_opt.state import build_swarm_state, validate_swarm_state  # noqa: E402

DEFAULT_ROOT = ROOT / "data" / "processed" / "nifty_eod"
SUBSET_TRADING_DATES = 160  # enough for ~15-20 weekly episodes, well inside TRAIN
AGENTS = {
    "direction": DirectionAgent,
    "volatility": VolatilityAgent,
    "regime": RegimeAgent,
}
LABEL_COLUMNS = {
    "direction": "direction_label",
    "volatility": "volatility_label",
    "regime": "regime_label",
}


def main() -> int:
    report: dict = {"steps": []}

    def step(name: str, ok: bool, detail: str = "") -> None:
        report["steps"].append({"name": name, "ok": ok, "detail": detail})
        print(f"  [{'OK' if ok else 'FAIL'}] {name} {detail}")
        if not ok:
            raise SystemExit(json.dumps(report, indent=2, default=str))

    provider = LocalParquetProvider(DEFAULT_ROOT)
    all_dates = provider.trading_dates(TRAIN.start, TRAIN.end)
    subset_dates = all_dates[:SUBSET_TRADING_DATES]
    step("load small chronological TRAIN subset", len(subset_dates) > 100, f"{len(subset_dates)} dates")

    dataset = build_dataset(provider, "NIFTY", subset_dates[0], subset_dates[-1])
    step(
        "build point-in-time features and targets",
        len(dataset.features) == len(subset_dates) and len(dataset.episodes_kept) > 5,
        f"{len(dataset.features)} feature rows, {len(dataset.episodes_kept)} episodes",
    )

    assert_no_future_features(dataset.features)
    assert_feature_label_alignment(dataset.features, dataset.targets)
    assert_no_cross_split_labels(dataset.targets)
    assert_duplicate_free(dataset.features, "date")
    step("temporal leakage audit", True)

    # last 20% of episodes by decision date = "a permitted later observation"
    sorted_eps = sorted(dataset.targets["decision_date"])
    split_idx = int(len(sorted_eps) * 0.8)
    train_cutoff = sorted_eps[split_idx]
    train_targets = dataset.targets[dataset.targets["decision_date"] < train_cutoff]
    holdout_targets = dataset.targets[dataset.targets["decision_date"] >= train_cutoff]
    step(
        "chronological in-subset train/holdout split",
        len(train_targets) > 5 and len(holdout_targets) > 0,
        f"train={len(train_targets)} holdout={len(holdout_targets)}",
    )

    tmp_dir = Path(tempfile.mkdtemp(prefix="swarm_opt_smoke_"))
    agent_preds_holdout = {}
    for name, cls in AGENTS.items():
        label_col = LABEL_COLUMNS[name]
        X_train, y_train, ts_train = make_xy(dataset, train_targets, label_col)
        X_hold, y_hold, ts_hold = make_xy(dataset, holdout_targets, label_col)
        step(f"{name}: usable rows", len(X_train) > 5 and len(X_hold) > 0,
             f"train={len(X_train)} holdout={len(X_hold)}")

        agent = cls(seed=0)
        agent.fit(X_train, y_train, FEATURE_SCHEMA_VERSION, (str(subset_dates[0]), str(train_cutoff)))
        preds = agent.predict(X_hold, ts_hold)
        step(f"{name}: predictions valid", len(preds) == len(X_hold))

        agent.save(tmp_dir / name)
        reloaded = cls.load(tmp_dir / name)
        reloaded_preds = reloaded.predict(X_hold, ts_hold)
        reproducible = all(
            p1.predicted_class == p2.predicted_class and p1.probabilities == p2.probabilities
            for p1, p2 in zip(preds, reloaded_preds)
        )
        step(f"{name}: save/reload reproducible", reproducible)
        agent_preds_holdout[name] = (reloaded, X_hold, ts_hold)

    # build one swarm state for the last holdout observation
    last_idx = -1
    observation_date = pd.Timestamp(ts_hold.iloc[last_idx]).date()
    agent_predictions = {
        name: agent.predict(X_hold.iloc[[last_idx]], ts_hold.iloc[[last_idx]])[0]
        for name, (agent, X_hold, ts_hold) in agent_preds_holdout.items()
    }
    step("agent predictions for later permitted observation", True, str(observation_date))

    chain = provider.option_chain("NIFTY", observation_date)
    greeks_result = compute_greeks(chain, pd.Timestamp(observation_date), "NIFTY")
    step("greeks agent", greeks_result is not None)

    row = dataset.features[dataset.features["date"] == pd.Timestamp(observation_date)].iloc[0]
    market_features = {c: (None if pd.isna(row[c]) else float(row[c])) for c in FEATURE_NAMES}
    market_features["has_full_lookback"] = bool(row["has_full_lookback"])
    market_features["atm_iv_available"] = bool(row["atm_iv_available"])

    state = build_swarm_state(
        observation_timestamp=observation_date,
        underlying="NIFTY",
        market_features=market_features,
        agent_predictions=agent_predictions,
        greeks_result=greeks_result,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
    )
    errors = validate_swarm_state(state)
    step("SWARM_STATE_V1 built and validated", len(errors) == 0, str(errors))

    round_tripped = type(state).from_dict(json.loads(json.dumps(state.to_dict(), default=str)))
    step("SWARM_STATE_V1 serialization round-trip", round_tripped.to_dict() == json.loads(json.dumps(state.to_dict(), default=str)))

    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
