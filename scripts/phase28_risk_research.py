"""Phase 2.8 Workstream A: Greek-aware risk RESEARCH, not a Risk Engine.

Builds a handful of synthetic multi-leg positions at a real, historical
ATM strike/forward/IV (read from the existing VALIDATION-period chain data,
read-only), aggregates their portfolio Greeks via
`swarm_opt.agents.portfolio_greeks`, and checks them against a few
illustrative, clearly-labeled exposure thresholds -- purely to establish
whether the aggregation machinery produces sane, decision-useful numbers.
This is a unit-of-analysis research script, not a production Risk Engine,
not the Phase 3 Decision Engine, and it recommends no trade.

    python scripts/phase28_risk_research.py --date 2025-06-05
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

from swarm_opt.agents.portfolio_greeks import OptionPosition, aggregate_greeks  # noqa: E402
from swarm_opt.data.providers.local_parquet_provider import LocalParquetProvider  # noqa: E402
from swarm_opt.features.enrich import enrich_chain  # noqa: E402

DEFAULT_ROOT = ROOT / "data" / "processed" / "nifty_eod"
OUT = ROOT / "data" / "models" / "phase2_8" / "risk_research_illustrative_run.json"

#: Illustrative thresholds only -- not calibrated against real margin,
#: capital, or risk-appetite data. Labeled explicitly as such throughout.
ILLUSTRATIVE_NET_DELTA_LIMIT_UNITS = 2 * 75  # ~ 2 lots of pure directional exposure
ILLUSTRATIVE_GAMMA_CONCENTRATION_DTE = 2      # "near expiry" for the gamma-concentration flag


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date", required=True, type=date.fromisoformat)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return p.parse_args(argv)


def build_leg(enriched: pd.DataFrame, strike: float, option_type: str, side: str, qty: int, lot: int) -> OptionPosition | None:
    row = enriched[(enriched["strike"] == strike) & (enriched["option_type"] == option_type)]
    if row.empty:
        return None
    r = row.iloc[0]
    import numpy as np
    valid = bool(np.isfinite(r["implied_volatility"])) and all(np.isfinite(r[g]) for g in ("delta", "gamma", "theta", "vega"))
    return OptionPosition(
        option_type=option_type, strike=float(strike), expiry=str(r["expiry"].date()),
        side=side, quantity_contracts=qty, lot_size=lot,
        delta=float(r["delta"]) if valid else float("nan"),
        gamma=float(r["gamma"]) if valid else float("nan"),
        theta=float(r["theta"]) if valid else float("nan"),
        vega=float(r["vega"]) if valid else float("nan"),
        valid=valid,
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    provider = LocalParquetProvider(args.root)
    chain = provider.option_chain("NIFTY", args.date)
    if chain.empty:
        raise SystemExit(f"no chain for {args.date}")

    near = chain[chain["days_to_expiry"] <= 10]
    if near.empty:
        raise SystemExit(f"no near-expiry chain for {args.date}")
    nearest_expiry = near["expiry"].min()
    leg_chain = near[near["expiry"] == nearest_expiry]
    spot = float(leg_chain["spot"].iloc[0])
    lot_size = int(leg_chain["lot_size"].iloc[0])
    enriched = enrich_chain(leg_chain)
    atm_strike = float(enriched.loc[(enriched["strike"] - spot).abs().idxmin(), "strike"])

    strikes = sorted(enriched["strike"].unique())
    idx = strikes.index(atm_strike)
    wing_up = strikes[min(idx + 4, len(strikes) - 1)]
    wing_dn = strikes[max(idx - 4, 0)]

    candidates = {
        "long_straddle": [
            build_leg(enriched, atm_strike, "CE", "LONG", 1, lot_size),
            build_leg(enriched, atm_strike, "PE", "LONG", 1, lot_size),
        ],
        "short_straddle": [
            build_leg(enriched, atm_strike, "CE", "SHORT", 1, lot_size),
            build_leg(enriched, atm_strike, "PE", "SHORT", 1, lot_size),
        ],
        "iron_condor": [
            build_leg(enriched, atm_strike, "CE", "SHORT", 1, lot_size),
            build_leg(enriched, wing_up, "CE", "LONG", 1, lot_size),
            build_leg(enriched, atm_strike, "PE", "SHORT", 1, lot_size),
            build_leg(enriched, wing_dn, "PE", "LONG", 1, lot_size),
        ],
        "naked_long_call": [build_leg(enriched, atm_strike, "CE", "LONG", 3, lot_size)],
    }

    results = {}
    for name, legs in candidates.items():
        legs = [l for l in legs if l is not None]
        if not legs:
            results[name] = {"status": "no priceable legs at this strike/date"}
            continue
        port = aggregate_greeks(legs)
        flags = []
        if abs(port.delta) > ILLUSTRATIVE_NET_DELTA_LIMIT_UNITS:
            flags.append(f"net delta {port.delta:.1f} exceeds illustrative limit "
                         f"{ILLUSTRATIVE_NET_DELTA_LIMIT_UNITS} (NOT a real risk limit)")
        if int(leg_chain["days_to_expiry"].iloc[0]) <= ILLUSTRATIVE_GAMMA_CONCENTRATION_DTE and abs(port.gamma) > 0:
            flags.append(f"gamma concentration near expiry (dte<= {ILLUSTRATIVE_GAMMA_CONCENTRATION_DTE}), "
                         f"net gamma={port.gamma:.4f} (illustrative flag only)")
        results[name] = {
            "legs": [l.to_dict() for l in legs],
            "portfolio_greeks": port.to_dict(),
            "illustrative_flags": flags,
        }
        print(f"{name}: delta={port.delta:.2f} gamma={port.gamma:.4f} theta={port.theta:.2f} "
              f"vega={port.vega:.2f} flags={flags}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "observation_date": args.date.isoformat(),
        "disclaimer": (
            "Illustrative research only. Thresholds are not calibrated against real "
            "margin, capital, or risk-appetite data. This is not a Risk Engine, Decision "
            "Engine, or Trade Constructor. Correct Greeks do not imply profitable trading."
        ),
        "candidates": results,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
