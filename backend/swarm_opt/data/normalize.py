"""Map raw NSE F&O bhavcopy frames onto the normalized contract.

NSE serves two incompatible bhavcopy schemas and NseKit passes both through
unchanged:

* ``udiff``  (2024-01-02 onward, 34 cols) - carries the underlying price and
  the board lot on every row.
* ``legacy`` (through 2023-12-29, 16 cols) - carries neither, so the spot is
  joined from the index series and the lot size is reconciled from turnover.

Both encode volume as *contracts* and turnover as *notional INR*. The exact
NSE identity, verified on every traded row of both schemas, is

    turnover == contracts * lot * (strike + premium)

*not* ``contracts * lot * spot``; the two only coincide near the money.

Nothing here fabricates a field. Implied volatility is not published by NSE in
either schema and is emitted as NaN; it is solved for in the features layer
against the forward, never against the index spot (see `forward` below).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import DTYPES, NORMALIZED_COLUMNS

UDIFF_MARKER = "FinInstrmTp"
LEGACY_MARKER = "INSTRUMENT"

#: Board lots NSE has used for index derivatives. Used only to snap the
#: turnover-reconciled estimate on the legacy schema to an admissible value.
CANDIDATE_LOTS = (10, 15, 20, 25, 30, 40, 50, 65, 75)

#: Tradability filter, applied identically to both schema eras.
#:
#: The two eras do not arrive with the same row universe: NseKit drops rows
#: whose OI, change-in-OI, volume and turnover are all zero, but that filter is
#: positional and silently no-ops on the 16-column legacy frame. Left alone,
#: the legacy era would expose dead strikes that the modern era cannot see, and
#: a backtest spanning both would be selecting from two different universes.
#: Requiring open interest imposes one universe on both, and drops exactly the
#: contracts that carry no position and therefore could not be traded out of.
MIN_OPEN_INTEREST = 1

#: Minimum number of strikes with BOTH legs traded before a put-call-parity
#: forward is trusted. Below this the expiry is handed to the carry curve.
MIN_PARITY_PAIRS = 5


class SchemaError(ValueError):
    pass


def detect_schema(raw: pd.DataFrame) -> str:
    if UDIFF_MARKER in raw.columns:
        return "udiff"
    if LEGACY_MARKER in raw.columns:
        return "legacy"
    raise SchemaError(f"unrecognised bhavcopy schema: {list(raw.columns)[:8]}")


# --------------------------------------------------------------------------- #
# lot size
# --------------------------------------------------------------------------- #
def infer_lot_size(
    contracts: pd.Series,
    notional: pd.Series,
    strike: pd.Series,
    premium: pd.Series,
) -> int:
    """Recover the board lot from ``notional == contracts * lot * (strike + premium)``.

    That identity is exact at every strike, so this needs no spot and does not
    degrade away from the money. Uses the most heavily traded rows so a single
    stale print cannot move it, then snaps to the nearest admissible lot.
    Raises if the snap is not close, rather than guessing.
    """
    unit = strike + premium
    mask = (contracts > 0) & (notional > 0) & (unit > 0)
    if not mask.any():
        raise SchemaError("cannot reconcile lot size: no traded rows")

    top = contracts[mask].nlargest(50).index
    est = float(np.median(notional.loc[top] / (contracts.loc[top] * unit.loc[top])))
    lot = min(CANDIDATE_LOTS, key=lambda c: abs(c - est))
    if abs(est - lot) / lot > 0.05:
        raise SchemaError(f"lot size {est:.2f} does not match any known lot")
    return int(lot)


# --------------------------------------------------------------------------- #
# forward
# --------------------------------------------------------------------------- #
def _parity_forward(chain: pd.DataFrame, spot: float, window: float = 0.03) -> float:
    """Forward implied by put-call parity on near-ATM traded strikes.

    NIFTY options are priced off the future, not the index. Solving IV against
    the spot would manufacture a skew that is not in the market, so every expiry
    gets its own forward.

    Only rows that actually traded are used, with no fallback to untraded ones:
    an untraded contract keeps a stale close that can sit thousands of points
    off fair value, and a single such strike drags the fit. A thinly quoted
    expiry is therefore left to the carry curve rather than priced off stale
    marks -- which is what keeps the far-dated end of the curve monotone.
    """
    traded = chain[chain["volume"] > 0]
    both = traded.pivot_table(index="strike", columns="option_type", values="close")
    if not {"CE", "PE"}.issubset(both.columns):
        return float("nan")
    both = both.dropna()
    if len(both) < MIN_PARITY_PAIRS:
        return float("nan")

    near = both.loc[abs(both.index - spot) <= window * spot]
    if len(near) < MIN_PARITY_PAIRS:
        order = np.argsort(np.abs(both.index.to_numpy() - spot))[:8]
        near = both.iloc[order]
    if len(near) < MIN_PARITY_PAIRS:
        return float("nan")
    return float(np.median(near["CE"] - near["PE"] + near.index))


def _interpolate_carry(
    fwd: dict, src: dict, spot: float, date: pd.Timestamp
) -> tuple[dict, dict]:
    """Fill gaps in the forward curve along the implied carry term structure.

    ``F = S * exp(c * T)`` defines an implied carry ``c`` at every expiry that
    priced. ``c`` is a smooth, slowly varying function of ``T``, so a thinly
    quoted expiry sandwiched between two priced ones can be recovered by
    interpolating ``c`` rather than left unpriceable.

    This uses only forwards observed on the same date, so it introduces no
    lookahead; the result is labelled ``interpolated`` so it stays auditable and
    can be excluded downstream. Expiries outside the range of priced ones are
    left missing rather than extrapolated.
    """
    known = sorted(
        (e, f) for e, f in fwd.items() if np.isfinite(f) and src[e] != "missing"
    )
    if len(known) < 2 or not np.isfinite(spot) or spot <= 0:
        return fwd, src

    def years(e) -> float:
        return max((e - date).days, 1) / 365.0

    ts = np.array([years(e) for e, _ in known])
    cs = np.array([np.log(f / spot) / t for (_, f), t in zip(known, ts)])

    for expiry, f in fwd.items():
        if np.isfinite(f) and src[expiry] != "missing":
            continue
        t = years(expiry)
        if t < ts[0] or t > ts[-1]:
            continue  # do not extrapolate the curve past what was observed
        fwd[expiry] = float(spot * np.exp(float(np.interp(t, ts, cs)) * t))
        src[expiry] = "interpolated"

    return fwd, src


def attach_forward(options: pd.DataFrame, futures: pd.DataFrame) -> pd.DataFrame:
    """Per-expiry forward: listed future, else put-call parity, else carry curve."""
    fut_px = futures.set_index("expiry")["close"].to_dict() if len(futures) else {}

    fwd: dict = {}
    src: dict = {}
    for expiry, chain in options.groupby("expiry"):
        listed = fut_px.get(expiry)
        if listed is not None and np.isfinite(listed):
            fwd[expiry], src[expiry] = float(listed), "futures"
            continue
        f = _parity_forward(chain, float(chain["spot"].iloc[0]))
        fwd[expiry] = f
        src[expiry] = "put_call_parity" if np.isfinite(f) else "missing"

    spot = float(options["spot"].iloc[0])
    date = options["date"].iloc[0]
    fwd, src = _interpolate_carry(fwd, src, spot, date)

    out = options.copy()
    out["forward"] = out["expiry"].map(fwd)
    out["forward_source"] = out["expiry"].map(src)
    return out


# --------------------------------------------------------------------------- #
# schema mappers
# --------------------------------------------------------------------------- #
def _from_udiff(raw: pd.DataFrame, underlying: str):
    sel = raw[raw["TckrSymb"] == underlying]
    opt = sel[sel["FinInstrmTp"].isin(["IDO", "STO"])].copy()
    fut = sel[sel["FinInstrmTp"].isin(["IDF", "STF"])].copy()

    o = pd.DataFrame({
        "date": pd.to_datetime(opt["TradDt"]),
        "underlying": underlying,
        "spot": opt["UndrlygPric"].astype(float),
        "expiry": pd.to_datetime(opt["XpryDt"]),
        "strike": opt["StrkPric"].astype(float),
        "option_type": opt["OptnTp"].astype(str),
        "open": opt["OpnPric"].astype(float),
        "high": opt["HghPric"].astype(float),
        "low": opt["LwPric"].astype(float),
        "close": opt["ClsPric"].astype(float),
        "volume": opt["TtlTradgVol"].astype("int64"),
        "open_interest": opt["OpnIntrst"].astype("int64"),
        "change_in_open_interest": opt["ChngInOpnIntrst"].astype("int64"),
        "settle_price": opt["SttlmPric"].astype(float),
        "lot_size": opt["NewBrdLotQty"].astype("int64"),
        "turnover_notional": opt["TtlTrfVal"].astype(float),
    })
    f = pd.DataFrame({
        "expiry": pd.to_datetime(fut["XpryDt"]),
        "close": fut["ClsPric"].astype(float),
    })
    return o, f


def _from_legacy(raw: pd.DataFrame, underlying: str, spot: float):
    sel = raw[raw["SYMBOL"] == underlying]
    opt = sel[sel["INSTRUMENT"].isin(["OPTIDX", "OPTSTK"])].copy()
    fut = sel[sel["INSTRUMENT"].isin(["FUTIDX", "FUTSTK"])].copy()

    notional = opt["VAL_INLAKH"].astype(float) * 1e5
    lot = infer_lot_size(
        opt["CONTRACTS"].astype(float),
        notional,
        opt["STRIKE_PR"].astype(float),
        opt["CLOSE"].astype(float),
    )

    o = pd.DataFrame({
        "date": pd.to_datetime(opt["TIMESTAMP"], format="%d-%b-%Y"),
        "underlying": underlying,
        "spot": float(spot),
        "expiry": pd.to_datetime(opt["EXPIRY_DT"], format="%d-%b-%Y"),
        "strike": opt["STRIKE_PR"].astype(float),
        "option_type": opt["OPTION_TYP"].astype(str),
        "open": opt["OPEN"].astype(float),
        "high": opt["HIGH"].astype(float),
        "low": opt["LOW"].astype(float),
        "close": opt["CLOSE"].astype(float),
        "volume": opt["CONTRACTS"].astype("int64"),
        "open_interest": opt["OPEN_INT"].astype("int64"),
        "change_in_open_interest": opt["CHG_IN_OI"].astype("int64"),
        "settle_price": opt["SETTLE_PR"].astype(float),
        "lot_size": lot,
        "turnover_notional": notional.to_numpy(),
    })
    f = pd.DataFrame({
        "expiry": pd.to_datetime(fut["EXPIRY_DT"], format="%d-%b-%Y"),
        "close": fut["CLOSE"].astype(float),
    })
    return o, f


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def normalize_bhavcopy(
    raw: pd.DataFrame,
    underlying: str = "NIFTY",
    spot: float | None = None,
    source: str = "nsekit",
    min_open_interest: int = MIN_OPEN_INTEREST,
) -> pd.DataFrame:
    """Raw bhavcopy -> `contracts.NORMALIZED_COLUMNS`.

    `spot` is required for the legacy schema, which does not carry the
    underlying price; pass that date's index close.

    `min_open_interest` is the tradability filter and is applied identically to
    both schema eras, before the forward is computed. See MIN_OPEN_INTEREST.
    """
    schema = detect_schema(raw)
    if schema == "udiff":
        opts, futs = _from_udiff(raw, underlying)
    else:
        if spot is None:
            raise SchemaError("legacy bhavcopy needs an explicit `spot`")
        opts, futs = _from_legacy(raw, underlying, spot)

    if opts.empty:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS).astype(DTYPES)

    opts = opts[opts["option_type"].isin(["CE", "PE"])]
    opts = opts[opts["open_interest"] >= min_open_interest]
    if opts.empty:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS).astype(DTYPES)

    opts = attach_forward(opts, futs)

    # NSE publishes no IV in either schema. Left NaN, solved for downstream.
    opts["implied_volatility"] = np.nan
    opts["is_traded"] = opts["volume"] > 0
    opts["days_to_expiry"] = (opts["expiry"] - opts["date"]).dt.days
    opts["source"] = source
    opts["schema_version"] = schema

    out = opts[NORMALIZED_COLUMNS].astype(DTYPES)
    return out.sort_values(["expiry", "strike", "option_type"]).reset_index(drop=True)
