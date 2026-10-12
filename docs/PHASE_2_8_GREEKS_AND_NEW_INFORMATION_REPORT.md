# Phase 2.8 — Greeks Engine Validation and New Information Research

Machine-readable companion: `data/models/phase2_8/phase28_experiment_registry.json`. All previous reports and artifacts preserved unchanged (verified, Section 1 and Section 13).

## 1. Repository baseline and preservation checks

HEAD unchanged at `d632151`. Original fingerprint `653ccb70c795d86f` and all 8 Phase 2.2 baseline artifact hashes: **verified byte-identical** at task start and end. Phase 2.5 reference (`data/models/extended_2019_2021/training_report.json`, Direction VAL accuracy 0.6765) and the Phase 2.7 registry: both present and unread-from-for-modification. `scripts/run_phase2_diagnostics.py` re-run after every structural code change in this phase and still reproduces the original baseline exactly.

## 2. Existing Greeks implementation audit

`backend/swarm_opt/features/pricing.py` (unchanged): Black-76, undiscounted (NSE futures-style convention; risk-free rate fixed at 0.0 by documented project convention — there is no discount factor to audit because none is applied). IV solved via `scipy.optimize.brentq` over `[1e-4, 5.0]`, returning NaN (not a clamped bracket edge) for any quote at or outside the no-arbitrage band. Greek units, confirmed by reading the code and cross-checked against the existing test suite: **Delta** per underlying unit ∈ [-1,1]/[-1,0]; **Gamma/Vega** per underlying unit, positive for long options, **Vega per 1.00 of volatility** (not per percentage point — a convention worth restating since it's easy to misread); **Theta** per calendar day. `agents/greeks_agent.py` reports **contract-level, per-underlying-unit** Greeks for the ATM call/put of the nearest expiry; lot size is a separate schema column never applied upstream of this phase's new portfolio module — confirmed by reading every call site between `pricing.greeks()` and `ContractGreeks`.

**Existing test coverage already discovered to be extensive** (`tests/test_pricing.py`): put-call parity, closed-form ATM check, no-arbitrage bounds, strike/vol monotonicity, IV round-trips across a grid, unsolvable/outside-band → NaN, vectorized-vs-scalar consistency, **analytical-vs-central-finite-difference Delta/Gamma/Vega already present**, NaN propagation, expiry-day stability, ACT/365 year fraction. This was not duplicated.

## 3. Pricing and Greek validation results

`tests/test_pricing_phase28_edge_cases.py` (144 cases, new): a broad multi-dimensional grid over 4 forwards × 7 strike offsets × 5 volatilities × 4 maturities (strided for runtime) checking price bounds and put-call parity hold everywhere, not just at the single `(F,K,T)` triple the existing suite uses; extreme moneyness (10× OTM, 0.1× ITM) stays finite and correct; IV solver bracket-edge round trips (`IV_LOW`, `IV_HIGH`) recover correctly; price does not overflow at 500% vol over a full year or underflow to negative at near-zero vol. **No bugs found** — every new case passed against the existing, unmodified implementation.

## 4. Implied-volatility solver validation

Reused and confirmed via `tests/test_pricing.py` (already thorough — grid across moneyness/maturity/option-type, invalid-price handling, unbracketable-root handling, deep-wing vs. deep-ITM distinction) plus the new bracket-edge tests in Section 3. **No gap found requiring new IV-solver tests beyond what already existed.**

## 5. Portfolio aggregation conventions and tests

New, additive module `backend/swarm_opt/agents/portfolio_greeks.py`: `OptionPosition` (one leg) + `aggregate_greeks` (sum across valid legs). Convention, stated once: `position_greek = signed_contracts × lot_size × per_underlying_unit_greek`, applied **exactly once** (confirmed by a dedicated test that distinguishes `lot × lot` from `lot`).

`tests/test_portfolio_greeks.py` (23 cases): construction validation (invalid side/quantity/lot/option-type all raise); lot-size-applied-once; reversing a position's side exactly reverses its signed Greek exposure; an invalid leg (unsolvable IV) is **excluded**, not zeroed, from the aggregate; standard strategies (long/short call/put, bull call spread, bear put spread, long straddle, iron condor) checked for correct sign/magnitude (e.g. short straddle: negative gamma, positive theta; iron condor: net short gamma, net positive theta). **Portfolio-level finite-difference validation**: perturbing the forward and repricing every leg at the SAME bumped forward reproduces the aggregated Delta (rel. tol 1e-3) and Gamma (second-order, rel. tol 1e-2); perturbing volatility reproduces Vega (rel. tol 1e-3); a one-day time-advance reproduces Theta within a documented loose tolerance (15% — theta is a local derivative, a full day is not infinitesimal, stated explicitly as the task required rather than picking an arbitrary number).

## 6. Bugs found and regression tests

**None in the pricing or Greeks implementation.** The only correctness issue found anywhere in this phase was in Phase 2.8's own new `scripts/phase28_risk_research.py` draft logic during development (not shipped broken) — no production regression was needed. All 23 portfolio tests and 144 pricing edge-case tests pass against the unmodified Week 1 implementation.

## 7. New-information-source feasibility

| Source | Feasible? | Evidence |
|---|---|---|
| Market breadth (B1) | **No** | NseKit's only constituent method returns LIVE data only, no date parameter, no point-in-time membership source |
| India VIX (B2) | **Yes** | `NseKitProvider.underlying_history('INDIA VIX', ...)` works with **zero code changes** — the provider's `INDEX_NAME` map falls through to the raw string, which NSE's own endpoint already accepts |
| USD/INR (B2) | **No** | NseKit exposes it only via a "live Gift Nifty" method, no historical series |
| Order flow (B3) | **No** | NSE's public EOD bhavcopy archive (the only authorized source) contains no tick/aggressor/order-book data; no proxy was fabricated |

**India VIX is the one feasible new source**, confirmed by a single bounded check (same pattern as Phase 2.4's reconnaissance), not a bulk acquisition.

## 8. Historical data coverage and timestamp audits

India VIX cache (`scripts/build_india_vix_history.py`, `data/features/india_vix_history.parquet`): **2019-02-18 through 2026-08-18, 1857 rows**, built with **one bulk range request** (same pattern as the project's own NIFTY index history, not a per-date loop). Two dates print `open=high=low=0.0` with a nonzero close — a genuine NSE data-quality artifact, flagged and set to NaN rather than used; one of those two dates is **the same 2021-03-30** the F&O bhavcopy extension build separately found missing in Phase 2.5, suggesting a shared upstream NSE data gap on that specific date across series, not a bug in either build. **Timestamp alignment**: India VIX is published at the same EOD close as NIFTY's own close — using `VIX_close[d]` for a decision made as of EOD `d` introduces no lookahead, by the identical reasoning already governing every other close-derived feature in this project.

## 9. Feature-family ablation results

New `pointintime_v3` (19 features = `pointintime_v2`'s 17 + `india_vix_level`, `india_vix_chg_5`), added **additively** — `FEATURE_NAMES`/`FEATURE_SCHEMA_VERSION` (v1) and `FEATURE_NAMES_V2` are completely unaffected (verified: `build_market_features()`'s default behavior is byte-identical, confirmed by re-running `scripts/run_phase2_diagnostics.py`).

Same chronological walk-forward protocol as Phase 2.6/2.7, VALIDATION read once per agent per feature set:

| Agent | Walk-fwd bal. acc. Δ | VALIDATION bal. acc. Δ | Persists? |
|---|---|---|---|
| Direction | -0.005 | **-0.056** | **No** |
| Volatility | +0.015 | 0.000 | **No** |
| Regime | +0.020 | -0.017 | **No** |

## 10. Agent-by-agent predictive improvement: none

**India VIX does not persist as an incremental improvement for any agent.** Direction actually gets measurably worse on VALIDATION. This is an honest negative result for a well-justified, high-quality (99.9% coverage), genuinely new data source — consistent with, and reinforcing, Phase 2.6/2.7's broader finding that the bottleneck is not "which features," full stop.

## 11. Test results and experiment failures

`pytest` → **452 passed, 0 failed** (273 prior + 179 new: 144 pricing edge cases, 23 portfolio Greeks, 8 India VIX feature tests, 4 SWARM_STATE Greeks compatibility tests). No experiment in this phase failed to run; the India VIX ablation produced a clean negative result, not an error.

## 12. Remaining numerical, data, and research limitations

- The Black-76/Greeks implementation now has broader numerical-stability evidence (extreme moneyness, bracket edges, overflow/underflow) but this does not newly prove correctness beyond what the existing finite-difference tests already established — it extends coverage, not the underlying claim.
- Portfolio aggregation has no margin, capital, or real risk-limit calibration — `scripts/phase28_risk_research.py`'s thresholds are explicitly illustrative, stated in its own output disclaimer.
- India VIX's null predictive result is one feature family tested once; it does not rule out a genuinely different functional use (e.g. as a regime-conditioning signal rather than a flat additive feature), which was out of this phase's bounded scope.
- Market breadth and order-flow remain genuinely unavailable through the current authorized data path — not "not yet tried," but blocked on data access this project has not been given.

## 13. Recommendation for the next phase

**Greeks engine: trustworthy and now more thoroughly evidenced** — safe to rely on for Phase 3's eventual risk-aware decision logic, with the explicit caveat that no margin/capital calibration exists yet (that is Phase 3's own job, not retroactively claimed here). **ML agents: no change to Phase 2.7's `ACCEPT CURRENT LIMITATION` verdict** — a fourth independent line of attack (new cross-asset information) converges on the same null result as targets, horizons, features, and models. The next phase should treat agent outputs as weak, unconfirmed priors (per Phase 2.7) and focus Phase 3's actual value proposition on whether a rule-based or Greek-aware trading engine can still behave sensibly *despite* that — which is the question this project has not yet tested and is the one still worth testing.

## 14. Final verification

- Original fingerprint `653ccb70c795d86f`: **unchanged**.
- All 8 Phase 2.2 baseline artifact hashes: **unchanged**.
- Phase 2.5 reference (`extended_2019_2021/training_report.json`): **unchanged**, re-read not re-written.
- Phase 2.6/2.7 registries: **unchanged**, read-only references.
- TEST: **never accessed**.
- No PPO, Decision Engine, Trade Constructor, Risk Engine, or EOD Simulator code was added.
