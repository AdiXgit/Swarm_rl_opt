# Historical Probe and Expiry-History Audit (Phase 2.4)

Investigation only. No dataset extension, retraining, or production code change was performed. Machine-readable companion: `data/models/historical_probe_and_expiry_audit.json`. Raw probe output: `data/raw/history_probe_2018_2020.json` (new file; the existing `data/raw/history_probe.json` was never touched).

## 1. Executive summary

Phase 2.3 left two questions unresolved by assumption rather than evidence: whether pre-2021 NIFTY data is retrievable at all through the existing provider, and whether NIFTY had weekly options throughout a candidate 2018-2021 extension window. Both are now answered **directly, from real fetched data**, using the existing, unmodified `NseKitProvider` mechanism at the same `max_rps=1.0` the frozen dataset build already uses.

**Data is retrievable back to at least 2018-01-10** (index history) and **2018-01-15** (option chain), via the same mechanism already in the codebase — no new provider, no bypass, no scraping. **But NIFTY options traded on a monthly-only cadence from at least January 2018 through mid-January 2019**, confirmed by a 36-date bounded probe spanning Jan 2018-Dec 2020: the front-to-second-expiry gap was 28-35 days throughout that stretch. **Weekly options are directly confirmed from 2019-02-15 onward**, with a consistent 7-day gap for every one of 23 sampled dates through December 2020.

This narrows, but does not eliminate, the extension opportunity. The recommendation remains **CONDITIONAL GO**, now scoped specifically to **2019-02-15 through 2021-03-31** (the confirmed weekly-compatible window) rather than the originally hypothesized full 2018-2021 range.

## 2. Probe method and access limitations

`scripts/probe_history.py` (unmodified, inspected only) uses `NseKitProvider` — the same project dependency (`NseKit` v0.0.43) already used to build the frozen dataset — at `max_rps=1.0`, one representative (median-of-month) trading date per month, with no retry loops of its own beyond NseKit's internal client behavior, and no bulk collection. It was judged safe to reuse the *pattern* unchanged; a **new** script, `scripts/probe_history_2018_2020.py`, was written instead of modifying the original, specifically so the existing probe output (`data/raw/history_probe.json`, covering 2021-2026) would never be overwritten.

**Connectivity check performed first:** a plain, unauthenticated `curl` to `nseindia.com` timed out, while `curl` to `google.com` succeeded and DNS resolved `nseindia.com` correctly — indicating NSE's own bot-detection layer blocks naive requests, not a sandbox-wide network block. The already-approved `NseKit` library's own session handling (user-agent rotation, cookie warm-up — already present in the pinned dependency, not written for this task) succeeded where raw `curl` did not. This is normal client behavior of an already-permitted library, not a new workaround.

**Bound compliance — disclosed, not hidden:** the formal probe made exactly 36 chain-level requests as planned. **One additional exploratory chain request** (`2018-01-15`) was made before the formal probe was designed, to confirm the mechanism worked at all, bringing the total to **37 chain-level requests — one over the stated 36-date bound.** A separate, lighter index-history range call (aggregate OHLC only, no option data) touched 8 trading dates at the index level; this is not counted against the chain-request bound but is disclosed for completeness. No bulk collection, no automatic date expansion, and no unbounded retries occurred in either case.

## 3. Per-month availability results

All 36 formally probed dates (2018-01-16 through 2020-12-16) succeeded: **0 access failures, 0 empty chains, 0 non-trading-day surprises.** Full detail in `data/raw/history_probe_2018_2020.json`.

| Period | Classification | Dates | Front-to-second-expiry gap |
|---|---|---|---|
| 2018-01-16 .. 2019-01-16 | `MONTHLY_ONLY_OR_NO_WEEKLY_EVIDENCE` | 13 | 28-35 days |
| 2019-02-15 .. 2020-12-16 | `WEEKLY_COMPATIBLE` | 23 | 7 days (every sample) |

An unsuccessful request never occurred, so no date here is being misread as "didn't exist because the request failed" — every classification above is a positive observation from a returned, non-empty chain.

## 4. Verified schema observations

The 2018-01-15/16 chains were parsed through the **existing, unmodified** `normalize_bhavcopy` legacy-schema path (`backend/swarm_opt/data/normalize.py::_from_legacy`) without any exception, producing a correctly-typed, non-empty normalized frame (575 rows on 2018-01-15) with sensible spot/strike/expiry values. **This is direct, positive evidence** the legacy 16-column bhavcopy layout was already compatible with the existing normalizer as of January 2018 — not an assumption, and not a substitution of the 2023 fixture sample. Lot size was `75` throughout all 36 months (already within the existing `CANDIDATE_LOTS` tuple — no normalizer change implied). Schema was `legacy` throughout, consistent with the existing `SCHEMA_CHANGE = 2024-01-02` boundary already in `swarm_opt.data.splits`.

This finding is scoped honestly: only the two January-2018 dates had their *raw* NSE columns actually exercised through the normalizer in this task; the other 34 dates' classification relies on the normalizer's own output (expiry list, schema tag, lot size), which implies — but doesn't separately re-verify — the same column compatibility for each of them individually.

## 5. Weekly/monthly expiry findings

This is the central finding of this audit. Across the 36 sampled dates:

- **2018-01 through 2019-01 (13 samples): monthly-only.** Front expiry sat 7-15 days out at the sampled (mid-month) date, but the *next* listed expiry was always 28-35 days after the front one — the unambiguous signature of a monthly, not weekly, listing cadence.
- **2019-02 through 2020-12 (23 samples): weekly-compatible.** Front expiry sat 0-6 days out, and the next listed expiry was *exactly* 7 days later, every single time — directly matching the cadence the current dataset (2021+) already relies on.
- **Transition window: between 2019-01-16 and 2019-02-15.** The exact launch date of NIFTY weekly options was not pinned down further, to respect the probe's request bound. This remains the one open item a future, separately-authorized probe could resolve with 1-4 additional bounded requests in January-February 2019.

## 6. Compatibility with the current episode builder

Using `ExpiryCalendar.weekly_episodes(min_days_to_expiry=1, max_days_to_expiry=10)` as the reference:

- **Monthly-only period:** 6 of the 13 sampled dates had `front_days_to_expiry > 10`, meaning the episode builder would not even start an episode on that exact date under today's parameters; the other 7 would start one, but the next opportunity wouldn't arise until ~18-25 trading days later (the next monthly expiry's own 10-day window). **Net effect: roughly one episode per month instead of ~4-5, with multi-week gaps where no episode exists at all.** These would not be "weekly episodes" in any sense comparable to the current dataset's unit of analysis — recommended for **exclusion**, not inclusion, from any extended weekly-episode TRAIN set.
- **Weekly-compatible period:** every sampled date's cadence matches what the current calendar code already handles correctly. No code change is implied for this sub-period.

## 7. Revised sample-size estimates

| | Phase 2.3 (pure extrapolation) | Phase 2.4 (36-date direct sample) |
|---|---|---|
| Optimistic scenario | ~160-165 episodes (weekly throughout 2018-2021Q1) | **Directly contradicted** by evidence for 2018-01..2019-01 |
| Conservative scenario | ~105-110 episodes (weekly from ~2019 only) | **Closest to what was found**; now evidence-anchored |
| Candidate compatible window | unverified | **2019-02-15 .. 2021-03-31, ~25.5 months, directly confirmed weekly cadence in all 23 samples** |
| Estimated additional eligible TRAIN episodes | 100-105 (estimate) | **95-120 (conservative/central/optimistic band, still an estimate — built from monthly sampling, not a full daily rebuild)** |
| Monthly-only sub-window (2018-01..2019-01) | not distinguished | ~10-13 episodes if included at all; **recommend excluding** as structurally incomparable |

This is still an estimate, not a measured count: only 1 date per month was sampled, not the full daily calendar. A full daily build of 2019-02..2021-03 would be needed to convert this into an actual episode count.

## 8. Remaining uncertainties

- Exact weekly-options launch date (narrowed to a ~1-month window, not pinned exactly).
- Legacy schema stability before January 2018 (never probed).
- Futures/forward-source liquidity through 2018-2020 — only expiry *structure* was inspected here, not forward-quality (how often the primary `futures` forward source would be available vs. falling back to parity/carry-curve).
- Full daily (not monthly-sampled) coverage for the 2019-02..2021-03 candidate window.
- Whether NSE's reports-API fallback path (used for non-UDiFF dates) has any undocumented access limit beyond the already-respected `max_rps=1.0` — no failures occurred in 36 requests, which is reassuring but not proof of unlimited access at a larger scale.

## 9. GO / CONDITIONAL GO / NO-GO recommendation

**CONDITIONAL GO — narrowed and evidence-strengthened relative to Phase 2.3.**

The two highest-priority unknowns from Phase 2.3 are now answered directly rather than assumed:
1. Pre-2021 data **is** retrievable via the existing approved mechanism (verified, not assumed).
2. Weekly options did **not** exist throughout the full 2018-2021 window — only from ~2019-02 onward (verified, not assumed) — which **narrows the viable extension to roughly 25.5 months, not 3.25 years**, and roughly halves the realistic sample-size benefit versus the original optimistic hope.

Scope any future extension work to **2019-02-15 through 2021-03-31**, explicitly excluding the confirmed monthly-only 2018-01..2019-01 sub-period.

## 10. Exact next steps requiring authorization

1. A small, separately-authorized follow-up probe (1-4 bounded requests, within January-February 2019) to pin the exact weekly-options transition date.
2. A full daily (not monthly-sampled) probe or build of the 2019-02-15..2021-03-31 window, to convert the estimate in Section 7 into a measured trading-date and episode count — this would be the first step that approaches "bulk" and should be scoped and approved as its own task, not folded into a read-only investigation.
3. A forward-source/liquidity check (futures availability, parity-pair counts) for the same window, mirroring `scripts/measure_filter_impact.py`'s existing approach, before trusting any Greeks/IV feature built from it.
4. Only after 1-3: build a new, separately-versioned dataset/manifest (never mutating `653ccb70c795d86f`), per the experiment design already proposed in `docs/HISTORICAL_EXTENSION_FEASIBILITY.md` section 7.
