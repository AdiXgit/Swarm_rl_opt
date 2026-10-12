# Historical Data Extension Feasibility Audit (Phase 2.3)

Investigation-only. No dataset extension, retraining, or production code change was performed. Machine-readable companion: `data/models/historical_extension_feasibility.json`.

## 1. Executive summary

`docs/EXPERIMENTS.md` asserts "data exists back to at least 2018" as the lever for the swarm agents' small-sample problem (189 TRAIN / 34 VALIDATION episodes). An exhaustive search of this repository found **that sentence is the only mention of 2018 anywhere in the codebase** — no probe, raw file, test, or manifest entry corroborates it. The earliest date this repository has ever actually fetched or cached is **2021-01-15** (`data/raw/history_probe.json`), five days before the frozen dataset's own 2021-04-01 start.

More importantly, this audit surfaced a structural risk the existing documentation does not address at all: **NIFTY weekly options are commonly understood to be a comparatively recent instrument** (monthly-only expiries for a substantial earlier period). If true for any part of a 2018-2021 extension, the current `ExpiryCalendar.weekly_episodes()` logic would not produce comparable weekly episodes for that period — it would produce few or none, since `max_days_to_expiry=10` requires a near-dated expiry to already be listed. This is unverified here (no live NSE call was made, per the task's constraints) but it is the single highest-priority item to resolve before any extension work.

**Recommendation: CONDITIONAL GO.** The idea is not rejected, but nothing about its feasibility is currently verified — only asserted. A short, explicitly-approved, lightweight probing step (same pattern as the existing `scripts/probe_history.py`, not a bulk download) is required before any implementation decision.

## 2. Actual historical coverage found

| Artifact | Earliest date | Latest date | Covers |
|---|---|---|---|
| `data/raw/history_probe.json` | 2021-01-15 | 2026-08-11 | One probed trading date per month (`scripts/probe_history.py`) |
| `data/raw/*.parquet` raw samples | 2023-07-17 (legacy) | 2026-08-18 (udiff) | 6 individual raw bhavcopy files, validation fixtures only |
| `data/processed/nifty_eod/` (frozen dataset) | 2021-04-01 | 2026-08-18 | The full, authoritative dataset — unchanged by this audit |

**Nothing in this repository has ever touched, fetched, cached, or referenced a specific 2018, 2019, or 2020 date.** The frozen dataset's own start (2021-04-01) was a modeling choice, not a data-availability limit that was tested at the time — `docs/PROJECT_SPEC.md` calls for "approximately 3-5 years... validate with a small sample first," and the validation sample that exists (`history_probe.json`) only ever covered January 2021 onward.

## 3. Verified vs. unverified vs. externally referenced

| Claim | Status |
|---|---|
| Frozen dataset range 2021-04-01..2026-08-18, fingerprint `653ccb70c795d86f`, 278 kept episodes (194/34/50) | **Verified** — read directly from `manifest_splits.json` in this task |
| NseKit (v0.0.43) is installed, has two bhavcopy code paths (direct UDiFF archive, NSE reports-API fallback) | **Verified** — read from `.venv/Lib/site-packages/NseKit/NseKit.py` |
| Data is retrievable for 2021-01-15 through 2026-08-11 | **Verified** — `history_probe.json` |
| "Data exists back to at least 2018" | **Unverified** — asserted once in `docs/EXPERIMENTS.md`, no corroboration anywhere |
| NSE reports-API fallback's actual historical retention window | **Unverified** — not documented by NseKit, not probed in this task |
| Legacy bhavcopy's 16-column schema is stable back to 2018 | **Unverified** — only one legacy sample exists locally (2023-07-17) |
| NIFTY weekly options didn't exist, or existed differently, before some point circa 2019 | **Externally referenced common market history, not verified against this repository or any live source in this task** |
| NIFTY lot-size history before 2021 fits the existing `CANDIDATE_LOTS` tuple | **Unverified** — only 2021-2026 lot sizes (25/50/65/75) have ever been observed here |

## 4. Schema compatibility assessment

Using `backend/swarm_opt/data/normalize.py` / `contracts.py` as the reference contract:

**Resolvable through existing normalization** (already implemented, already exercised on 2021-2023 legacy-schema data):
- 16-column legacy bhavcopy mapping (`_from_legacy`), spot lookup via the index series for schema eras with no underlying-price column, lot-size reconciliation via the turnover identity (`infer_lot_size`), and the futures/parity/carry-curve forward hierarchy.

**Would need new verification, not new code, if the schema turns out identical:**
- Whether the exact legacy column set (`SYMBOL, INSTRUMENT, EXPIRY_DT, STRIKE_PR, OPTION_TYP, OPEN, HIGH, LOW, CLOSE, CONTRACTS, OPEN_INT, CHG_IN_OI, SETTLE_PR, VAL_INLAKH, TIMESTAMP`) is stable all the way to 2018, or whether NSE used a different CSV layout at some earlier point — can't be checked without an actual pre-2021 sample file, which does not exist locally.
- Lot sizes outside `(10, 15, 20, 25, 30, 40, 50, 65, 75)` would need the tuple extended — a small, localized, low-risk change, but still a change to code the current frozen dataset depends on, so it would need its own regression test before touching anything.

**Could prevent a reliable comparison outright if unresolved:**
- A monthly-only-expiry period mixed into TRAIN without a regime flag would reintroduce exactly the kind of confound `docs/EXPERIMENTS.md`'s mandatory Experiment 4 (Thursday/Tuesday expiry-weekday change) was designed to isolate — except worse, because the unit of analysis itself (a weekly episode) would not exist consistently across the boundary.

## 5. Implications per agent (benefits, risks, unknowns — not guaranteed outcomes)

**Direction Agent** — Benefit: more weekly return observations, potentially spanning different macro regimes (2018 India market conditions, pre-COVID, COVID crash/recovery). Risk: 2018-2020 includes the COVID volatility shock, which could dominate the added sample with an extreme, non-representative regime rather than adding "more of the same." Unknown: whether return-threshold scale (frozen at 0.5×std of TRAIN) would need to be very different for a pre-2021 regime — the current threshold is frozen from 2021-2024 data only and the task's constraints forbid recomputing or changing it here.

**Volatility Agent** — Benefit: more realized-vol observations. Risk: IV must still be solved via Black-76 against a reconstructed forward, same as today — if futures liquidity is thinner for far-history dates, more rows would fall back to put-call-parity or carry-curve forwards (or `missing`), degrading the ATM IV/skew cache's coverage exactly where it's most needed. Unknown: whether 2018-2020 option-chain liquidity (strikes traded, OI) is comparable to 2021+ at all — not inspected here, would need its own `measure_filter_impact.py`-style audit once real data exists.

**Regime Agent** — Benefit: more regime diversity (COVID crash is a textbook `HIGH_VOLATILITY`/`TRENDING_DOWN` period the current TRAIN set may underrepresent). Risk: the five-class priority rule and its fixed ratio thresholds (1.3/0.7) were frozen against 2021-2024 volatility statistics; whether they remain a sensible mutually-exclusive partition under a structurally different (e.g., pre-weekly-options, different lot-size) regime is unknown and must not be "fixed" by re-tuning against the new data without an explicit, separate approval — that would violate the same frozen-threshold discipline that currently protects VALIDATION/TEST integrity.

**Greeks Agent** — Low risk if the data is retrievable: it reuses the same validated Black-76 implementation and forward hierarchy with no new logic needed. The only real exposure is numerical: near-expiry and deep-ITM/OTM behavior near zero-IV-content quotes is already handled (and was just hardened in Phase 2.2's `enrich.py` fix), so a wider date range mostly exercises existing, already-tested code paths rather than new ones.

## 6. Estimated additional sample size (explicitly estimates, not measurements)

Extending from 2018-01-01 to the current TRAIN start (2021-03-31) is about **3.25 candidate years**. Using the current dataset's own observed annual rate (~50-51 weekly episodes/year, from VALIDATION's 34 episodes over 8 months and TEST's 50 over ~11.5 months):

| Scenario | Additional trading dates | Additional weekly episodes (before filtering) | Surviving lookback + purge |
|---|---|---|---|
| Optimistic (weekly options available throughout) | ~800 | ~160-165 | ~153-158 |
| Conservative (weekly options only from ~2019) | ~800 calendar dates, but only ~2.1 years produce weekly cadence | ~105-110 | ~100-105 |
| Minimal (pre-2021 data not actually retrievable) | 0 | 0 | 0 |

Under the optimistic scenario, TRAIN would grow from 189 to roughly **340-350 eligible episodes** — a ~1.8x increase, still small by conventional ML standards. **None of these numbers come from inspecting a single real pre-2021 row; they are rate extrapolations and are labeled as such.**

## 7. Proposed experiment design (not implemented)

1. **New, separately versioned dataset.** e.g. `nifty_eod_2018-01-01_2026-08-18`, its own `manifest.json`/`manifest_splits.json`/content fingerprint, built by the *existing* `scripts/build_dataset.py` + `scripts/audit_dataset.py` pipeline extended only in `--start`. The current `nifty_eod_2021-04-01_2026-08-18` dataset and fingerprint `653ccb70c795d86f` are never touched, so every Phase 2/2.1/2.2 result remains reproducible exactly as-is.
2. **Compatibility gate before any build.** Run the lightweight monthly probe (`scripts/probe_history.py --start 2018-01-01`) first; require it to return `ok: true` schema/lot/expiry readings for a representative sample before committing to a full build.
3. **Same normalization, same point-in-time pipeline.** `swarm_opt.data.normalize`, `swarm_opt.features.pointintime`, `swarm_opt.features.targets` are reused unchanged — the whole point is to test a bigger sample through the identical, already-leakage-audited pipeline, not a new one.
4. **Chronological split, no new peeking.** The extended TRAIN would be `2018-01-01..2024-12-31` (old TRAIN start moves earlier only); **VALIDATION (2025-01-01..2025-08-31) and TEST (2025-09-01..2026-08-18) must stay byte-identical to today** — same dates, same purge behavior, same thresholds (re-fit only on the new, larger TRAIN, exactly as `fit_thresholds_on_train` already does, never on VALIDATION/TEST).
5. **Lookback/purge unchanged in mechanism.** `swarm_opt.data.splits.purge_straddling` and `pointintime.MAX_LOOKBACK_DAYS` apply exactly as today; the only change is a longer runway before TRAIN's first usable episode.
6. **Structural-break handling.** Any schema change (legacy format drift), lot-size regime, or expiry-cadence change (monthly-vs-weekly) discovered in the extended range must be recorded as a new, explicit regime boundary — the same pattern `swarm_opt.data.splits.SCHEMA_CHANGE` and `EXPIRY_REGIME_CHANGE` already use — not silently absorbed into TRAIN. If a monthly-only-expiry sub-period is found, it should be **excluded from the weekly-episode TRAIN set entirely** and reported as a known exclusion, not forced into the current weekly target definition.
7. **Regime-specific reporting required.** Given `docs/EXPERIMENTS.md`'s existing mandate for a Thursday/Tuesday expiry-regime robustness check, an extended TRAIN would need its own per-regime episode/class-distribution breakdown (same style as `manifest_splits.json`'s `expiry_regimes`/`schema_eras` fields) before any metric is trusted.
8. **Fair comparison with the current benchmark.** Train the same three agents, same seeds, same hyperparameter protocol as Phase 2.2's frozen config, on the extended TRAIN; evaluate on the **unchanged** VALIDATION split; compare against `data/models/phase2_training_report.json` and the Phase 2.2 experiment registry side by side — never substituting one for the other, exactly as `docs/EXPERIMENTS.md`'s existing robustness-check discipline already requires for the expiry-weekday change.

## 8. Go/No-Go recommendation

### CONDITIONAL GO

| Requirement | Evidence found | Status | Remaining work |
|---|---|---|---|
| Confirm NseKit can retrieve pre-2021 data at all | None — earliest confirmed probe is 2021-01-15 | **UNRESOLVED** | Lightweight monthly probe back to 2018-01-01 (explicitly approved as its own step, not bulk) |
| Confirm legacy schema stability pre-2021 | Only one legacy sample exists (2023-07-17) | **UNRESOLVED** | Fetch and diff ≥1 sample from 2018 and 2019 against `normalize._from_legacy` |
| Confirm whether/when NIFTY weekly options began | No artifact addresses this; externally referenced only | **UNRESOLVED — highest priority** | Inspect the probed expiry ladder once pre-2021 data is fetched |
| Confirm lot-size history fits `CANDIDATE_LOTS` | Only 2021-2026 lots observed, all covered | **Likely resolvable** | Extend the tuple if a new lot size appears; low risk |
| Preserve the frozen benchmark | Fingerprint `653ccb70c795d86f` verified unchanged in this task | **Satisfied** | None — any extension must version a new manifest, never mutate this one |

### Conditions that must be satisfied before implementation begins
1. A lightweight (not bulk) monthly probe to 2018-01-01 confirms non-empty chains/index history exist.
2. That probe's expiry ladder shows weekly (not purely monthly) listings for a meaningful fraction of the candidate window.
3. ≥1 real pre-2021 raw sample is fetched and schema-diffed against the existing legacy mapping.
4. A new, separately-versioned manifest/fingerprint is produced; today's artifacts are left untouched.
5. The extended run's split never alters or re-derives the current VALIDATION/TEST windows.

## 9. Limitations and unresolved questions

- No live network call to NSE or NseKit was made in this task — every "unresolved" item above is unresolved precisely because of that, by design (investigation-only, no bulk collection).
- The "weekly options didn't always exist" risk is stated from general market knowledge, not from this repository or any source fetched during this task; it must be treated as a hypothesis to verify, not a fact to design around yet.
- Sample-size estimates in Section 6 are rate extrapolations from 2021-2026 behavior and could be wrong in either direction if 2018-2020 liquidity, volatility, or contract availability differed substantially (plausible, given the COVID period falls inside that window).
- This audit did not examine whether a *licensed* historical data vendor (as opposed to NSE's own public archive via NseKit) would be a faster or more reliable path — that question was out of scope for a read-only repository investigation and was not pursued.
