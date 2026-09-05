# STATE_REPORT.md — co-president-2026 Project Assessment

**Date**: 2026-07-21
**Branch assessed**: `feat/spec-42-final-forecast` (7 modified files + 1 untracked script, uncommitted)
**Assessment type**: Read-only forensic review. No code was changed.

---

## 1. Executive Summary

**The 2022 MVP is sound. The 2026 pipeline is broken end-to-end, and the failure is primarily a data problem, secondarily an architecture problem, and ultimately a process problem.**

The headline numbers:

| Output | Claimed | Reality |
|---|---|---|
| 2022 backtest R1 (with result likelihood) | MAE 0.14pp | ✅ Reproducible, but near-tautological (φ_elec=5,000 pins the answer) |
| 2022 true forward R1 | MAE 2.00pp | ✅ Documented, but 95% CI coverage only 67% |
| 2026 backtest R1 (with result likelihood) | MAE 0.11pp | ⚠️ Reproduces, but meaningless as validation (same tautology) |
| **2026 true forward R1** | — | ❌ **MAE 14.29pp, 0% CI coverage, winner-call wrong** |
| **2026 runoff forecast** | "statistically significant" | ❌ **Degenerate 100%/0% probabilities; report contradicts itself** |

Three root causes, in order of impact:

1. **Data ingestion silently drops 18 of 29 CNE poll bundles, and 2 of the 4 firms that survive produce corrupt toplines.** The forward model runs on ~8 polls dominated by a single firm (Atlas Intel), polluted by a garbage GAD3 extraction (Valencia shown at 43.1% vs. 13.0% in the independent La Silla Vacía source) and a Tempo poll whose shares sum to 39% instead of 100%.
2. **The empirical runoff path (SPEC-34) is mathematically broken at three levels** — extraction semantics, aggregation counts, and orientation handling — producing `Beta(total+0.5, 0.5)` posteriors, i.e., 100% win probability for every candidate against every candidate.
3. **The forward model has no anchor**: the 2026 consultation registry is empty (uniform prior), the municipal prior silently degrades to flat for non-2022 years, and there is no election-day fundamentals term — so with ~8 noisy polls the posterior collapses toward the poll-set average with far-too-narrow intervals.

The "black box" feeling is accurate and structural: every pipeline stage can fail silently (bare `except Exception`, monkey-patched validation thresholds, dead-code validators), no phase writes inspectable QC artifacts, and no 2026 plots exist anywhere in `results/`.

**Process state is also red**: all three quality gates fail on the working tree (ruff: 9 errors, pyright: 2 errors, pytest: 17 failures), the entire 2026 pipeline lives on **7 stacked unmerged PRs** (#267–#274), and documentation (STATUS.md, CHANGELOG) is 5–10 SPECs behind reality.

---

## 2. Scope & Method

Examined:

- **Root docs**: `MVP_SPECS_GUIDE.md`, `README.md`, `AGENTS.md`, `CHANGELOG.md`, `GUIDE.md`
- **`docs/`**: `architecture/` (all 6), `results/` (both), `specs/STATUS.md`, `plans/` (all 3), `reference/` (all 5), `data/`, `archive/`, `SPEC-41-FNN-CLR-BENCHMARK-REPORT.md`
- **`reference/` (gitignored root dir)**: all 9 markdown tech guides (co_elect, pollsposition, 538, us_potus ×2, 2024_potus, elex, us_election2024, forecast_2016), the Colombia research reference, PDF inventory
- **Git**: log (~50 commits), branches (9), working-tree diff (642 insertions, 1,065 deletions)
- **GitHub**: 100 issues (7 open), 30 most recent PRs (7 open)
- **Code**: `data_cne_2026.py`, `data_results.py`, `empirical_runoff.py`, `model_round1.py`, `model_municipal.py`, `model_runoff_matrix.py`, `config.py`, `report_utils.py`, all 2026 scripts
- **Data**: `2026_topline.parquet`, `2026_runoff_pairings.parquet`, La Silla Vacía `encuestas_detalle.csv` + `ponderacion.csv`, 2026 MMV departmental results (independent re-aggregation)
- **Quality gates**: ran `make test-fast` (17 failed / 1,145 passed), `ruff check` (9 errors), `pyright src/` (2 errors)
- **Cross-validation**: every 2026 topline poll checked against La Silla Vacía's independent aggregator

---

## 3. Current State Snapshot

### 3.1 Branch / PR topology

```
main ── dev ── (last merged: PR #266, docs overhaul)
              └─ feat/year-agnostic-scaffold      PR #267 OPEN  (Phase 0)
                   └─ feat/spec-31-cne-ingest     PR #268 OPEN
                        └─ feat/spec-32-cne-validation  PR #269 OPEN
                             └─ feat/spec-33-cleanpolls-2026  PR #270 OPEN
                                  └─ feat/spec-34-empirical-runoff  PR #271 OPEN
                                       └─ feat/spec-35-refit-2026   PR #272 OPEN
                                            └─ feat/spec-42-final-forecast  PR #274 OPEN  ← HEAD
                                                 (plus uncommitted changes)
```

**The entire 2026 pipeline (SPEC-31→35, 42) exists only on this unmerged stack.** `dev` contains none of it. Issues #221–#226 and #273 remain open despite their code existing on branches — the board says "ready/in-progress" while branches say "done".

### 3.2 Quality gates (working tree, 2026-07-21)

| Gate | Result | Detail |
|---|---|---|
| `ruff check src/ tests/ scripts/` | ❌ **9 errors** | incl. unused variables in modified files |
| `pyright src/` | ❌ **2 errors** | `data_cne_2026.py:1288` unused variable etc. |
| `pytest` (fast) | ❌ **17 failed** / 1,145 passed | municipal OOS (11), benchmark compositional (4), aggregation trends (2) |
| CodeRabbit review | ❌ Not run | uncommitted work per §9 of AGENTS.md requires it |

The 11 municipal-OOS failures are a **regression from the Phase-0 year-agnostic refactor**: `ValueError: No CANDIDATES registry for year 2014` — `CANDIDATES_BY_YEAR` only has 2022/2026, but SPEC-26's OOS framework backtests on 2014/2018 holdouts. A previously-green suite was broken by the scaffold PR and never repaired.

### 3.3 Uncommitted working-tree changes (mid-rewrite)

| File | Change | Assessment |
|---|---|---|
| `scripts/forward_2026_backtest.py` | **1,203 → ~220 lines** | Google Trends, municipal features, runoff digital signals all stripped. Header now reads *"No Google Trends. No municipal features. Simple, fast, forward."* — directly contradicts `PLAN_2026_forecast_execution.md` §1.2 ("Digital signals stay in the model", "TerriData enrichment must be in the first backtest", "approval included") |
| `scripts/run_2026_forecast.py` | new, untracked | Parallel script with the *opposite* philosophy (trends hard-required, approval wired). Two divergent 2026 runners now exist |
| `src/co_president/data_cne_2026.py` | +245 lines | `_load_generic_xlsx` heuristic loader for 5 firms; more whack-a-mole candidate-name mappings; `extract_approval_series()` |
| `src/co_president/data_results.py` | +138 lines | 2026 MMV loader; hardcoded `registered_voters=39_000_000`, `polling_stations=110_000` |
| `scripts/report_utils.py` | +63 lines | `results_r1=None` path fabricates zero-valued metrics; runoff short-circuit returns all-NaN metrics |
| `src/co_president/model_round1.py` | +45 lines | approval covariate (merge_asof, `fillna(50.0)`) |
| `src/co_president/model_municipal.py` | +4 lines | year-aware candidate keys |
| `scripts/run_100k_report.py` | +7 lines | `year=` param + monkey-patch block |

---

## 4. What Works (and Should Be Preserved)

1. **2022 MVP pipeline** — ingestion, cleaning (Invamer date fix, undecided redistribution, dedup), Registraduría/MOE cross-validation, Dirichlet-Multinomial + reverse-time RW, non-centered parameterization (R-hat 1.000, 0 divergences), K=3 honest runoff with transfer prior, rest_blanco override, drift. The *design* is validated and well-documented in `docs/architecture/`.
2. **Atlas Intel 2026 ingestion** — verified against La Silla Vacía: all 8 Atlas polls match within ~1pp (e.g., 2026-05-21: ours Cepeda 38.70/DLE 37.22/Valencia 14.23 vs. LSV 37.7/36.3/13.9). The per-firm loader pattern works when written carefully.
3. **2026 MMV results loader** (uncommitted) — independently re-aggregated the 33 departmental files: DLE 43.51% (10.05M), Cepeda 41.31% (9.54M), Valencia 6.87%, Fajardo 4.23%, blanco 1.74%, rest 1.39%, C. López 0.94%. Matches the backtest report. The actual top-two was **de la Espriella vs. Cepeda**.
4. **Municipal ML benchmark** (SPEC-41) — FNN+CLR R² 0.987–0.995 on 3-class holdout with year-specific TerriData features; honest documentation of overfitting traps (RFR 0.999 flagged as memorization).
5. **Documentation substrate** — `docs/architecture/`, `docs/reference/` synthesis docs, and GUIDE.md/MANIFEST.yaml convention are genuinely good.

---

## 5. Evidence: The 2026 Forecast Failure

### 5.1 Forward R1 (true forecast, `results/forward_2026_report.md`, 2026-06-22)

| Candidate | Actual (MMV) | Predicted | Error |
|---|---|---|---|
| de la Espriella | 43.51% | 21.06% | **−22.5pp** |
| Cepeda | 41.31% | 24.52% | **−16.8pp** |
| Valencia | 6.87% | 24.79% | **+17.9pp** |
| Fajardo | 4.23% | 10.29% | +6.1pp |
| C. López | 0.94% | 7.39% | +6.5pp |

MAE 14.29pp, **95% CI coverage 0% (0/7)**. The posterior compresses everyone toward ~10–25% — the signature of a flat prior + tiny, noisy poll set + no anchor. (The report's "Actual 0.00%" column comes from a synthetic placeholder `RoundResult`; the MAE printed is against 0.00%, so even the printed 14.29pp is a fabricated metric.)

### 5.2 Runoff forecast (`results/runoff_forecast_2026_report.md`, 2026-06-22)

- Every pairing: **P(first wins) = 100.00%**, CI [99.5%, 100.0%].
- "Statistical significance" section declares **both** "de la Espriella >95% beats Cepeda" **and** "Cepeda >95% beats de la Espriella" — a logical impossibility, printed verbatim in the final deliverable.
- The two report files (`forward_2026_report.md` vs. `runoff_forecast_2026_report.md`) contain **two different R1 posteriors** (Valencia 24.79% vs. 23.26%; DLE 21.06% vs. 30.74%) — generated by different code versions ~20 minutes apart. Which model produced the deliverable is untraceable.
- `baseline_100k_report_2026.md`: **max R-hat 1.52** (non-converged) yet the interpretation text declares success.
- Template bugs: `Pairing ( vs ) not found in matrix`, `P( wins): N/A` — empty candidate names rendered into reports.

### 5.3 Data-layer evidence (the primary cause)

`2026_topline.parquet` — **11 rows from 29 bundles**:

| firm | bundles | rows in parquet | verdict (vs. La Silla Vacía) |
|---|---|---|---|
| atlas_intel | 9 | 8 | ✅ ≤1pp error — good |
| gad3 | 2 | 1 | ❌ **Garbage**: Valencia 43.13% vs. LSV 13.0%; Cepeda 7.19% vs. 36.0%; rest 30.93% vs. ~1.9% |
| genesis_crea | 3 | 1 | ⚠️ Valencia +4.0pp vs. LSV (SPEC-32 threshold is 0.5pp) |
| tempo | 1 | 1 | ❌ **Shares sum to 39.05**, not 100 — uniform ×0.39 under-scaling of every candidate |
| invamer | 2 | 0 | ❌ silently dropped |
| cnc | 4 | 0 | ❌ silently dropped (incl. a 1.7 GB bundle) |
| guarumo_ecoanalitica | 3 | 0 | ❌ silently dropped |
| corp_miguel_maldonado | 2 | 0 | ❌ silently dropped |
| analizar_lombana | 1 | 0 | ❌ silently dropped |

`2026_runoff_pairings.parquet` — **15 rows, all from one GAD3 poll (2026-04-22)**, containing:

- Self-pairings: `cepeda vs cepeda`, `de_la_espriella vs de_la_espriella`, `valencia vs valencia`
- Unmapped raw ballot codes as candidates: `"4"`, `"5"`
- **`n_a == n_b` in every row** — no actual vote split was ever computed

La Silla Vacía's `encuestas_detalle.csv` contains ~27 polls across 33 field dates — **2.5× more polls than the ingestion pipeline produces**, and it is already clean, weighted, and includes the `Prediccion_30d` validation target. It sits in `data/2026-polls/silla_vacia_ponderador/`, used only as an (unenforced) cross-check.

---

## 6. Findings by Category

### A. Data Ingestion & Cleaning — **FATAL**

**A1. Silent bundle dropout.** `build_cne_2026_tables` wraps every per-bundle load in bare `except Exception: logger.exception(...)` (`data_cne_2026.py:~1120`). A bundle that fails vanishes with a log line; nothing aggregates, counts, or surfaces the loss. 18 of 29 bundles (62% of the raw data) are lost this way. There is no ingestion coverage report — SPEC-15's `report.py` pattern was never applied to the CNE pipeline.

**A2. GAD3 first-round extraction is wrong.** `_gad3_extract_r1_votes` treats sorted `Q06.*` columns as one-hot vote dummies via `_GAD3_R1_INDEX_MAP` (`data_cne_2026.py:272-280, 345-376`). The output contradicts the independent aggregator by up to **30pp per candidate**. Either the Q06 columns are not one-hot vote dummies, or the index map/sort (`_gad3_sort_q06_col` extracts only the first digit group — `"Q06"` → key 6 for *every* column, making the sort a degenerate no-op) misaligns labels. This single poll is responsible for most of the Valencia +17.9pp forward error.

**A3. Tempo shares are not normalized.** The `_load_generic_xlsx` path (uncommitted) produced shares summing to 39.05. Ratios are correct (~×0.39 uniform), so the extraction found the right columns but a wrong denominator/weight. **No stage of the pipeline checks that a poll's candidate shares sum to ~100%** — the 2022 `CleanPolls` normalization (`normalize_undecided`) is not applied to the 2026 table.

**A4. `_load_generic_xlsx` is an unvalidated heuristic now covering 5 firms.** "First XLSX with ≥20 rows, first column containing a weight keyword, Invamer's vote-column detector" (`data_cne_2026.py:761-856`). It produced findings A2–A3. It replaced previously-explicit per-firm loaders with no per-firm golden test.

**A5. Fabricated data fills real gaps.** `muestra = 1200` hardcoded when missing (`data_cne_2026.py:1213,1221` — feeds the Dirichlet-Multinomial `n`, directly corrupting uncertainty); `registered_voters = 39_000_000`, `polling_stations = 110_000` hardcoded for the 2026 canonical result (`data_results.py:~790`).

**A6. SPEC-32 cross-validation is dead code.** `validation_cne_2026.py` exists, has tests, and is imported by **nothing** outside itself. The mandated "discrepancy report in `docs/results/`" was never generated. Had it been wired as a gate, findings A2–A4 would have been caught on day one — the LSV comparison in §5.3 took minutes to perform manually.

**A7. 2026 results have no second-source validation.** The 2022 design cross-validates Registraduría vs. MOE; the 2026 loader (`data_results.py`, uncommitted) is single-source with no MOE check and no participation files.

**A8. Candidate-name mapping is whack-a-mole.** `CANDIDATE_KEY_MAP_2026` grows by string patches per commit (latest: `"no Sabe/No Responde"`, `"Blanco"`, `"Roy Barrera"`...). `"Blanco" → blanco` risks double-counting where blanco is already a computed column. Unmapped names silently become `rest` or `ns_nr` in some loaders, pass through raw in others (`"4"`, `"5"` in pairings).

### B. Runoff Pipeline (SPEC-34) — **FATAL, mathematically broken at 3 levels**

**B1. Extraction semantics.** `_gad3_extract_runoff_pairings` (`data_cne_2026.py:379-411`) hardcodes Q09A→cepeda, Q09B→de_la_espriella, Q09C→valencia and treats the *response* as "candidate_b". The actual opponent in each head-to-head question is never known — the pairing is fabricated. Responses equal to the column candidate create self-pairs; unmapped numeric codes pass through.

**B2. Aggregation assigns all votes to both sides.** `extract_runoff_pairings` sets `n_a = total_w; n_b = total_w  # same denominator for paired responses` (`data_cne_2026.py:1013-1014`). The comment reveals the conceptual error: `n_a`/`n_b` should be the weighted counts *choosing* each side; instead both get the group's full weight. Then `compute_per_pairing_stats` (`empirical_runoff.py:121-147`) sums `n_total` for the canonical-first candidate across both orientations and computes `n_b = n_total − n_a = 0`. Result: every pairing's posterior is **Beta(total+0.5, 0.5)** — point mass at 1.0.

**B3. Orientation ignored.** `sample_empirical_transfer` takes `candidate_a`/`candidate_b` as unused arguments (`empirical_runoff.py:186-188`) and samples the canonical-alphabetical Beta. `estimate_runoff_matrix` (`model_runoff_matrix.py:557-568`) attributes the result to whichever candidate is "first" in the top-two ordering. Hence the report's self-contradiction: (DLE, Cepeda) and (Cepeda, DLE) both print "first wins 100%".

**B4. Empirical path bypasses every validated 2022 mechanism.** When `alpha+beta ≥ 50.5`, the empirical path *replaces* the K=3 model — no transfer prior, no rest_blanco override (shares forced to sum to 1.0 between the two candidates; `share_rest = NaN`), no drift, no runoff-poll likelihood, no MCMC diagnostics. The careful "honest forecast" architecture documented in `docs/architecture/honest-forecast.md` applies only to the path the empirical branch discards. And the RNG is unseeded (`np.random.default_rng()`, `empirical_runoff.py:207`) — outputs are not reproducible.

**B5. No guardrails on the pairing data.** The SPEC-34 design said "empirical Beta when n_eff ≥ 50, Dirichlet fallback otherwise". Nothing validates that the pairing table is sane (no self-pairs, mapped labels, plausible splits) before it overrides the validated path.

### C. Architecture / Conceptual — **HIGH**

**C1. The forward R1 model has no anchor — by construction.** For 2026: `CONSULTATION_VOTES_2026 = {}` (`config.py:417`) → `consultation_log_share_prior` returns uniform −1.0 logits (~14%/candidate); the municipal prior for `target_year=2026` finds no 2026 historical records → `clr_left_share = 0.0` everywhere (`model_municipal.py:92-104`), and shape mismatches trigger the silent flat-prior fallback (`model_round1.py:349-362`, commit 598fdc7's "guard"). The random walk's oldest point is thus uninformative and election day is free — the posterior is 100% poll-driven with ~8 polls.

**C2. Prior anchoring is inverted relative to the reference methodology.** The Economist (`us_potus` §13.1) and Rieke 2024 anchors place the *fundamentals prior on election day* and walk backward; this project anchors the informative prior at the *oldest* time point (`docs/architecture/model-design.md` §"Why Reverse-Time Random Walk"). In backtest mode this doesn't matter (φ_elec=5,000 pins election day). In forward mode the endpoint estimate = flat prior + accumulated RW noise × T — the exact configuration the references are built to avoid. There is no compensating time-calibrated uncertainty schedule (`fit_rmse_day_x`, ω-schedule).

**C3. Backtest validation is near-tautological.** With φ_elec=5,000 the "backtest" re-reads the answer (2022: MAE 0.14pp; 2026: 0.11pp) while the honest forward number is 14–20× worse (2022: 2.00pp; 2026: 14.29pp). Headline claims in README ("R1 MAE 0.14pp") describe the pinned configuration. Forward 95%-CI coverage was already 67% in 2022 — measured overconfidence, never addressed by a calibration layer (no Brier score, no conformal, no multi-cycle holdout in the reported metrics).

**C4. No systematic-bias component.** The references carry explicit national polling-bias terms (`polling_bias`, partisan non-response AR(1), market_bias). This project patches the same channel with hardcoded constants: blanco prior pinned at 2.0% (polls say ~10.6%), rest_blanco pinned at 2.3%, runoff digital signals disabled outright, `concentration_t1_boost = 0.0`. Each patch is documented and individually justified — but they are symptoms of a missing model component, and each is a 2022-calibrated constant now applied (or not) to 2026 without revalidation.

**C5. No multi-election borrowing.** pollsposition trains on 2002–2017 with shared posteriors; the gap analysis flags this as **P0/critical** for 2026 ("with only 5–10 polls, borrowing strength from 2022 is the only way to converge"). SPEC-40 (multi-election refactor) shipped for the model builders but is not wired into the 2026 forecast path — the 2026 model fits on 2026 data alone.

**C6. No candidate-dropout mechanism.** Claudia López polls 0.0% in June Atlas bundles (withdrawn) yet the forward RW extrapolates her to 7.39%. pollsposition's −10 logit offset for non-competing parties was never adopted. Symmetrically, the June Atlas polls (field_end 06-02…06-11, *after* R1 on 05-31) sit in the same parquet as R1 polls — classification depends on each script's own cutoff logic, and the gutted forward script uses `field_end <= r1_date` while the plan says ≤ 2026-05-22.

**C7. Provenance misattribution in docs.** `docs/reference/adopted.md` and `key-findings.md` credit the co_elect "Ajiaco" guide for the reverse-time RW, house effects, zero-sum constraint, and consultation prior. The actual `reference/co_elect-tech_guide.md` contains **none of these** — its model is a static Dirichlet(2×6)+Multinomial with no time dimension (its own guide flags its runoff as "completely meaningless"). The true source in the reference set is `us_potus_tech_guide.md`. The citations should be corrected; the current ones make architecture review harder.

**C8. Turnout channel absent.** The research reference elevates turnout modeling (and the 2022 R1→R2 abstention structural break, −4.24pp) to a first-class feature; no turnout model exists. Vote intention → vote translation ignores mobilization.

### D. Feature Engineering / Model Wiring — **HIGH**

**D1. Inverted house-effect flag.** `run_2026_forecast.py:261`: `no_house_effects=(mode == "backtest")` — house effects are *disabled* in backtest and *enabled* in forward mode, where 3 of 4 firms have exactly one poll each and house effects are unidentified (a per-pollster free parameter per candidate with one observation absorbs that poll's deviation as "bias").

**D2. Monkey-patched validation minimums, three different values.** `_dp._MIN_ROUND1_POLLSTERS = 3` in `forward_2026_backtest.py`, `= 2` in `run_2026_forecast.py`, `= 3` in `run_100k_report.py`; `_MIN_ROUND2_POLLSTERS = 1`. The `CleanPolls.__post_init__` diversity checks (SPEC-04, designed to guarantee house-effect identifiability) are bypassed by reaching into module privates — and each script picks a different number.

**D3. Tautological branches.** `run_2026_forecast.py:269-270`: `_r1_for_r2 = results_r1 if mode == "backtest" else results_r1` — both branches identical; same for `_r2_for_model`. In forward mode `results_r2` (an empty synthetic `RoundResult`) flows into `run_round2_survey`, which short-circuits to all-NaN metrics — so the "runoff" section of a forward report is structurally guaranteed to be empty unless the empirical path fills it.

**D4. Fake metrics on the `None` path.** `report_utils.py` `run_round1_survey` with `results_r1=None` emits metric dicts with `actual: 0.0, error: 0.0, mae: 0.0` — indistinguishable downstream from real metrics. Report consumers cannot tell "no ground truth" from "perfect".

**D5. Approval covariate is fragile.** `extract_approval_series` swallows all exceptions per bundle (`except Exception: continue`), supports only Atlas-style columns, and the model fills missing dates with 50.0 (`model_round1.py:311`). No test validates the sign or magnitude of its effect; the plan called it an "explanatory covariate" but it's wired as a poll-level linear term with no posterior summary in the report.

**D6. Silent 2022 fallback for 2026 trends queries.** `run_2026_forecast.py:133`: `CANDIDATE_QUERY_MAPS.get(str(year)) or CANDIDATE_QUERY_MAPS["2022"]` — a missing 2026 map silently fetches Petro-era queries.

**D7. Two divergent 2026 runners.** `forward_2026_backtest.py` (gutted: no trends, no features, synthetic results) vs. `run_2026_forecast.py` (trends hard-required, approval, features). Which one is SPEC-42's deliverable script is undefined; the issue text names the former, the newest artifact came from neither consistently.

**D8. Municipal model still requires `pop_2022`** (`model_municipal.py` docstring, required columns) — the plan's "use `pop_2026`" was never implemented in the model layer.

### E. Validation & Observability (the "black box") — **HIGH**

**E1. No phase artifacts.** Nothing between "raw zips" and "final report" is persisted in inspectable form: no ingestion coverage table (bundle → rows → status → reason), no topline-vs-LSV discrepancy report, no pairing QC table, no record of which prior (flat vs. municipal) was actually used, no poll-level input dump with shares-sums.

**E2. No 2026 plots.** `results/` contains zero PNGs for 2026. `plotting.py` is 2022-oriented; notebooks are 5–8KB stubs. There is no visual check of poll series vs. posterior vs. actuals — the exact tool that would have made the GAD3 outlier obvious in seconds.

**E3. Silent degradation is the default everywhere.** Bundle loads, approval extraction, trends fetch, municipal prior shape, consultation registry, runoff short-circuit — every failure path logs and continues (or continues without logging). Nothing ever *stops the pipeline* on bad data. The one validator that would (`CleanPolls.__post_init__`) is monkey-patched away.

**E4. Report renderer doesn't validate its inputs.** Empty candidate names (`Pairing ( vs ) not found`), `Actual 0.00%`, "Converged: Y" next to "WARNING: Chains did not fully converge", NaN margins — all rendered into final deliverables without a single assertion.

**E5. The 2026 R2 already happened (2026-06-21); the repo has no `2026-presidential2/` data.** The ultimate validation of SPEC-42's deliverable — comparing the forecast to the actual runoff — is not possible in the current repo state, and the forecast artifacts were generated 2026-06-22 with no subsequent review.

### F. Process / Repository Hygiene — **MEDIUM-HIGH**

**F1. All quality gates fail** (§3.2). Per AGENTS.md §4, nothing on this branch is committable; per §9, CodeRabbit review hasn't run on the uncommitted work.

**F2. Seven stacked unmerged PRs.** The entire 2026 pipeline is absent from `dev`. Reviews can't see the full picture; each PR's base is the previous unmerged branch. The Phase-0 scaffold (PR #267) broke 11 previously-green tests and was never fixed before 5 more branches stacked on top.

**F3. Documentation drift.** `STATUS.md` lists SPEC-31→36 as "📋 Planned" (they're shipped-on-branch) and has no SPEC-42 entry; `CHANGELOG.md` stops at SPEC-22; README's results section presents 2022 numbers without noting the 2026 state; `docs/plans/PLAN_2026_forecast_execution.md` §1.2 decisions (trends stay, TerriData in first backtest, approval in model) are contradicted by the gutted script on disk.

**F4. A test was skipped to buy time** (`d534ee2 test(SPEC-34): skip 2026 forecast test pending Phase 5 revalidation`) and never unskipped.

**F5. Inconsistent artifacts.** `results/` mixes outputs from at least three code versions with no provenance (no git SHA, no config hash in report headers).

---

## 7. Root-Cause Chain (why the numbers are what they are)

```
18/29 bundles silently dropped            GAD3 garbage + Tempo ×0.39 + Génesis +4pp
            │                                          │
            ▼                                          ▼
     ~8 usable R1 polls, 74% Atlas ──────► poll set mean pulled toward Valencia
            │                                          │
            ▼                                          ▼
  flat prior (empty consultation) + no election-day anchor + no dropout handling
            │                                          │
            └──────────────┬───────────────────────────┘
                           ▼
        Forward R1 posterior ≈ shrunk poll average, CIs far too narrow
        (DLE −22.5pp, Valencia +17.9pp, coverage 0%)
                           │
                           ▼
        Top-two pairing (DLE, Cepeda) only 33.91% probable
                           │
                           ▼
   GAD3 Q09 fabrication → n_a=n_b → Beta(total, 0.5) → orientation ignored
                           │
                           ▼
        100%/0% win probabilities for every pairing, both directions
                           │
                           ▼
        "Statistically significant" deliverable is incoherent
```

The backtest numbers stayed pretty (0.11pp) **only because φ_elec=5,000 re-reads the answer** — masking every upstream failure. This is why the breakage was invisible until the true forward run.

---

## 8. Remediation Roadmap (prioritized)

### P0 — Stop the line (before any new model work)

1. **Do not merge PR #274 (or the stack) as-is.** The deliverable contradicts itself.
2. **Fix the gates**: repair the 11 municipal-OOS regressions (register 2014/2018 candidates or scope `get_active_candidates` fallback), the 4 benchmark + 2 aggregation failures, 9 ruff errors, 2 pyright errors. CI goes green before anything else merges.
3. **Kill silent failure**: `build_cne_2026_tables` must emit (and save) a per-bundle ingestion report — firm, loader, rows, status, failure reason — and fail loudly when a firm produces zero rows. Same pattern for approval extraction and pairings.

### P1 — Fix the data (the actual cause)

4. **Wire `validation_cne_2026.py` as a hard gate** in the pipeline: every ingested poll compared to La Silla Vacía by firm+field_end; >0.5pp per-candidate → quarantined with a written reason. Generate the missing `docs/results/` discrepancy report.
5. **Until fixed, drop the three heuristic-loader polls** (GAD3, Tempo, Génesis) from the model input — 8 clean Atlas polls beat 11 polluted ones. Fix GAD3 Q06/Q09 against the actual questionnaire; fix Tempo's denominator; add a pipeline invariant: *every poll's candidate shares must sum to 100 ± 2%*.
6. **Use La Silla Vacía toplines as a first-class source** (~27 polls, already weighted, includes `Prediccion_30d` for validation). Even as a supplement it triples the poll base and diversifies away from Atlas.
7. **Rebuild or abandon the empirical pairing table**: pairings need real opponent metadata (questionnaire mapping), mapped labels (no "4"/"5"), no self-pairs, and true per-side counts (`n_a` = weight choosing A). Then fix `compute_per_pairing_stats` (no double-count), respect orientation in `sample_empirical_transfer`, and seed the RNG.
8. **No fabricated inputs**: real `muestra` per poll (or model-imputed with documentation), real 2026 census/polling-station figures, MOE second-source for 2026 results.

### P2 — Fix the model wiring

9. **Restore the validated 2022 honest-runoff path as the 2026 default** (K=3 + transfer prior + rest override + drift); gate the empirical Beta path behind pairing-data QC and use it only as a *likelihood inside* the K=3 model, not a replacement for it.
10. **Anchor the 2026 prior**: populate `CONSULTATION_VOTES_2026` from the March 8 consultations, or use the LSV early-window average as an informative prior; fix the municipal prior for 2026 (explicit candidate→bloc mapping, `pop_2026`, defined `clr_left` behavior for history-less target years); never fall back to flat silently — log it in the report.
11. **Handle dropouts** (C. López): pollsposition-style −10 logit offset after withdrawal, or remove from K with redistribution.
12. **Fix the inverted `no_house_effects` flag**; require ≥2 polls per firm for house effects or pool low-count firms into one "other" house.
13. **One 2026 runner**, implementing the plan's §1.2 decisions (trends, features, approval) with a `--fast` flag — not two divergent scripts. Delete the synthetic-`RoundResult` pattern; forward reports must say "no ground truth", not print fake zeros.

### P3 — Rebuild trust & observability

14. **Phase QC artifacts at every stage**: ingestion coverage table, topline validation report, pairing QC, prior-used record, poll input dump (firm × date × shares × sum-check), posterior-vs-polls-vs-actuals plots for both rounds. If a phase can't be plotted, it can't be trusted.
15. **Calibration honesty**: report Brier score and CI coverage next to MAE everywhere; state explicitly in README/reports that φ_elec backtests are pinned-run validations, and the forward number is the honest one.
16. **Add 2026 R2 results** (`2026-presidential2/` MMV) and score the June-21 forecast against reality — the ultimate, free validation of the whole pipeline.
17. **Update the docs**: STATUS.md (SPEC-31→42 real status), CHANGELOG (SPEC-23→42), correct the co_elect provenance in `adopted.md`/`key-findings.md`, record artifact provenance (git SHA + config hash) in every report header.
18. **Unstack the PRs**: merge sequentially (#267→#274) only after each passes `make check` + CodeRabbit, starting with the gate repairs. Consider folding SPEC-42 into a re-run after P1–P2 rather than merging the current artifacts.

### Estimated effort

| Phase | Scope | Effort |
|---|---|---|
| P0 gates + silence-killing | 3–5 files | 1–2 days |
| P1 data fixes + LSV integration | ingestion + validation + tests | 3–5 days |
| P2 model wiring | config + round1/runoff + scripts | 2–4 days |
| P3 observability + docs + unstacking | plots, QC artifacts, docs, PRs | 2–3 days |

---

## 9. Appendix — Key Evidence Index

| Claim | Evidence |
|---|---|
| Forward R1 MAE 14.29pp, 0% coverage | `results/forward_2026_report.md` |
| 100%/0% + contradictory significance | `results/runoff_forecast_2026_report.md:37-56` |
| `n_a = n_b = total_w` | `src/co_president/data_cne_2026.py:1013-1014` |
| Double-count → Beta(total, 0.5) | `src/co_president/empirical_runoff.py:121-147` |
| Orientation ignored + unseeded RNG | `src/co_president/empirical_runoff.py:184-208` |
| Silent bundle dropping | `src/co_president/data_cne_2026.py:~1118-1124` (`except Exception`) |
| GAD3 index map + degenerate sort | `src/co_president/data_cne_2026.py:272-280, 340-342` |
| GAD3/Tempo/Génesis vs. LSV ground truth | §5.3 tables (manual cross-check, 2026-07-21) |
| Empty 2026 consultation registry → flat prior | `src/co_president/config.py:415-422`; `model_round1.py:316-330` |
| Municipal prior inert for 2026 (`clr_left=0`) | `src/co_president/model_municipal.py:92-104` |
| Flat-prior silent fallback | `src/co_president/model_round1.py:349-362` |
| Fabricated `muestra=1200` | `src/co_president/data_cne_2026.py:1213,1221` |
| Fabricated census/stations | `src/co_president/data_results.py` (2026 path, uncommitted) |
| Inverted house-effects flag | `scripts/run_2026_forecast.py:261` |
| Tautological branches | `scripts/run_2026_forecast.py:269-270` |
| Monkey-patched minimums (3 values) | `scripts/forward_2026_backtest.py`, `scripts/run_2026_forecast.py:103-104`, `scripts/run_100k_report.py` |
| Fake zero metrics | `scripts/report_utils.py` (`run_round1_survey`, `results_r1=None` path) |
| SPEC-32 validator dead | `src/co_president/validation_cne_2026.py` imported only by its test |
| 17 test failures | `make test-fast` run 2026-07-21 (municipal OOS ×11, benchmark ×4, aggregation ×2) |
| R-hat 1.52 reported as success | `results/baseline_100k_report_2026.md` |
| Two different R1 posteriors in deliverables | `results/forward_2026_report.md` vs. `results/runoff_forecast_2026_report.md` |
| Inverted prior anchoring vs. references | `docs/architecture/model-design.md` vs. `reference/us_potus_tech_guide.md` §13.1 |
| Provenance misattribution | `docs/reference/adopted.md` vs. `reference/co_elect-tech_guide.md` §10 |
| 2026 actual R1 (verified from MMV) | DLE 43.51%, Cepeda 41.31%, Valencia 6.87%, Fajardo 4.23% — §4 item 3 |
| Stacked unmerged PRs | `gh pr list`: #267, #268, #269, #270, #271, #272, #274 all OPEN |
| Docs drift | `docs/specs/STATUS.md` (SPEC-31→36 "Planned"), `CHANGELOG.md` (ends SPEC-22) |

---

*Generated by read-only forensic assessment. All file:line citations refer to the working tree on branch `feat/spec-42-final-forecast` as of 2026-07-21.*
