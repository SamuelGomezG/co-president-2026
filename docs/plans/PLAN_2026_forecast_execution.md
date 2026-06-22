# 2026 Presidential Election Forecast — Execution Plan

## 1. Overview

### 1.1 Ultimate goals

1. Run a **2026 Round 1 (R1) backtest** in the same manner as `scripts/run_100k_report.py`.
2. Run a **true 2026 R1 forward backtest** in the same manner as `scripts/forward_backtest.py`.
3. Achieve results good enough in (1) and (2) to produce the **final project deliverable**: a statistically significant 2026 runoff forecast.

### 1.2 Decisions already taken

- **Candidate registry (2026 R1)**: `cepeda`, `de_la_espriella`, `valencia`, `fajardo`, `claudia_lopez`, `rest`, `blanco`.
- **Election dates**: R1 = 2026-05-31, runoff = 2026-06-21, consultations = 2026-03-08.
- **R1 poll cutoff**: field end ≤ 2026-05-22 (latest May bundle field dates). Runoff polls are the June bundles.
- **PDF-only firms** are included in ingestion, with a documented fallback/extraction strategy.
- **Digital signals stay in the model**. Google Trends cutoff for the forward backtest = 2026-05-30.
- **Petro government approval** (`approve_disapprove_federal_government`) is included as an explanatory covariate.
- **TerriData enrichment** (and other post-2022 covariates) must be in the **first backtest**.
- **2026 canonical R1 result** must be built from the 33 departmental MMV CSVs in `data/2026-elections/2026-presidential1/`.
- **Final report format** must match the existing 2022 reports.

### 1.3 Data already available

| Location | Contents | Notes |
|---|---|---|
| `data/2026-polls/` | 23 CNE ZIP bundles (~2.5 GB) from 9 firms | Atlas CSV/XLSX/PDF, Invamer XLSX, CNC SPSS `.sav`, GAD3 XLSX, several PDF-only |
| `data/2026-polls/silla_vacia_ponderador/encuestas_detalle.csv` | Long-format toplines (465 rows) | Cross-validation source |
| `data/2026-polls/silla_vacia_ponderador/ponderacion.csv` | Weekly weighted snapshots + `Prediccion_30d` | Validation target |
| `data/2026-elections/2026-presidential1/` | 33 departmental MMV CSVs (~922 k rows) | Need aggregation to national canonical result |
| `data/processed/municipal_feature_matrix.parquet` | 1143 municipalities × 169 features | Already contains `pop_2026` |
| `data/raw/TerriData/` | Fiscal/economic municipal indicators | Use existing `benchmarks/terridata.py` |
| `data/digital_signals/` | 2022 Twitter signals only | Google Trends must be re-fetched for 2026 |

### 1.4 Major blockers

- The whole pipeline (`load_and_clean_all`, `model_round1.py`, `model_runoff_matrix.py`, `model_runoff_simple.py`, `report_utils.py`) is hard-wired to 2022 dates, candidate names, pollster ratings, and results.
- No national consolidated 2026 R1 result file exists yet.
- No runoff results exist yet (final deliverable must therefore be a true forecast).
- Google Trends query map still points to 2022 candidates.
- Several 2026 bundles are PDF-only and need extraction.

---

## 2. Plan Items

Each item is written in the style of `.github/ISSUE_TEMPLATE/spec-issue.yml` so it can be copied directly into GitHub issues.

---

### Phase 0 — Year-agnostic pipeline scaffold (standalone PR)

**Title:** `refactor(SPEC-30): year-agnostic pipeline scaffold for 2026`

**Labels:** `type:refactor`, `spec:30`, `priority:critical`, `status:ready`

#### Reason

Every shared helper and model module currently uses 2022 literals (`ELECTION_DATE_ROUND1`, `FIRST_ROUND_CANDIDATES`, `POLLSTER_RATINGS`, `CONSULTATION_VOTES`, `pop_2022`, etc.). Before any 2026 data can be ingested or modeled, the code must accept a `year` parameter and resolve dates, registries, ratings, and result loaders per year. This must be a standalone PR so that 2026 work branches from a clean, working 2022 baseline.

#### Description

Refactor configuration, data loading, model building, and reporting to be year-aware. Add a `get_election_dates(year)` helper, populate 2026 registries, make municipal feature columns year-aware, and ensure 2022 backtests remain bit-for-bit identical. No 2026-specific business logic beyond configuration belongs in this PR.

#### Dependencies

- None — this is the foundation for SPEC-31 → SPEC-36.

#### Task List

- [ ] Add `get_election_dates(year)` returning `(r1_date, runoff_date, consultation_date)`.
- [ ] Add constants `ELECTION_DATE_ROUND1_2026`, `ELECTION_DATE_ROUND2_2026`, `CONSULTATION_DATE_2026`.
- [ ] Populate `FIRST_ROUND_CANDIDATES_2026` and `CANDIDATE_KEY_MAP_2026` with aliases.
- [ ] Add `POLLSTER_RATINGS_2026` (initially empty or seeded from SPEC-32).
- [ ] Add `CONSULTATION_VOTES_2026` / `CONSULTATION_KEY_MAP_2026` (or a disable-consultation flag).
- [ ] Make municipal feature extraction year-aware so it can use `pop_2026`.
- [ ] Refactor `load_and_clean_all()` to accept `year=2022` and dispatch paths/registry/dates by year.
- [ ] Refactor `model_round1.py` to accept `election_date` and candidate registry instead of 2022 literals.
- [ ] Refactor `model_runoff_matrix.py` / `model_runoff_simple.py` to accept runoff date and candidate registry.
- [ ] Update `report_utils.py` (`ReportInputs`, formatting helpers) to use year-aware candidate registry.
- [ ] Update `load_canonical_results()` to accept `year=2022`.
- [ ] Update `scripts/run_100k_report.py` and `scripts/forward_backtest.py` to pass `year=2022` explicitly.
- [ ] Write unit tests for year-aware helpers.
- [ ] Run `make check`.

#### Completion Criteria

- [ ] `make check` exits 0.
- [ ] 2022 `run_100k_report.py` and `forward_backtest.py` outputs are identical to the pre-refactor baseline.
- [ ] 2026 config registries are populated and unit-tested.
- [ ] No literal 2022 dates or candidate names remain in shared helper code.

#### References

- `MVP_SPECS_GUIDE.md` §configuration
- `AGENTS.md` §3, §5
- `scripts/run_100k_report.py`
- `scripts/forward_backtest.py`
- `src/co_president/config.py`
- `src/co_president/data_polls.py`
- `src/co_president/model_round1.py`
- `src/co_president/model_runoff_matrix.py`
- `src/co_president/model_runoff_simple.py`
- `src/co_president/report_utils.py`

---

### Phase 1 — SPEC-31: Ingest 2026 CNE poll bundles

**Title:** `data(SPEC-31): ingest 23 CNE 2026 poll bundles into machine-readable parquet files`

**Labels:** `type:data`, `spec:31`, `priority:critical`, `status:ready`

#### Reason

GitHub issue #221 / SPEC-31 requires converting the ~1.2 GB of heterogeneous CNE 2026 microdata bundles into a canonical, weighted topline table and a runoff head-to-head pairing table. This is the single highest-risk data blocker because formats vary across CSV, Excel, SPSS `.sav`, and PDF.

#### Description

Create `src/co_president/data_cne_2026.py` with a canonical candidate normalizer, per-firm loaders, and a PDF extraction fallback. Produce `data/2026-polls/_processed/2026_topline.parquet` and `2026_runoff_pairings.parquet`. Weighted toplines must respect each bundle's design weights when available.

#### Dependencies

- Phase 0 (year-agnostic scaffold) — candidate registry and pollster map must exist.

#### Task List

- [ ] Define canonical candidate normalizer mapping raw labels → keys (`cepeda`, `de_la_espriella`, `valencia`, `fajardo`, `claudia_lopez`, `rest`, `blanco`).
- [ ] Implement Atlas Intel CSV loader (R1 variable `presidential_election_2026`, weight column).
- [ ] Implement Atlas Intel runoff loader (`second_round_president_2026_co`, `rejection_cepeda_vs_espriella`).
- [ ] Implement Invamer Excel microdata loader.
- [ ] Implement CNC SPSS `.sav` loader.
- [ ] Implement GAD3 Excel loader.
- [ ] Implement PDF extraction fallback for Guarumo/Ecoanalítica, Génesis Crea, TEMPO, Corporación Miguel Maldonado, Analizar + Lombana.
- [ ] Compute weighted toplines per poll (apply design weights, scale to percentages).
- [ ] Extract head-to-head runoff pairings where microdata variable exists.
- [ ] Cache outputs to `data/2026-polls/_processed/`.
- [ ] Add ingestion tests using tiny synthetic fixtures for each format.
- [ ] Document extraction gaps and fallback assumptions.
- [ ] Run `make check`.

#### Completion Criteria

- [ ] `src/co_president/data_cne_2026.py` exists and is tested.
- [ ] `2026_topline.parquet` exists with columns: `fecha`, `encuestadora`, `field_end`, `muestra`, candidate columns, `metodologia`, etc.
- [ ] `2026_runoff_pairings.parquet` exists with columns: `fecha`, `encuestadora`, `candidate_a`, `candidate_b`, `n_a`, `n_b`, `n_total`.
- [ ] PDF-only firms are either extracted or explicitly flagged with a documented reason.
- [ ] `make check` exits 0.

#### References

- GitHub #221
- `docs/plans/PLAN_cne_2026_ingestion.md`
- `data/2026-polls/` bundle inventory
- `src/co_president/ingestion/` patterns

---

### Phase 2 — SPEC-32: Cross-validate 2026 toplines and define 2026 pollster ratings

**Title:** `data(SPEC-32): cross-validate 2026 toplines vs La Silla Vacía and add POLLSTER_RATINGS_2026`

**Labels:** `type:data`, `spec:32`, `priority:high`, `status:ready`

#### Reason

GitHub issue #222 / SPEC-32 requires validating that the ingested CNE toplines match the independent La Silla Vacía `encuestas_detalle.csv` within 0.5 pp, and assigning the 2026 pollster ratings that the model will use as quality weights.

#### Description

Match CNE-ingested polls to La Silla Vacía rows by firm and field end date. Compute absolute per-candidate differences, flag >0.5 pp discrepancies, resolve or document them, and add `POLLSTER_RATINGS_2026` to `config.py`.

#### Dependencies

- SPEC-31 outputs (`2026_topline.parquet`).

#### Task List

- [ ] Load `data/2026-polls/silla_vacia_ponderador/encuestas_detalle.csv` and pivot to poll-level.
- [ ] Match polls by firm alias and `field_end` / `Fecha_Fin_Campo`.
- [ ] Compute per-candidate absolute differences.
- [ ] Flag polls with any difference >0.5 pp.
- [ ] Produce a discrepancy report (Markdown) in `docs/results/`.
- [ ] Resolve discrepancies where due to weighting or candidate aliasing; document irreducible ones.
- [ ] Add `POLLSTER_RATINGS_2026` dict with 8 firms and their ratings (Atlas Intel 5.9, CNC 5.8, GAD3 6.2, Guarumo 5.1, Invamer 5.3, Génesis Crea 2.0, TEMPO 2.0, Corporación MMM 2.0).
- [ ] Add validation tests.
- [ ] Run `make check`.

#### Completion Criteria

- [ ] Discrepancy report generated.
- [ ] Less than 5% of matched polls remain flagged with unresolved >0.5 pp differences (or each is documented).
- [ ] `POLLSTER_RATINGS_2026` is defined and unit-tested.
- [ ] `make check` exits 0.

#### References

- GitHub #222
- `docs/plans/PLAN_cne_2026_ingestion.md`
- `data/2026-polls/silla_vacia_ponderador/`
- `src/co_president/config.py`

---

### Phase 3 — SPEC-33: Build 2026 CleanPolls pipeline and national canonical R1 result

**Title:** `feat(SPEC-33): build 2026 CleanPolls pipeline and national canonical R1 result`

**Labels:** `type:feat`, `spec:33`, `priority:high`, `status:ready`

#### Reason

GitHub issue #223 / SPEC-33 requires a model-ready `CleanPolls` object for 2026 and canonical 2026 election dates. The 2026 R1 result is only available as 33 departmental MMV CSVs, so a national aggregator is also required.

#### Description

Implement `build_clean_polls_2026()` returning `CleanPolls`, using the ingested 2026 topline and runoff tables. Classify polls as R1 (field end ≤ 2026-05-22) or runoff (June). Aggregate departmental MMV CSVs into a national `RoundResult` and expose it via `load_canonical_results(2026)`. Add `CONSULTATION_VOTES_2026` if consultation data is available; otherwise implement a disable-consultation flag.

#### Dependencies

- Phase 0
- SPEC-31 (`2026_topline.parquet`, `2026_runoff_pairings.parquet`)
- SPEC-32 (`POLLSTER_RATINGS_2026`)

#### Task List

- [ ] Implement `build_clean_polls_2026()` returning `CleanPolls`.
- [ ] Classify R1 vs runoff polls using 2026 cutoff dates.
- [ ] Apply deduplication, undecided redistribution, and pollster-rating weights.
- [ ] Aggregate 33 departmental MMV CSVs into national vote totals by candidate.
- [ ] Add `load_canonical_results(2026)` returning `(r1_result, r2_result)` with `r2_result=None`.
- [ ] Define `CONSULTATION_VOTES_2026` / `CONSULTATION_KEY_MAP_2026` or add a consultation-prior disable flag.
- [ ] Update `scripts/run_2026_backtest.py` to load 2026 data instead of 2022.
- [ ] Add tests using a synthetic departmental MMV fixture.
- [ ] Run `make check`.

#### Completion Criteria

- [ ] `build_clean_polls_2026()` returns a valid `CleanPolls` object.
- [ ] National 2026 R1 result matches the sum of departmental CSVs within rounding.
- [ ] `scripts/run_2026_backtest.py` no longer loads 2022 polls or results.
- [ ] `make check` exits 0.

#### References

- GitHub #223
- `docs/plans/PLAN_cne_2026_ingestion.md`
- `data/2026-elections/2026-presidential1/`
- `src/co_president/data_polls.py`

---

### Phase 4 — SPEC-34: Empirical runoff matrix from CNE head-to-head microdata

**Title:** `feat(SPEC-34): empirical Beta posteriors for runoff matrix from CNE head-to-head microdata`

**Labels:** `type:feat`, `spec:34`, `priority:critical`, `status:ready`

#### Reason

GitHub issue #224 / SPEC-34 requires upgrading `model_runoff_matrix.py` so that runoff pairing estimates are driven by observed head-to-head CNE microdata rather than only simulated transfer priors. This is essential for a credible 2026 runoff forecast because the 2022 transfer structure may not apply.

#### Description

Compute effective sample sizes from design weights per head-to-head pairing. Fit `Beta(a, b)` posteriors and cache them. Modify `estimate_runoff_matrix` to sample from the empirical posterior when `n_eff >= 50`, falling back to the existing Dirichlet-Multinomial simulation for unpolled or sparse pairings.

#### Dependencies

- Phase 0
- SPEC-31 (`2026_runoff_pairings.parquet`)

#### Task List

- [ ] Compute `n_eff` per pairing using design weights.
- [ ] Fit `Beta(a, b)` per head-to-head observation; pool across polls or by firm.
- [ ] Cache empirical transfer posterior in `results/`.
- [ ] Modify `estimate_runoff_matrix` to prefer empirical Beta posterior when `n_eff >= 50`.
- [ ] Preserve Dirichlet-Multinomial fallback for pairings below the threshold.
- [ ] Validate that transfer rates satisfy sum and positivity constraints.
- [ ] Add unit tests with synthetic head-to-head data.
- [ ] Run `make check`.

#### Completion Criteria

- [ ] Empirical Beta posterior is used for all pairings with `n_eff >= 50`.
- [ ] Fallback path unchanged and tested for unpolled pairings.
- [ ] 2022 backtest still passes with the refactored matrix (no regression).
- [ ] `make check` exits 0.

#### References

- GitHub #224
- `docs/plans/PLAN_cne_2026_ingestion.md`
- `src/co_president/model_runoff_matrix.py`
- `src/co_president/model_transfer.py`

---

### Phase 5 — SPEC-35: Refit models on 2026 data and validate (first backtest)

**Title:** `feat(SPEC-35): refit R1 and runoff models on 2026 data and validate vs La Silla Vacía`

**Labels:** `type:feat`, `spec:35`, `priority:high`, `status:ready`

#### Reason

GitHub issue #225 / SPEC-35 requires running the first real 2026 backtest. This is where the project discovers whether the model, signals, and covariates produce a usable forecast. It must include digital signals, Petro approval, and TerriData enrichment per the user's decision.

#### Description

Create `scripts/run_2026_backtest.py` that loads 2026 `CleanPolls`, canonical result, features, Google Trends, and Petro approval; runs the R1 model and empirical runoff matrix; and produces a report matching the 2022 format. Validate R1 posterior means against `Prediccion_30d` and the May 31 actuals.

#### Dependencies

- Phase 0
- SPEC-33 (`build_clean_polls_2026`, canonical 2026 result)
- SPEC-34 (empirical runoff matrix)

#### Task List

- [ ] Update `CANDIDATE_QUERY_MAP_2026` in `trends_keywords.py` with real 2026 candidate keys.
- [ ] Fetch/cache Google Trends through 2026-05-30.
- [ ] Extract Petro government approval series (`approve_disapprove_federal_government`) from Atlas/Invamer microdata and expose as a digital/covariate signal.
- [ ] Integrate TerriData/GEIH/ECV/IPM/MPMD covariates via existing `benchmarks/terridata.py` and fundamentals pipeline for the first backtest.
- [ ] Create `scripts/run_2026_backtest.py` analogous to `run_100k_report.py`.
- [ ] Run R1 model with 2026 polls, features, digital signals, and approval covariate.
- [ ] Run empirical runoff matrix on top two candidates.
- [ ] Validate R1 posterior mean vs La Silla Vacía `Prediccion_30d` and vs national May 31 result; compute MAE and coverage.
- [ ] Generate `results/baseline_2026_report.{md,json}` matching 2022 report format.
- [ ] Document model performance and sensitivity.
- [ ] Add tests for new signal/covariate loaders.
- [ ] Run `make check`.

#### Completion Criteria

- [ ] `scripts/run_2026_backtest.py` runs end-to-end without errors.
- [ ] `results/baseline_2026_report.{md,json}` is generated and matches 2022 report structure.
- [ ] R1 MAE and coverage are computed and documented.
- [ ] `make check` exits 0.

#### References

- GitHub #225
- `docs/plans/PLAN_cne_2026_ingestion.md`
- `scripts/run_100k_report.py`
- `scripts/run_2026_backtest.py` (existing prototype)
- `src/co_president/ingestion/trends_keywords.py`
- `src/co_president/benchmarks/terridata.py`
- `data/digital_signals/`
- `data/raw/TerriData/`

---

### Phase 6 — SPEC-36 (stretch): Demographic / primary-recall hierarchical offsets in 2026 runoff

**Title:** `feat(SPEC-36): demographic and primary-recall hierarchical offsets for 2026 runoff`

**Labels:** `type:feat`, `spec:36`, `priority:low`, `status:ready`

#### Reason

GitHub issue #226 / SPEC-36 is a stretch goal to improve runoff inference by modeling respondent-level heterogeneity from the CNE microdata.

#### Description

Add hierarchical group-level intercepts to the 2026 runoff model based on demographics (age, gender, education, income, region) and primary/recall vote (`vote_2021_presidential_elections_firstround_final_choice`, `vote_primaries_2026`). Gate the feature via `ModelConfig` flags so it is opt-in.

#### Dependencies

- SPEC-35 (working 2026 backtest)
- SPEC-31 (microdata with demographic/recall columns)

#### Task List

- [ ] Define demographic groups from microdata.
- [ ] Add group-level intercepts to `model_runoff_simple.py` for 2026.
- [ ] Add primary-recall offsets.
- [ ] Add `ModelConfig` flags to enable/disable offsets.
- [ ] Run convergence and sanity tests on minimal data.
- [ ] Run slow MCMC tests.
- [ ] Document runtime and convergence behavior.

#### Completion Criteria

- [ ] Model builds and converges on 2026 data when the flag is enabled.
- [ ] Flag works and defaults to off.
- [ ] `make check` exits 0; slow tests pass or have documented acceptable R-hat.

#### References

- GitHub #226
- `docs/plans/PLAN_cne_2026_ingestion.md`
- `src/co_president/model_runoff_simple.py`
- `src/co_president/config.py`

---

### Phase 7 — Final deliverable: 2026 runoff forecast with statistical significance

**Title:** `feat(SPEC-37): final 2026 runoff forecast with statistical significance`

**Labels:** `type:feat`, `spec:37`, `priority:critical`, `status:ready`

#### Reason

This is the ultimate project goal. After the backtests validate the pipeline, produce a true forward forecast for 2026 R1 and the runoff, with credible intervals and a clear statement of statistical significance.

#### Description

Run a true forward backtest for 2026 R1 without using the election result. Use the R1 posterior to identify the top two finalists and the empirical runoff matrix to estimate runoff outcomes. Produce final reports with 95% credible intervals, winner probabilities, and significance thresholds.

#### Dependencies

- SPEC-35 (validated 2026 backtest)
- SPEC-34 (empirical runoff matrix)
- SPEC-36 optional (demographic offsets)

#### Task List

- [ ] Create `scripts/forward_2026_backtest.py` analogous to `scripts/forward_backtest.py`.
- [ ] Use polls through 2026-05-22 and digital signals through 2026-05-30.
- [ ] Run R1 model **without** 2026 canonical result.
- [ ] Identify top two candidates by posterior probability of finishing first/second.
- [ ] Run empirical runoff matrix using R1 posterior and head-to-head microdata.
- [ ] Compute 95% credible intervals and `p(winner)` for the runoff.
- [ ] Produce `results/forward_2026_report.{md,json}` matching 2022 forward backtest format.
- [ ] Produce final deliverable `results/runoff_forecast_2026_report.{md,json}` with narrative, tables, and charts.
- [ ] Write statistical significance statement (e.g., "Candidate X has a posterior probability >95% of winning the runoff").
- [ ] Run CodeRabbit review before committing.
- [ ] Run `make check`.

#### Completion Criteria

- [ ] `scripts/forward_2026_backtest.py` runs end-to-end.
- [ ] `results/forward_2026_report.{md,json}` is generated.
- [ ] `results/runoff_forecast_2026_report.{md,json}` is generated.
- [ ] Runoff forecast includes 95% credible intervals and a statistical significance statement.
- [ ] `make check` exits 0.

#### References

- `scripts/forward_backtest.py`
- `MVP_SPECS_GUIDE.md`
- `docs/results/`
- `AGENTS.md` §9 (CodeRabbit review)

---

## 3. Branch and sequencing summary

| Phase | Branch | Main output | PR base |
|---|---|---|---|
| 0 | `feat/spec-30-year-agnostic-scaffold` | Year-aware config, data, model, report helpers | `dev` |
| 1 | `feat/spec-31-cne-ingest` | `data_cne_2026.py`, `2026_topline.parquet`, `2026_runoff_pairings.parquet` | `dev` (after Phase 0 merges) |
| 2 | `feat/spec-32-cne-validation` | Discrepancy report, `POLLSTER_RATINGS_2026` | `dev` |
| 3 | `feat/spec-33-cleanpolls-2026` | `build_clean_polls_2026()`, national 2026 R1 result | `dev` |
| 4 | `feat/spec-34-empirical-runoff` | Empirical Beta transfer posteriors | `dev` |
| 5 | `feat/spec-35-refit-2026` | `scripts/run_2026_backtest.py`, baseline report | `dev` |
| 6 | `feat/spec-36-demographic-offsets` | Hierarchical offsets, opt-in flags | `dev` |
| 7 | `feat/spec-37-final-forecast` | Forward + runoff deliverable reports | `dev` |

---

## 4. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| PDF-only bundles cannot be extracted accurately | Medium | High | Use multiple extractors, document gaps, keep PDF toplines as a separate lower-confidence tier |
| 2026 runoff result unavailable before final forecast | High | Medium | Final deliverable is a true forecast; validate via backtests and sensitivity analysis |
| Google Trends rate-limits or missing candidate queries | Medium | Medium | Cache early, add fallback to multi-candidate normalization, document signal absence |
| CleanPolls threshold fails for 2026 (few R1/R2 pollsters) | Low | High | Relax thresholds for 2026 or document exception; ensure La Silla data can supplement if needed |
| Head-to-head microdata too sparse for empirical Beta posteriors | Medium | High | Lower `n_eff` threshold or pool across firms; keep Dirichlet fallback |
| 100K MCMC runtime too long for iteration | Medium | Medium | Use `numpyro` sampler, reduce to 5K/5K for development, reserve 100K for final report |
| TerriData/GEIH/ECV integration adds noise rather than signal | Low | Medium | Evaluate covariate importance in first backtest; drop if harmful |

---

## 5. Definition of done for the entire plan

- [ ] Phase 0 scaffold is merged as a standalone PR and 2022 backtests are regression-free.
- [ ] SPEC-31 → SPEC-36 are implemented, reviewed, and merged into `dev`.
- [ ] `scripts/run_2026_backtest.py` produces a 2026 R1 backtest report matching the 2022 format.
- [ ] `scripts/forward_2026_backtest.py` produces a true forward 2026 R1 + runoff forecast report.
- [ ] Final runoff forecast report includes statistical significance statement.
- [ ] All quality gates (`make check`, CodeRabbit review) pass for every PR.
