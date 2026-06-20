# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), grouped by milestone.

---

## [Unreleased]

### Added
- docs/specs/STATUS.md — centralized SPEC status tracker
- docs/README.md — documentation directory index
- CONTRIBUTING.md — human-readable contribution guide
- `docs/archive/` — superseded planning documents
- SPEC-23 → SPEC-29: Extension feature shipments (MOE PDF X-validation, ECP signal, AS/COA polls, OOS validation, horseshoe regularization, 2026 forecast mode, generalization audit)
- SPEC-30: Transfer-rate estimation model (`model_transfer.py`) with probabilistic pairing matrix
- SPEC-37: Compositional ALR/ILR transforms for target-side application (`benchmarks/transforms.py`)
- SPEC-38: Google Trends ingestion pipeline (`ingest_trends.py`, `trends_keywords.py`)
- SPEC-39: Twitter sentiment pipeline (RoBERTuito, GPT-4o-mini, spam filter, sentiment series)
- SPEC-40: Multi-election hierarchical model refactor for 2002-2026 training
- SPEC-41: FNN+CLR ML benchmark with SVR/RF/GB/KNN baselines (R² 0.94)
- `SPEC-41-FNN-CLR-BENCHMARK-REPORT.md` — FNN+CLR benchmark report documentation

### Changed
- `AGENTS.md` — fixed PyMC version reference (≥6.0 → ≥5.15), removed stale pytest-fast hook claim, removed unavailable python-executor skill
- `docs/specs/STATUS.md` — full SPEC status reconciliation (23→29 shipped, 30 in progress)
- `ENHANCEMENT_ROADMAP.md` — updated scope header and last-updated date
- `README.md` — updated project structure and specs roadmap
- `.pre-commit-config.yaml` — removed stale pytest-fast hook
- `.github/workflows/slow-tests.yml` — reconciled setup-uv to v6
- `.gitignore` — added 2026 data dirs and parallel coverage artifacts
- `pyproject.toml` — added PD013 per-file-ignore to model_municipal.py

### Removed
- `HANDOFF.md`, `.pr_body.md`, `branch-protection-ci-removal.md` — root-level leftovers
- `.claude/skills/pandas-data-analysis` — broken symlink
- `CNE_DATA_PLAN.md` — archived (superseded by `docs/PLAN_cne_2026_ingestion.md`)
- `results/ROADMAP.md` — superseded by `ENHANCEMENT_ROADMAP.md`

---

## v0.1.0 — 2026-06-09

### MVP: National Polling Model (SPEC-01 → SPEC-11)

Initial release. Bayesian presidential election forecaster for Colombia, backtested against 2022 results.

#### SPEC-01: Project scaffolding
- Python 3.12 project with uv/hatchling build
- `paths.py` — data directory resolution
- GitHub Actions CI with 5 parallel workflows
- Pre-commit hooks (ruff, pyright, detect-private-key, large-files)
- `pyproject.toml` with ruff ALL rules, pyright strict
- AGENTS.md — agent operating manual

#### SPEC-02: Configuration
- `Candidate` and `ModelConfig` dataclasses
- All 2022 candidates, pollster ratings, election dates
- Transfer constants for runoff vote projection
- Consultation prior strengths
- Pollster weight helper functions

#### SPEC-03: Results consolidation
- `load_canonical_results()` — Registraduría MMV loader
- MOE cross-validation
- `RoundResult` dataclass with participation data
- Support for both rounds

#### SPEC-04: Poll loading & cleaning
- `load_and_clean_all()` — full pipeline
- Undecided voter redistribution
- Invamer date correction (2022-04-19 → 2022-05-19)
- Deduplication and forced-choice detection
- GAD3 tracking wave integration (11 waves)
- Cross-pollster consistency validation

#### SPEC-05: Baseline aggregation
- Time-decay sample-size weighting
- Pollster rating weighting
- `compute_weighted_poll_average()`

#### SPEC-06: 1st-round Bayesian model
- Dirichlet-Multinomial + reverse-time random walk
- Hierarchical pollster house effects (zero-sum constrained)
- Sample-size-dependent Dirichlet concentration
- 4-phase MCMC tests (graph, prior predictive, convergence, sanity)
- `simulate_elections()` with outright win and runoff scenarios

#### SPEC-07: Runoff simple (K=3)
- Dirichlet runoffs (A vs B vs rest+blanco)
- Informed prior with default pollster weight

#### SPEC-08: Runoff pairing matrix
- `estimate_runoff_matrix()` — head-to-head polling
- Transfer heuristic fallback (73/27 aggregate split)
- `overall_win_probability()` — top-two win probabilities

#### SPEC-09: Validation & plotting
- MAE, RMSE, Brier score metrics
- Calibration analysis (50%, 95% HDI)
- Rolling forecast snapshots
- Sensitivity flags (high NS/NR, cross-pollster disagreement)
- Visualization: forecast evolution, calibration, error-over-time

#### SPEC-10: CLI
- `co-president run` — full pipeline (validate → aggregate → run → forecast)
- `co-president validate`, `aggregate`, `plot`, `ingest`, `forecast`, `config`
- `--config-override KEY=VALUE` parsing

#### SPEC-11: Data quality diagnostics
- Pollster rating validation
- Methodology effect analysis
- Time-decay regression
- Cross-pollster consistency checks

---

### Post-MVP: Municipal Hierarchical Model Foundation (SPEC-12 → SPEC-22)

#### SPEC-12.1: DIVIPOLA master registry
- 1,122 municipalities from datos.gov.co Socrata API (sodapy)
- Code validation, type normalization, fundamentals export

#### SPEC-12.2: Historical election results
- CEDAE local file reader (2002-2022)
- Candidate mapping, vote aggregation by municipality
- Lagged historical features
- Bulk CEDAE downloader annex

#### SPEC-13a: Socioeconomic demographics
- DANE socioeconomic/demographic data ingest (ethnicity, education, internet)
- Stub fallback (Bogotá/Medellín/Cali) when remote unavailable

#### SPEC-13.2 / SPEC-20: Electoral risk
- Three-tier cascade (local CSV → remote API → hardcoded fallback)
- MOE risk maps, INDEPAZ conflict data, PDET presence, UNODC coca
- 170 PDET municipality hardcoded fallback from official ART list

#### SPEC-14: Feature matrix
- `build_feature_matrix.py` — joins 8+ fundamental components
- Writes `municipal_feature_matrix.{csv,parquet}`
- 1,123 rows × 129 columns

#### SPEC-15: Coverage report
- `report.py` — pipeline provenance, source_map, COVERAGE.md
- Legislative schema cross-audit

#### SPEC-16: Sabaneta fixture
- Sabaneta municipal Cámara de Representantes CSV parser
- Validation, matrix builder, CLI integration

#### SPEC-17: CNPV 2018 census
- 33 departmental ZIP microdata (F8/F9/F11 CSVs inside nested zips)
- 300+ feature columns per census form
- Downloader script, ingestion pipeline, feature matrix merge

#### SPEC-18: Poverty indicators
- **NBI** (primary): CNPV 2018 needs-basic-insatisfechas at municipal level
- **IPM** (secondary): ECV 2018/2022 multidimensional poverty at dept level
- R² fidelity guard for departamento→municipio propagation

#### SPEC-19: Population projections
- DANE PPED projections 2018-2026
- 1,122 municipalities, 8 years
- Population-weighted target candidate computation

#### SPEC-20: Risk indicators (extended)
- Three-tier cascade (local/remote/fallback)
- Expanded fallback list for all 1,122 municipalities
- PDET, INDEPAZ, UNODC signals

#### SPEC-21: Municipal features API
- `MunicipalFeatures` — 64-field frozen dataclass
- `load_features()` — typed schema validation with Series dtype checks
- CLR and logit compositional data transformation helpers

#### SPEC-21a: Fiscal autonomy
- TerriData fiscal autonomy variables (.xlsx.zip)
- 8-year averaging (2018-2024)
- Pct-ingresos-propios, pct-gastos-funcionamiento

#### SPEC-21b: Bogotá localidad disaggregation
- Bogotá D.C. (cod 11001) → 20 localidades + SIN COMUNA
- Join MMV with participation, propagate upstream

#### SPEC-21c: Compositional transforms
- `clr()`, `logit()`, `inv_logit()` with input validation
- Log-ratio transform design for municipal feature space

#### SPEC-22: Municipal hierarchical model
- 3-layer logistic-normal / Dirichlet-Multinomial
- η_local (municipal) → η_dept (dept) → η_natl (national) pooling
- 4-phase MCMC tests (graph, prior predictive, convergence, sanity)
- Municipal `sample_size` and `effective_n` application
