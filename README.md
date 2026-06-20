<div align="center">

# co-president-2026

**A Bayesian presidential election forecaster in Python — backtested on 2022 data, ready for 2026.**

[![license: MIT + CC BY 4.0](https://img.shields.io/badge/license-MIT%20%2B%20CC%20BY%204.0-blue.svg)](./LICENSE)
[![python: 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![pymc: 5.x](https://img.shields.io/badge/PyMC-5.x-orange.svg)](https://www.pymc.io/)
[![ruff](https://img.shields.io/badge/ruff-enabled-a307e1.svg)](https://docs.astral.sh/ruff/)
[![pyright](https://img.shields.io/badge/pyright-strict-4ade80.svg)](https://microsoft.github.io/pyright/)

</div>

---

## Table of contents

- [What is it?](#what-is-it)
- [Features](#features)
- [How it works](#how-it-works)
- [About the model](#about-the-model)
- [Data sources (MVP)](#data-sources-mvp)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Commands reference](#commands-reference)
- [Project structure](#project-structure)
- [Development](#development)
  - [Getting started](#getting-started)
  - [TDD workflow](#tdd-workflow)
  - [Makefile targets](#makefile-targets)
  - [Quality gates](#quality-gates)
- [Tests](#tests)
- [Specs roadmap](#specs-roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## What is it?

`co-president-2026` ingests Colombian presidential poll data, cross-validates it against official election results from two independent sources, fits a Bayesian Dirichlet-Multinomial model with a reverse-time random walk, and produces probabilistic forecasts for both the first round and a potential runoff.

The MVP backtests the pipeline against the **2022 Colombian presidential election** using only national-level polls. Once the model can predict 2022 within ±5 percentage points, the same code adapts to 2026 by swapping data files and updating candidate configurations.

The stack: **Python 3.12 + PyMC** for Bayesian inference, **pandas + numpy** for data wrangling, **ArviZ** for MCMC diagnostics, and **matplotlib** for visualization.

---

## Features

- **Ingestion & cleaning** — loads poll data from multiple pollsters, corrects known data-entry errors (e.g., the Invamer 2022-04-19 → 2022-05-19 fix), deduplicates same-date/same-pollster entries, and redistributes undecided voters proportionally across all candidates.

- **Results consolidation** — aggregates polling-station-level results from the **Registraduría Nacional** (~1.1M rows across two rounds) and independently from the **MOE** (Misión de Observación Electoral), cross-validates the totals, and produces a single canonical `RoundResult` per round.

- **Bayesian modeling** — Dirichlet-Multinomial observation model with softmax-transformed logit-scale vote intentions, a reverse-time random walk anchored at election day, and hierarchical pollster house effects with a zero-sum constraint. An informative prior is derived from the March 2022 inter-party consultation results.

- **Probabilistic forecasts** — for the multi-candidate first round: per-candidate mean share, 50% and 95% credible intervals, probability of finishing 1st/2nd, and overall win probability. For the runoff: head-to-head probabilities and margin distributions.

- **Backtesting & validation** — MAE, RMSE, Brier score, calibration checks, and rolling forecasts that re-fit the model at successive cutoff dates to show prediction accuracy tightening as election day approaches.

- **Strict TDD + type safety** — every module is tested before implementation, all functions are fully annotated, and three gates (`ruff`, `pyright`, `pytest`) must pass before every commit.

---

## How it works

```
  encuestas_2022.csv ──┐
                        │     load_and_clean_all()
  consultas.csv ────────┤     ┌───────────────────────┐
                        ├─────▶  Poll cleaning         │
                        │     │  - fix date errors     │
                        │     │  - normalize undecided │
                        │     │  - infer round number  │
                        │     │  - deduplicate         │
                        │     └───────────────────────┘
                        │              │
                        │              ▼
  Registraduría ─────┐  │     ┌───────────────────────┐
  MMV files          │  │     │  CleanPolls            │
  (2 rounds)         ├──┤     │  round1: DataFrame     │
                     │  │     │  round2: DataFrame     │
  MOE files          │  │     │  consultation: list    │
  (2 rounds)         ├──┘     └───────────────────────┘
                     │                  │
                     ▼                  ▼
  ┌───────────────────────┐  ┌───────────────────────┐
  │  load_canonical_result│  │  Bayesian Model        │
  │  RoundResult (1)      │  │  Dirichlet-Multinomial │
  │  RoundResult (2)      │  │  + reverse-time random │
  └───────────────────────┘  │    walk + house effects│
              │              └───────────────────────┘
              │                          │
              ▼                          ▼
  ┌───────────────────────┐  ┌───────────────────────┐
  │  Validation            │  │  Forecast              │
  │  - MAE, RMSE, Brier    │  │  Round1Forecast        │
  │  - calibration checks  │  │  RunoffForecast        │
  │  - rolling forecast    │  │  RunoffMatrix          │
  │  - plots               │  │  prob_runoff, CI       │
  └───────────────────────┘  └───────────────────────┘
```

---

## About the model

The core model is a **Dirichlet-Multinomial** with three key components:

**Reverse-time random walk.** Instead of a forward random walk starting from a vague prior, the walk is anchored at election day (`θ[0]`) and uncertainty grows as we move backward in time. This matches the intuition that we know less the further we are from election day. The earliest time point (`θ[T-1]`) receives an informative prior derived from the March 2022 inter-party consultation results — real voter behavior, not a diffuse prior.

**House effects.** Each pollster gets a per-candidate bias parameter on the logit scale. The biases are zero-sum constrained across pollsters for each candidate, meaning they represent *relative* rather than absolute shifts. A pollster that consistently overestimates a candidate by 2 points is captured here, rather than distorting the latent vote intention.

**Election-day anchoring.** The actual election result is modeled as a second likelihood with its own Dirichlet-Multinomial observation, using a much higher concentration parameter (φ_elec >> φ_poll). This reflects that election results are far less noisy than polls while still respecting the stochastic nature of a single observed outcome.

The model is implemented in **PyMC** using the NUTS sampler. MCMC diagnostics (R-hat, ESS) are checked on every fit; the model is reparameterized as needed to avoid divergent transitions.

---

## Data sources (MVP)

| File | Source | Rows | Columns | Granularity | Encoding |
|------|--------|------|---------|-------------|----------|
| `encuestas_2022.csv` | recetas-electorales.com (Nelson Amaya) | 46 | 27 | National polls | UTF-8, comma-delimited |
| `consultas.csv` | recetas-electorales.com | 65 | 7 | Coalition internal polls | ISO-8859-1, comma-delimited |
| `MMV_NACIONAL_PRESIDENTE_2022_1v.csv.gz` | Registraduría Nacional | 727,510 | 16 | Polling-station-level | ISO-8859-1, semicolon-delimited |
| `MMV_NACIONAL_PRESIDENTE_2022_2v.csv.gz` | Registraduría Nacional | 398,387 | 16 | Polling-station-level | ISO-8859-1, semicolon-delimited |
| `moe_vuelta1.csv` | MOE | 11,864 | 15 | Municipal-level | UTF-8, comma-delimited |
| `moe_vuelta2.csv` | MOE | 5,550 | 15 | Municipal-level | UTF-8, comma-delimited |
| `reg_participacion_vuelta1.csv` | Registraduría Nacional | ~33,000 | 17 | Polling-station-level | UTF-8-BOM, comma-delimited |
| `reg_participacion_vuelta2.csv` | Registraduría Nacional | ~33,000 | 17 | Polling-station-level | UTF-8-BOM, comma-delimited |

Polls are entirely national-level — no departmental or regional breakdowns. Results data is aggregated from station/municipal level to national totals for cross-validation and model anchoring.

---

## Prerequisites

- **Python 3.12+** — install via [pyenv](https://github.com/pyenv/pyenv) or your OS package manager.
- **`uv`** — for dependency management. See [uv installation](https://docs.astral.sh/uv/getting-started/installation/).
- **Data decompression** — The Registraduría MMV files must be decompressed once after cloning:

```bash
gunzip data/2022-presidential-results/MMV_NACIONAL_PRESIDENTE_2022_*.csv.gz
```

The `.gitignore` ignores the uncompressed CSVs to prevent accidental re-commits. The `Makefile` `sync` target does NOT auto-decompress — it's a manual one-time step.

---

## Installation

```bash
# Clone the repo
git clone https://github.com/<your-org>/co-president-2026
cd co-president-2026

# Install dependencies and the local package (editable)
make setup

# Verify installation
uv run python -c "import co_president; print(co_president.__version__)"
```

To update all dependencies: `uv sync --upgrade`.

---

## Usage

```bash
# Full pipeline: clean data, fit model, validate, print results
uv run python -m co_president run

# Run without MCMC (use baseline weighted average instead)
uv run python -m co_president run --no-sample

# Baseline weighted average only (quick diagnostics)
uv run python -m co_president aggregate

# Validate against saved model output
uv run python -m co_president validate

# Generate all plots
uv run python -m co_president plot --output-dir results/

# Show current configuration
uv run python -m co_president config

# Ingest fundamental data components (default: sabaneta; use --component for others)
uv run python -m co_president ingest                  # sabaneta (default)
uv run python -m co_president ingest --component cnpv # specific component

# 2026 forecast mode
uv run python -m co_president forecast --mode prior_only

# Override config from the command line
uv run python -m co_president run --config-override seed=42
```

The `run` command produces a summary table:

```
======================================================================
  CO-PRESIDENT 2026 — 2022 Backtesting Results
======================================================================

  FIRST ROUND (May 29, 2022)
  +-------------------+----------+----------+-----------+------------------+
  | Candidate         | Predicted| Actual   | Error     | 95% CI            |
  +-------------------+----------+----------+-----------+------------------+
  | Gustavo Petro     |   41.2%  |  40.34%  |  +0.86pp  | [37.1%, 45.3%]    |
  | Rodolfo Hernández |   27.8%  |  28.15%  |  -0.35pp  | [23.4%, 32.1%]    |
  | Federico Gutiérrez|   22.5%  |  23.89%  |  -1.39pp  | [18.2%, 26.8%]    |
  | Sergio Fajardo    |    5.1%  |   4.39%  |  +0.71pp  | [ 3.2%,  7.0%]    |
  | Ingrid Betancourt |    0.8%  |   0.40%  |  +0.40pp  | [ 0.3%,  1.3%]    |
  +-------------------+----------+----------+-----------+------------------+
  MAE: 0.74pp  |  RMSE: 0.89pp  |  Brier: 0.008
  Probability of runoff: 99.2%

  RUNOFF (June 19, 2022) — Petro vs. Hernández
  Gustavo Petro:    51.8% [48.2%, 55.4%]  ->  Win probability: 78.3%
  Rodolfo Hernández: 48.2% [44.6%, 51.8%]  ->  Win probability: 21.7%
  MAE: 1.36pp  |  Expected margin: +3.6pp Petro
======================================================================
```

### Commands reference

| Command | Description |
|---|---|
| `run` | Full pipeline: load → clean → fit → validate → print table |
| `run --no-sample` | Same but skip MCMC; use baseline weighted averages |
| `aggregate` | Baseline weighted polling average only |
| `validate` | Re-run validation on saved model output |
| `plot` | Generate all figures (forecast evolution, calibration, error) |
| `ingest` | Ingest fundamental data components (default: sabaneta; use `--component` for others) |
| `forecast` | 2026 forecast mode with `--mode off / prior_only / joint` |
| `config` | Print current configuration (candidates, dates, pollster ratings) |
| `--config-override` | Override config keys from the command line (`KEY=VALUE`) |

---

## Project structure

```
co-president-2026/
├── pyproject.toml                  # Project metadata, dependencies, tool config
├── uv.lock                         # Lockfile (generated by uv)
├── CHANGELOG.md                    # Milestone-based changelog
├── CONTRIBUTING.md                 # Contribution guide
├── AGENTS.md                       # AI agent operating manual
├── MVP_SPECS_GUIDE.md              # MVP technical specification (SPEC-01→11)
├── ENHANCEMENT_ROADMAP.md          # Post-MVP enhancement plan (SPEC-12→41)
├── src/
│   └── co_president/
│       ├── __init__.py             # Package version
│       ├── __main__.py             # CLI entry point (8 subcommands)
│       ├── config.py               # Candidate maps, dates, pollster ratings, hyperparameters
│       ├── data.py                 # Re-export facade for data_polls + data_results
│       ├── data_polls.py           # Poll loading, cleaning, normalization (SPEC-04)
│       ├── data_results.py         # Results consolidation (SPEC-03)
│       ├── aggregation.py          # Baseline weighted polling averages (SPEC-05)
│       ├── model_round1.py         # Dirichlet-Multinomial first-round model (SPEC-06)
│       ├── model_runoff_simple.py  # K=3 Dirichlet runoff model (SPEC-07)
│       ├── model_runoff_matrix.py  # Probabilistic pairing matrix (SPEC-08)
│       ├── model_municipal.py      # 3-layer hierarchical municipal model (SPEC-22)
│       ├── model_transfer.py       # Transfer-rate estimation model (SPEC-30)
│       ├── model_utils.py          # Shared model utilities
│       ├── validation.py           # Backtesting, calibration, rolling forecasts (SPEC-09)
│       ├── validation/             # Extended validation modules
│       │   ├── __init__.py
│       │   ├── compositional.py    # Compositional data validation (SPEC-37)
│       │   └── municipal_oos.py    # OOS validation framework (SPEC-26)
│       ├── plotting.py             # Visualization (SPEC-09)
│       ├── _data_quality.py        # Data quality diagnostics (SPEC-11)
│       ├── paths.py                # Data directory resolution (SPEC-01)
│       ├── fundamentals/           # Municipal features API (SPEC-21)
│       │   ├── __init__.py
│       │   └── features.py         # MunicipalFeatures dataclass, loaders, CLR/logit
│       ├── benchmarks/             # ML benchmark suite (SPEC-41)
│       │   ├── __init__.py
│       │   ├── transforms.py       # Compositional transforms (SPEC-37)
│       │   ├── fnn_clr.py          # FNN+CLR model (SPEC-41)
│       │   ├── runner.py           # Benchmark runner (SPEC-41a-c)
│       │   └── terridata.py        # TerriData feature loader
│       └── ingestion/              # Post-MVP data ingestion pipeline
│           ├── __init__.py         # Re-exports all ingestion symbols
│           ├── ingest_divipola.py  # DIVIPOLA master registry (SPEC-12.1)
│           ├── ingest_historical.py# Historical results (SPEC-12.2)
│           ├── ingest_cnpv.py      # CNPV 2018 census (SPEC-17)
│           ├── ingest_nbi.py       # NBI poverty indicators (SPEC-18)
│           ├── ingest_ipm.py       # IPM poverty indicators (SPEC-18)
│           ├── ingest_population.py# Population projections (SPEC-19)
│           ├── ingest_risk.py      # Electoral risk indicators (SPEC-13.2/20)
│           ├── ingest_socioeconomic.py # Socioeconomic demographics (SPEC-13a)
│           ├── ingest_fiscal.py    # Fiscal autonomy (SPEC-21a)
│           ├── ingest_bogota.py    # Bogotá localidad disaggregation (SPEC-21b)
│           ├── ingest_sabaneta.py  # Sabaneta fixture loader (SPEC-16)
│           ├── ingest_ecp.py       # ECP cultural attitudinal signal (SPEC-24)
│           ├── ingest_trends.py    # Google Trends ingestion (SPEC-38)
│           ├── trends_keywords.py  # Google Trends keyword definitions
│           ├── twitter_preprocess.py # Twitter data preprocessing (SPEC-39)
│           ├── twitter_denoise.py  # Twitter spam/noise filtering (SPEC-39)
│           ├── bert_sentiment.py   # BERT sentiment classification (SPEC-39)
│           ├── llm_sentiment.py    # LLM sentiment classification (SPEC-39)
│           ├── sentiment_series.py # Sentiment time-series builder (SPEC-39)
│           ├── build_feature_matrix.py # Feature matrix builder (SPEC-14)
│           ├── report.py           # Coverage report generator (SPEC-15)
│           └── _download_cedae.py  # CEDAE bulk downloader
├── data/
│   ├── 2022-polls/                 # encuestas_2022.csv, consultas.csv, as_coa/
│   ├── 2022-presidential-results/  # Registraduría + MOE files
│   └── fundamentals/              # Built-in fundamental component matrices
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Shared pytest fixtures (data_dir, sabaneta_fixture)
│   ├── test_paths.py               # SPEC-01
│   ├── test_config.py              # SPEC-02
│   ├── test_data_results.py        # SPEC-03
│   ├── test_data_polls.py          # SPEC-04
│   ├── test__data_quality.py       # SPEC-11
│   ├── test_aggregation.py         # SPEC-05
│   ├── test_model.py               # SPEC-06/07/08 — all 3 Bayesian models
│   ├── test_model_municipal.py     # SPEC-22
│   ├── test_validation.py          # SPEC-09
│   ├── test_validation_municipal_oos.py   # SPEC-26 — OOS validation framework
│   ├── test_compositional.py       # SPEC-37 — compositional data transforms
│   ├── test_validation_compositional.py   # SPEC-37 — compositional validation
│   ├── test_fundamentals_features.py      # SPEC-21 — municipal features
│   ├── test_cli.py / test___main___forecast.py  # SPEC-10 / SPEC-28 — CLI
│   ├── test_bert_sentiment.py / test_llm_sentiment.py  # SPEC-39 — sentiment
│   ├── test_twitter_preprocess.py / test_twitter_denoise.py  # SPEC-39 — twitter
│   ├── test_sentiment_series.py   # SPEC-39 — sentiment time series
│   ├── test_benchmarks_fnn_clr.py / test_benchmarks_runner.py  # SPEC-41 — benchmarks
│   ├── test_benchmarks_transforms.py / test_benchmark_compositional.py  # SPEC-37/41
│   └── test_ingestion_*.py         # 16 files, one per ingestion module
├── docs/
│   ├── README.md                   # Documentation index
│   ├── specs/STATUS.md             # Full SPEC status tracker (01→41)
│   ├── PLAN_demographic_ingestion.md # 12-day demographic plan (SPEC-15→30)
│   ├── PLAN_cne_2026_ingestion.md   # CNE 2026 ingestion plan (SPEC-31→36)
│   └── archive/                    # Superseded planning documents
├── scripts/                        # Utility scripts (CNPV downloader, 100K report)
└── notebooks/                      # Untracked; manual EDA
```

Tests mirror the source tree one-to-one: `src/co_president/config.py` → `tests/test_config.py`, etc.

---

## Development

### Getting started

```bash
git clone https://github.com/<your-org>/co-president-2026
cd co-president-2026
make setup

# Quick check that everything is wired correctly
uv run ruff check src/ tests/
uv run pyright src/
uv run pytest tests/ -v
```

### TDD workflow

This project follows strict TDD. Each spec is implemented in this order:

```
  Step 1: Write failing tests that define the contract (RED)
  Step 2: Write the minimum code to make them pass (GREEN)
  Step 3: Refactor — extract helpers, improve names (REFACTOR)
  Step 4: Type-check — pyright src/ must pass (TYPE-CHECK)
  Step 5: Lint — ruff check src/ tests/ must pass (LINT)
  Step 6: Commit — only when all three gates pass
```

### Makefile targets

| Target | Description |
|---|---|
| `make check` | Runs fmt + lint + typecheck + test in sequence. Run before every commit. |
| `make fmt` | Auto-format all source and test files with `ruff format`. |
| `make lint` | Lint check with `ruff check` (all rules). |
| `make typecheck` | Static type check with `pyright` (strict mode). |
| `make test` | Run the full test suite with verbose output. |
| `make test-fast` | Run only unit tests (skip slow MCMC model tests). |
| `make test-model` | Run fast model tests only (graph checks, prior predictive). |
| `make test-model-slow` | Run slow model tests only (convergence, sanity checks). |
| `make dev` | Quick smoke test: run the pipeline without MCMC sampling. |
| `make sync` | Install or update all dependencies (`uv sync`). |
| `make clean` | Remove `__pycache__`, `.pyc` files, `results/`, and `*.nc` artifacts. |

### Quality gates

These three commands must all return zero errors before every commit (`make check` runs them all):

```bash
uv run ruff check src/ tests/       # zero lint errors
uv run pyright src/                 # zero type errors
uv run pytest tests/ -v            # all tests pass
```

---

## Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run a specific test file
uv run pytest tests/test_data_polls.py -v

# Run MCMC-related model tests (slow; may take several minutes)
uv run pytest tests/test_model.py -v

# With full output (no pytest capture)
uv run pytest tests/ -v --capture=no
```

**MCMC test strategy.** Because posterior draws are stochastic, model tests use a multi-phase approach:
- **Graph test:** model builds without error; expected number of free RVs and observed variables are checked.
- **Prior predictive test:** `pm.sample_prior_predictive()` runs; predicted shares are in [0, 1].
- **Convergence test** (slow): fit on minimal synthetic data; R-hat < 1.10 for all parameters.
- **Sanity test:** 3 artificial polls with known ground truth; posterior mean within 5pp of truth.

---

## Specs roadmap

### MVP: National polling model (shipped)
- [x] **SPEC-01** — Project scaffolding (pyproject.toml, uv, linting, typing)
- [x] **SPEC-02** — Configuration and constants (candidate maps, pollster ratings, hyperparameters)
- [x] **SPEC-03** — Election results consolidation (Registraduría + MOE cross-validation)
- [x] **SPEC-04** — Poll data loading and cleaning (date fixes, undecided redistribution, deduplication)
- [x] **SPEC-05** — Baseline weighted polling average (benchmark)
- [x] **SPEC-06** — Bayesian first-round model (Dirichlet-Multinomial + reverse-time random walk)
- [x] **SPEC-07** — Runoff model (simple top-2, K=3)
- [x] **SPEC-08** — Runoff probability matrix (all pairings, transfer heuristic)
- [x] **SPEC-09** — Validation metrics and visualization (MAE, RMSE, calibration, rolling forecasts)
- [x] **SPEC-10** — CLI entry point and integration
- [x] **SPEC-11** — Data quality diagnostics

### Post-MVP: Full feature suite (SPEC-12→41 shipped, 30 in progress, 31→36 planned)
See [`docs/specs/STATUS.md`](docs/specs/STATUS.md) for the complete status of every SPEC (01–41).

**MVP success criterion:** achieved — the model predicts 2022 results within ±5 percentage points of actual vote shares using only national poll data.

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full contribution guide. Key points:

1. Create feature branches off `dev` as `feat/spec-XX-description`.
2. Follow the TDD cycle (RED → GREEN → REFACTOR).
3. Run `make check` before every commit (fmt + lint + typecheck + test).
4. Run CodeRabbit (`cr`) before every commit — never commit without a passing review.
5. Use semantic commit format: `feat(SPEC-XX):`, `test(SPEC-XX):`, `fix(SPEC-XX):`, or `chore:` only.

---

## License

Dual-licensed under the terms below:

- **Source code** — [MIT](./LICENSE)
- **Outputs, graphs, and documentation** — [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)

Copyright (c) 2026 Samuel Gomez Guio.
