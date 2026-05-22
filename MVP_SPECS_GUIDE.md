# co-president-2026 — Technical Specification & Implementation Guide

**Project**: Colombian Presidential Elections 2026 — Polling-Based Bayesian Forecast  
**MVP Scope**: Predict 2022 election results from national polls only, no demographic/regional data  
**Language**: Python 3.12  
**Last Updated**: 2026-05-13

---

## 1. Overview

`co-president-2026` is a Python package that:

1. **Ingests** Colombian presidential poll data (CSV files from multiple pollsters)
2. **Cleans & normalizes** poll data (undecided voter redistribution, date corrections, deduplication)
3. **Consolidates** official election results from two independent sources (Registraduría Nacional and MOE) into a single source of truth
4. **Fits** a Bayesian Dirichlet-Multinomial model with a **reverse-time random walk** for latent vote intentions, hierarchical pollster house effects, and time-varying uncertainty
5. **Validates** predictions against 2022 actual results via backtesting
6. **Produces** probabilistic forecasts for both the first round (multi-candidate) and runoff (top-2 head-to-head)

### MVP End State

By the end of this MVP, the model must predict the 2022 Colombian presidential election results within ±5% of the actual vote shares, given only the poll data available up to election day. Once validated, the same pipeline can be adapted for 2026 by swapping the data source and updating candidate configurations.

### Data Inventory

| File | Source | Rows | Columns | Granularity | Encoding |
|------|--------|------|---------|-------------|----------|
| `encuestas_2022.csv` | recetas-electorales.com (Nelson Amaya) | 46 | 27 | National polls | UTF-8, comma-delimited |
| `consultas.csv` | recetas-electorales.com | 65 | 7 | Coalition internal polls | ISO-8859-1, comma-delimited |
| `MMV_NACIONAL_PRESIDENTE_2022_1v.csv` | Registraduría Nacional | 727,510 | 16 | Polling-station-level | ISO-8859-1, semicolon-delimited |
| `MMV_NACIONAL_PRESIDENTE_2022_2v.csv` | Registraduría Nacional | 398,387 | 16 | Polling-station-level | ISO-8859-1, semicolon-delimited |
| `moe_vuelta1.csv` | MOE (Misión de Observación Electoral) | 11,864 | 15 | Municipal-level | UTF-8, comma-delimited |
| `moe_vuelta2.csv` | MOE | 5,550 | 15 | Municipal-level | UTF-8, comma-delimited |
| `reg_participacion_vuelta1.csv` | Registraduría Nacional | ~33,000 | 17 | Polling-station-level | UTF-8-BOM, comma-delimited |
| `reg_participacion_vuelta2.csv` | Registraduría Nacional | ~33,000 | 17 | Polling-station-level | UTF-8-BOM, comma-delimited |

**Key insight**: The poll data is entirely national-level. The results data is municipal/station-level and can be aggregated to national totals. No departmental-level poll data exists. The MVP operates at the **national level only**.

---

## 2. Documentation & Package References

All packages used in this project are pinned to compatible versions with Python 3.12. Below are links to the up-to-date documentation for each dependency.

### 2.1 Runtime Dependencies

| Package | Version | Documentation | Purpose |
|---|---|---|---|
| **Python** | 3.12.x | [docs.python.org/3.12](https://docs.python.org/3.12/) | Core language |
| **pandas** | ≥2.2 | [pandas.pydata.org/docs](https://pandas.pydata.org/docs/) | Data loading, cleaning, pivoting, grouping |
| **numpy** | ≥1.26 | [numpy.org/doc/stable](https://numpy.org/doc/stable/) | Numerical arrays, random number generation, log/exp ops |
| **PyMC** | ≥5.15 | [pymc.io](https://www.pymc.io/welcome.html) | Bayesian modeling, MCMC sampling via NUTS |
| **ArviZ** | ≥0.18 | [python.arviz.org](https://python.arviz.org/) | MCMC diagnostics (`r_hat`, `ess`, `summary`), posterior analysis, trace plotting |
| **matplotlib** | ≥3.8 | [matplotlib.org/stable](https://matplotlib.org/stable/) | Visualization (evolution plots, calibration plots) |
| **pyarrow** | ≥15.0 | [arrow.apache.org/docs/python](https://arrow.apache.org/docs/python/) | Fast CSV/parquet I/O backend for pandas |

### 2.2 Dev Dependencies

| Package | Version | Documentation | Purpose |
|---|---|---|---|
| **pytest** | ≥8.0 | [docs.pytest.org](https://docs.pytest.org/) | Test framework — fixtures, parametrization, conftest |
| **pytest-cov** | ≥5.0 | [pytest-cov.readthedocs.io](https://pytest-cov.readthedocs.io/) | Coverage reporting for pytest |
| **ruff** | ≥0.4 | [docs.astral.sh/ruff](https://docs.astral.sh/ruff/) | Linting (all rules) + formatting + isort |
| **pyright** | ≥1.1 | [microsoft.github.io/pyright](https://microsoft.github.io/pyright/) | Static type checking (strict mode) |
| **pandas-stubs** | ≥2.2 | [pypi.org/project/pandas-stubs](https://pypi.org/project/pandas-stubs/) | Type stubs for pandas |
| **pre-commit** | ≥3.8 | [pre-commit.com](https://pre-commit.com/) | Git hook framework |
| **pip-audit** | ≥2.7 | [pypi.org/project/pip-audit](https://pypi.org/project/pip-audit/) | Vulnerability scanning for Python packages |
| **bandit** | ≥1.7 | [bandit.readthedocs.io](https://bandit.readthedocs.io/) | Static security analysis |

### 2.3 Tooling References

| Tool | Documentation | Purpose |
|---|---|---|
| **uv** | [docs.astral.sh/uv](https://docs.astral.sh/uv/) | Python package/project manager, virtual env creation, lockfile generation |
| **pyproject.toml** | [packaging.python.org/en/latest/guides/writing-pyproject-toml](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/) | Project metadata standard (PEP 621) |
| **Stan Reference Manual** | [mc-stan.org/docs/reference-manual](https://mc-stan.org/docs/reference-manual/) | Reference for MCMC algorithms, diagnostics, model block syntax (conceptual reference; PyMC uses its own NUTS implementation) |
| **Dirichlet-Multinomial** | [Wikipedia](https://en.wikipedia.org/wiki/Dirichlet-multinomial_distribution) | Mathematical foundation of the core observation and election models |

### 2.4 Key PyMC API References

These specific PyMC classes and functions will be used extensively in SPEC-06 through SPEC-08:

| API | Reference | Role in Our Model |
|---|---|---|
| `pm.Model` | [link](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.Model.html) | Model container (context manager) |
| `pm.Normal` | [link](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.Normal.html) | Prior for latent vote intention vector `θ[t]` on logit scale; prior for raw house effects |
| `pm.HalfNormal` | [link](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.HalfNormal.html) | Prior for standard deviations `σ_rw` (random walk step size), `σ_house` (house effect spread) |
| `pm.DirichletMultinomial` | [link](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.DirichletMultinomial.html) | Observation model for poll vote shares; election likelihood |
| `pm.Gamma` | [link](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.Gamma.html) | Prior for Dirichlet concentration parameters (separate for polls vs. election results) |
| `pm.Deterministic` | [link](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.Deterministic.html) | Zero-sum constrained house effects, softmax-transformed probabilities |
| `pm.sample` | [link](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample.html) | NUTS MCMC sampler (draws, tune, chains, cores, seed, target_accept) |
| `pm.Data` | [link](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.Data.html) | Mutable data containers enabling posterior predictive with different inputs |
| `pm.sample_prior_predictive` | [link](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample_prior_predictive.html) | Prior predictive sampling (model validation before fitting) |
| `pm.sample_posterior_predictive` | [link](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample_posterior_predictive.html) | Posterior predictive sampling (generate mock election outcomes) |
| `pm.math.softmax` | [link](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.math.softmax.html) | Logit-to-probability-simplex transformation: `softmax(θ) = exp(θ) / Σexp(θ)` |
| `az.summary` | [link](https://python.arviz.org/en/stable/api/generated/arviz.summary.html) | MCMC diagnostics table (r_hat, ess_bulk, ess_tail, mean, sd, hdi) |
| `az.plot_trace` | [link](https://python.arviz.org/en/stable/api/generated/arviz.plot_trace.html) | Trace plots for visual convergence diagnostics |

### 2.5 Project Structure Expected After SPEC-01

```
co-president-2026/
├── pyproject.toml                  # Project config, deps, ruff, pyright
├── uv.lock                         # Lockfile (generated by uv)
├── .python-version                 # "3.12"
├── .gitignore                      # Excludes .venv, __pycache__, notebooks, results/, *.nc, .Rproj.*, .Rhistory, .DS_Store
├── MVP_SPECS_GUIDE.md               # This document
├── src/
│   └── co_president/
│       ├── __init__.py
│       ├── config.py               # SPEC-02: Candidate maps, dates, pollster ratings, model hyperparameters
│       ├── data.py                  # SPEC-03+04: Results consolidation + poll loading/cleaning
│       ├── aggregation.py          # SPEC-05: Baseline weighted polling averages
│       ├── model_round1.py         # SPEC-06: Dirichlet-Multinomial + reverse-time RW (1st round)
│       ├── model_runoff_simple.py  # SPEC-07: Dirichlet-Multinomial(K=3) runoff model
│       ├── model_runoff_matrix.py  # SPEC-08: Full probabilistic pairing matrix
│       ├── validation.py           # SPEC-09: Backtesting, calibration, rolling forecast
│       ├── plotting.py             # SPEC-09: Visualization functions
│       └── __main__.py             # SPEC-10: CLI entry point
├── data/
│   ├── 2022-polls/
│   │   ├── encuestas_2022.csv
│   │   └── consultas.csv
│   └── 2022-presidential-results/
│       ├── MMV_NACIONAL_PRESIDENTE_2022_1v.csv
│       ├── MMV_NACIONAL_PRESIDENTE_2022_2v.csv
│       ├── moe_vuelta1.csv
│       ├── moe_vuelta2.csv
│       ├── reg_participacion_vuelta1.csv
│       └── reg_participacion_vuelta2.csv
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_data.py
│   ├── test_aggregation.py
│   ├── test_model.py
│   └── test_validation.py
└── notebooks/
    └── exploration.ipynb           # Untracked; manual EDA and ad-hoc analysis
```

---

## 3. Development Philosophy

### 3.1 Test-Driven Development (TDD)

This project follows a strict TDD discipline. Every spec MUST be implemented in this exact order:

```
┌──────────────────────────────────────────────────────────────────────┐
│  1. RED                                                              │
│  Write a failing test that defines the expected behavior.           │
│  The test IS the contract: inputs, outputs, edge cases, error       │
│  conditions. No implementation code exists yet.                      │
│                                                                      │
│  2. GREEN                                                            │
│  Write the MINIMUM code to make the test pass. No over-engineering. │
│  No code path without a corresponding test. Use the simplest        │
│  possible implementation that satisfies the test.                   │
│                                                                      │
│  3. REFACTOR                                                         │
│  Clean up: extract helpers, improve names, reduce duplication,      │
│  add docstrings. Keep ALL tests green throughout.                   │
│                                                                      │
│  4. TYPE-CHECK                                                       │
│  Run `pyright src/`. Fix ALL type errors. Every function parameter  │
│  and return type MUST be annotated. Every class attribute MUST be   │
│  annotated. No `Any` unless mathematically justified.               │
│                                                                      │
│  5. LINT                                                             │
│  Run `ruff check src/ tests/`. Fix ALL lint errors. The only        │
│  allowed suppression is `# noqa: S101` for test assertions.         │
│                                                                      │
│  6. COMMIT                                                           │
│  Commit only when all three gates pass:                             │
│    $ ruff check src/ tests/        # zero errors                    │
│    $ pyright src/                   # zero type errors              │
│    $ pytest tests/ -v               # all tests pass                │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.2 Modus Operandi (M.O.)

1. **Specs are the source of truth.** Each SPEC-XX defines what must happen. If a behavior is not in the spec, it is OUT OF SCOPE until explicitly added. No speculative features.

2. **Tests before code, always.** No line of implementation code is written before a test exists that fails because the code is not yet written. This applies to:
   - Utility functions (data cleaning, normalization, weighting)
   - Statistical functions (weighted averages, time decay, undecided redistribution)
   - Model construction (the PyMC model graph must be testable — verify graph structure, not just posterior)
   - Validation metrics (MAE, RMSE, Brier score — test with known inputs/outputs)
   - Visualization code (at minimum: assert figure object is returned, no exceptions)
   - CLI entry points (test subprocess exit codes and stdout content)

3. **One spec at a time, sequentially.** SPEC-03 does not begin until SPEC-02 is fully tested and green. Specs that depend on earlier specs inherit all prior tests and MUST keep them green.

4. **Tests mirror `src/` one-to-one:**
   ```
   src/co_president/config.py          →  tests/test_config.py
   src/co_president/data.py            →  tests/test_data.py
   src/co_president/aggregation.py     →  tests/test_aggregation.py
   src/co_president/model_round1.py    →  tests/test_model.py
   src/co_president/model_runoff_simple.py  →  tests/test_model.py
   src/co_president/model_runoff_matrix.py  →  tests/test_model.py
   src/co_president/validation.py      →  tests/test_validation.py
   src/co_president/plotting.py        →  tests/test_validation.py
   src/co_president/__main__.py        →  tests/test_cli.py (created in SPEC-10)
   ```

5. **Test data is deterministic and minimal.** Use small, hardcoded DataFrames (3-5 rows) for unit tests. Do NOT load the full CSV files in unit tests. The full dataset is only used in:
   - Integration-level validation (SPEC-09)
   - CLI commands (SPEC-10)
   - The exploration notebook (not tracked, not tested)

6. **MCMC tests have special rules.** Bayesian models cannot be tested with exact numeric assertions because posterior draws are stochastic. MCMC test strategy:
   - **Graph test**: Call `pm.Model()` and verify the model builds without error. Assert expected number of free parameters (`RVs`), observed variables (`observed_RVs`), and deterministic transforms (`deterministics`).
   - **Prior predictive test**: Call `pm.sample_prior_predictive()`. Assert prior predictive samples fall within mathematically valid ranges (e.g., vote shares on [0, 1]).
   - **Convergence test** (slow; runs in CI only): Fit on a minimal dataset (3 polls, 2 candidates). Assert R-hat < 1.10 for all parameters.
   - **Small-data sanity test**: Construct 3 artificial polls with known ground truth (e.g., Candidate A at 60%, Candidate B at 40%). Fit the model. Assert the posterior mean is within 5 percentage points of the ground truth.

7. **No ad-hoc scripting in the codebase.** All exploration happens in `notebooks/exploration.ipynb` (gitignored). Any logic that proves useful during exploration MUST be extracted into a tested function in `src/co_president/`.

8. **Type safety gate.** The following three commands MUST all return zero errors before every commit:
   ```bash
   ruff check src/ tests/       # Must return exit code 0
   pyright src/                  # Must return exit code 0
   pytest tests/ -v              # Must return exit code 0 (all tests pass)
   ```

---

## 4. SPEC-01: Project Scaffolding

**Goal**: Set up the Python 3.12 project with strict typing, linting, and dependency management.

### 4.1 Requirements

1. **Python version**: 3.12.x, managed via `uv` and declared in `.python-version`.

2. **`pyproject.toml`** with the following sections:

   **`[project]` block**:
   ```toml
   [project]
   name = "co-president"
   version = "0.1.0"
   description = "Bayesian presidential election forecast for Colombia using polling data"
    readme = "MVP_SPECS_GUIDE.md"
   requires-python = ">=3.12"
   license = {text = "MIT"}
   dependencies = [
       "pandas>=2.2",
       "numpy>=1.26",
       "pymc>=5.15",
       "arviz>=0.18",
       "matplotlib>=3.8",
       "pyarrow>=15.0",
   ]

    [project.optional-dependencies]
    dev = [
        "pytest>=8.0",
        "pytest-cov>=5.0",
        "ruff>=0.4",
        "pyright>=1.1",
        "pandas-stubs>=2.2",
        "pre-commit>=3.8",
        "pip-audit>=2.7",
        "bandit>=1.7",
    ]
   ```

   **`[tool.ruff]` block**:
   ```toml
   [tool.ruff]
   target-version = "py312"
   line-length = 100
   src = ["src", "tests"]
   extend-exclude = ["notebooks", ".venv"]

   [tool.ruff.lint]
    select = ["ALL"]
    ignore = [
        "D100",     # Missing module docstring — module layout and usage are documented in MVP_SPECS_GUIDE; only functions/classes get docstrings
        "D104",     # Missing package docstring — __init__.py files are minimal stubs
        "D203",     # Conflicts with D211 (Google-style blank line before class)
        "D213",     # Conflicts with D212 (Google-style summary on first line)
        "COM812",   # Conflicts with ruff format trailing comma handling; intentional with isort force-sort-within-sections
    ]

    [tool.ruff.lint.per-file-ignores]
    "tests/**/*.py" = ["S101", "PLR2004"]

    [tool.ruff.format]
    quote-style = "double"
    indent-style = "space"
    skip-magic-trailing-comma = false
    line-ending = "lf"

    [tool.ruff.lint.isort]
    known-first-party = ["co_president"]
    force-sort-within-sections = true
   ```

   **`[tool.pyright]` block**:
   ```toml
   [tool.pyright]
   pythonVersion = "3.12"
   typeCheckingMode = "strict"
   reportMissingTypeStubs = true
   reportMissingImports = true
   reportUnusedImport = true
   include = ["src", "tests"]
   venvPath = "."
   venv = ".venv"
   ```

 3. **`.gitignore`**: Must exclude `.venv/`, `__pycache__/`, `*.pyc`, `*.Rproj*`, `.Rhistory`, `.RData`, `.DS_Store`, `notebooks/`, `results/` (model output directory), `*.nc` (ArviZ netCDF files).

4. **`uv.lock`**: Generated by `uv lock` or `uv sync`. Committed to version control.

 5. **Source layout**: `src/co_president/` with an `__init__.py` that exports `__version__ = "0.1.0"` and imports nothing else by default.

6. **Tests layout**: `tests/` with `__init__.py` and `conftest.py` containing shared pytest fixtures (the minimal fixture is a path to the data directory).

### 4.2 Acceptance Criteria

- `uv sync` installs all dependencies without error
- `uv run ruff check src/ tests/` passes with zero errors
- `uv run pyright src/` passes with zero errors
- `uv run pytest tests/` runs (even if all tests are skipped, but at minimum `conftest.py` must exist and be valid)
- `uv run python -c "import co_president; print(co_president.__version__)"` prints `0.1.0`
- `.gitignore` contains all required entries

### 4.3 TDD Steps

1. **Red**: Write `tests/test_config.py` with a single placeholder test `def test_project_exists() -> None: assert True`. It should pass, establishing the test infrastructure.
2. **Green**: Run `uv sync`. Create `pyproject.toml`, `.python-version`, `.gitignore`, `src/co_president/__init__.py`, `tests/__init__.py`, `tests/conftest.py`.
3. **Type-check**: `pyright src/` must pass.
4. **Lint**: `ruff check` must pass.
5. **Commit**.

---

## 5. SPEC-02: Configuration & Constants

**Goal**: Define all configuration constants — candidate mappings, date thresholds, pollster ratings, and actual election results — in a single, fully typed module. Every downstream module imports from `co_president.config`, never hardcodes these values.

### 5.1 Requirements

#### 5.1.1 Candidate Dataclass

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Candidate:
    """A presidential candidate across the election cycle."""
    key: str                     # Internal key matching column names in polls
    display_name: str            # Human-readable name
    coalition: str | None        # Coalition the candidate runs under (e.g., "Pacto Histórico")
    first_round: bool = True     # Whether the candidate ran in the first round
    runoff: bool = False         # Whether the candidate made the runoff
```

#### 5.1.2 Model Hyperparameters Dataclass

```python
    @dataclass(frozen=True)
    class ModelConfig:
        """Hyperparameters for the Bayesian election model."""
        random_walk_sigma_prior: float = 0.5         # HalfNormal sigma for RW step size
        concentration_poll_prior_mean: float = 5.0   # Gamma mean for poll concentration
        concentration_election_prior_mean: float = 50.0  # Gamma mean for election concentration
        house_effect_sigma_prior: float = 1.0        # HalfNormal sigma for house effects
        mcmc_draws: int = 4000                       # Number of posterior draws per chain
        mcmc_tune: int = 1000                        # Number of tuning/warmup draws per chain
        mcmc_chains: int = 4                         # Number of chains
        mcmc_cores: int = 4                          # Number of CPU cores for parallel chains
        target_accept: float = 0.95                  # NUTS target acceptance rate
        seed: int = 332211                           # RNG seed for reproducibility
        time_decay_half_life_days: float = 30.0      # Days for poll weight to halve
        consultation_prior_strength: float = 0.5   # Sigma for Normal prior on theta[T-1] derived from consultation results
```

#### 5.1.3 Candidate Registry

`FIRST_ROUND_CANDIDATES: dict[str, Candidate]` mapping CSV column names in `encuestas_2022.csv` to `Candidate` objects:

| CSV Column | Candidate | Coalition | 1st Round | Runoff |
|---|---|---|---|---|
| `gustavo_petro` | Gustavo Petro | Pacto Histórico | Yes | Yes |
| `federico_gutierrez` | Federico Gutiérrez | Equipo por Colombia | Yes | No |
| `rodolfo_hernandez` | Rodolfo Hernández | Liga de Gobernantes Anticorrupción | Yes | Yes |
| `sergio_fajardo` | Sergio Fajardo | Centro Esperanza | Yes | No |
| `ingrid_betancourt` | Ingrid Betancourt | (Independent, formerly Centro Esperanza) | Yes | No |

Plus sentinel entries for `rest` (aggregate of "otros" + all minor candidates) and `blanco`.

#### 5.1.4 Coalition-to-Candidate Mapping

`COALITION_TO_CANDIDATE: dict[str, str]` for translating Registraduría/MOE coalition names to candidate keys:

| Source Coalition Name | → | Candidate Key |
|---|---|---|
| `"COALICION PACTO HISTORICO"` | → | `gustavo_petro` |
| `"COALICION EQUIPO POR COLOMBIA"` | → | `federico_gutierrez` |
| `"LIGA DE GOBERNANTES ANTICORRUPCION"` | → | `rodolfo_hernandez` |
| `"COALICION CENTRO ESPERANZA"` | → | `sergio_fajardo` |
| `"VOTOS EN BLANCO"` (code 996) | → | `blanco` |
| `"VOTOS NULOS"` (code 997) | → | `nulos` (tracked but excluded from valid vote share) |
| `"VOTOS NO MARCADOS"` (code 998) | → | `no_marcados` (excluded from valid vote share) |

Minor parties map to `"rest"`: `"COLOMBIA JUSTA LIBRES"`, `"PARTIDO MOVIMIENTO DE SALVACION NACIONAL"`, `"COLOMBIA PIENSA EN GRANDE"`, `"PARTIDO VERDE OXIGENO"`.

**Note on Centro Esperanza**: Fajardo and Betancourt both ran under Centro Esperanza branding initially. At the official election, the coalition's votes counted entirely for Fajardo (Betancourt withdrew). In the polls, they are tracked separately. The mapping treats Centro Esperanza votes in Reg/MOE as Fajardo's. Betancourt individual votes are tracked in polls only (not in consolidated results separately).

#### 5.1.5 Election Dates

```python
ELECTION_DATE_ROUND1: date = date(2022, 5, 29)
ELECTION_DATE_ROUND2: date = date(2022, 6, 19)
CONSULTATION_DATE: date = date(2022, 3, 13)
```

`CONSULTATION_DATE` marks when inter-party consultation results became known. Polls before this date feature candidates who later dropped out (Gaviria, Galán, Char, Barguil, Peñalosa, Zuluaga). Polls after this date are post-consultation with the final five candidate field.

`CONSULTATION_VOTES: dict[str, int]` records the total votes cast in each coalition's March 13 consultation. These serve as an **informative prior** for the model's earliest time point (SPEC-06), grounding `theta[T-1]` in real voter behavior rather than a vague prior:

```python
CONSULTATION_VOTES: dict[str, int] = {
    "gustavo_petro": 5_500_000,          # Pacto Histórico total (5.5M votes across all candidates)
    "federico_gutierrez": 3_800_000,      # Equipo por Colombia total (3.8M)
    "sergio_fajardo": 1_800_000,          # Centro Esperanza total (1.8M)
    "rodolfo_hernandez": 0,               # No coalition consultation (independent)
    "ingrid_betancourt": 0,               # Ran in Centro Esperanza consultation but withdrew
}
```

Values are approximate (±200K) and only rough proxies for coalition base support. They are used as a **soft** prior — the model can deviate if the poll data disagrees.

`consultation_log_share_prior() -> dict[str, float]` computes `log(share)` for each candidate with non-zero consultation votes, where `share = votes / sum(votes)`. For candidates with zero consultation votes, returns the log of the minimum non-zero share divided by 2 (a small placeholder). This ensures every candidate has a prior while respecting the known consultation signal.

#### 5.1.6 Pollster Ratings

`POLLSTER_RATINGS: dict[str, float]` — La Silla Vacía pollster quality ratings (0–10 scale):

```python
POLLSTER_RATINGS: dict[str, float] = {
    "Invamer": 10.00,
    "CNC": 8.10,
    "GAD3": 8.10,
    "Guarumo": 8.00,
    "TYSE": 6.30,
    "CELAG": 5.90,
    "AtlasIntel": 5.40,
    "MassiveCaller": 5.10,
    "Medilab": 3.80,
    "YanHaas": 3.60,
    "CifrasYConceptos": 3.60,
    "Datexco": 3.23,
    "Mosqueteros": 1.00,
}
```

`POLLSTER_WEIGHT_FORMULA`: `weight = rating × 0.02 + 0.8`. This constrains weights to ±20% of 1.0. For unrated pollsters, use the median of all known ratings as the default.

#### 5.1.7 Actual Election Results (Lazy-Loaded)

`load_actual_results() -> tuple[RoundResult, RoundResult]` is a lazy-loading function defined in `data.py` (SPEC-03). Downstream models call this function at runtime — `config.py` does NOT import from `data.py`. The function returns both rounds' `RoundResult` objects populated from consolidated election results. This breaks the circular dependency between `config.py` and `data.py`.

#### 5.1.8 Active Candidate Helpers

```python
def get_active_candidates(round_number: Literal[1, 2]) -> list[Candidate]: ...
```
Returns the list of `Candidate` objects active in the given round.

```python
def get_candidate_column_map() -> dict[str, str]: ...
```
Returns mapping from CSV column name (`gustavo_petro`) to candidate key (`gustavo_petro`). Used by the poll data loader to identify which columns to extract.

#### 5.1.9 Transfer Heuristic Constants (SPEC-08)

These are placeholder values for the runoff probability matrix transfer heuristic (SPEC-08). They MUST be refined with actual legislative election data before the 2026 cycle.

```python
TRANSFER_FAJARDO_PETRO: float = 0.60       # Fraction of Fajardo voters going to Petro
TRANSFER_FAJARDO_HERNANDEZ: float = 0.40   # Fraction going to Hernández (1 - above)
TRANSFER_GUTIERREZ_HERNANDEZ: float = 0.70 # Fraction of Gutiérrez voters going to Hernández
TRANSFER_GUTIERREZ_PETRO: float = 0.30     # Fraction going to Petro (1 - above)
TRANSFER_BLANCO_SPLIT: float = 0.5   # Fraction of blank votes going to candidate A (Petro by convention); 1-split goes to B

# Mapping from consultation candidate display names to canonical keys
CONSULTATION_KEY_MAP: dict[str, str] = {
    "Gustavo Petro": "gustavo_petro",
    "Federico Gutiérrez": "federico_gutierrez",
    "Sergio Fajardo": "sergio_fajardo",
    "Ingrid Betancourt": "ingrid_betancourt",
    "Rodolfo Hernández": "rodolfo_hernandez",
    # ... additional mappings populated as needed from consultas.csv
}
```

### 5.2 Acceptance Criteria

- All constants are typed with explicit type annotations on `dict`, `Candidate`, `ModelConfig`, function signatures
- `pyright src/co_president/config.py` passes with zero errors
- `ruff check src/co_president/config.py` passes
- `get_active_candidates(1)` returns 5 candidates + `rest` + `blanco`
- `get_active_candidates(2)` returns 2 candidates + `blanco`
- `POLLSTER_WEIGHT_FORMULA(10.0) == 1.0` and `POLLSTER_WEIGHT_FORMULA(0.0) == 0.8`
- `COALITION_TO_CANDIDATE["COALICION PACTO HISTORICO"] == "gustavo_petro"`

### 5.3 TDD Steps

1. **Red**: Write `tests/test_config.py` with tests for:
   - `Candidate` dataclass instantiation and immutability
   - `ModelConfig` default values
   - `FIRST_ROUND_CANDIDATES` has 5 entries + `rest` + `blanco`
   - `COALITION_TO_CANDIDATE` coverage for all 4 major coalitions + blank/null
   - `POLLSTER_WEIGHT_FORMULA` formula correctness
    - `get_active_candidates(1)` count and naming
    - `get_active_candidates(2)` count and naming
    - `CONSULTATION_VOTES` has correct keys and non-negative values
    - `consultation_log_share_prior()` returns finite values for all candidates (no -inf for zero-vote candidates)
    - `consultation_log_share_prior()` for a candidate with non-zero votes > log-share for a zero-vote candidate
2. **Green**: Implement `config.py`.
3. **Type-check + Lint + Commit**.

---

## 6. SPEC-03: Election Results Data Consolidation

**Goal**: Ingest both Registraduría (MMV) and MOE result files, aggregate each to national vote totals per candidate, cross-validate them, and produce a single canonical `ElectionResult` object for each round. This is our **source of truth** for validation (SPEC-09) and for the election-day anchor in the Bayesian model (SPEC-06).

### 6.1 Requirements

#### 6.1.1 Data Loading

 1. **`load_registraduria_round1() -> pd.DataFrame`**: Read `data/2022-presidential-results/MMV_NACIONAL_PRESIDENTE_2022_1v.csv`.
   - Encoding: ISO-8859-1 (latin-1)
   - Delimiter: `;` (semicolon)
   - Select columns: `CANNOMBRE`, `CAN` (candidate code), `VOTOS`
   - Group by `CANNOMBRE`, sum `VOTOS`
   - Map result candidate names to canonical keys via `COALITION_TO_CANDIDATE`
   - Exclude `VOTOS NULOS`, `VOTOS NO MARCADOS` from valid vote total (track separately)
   - Compute `vote_share = votes / total_votes_incl_blank` (candidates + blank votes, matching the official Colombian percentage calculation)

 2. **`load_registraduria_round2() -> pd.DataFrame`**: Same for `MMV_NACIONAL_PRESIDENTE_2022_2v.csv`.

 3. **`load_moe_round1() -> pd.DataFrame`**: Read `moe_vuelta1.csv`.
   - Encoding: UTF-8
   - Delimiter: `,` (comma)
   - Select columns: `nomparti` (coalition/party name), `votos`
   - Group by `nomparti`, sum `votos`
   - Map coalition names to canonical keys via `COALITION_TO_CANDIDATE`
   - Exclude `VOTOS NULOS`, `VOTOS NO MARCADOS` from valid vote total

 4. **`load_moe_round2() -> pd.DataFrame`**: Same for `moe_vuelta2.csv`.

 4b. **`load_participation_round1() -> pd.DataFrame`**: Read `data/2022-presidential-results/reg_participacion_vuelta1.csv`.
     - Encoding: UTF-8 with BOM (the file starts with `\ufeff`)
     - Delimiter: `,` (comma)
     - Aggregate `Total censo` to get national registered voters
     - Count unique `Código Puesto` to get total polling stations
     - Sum `Voto total`, `Voto blanco`, `Voto nulo`, `Voto no marcado` for validation

 4c. **`load_participation_round2() -> pd.DataFrame`**: Same for `reg_participacion_vuelta2.csv`.

#### 6.1.2 Data Structures

```python
@dataclass(frozen=True)
class CandidateResult:
    """A single candidate's result in an election round."""
    candidate_key: str
    votes: int
    vote_share: float  # votes / total_votes_incl_blank in that round


@dataclass(frozen=True)
class RoundResult:
    """Complete election results for one round."""
    round_number: Literal[1, 2]
    date: date
    total_valid_votes: int
    total_votes_incl_blank: int
    registered_voters: int       # From reg_participacion_vueltaX.csv: sum of Total censo
    polling_stations: int        # From reg_participacion_vueltaX.csv: count of unique Código Puesto
    candidates: tuple[CandidateResult, ...]
    blank_votes: int
    null_votes: int
    unmarked_votes: int

    def get_share(self, candidate_key: str) -> float:
        ...  # Raises KeyError if candidate_key not found in this round
    def get_candidates_above(self, threshold_pct: float) -> list[CandidateResult]:
        ...  # Returns empty list if none meet threshold (not an error)
    def top_two(self) -> tuple[CandidateResult, CandidateResult]:
        ...  # Raises ValueError if fewer than 2 candidates exist in this round
    def turnout(self) -> float:
        ...  # total_votes_incl_blank / registered_voters; raises ZeroDivisionError if registered_voters is 0
```

#### 6.1.3 Cross-Validation & Consolidation

5. **`cross_validate(registraduria: RoundResult, moe: RoundResult) -> list[str]`**: Compare two source results for the same round.
   - Compare total valid votes: if difference > 0.10%, log a WARNING
   - Compare each candidate's vote share: if any differ by >0.50%, log a WARNING
   - Return list of warning messages (empty list = perfect match)

6. **`consolidate_round(registraduria: RoundResult, moe: RoundResult, round_number: Literal[1, 2]) -> RoundResult`**: Returns the Registraduría result as canonical. Logs the MOE comparison. If discrepancies exceed tolerance, raise a `ValueError` with the cross-validation report.

7. **`load_canonical_results() -> tuple[RoundResult, RoundResult]`**: Loads both rounds from Registraduría, cross-validates against MOE, returns `(round1, round2)`. This is the single entry point for election results throughout the codebase.

### 6.2 Acceptance Criteria

- Round 1 vote shares match official results within ±0.02%:
  - Gustavo Petro: 40.34%
  - Rodolfo Hernández: 28.15%
  - Federico Gutiérrez: 23.89%
  - Sergio Fajardo: 4.39% (as Centro Esperanza coalition; Betancourt's individual votes are only in polls)
  - Blanco: 2.72% (actual blank votes, not poll "blanco" column)
- Round 2 vote shares match official results within ±0.02%:
  - Gustavo Petro: 50.44%
  - Rodolfo Hernández: 47.26%
- Cross-validation between Registraduría and MOE produces zero warnings (total votes match within 0.1%)
- `RoundResult.top_two()` for Round 1 returns `[Petro, Hernández]`
- `RoundResult.get_candidates_above(20.0)` for Round 1 returns `[Petro, Hernández, Gutiérrez]`
- `RoundResult.registered_voters` is > 0 and within ±1% of the official census figure (~39 million for 2022)
- All functions fully typed; `pyright` passes

### 6.3 TDD Steps

1. **Red**: Write `tests/test_data.py` with tests for:
   - `load_registraduria_round1()` returns a `DataFrame` with expected candidate keys
   - `load_moe_round1()` returns a `DataFrame` with expected coalition names
   - `load_participation_round1()` / `load_participation_round2()` return DataFrames with `Total censo` and `Código Puesto` columns
   - `RoundResult` dataclass instantiation and `get_share()`, `get_candidates_above()`, `top_two()` methods
   - `cross_validate()` with two identical `RoundResult`s → zero warnings
   - `cross_validate()` with differing shares → at least one warning returned
   - `consolidate_round()` returns Registraduría result unchanged when MOE matches
   - `consolidate_round()` raises `ValueError` when discrepancy exceeds tolerance
2. **Green**: Implement the loaders, data classes, cross_validate, consolidate_round, load_canonical_results.
3. **Type-check + Lint + Commit**.

---

## 7. SPEC-04: Poll Data Loading & Cleaning

**Goal**: Load, clean, normalize, and filter the poll data into a typed, analysis-ready container.

### 7.1 Requirements

#### 7.1.1 CandidateShares Dataclass

```python
@dataclass(frozen=True)
class CandidateShares:
    """Vote share breakdown for a single poll."""
    candidates: dict[str, float]   # candidate_key → normalized vote share (%)
    ns_nr: float                    # Undecided / no response (%)
    blanco: float                   # Blank vote (%)
    otros: float                    # "Other" candidates (%)

    def total(self) -> float:
        """Sum of all shares including ns_nr, blanco, otros. Must be 100 (±1% rounding tolerance) after normalize_undecided."""
        return sum(self.candidates.values()) + self.ns_nr + self.blanco + self.otros
```

#### 7.1.2 PollRow Dataclass

```python
@dataclass(frozen=True)
class PollRow:
    """One row of cleaned poll data."""
    date: date
    pollster: str
    sample_size: int
    sample_voting: int | None       # Effective voting sample (muestra_int_voto)
    margin_of_error: float | None   # Percentage points
    survey_method: str              # "presencial", "telefonica", "digital", "telefonico y presencial"
    round_number: Literal[1, 2]
    shares: CandidateShares         # Vote share breakdown
```

#### 7.1.3 ConsultationPoll Dataclass

```python
@dataclass(frozen=True)
class ConsultationPoll:
    """One row of inter-party consultation poll data."""
    date: date
    pollster: str
    coalition: str                  # "Pacto Historico", "Centro Esperanza", "Equipo por Colombia"
    candidate: str                  # Raw candidate name from CSV
    candidate_key: str              # Canonical key (e.g., "gustavo_petro")
    share: float                    # Vote intention within the coalition (%)
    sample_size: int
    margin_of_error: float | None
```

#### 7.1.4 Loading Functions

1. **`load_raw_polls(path: str | None = None) -> DataFrame`**: Read `encuestas_2022.csv`. Return a typed DataFrame. If `path` omitted, defaults to `data/2022-polls/encuestas_2022.csv`.

 2. **`load_raw_consultas(path: str | None = None) -> DataFrame`**: Read `consultas.csv` (ISO-8859-1). Same return schema.

 2b. **`parse_consultations(df: DataFrame) -> list[ConsultationPoll]`**: Convert the raw consultation DataFrame into typed `ConsultationPoll` objects. Calls `map_consultation_name_to_key()` to resolve canonical keys. Coalition field is derived from the source file's own coalition column.

 2c. **`map_consultation_name_to_key(name: str) -> str`**: Look up `name` in `CONSULTATION_KEY_MAP` (imported from `config`). Raises `ValueError` with the unrecognized name if no mapping exists. This ensures all consultation names are accounted for and no silent data loss occurs.

#### 7.1.5 Cleaning Functions

3. **`fix_invamer_date(df: DataFrame) -> DataFrame`**: The Invamer poll with `fecha == "2022-04-19"` and `encuestadora == "Invamer"` must be corrected to `2022-05-19`. This is a known data-entry error documented in `co_elect` (line `1a_v.R:9-12`).

 4. **`normalize_undecided(df: DataFrame) -> DataFrame`**: For every row where `ns_nr > 0`:
    ```
    candidate_normalized[i] = candidate_raw[i] / (100 - ns_nr) × 100
    ```
    where `i` ranges over **all** non-`ns_nr` columns: named candidates, `blanco`, and `otros`. This formula redistributes the undecided vote proportionally to all vote options. This is the formula from `co_elect` (`1a_v.R:14-36`).

5. **`retain_active_candidates(df: DataFrame, candidates: list[str]) -> DataFrame`**: Drop candidate columns whose share is `NA` for all rows (post-consultation, many candidates drop out). Keep only the columns corresponding to candidates who are `active` per `get_active_candidates()`.

6. **`infer_round_number(df: DataFrame) -> DataFrame`**: Assign `round_number` as:
   - `1` for polls where `gustavo_petro`, `federico_gutierrez`, `rodolfo_hernandez` are non-NA, and at least one of `sergio_fajardo` or `ingrid_betancourt` is also non-NA. Date must be after `CONSULTATION_DATE`. This relaxed criteria handles polls where some post-consultation candidates have NA shares.
   - `2` for polls where `gustavo_petro` and `rodolfo_hernandez` have non-NA values AND `federico_gutierrez`, `sergio_fajardo`, `ingrid_betancourt` are ALL NA. Date must be after `ELECTION_DATE_ROUND1`.
   - `None` for polls that don't fit either category (pre-consultation, mixed, etc.).
   Use the candidate column presence/absence as the primary signal; use dates as a secondary check.

 7. **`deduplicate_polls(df: DataFrame) -> DataFrame`**: When a pollster has multiple polls on the same date (same `fecha` and `encuestadora`), fill any NA `muestra_int_voto` with `muestra`, then keep only the row with the largest filled `muestra_int_voto`. This ensures each pollster contributes at most one data point per date.

#### 7.1.6 Container Dataclass

```python
@dataclass(frozen=True)
class CleanPolls:
    """Cleaned poll dataset split by round."""
    round1: DataFrame      # Post-consultation 1st round polls (>= 5 unique pollsters expected)
    round2: DataFrame      # Post-round1 runoff polls (>= 2 unique pollsters expected)
    consultation: list[ConsultationPoll]  # Pre-consultation coalition polls (typed)
    all_polls: DataFrame    # Unfiltered, all rows, for exploration

    def __post_init__(self) -> None:
        """Validate minimum pollster diversity per round."""
        round1_pollsters = self.round1["encuestadora"].nunique()
        round2_pollsters = self.round2["encuestadora"].nunique()
        if round1_pollsters < 5:
            msg = f"Round 1 needs >= 5 pollsters, got {round1_pollsters}"
            raise ValueError(msg)
        if round2_pollsters < 2:
            msg = f"Round 2 needs >= 2 pollsters, got {round2_pollsters}"
            raise ValueError(msg)
```

#### 7.1.7 Orchestration

 8. **`load_and_clean_all() -> CleanPolls`**: Calls all the above in sequence — including `parse_consultations` on the raw consultation data to produce typed `ConsultationPoll` objects — and returns a `CleanPolls`.

### 7.2 Acceptance Criteria

- `CleanPolls.round1` contains >= 5 unique pollsters representing post-consultation first-round polls
- `CleanPolls.round2` contains >= 2 unique pollsters representing runoff polls
- Normalized shares (candidates + blanco + otros) sum to 100% (±1% rounding tolerance) after undecided redistribution
- Invamer poll dated April 19 is corrected to May 19 in the output
- No pollster appears more than once on the same date in either round1 or round2
- `CleanPolls.__post_init__` validation passes on the actual 2022 data
- All functions fully typed; `pyright` passes

### 7.3 TDD Steps

 1. **Red**: Write `tests/test_data.py` (extend the file from SPEC-03) with tests for:
    - `fix_invamer_date` on a minimal 2-row DataFrame
    - `normalize_undecided` on a single row with 10% undecided: verify each candidate share is scaled by 100/(100-10) and sum = 100
    - `normalize_undecided` on a row with `ns_nr = 0`: no change
    - `retain_active_candidates` drops columns with all NA
    - `infer_round_number` correctly classifies Round 1 polls (5 active candidates)
    - `infer_round_number` correctly classifies Round 2 polls (only Petro + Hernández)
    - `deduplicate_polls` keeps the larger sample when pollster+date duplicates exist
     - `load_raw_consultas` returns a DataFrame; `parse_consultations(df)` returns a list of ConsultationPoll with correct coalition/candidate mapping
    - ConsultationPoll dataclass immutability and field types
2. **Green**: Implement the functions.
3. **Type-check + Lint + Commit**.

---

## 8. SPEC-05: Poll Aggregation (Baseline Comparison)

**Goal**: Produce a simple weighted polling average as a **baseline** for comparison with the Bayesian model. This is NOT the final forecast — it's a benchmark to validate that the Bayesian model adds value over simple heuristics.

### 8.1 Requirements

#### 8.1.1 Weighting Functions

1. **`time_weight(dates: Series[date], election_date: date, half_life: float = 30.0) -> Series[float]`**: Compute exponential decay weights relative to the election date:
   ```
   w_time = 0.5 ** (abs((election_date - poll_date).days) / half_life)
   ```
   Polls closer to the election date get higher weight. A poll on election day has weight 1.0. A poll 30 days before has weight 0.5.

2. **`sample_size_weight(sample_sizes: Series[int]) -> Series[float]`**: Compute diminishing-returns weighting:
   ```
   w_sample = log(sample_size + 1)
   ```
   This is normalized so that all weights are relative (normalization happens in `combined_weight`).

 3. **`pollster_weight_map(pollsters: Series[str], ratings: dict[str, float]) -> Series[float]`**: Map each pollster name to its rating-based weight using `POLLSTER_WEIGHT_FORMULA`. For pollsters not in the ratings dict, use the median weight of all rated pollsters.

 4. **`combined_weight(df: DataFrame, election_date: date, ratings: dict[str, float], half_life: float = 30.0) -> Series[float]`**: Compute `w = w_time × w_sample × w_pollster` per poll, then normalize globally so `Σ w = 1.0`. All candidates in the same poll share the same weight.

#### 8.1.2 Aggregation Functions

 5. **`weighted_average(df: DataFrame, candidates: list[str], election_date: date, ratings: dict[str, float]) -> dict[str, float]`**: For each candidate in `candidates`, compute the weighted mean of their normalized vote share across all polls using `combined_weight`. Return `{candidate_key: weighted_mean}`. Note: `candidates` includes all named candidates plus `"rest"` and `"blanco"` — the caller (e.g., SPEC-10's `aggregate` command) passes the full list.

 6. **`aggregation_snapshot(df: DataFrame, candidates: list[str], as_of_date: date, ratings: dict[str, float]) -> dict[str, float]`**: Same as `weighted_average` but using only polls with `fecha <= as_of_date`. This simulates "what did the forecast say on date X?". Same note: `candidates` includes `"rest"` and `"blanco"`.

 7. **`evolution_series(df: DataFrame, candidates: list[str], election_date: date, ratings: dict[str, float], n_snapshots: int = 15) -> DataFrame`**: Compute `aggregation_snapshot` for `n_snapshots` evenly spaced dates between `CONSULTATION_DATE` and 2 days before `election_date`. Return a DataFrame in **long format** with columns: `as_of_date` (date), `candidate` (str), `weighted_average` (float). Long format is chosen because it integrates directly with matplotlib and other plotting libraries without reshaping.

### 8.2 Acceptance Criteria

- Weighted averages sum to ~100% across all active candidates + rest + blanco (±2% due to undecided redistribution)
- Weights are poll-level: all candidates in the same poll share the same weight
- `time_weight` for a poll on election day = 1.0
- `time_weight` for a poll 30 days before election day = 0.5
- `sample_size_weight` is monotonic and convex (diminishing returns)
- `pollster_weight_map` for Invamer (rating 10) = 1.0, for Mosqueteros (rating 1.0) = 0.82
- Evolution series produces monotonically changing averages (no wild jumps unless data justifies)
- The final pre-election weighted average for Petro is within ±5 percentage points of 40.34%
- All functions fully typed; `pyright` passes

### 8.3 TDD Steps

1. **Red**: Write `tests/test_aggregation.py` with tests for:
   - `time_weight`: election day poll → 1.0; 30-day-ago poll → 0.5
   - `sample_size_weight`: monotonic; value for N=1000 > value for N=100
   - `pollster_weight_map`: known pollster → correct weight; unknown pollster → median fallback
   - `combined_weight`: all weights positive; normalization property
   - `weighted_average` on a 3-row synthetic DataFrame: verify arithmetic correctness
   - `aggregation_snapshot`: polls after cut date are not included
   - `evolution_series`: correct number of rows; dates are in chronological order
2. **Green**: Implement the functions.
3. **Type-check + Lint + Commit**.

---

## 9. SPEC-06: Bayesian Model — First Round

**Goal**: Implement the core forecasting model for the multi-candidate first round using a Dirichlet-Multinomial observation model with a **reverse-time random walk** for latent vote intention and hierarchical pollster house effects. This is the heart of the MVP.

### 9.1 Mathematical Specification

Let:
- `K` = number of active candidates + 2 (for `rest` and `blanco` categories kept separate)
- `T` = number of discrete time points (days before election, indexed 0 to T-1, where 0 is Election Day)
- `P` = number of pollsters
- `N` = number of polls

The model:

```
# Priors
σ_rw ~ HalfNormal(0.5)                    # Random walk step size
σ_house ~ HalfNormal(1.0)                  # House effect spread
φ_poll ~ Gamma(2, 2/μ_poll)               # Nominal concentration for polls (μ_poll = 5)
φ_elec ~ Gamma(5, 5/μ_elec)               # Nominal concentration for election (μ_elec = 50)

# Reverse-time random walk (logit scale)
# Initial prior: informed by March 13 consultation results
μ_consulta[k] = log(consultation_votes[k] / Σ consultation_votes)
# For candidates with 0 consultation votes: μ = log(min_nonzero / 2)
θ[T-1] ~ Normal(μ_consulta, σ_consulta_prior)  # Informative prior from real consultation votes
for t in T-2, ..., 0:
    θ[t] ~ Normal(θ[t+1], σ_rw)            # Walk backward toward election day

# House effects (ZeroSumNormal across pollsters)
η_raw[p] ~ Normal(0, σ_house)             # Raw per-pollster effects
η[p, k] = η_raw[p] - mean(η_raw)         # Zero-sum constraint, per candidate k
# η[p] is (P, K) matrix; zero-sum means Σ_p η[p,k] = 0 for each k

# Observation model (polls)
θ_adj[n] = θ[t_n] + η[pollster[n]]        # Logit-scale intention adjusted for house effects
p_adj[n] = softmax(θ_adj[n])              # Convert to probability simplex
α_poll[n] = p_adj[n] × φ_poll             # Concentration vector (DirichletMultinomial parameter)
observed[n] ~ DirichletMultinomial(       # n-th poll observation
    n = sample_size[n],
    a = α_poll[n]                         # PyMC: a is the concentration vector
)

# Election result likelihood
p_elec = softmax(θ[0])                     # Election-day vote share probabilities
α_elec = p_elec × φ_elec                  # Much higher concentration → election less noisy
actual ~ DirichletMultinomial(
    n = total_votes,
    a = α_elec
)
```

### 9.2 Requirements

#### 9.2.1 Model Construction

1. **`build_round1_model(polls: DataFrame, results: RoundResult | None, config: ModelConfig) -> pm.Model`**:
   - Extract candidate column names from config
   - Build time indices: `days_before_election = (ELECTION_DATE - poll_date).days`
   - Create integer time indices mapping each unique `days_before_election` value to `0, 1, ..., n_time_points - 1` where `0` maps to `days_before_election = 0` (election day)
   - Build pollster indices: unique pollster names → integer indices `0, 1, ..., P-1`
    - Implement the reverse-time random walk: initialize `θ[T-1]` with a Normal prior whose mean comes from `consultation_log_share_prior()` (computed from `CONSULTATION_VOTES` in config) and sigma from `config.consultation_prior_strength`. Then create a chain of `pm.Normal` variables with `θ[t] ~ Normal(θ[t+1], σ_rw)` for t in T-2, ..., 0.
   - Implement hierarchical house effects as raw effects with a zero-sum deterministic transform
   - If `results is not None`, observe the election result likelihood
   - If `results is None`, do not include the election likelihood (this is the "forecast" mode where we only have polls)

2. **Zero-sum house effects implementation**:
   ```python
   # Not using a built-in ZeroSumNormal (PyMC doesn't have one).
   # Instead, enforce zero-sum via deterministic centering:
   raw_house = pm.Normal("raw_house", mu=0, sigma=sigma_house, shape=(n_pollsters, K))
   house_effects = pm.Deterministic(
       "house_effects",
       raw_house - raw_house.mean(axis=0, keepdims=True)
   )
   # Now for each candidate k: Σ_p house_effects[p, k] = 0
   ```

3. **Softmax transformation**:
   ```python
   # On the logit scale, θ[t] is a vector of length K
   # softmax(θ[t]) produces a probability simplex of length K
   p_adj = pm.Deterministic("p_adj", pm.math.softmax(theta[time_idx] + house_effects[pollster_idx]))
   ```

4. **`sample_round1(model: pm.Model, config: ModelConfig) -> InferenceData`**: Run `pm.sample()` with:
   - `draws=config.mcmc_draws`
   - `tune=config.mcmc_tune`
   - `chains=config.mcmc_chains`
   - `cores=config.mcmc_cores`
   - `target_accept=config.target_accept`
   - `random_seed=config.seed`

#### 9.2.2 Forecast Data Structures

5. **`CandidateForecast` dataclass**:
   ```python
   @dataclass(frozen=True)
   class CandidateForecast:
       candidate_key: str
       mean_share: float
       median_share: float
       ci_50: tuple[float, float]     # 50% credible interval
       ci_95: tuple[float, float]     # 95% credible interval
       prob_first: float              # P(candidate finishes 1st)
       prob_second: float             # P(candidate finishes 2nd)
       prob_top_two: float            # P(candidate finishes 1st OR 2nd)
       prob_win_outright: float       # P(candidate vote share > 50%)
   ```

6. **`Round1Forecast` dataclass**:
   ```python
   @dataclass(frozen=True)
   class Round1Forecast:
       candidates: list[CandidateForecast]
       prob_runoff: float             # P(no candidate wins outright)
       round_number: Literal[1] = 1
   ```

 6b. **`to_json()` / `from_json()` helpers**:
     ```python
     # On Round1Forecast:
     @staticmethod
     def from_dict(d: dict) -> Round1Forecast: ...
     def to_dict(self) -> dict: ...
     def to_json(self) -> str: ...          # json.dumps(self.to_dict())
     @staticmethod
     def from_json(s: str) -> Round1Forecast: ...  # from_dict(json.loads(s))
     ```
     Uses `dataclasses.asdict()` internally for `to_dict()` and reconstructs `CandidateForecast` and tuple CI fields in `from_dict()`. Used by `save_rolling_snapshot` / `load_rolling_snapshots` (SPEC-09).

7. **`forecast_round1(idata: InferenceData, candidates: list[str]) -> Round1Forecast`**: Extract posterior draws of election-day probabilities `softmax(θ[0])`. For each posterior draw, rank candidates and count frequencies to compute `prob_first`, `prob_second`, `prob_top_two`, `prob_win_outright`. Compute mean, median, and HDI for each candidate's share.

#### 9.2.3 Posterior Predictive

8. **`simulate_elections(idata: InferenceData, candidates: list[str], n_simulations: int = 10000) -> DataFrame`**: From the posterior, draw `n_simulations` sets of election-day vote shares. For each simulation, determine the winner (and whether there's a runoff). Return a DataFrame with columns: `sim_id`, `candidate`, `share`, `rank`, `win_outright`, `goes_to_runoff`.

### 9.3 Acceptance Criteria

- Model builds without PyMC errors (graph test passes)
- MCMC converges for 2022 data:
  - R-hat < 1.05 for all `theta[0]` parameters (election-day latent shares)
  - R-hat < 1.05 for `sigma_rw`, `sigma_house`
  - Effective sample size (ESS bulk) > 100 for all parameters
  - No divergent transitions
- Posterior means for major candidates are within ±5% of actual 2022 results:
  - Petro: within [35.34%, 45.34%] (actual: 40.34%)
  - Hernández: within [23.15%, 33.15%] (actual: 28.15%)
  - Gutiérrez: within [18.89%, 28.89%] (actual: 23.89%)
- House effects are non-zero for at least 2 pollsters (model captures pollster bias)
- `prob_runoff` > 0.90 (the actual 2022 election went to a runoff)
- `prob_top_two` for Petro > 0.95 and Hernández > 0.85
- Model runtime < 15 minutes on a standard laptop for the 2022 dataset
- All functions fully typed; `pyright` passes

### 9.4 TDD Steps

 1. **Red**: Write `tests/test_model.py` with tests for:
     - **Phase A (minimal model)**: `build_round1_model` on a 2-candidate, 2-poll synthetic DataFrame with NO house effects and a **single time point** (T=1, both polls on election day). Model builds. PyMC graph has 2 free RVs (`sigma_rw`, `theta[0]` shape 2), 1 deterministic (`p_adj` softmax), 1 observed RV (`DirichletMultinomial`).
     - **Phase B (add house effects)**: Same synthetic data with a single time point but 2 pollsters. PyMC graph has 4 free RVs (`sigma_rw`, `theta[0]`, `sigma_house`, `raw_house` shape 2×2), 2 deterministics (`house_effects` zero-sum, `p_adj` softmax), 1 observed RV.
    - **Phase C (full model)**: 3-row synthetic DataFrame with 3 pollsters, 3 candidates. Model builds with expected RV count. `pm.sample_prior_predictive` returns shares in [0, 1].
    - **Phase D (forecast mode)**: `results=None` → no election likelihood term present in model.
     - **Phase E (backtest mode)**: `results=something` → election likelihood term present.
     - **Consultation prior**: `consultation_log_share_prior()` returns log-shares derived from `CONSULTATION_VOTES`. For a candidate with zero consultation votes, the return value is `log(min_nonzero / 2)`, not `-inf`. In Phase C, the theta prior mean matches the consultation log-shares for the synthetic candidates.
     - `CandidateForecast` and `Round1Forecast` dataclass validation: all probabilities in [0, 1]; mean_share >= 0.
     - `Round1Forecast.to_json()` / `Round1Forecast.from_json()` roundtrip: serialized → deserialized values match original (no CI precision loss).
     - `forecast_round1` on a manually constructed `InferenceData` with 2 candidates: verifies CI width is positive.
 2. **Green**: Implement Phase A first (2-candidate model, no house effects). Make it pass. Then incrementally add Phases B through E. Each phase is a separate commit.
 3. **Full-data convergence test** (runs in CI or manually, not on every commit): Fit on real data. Check R-hat < 1.05 and ESS > 100.
 4. **Type-check + Lint + Commit**.

---

## 10. SPEC-07: Bayesian Model — Runoff (Simple)

**Goal**: Given the actual top-2 candidates from Round 1, predict the runoff using a Dirichlet(3) model with the same reverse-time random walk structure and house effects.

### 10.1 Mathematical Specification

The runoff model is structurally identical to the Round 1 model but with `K = 3` (Candidate A, Candidate B, rest/blanco):

```
K_runoff = 3  # A, B, rest

# Reverse-time random walk, same house effects, same DirichletMultinomial observation
# Differences:
# 1. θ has dimension 3 instead of 7+
# 2. The prior for θ[T-1] uses a reference parameterization (rest category is reference = 0):
#    θ_A[T-1] ~ Normal(log(p_A / p_rest), 0.5)
#    θ_B[T-1] ~ Normal(log(p_B / p_rest), 0.5)
#    θ_rest[T-1] = 0  # fixed reference
#    where p_A, p_B, p_rest are the Round 1 vote shares of the two runoff candidates and rest/blanco
#    e.g., if Petro got 40.34% and Hernández got 28.15%,
#    with round1 rest+blanco = 100 - 40.34 - 28.15 = 31.51%,
#    the prior logit for Petro is log(40.34 / 31.51) = 0.247
# 3. Election date is June 19 instead of May 29
```

### 10.2 Requirements

1. **`build_runoff_simple_model(polls: DataFrame, results: RoundResult, round1_idata: InferenceData | None, config: ModelConfig) -> pm.Model`**:
   - Filter polls to the two actual runoff candidates from `results.top_two()`
   - Set `K = 3` (A, B, rest)
   - Time indices anchored on `ELECTION_DATE_ROUND2`
   - If `round1_idata` is provided, use the Round 1 posterior of the top-two's relative margins as an informative prior for the runoff's starting point
   - Otherwise, use a vague prior (this is the fallback for when Round 1 model hasn't been run)

2. **`sample_runoff(model: pm.Model, config: ModelConfig) -> InferenceData`**: Same MCMC configuration as SPEC-06.

 3. **`RunoffForecast` dataclass**:
    ```python
    @dataclass(frozen=True)
    class RunoffForecast:
        candidate_a_key: str
        candidate_b_key: str
        prob_a_wins: float
        prob_b_wins: float
        mean_share_a: float
        mean_share_b: float
        mean_margin: float             # mean(A) - mean(B)
        ci_95_a: tuple[float, float]
        ci_95_b: tuple[float, float]
    ```

4. **`forecast_runoff_simple(idata: InferenceData, candidate_a: str, candidate_b: str) -> RunoffForecast`**: Extract posterior predictions and compute the forecast.

### 10.3 Acceptance Criteria

- Model correctly identifies Petro as the runoff winner with probability > 0.55 when using all 2nd round polls
- Posterior mean vote shares within ±3% of actual runoff results:
  - Petro: within [47.44%, 53.44%] (actual: 50.44%)
  - Hernández: within [44.26%, 50.26%] (actual: 47.26%)
- Expected margin (Petro − Hernández) within [0%, 8%] (actual: 3.18%)
- R-hat < 1.05 for all parameters
- All functions fully typed; `pyright` passes

### 10.4 TDD Steps

1. **Red**: Write additional tests in `tests/test_model.py`:
   - `build_runoff_simple_model` on synthetic 3-row runoff data: model builds, has K=3
   - The Round 1 prior is correctly incorporated when `round1_idata` is provided
   - `RunoffForecast` dataclass validation
   - `forecast_runoff_simple` on synthetic posterior: verifies probability ranges
2. **Green**: Implement.
3. **Type-check + Lint + Commit**.

---

## 11. SPEC-08: Runoff Probability Matrix (Enhanced)

**Goal**: From the Round 1 posterior, compute the probability of every possible runoff pairing and, for each pairing, estimate who would win. This provides a complete probabilistic picture *before* Round 1 results are known.

### 11.1 Requirements

1. **`compute_top_two_probabilities(round1_idata: InferenceData, candidates: list[str]) -> dict[tuple[str, str], float]`**: From the posterior draws of election-day vote shares, for each draw, determine the top two candidates by share. Count frequencies of each ordered pair `(first_place, second_place)`. Return `{(cand_a, cand_b): probability}`.

2. **`PairingForecast` dataclass**:
   ```python
   @dataclass(frozen=True)
   class PairingForecast:
       candidate_first: str
       candidate_second: str
       prob_pairing: float           # P(this exact pairing occurs)
       prob_first_wins: float        # P(first candidate wins runoff)
       prob_second_wins: float       # P(second candidate wins runoff)
       mean_margin: float            # Expected vote margin
   ```

3. **`RunoffMatrix` dataclass**:
   ```python
   @dataclass(frozen=True)
   class RunoffMatrix:
       pairings: list[PairingForecast]
       prob_runoff: float            # Σ pairing probabilities = P(no outright winner)
       ordered_by_likelihood: tuple[tuple[str, str], ...]  # Pairings from most to least likely
   ```

 4. **`estimate_runoff_matrix(round1_idata: InferenceData, results: tuple[RoundResult, RoundResult], round2_polls: DataFrame | None, config: ModelConfig) -> RunoffMatrix`**:
   - Call `compute_top_two_probabilities`
   - For each pairing with probability > 0.01:
     - If `round2_polls` contains head-to-head polls for these two candidates, run `build_runoff_simple_model` with those polls filtered
     - If no head-to-head polls exist, use a **transfer heuristic**: assume eliminated candidates' Round 1 votes redistribute proportionally based on coalition alignment. For example, Gutiérrez voters (Equipo por Colombia, right-wing) are more likely to transfer to Hernández (anti-establishment right) than to Petro (left).
   - The transfer heuristic uses a simple rule-based system:
     - Blanco votes split per `TRANSFER_BLANCO_SPLIT` (default 0.5 = equal split) — TODO: refine with legislative data
     - "Rest" minor candidates split proportionally to the major candidate shares in the Round 1 posterior
     - Centro Esperanza (Fajardo) votes split per `TRANSFER_FAJARDO_PETRO` (default 0.60) toward Petro — TODO: refine with legislative data
     - Equipo por Colombia (Gutiérrez) votes split per `TRANSFER_GUTIERREZ_HERNANDEZ` (default 0.70) toward Hernández — TODO: refine with legislative data
   - Run the runoff model for each pairing
   - Aggregate into `RunoffMatrix`

 5. **`overall_win_probability(matrix: RunoffMatrix, prob_win_outright: dict[str, float]) -> dict[str, float]`**: Compute the total probability that each candidate becomes president:
    ```
    P(Candidate X wins) = Σ_{pairings containing X} prob_pairing × prob_X_wins_in_that_pairing
    + prob_win_outright[X]
    ```
    `prob_win_outright` is provided by the caller from SPEC-06's `Round1Forecast.candidates[n].prob_win_outright`.

### 11.2 Acceptance Criteria

- Petro vs Hernández pairing probability > 90% (this was the actual runoff)
- All pairing probabilities in the matrix sum to `1 - P(win_outright)` within ±1%
- `overall_win_probability` for Petro > 0.55 (Petro was the eventual winner)
- The transfer heuristic produces plausible results (no candidate gets >100% of an eliminated candidate's votes)
- All functions fully typed; `pyright` passes

### 11.3 TDD Steps

1. **Red**: Write additional tests in `tests/test_model.py`:
   - `compute_top_two_probabilities` on a synthetic posterior with known top-two: produces the correct pair and probability ≈ 1.0
   - `PairingForecast` and `RunoffMatrix` dataclass validation
   - `overall_win_probability` edge case: only one candidate > 50% in Round 1 → probability = 1.0 for that candidate
   - Transfer heuristic: eliminated candidate votes sum to 100% for each pairing (no votes lost)
2. **Green**: Implement.
3. **Type-check + Lint + Commit**.

---

## 12. SPEC-09: Validation & Backtesting

**Goal**: Systematically compare model predictions against actual 2022 results using multiple metrics, and produce visualizations.

### 12.1 Requirements

#### 12.1.1 Validation Data Structures

1. **`CandidateValidation` dataclass**:
   ```python
   @dataclass(frozen=True)
   class CandidateValidation:
       candidate_key: str
       actual_share: float
       predicted_mean: float
       predicted_median: float
       error: float                    # predicted_mean - actual_share
       abs_error: float                # |error|
       within_95ci: bool              # Is actual_share within 95% credible interval?
       within_50ci: bool              # Is actual_share within 50% credible interval?
   ```

2. **`RoundValidation` dataclass**:
   ```python
   @dataclass(frozen=True)
   class RoundValidation:
       round_number: Literal[1, 2]
       candidates: list[CandidateValidation]
       mae: float                      # Mean Absolute Error across candidates
       rmse: float                     # Root Mean Square Error
       calibration_95: float           # Fraction of candidates whose actual is within 95% CI
       calibration_50: float           # Fraction of candidates whose actual is within 50% CI
   ```

#### 12.1.2 Validation Functions

3. **`validate_round1(forecast: Round1Forecast, results: RoundResult) -> RoundValidation`**: Compute `CandidateValidation` for each candidate in the forecast. Compare forecast means/CIs against actual vote shares from `results`.

4. **`validate_runoff(forecast: RunoffForecast, results: RoundResult) -> RoundValidation`**: Same for the runoff.

5. **`brier_score_round1(forecast: Round1Forecast, results: RoundResult) -> float`**: For the binary outcome "candidate makes the runoff":
   ```
   Brier = (1/N) × Σ (p_i - o_i)²
   ```
   where `p_i` = `prob_top_two[i]` and `o_i` = 1 if candidate actually made the runoff, 0 otherwise.

6. **`rolling_forecast(polls: CleanPolls, results: tuple[RoundResult, RoundResult], config: ModelConfig, n_snapshots: int = 10) -> list[tuple[date, Round1Forecast]]`**: Re-fit the Round 1 model using only polls available up to each cutoff date. Produces a time-series of forecasts showing how predictions evolve as election day approaches. Cutoff dates are evenly spaced from `CONSULTATION_DATE + 30 days` to 2 days before `ELECTION_DATE_ROUND1`. Return a list of `(cutoff_date, forecast)`.

 7. **`compute_rolling_errors(rolling: list[tuple[date, Round1Forecast]], results: RoundResult) -> DataFrame`**: For each forecast in the rolling series, compute MAE and RMSE. Return a DataFrame tracking error over time.

 7b. **`save_rolling_snapshot(snapshot: tuple[date, Round1Forecast], output_dir: str) -> None`**: Save a single rolling forecast snapshot to disk as **JSON** (human-readable, debuggable). Uses a `to_json()` helper on `Round1Forecast` / `CandidateForecast` that converts dataclasses to dicts via `dataclasses.asdict()`. File extension: `.json`. Enables resuming interrupted rolling forecasts without re-running all prior snapshots.

 7c. **`load_rolling_snapshots(output_dir: str) -> list[tuple[date, Round1Forecast]]`**: Load all saved snapshots from disk. Used to resume a rolling forecast that was interrupted.

#### 12.1.3 Visualization Functions (in `plotting.py`)

8. **`plot_forecast_evolution(rolling: list[tuple[date, Round1Forecast]], results: RoundResult) -> Figure`**: Line chart with:
   - X-axis: date
   - Y-axis: vote share (%)
   - One line per major candidate showing predicted mean
   - Shaded band showing 95% CI
   - Horizontal dashed lines showing actual results
   - Must be readable in monochrome and color

 9. **`plot_calibration(validation: RoundValidation) -> Figure`**: Scatter plot of predicted vs actual:
    - X-axis: predicted mean share (%)
    - Y-axis: actual result (%)
    - One point per candidate with horizontal error bars showing 95% CI
    - 45-degree reference line (y=x): points on the line are perfectly calibrated
    - Points above the line = model overpredicted; below = underpredicted
    - A separate `plot_forecast_comparison` (dot plot with candidate names on X-axis) is left for a later phase

10. **`plot_error_over_time(rolling_errors: DataFrame) -> Figure`**: MAE and RMSE vs time, showing the model's accuracy improving as election day approaches.

### 12.2 Acceptance Criteria

- `validate_round1` computes all metrics correctly (MAE verified against manual calculation)
- 95% CIs contain actual results for all major candidates (Petro, Hernández, Gutiérrez)
- Calibration: at least 80% of candidates have actuals within 95% CI
- Rolling forecast shows MAE decreasing (monotonically or near-monotonically) as election day approaches
- `brier_score_round1` is lower (better) than the baseline (always predicting the polling average)
- All plots render without errors
- All functions fully typed; `pyright` passes

### 12.3 TDD Steps

1. **Red**: Write `tests/test_validation.py` with tests for:
   - `CandidateValidation` with known inputs/outputs: verify error, abs_error, CI containment
   - `RoundValidation` MAE and RMSE on small manual dataset
   - `brier_score_round1` on known probabilities vs known outcomes
    - `compute_rolling_errors` on synthetic 3-point rolling forecast
    - `Round1Forecast.to_json()` / `Round1Forecast.from_json()` roundtrip: serialize, deserialize, verify all fields match
    - Plot functions: at minimum assert figure object is returned, no exceptions raised
2. **Green**: Implement.
3. **Type-check + Lint + Commit**.

---

## 13. SPEC-10: Integration & CLI

**Goal**: Provide a command-line interface to run the full pipeline end-to-end, and ensure all modules integrate correctly.

### 13.1 Requirements

#### 13.1.1 CLI Entry Point (`__main__.py`)

```bash
# Run the full pipeline
python -m co_president run [--no-sample] [--config-override KEY=VALUE ...]

# Run backtesting only (assumes model outputs exist)
python -m co_president validate

# Run baseline weighted average only
python -m co_president aggregate

# Generate all visualization plots
python -m co_president plot [--output-dir results/]

# Show configuration
python -m co_president config
```

1. **`run` command**: Executes the full pipeline:
   - `load_canonical_results()` → round1_results, round2_results
   - `load_and_clean_all()` → clean_polls
   - `build_round1_model()` → fit → `forecast_round1()`
   - `build_runoff_simple_model()` → fit → `forecast_runoff_simple()`
   - `estimate_runoff_matrix()`
   - `validate_round1()`, `validate_runoff()`
   - Prints summary table to stdout
   - Saves `InferenceData` to `results/` directory
    - `--no-sample` flag: skips MCMC sampling entirely. Instead of model predictions, prints the baseline weighted average from SPEC-05 as the forecast in the same summary table format. Useful for testing the data pipeline end-to-end without waiting for MCMC.

2. **`validate` command**: Loads saved `InferenceData` from `results/` and runs validation only.

3. **`aggregate` command**: Runs only the baseline weighted average (SPEC-05) for quick diagnostics.

4. **`plot` command**: Generates all figures from SPEC-09 and saves them to the specified output directory.

5. **`config` command**: Prints the current configuration (candidates, dates, pollster ratings, model hyperparameters).

#### 13.1.2 Config Override Specification

The `--config-override KEY=VALUE` flag applies to `ModelConfig` fields only:

- `KEY` must match a field name in `ModelConfig` (e.g., `random_walk_sigma_prior`, `mcmc_draws`, `target_accept`).
- `VALUE` is type-cast using the field's annotation (`int` or `float`).
- Invalid `KEY` raises `ValueError` with available keys listed in the error message.
- Multiple overrides are supported: `--config-override key1=val1 key2=val2`.
- Overrides take effect only for the current invocation; they do not modify the config file.

#### 13.1.3 Output Format

The `run` command prints a summary table:

```
======================================================================
  CO-PRESIDENT 2026 — 2022 Backtesting Results
======================================================================

  FIRST ROUND (May 29, 2022)
  +-------------------+----------+----------+-----------+------------------+
  | Candidate         | Predicted| Actual   | Error     | 95% CI            |
  +-------------------+----------+----------+-----------+------------------+
  | Gustavo Petro     |   41.2%  |  40.34%  |  +0.86pp  | [37.1%, 45.3%]    |
  | Rodolfo Hernandez |   27.8%  |  28.15%  |  -0.35pp  | [23.4%, 32.1%]    |
  | Federico Gutierrez|   22.5%  |  23.89%  |  -1.39pp  | [18.2%, 26.8%]    |
  | Sergio Fajardo    |    5.1%  |   4.39%  |  +0.71pp  | [ 3.2%,  7.0%]    |
  | Ingrid Betancourt |    0.8%  |   0.40%  |  +0.40pp  | [ 0.3%,  1.3%]    |
  +-------------------+----------+----------+-----------+------------------+
  MAE: 0.74pp  |  RMSE: 0.89pp  |  Brier: 0.008
  Probability of runoff: 99.2%

  RUNOFF (June 19, 2022) — Petro vs. Hernandez
  Gustavo Petro:    51.8% [48.2%, 55.4%]  ->  Win probability: 78.3%
  Rodolfo Hernandez: 48.2% [44.6%, 51.8%]  ->  Win probability: 21.7%
  MAE: 1.36pp  |  Expected margin: +3.6pp Petro
======================================================================
```

#### 13.1.4 Error Handling

6. **Graceful degradation**: If a model fails to converge (R-hat > 1.10), print a WARNING to stderr and continue with the rest of the pipeline. Do not crash.

7. **Data validation on startup**: Before MCMC sampling, validate:
   - At least 5 unique pollsters in round 1 data
   - At least 2 unique pollsters in round 2 data
   - No candidate shares > 100% or < 0%
   - Election results loaded and valid (total shares ≈ 100%)

8. **Logging**: Use Python's `logging` module. INFO level for progress. WARNING for data quality issues. ERROR for fatal issues.

### 13.2 Acceptance Criteria

- `python -m co_president run --no-sample` completes without error (tests data pipeline end-to-end)
- `python -m co_president run` completes and prints the summary table
- `python -m co_president validate` loads saved InferenceData and prints validation metrics
- `python -m co_president aggregate` prints baseline weighted averages
- `python -m co_president plot` generates PNG files in the specified output directory
- `python -m co_president config` prints all configuration values
- All functions fully typed; `pyright` passes

### 13.3 TDD Steps

1. **Red**: Write `tests/test_cli.py` with tests for:
   - `python -m co_president config` prints expected keys
   - `python -m co_president aggregate` exits with code 0
   - `python -m co_president run --no-sample` exits with code 0
   - Data validation catches obviously bad inputs (negative sample sizes, shares > 100%)
2. **Green**: Implement `__main__.py`.
3. **Type-check + Lint + Commit**.

---

## 14. Implementation Order & Dependency Graph

```
  SPEC-01 (Scaffolding)
    ├── SPEC-02 (Config & Constants)
    │     ├── SPEC-03 (Results Consolidation) ─────────────────────┐
    │     │     └── SPEC-04 (Poll Loading & Cleaning)              │
    │     │           ├── SPEC-05 (Baseline Aggregation)           │
    │     │           │     └── benchmark for comparison           │
    │     │           ├── SPEC-06 (Bayesian 1st Round)             │
    │     │           │     ├── SPEC-07 (Runoff Simple)            │
    │     │           │     │     ├── SPEC-08 (Runoff Matrix)      │
    │     │           │     │     │     └── SPEC-09 (Validation)   │
    │     │           │     │     │           └── SPEC-10 (CLI)    │
    │     │           │     │     └── SPEC-09 (Validation)         │
    │     │           │     └── SPEC-09 (Validation)               │
    │     │           └── SPEC-09 (Validation)                     │
    │     └── SPEC-09 (Validation)                                 │
    └── (all specs depend on SPEC-01 tooling)                      │
```

### 14.1 Sequential Build Order

| Step | Spec | Files | Estimated Effort | Must Have Before |
|------|------|-------|------------------|------------------|
| 1 | SPEC-01 | `pyproject.toml`, `uv.lock`, `.python-version`, `.gitignore`, `conftest.py` | Small | — |
| 2 | SPEC-02 | `config.py`, `test_config.py` | Small | SPEC-01 |
| 3 | SPEC-03 | `data.py` (results part), `test_data.py` (results tests) | Medium | SPEC-02 |
| 4 | SPEC-04 | `data.py` (polls part), `test_data.py` (polls tests) | Medium | SPEC-02, SPEC-03 data loading established |
| 5 | SPEC-05 | `aggregation.py`, `test_aggregation.py` | Small | SPEC-04 |
| 6 | SPEC-06 | `model_round1.py`, `test_model.py` | Large | SPEC-04, SPEC-02 |
| 7 | SPEC-07 | `model_runoff_simple.py`, `test_model.py` (extend) | Medium | SPEC-06, SPEC-03 |
| 8 | SPEC-09 | `validation.py`, `plotting.py`, `test_validation.py` | Medium | SPEC-06, SPEC-07, SPEC-03 |
| 9 | SPEC-08 | `model_runoff_matrix.py`, `test_model.py` (extend) | Medium | SPEC-06, SPEC-07, SPEC-09 (validation verifies the matrix makes sense) |
| 10 | SPEC-10 | `__main__.py`, `test_cli.py` | Small | All above |

### 14.2 TDD Execution Per Spec

For each spec in the build order:

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: WRITE TESTS                                         │
│    - Read the spec's "TDD Steps" section                     │
│    - Write all test cases in the listed test file            │
│    - Run `pytest tests/ -v -k test_<spec>`                   │
│    - Confirm ALL tests FAIL (they should: no code exists yet) │
│                                                              │
│  Step 2: IMPLEMENT                                           │
│    - Write the minimum code to make all tests pass           │
│    - Run `pytest tests/ -v`                                  │
│    - Confirm ALL tests PASS (including tests from prior specs)│
│                                                              │
│  Step 3: TYPE-CHECK                                          │
│    - Run `pyright src/`                                      │
│    - Fix ALL type errors                                     │
│                                                              │
│  Step 4: LINT                                                │
│    - Run `ruff check src/ tests/`                            │
│    - Fix ALL lint errors                                     │
│                                                              │
│  Step 5: COMMIT                                              │
│    - `git add -A`                                            │
│    - `git commit -m "feat(SPEC-XX): <description>"`          │
└─────────────────────────────────────────────────────────────┘
```

### 14.3 CI Pipeline (Recommended)

```yaml
# .github/workflows/ci.yml
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync
      - run: uv run ruff check src/ tests/
      - run: uv run pyright src/
      - run: uv run pytest tests/ -v --ignore=tests/test_model.py
        # Unit tests (everything except model tests, which are slow)
  model-tests:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync
      - run: uv run pytest tests/test_model.py -v -m "not slow"
        # Fast model tests only (graph tests, prior predictive)
      - run: uv run pytest tests/test_model.py -v -m "slow"
        # Slow model tests (convergence, small-data sanity) — can be manual trigger
```

---

## 15. Appendix: Key Formulas Reference

| Formula | Expression | Source |
|---|---|---|
| Pollster weight | `w = rating × 0.02 + 0.8` | `co_elect` line `1a_v.R:43` |
| Undecided normalization | `c_norm = c_raw / (100 − ns_nr) × 100` | Applied to all non-ns_nr columns (candidates + blanco + otros). `co_elect` lines `1a_v.R:14-36` |
| Exponential time decay | `w_time = 0.5^(t_days / half_life)` | 538 + Pollsposition |
| Sample size weight | `w_sample = log(size + 1)` | Derived from 538 effective sample size concept |
| Reverse-time random walk | `θ[t] ~ Normal(θ[t+1], σ_rw²)`, with `θ[T] = election day` | Economist (Gelman/Heidemanns) |
| Zero-sum house effects | `η[p,k] = η_raw[p,k] − mean_p(η_raw[:,k])` | Pollsposition |
| Softmax transformation | `p = softmax(θ) = exp(θ) / Σexp(θ)` | Standard logit-to-simplex mapping |
| Dirichlet-Multinomial likelihood | `y ~ DM(n, α, p)` with concentration `α` | Pollsposition + `co_elect` |
| Election concentration prior | `φ_elec = 50` (much higher than polls' φ) → election is lower-noise | Pollsposition |
| MAE (Mean Absolute Error) | `MAE = (1/N) × Σ|pred_i − actual_i|` | Standard |
| RMSE (Root Mean Square Error) | `RMSE = √((1/N) × Σ(pred_i − actual_i)²)` | Standard |
| Brier score | `(1/N) × Σ(p_i − o_i)²` | Standard (binary outcome scoring) |

---

## 16. Appendix: Known Anomalies in 2022 Data

These are documented for awareness during implementation:

1. **Invamer date error**: Row `n=26` has `fecha=2022-04-19` but should be `2022-05-19`. The `co_elect` reference documents this and applies a correction. SPEC-04 must include this correction.

2. **MassiveCaller duplicates**: MassiveCaller appears 10 times in the dataset (roughly once per week). These are all automated IVR (interactive voice response) polls. Their `muestra_int_voto` is always the same as `muestra` (1000). They appear in both Round 1 and Round 2. The deduplication in SPEC-04 should handle these.

3. **Centro Esperanza coalition split**: In the polls, Fajardo and Betancourt are tracked separately. In the official results, Centro Esperanza is a single coalition. Fajardo won the coalition's primary, so all Centro Esperanza votes in the Registraduría/MOE data are his. Betancourt's `0.40%` in the official results comes from votes she received independently after withdrawing from the coalition. SPEC-03 must handle this correctly.

4. **GAD3 runoff polls — originally not in main CSV; now recovered**:
   - The 10 "missing" GAD3 tracking poll waves (May 31 – June 10) were **found** documented on the Spanish Wikipedia page [Sondeos de intención de voto para las elecciones presidenciales de Colombia de 2022](https://es.wikipedia.org/wiki/Anexo:Sondeos_de_intenci%C3%B3n_de_voto_para_las_elecciones_presidenciales_de_Colombia_de_2022) with RCN Radio as the primary source.
   - **Verification**: 7 of 10 waves were cross-checked against Wayback Machine archives of the original RCN Radio articles. All confirmed values matched Wikipedia exactly.
   - **Status**: Waves 1–10 have been added to `encuestas_2022.csv`. Wave 11 (June 11) was already present as row 41.
   - These polls include a `blanco` category (2.6%–5.8%), so they are compatible with the K=3 Dirichlet-Multinomial runoff model.

| Wave | Date Range | Hernández | Petro | Blanco | Sample |
|---|---|---|---|---|---|
| 1 | May 30–31 | 52.5% | 44.8% | 2.7% | 1,200 |
| 2 | May 30–Jun 1 | 52.3% | 45.1% | 2.6% | 1,755 |
| 3 | May 30–Jun 2 | 50.4% | 45.6% | 4.0% | 2,308 |
| 4 | May 30–Jun 3 | 47.9% | 46.3% | 5.8% | 2,840 |
| 5 | May 30–Jun 4 | 48.1% | 46.8% | 5.1% | 3,240 |
| 6 | May 30–Jun 6 | 47.8% | 46.8% | 5.4% | 3,641 |
| 7 | May 30–Jun 7 | 47.1% | 47.8% | 5.1% | 4,041 |
| 8 | May 30–Jun 8 | 46.7% | 48.5% | 4.9% | 4,438 |
| 9 | May 30–Jun 9 | 46.8% | 48.1% | 5.1% | 4,836 |
| 10 | May 30–Jun 10 | 47.9% | 47.1% | 5.0% | 5,236 |
| 11 | May 30–Jun 11 | 47.9% | 47.1% | 5.0% | 5,236 |

Recommendation: Before SPEC-07 (runoff model), ensure `encuestas_2022.csv` includes these GAD3 waves. If convergence diagnostics are still poor (R-hat > 1.10) after adding them, the June 3–7 gap (now partially filled) may still contribute, but other model issues should be investigated first.

5. **ISO-8859-1 encoding**: `consultas.csv` and the MMV files use ISO-8859-1 (Latin-1) encoding with accented characters. Pandas must be told to use `encoding="latin-1"` or `encoding="iso-8859-1"`. Failure to do this will produce garbled Spanish characters (e.g., `RODOLFO HERN┴NDEZ` instead of `RODOLFO HERNÁNDEZ`).

6. **MOSQUETEROS massive sample**: Row 27 (`Mosqueteros`, `2022-05-19`) has `muestra=6000` but `muestra_int_voto` is `NA`. Ditto for row 35 (runoff). For sample-size weighting, when `muestra_int_voto` is NA, fall back to `muestra`.

---

## 17. Appendix: Glossary

| Term | Definition |
|---|---|
| **Dirichlet-Multinomial (DM)** | A compound distribution where the multinomial's probability vector is drawn from a Dirichlet. Our observation model: poll/shares is multinomial with Dirichlet-distributed probabilities. |
| **Reverse-time random walk** | A random walk anchored at the endpoint (election day) instead of the start. Modelled as `θ[t] ~ Normal(θ[t+1], σ²)`. The uncertainty grows as we move backward. |
| **House effects** | Systematic biases specific to each pollster, e.g., Pollster X routinely overestimates Petro by 2 percentage points. Modeled as additive offsets on the logit scale. |
| **Zero-sum constraint** | The constraint that the sum of all pollster biases for a given candidate equals zero: `Σ_p η[p,k] = 0`. This forces the effects to represent relative biases, not absolute shifts. |
| **Softmax** | A function that maps a vector of real numbers (logits) to a probability simplex: `softmax(x)_i = exp(x_i) / Σ exp(x_j)`. |
| **NUTS** | No-U-Turn Sampler, the default MCMC algorithm in PyMC. An adaptive variant of Hamiltonian Monte Carlo. |
| **R-hat (Gelman-Rubin statistic)** | A convergence diagnostic. R-hat < 1.05 indicates chains have converged. R-hat > 1.10 indicates lack of convergence. |
| **ESS (Effective Sample Size)** | The number of independent draws the MCMC chain effectively represents, accounting for autocorrelation. |
| **HDI (Highest Density Interval)** | The narrowest interval containing a given probability mass of the posterior (e.g., 95% HDI). |
| **MOE** | Misión de Observación Electoral, a Colombian civil society organization that independently tabulates election results. |
| **Registraduría Nacional** | The official Colombian electoral authority responsible for vote counting and certification. |
| **Consultas** | Inter-party primaries/consultations held in March 2022. Three coalitions held binding consultations: Pacto Histórico (left), Centro Esperanza (center), and Equipo por Colombia (right). Winners became the coalition's presidential candidate. |
| **Coalition-to-Candidate mapping** | Translation table from Registraduría/MOE coalition/party names to our internal candidate keys. |
