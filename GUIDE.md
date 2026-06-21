# co-president-2026 Project Root

**Purpose**: Bayesian presidential election forecaster for Colombia. Predicts 2022 within <1pp MAE (R1) and <0.7pp MAE (runoff). Validated and ready for 2026.
**Category**: root

## Contents

| Name | Type | Description |
|------|------|-------------|
| `src/` | directory | Core Python package (`co_president`) |
| `tests/` | directory | Test suite mirroring `src/` one-to-one |
| `scripts/` | directory | Report generation, backtesting, ingestion runners |
| `docs/` | directory | Architecture, results, data reports, plans, reference |
| `reference/` | directory | Academic PDFs, tech guides, bibliography |
| `data/` | directory | Raw and processed data files |
| `benchmarks/` | directory | Root-level benchmark scripts (ML baselines) |
| `results/` | directory | Generated reports, traces, cached data |
| `pyproject.toml` | config | Project metadata, dependencies, tool config |
| `Makefile` | script | Developer commands (fmt, lint, typecheck, test) |
| `MVP_SPECS_GUIDE.md` | doc | Authoritative technical specification |
| `AGENTS.md` | doc | Agent operating manual |
| `README.md` | doc | Public-facing project documentation |
| `CHANGELOG.md` | doc | Milestone-based changelog |
| `CONTRIBUTING.md` | doc | Contribution guide |
| `.gitignore` | config | Git exclusion rules |
| `.pre-commit-config.yaml` | config | Pre-commit hook configuration |
| `.python-version` | config | Python 3.12 pin |

## Subdirectories

| Name | Purpose |
|------|---------|
| `src/` | `co_president` package — models, data loading, ingestion, benchmarks |
| `tests/` | Pytest test suite, fixtures, integration tests |
| `scripts/` | CLI runners: `run_100k_report.py`, `report_utils.py`, `run_2026_backtest.py` |
| `docs/` | Architecture docs, results, data reports, plans, specs, reference |
| `reference/` | Academic PDFs, tech guides, bibliography, key findings |
| `data/` | Poll data, election results, census data, processed features |
| `benchmarks/` | Root-level ML benchmark scripts |
| `results/` | Generated MCMC traces, reports, trend caches |

## Cross-References

- **Specs**: `MVP_SPECS_GUIDE.md` for SPEC-01→SPEC-11; `docs/specs/STATUS.md` for all SPECs.
- **Architecture**: `docs/architecture/` for design rationale and citations.
- **Results**: `docs/results/` for backtest and forward forecast tables.
- **Data**: `docs/data/` for data source executive reports.
