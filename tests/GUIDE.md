# tests/

**Purpose**: Pytest test suite mirroring `src/` one-to-one. Model tests use a four-phase strategy: graph test → prior predictive → convergence → sanity.
**Category**: test

## Contents (Key)

| Name | Type | Description |
|------|------|-------------|
| `test_model.py` | module | Model graph tests, prior predictive, convergence, sanity |
| `test_config.py` | module | Candidate, ModelConfig, constants, formulas |
| `test_data_polls.py` | module | Poll loading, cleaning, undecided redistribution |
| `test_data_results.py` | module | Results consolidation, cross-validation |
| `test_aggregation.py` | module | Weighted averages, time decay, evolution series |
| `test_validation.py` | module | Backtesting metrics, CI coverage, forecast dataclasses |
| `test_data.py` | module | Data loading facade tests |
| `test_model_integration.py` | module | End-to-end model integration tests |
| `test_ingestion_*.py` | module | Ingestion pipeline tests (one per pipeline) |
| `test_fundamentals_features.py` | module | MunicipalFeatures dataclass, CLR transforms |
| `test_benchmarks.py` | module | ML benchmark tests |

## Contents (Secondary)

| Name | Type | Description |
|------|------|-------------|
| `conftest.py` | module | Shared fixtures: data_dir, sample DataFrames |
| `__init__.py` | module | Package init |

## Subdirectories

| Name | Purpose |
|------|---------|
| `fixtures/` | Test fixtures: synthetic data, mini DataFrames |
| `integration/` | Integration tests: end-to-end pipeline |

## Cross-References

- `src/` mirrors this directory one-to-one.
- `MVP_SPECS_GUIDE.md` §3 for TDD methodology.
