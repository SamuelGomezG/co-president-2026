# co_president/

**Purpose**: Core Python package for the co-president-2026 Bayesian election forecasting model.
**Category**: source

## Contents (Key)

| Name | Type | Description | SPEC |
|------|------|-------------|------|
| `config.py` | module | Candidate dataclass, ModelConfig, election dates, pollster ratings | SPEC-02 |
| `model_round1.py` | module | Dirichlet-Multinomial + reverse-time RW + house effects (first round) | SPEC-06 |
| `model_runoff_simple.py` | module | K=3 Dirichlet runoff model with drift and rest_blanco override | SPEC-07 |
| `model_runoff_matrix.py` | module | Probabilistic pairing matrix (all pairings, transfer heuristic) | SPEC-08 |
| `model_municipal.py` | module | Hierarchical municipal prior model (demographics → vote shares) | SPEC-22 |
| `model_transfer.py` | module | Bayesian ecological inference for runoff transfer rates | SPEC-30 |
| `model_utils.py` | module | Shared utilities: election day array, candidate ordering | — |
| `data.py` | module | Re-export facade for data_polls + data_results | — |
| `data_polls.py` | module | Poll loading, cleaning, undecided redistribution, deduplication | SPEC-04 |
| `data_results.py` | module | Results consolidation (Registraduría + MOE cross-validation) | SPEC-03 |
| `aggregation.py` | module | Baseline weighted polling averages, time decay, evolution series | SPEC-05 |
| `validation.py` | module | Backtesting, MAE/RMSE/Brier, calibration, rolling forecasts | SPEC-09 |
| `plotting.py` | module | Visualization: evolution plots, calibration, error | SPEC-09 |
| `__main__.py` | module | CLI entry point (8 subcommands: run, aggregate, validate, plot, etc.) | SPEC-10 |
| `_data_quality.py` | module | Pollster accuracy monitoring, time decay validation | SPEC-11 |
| `paths.py` | module | Data directory resolution | SPEC-01 |

## Contents (Secondary)

| Name | Type | Description |
|------|------|-------------|
| `__init__.py` | module | Package version `__version__ = "0.1.0"` |
| `fundamentals/` | directory | MunicipalFeatures dataclass, CLR/logit transforms |
| `ingestion/` | directory | Data acquisition pipelines (Google Trends, census, historical) |
| `validation/` | directory | Extended validation modules (compositional, municipal OOS) |
| `benchmarks/` | directory | ML benchmark suite (FNN+CLR, SVR, RF, GB, KNN) |

## Cross-References

- `docs/architecture/` for design rationale and citations.
- `reference/key-findings.md` for literature support.
- `tests/test_model.py` for model graph and MCMC tests.
- `tests/test_data*.py` for data loading tests.
