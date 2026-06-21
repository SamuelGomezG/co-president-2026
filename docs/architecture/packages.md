# Package Organization

## Package: `co_president`

### Entry Point
- `__init__.py` — package version
- `__main__.py` — CLI entry point (8 subcommands: run, aggregate, validate, plot, ingest, forecast, config)

### Configuration & Data
- `config.py` — Candidate dataclass, ModelConfig, election dates, pollster ratings, consultation votes
- `data.py` — Re-export facade for data_polls + data_results
- `data_polls.py` — Poll loading, cleaning, undecided redistribution, deduplication
- `data_results.py` — Results consolidation (Registraduría + MOE cross-validation)
- `paths.py` — Data directory resolution

### Aggregation Baseline (SPEC-05)
- `aggregation.py` — Weighted polling averages, time decay, evolution series

### Model Stage 1: First Round (SPEC-06)
- `model_round1.py` — Dirichlet-Multinomial + reverse-time RW + house effects + election likelihood + digital signals + municipal prior
- `model_municipal.py` — Hierarchical municipal prior (demographics → vote shares)
- `model_utils.py` — Shared utilities (election day extraction, candidate ordering)

### Model Stage 2: Transfer Model (SPEC-30)
- `model_transfer.py` — Bayesian ecological inference model for runoff transfer rates
  - Trains on 2010/2014/2018 historical R1→R2 + municipal demographics + legislative shares
  - Produces per-candidate transfer rate posteriors

### Model Stage 3: Runoff (SPEC-07/08)
- `model_runoff_simple.py` — K=3 Dirichlet runoff model with drift
- `model_runoff_matrix.py` — Probabilistic pairing matrix (identifies most likely pairings from R1 posterior)

### Integration & Validation (SPEC-22/26)
- `fundamentals/features.py` — MunicipalFeatures dataclass, CLR/logit transforms, feature loading
- `validation.py` — Backtesting, MAE/RMSE/Brier, calibration, rolling forecasts
- `validation/municipal_oos.py` — OOS validation framework (leave-one-year-out)
- `validation/compositional.py` — Compositional data validation (Aitchison distance)
- `validation/__init__.py` — Extended validation modules
- `plotting.py` — Visualization (evolution plots, calibration, error)
- `_data_quality.py` — Pollster accuracy monitoring, time decay validation

### Ingestion Pipeline
- `ingestion/__init__.py` — Re-exports all ingestion symbols
- `ingestion/ingest_trends.py` — Google Trends data (pytrends + prop_fav computation)
- `ingestion/trends_keywords.py` — Candidate-specific query maps
- `ingestion/ingest_divipola.py` — DIVIPOLA master registry
- `ingestion/ingest_historical.py` — Historical results (CEDAE)
- `ingestion/ingest_cnpv.py` — CNPV 2018 census
- `ingestion/ingest_nbi.py` — NBI poverty indicators
- `ingestion/ingest_ipm.py` — IPM poverty indicators
- `ingestion/ingest_population.py` — Population projections
- `ingestion/ingest_risk.py` — Electoral risk indicators
- `ingestion/ingest_socioeconomic.py` — Socioeconomic demographics
- `ingestion/ingest_fiscal.py` — Fiscal autonomy indicators
- `ingestion/ingest_bogota.py` — Bogotá localidad disaggregation
- `ingestion/ingest_sabaneta.py` — Sabaneta fixture
- `ingestion/build_feature_matrix.py` — Join all components into feature matrix

### Benchmarks (SPEC-37/41)
- `benchmarks/transforms.py` — Compositional ALR/ILR/CLR transforms
- `benchmarks/fnn_clr.py` — FNN+CLR ML benchmark
- `benchmarks/runner.py` — Benchmark runner (SVR/RF/GB/KNN)

## Why Separate Model Files?

`model_round1.py` and `model_municipal.py` are separate because the municipal model uses a 3-layer hierarchical structure with 1,122+ parameters. Putting them in one file would make the file unmanageable and create import dependencies.

`model_runoff_simple.py` and `model_runoff_matrix.py` are separate because the simple model handles one specific pairing, while the matrix handles all possible pairings probabilistically.

`model_transfer.py` is separate because it's a fundamentally different model type (ecological inference / Beta-binomial, not Dirichlet-Multinomial).

## Tests

Tests mirror `src/` one-to-one:

| src/ module | test file |
|-------------|-----------|
| `config.py` | `test_config.py` |
| `data.py` / `data_polls.py` / `data_results.py` | `test_data*.py` |
| `aggregation.py` | `test_aggregation.py` |
| `model_round1.py` / `model_runoff_simple.py` / `model_runoff_matrix.py` | `test_model.py` |
| `model_municipal.py` | `test_model.py` |
| `model_transfer.py` | `test_model.py` |
| `validation.py` / `plotting.py` | `test_validation.py` |
| `model_runoff_matrix.py` | `test_validation.py` |
| `fundamentals/features.py` | `test_fundamentals_features.py` |
| `ingestion/*.py` | `test_ingestion_*.py` |
| `benchmarks/*.py` | `test_benchmarks.py` |

All MCMC tests follow a four-phase strategy: graph test → prior predictive test → convergence test (@pytest.mark.slow) → sanity test (@pytest.mark.slow).
