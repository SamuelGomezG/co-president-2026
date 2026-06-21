# scripts/

**Purpose**: CLI runners for backtesting, reports, and 2026 forecast.
**Category**: script

## Contents

| Name | Type | Description |
|------|------|-------------|
| `run_100k_report.py` | module | Full 5000×4 backtest pipeline: R1 → runoff → report |
| `forward_backtest.py` | module | True forward backtest: both rounds without election results |
| `run_2026_backtest.py` | module | 2026 backtest with updated dates and candidate config |
| `report_utils.py` | module | Shared utilities: survey runners, accuracy computation, report generation |
| `download_cnpv_2018.py` | module | CNPV 2018 census download script |
| `plot_validation.py` | module | Validation plot generation |
| `run_baseline.py` | module | Baseline weighted-averaging runner |

## Cross-References

- `docs/results/` for generated reports.
- `results/` directory for output files.
