# ingestion/

**Purpose**: Data acquisition pipelines for the co-president-2026 model.
**Category**: source

## Contents (Key)

| Name | Type | Description | SPEC |
|------|------|-------------|------|
| `ingest_trends.py` | module | Google Trends data: fetch, compute_prop_fav, pytrends integration | SPEC-38 |
| `trends_keywords.py` | module | Candidate-specific query maps: single-name per SciELO 2023 | — |
| `ingest_divipola.py` | module | DIVIPOLA master registry, Socrata API, 1,122 municipalities | SPEC-12 |
| `ingest_historical.py` | module | CEDAE historical results 2002-2018, MOE legislative 2022 | SPEC-12 |
| `ingest_cnpv.py` | module | CNPV 2018 census: demographics, chunked processing | SPEC-17 |
| `ingest_nbi.py` | module | NBI poverty indicators (primary poverty feature) | SPEC-18 |
| `ingest_ipm.py` | module | IPM multidimensional poverty (secondary, ~80% imputed) | SPEC-18 |
| `ingest_population.py` | module | DANE population projections 2018-2026 | SPEC-19 |
| `ingest_risk.py` | module | Electoral risk: MOE, INDEPAZ, PDET, UNODC coca | SPEC-20 |
| `ingest_socioeconomic.py` | module | Socioeconomic demographics (stub fallback) | SPEC-13 |
| `ingest_fiscal.py` | module | Fiscal autonomy indicators: TerriData | SPEC-21a |
| `ingest_bogota.py` | module | Bogotá disaggregation: 21 localidades | SPEC-21b |
| `ingest_sabaneta.py` | module | Sabaneta fixture loader | SPEC-16 |
| `build_feature_matrix.py` | module | Join all components → municipal feature matrix | SPEC-14 |

## Contents (Secondary)

| Name | Type | Description |
|------|------|-------------|
| `__init__.py` | module | Re-exports all ingestion symbols |

## Cross-References

- `data/fundamentals/` for output data files.
- `docs/data/` for data source executive reports.
- `docs/architecture/features.md` for feature documentation.
