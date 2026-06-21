# Data Source Reports

This directory contains executive reports for every data source used by the co-president-2026 model. Each report covers provenance, key findings, features extracted, and the ingestion pipeline.

## Reports

| File | Content | Years |
|------|---------|-------|
| `polls.md` | Presidential polls: encuestas_2022.csv, consultas.csv, AS/COA, CNE microdata, La Silla Vacía | 2022, 2026 |
| `election-results.md` | Official results: Registraduría MMV, MOE, participation | 2022, 2026 |
| `census-demographics.md` | DANE census data: CNPV 2018, NBI, IPM, population | Both (time-invariant) |
| `historical-results.md` | Historical: CEDAE 2002-2018, MOE legislative 2022, Sabaneta | 2002-2022 |
| `risk-and-conflict.md` | MOE risk, PDET, INDEPAZ, UNODC | Both (updated 2026) |
| `digital-signals.md` | Google Trends via pytrends | 2022, 2026 |
| `geographic-reference.md` | DIVIPOLA registry, Bogotá localidades | Both (time-invariant) |
| `2026-candidate-config.md` | 2026 timeline, pollster ratings, differences from 2022 | 2026 only |

## Cross-References

- **Model features**: `docs/architecture/features.md` catalogs every feature from these data sources.
- **Literature support**: `docs/reference/key-findings.md` discusses key findings from data analysis.
- **Citations**: `docs/reference/bibliography.md` has full citations for every data source.

## Data Tracking Policy

External raw data files (Registraduría MMV, MOE CSVs, CEDAE historical, recetas-electorales polls, AS/COA, CNE microdata) are **not tracked in git**. They are documented here with download/access instructions. Our processed/derived files (`data/fundamentals/`, `data/processed/`) are tracked as our work.
