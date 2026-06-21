# Census & Demographic Data Sources

All time-invariant — used for both 2022 and 2026 models via `load_features()` → `MunicipalFeatures`.

## DANE CNPV 2018 Census

**Provenance**: Departamento Administrativo Nacional de Estadística (DANE). [https://www.dane.gov.co/](https://www.dane.gov.co/) — Censo Nacional de Población y Vivienda 2018.

**Local files**: `data/fundamentals/cnpv_2018.csv` (our processed output). Raw data from 33 departmental ZIPs at `data/raw/cnpv-2018/` (not tracked).

**Key findings**:
- 1,122 municipalities fully covered (plus 21 Bogotá localidades = 1,143 rows)
- 44.8M total population in 2018, projected to ~52M in 2026
- Internet access rate: national average ~36% in 2018, population-weighted ~80% with soft deflation
- Ethnic composition: ~9% afro-Colombian, ~4% indigenous (highly variable by region)
- Rural dispersion: ~24% of population in rural dispersed areas (key for clientelism/gamonalismo analysis)

**Features extracted**:
| Feature | Computation | Model role |
|---------|-------------|-----------|
| pct_afro_colombian | Z-scored | β_ethnicity (municipal model) |
| pct_indigenous | Z-scored | β_indigenous (municipal model) |
| pct_rural_disperso | Z-scored | β_rural (municipal model) |
| years_schooling_promedio | Z-scored | β_education (municipal model) |
| internet_access_rate | Z-scored | β_internet (municipal model), digital signal weighting |
| pct_age_* | Z-scored | β_age (future) |
| poblacion_total | Population weight | Pop-weighted rollup |

**Ingestion**: `ingest_cnpv.py` → `build_cnpv_features()` → `data/fundamentals/cnpv_2018.csv`.

## DANE NBI Poverty (Primary)

**Provenance**: DANE. Necesidades Básicas Insatisfechas (NBI) from CNPV 2018. Complete municipal coverage — **primary poverty feature**.

**Local files**: `data/fundamentals/nbi_2018.csv` (our processed output). Raw XLSX at `data/raw/DANE-NBI/CNPV-2018-NBI.xlsx` (not tracked).

**Key findings**:
- **0% imputation**: all 1,122 municipalities have NBI values (unlike IPM which is ~80% imputed)
- Range: ~0.03 (Bogotá) to ~0.80+ (rural Chocó, Guainía)
- NBI in rural areas is systematically higher than urban (2-5x) — captures structural poverty that IPM misses
- CLR-transformed NBI is used in the municipal model's logit predictor

**Features extracted**: `nbi_rate` → `clr_nbi = log(nbi / (1-nbi))` → Z-scored → `β_poverty`.

## DANE IPM (Secondary)

**Provenance**: DANE. Índice de Pobreza Multidimensional de fuente censal. Only ~33 cabeceras municipales have direct measurements — remainder imputed.

**Local files**: `data/fundamentals/ipm_2018.csv` (our processed output). Not tracked in git.

**Key findings**:
- ~80% of municipalities are imputed (IPM values estimated from regression model on 33 cabeceras)
- R² guard: IPM column excluded from feature matrix if imputation R² < 0.5
- Secondary to NBI — only used when explicit `--component ipm` flag is passed

**Features extracted**: `ipm_2018`, `ipm_2018_imputed` (boolean flag).

## DANE Population Projections

**Provenance**: DANE. PPED (Proyecciones de Población) 2018-2042 by municipality and area type.

**Local files**: `data/fundamentals/population_2018_2026.csv` (our processed output). Raw XLSX at `data/raw/PPED-AreaMun-2018-2042_VP.xlsx` (not tracked).

**Key findings**:
- Extends to 2026 — enables 2026 turnout-weighted rollup
- Used to compute `effective_pop = pop * turnout / mean(turnout)` in municipal model
- Population growth concentrated in major cities (Bogotá, Medellín, Cali); rural municipalities are stable or declining

**Features extracted**: `pop_2022` (current), `pop_2026` (for 2026 mode). Also: `pop_2018`–`pop_2025` for historical alignment.

---

## References

See `docs/reference/key-findings.md` §1 (Compositional Data) and §5 (Municipal Model). Full features documented in `docs/architecture/features.md` §4 (Municipal Hierarchical Model Features).
