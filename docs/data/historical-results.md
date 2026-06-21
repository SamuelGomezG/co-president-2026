# Historical Election Results Data

## CEDAE Historical Results (2002-2018)

**Provenance**: CEDAE (Centro de Datos Electoral) — electoral results database. [https://cedae.datasketch.co/](https://cedae.datasketch.co/)

**Local files**: `data/raw/cedae/` — 15 compressed `.dta.csv.gz` files covering 2002, 2006, 2010, 2014, 2018. Not tracked in git.

**Coverage**: Presidential R1/R2 + Cámara de Representantes + Senado for each election year (2002-2018).

**Key findings**:
- Municipal-level vote shares for 6 election cycles (2002-2018)
- Used to compute CLR-transformed historical left-share for each municipality
- 2018 is the most comparable to 2022: Fajardo (center) was eliminated and his voters split ~53/47 between Petro and Duque — this split directly informs the SPEC-30 transfer model's prior for Fajardo→X rates in 2022
- Turnout patterns are municipality-specific and stable across years (used for effective_pop computation)

**Features extracted**:
| Feature | Computation | Model role |
|---------|-------------|-----------|
| clr_left_share | CLR of historical left-vote share per muni | β_historical (municipal model) |
| historical_turnout_m | Mean turnout per muni across years | Effective pop weight |
| left_r1, left_r2, elim_r1 | R1/R2 per muni per year | Transfer model target variables |
| camara_left_share, senado_left_share | Legislative left-vote share | Transfer model covariates |

**Ingestion**: `ingest_historical.py` → `build_historical_matrix()` → `data/fundamentals/historical_results.csv`.

## MOE Legislative 2022

**Provenance**: MOE. Cámara de Representantes and Senado results at the municipal level for 2022.

**Local files**: `data/raw/MOE-2022-legislativas/moe_camara_territorial_2022.csv` (~13K rows), `moe_senado_nacional_2022.csv` (~20K rows). Not tracked in git.

**Key findings**:
- Closes the 2022 legislative data gap (CEDAE does not have 2022 legislative files)
- Cámara: department-level representation — captures local machine politics (gamonalismo)
- Senado: national at-large — captures ideological alignment

**Features extracted**: camara_left_share, senado_left_share — used as features in SPEC-30 transfer rate estimation.

## Sabaneta Cámara Fixture

**Provenance**: Sabaneta municipality (Antioquia) — publicly published Cámara de Representantes results 2002-2022.

**Local files**: `data/sabaneta/Resultados_de_elecciones_para_Cámara...csv` (not tracked in git). Processed version at `data/fundamentals/sabaneta_camara.csv`.

**Key findings**:
- Gold-standard verification fixture for the data pipeline: 4 electoral periods, 16 parties, consistent schema
- Confirms that the CEDAE-to-canonical mapping works correctly on real data

## MOE Censo Electoral

**Provenance**: MOE. Historical census-level electoral data for multiple election cycles.

**Local files**: `data/moe_censo_electoral/` — 14 CSVs covering 2006-2023. Not tracked in git.

**Coverage**: Congressional, local, presidential, and plebiscite elections. Used for cross-validation of CEDAE totals.

---

## Transfer Model Training

The SPEC-30 transfer model trains on 2010/2014/2018 data (2022 excluded via `exclude_year=2022`). For each eliminated candidate, the model estimates per-municipality transfer rates to the left-runoff candidate using:

- Historical R1/R2 vote shares per municipality (from CEDAE)
- Demographic covariates (from CNPV 2018 census)
- Legislative shares — Cámara (local machine) and Senado (national ideology) — from MOE 2022

**References**: See `docs/reference/key-findings.md` §6 (Transfer Rate Estimation). Full citations in `docs/reference/bibliography.md` `[King1997]`, `[USANTOMAS2025]`.
