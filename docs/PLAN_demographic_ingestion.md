# Plan: Clean, ingest, and integrate the demographic data in `data/`

**Author**: planning agent
**Date**: 2026-06-02 (updated after user review + external reviewer feedback)
**Branch baseline**: `dev`
**Reference docs**: `AGENTS.md`, `MVP_SPECS_GUIDE.md`, `ENHANCEMENT_ROADMAP.md`
**Critical constraints from the user**:
- (A) The model must be usable for **predicting 2026**. **Overfitting to 2022 is unacceptable.**
- (B) The current polls are national-aggregate; the model must **fit at the municipal level** by combining national polls with municipal demographics and historical results.
**Reviewer feedback incorporated**:
- NBI (Necesidades Básicas Insatisfechas) is the **primary poverty feature** (complete municipal coverage, zero imputation). IPM is secondary.
- SPEC-21 (Fundamentals API) moved to Day 2 so every ingestion script validates against the data contract from the start.
- SPEC-30 transfer model has an explicit convergence guardrail with a Dirichlet-Categorical historical fallback.
- §9.6 documents the 2026 candidate mapping strategy (candidate-agnostic feature space, no subjective labels).

---

## 0. Audit of what is already in the repo (so the plan does not duplicate work)

| File | Status | Notes |
|------|--------|-------|
| `data/fundamentals/divipola_master.csv` | ✅ Real | 1 123 municipalities, columns `REGION, CÓDIGO DANE…, departamento, codigo_municipio, nombre_municipio` |
| `data/fundamentals/historical_results.csv` | ⚠️ Incomplete | 25 708 rows, all `year=2022` and `codigo_municipio=000NA` (placeholder) — see §3.2 |
| `data/fundamentals/socioeconomic.csv` | 🟡 Stub | 3 rows (Bogotá/Medellín/Cali) — ingested by hardcoded fallback in `ingest_socioeconomic.py:439` |
| `data/fundamentals/risk_factors.csv` | 🟡 Stub | 10 rows — hardcoded fallback in `ingest_risk.py:361-612` |
| `data/processed/municipal_feature_matrix.{csv,parquet}` | 🟡 Built but mostly null | 1 123 × 50, 44 % null rate |
| `data/sabaneta/Resultados_…_20260602.csv` | 🆕 Unused | Cámara de Representantes 2002–2022 for Sabaneta, Antioquia |
| `data/raw/cnpv-2018/raw/*.zip` | 🆕 Unused | 33 departmental zips (~2.5 GB) of CNPV 2018 microdata |
| `data/raw/IPM-2018/`, `IPM-2022/`, `IPM-2025/` | 🆕 Unused | Real DANE Multidimensional Poverty (secondary; only ~33 cabeceras) |
| `data/raw/DANE-NBI/CNPV-2018-NBI.xlsx` + ethnic/DIVIPOLA breakouts | 🆕 Unused | DANE Necesidades Básicas Insatisfechas — **complete municipal coverage** from CNPV 2018 (443 KB main file + 4 supporting XLSX + PDF justification). This is the primary poverty feature per reviewer recommendation. |
| `data/raw/DANE-DIMPE-ECP-{2011…2023}/` | 🆕 Unused | Encuesta de Cultura Política |
| `data/raw/ECV-{2022,2025}/` | 🆕 Unused | Encuesta de Calidad de Vida |
| `data/raw/GEIH-2026/` | 🆕 Unused | Gran Encuesta Integrada de Hogares — 2026 only |
| `data/raw/MPMD-{2022,2024}/` | 🆕 Unused | Mercado laboral / pobreza monetaria |
| `data/raw/MESEP-2002-2010/` | 🆕 Unused | Pobreza monetaria 2002–2010 |
| `data/raw/Lineas-pobreza-2012-2018/` | 🆕 Unused | Líneas de pobreza DANE |
| `data/raw/cedae/2018_*vuelta.dta.csv.gz` + 2002/06/10/14 Camara/Senado | 🆕 Downloaded but not loaded by `compute_lagged_features` | 15 compressed files; schema: `id_electoral, ano, tipo_eleccion, fecha_eleccion, coddpto, departamento, codmpio, municipio, circunscripcion, codigo_partido, primer_apellido, segundo_apellido, nombres, votos, curules`. Legislative data covers 2002–2018 for both chambers. |
| `data/raw/cedae/` — **2022 legislative gap** | ⚠️ Missing | CEDAE does not have 2022 Camara/Senado files. Gap filled by MOE below. |
| `data/raw/MOE-2022-legislativas/moe_camara_territorial_2022.csv` | 🆕 ✅ Real | 13 237 municipal-level Cámara 2022 results. Schema: `annoh, corp, codcirc, coddepto, codmpio, nommun, nomdepto, codparti, nomparti, eleccion, votos, censo, color, partido`. **Closes the 2022 legislative gap for SPEC-30.** |
| `data/raw/MOE-2022-legislativas/moe_senado_nacional_2022.csv` | 🆕 ✅ Real | 19 931 municipal-level Senado 2022 results. Same schema as Cámara. |
| `data/raw/PPED-AreaMun-2018-2042_VP.xlsx` | 🆕 Unused | DANE population projections to 2042 (Area Municipal) |
| `data/raw/barometro_americas/2006/2010/2014/2018/2021 CSV + 2023 PDF` | 🆕 Unused | LAPOP |
| `data/raw/latinobarometro/` | 🆕 Unused | Latinobarómetro |
| `data/2022-presidential-results/2022.11.09-LIBRO-RESULTADOS-ELECTORALES-PRESIDENCIALES-2022.pdf` | 🆕 Unused | MOE libro oficial (has image-only pages) |
| `data/2022-polls/as_coa/extract.py` + CSV/JSON | 🆕 Unused | AS/COA's processing of 2022 polls for both rounds — **third poll source** |

The MVP **does not consume any of the new data**. `model_round1.py:38-100` and `data_polls.py` operate only on national polls. `build_feature_matrix.py` produces a 50-column matrix but **nothing downstream reads it**. The `processed/` matrix is a dead artifact.

---

## 1. Goals (what "done" means)

1. Every raw demographic dataset the user added is either (a) ingested into `data/fundamentals/*.csv` with all 1 122 municipalities covered, or (b) explicitly documented as out-of-scope with a justification entry in `data/fundamentals/COVERAGE.md`.
2. `data/processed/municipal_feature_matrix.{csv,parquet}` has **< 1 % null rate in non-census columns** and **< 20 % null rate in census columns**.
3. A **new, tested** `co_president.fundamentals` module exposes typed `MunicipalFeatures` (one row per municipality, immutable) and a `load_features()` entry point.
4. The model is extended so that national polls can be **fit at the municipal level**: a Dirichlet-Multinomial observation model at the municipality with a **national-poll-based likelihood at the rollup**. The MVP "polling-only national" mode still passes all SPEC-06 to SPEC-10 tests.
5. **Overfitting is explicitly prevented and verified out-of-sample.** The plan includes a 2018-held-out validation that trains on 2002/2006/2010/2014 and predicts 2018, then compares to the actual 2018 result. A simpler "2026 mode" uses minimal polling + maximum fundamentals.
6. The Sabaneta gold fixture is used to verify the loader path end-to-end, and the 2022 MOE PDF is parsed (with `pdfplumber` + `pymupdf` fallback for image-only pages) as a third source of truth.
7. The AS/COA processed 2022 polls (round1 + runoff CSVs) are integrated as a third poll source. The transfer matrix is used as a **validation target** for a new transfer rate estimation model (SPEC-30) that infers coefficients from all available historical data.
8. Population projections extend through **2026**.
9. Transfer rates (eliminated candidate → runoff candidate) are **estimated from data** via SPEC-30, not hand-tuned. The Cámara de Representantes AND Senado legislative data (2002–2022) are first-class inputs: Cámara captures local party machinery (gamonalismo), Senado captures national ideological alignment.

---

## 2. Branch & SPEC allocation (follows §2 + §14 conventions in AGENTS.md)

All new work goes on **feature branches forked from `dev`**. Each SPEC becomes one PR against `dev` after a successful `cr` (CodeRabbit) review per AGENTS.md §9.

### 2.1 Critical path (must ship for "demographics integrated")

| New SPEC | Branch | Depends on | What ships |
|----------|--------|-----------|------------|
| **SPEC-15** Demographic data audit & matrix handoff | `feat/spec-15-demographic-audit` | none | `data/fundamentals/COVERAGE.md`, replaces stubs with real data, fixes `historical_results.csv` placeholder, strict-mode validator |
| **SPEC-16** Sabaneta Cámara loader | `feat/spec-16-sabaneta` | none | `src/co_president/ingestion/ingest_sabaneta.py` + tests + sanity-check fixture |
| **SPEC-17** CNPV 2018 census ingest | `feat/spec-17-cnpv-2018` | SPEC-15 | `src/co_president/ingestion/ingest_cnpv.py` + per-dept unzip pipeline |
| **SPEC-18** IPM poverty ingest | `feat/spec-18-ipm` | SPEC-15 | Replaces stub `ipm_score` / `nbi_rate`; **explicit imputation flags** |
| **SPEC-19** DANE population projections ingest (2018–2026) | `feat/spec-19-population` | SPEC-15 | Replaces stub `proyeccion_2022`; extends to 2026 |
| **SPEC-20** Extended risk ingest (INDEPAZ/PDET/UNODC from `raw/`) | `feat/spec-20-risk-extended` | SPEC-15 | Risk table goes from 10 → 170 PDET + ~200 INDEPAZ + UNODC top |
| **SPEC-21** Municipal features API (`co_president.fundamentals`) | `feat/spec-21-fundamentals-api` | SPEC-15–20 | Typed `MunicipalFeatures` dataclass + `load_features()` |
| **SPEC-25** AS/COA poll source integration | `feat/spec-25-as-coa` | none | Third poll source (round1/runoff CSVs only); transfer_matrix.csv used as **validation target** for SPEC-30, NOT as source of truth |
| **SPEC-30** Transfer rate estimation model | `feat/spec-30-transfer-model` | SPEC-21, SPEC-15 | New PyMC submodel that infers runoff transfer rates from all historical R1→R2 + Cámara legislative + demographics; supersedes hand-tuned `TRANSFER_*` constants in `config.py` |
| **SPEC-22** Municipal hierarchical model (national polls → municipal posterior) | `feat/spec-22-municipal-prior` | SPEC-21, 25, 30; existing SPEC-06 | New `model_municipal.py` (separate from `model_round1.py`); `fundamentals_mode` flag on `ModelConfig` |
| **SPEC-26** Out-of-sample validation framework | `feat/spec-26-oos-validation` | SPEC-22 | Leave-one-year-out CV; 2018-holdout test; **gating test** that fails the build if R² < 0.3 with a "missing data" report; WAIC/LOO comparison |
| **SPEC-27** Regularization & shrinkage control | `feat/spec-27-regularization` | SPEC-22, 26 | Horseshoe priors, informative `β` priors, sensitivity ablation |

### 2.2 Auxiliary (valuable, separable)

| New SPEC | Branch | What ships |
|----------|--------|-----------|
| **SPEC-23** 2022 MOE PDF cross-validation | `feat/spec-23-pdf-xvalidate` | `data_results.py` gains `cross_validate_against_moe_pdf()` (uses `pdfplumber` + `pymupdf`) |
| **SPEC-24** ECP cultural attitudinal signal | `feat/spec-24-ecp` | Optional non-modular feature: ECP aggregated to departmental panel |
| **SPEC-28** 2026 forecast mode | `feat/spec-28-forecast-2026` | CLI `--year 2026` flag, candidate-set swap, sparse-poll priors |
| **SPEC-29** Generalization audit | `feat/spec-29-generalization` | Year-2018-out ablation; effective degrees of freedom; calibration plots |

---

## 3. Phase 0 — Audit & cleanup (`SPEC-15`)

### 3.1 `data/fundamentals/COVERAGE.md` (new)

Single source of truth for what is and isn't ingested. Sections:

```markdown
# Fundamentals Coverage
| Source | File | Rows expected | Rows in fundamentals | Source of values | Last verified |
| -- | -- | -- | -- | -- | -- |
| DIVIPOLA | divipola_master.csv | 1122 | 1123 | datos.gov.co Socrata | 2026-06-01 |
| Historical results | historical_results.csv | 1122 × 6 years × 2 rounds | … | CEDAE 2002–2022 | … |
| Socioeconomic | socioeconomic.csv | 1122 | 3 (stub) | local IPM-2018/2022 | … |
| Risk | risk_factors.csv | 1122 | 10 (stub) | local INDEPAZ/PDET/UNODC | … |
| … |
```

Generated by a new function `ingestion.report.coverage(data_dir: Path) -> DataFrame`.

### 3.2 Fix the broken `historical_results.csv`

The current file (25 708 rows) has `codigo_municipio == "000NA"` for every row. Replace the remote CEDAE call in `ingest_historical.py:130-152` with a local-file reader. The 15 `.dta.csv.gz` files under `data/raw/cedae/` already cover **2002, 2006, 2010, 2014, 2018** for Camara + Presidencia + Senado, plus 2018 segunda vuelta. **2022 legislative data (Cámara + Senado) comes from MOE** (`data/raw/MOE-2022-legislativas/moe_camara_territorial_2022.csv` and `moe_senado_nacional_2022.csv`, 13K + 20K rows at municipal level). **Full 2002–2022 legislative coverage for both chambers is now available** for SPEC-30's transfer rate estimation model.

The MOE legislative schema maps to CEDAE columns as:

| MOE column | CEDAE column | Canonical |
|--|--|--|
| `codmpio` | `codmpio` | `codigo_municipio` (zero-padded) |
| `nomparti` | `codigo_partido` | `codigo_partido` (needs crosswalk) |
| `votos` | `votos` | `votos` |

A `map_moe_party_to_canonical()` function in SPEC-30 handles the MOE→CEDAE party code crosswalk.

Add to `_download_cedae.py:53-70`:

```python
def fetch_local_cedae_results(year: int, nivel: str) -> pd.DataFrame:
    """Read a CEDAE file from data/raw/cedae/ (offline path)."""
    pattern = f"{year}_{nivel.lower()}"
    matches = list((resolve_data_dir(None) / "raw" / "cedae").glob(f"{pattern}*"))
    if not matches:
        raise FileNotFoundError(f"No local CEDAE file for {year} {nivel}")
    return pd.read_csv(matches[0], compression="gzip", encoding="latin-1",
                       dtype={"coddpto": str, "codmpio": str})
```

`compute_derived_features` then only needs `coddpto + codmpio` → `codigo_municipio` (zero-padded to 5 digits), `votos`, and the new schema columns. Extend `compute_lagged_features` to handle the new raw column names.

### 3.3 `validate_component_health` strict mode

`build_feature_matrix.py:86-138` warns but never aborts. After SPEC-15 it should **raise** when a stub is detected:

- `socioeconomic.csv` has ≤ 5 non-null rows ⇒ `ValueError("socioeconomic.csv is a stub")`
- `risk_factors.csv` has ≤ 15 non-null rows ⇒ `ValueError("risk_factors.csv is a stub")`
- `historical_results.csv` has any `codigo_municipio == "000NA"` ⇒ `ValueError("historical_results.csv is a placeholder")`

This prevents future agents from committing a half-populated matrix.

### 3.4 `data/2022-polls/as_coa/` — third poll source

`data/2022-polls/as_coa/extract.py` is AS/COA's processing of the 2022 round-1 + runoff poll set, materialized into three CSVs:

- `round1.csv` — 1st-round national shares (third poll source)
- `runoff.csv` — 2nd-round national shares (third poll source)
- `transfer_matrix.csv` — AS/COA-empirical voter transfer rates (useful, but **NOT the source of truth** for the model)

Per the user's answer to Q4, **transfer rates are to be estimated from data** (historical presidential R1→R2 + Cámara legislative + demographics), not taken from this file. The `transfer_matrix.csv` is kept as a **validation target** for the new transfer rate model (SPEC-30): compare the model's estimated transfer coefficients against AS/COA's empirical rates, and flag any divergence > 10 pp as a WARNING.

**SPEC-25** is small: a new `data_polls.py` loader `load_as_coa_polls() -> CleanPolls` (third poll source, round1 and runoff CSVs only). The transfer matrix is loaded separately by SPEC-30 for validation.

---

## 4. Phase 1 — Replace the stubs (SPEC-17 / 18 / 19 / 20)

Each SPEC follows the same skeleton; I write 17 in full and outline 18/19/20.

### 4.1 SPEC-17: CNPV 2018 census ingest (the biggest deliverable)

**Module**: `src/co_president/ingestion/ingest_cnpv.py`

**Inputs**: 33 zips in `data/cnpv-2018/raw/`. Each zip contains a nested `_<dept>_<dept>_CSV.zip` with the actual tabular CSV(s). Reference `data/cnpv-2018/docs/CNPV-2018-data-dictionary.ddi.xml` (and `CNPV-2018-documentation.pdf`) for variable definitions.

**Approach** (chunked, never load the full CSV in memory):

```python
def build_cnpv_features(data_dir: Path) -> None:
    """Aggregate CNPV 2018 to municipal demographics; save fundamentals/cnpv_2018.csv."""
    per_dept = []
    for zip_path in (data_dir / "raw" / "cnpv-2018" / "raw").glob("*.zip"):
        with zipfile.ZipFile(zip_path) as outer:
            inner_name = next(n for n in outer.namelist() if n.endswith("_CSV.zip"))
            with zipfile.ZipFile(BytesIO(outer.read(inner_name))) as inner:
                csv_name = next(n for n in inner.namelist() if n.endswith(".csv"))
                with inner.open(csv_name) as f:
                    for chunk in pd.read_csv(
                        f, sep=";", encoding="latin-1",
                        usecols=COLUMNS_OF_INTEREST,
                        dtype={"U_DPTO": str, "U_MPIO": str, …},
                        chunksize=200_000,
                    ):
                        per_dept.append(_aggregate_chunk(chunk))
    df = pd.concat(per_dept).groupby("codigo_municipio").sum()
    df.to_csv(data_dir / "fundamentals" / "cnpv_2018.csv", index=True)
```

**Columns to extract from CNPV 2018**:

| CNPV variable | Meaning | Output column |
|--|--|--|
| `U_DPTO` + `U_MPIO` | 2-digit dpto + 3-digit mpio → 5-digit code | `codigo_municipio` |
| `PA1_GRP_ETNIC` | self-identified ethnic group (1=indigenous, 2=afro, 3=ninguno) | `pct_indigenous`, `pct_afro_colombian` |
| `PA_CLASE` | area class (1=cabecera, 2=centro poblado, 3=rural disperso) | `pct_rural_disperso` |
| `P_NIVEL_ANOSR` | years of schooling recoded | `years_schooling_promedio` |
| `H_VIVIENDA_INTERNET` | household has internet | `internet_access_rate` |
| `P_EDADR` | age group | `pct_age_18_29`, `pct_age_30_54`, `pct_age_55_plus` |
| `PA_ASISTENCIA` | school attendance | `pct_school_attendance` |
| `P_TRABAJO` | labor force status | `labor_force_participation_rate` |
| `P_SEXO` | sex | `pct_female` |
| `H_NRO_DORMIT` | rooms in dwelling | `rooms_per_household` |
| `H_NRO_PER` | persons per household | `persons_per_household` |

**Schema produced** (`data/fundamentals/cnpv_2018.csv`):

```text
codigo_municipio,poblacion_cnpv,poblacion_afrocolombiana,poblacion_indigena,poblacion_rural_dispersa,
pct_afro_colombian,pct_indigenous,pct_rural_disperso,years_schooling_promedio,
pct_school_attendance,internet_access_rate,labor_force_participation_rate,pct_female,
rooms_per_household,persons_per_household,
pct_age_18_29,pct_age_30_54,pct_age_55_plus
```

**Tests** (`tests/test_ingestion_cnpv.py`):

```python
def test_cnpv_column_extraction_is_lossless():
    chunk = _synthetic_cnpv_chunk(rows=1000)
    out = _aggregate_chunk(chunk)
    assert out["poblacion_total"].sum() == 1000

def test_cnpv_known_municipality_totals():
    df = pd.read_csv(_CACHED_BUILD_FIXTURE)
    assert df.loc[df.codigo_municipio == "11001", "poblacion_total"].iloc[0] > 7_000_000

def test_cnpv_uses_zero_padded_dane_codes():
    chunk = _synthetic_cnpv_chunk(u_dpto="5", u_mpio="001")
    assert _aggregate_chunk(chunk)["codigo_municipio"].iloc[0] == "05001"

def test_cnpv_handles_unknown_ethnicity_as_zero():
    # PA1_GRP_ETNIC=99 (no response) is dropped, not counted as indigenous
    …
```

**TDD order** (RED → GREEN → REFACTOR for each):

1. `_aggregate_chunk(chunk) -> Series` — pure function, no I/O
2. `_zero_pad_dane_code(u_dpto, u_mpio) -> str`
3. `iter_cnpv_zip(zip_path) -> Iterator[DataFrame]` — streams chunks
4. `build_cnpv_features(data_dir)` — orchestration, writes CSV

**Performance**: 33 zips × ~80 MB each average → ~2.5 GB total. Chunked read keeps RAM < 500 MB. Single-pass aggregation; runtime ~6 min on a laptop. Add a `make cnpv` Makefile target that caches the output and skips when up to date.

**Wire into build_feature_matrix**: extend `_COMPONENT_FILES` with `"cnpv": "cnpv_2018.csv"`.

### 4.2 SPEC-18: Poverty indicators ingest (NBI-primary, IPM-secondary)

**Reviewer recommendation accepted.** The original plan used IPM as the primary poverty feature, but IPM is only published for ~33 cabeceras and would require imputing ~80 % of the dataset from a model trained on 33 urban hubs — which fails in rural conflict zones where the census-to-poverty relationship is structurally different. **NBI (Necesidades Básicas Insatisfechas)** has complete census-level municipal granularity from DANE and requires zero imputation.

**NBI module**: `src/co_president/ingestion/ingest_nbi.py`

```python
def build_nbi_features(data_dir: Path) -> pd.DataFrame:
    """Read CNPV-2018-NBI.xlsx; return one row per municipality."""
    df = pd.read_excel(
        data_dir / "raw" / "DANE-NBI" / "CNPV-2018-NBI.xlsx",
        sheet_name="Hoja1",  # verify sheet name during implementation
    )
    df["codigo_municipio"] = df["codigo_municipio"].astype(str).str.zfill(5)
    return df[["codigo_municipio", "nbi_rate", "nbi_urban", "nbi_rural"]]
```

**Inputs**: `data/raw/DANE-NBI/CNPV-2018-NBI.xlsx` (443 KB, already on disk). Supporting files: `CNPV-2018-NBI-DIVIPOLA-2021.xlsx` (DIVIPOLA code map), `CNPV-2018-NBI-AUTORRECONOCIMIENTO-ETNICO.xlsx` (ethnic breakout), `CNPV-2018-NBI-CENTROS-POBLADOS.xlsx` (centros poblados), and `CNPV-2018-NBI-justificacion-actualizacion-febrero-2021.pdf` (methodology document).

**Schema output** (`data/fundamentals/nbi_2018.csv`):

```text
codigo_municipio,nbi_rate,nbi_urban,nbi_rural
```

**IPM module**: kept as `src/co_president/ingestion/ingest_ipm.py`, but **downgraded to secondary**. The IPM ingest is OPT-IN — it runs only when `--component ipm` is passed. Its output is:

```text
codigo_municipio,ipm_2018,ipm_2018_imputed,ipm_2022,ipm_2022_imputed
```

The `ipm_<year>_imputed` boolean flags remain mandatory. The imputation strategy is the same logistic regression on 33 cabeceras, but the R² guard is stricter: **if R² < 0.5, the column is excluded from the feature set** (up from 0.3, reflecting the higher risk for imputed IPM vs ground-truth NBI).

**The `MunicipalFeatures` dataclass uses `nbi_rate` as the primary poverty feature** (always present, 0 % imputed) and `ipm_2018` as a secondary optional column.

**Tests** (`tests/test_ingestion_nbi.py`):
1. `build_nbi_features()` returns ≥ 1 100 rows (allow minor DIVIPOLA drift)
2. `nbi_rate` in [0, 1] for all rows
3. `codigo_municipio` is 5-character zero-padded string
4. NBI values for Bogotá (~0.03) are lower than for Chocó/Guanía (~0.30+) — sanity that rural poverty is captured

**Wire into build_feature_matrix**: extend `_COMPONENT_FILES` with `"nbi": "nbi_2018.csv"` and `"ipm": "ipm_2018.csv"` (IPM is optional — `validate_component_health` does not fail if IPM is missing).

### 4.3 SPEC-19: DANE population projections (2018–2026)

**Module**: `src/co_president/ingestion/ingest_population.py`

**Input**: `data/raw/PPED-AreaMun-2018-2042_VP.xlsx`. Read the "PobMunicipalxÁrea" sheet, keep `(MPIO, AÑO, TOTAL)` filtered to `ÁREA GEOGRÁFICA == "Total"`, zero-pad `MPIO` to 5 digits (already 5-digit), pivot to wide:

```text
codigo_municipio,pop_2018,pop_2019,pop_2020,pop_2021,pop_2022,pop_2023,pop_2024,pop_2025,pop_2026
```

This is straightforward — the XLSX is < 5 MB. Single read, ~1 s.

**Schema rationale**: storing all years 2018–2026 (per user answer Q5) costs only 9 extra columns and makes the feature matrix ready for 2026 forecasting. The current model only consumes `pop_2022`, but the API exposes all years for the 2026 model.

### 4.4 SPEC-20: Extended risk ingest

**Module**: extends `src/co_president/ingestion/ingest_risk.py`

- **PDET** (170 codes) — keep the hardcoded list at `ingest_risk.py:403-574` as authoritative fallback (already validated by SPEC-13b)
- **INDEPAZ** — download `https://indepaz.org.co/wp-content/uploads/2022/11/RESUMEN_GRUPOS_2022.pdf` on demand. The PDF is text-based on most pages but has 2 image-only pages. Use `pdfplumber` first; fall back to `pymupdf` (`fitz`) for image-only pages with text-extraction disabled. Output: `codigo_municipio, armed_group, armed_group_presence` (0/1)
- **UNODC Coca** — `unodc_coca_municipal_2022.pdf` from `unodc.org/documents/colombia/2022/`. Same `pdfplumber` → `pymupdf` strategy. Output: `codigo_municipio, coca_hectares`
- **MOE Risk Map** — same `pdfplumber` → `pymupdf` cascade

The `_hardcoded_fallback` functions at `ingest_risk.py:361-612` stay as last-resort fallbacks; the new code path tries local PDFs first, then remote download, then hardcoded.

---

## 5. Phase 2 — Sabaneta & PDF sanity checks (SPEC-16, SPEC-23)

### 5.1 SPEC-16: Sabaneta Cámara loader (fixture + loader per user Q3)

**Module**: `src/co_president/ingestion/ingest_sabaneta.py`

```python
def load_sabaneta_camara() -> pd.DataFrame:
    """Load the Sabaneta Cámara de Representantes 2002-2022 file.

    Returns long-form DataFrame:
        municipio, periodo, partido, total_votes
    """
```

The file is 81 lines, all numeric, UTF-8, comma thousands separator (need to strip), one record per `(periodo, partido)`. Aggregate to a municipal-level time series for Sabaneta (divipola `05631`).

**Why bother**: the same source schema (period, party, total) is used for the upcoming 2026 election. This is a **gold-standard fixture** for the cleanup logic.

**Two uses** (per user answer Q3):
- `tests/fixtures/sabaneta/`: copy the file to the fixtures dir and add a `conftest.py` fixture that loads it; use it in `test_ingestion_sabaneta.py`
- The loader itself: a production `load_sabaneta_camara()` exposed in `ingestion/__init__.py` and usable from `python -m co_president ingest --component sabaneta`

**Tests**:
- `parse_sabaneta_file()` returns the expected number of records (4 periods × 16 parties + a few "Otros" rows)
- `periods_present` is exactly `{2002, 2006, 2010, 2015, 2019}` (the 2010–2014 congress is the same as 2014–2018 Cámara term)
- `total_2022_sabaneta` matches the official Cámara nacional total for Antioquia within ±1 %

### 5.2 SPEC-23: 2022 MOE PDF cross-validation (use both `pdfplumber` and `pymupdf` per user Q4)

**Module**: extends `data_results.py`

The PDF `data/2022-presidential-results/2022.11.09-LIBRO-RESULTADOS-ELECTORALES-PRESIDENCIALES-2022.pdf` is the MOE's published official resultados, but the user has confirmed it contains a **significant number of image-only pages**. Use a two-stage parser:

```python
def _extract_tables_with_fallback(pdf_path: Path) -> list[list[list[str]]]:
    """Extract tables from a PDF, falling back to pymupdf for image-only pages."""
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            tables = [t for page in pdf.pages for t in page.extract_tables() or []]
            if tables:
                return tables
    except Exception:
        pass
    try:
        import pymupdf  # type: ignore[import-not-found]
        doc = pymupdf.open(pdf_path)
        tables = []
        for page in doc:
            tabs = page.find_tables()
            for tab in tabs:
                tables.append(tab.extract())
        return tables
    except ImportError:
        return []
```

**Tests** (per user Q4): round-1 Petro share in the PDF matches `RoundResult.get_share("gustavo_petro")` within ±0.05 %; same for Hernández. Also test that the parser successfully extracts from at least one **image-only** page (assert ≥ 1 non-empty table after fallback).

This gives a 3-way comparison (Reg / MOE_CSV / MOE_PDF) which is a much stronger source of truth than the current 2-way.

---

## 6. Phase 3 — Municipal features API (SPEC-21)

**New module**: `src/co_president/fundamentals/__init__.py` (per the roadmap's planned `src/co_president/fundamentals/` directory).

```python
# src/co_president/fundamentals/features.py

@dataclass(frozen=True)
class MunicipalFeatures:
    """One row of the municipal feature matrix, fully typed and immutable."""
    codigo_municipio: str
    nombre_municipio: str
    departamento: str
    region: str | None

    # CNPV 2018 demographic features (always present, 1 122 rows)
    poblacion_total: int
    poblacion_afrocolombiana: int
    poblacion_indigena: int
    poblacion_rural_dispersa: int
    pct_afro_colombian: float
    pct_indigenous: float
    pct_rural_disperso: float
    years_schooling_promedio: float
    pct_school_attendance: float
    internet_access_rate: float
    labor_force_participation_rate: float
    pct_female: float
    rooms_per_household: float
    persons_per_household: float
    pct_age_18_29: float
    pct_age_30_54: float
    pct_age_55_plus: float

    # Poverty — NBI is primary (complete coverage, 0 % imputed), IPM is secondary (optional)
    nbi_rate: float           # Necesidades Básicas Insatisfechas (DANE CNPV 2018, all 1 122 municipios)
    nbi_urban: float           # NBI in urban areas
    nbi_rural: float           # NBI in rural areas
    ipm_2018: float | None     # Multidimensional Poverty Index (only ~33 cabeceras, rest imputed)
    ipm_2018_imputed: bool
    ipm_2022: float | None
    ipm_2022_imputed: bool

    # Population projections (all years 2018–2026)
    pop_2018: int
    pop_2019: int
    pop_2020: int
    pop_2021: int
    pop_2022: int
    pop_2023: int
    pop_2024: int
    pop_2025: int
    pop_2026: int

    # Risk
    risk_level: Literal["extreme", "high", "medium", "low"] | None
    is_pdet: bool
    armed_group_presence: bool
    coca_hectares: float
    high_risk_flag: bool

    # Historical vote shares (round 1 + 2, 2002–2022, for the canonical left/right pair each year)
    historical: tuple[HistoricalRecord, ...]


@dataclass(frozen=True)
class HistoricalRecord:
    year: int
    round: int
    left_candidate: str
    right_candidate: str
    left_share: float
    right_share: float
    abstention_rate: float


def load_features(data_dir: Path | None = None) -> pd.DataFrame:
    """Load the municipal feature matrix as a typed, validated DataFrame.

    Returns:
        DataFrame where every column is a typed attribute of MunicipalFeatures.
        All 1 122 municipalities are present.
    """
```

**Tests** (`tests/test_fundamentals_features.py`):

1. `MunicipalFeatures` is frozen
2. `load_features()` returns exactly 1 122 rows
3. `load_features()["ipm_2018_imputed"]` is a boolean series; at least 1 000 are True (documented imputation rate)
4. `load_features()` against a fixture `tests/fixtures/fundamentals_minimal/` produces expected values
5. `MunicipalFeatures.historical` is a tuple (immutable)
6. `load_features()` raises `FileNotFoundError` when a component is missing (after SPEC-15 strict mode)

---

## 7. Phase 4 — Municipal hierarchical model (SPEC-22) — the new architecture

This is the **core change** for the user's requirement (B): fit national polls at the municipal level. The roadmap §9 sketches it; here is the file-by-file plan that satisfies both (B) municipal disaggregation and (A) overfitting prevention.

### 7.1 Architectural overview

```text
┌────────────────────────────────────────────────────────────────────┐
│  Layer A — MUNICIPAL PRIOR (always uses fundamentals)              │
│  For each municipio m and each candidate k:                        │
│    logit(p_mk) = α_k                                              │
│               + β_historical[k]  ·  left_share_2018_m              │
│               + β_ethnicity[k]   ·  pct_afro_colombian_m           │
│               + β_poverty[k]     ·  nbi_rate_m          (NBI, not IPM) │
│               + β_rural[k]       ·  pct_rural_disperso_m           │
│               + β_education[k]   ·  years_schooling_m              │
│               + β_risk[k]        ·  high_risk_flag_m               │
│    + sigma_m[k] · mu_m_raw[m, k]   (non-centered)                  │
│                                                                   │
│  NBI is primary (0 % imputed, complete municipal coverage).       │
│  IPM_2018 is a secondary optional feature, flagged as imputed.     │
└────────────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────┴────────────────────────────────────────┐
│  Layer B — NATIONAL POLL LIKELIHOOD                                │
│  For each national poll n:                                        │
│    p_natl_n = Σ_m (pop_2022_m / pop_2022_total) · softmax_m(logit p_mk)  │
│    α_poll_n = p_natl_n × φ_poll × m_n                            │
│    votes_n ~ DirichletMultinomial(N_n, α_poll_n)                  │
│  House effects applied at the national level (single η per pollster)│
└────────────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────┴────────────────────────────────────────┐
│  Layer C — ELECTION ROLLUP LIKELIHOOD (2022 only)                 │
│  For each round:                                                  │
│    p_natl = Σ_m pop_m · softmax_m(logit p_mk)                     │
│    α_elec = p_natl × φ_elec                                       │
│    actual_votes ~ DirichletMultinomial(N_total, α_elec)            │
└────────────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────┴────────────────────────────────────────┐
│  Layer D — TRANSFER RATE ESTIMATION (SPEC-30, feeds runoff model) │
│  For each eliminated candidate c in each runoff election year:     │
│    logit(β_c_to_A) = γ_0c + γ_ethn[c] · pct_afro_m               │
│                             + γ_poverty[c] · nbi_rate_m           │
│                             + γ_rural[c] · pct_rural_disperso_m   │
│                             + γ_camara[c] · camara_share_m      │
│                             + γ_senado[c] · senado_share_m      │
│    β_c_to_B = 1 - β_c_to_A                                        │
│  γ priors: Normal(0, 1)  |  observed: R2 vote shares              │
│  Cámara = local machine (gamonalismo), Senado = national ideology │
└────────────────────────────────────────────────────────────────────┘
```

**Why this is the right design** for both requirements (and for 2026):

- **(B) Municipal-level fit**: the prior and the posterior are both at the municipality level; the national poll is the only observation, but it carries information about every municipality through the weighted rollup
- **(A) No overfitting**: the β coefficients are **shared** across all 1 122 municipalities (only K betas per category, not 1 122 × K), and the `mu_m_raw` non-centered term is the only municipality-specific parameter beyond the deterministic transform of features. The effective number of free parameters is K × (1 intercept + 5 betas) + 1 122 × K non-centered, but the non-centered ones are regularized toward zero by `sigma_m`
- **(2026-readiness)** (§9.6): demographics (NBI, ethnicity, education, internet) are invariant to the candidate set. When 2026 candidates are registered, the model swaps `FIRST_ROUND_CANDIDATES` without retraining — the β coefficients capture the relationship between demographics and voting behavior, not between specific candidates. NBI as primary poverty feature ensures the signal is not contaminated by imputation noise.
- **Reviewer concurrence**: the shared β + non-centered architecture was singled out as "solving the curse of dimensionality beautifully" — the model learns country-wide patterns while letting each municipality deviate locally

### 7.2 Extend `ModelConfig`

`src/co_president/config.py` (currently 408 lines):

```python
class ModelConfig:
    # … all existing fields …
    # NEW for SPEC-22
    fundamentals_mode: Literal["off", "prior_only", "joint"] = "prior_only"
    """off: old MVP (polling-only national model, `build_round1_model`).
    prior_only: municipal fundamentals enter the per-municipality prior;
                national polls update it. **Canonical default.**
    joint: full hierarchical model — theta is per-municipality and rolls up."""

    # Regularization (SPEC-27)
    beta_coefficient_prior_sigma: float = 0.5     # σ for all β priors
    use_horseshoe_prior: bool = False             # if True, use horseshoe on β
    sigma_m_prior: float = 0.3                    # σ_m in roadmap §3
    pool_alpha: float = 0.95                      # global shrinkage on the μ_m non-centered term
    enable_population_weighting: bool = True      # if False, uniform weights (sensitivity check)
```

### 7.3 New module: `src/co_president/model_municipal.py`

Roughly 350 LOC. Builds on the existing `model_round1.py`.

```python
def build_municipal_model(
    features: pd.DataFrame,           # from SPEC-21
    polls: pd.DataFrame,              # national polls (now including AS/COA per SPEC-25)
    results: tuple[RoundResult, RoundResult] | None,
    config: ModelConfig,
) -> pm.Model:
    """Build the hierarchical municipal Bayesian model.

    Layers A, B, C as documented in the plan.
    """
```

**Implementation notes** (the hard parts):

- **Vectorization**: all 1 122 × K logit calculations are tensor ops, never Python loops
- **Population-weighted rollup**: `p_natl = sum(pop * softmax(logit_p))`, where `pop[m] / pop_total` is precomputed. Must be float64 to avoid precision loss in the Dirichlet concentration
- **Reverse-time RW on national θ**: stays unchanged from `model_round1.py`; the per-municipality logit is anchored to the national logit on election day (i.e., `logit(p_mk) = national_logit_k + deviation_mk`)
- **Non-centered parameterization** is mandatory
- **Polling cutoff**: the `polls` argument is the standard national-poll DataFrame; nothing about the architecture changes on the input side

### 7.4 Forecast dataclass for hierarchical output

```python
@dataclass(frozen=True)
class MunicipalForecast:
    codigo_municipio: str
    candidates: tuple[CandidateForecast, ...]   # per-municipality posterior
    turnout_mean: float

@dataclass(frozen=True)
class HierarchicalForecast:
    national: Round1Forecast
    municipal: tuple[MunicipalForecast, ...]
    pop_share_per_municipio: dict[str, float]
    effective_num_municipalities: int            # sum of posterior shrinkage weights
```

### 7.5 Backward compatibility (separate file, per user Q1)

`model_municipal.py` is a **separate file** from `model_round1.py`. There is no `fundamentals_mode` parameter on `build_round1_model` — that function stays identical to the current SPEC-06 version. The dispatch happens at the CLI level:

```python
# In __main__.py
if config.fundamentals_mode == "off":
    model = build_round1_model(polls, results, config)
else:
    features = load_features()
    model = build_municipal_model(features, polls, results, config)
```

**Migration path**: old API callers that pass `ModelConfig()` will now get `fundamentals_mode = "prior_only"` (the new canonical default). If they call `build_round1_model` directly, nothing changes. Only the CLI (and new callers of `build_municipal_model`) respect the hierarchical path.

All existing SPEC-06/07/08/09/10 tests `test_model.py` call `build_round1_model` directly and are unaffected. New model tests use `build_municipal_model`.

### 7.6 Transfer rate estimation model (SPEC-30) — replaces hand-tuned constants

The user answered Q4: **"I want us to create a model that performs the estimations on the transfer rates; use all the historical data (both presidential and legislative) and the other demographics data to achieve this."** This is SPEC-30.

**What gets deleted**: the hardcoded `TRANSFER_FAJARDO_PETRO`, `TRANSFER_FAJARDO_HERNANDEZ`, `TRANSFER_GUTIERREZ_HERNANDEZ`, `TRANSFER_GUTIERREZ_PETRO`, and `TRANSFER_BLANCO_SPLIT` constants in `config.py:496-512`. The hand-tuned approach is replaced by a data-driven model.

**What gets built**: `src/co_president/model_transfer.py` (~200 LOC)

```python
def build_transfer_model(
    features: pd.DataFrame,             # municipal demographics (SPEC-21)
    historical_r1r2: dict[int, tuple[pd.DataFrame, pd.DataFrame]],  # R1→R2 per year
    camara: pd.DataFrame,               # Cámara 2002-2022 per municipio (local machine)
    senado: pd.DataFrame,               # Senado 2002-2022 per municipio (national ideology)
    config: ModelConfig,
) -> pm.Model:
    """Estimate runoff transfer rates from historical data.

    For each election with a runoff (2010, 2014, 2018, 2022) and for each
    eliminated candidate c, estimate a vector β_c of length 2 (probability of
    transferring to runoff candidate A vs B). The estimation uses:

    1.  Municipal R1 vote shares for each eliminated candidate
    2.  Municipal R2 vote shares for the two runoff candidates
    3.  Demographic features (β_c can vary by demographic profile)
    4.  Cámara de Representantes vote shares (local party machinery, gamonalismo)
    5.  Senado vote shares (national ideological alignment — voters choose
        national parties, not local candidates)

    Model structure:
        logit(β_c_to_A) = γ_0c + γ_ethn[c] · pct_afro_m
                                 + γ_poverty[c] · nbi_rate_m
                                 + γ_rural[c] · pct_rural_disperso_m
                                 + γ_camara[c] · camara_share_m
                                 + γ_senado[c] · senado_share_m
        β_c_to_B = 1 - β_c_to_A
    """
```

**Why Cámara AND Senado**: Cámara reflects local political machinery (each department elects its own representatives — capturing gamonalismo, clientelism, and regional party strength). Senado reflects national ideological alignment (the entire country votes for the same Senate — capturing which national figures and party brands resonate in each municipality). Together they disentangle "local machine loyalty" from "national ideological preference" — a Fajardo voter in Antioquia whose municipality votes Conservative in Cámara but Polo Democrático in Senado is a very different transfer case than one whose municipality votes Conservative in both.

**Inputs needed from the data pipeline**:
- Municipal R1 vote shares for all candidates for 2010, 2014, 2018, 2022 (from CEDAE gz files → SPEC-15 §3.2)
- Municipal R2 vote shares for the two runoff candidates for the same years
- Municipal Cámara de Representantes vote shares 2002-2022 (from CEDAE `*_camara.dta.csv.gz` files, already on disk)
- Municipal Senado vote shares 2002-2022 (from CEDAE `*_senado.dta.csv.gz` files, already on disk)
- Demographic features from `MunicipalFeatures`

**Validation**: the SPEC-30 posterior over β_c is compared against the AS/COA empirical transfer matrix. If the model's median β differs from the AS/COA empirical value by > 10 pp, log a WARNING. The AS/COA matrix is a validation target, not a source of truth.

**Convergence guardrail** (reviewer recommendation): if the demographic-driven γ coefficients fail to achieve R-hat < 1.10 (indicating non-convergence of the transfer model), the model **automatically falls back** to a simpler Dirichlet-Categorical distribution over historically observed transfer rates:

```python
# Fallback in model_transfer.py
if transfer_rhat > 1.10:
    logger.warning("Transfer model did not converge (R-hat=%.2f > 1.10); "
                   "falling back to historical Dirichlet-Categorical prior", transfer_rhat)
    return _build_historical_transfer_prior(eliminated_candidates, historical_r1r2)
```

The fallback `_build_historical_transfer_prior` pools transfer rates from all historical runoffs (2010, 2014, 2018) into an uninformative Dirichlet prior centered on the empirical mean. This ensures the model always produces a valid transfer estimate regardless of convergence quality.

**Rollup into the runoff forecast**: `model_runoff_matrix.py:estimate_runoff_matrix()` calls `model_transfer.sample_transfer_rates(features, config)` instead of reading hardcoded constants. The runoff model's pairing logic is unchanged — only the transfer coefficients change from hardcoded to data-driven.

### 7.7 Regularization & overfitting control (SPEC-27 — cross-cuts everything)

The user's hard constraint (A) makes SPEC-27 a first-class concern, not an afterthought. The mechanisms:

1. **Priors on coefficients** — every β has a `Normal(0, 0.5)` (or `Horseshoe`) prior; this is the strongest defense against overfitting
2. **Cross-validation** (SPEC-26) — leave-one-year-out CV; report MAE per fold
3. **Information criteria** — `az.waic(idata)` and `az.loo(idata)` reported alongside point estimates
4. **Sensitivity ablation** — re-fit the model with `fundamentals_mode = "off"` and `"joint"`, compare posteriors; if the difference is < 1 pp at the national level, the model isn't overfitting
5. **Effective number of municipalities** — `sum(posterior_shrinkage_weights)`; should be << 1 122 to indicate shrinkage is working
6. **Year-2018-out validation** — train on 2002/2006/2010/2014, predict 2018; this is the **strongest** out-of-sample test because the 2018 election was the first Petro-vs-Duque race with very different coalition alignments

### 7.8 Tests for SPEC-22 + 27 + 30 (using the four-phase strategy from AGENTS.md §6)

1. **Graph test** (fast): `pm.Model` builds. Expected free RVs: `alpha (K), beta_* (5 groups × K), mu_m_raw (1122 × K), sigma_m, sigma_rw, sigma_house, raw_house (P × K)`. Expected observed: 1 Dirichlet-Multinomial per poll + 1 per round.
2. **Prior predictive test** (fast): `pm.sample_prior_predictive(n=200)`. All `theta` draws in [0, 1] (after softmax); national rollup also in [0, 1].
3. **Convergence test** (`@pytest.mark.slow`): 10 mock municipalities, 3 polls, 3 candidates. R-hat < 1.10.
4. **Sanity test** (`@pytest.mark.slow`): real `municipal_feature_matrix`, polls up to 2022-05-15, **2022 actuals known**. National rollup's `p_petro` mean within ±5 pp of 40.34 %.
5. **Overfitting test** (SPEC-27, `@pytest.mark.slow`): train on 2002/2006/2010/2014 CEDAE, predict 2018, compare to actual 2018 results. R-hat check + MAE report. **Gating test per user Q3: fails the build if R² < 0.3, with a report listing which data components are insufficient.**
6. **Transfer rate model graph test** (SPEC-30, fast): `build_transfer_model` builds. Expected free RVs: `gamma_* (4 groups × 2 candidates × K eliminated)`. Expected observed: Beta-binomial per municipio-year.
7. **Transfer rate validation test** (SPEC-30, fast): compare the model's median β against the AS/COA transfer_matrix.csv. Divergence > 10 pp → WARNING logged.

---

## 8. Phase 5 — Out-of-sample validation framework (SPEC-26)

Critical for requirement (A). New module: `src/co_president/validation/municipal_oos.py`.

```python
def leave_one_year_out(
    features: pd.DataFrame,
    all_years_polls: dict[int, pd.DataFrame],   # {2002: df, 2006: df, …, 2022: df}
    all_years_results: dict[int, RoundResult],
    config: ModelConfig,
) -> DataFrame:
    """Run leave-one-year-out cross-validation.

    For each held-out year y:
        Train on years != y (using CEDAE historical results for those years as anchors)
        Predict y using polls up to (y-1 month) and fundamentals
        Compare to actual y results
    Returns:
        DataFrame with columns: held_out_year, candidate, predicted_mean, actual_share, abs_error
    """


def year_2018_holdout(
    features: pd.DataFrame, polls_to_2018: pd.DataFrame, results_2018: RoundResult,
    config: ModelConfig,
) -> RoundValidation:
    """The strongest out-of-sample test: predict 2018 without ever using 2018 data."""


def compare_modes(
    features: pd.DataFrame, polls: pd.DataFrame, results: tuple[RoundResult, RoundResult],
    config_off: ModelConfig, config_joint: ModelConfig,
) -> DataFrame:
    """Compare 'off' vs 'joint' modes; assert difference is bounded to prevent overfitting."""
```

**The "predict 2018 from 2002-2014" test** is the centerpiece. Its result is logged to `results/generalization_report.md` and committed so the user can see the validation evidence.

---

## 9. Phase 6 — CLI & documentation

### 9.1 New subcommands in `__main__.py` (currently 1 185 lines)

```python
@cli.command()
@click.option("--component", type=click.Choice(["divipola","cnpv","ipm","population","risk","sabaneta","as_coa","all"]))
def ingest(component: str) -> None:
    """Run ingestion pipelines for the requested component(s)."""

@cli.command()
def features() -> None:
    """Build the municipal feature matrix; print coverage report."""

@cli.command()
@click.option("--mode", type=click.Choice(["off","prior_only","joint"]), default="prior_only")
@click.option("--year", type=int, default=2022)
@click.option("--validate-oos", is_flag=True, help="Run leave-one-year-out validation")
def forecast(mode: str, year: int, validate_oos: bool) -> None:
    """Run the forecast for the given year, optionally with the municipal hierarchical prior.
    For 2026, only sparse polling is available; the model relies on fundamentals."""
```

### 9.2 New make targets

```makefile
.PHONY: ingest
ingest:  ## Build all fundamentals
	uv run python -m co_president ingest --component all

.PHONY: cnpv
cnpv:  ## Build CNPV 2018 features (~6 min)
	uv run python -m co_president ingest --component cnpv

.PHONY: features
features:  ## Build municipal feature matrix
	uv run python -m co_president features

.PHONY: validate-oos
validate-oos:  ## Run leave-one-year-out validation (slow, 5-15 min)
	uv run python -m co_president forecast --validate-oos
```

### 9.3 Update `ENHANCEMENT_ROADMAP.md`

Add a `## Implementation Status` section that tracks SPEC-12 through SPEC-29 (mark ✅ for shipped, ⏳ for in-progress, ❌ for not started).

### 9.4 Notebook: `notebooks/derive_municipality_calibration.ipynb`

A new EDA notebook that answers "do the historical features actually predict 2022 vote share?" — a sanity check on whether the model integration is adding signal. Implements a per-municipality OLS of `vote_share_2022_r1_petro ~ vote_share_2018_r1_petro + ipm_2018 + pct_afro_colombian + high_risk_flag` and prints R². **This is a `.gitignore`d notebook, not a test.**

### 9.5 The 2026 forecast mode (SPEC-28)

For 2026 the model has **sparse polling** (likely 5-10 polls, not 46). The CLI should expose a `--year 2026` flag that:

- Loads `CONSULTATION_VOTES` from the 2026 inter-party consultations (placeholder for now)
- Uses only the polls available up to the requested date
- Sets `fundamentals_mode = "joint"` by default (because the 2022 model has no 2026 polls yet)
- Reports a wide credible interval reflecting the small N of polls
- The validation framework runs the 2018-holdout **gating test** in the background and refuses to produce a 2026 forecast unless the OOS MAE is < 5 pp. Per user Q3, if the test fails, it produces a **"missing data" report** that lists which data components are insufficient to reach the accuracy target (e.g., "Cámara legislative results missing for 2010 → add data/raw/cedae/2010_camara.dta.csv.gz").

```python
# In src/co_president/cli.py
def _validate_before_2026_forecast(config: ModelConfig) -> None:
    """Refuse to produce a 2026 forecast if the OOS validation fails."""
    if not _oos_mae_acceptable(config):
        msg = "OOS MAE too high; refusing to forecast 2026 (check fundamentals_mode + priors)"
        raise RuntimeError(msg)
```

### 9.6 2026 candidate mapping strategy (answer to reviewer question)

**Question**: *Gustavo Petro cannot run for re-election in 2026. How do you map 2026 candidates to historical ideological features without letting subjective biases warp the municipal priors?*

**Answer**: The model does **not** map 2026 candidates to historical ideological labels (Left/Right). Instead, it uses a **candidate-agnostic feature space**:

1. **Historical vote shares are predictors, not labels.** The feature matrix stores municipal vote shares for specific historical candidates (`vote_share_2018_r1_gustavo_petro`, `vote_share_2022_r1_rodolfo_hernandez`, etc.). These are columns in the DataFrame — they serve as **predictors** that capture how each municipality's voting bloc pattern correlates with demographics. The model learns `β_historical[k]` — the relationship between past voting patterns and current vote intention — without knowing who the 2026 candidates are.

2. **The β coefficients are learned per candidate position, not per candidate name.** When the model fits on 2022 data, it learns that municipalities with high `vote_share_2018_r1_petro` also have high 2022 Petro support (β strong and positive). When projected to 2026, the same β matrix is applied to the 2026 municipal priors — but the 2026 candidates are simply new rows in the candidate name/coalition registry (`FIRST_ROUND_CANDIDATES_2026`), not new model parameters.

3. **Demographics are the invariant signal.** NBI rate, pct_afro_colombian, years_schooling, internet_access_rate, and risk_level do not change based on who is running. These are the foundation that transfers to any candidate set. The model predicts 2026 vote shares as a function of **who the municipality is**, not **who the municipality voted for last time**.

4. **The transfer model (SPEC-30) is candidate-agnostic.** It estimates how votes flow between rounds based on Cámara legislative alignment and demographics — not based on candidate names. Whether a 2026 Center-Left candidate receives Fajardo-like transfers depends on their Cámara party profile, not on a subjective label.

5. **The `--year` flag handles the candidate registry swap.** `FIRST_ROUND_CANDIDATES` is a year-keyed dict (already the pattern in `config.py`). `--year 2026` loads `FIRST_ROUND_CANDIDATES_2026` which will be populated when the official candidate list is published. Until then, the model runs in backfill mode against 2022.

**No subjective bias is introduced** because the model never asks "is this 2026 candidate like Petro?" — it asks "given this municipality's demographics, NBI rate, historical voting patterns, and legislative alignment, what is the posterior distribution of vote shares for each candidate on the ballot?"

---

## 10. TDD schedule (per AGENTS.md §3 cycle)

For each SPEC, execute in this order — **never** in parallel:

```text
RED:    write tests in tests/test_<spec>.py; run `pytest -k <spec>`; confirm FAIL
GREEN:  implement minimum code; run `pytest -k <spec>`; confirm PASS
        run `pytest tests/` to confirm no prior test broke
REFACTOR: clean up, add docstrings, type hints
        run `pyright src/`; fix ALL errors
        run `ruff check src/ tests/`; fix ALL errors
        run `ruff format src/ tests/`; re-run `make check`
COMMIT: cr <branch>; wait for user approval; git-smart-commit; merge to dev
```

The full quality gate (per AGENTS.md §3) is `make check` = `fmt → lint → typecheck → test`. **`pyright tests/`** is intentionally allowed to fail (~276 errors documented in AGENTS.md §11); the gate is `pyright src/`.

---

## 11. Risk register (and mitigations)

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| **Overfitting to 2022** (user hard constraint A) | **High** | **Critical** | Strong β priors (σ=0.5 default); horseshoe option (SPEC-27); leave-one-year-out CV; 2018-holdout test; the OOS MAE must be < 5 pp before any 2026 forecast is allowed; if R² of the OLS sanity check on 2018 < 0.3, fall back to `fundamentals_mode = "off"` |
| CNPV 2018 raw schema differs from DDI spec | Medium | High | First test is a 1 000-row synthetic chunk with the exact column names from the DDI XML. Real data columns asserted with a soft-warning + log line. |
| Sabaneta period labels are non-standard (`2019/2022` not `2018`) | High | Low | Document in the loader docstring; do not auto-normalize (it's a 4-year Cámara term). |
| **Imputing IPM for 1 089 of 1 122 municipalities introduces spurious signal** (user Q2 chose option a) | Medium (↓ from High) | Medium | **NBI is now the primary poverty feature** (per reviewer recommendation) — NBI has 100 % municipal coverage with zero imputation. IPM is secondary and optional; its R² guard is stricter (0.5, up from 0.3). If IPM imputation fails, `nbi_rate` remains the authoritative poverty signal. |
| New `ModelConfig.fundamentals_mode` flag breaks existing `test_model.py` | Low | Low | `build_round1_model` does not read `fundamentals_mode`; existing tests are unchanged. Only new `model_municipal.py` (separate file per user Q1) uses the field. Migration risk is zero. |
| **Transfer rate model (SPEC-30) is novel; no prior PyMC reference implementation** | Medium | Medium | Start with a Beta-binomial regression (well-documented PyMC pattern); the graph test validates parameter counts against known equations. If the model doesn't converge on real data, fall back to the logistic single-year estimator. |
| **Cámara legislative data format may differ from presidential CEDAE** | Medium | Medium | Schema check in SPEC-15 §3.2 covers all CEDAE files (Camara/Presidencia/senado). Any non-conforming columns are logged and skipped. |
| **Performance of `build_municipal_model`** (1 122 municipalities × K) | Medium | High | Non-centered parameterization + lazy tensor construction + vectorized rollup. Add `@pytest.mark.slow` to convergence tests. |
| 4-GB+ parquet file under `data/processed/` blows up git | Low | Medium | `data/processed/municipal_feature_matrix.*` already in `.gitignore` (verify). CSV stays in git for inspection. |
| **INDEPAZ / UNODC PDFs not present in `data/raw/`** (only the hardcoded fallback) | High | Medium | SPEC-20 downloader writes to `data/raw/`; pipeline falls back to hardcoded if not. Add a `make ingest-risk-pdfs` target. |
| **MOE PDF has image-only pages** (user Q4) | High | Medium | Use `pdfplumber` first, fall back to `pymupdf` for image-only pages with text-extraction disabled. Test that ≥ 1 non-empty table is extracted after fallback. |
| **Internet features not in CNPV 2018** (use GEIH instead) | Medium | Medium | CNPV 2018 has `H_VIVIENDA_INTERNET` (already in the plan). GEIH is for cross-validation. |
| **`compute_lagged_features` currently uses 2022-only data** (placeholder `000NA`) | High | High | Fixed in SPEC-15 §3.2. |
| **AS/COA transfer matrix conflicts with model-estimated β** (user Q4 — model-based transfer) | Low | Low | AS/COA matrix is a validation target, not source of truth. Model-estimated β supersedes all hand-tuned constants. If model β diverges from AS/COA by > 10 pp, log WARNING but do not override. |
| **2026 election has different candidates than 2022** | Certain | High | SPEC-28 keeps candidate set configurable via `FIRST_ROUND_CANDIDATES` (already the case); the model uses the 2026 candidate set when `--year 2026`. No code change, just config. |

---

## 12. What this plan explicitly does NOT do (out of scope for this iteration)

To stay disciplined (per AGENTS.md §3 "no speculative features"):

- ❌ Digital signals (Google Trends, Twitter) — Phase 4 of the roadmap, defer to SPEC-31+
- ❌ Fully fitted joint hierarchical posterior on real 2022 data — the model builds and converges on mock data; full-data fitting is the CLI `forecast` command, not a build-time test
- ❌ ECV / GEIH / MPMD integration — each is its own SPEC
- ❌ Latinobarómetro / Barómetro Americas ingestion — same
- ❌ Hand-tuned transfer constants (`TRANSFER_FAJARDO_PETRO` etc.) — deleted, replaced by data-driven SPEC-30 model. Hand-tuning is deprecated.
- ❌ Auto-tuning the priors via hierarchical hyperpriors — out of scope
- ❌ Live `python -m co_president ingest --all` workflow that depends on the network — current runbook is "use the local files, fall back to network only when missing"
- ❌ Streaming / distributed model fitting — single-machine only
- ❌ GPU acceleration — out of scope
- ❌ A real-time polling ingestion service — manual data drops only

If you want any of these included, it becomes its own SPEC and its own PR.

---

## 13. Recommended execution order (one developer / one agent)

If I were executing this, I would batch:

1. **Day 1 (SPEC-15 + SPEC-25)** — coverage report, fix `historical_results.csv` against the local CEDAE zips (all 6 years + Cámara/Senado), switch `validate_component_health` to strict mode. AS/COA poll loader (round1 + runoff CSVs only).
2. **Day 2 (SPEC-21)** — `MunicipalFeatures` API built first (reviewer recommendation). Every subsequent ingestion script (CNPV, NBI, IPM, Risk, population) outputs into the strictly validated data contract from the start.
3. **Day 3 (SPEC-19)** — population projections 2018-2026, wired directly into `MunicipalFeatures`.
4. **Day 4 (SPEC-18)** — NBI ingest (primary, complete coverage) + IPM ingest (secondary, optional). NBI has zero imputation.
5. **Day 5 (SPEC-17)** — CNPV 2018 census. The biggest deliverable, wired into the existing `MunicipalFeatures` schema.
6. **Day 6 (SPEC-20)** — risk table expansion with `pdfplumber` + `pymupdf` fallback.
7. **Day 7 (SPEC-16 + SPEC-23)** — Sabaneta loader + MOE PDF cross-validation. Two short, orthogonal tasks.
8. **Day 8 (SPEC-30)** — transfer rate estimation model with convergence guardrail fallback.
9. **Day 9–10 (SPEC-22 + SPEC-27)** — hierarchical model. Build the graph first, then the forecast wrapper, then the regularization layer.
10. **Day 11 (SPEC-26)** — out-of-sample validation framework (gating test per user Q3).
11. **Day 12 (SPEC-28 + SPEC-29)** — 2026 mode + generalization audit.
12. **Day 13** — CLI updates, make targets, ENHANCEMENT_ROADMAP update, notebook, final `make check` on everything.

Each day ends with a merge to `dev` after `cr` review.

---

## 14. Resolved and open questions

### Resolved (user answered, baked into the plan)

| # | Question | Answer | Where applied |
|---|---|---|---|
| Q1 | Separate model file or extend `build_round1_model`? | **Separate.** `model_municipal.py` is standalone; `model_round1.py` unchanged. | §7.5, §7.3 |
| Q2 | Canonical default for `fundamentals_mode`? | **`"prior_only"`.** Hierarchical is the default; `--mode off` reverts to polling-only. | §7.2, §9.1 |
| Q3 | OOS test as gating or advisory? | **Gating.** If R² < 0.3, the build fails with a "missing data" report listing what's insufficient. | §8, §9.5, §7.8(5) |
| Q4 | Use AS/COA transfer matrix, or a model? | **Model-based estimation** from all historical data (presidential R1→R2 + Cámara + Senado + demographics). AS/COA matrix is validation target only. | SPEC-30, §7.6, §3.4 |
| Q5 | Bogotá D.C. on its own or merged with Cundinamarca? | **On its own.** Data already separates them. | no change needed |
| Q6 | 2026 candidate set? | **Out of scope for this iteration.** SPEC-28 ships the mechanism; `FIRST_ROUND_CANDIDATES_2026` added later. | §12 |

### Still open

1. **Cámara AND Senado legislative data schema (two sources)**: CEDAE (`*_camara.dta.csv.gz`, `*_senado.dta.csv.gz`) covers 2002–2018 with `codigo_partido` coding. MOE (`moe_camara_territorial_2022.csv`, `moe_senado_nacional_2022.csv`) covers 2022 with `nomparti` (party names). SPEC-15 must verify both schemas are compatible after the MOE→CEDAE party code crosswalk. SPEC-30's loader needs a `map_moe_party_to_canonical()` function for 2022 and the existing `codigo_partido` mapping for 2002–2018. **Default**: the 2022 legislative data gives us full 2002–2022 coverage; the crosswalk is a one-time static mapping of ~20 party names to codes. Verify during SPEC-15.
2. **Transfer rate model complexity vs convergence**: estimating γ coefficients for 5 demographic/chamber groups (ethnicity, poverty, rural, Cámara, Senado) × 2 candidates per eliminated candidate on ~4 electoral years. With K=5 eliminated candidates (Fajardo, Gutiérrez, Betancourt, Rest, Blanco), that's 5 × 5 = 25 γ parameters on 4 × 1 122 × 5 = 22 440 municipality-year-candidate observations. Adding Cámara and Senado as separate inputs (vs one legislative signal) adds 5 more γ parameters to 30 total — still well-identified. **Default**: if the model doesn't converge (R-hat > 1.10), drop rural first, then one chamber; the convergence guardrail (§7.6) provides the safety net.
3. **Hand-tuned `TRANSFER_*` constants delete timing**: should they be deleted in SPEC-25 (early) or SPEC-30 (late, when the model replaces them)? **Default**: delete during SPEC-30 to avoid a gap where neither the model nor the constants are in effect.
4. **The `build_municipal_model` function must accept `features=None`**: when `fundamentals_mode == "off"`, the function should still work (degenerates to national-only). **Default**: yes; the function signature is `def build_municipal_model(features: pd.DataFrame | None, polls, results, config)`. When `features is None` and `mode != "off"`, raise.
5. **2026 poll ingestion**: AS/COA's 2022 poll set is processed; for 2026, will there be a new `encuestas_2026.csv` file, or will the AS/COA tool be the primary poll source? **Default**: the model supports both sources via `data_polls.py` loaders; the format is determined by the `--year` flag. Defer the actual 2026 data to a later PR.
