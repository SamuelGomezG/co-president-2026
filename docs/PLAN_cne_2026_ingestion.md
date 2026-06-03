# Plan: Ingest and Integrate 2026 CNE Microdata

**Author**: planning agent
**Date**: 2026-06-03
**Branch baseline**: `dev`
**Reference docs**: `AGENTS.md`, `MVP_SPECS_GUIDE.md`, `ENHANCEMENT_ROADMAP.md`, `docs/PLAN_demographic_ingestion.md`
**Critical constraints**:
- (A) The model must support predicting the 2026 election, treating the 2022 framework as a reusable backbone. Backward compatibility with 2022 validation must be preserved.
- (B) Respondent-level design weights (micro-weights) must be strictly separated from poll-level weights (macro-weights).
- (C) The new pipeline must explicitly parameterize the time axis `T` to align with the actual 2026 election dates (May 31 First Round, June 21 Runoff).

---

## 0. Audit of CNE Data Universe

| File/Directory | Status | Notes |
|------|--------|-------|
| `data/2026-polls/` | 🆕 Untouched | 23 ZIP bundles (~1.2 GB) from 9 firms. Covers March 10 to May 21, 2026. |
| `data/2026-polls/silla_vacia_ponderador/encuestas_detalle.csv` | 🆕 Untouched | La Silla Vacía raw aggregator baseline. |
| `data/2026-polls/silla_vacia_ponderador/ponderacion.csv` | 🆕 Untouched | La Silla Vacía weekly snapshots + June 21 `Prediccion_30d` target. |
| `data/2026-polls/silla_vacia_ponderador/Nota-tecnica*.pdf` | 🆕 Untouched | Methodology for La Silla Vacía macro-weights (1.0 in-person, 0.8 phone, 0.75 online). |
| `src/co_president/config.py` | ⚠️ Needs Update | `POLLSTER_RATINGS` only covers 2022 pollsters. Needs extension for 2026. |

**Observation**: Only 4 firms (Atlas Intel, Invamer, CNC, GAD3) provide machine-readable microdata (CSV/Excel/SPSS). 5 firms provide PDF-only reports.

---

## 1. Goals (what "done" means)

1. A new `src/co_president/data_cne_2026.py` module accurately parses the CNE microdata bundles and outputs two canonical files: `2026_topline.parquet` and `2026_runoff_pairings.parquet`.
2. The 11 PDF-only bundles have their runoff matrices extracted via OCR/table parsing (e.g., `pdfplumber`), falling back to topline baseline aggregation if impossible.
3. Our re-derived toplines from the CNE microdata successfully cross-validate against La Silla Vacía's `encuestas_detalle.csv`, with discrepancies > 0.5pp logged and reconciled.
4. `config.py` is safely extended with 2026 candidate definitions, dates, and `POLLSTER_RATINGS_2026` without breaking 2022 tests.
5. **SPEC-08 is fundamentally upgraded**: Instead of just simulating the runoff pairings from the first-round posterior, it now consumes an empirical matrix built from thousands of respondent-level head-to-head scenarios.
6. The Bayesian models extrapolate directly to May 31, 2026 (Round 1) and June 21, 2026 (Runoff). Model predictions are validated against the `Prediccion_30d` snapshot.

---

## 2. Branch & SPEC Allocation

Following §2 + §14 conventions in `AGENTS.md`. New work goes on feature branches forked from `dev`.

| New SPEC | Branch | Depends on | What ships |
|----------|--------|-----------|------------|
| **SPEC-31** CNE Ingest & Normalization | `feat/spec-31-cne-ingest` | none | `src/co_president/data_cne_2026.py`, canonical candidate mapping, and generation of `2026_topline.parquet` / `2026_runoff_pairings.parquet`. |
| **SPEC-32** Cross-Validation & Calibration | `feat/spec-32-cne-xvalidate` | SPEC-31 | Comparison script vs La Silla Vacía; Update `config.py` with 2026 pollster ratings. |
| **SPEC-33** Backfill Pipeline | `feat/spec-33-cne-backfill` | SPEC-32 | `build_clean_polls_2026()` returning `CleanPolls` dataclass; integration into SPEC-05 baseline. |
| **SPEC-34** Empirical Runoff Matrix (SPEC-08 Uplift) | `feat/spec-34-empirical-runoff` | SPEC-31, SPEC-33 | Upgraded `model_runoff_matrix.py` that utilizes observed head-to-head responses via empirical Beta posteriors. |
| **SPEC-35** Validation & Forecasting | `feat/spec-35-cne-forecast` | SPEC-34 | Refit models to 2026 dates (May 31 / Jun 21); output comparison to `Prediccion_30d`. |
| **SPEC-36** Stretch: Demographics & Covariates | `feat/spec-36-cne-demographics` | SPEC-35 | Integration of primary recall (`vote_primaries_2026`) and regional variables into the random walk prior. |

---

## 3. Phase 0 — Ingest, Extraction & Normalization (SPEC-31)

**Goal:** Convert heterogeneous CNE polls (CSV, Excel, SPSS, PDF) into standardized Parquet data structures.

**Module:** `src/co_president/data_cne_2026.py`

### 3.1 Canonical Candidate Normalizer
Creates a strict mapping of all 2026 string variations into canonical keys (matching `Candidate` objects).
```python
CANDIDATE_KEY_MAP_2026 = {
    "Paloma Valencia": "valencia",
    "Paloma valencia": "valencia",
    "Abelardo de la Espriella": "de_la_espriella",
    "Iván Cepeda": "cepeda",
    "Sergio Fajardo": "fajardo",
    "Voto en Blanco": "blanco",
    "No Sabe / No Responde": "ns_nr",
    "Ninguno": "ns_nr"
}
```
*Requirement*: `blanco` and `ns_nr` must be distinctly preserved. In the K=3 runoff model, these will aggregate into `rest`.

### 3.2 Firm-Specific Loaders
```python
def load_atlas_intel(zip_path: Path) -> pd.DataFrame:
    # Extracts the CSV from inside the zip, utilizes the `weight` column.
    
def load_invamer(zip_path: Path) -> pd.DataFrame:
    # Uses `openpyxl` to read raw data and dictionary mapping to resolve columns like `P245`.

def load_cnc(zip_path: Path) -> pd.DataFrame:
    # Uses `pyreadstat` to load SPSS `.sav` and extracts design weights from `FACTOR`.

def load_gad3(zip_path: Path) -> pd.DataFrame:
    # Loads Excel microdata and evaluates `Q06` (Round 1) and `Q08A-D` (Runoffs).
```

### 3.3 PDF-Only Table Extraction
For the 11 PDF-only bundles (Guarumo, Génesis Crea, etc.):
```python
def extract_pdf_runoff_tables(pdf_path: Path) -> pd.DataFrame | None:
    """Attempts to recover head-to-head runoff matrices via pdfplumber."""
```

### 3.4 Topline and Runoff Extraction
From respondent-level data, compute weighted percentages:
```python
def extract_topline(df: pd.DataFrame, firm: str, date: date) -> pd.DataFrame:
    """Compute weighted pct per candidate using respondent design weights."""
    
def extract_runoff_pairings(df: pd.DataFrame, firm: str, date: date) -> pd.DataFrame:
    """Extract all observed head-to-head scenarios as long-format (cand_a, cand_b, pct_a, pct_b, n_eff)."""
```

**Output Data:** `data/2026-polls/_processed/2026_topline.parquet`, `data/2026-polls/_processed/2026_runoff_pairings.parquet`.

**TDD Steps:**
1. **Red**: Test `load_atlas_intel` handles missing ZIP file.
2. **Red**: Test normalizer throws `KeyError` on unknown candidate.
3. **Green**: Implement loader extraction and canonical mapping.
4. **Refactor**: Add progress logging and strict schema validation.

---

## 4. Phase 1 — Cross-Validation & Calibration (SPEC-32)

**Goal:** Reconcile our extracted microdata with La Silla Vacía's published aggregator to ensure methodological consistency.

### 4.1 Comparison against Baseline
Load `data/2026-polls/silla_vacia_ponderador/encuestas_detalle.csv`.
Perform an outer join with `2026_topline.parquet` on `(firm, date, candidate)`.
```python
def validate_topline_crosscheck(microdata_toplines: pd.DataFrame, silla_vacia: pd.DataFrame) -> None:
    # Identify differences > 0.5pp
    # Output report to `data/2026-polls/_processed/crosscheck_report.csv`
```
*Resolution strategy*: If microdata correctly accounts for design weights but differs from Silla Vacía, log the discrepancy and **trust the microdata**.

### 4.2 Updating `POLLSTER_RATINGS` (SPEC-02)
Modify `src/co_president/config.py`:
```python
POLLSTER_RATINGS_2026: dict[str, float] = {
    "Atlas Intel": 5.9,
    "CNC": 5.8,
    "GAD3": 6.2,
    "Guarumo": 5.1,
    "Invamer": 5.3,
    "Génesis Crea": 2.0,
    "TEMPO": 2.0,
    "Corporación MMM": 2.0
}
```
*Note*: The 2026 ratings are completely isolated from the 2022 configuration (`POLLSTER_RATINGS`), ensuring SPEC-02 2022 tests do not fail.

---

## 5. Phase 2 — Backfill Pipeline (SPEC-33)

**Goal:** Connect the cleaned 2026 datasets into the standard model pipeline.

### 5.1 Building Clean Polls
In `data_cne_2026.py`:
```python
def build_clean_polls_2026() -> CleanPolls:
    """Build CleanPolls(round1=..., round2=...) from the 23 CNE bundles + historical pre-March 8 series."""
```

### 5.2 Timeline alignment for Bayesian Models
Update `config.py` to allow dynamic timelines based on the active cycle:
```python
def get_election_dates(cycle: Literal[2022, 2026]) -> tuple[date, date, date]:
    if cycle == 2026:
        return date(2026, 5, 31), date(2026, 6, 21), date(2026, 3, 8) # R1, R2, Consultas
```

**TDD Steps:**
1. **Red**: Test `build_clean_polls_2026` ensures >5 pollsters for R1.
2. **Green**: Implement timeline abstraction and backfilling.

---

## 6. Phase 3 — Empirical Runoff Matrix (SPEC-34)

**Goal:** Upgrade SPEC-08 to use *observed* head-to-head polling margins rather than relying solely on simulation from Round 1.

### 6.1 Empirical Beta Posterior
Add to `model_runoff_matrix.py`:
```python
def head_to_head_empirical(pairings: pd.DataFrame, prior: pm.Beta) -> pd.DataFrame:
    """
    Compute an empirical Beta posterior for pairs polled directly in the microdata.
    Weight observations by macro-weights (recency, quality).
    """
```

### 6.2 Upgrading `estimate_runoff_matrix`
Modify `estimate_runoff_matrix` to consume the `2026_runoff_pairings.parquet` file. 
If an explicit pairing exists (e.g., Cepeda vs. De la Espriella), use the `head_to_head_empirical` logic. 
If it does not exist (fringe pairing), fall back to the existing Dirichlet-Multinomial R1 simulation.

---

## 7. Phase 4 — Validation & Forecasting (SPEC-35)

**Goal:** Refit models for the 2026 race and validate out-of-sample against the aggregator's 30-day forecast.

### 7.1 Refitting the Models
- Train SPEC-06 (First Round) and SPEC-07/08 (Runoff) on `build_clean_polls_2026()` up to the `2026-05-21` boundary.

### 7.2 The June 21 Validation Target
Create `scripts/validate_2026_forecast.py` to compare model output to `data/2026-polls/silla_vacia_ponderador/ponderacion.csv` where `Tipo == "Prediccion_30d"`.
- Verify posterior medians fall within the `Ponderado_Min` and `Ponderado_Max` (`Banda`) intervals for Cepeda, De la Espriella, and Valencia.

**Deliverable**: `results/2026_forecast_validation.md` committed to version control.

---

## 8. Phase 5 — Stretch: Demographics & Covariates (SPEC-36)

**Goal:** Use the demographic and recall variables available in Atlas/Invamer to create a hierarchical offset in the runoff model.

### 8.1 Primary Recall Prior
Utilize `vote_primaries_2026` to inform coalition alignment (gamonalismo vs national ideology).
- Instead of hard-coded house effects, allow the random walk drift term to include a sub-group level offset (by region or primary recall).
- *Warning*: Maintain regularization (Horseshoe priors) to prevent overfitting the model to just Atlas and Invamer's respondent pools.