# 2026 Presidential Election Prediction Plan: Integrating CNE Microdata

## 1. Data Ecosystem and Context

To build our 2026 predictions, we will triangulate three data sources:
1. **The CNE Microdata (`data/2026-polls/`):** 23 official polls spanning March 10 to May 21, 2026. This dataset provides raw respondent-level data for major firms (Atlas Intel, Invamer, CNC, GAD3) and unlocks the ability to observe exact head-to-head runoff margins and demographic covariates.
2. **La Silla Vacía Aggregator (`silla_vacia_ponderador/`):** Provides a pre-computed topline baseline and an authoritative pollster quality rating methodology. This serves as our cross-validation benchmark.
3. **AS/COA Timeline & Framing:** Confirms the operative election timeline—First Round on **May 31, 2026**, and Runoff on **June 21, 2026**—and establishes that the runoff pairing we must optimize for is Iván Cepeda vs. Abelardo de la Espriella (or Paloma Valencia).

---

## 2. Consolidated Execution Plan

The integration of the CNE microdata into our existing MVP specifications (SPECs 02–10) will be handled in five phases. All identified improvements and blind spots are integrated directly into these execution steps to ensure backwards compatibility with the 2022 project framework.

### Phase 0: Ingest, Extraction & Normalization
**Goal:** Convert disparate formats (CSV, Excel, SPSS, PDF) into a unified Parquet dataset.
- **Module:** Create `src/co_president/data_cne_2026.py` as a parallel addition to `data.py` to maintain backwards compatibility (accepting a `cycle` parameter to avoid breaking the 2022 pipeline).
- **Tasks:**
  1. **Canonical Candidate Normalizer:** Create a robust dictionary mapping string variants. **Improvement:** This normalizer must rigorously map "Voto en Blanco", "No Sabe / No Responde", and "Ninguno" into standard `blanco` and `ns_nr` buckets, which are eventually grouped into the `rest` category for the K=3 runoff model.
  2. **Firm-Specific Loaders:** Write tailored extractors for Atlas (CSV), Invamer (Excel + dict), CNC (`pyreadstat`), and GAD3 (Excel).
  3. **Table Extraction for PDF-Only Polls (Improvement):** For the 11 PDF-only bundles (e.g., Guarumo, Génesis Crea), avoid data loss by using table-extraction tools (like `pdf-reader_read_pdf` with `include_tables=True`). Attempt to extract head-to-head runoff matrices from these PDFs to maximize SPEC-08 data, rather than strictly falling back to toplines.
  4. **Topline Extraction:** Compute weighted percentages per poll using respondent *design weights* (micro-weights), keeping them strictly separate from the *macro-weights* (recency, quality) applied later in SPEC-05.
- **Deliverables:** `2026_topline.parquet` and `2026_runoff_pairings.parquet`.

### Phase 1: Cross-Validation & Calibration
**Goal:** Reconcile our extracted microdata toplines with La Silla Vacía's published aggregator to ensure methodological soundness.
- **Tasks:**
  1. Join `2026_topline.parquet` with La Silla Vacía's `encuestas_detalle.csv`.
  2. Flag and resolve discrepancies > 0.5pp (checking for differences in likely-voter screens or handling of undecideds).
  3. **Update `POLLSTER_RATINGS` (SPEC-02):** Integrate the 2026 firm ratings (e.g., Atlas Intel 5.9, CNC 5.8) into the configuration module, safely sequestered from the 2022 rubrics.

### Phase 2: Backfill Pipeline (SPEC-04 & SPEC-05)
**Goal:** Route the cleaned 2026 data into the downstream Bayesian models.
- **Tasks:**
  1. Implement `build_clean_polls_2026()` to structure the data into `CleanPolls(round1=..., round2=...)`.
  2. Run the baseline weighted average (SPEC-05) on the 2026 data.
  3. **Timeline Alignment (Improvement):** Parameterize the time axis `T` so the models correctly extrapolate exactly to May 31 (R1) and June 21 (Runoff).

### Phase 3: Empirical Runoff Matrix (SPEC-08 Uplift)
**Goal:** Upgrade SPEC-08 by feeding it directly observed head-to-head polling responses, bypassing the need to simulate the runoff purely from first-round posteriors.
- **Tasks:**
  1. Aggregate the runoff pairing responses (`2026_runoff_pairings.parquet`) across polls, using SPEC-05's macro-weights (recency, quality).
  2. Create an `empirical_runoff_matrix()` function.
  3. For pairings with direct polling data (e.g., Cepeda vs. De la Espriella), use the empirical Beta posterior calculated from respondent microdata. For unpolled fringe pairings, fall back to the existing SPEC-08 Dirichlet-Multinomial simulation.

### Phase 4: Validation & Forecasting (SPEC-09 & SPEC-10)
**Goal:** Generate final 2026 predictions and validate against aggregator consensus.
- **Tasks:**
  1. Fit the SPEC-06 (First Round) and SPEC-07/08 (Runoff) Bayesian models on data up to the May 21 field date boundary.
  2. Compare the posterior predictive medians to La Silla Vacía's `Prediccion_30d` snapshot for June 21 to ensure the model behaves reasonably.
  3. Generate the final prediction reports based on the AS/COA three-way race framing.

### Phase 5 (Stretch): Demographics & Covariates
**Goal:** Enhance the reverse-time random walk with respondent-level signals.
- **Tasks:**
  1. Incorporate March 8 consultation recall (`vote_primaries_2026` from the Atlas and Invamer microdata) as an anchoring prior or hierarchical offset in the runoff model.
  2. Add regional clustering covariates to the Dirichlet-Multinomial structure.