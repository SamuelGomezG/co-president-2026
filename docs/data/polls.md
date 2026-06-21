# Poll Data Sources

## 2022 Polls

### encuestas_2022.csv

**Provenance**: recetas-electorales.com (Nelson Amaya). Compiled from publicly available Colombian presidential polls.

**License**: Third-party compiled data — not tracked in git. Download from recetas-electorales.com.

**Format**: 46 rows, 27 columns. UTF-8, comma-delimited. Columns: fecha, encuestadora, muestra, muestra_int_voto, margen, metodo, and candidate share columns.

**Key findings**:
- 30 first-round polls (post-consultation, post-03-13) from 7+ pollsters
- 19 runoff polls (post-05-29) from 6+ pollsters
- Known data-entry error: Invamer poll dated 2022-04-19 corrected to 2022-05-19 (confirmed against co_elect source)
- GAD3 contributed the most polls (11 of 19 runoff polls)

**Features extracted**:
| Feature | Module | How computed |
|---------|--------|-------------|
| fecha | `data_polls.py:load_raw_polls()` | Parsed via `pd.to_datetime()` |
| encuestadora, muestra, candidate shares | `data_polls.py` | Normalized, rounded to counts |
| days_before, time_idx, pollster_idx | `model_round1.py:202-229` | Derived during model build |
| rest_blanco (runoff K=3) | `model_runoff_simple.py:173` | Sum of eliminated candidates |

**Ingestion pipeline**: `load_and_clean_all()` → `data_polls.py`:
1. `load_raw_polls()` — reads CSV, standardizes columns
2. `fix_invamer_date()` — corrects known date error
3. `normalize_undecided()` — redistributes ns_nr proportionally
4. `infer_round_number()` — classifies R1 vs R2
5. `deduplicate_polls()` — keeps largest sample per pollster+date
6. `retain_active_candidates()` — drops withdrawn candidates

### consultas.csv

**Provenance**: recetas-electorales.com. Pre-election consultation polls (March 13, 2022).

**License**: Same as encuestas_2022.csv — not tracked in git.

**Format**: 65 rows. ISO-8859-1, comma-delimited.

**Key findings**:
- Coalition breakdown enables informative prior for theta[T-1]
- Pacto Histórico ~5.5M votes → Petro's consultation base
- Equipo por Colombia ~3.8M → Gutiérrez's base
- Centro Esperanza ~1.8M → Fajardo's base
- Zero-vote candidates (Hernández, Betancourt) receive default log-share prior

**Features extracted**: `consultation_log_share_prior()` → `log(votes / sum(votes))` per candidate.

### AS/COA Poll Tracker

**Provenance**: Americas Society / Council of the Americas. [https://www.as-coa.org/articles/poll-tracker-colombias-2022-presidential-election](https://www.as-coa.org/articles/poll-tracker-colombias-2022-presidential-election)

**Local files**: `data/2022-polls/as_coa/round1.csv`, `runoff.csv`, `transfer_matrix.csv` (not tracked in git).

**Key findings**: Third-party poll aggregation and empirical transfer rates. The transfer_matrix.csv is used as a validation target (not source of truth) for the SPEC-30 transfer model.

---

## 2026 Polls

### CNE Microdata (23 ZIP Bundles)

**Provenance**: Consejo Nacional Electoral (CNE) — Colombia's electoral authority. Official poll metadata repository.

**Local files**: `data/2026-polls/` — 23 ZIP bundles, not tracked in git.

**Format**: Mixed — CSV, Excel, SPSS (.sav), PDF. 4 firms provide machine-readable microdata (Atlas Intel, Invamer, CNC, GAD3); 5 firms are PDF-only (Guarumo, Génesis Crea, TEMPO, Corporación MMM, Analizar Lombana).

**Timeline**: March 10 – May 21, 2026. First Round: May 31, Runoff: June 21.

**Key findings**:
- 4 firms with design weights (microdata): Atlas Intel (4 waves), CNC (3 waves), GAD3 (2 waves), Invamer (2 waves)
- 5 PDF-only firms: topline extraction only (no respondent-level covariates)
- silla_vacia_ponderador/ provides reference aggregation for cross-validation
- Primary recall variable (`vote_primaries_2026`) available in Atlas and Invamer microdata — enables hierarchical prior for 2026 runoff

**Features to extract** (planned, SPEC-31):
| Feature | Source | How computed |
|---------|--------|-------------|
| Topline shares per candidate | Microdata via `extract_topline()` | Weighted using respondent design weights |
| Runoff pairings | Microdata via `extract_runoff_pairings()` | Head-to-head scenarios extracted from Q08A-D variables |
| Primary recall | `vote_primaries_2026` field | Hierarchical offset for theta[T-1] prior |
| Pollster methodology flag | Firm metadata | Macro-weights: in-person 1.0, phone 0.8, online 0.75 |

**Ingestion pipeline** (planned):
1. `load_atlas_intel()`, `load_invamer()`, `load_cnc()`, `load_gad3()` — firm-specific extractors
2. `extract_pdf_runoff_tables()` — table extraction for PDF-only firms
3. `extract_topline()` — microdata → weighted percentages
4. `extract_runoff_pairings()` — head-to-head scenarios → Beta posterior
5. Cross-validation against La Silla Vacía aggregator (see `[LaSillaVacia2026]` in reference docs)

### La Silla Vacía Aggregator

**Provenance**: La Silla Vacía. [https://lasillavacia.com/historias/silla-vacia/semaforo-de-encuestas/](https://lasillavacia.com/historias/silla-vacia/semaforo-de-encuestas/)

**Local files**: `data/2026-polls/silla_vacia_ponderador/encuestas_detalle.csv`, `ponderacion.csv` (not tracked in git).

**License**: Third-party compiled data — not tracked in git.

**Methodology**: See technical note in silla_vacia_ponderador/ and `docs/reference/gap_analysis_pdfs_2026-06-12.md` for methodology summary (`[LaSillaVacia2026]` in bibliography). Weighting: in-person 1.0, phone 0.8, online 0.75. Weekly ponderación provides recency-weighted averages. Predicción_30d (June 21) is the validation target for 2026 forecast.

---

## References

Poll data features are documented in `docs/architecture/features.md` §1. For the weighting methodology: `docs/reference/key-findings.md` §7 (Polling Aggregation). Full citations: `docs/reference/bibliography.md` `[CoElect]`, `[LaSillaVacia]`, `[LaSillaVacia2026]`.
