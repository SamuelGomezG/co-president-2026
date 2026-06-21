# Election Results Data

## 2022 Election Results

### Registraduría Nacional (MMV Files)

**Provenance**: Registraduría Nacional del Estado Civil. Official election results at the polling-station level.

**Local files**: `data/2022-presidential-results/MMV_NACIONAL_PRESIDENTE_2022_1v.csv.gz` (R1, ~727K rows), `_2v.csv.gz` (R2, ~398K rows). Not tracked in git. Must be decompressed once after cloning.

**Format**: ISO-8859-1, semicolon-delimited. 16 columns including `CANNOMBRE`, `CAN` (candidate code), `VOTOS`.

**Key findings**:
- 99.9% convergence rate between pre-count (preconteo) and official scrutiny (escrutinio) — single-source reliability confirmed
- Round 1: Petro 40.34%, Hernández 28.15%, Gutiérrez 23.89%, Fajardo 4.39%, Blanco 2.72%
- Round 2: Petro 50.42%, Hernández 47.35%, Rest (blank+null) 2.23%
- The rest category (~2.3%) is used as the prior anchor for the K=3 runoff model's rest_blanco parameter

**Features extracted**: Vote shares → election likelihood anchor in model. Distinct shares per round. Total votes for DirichletMultinomial n. blank+null rate for runoff rest prior.

### MOE Results (CSV + PDF)

**Provenance**: Misión de Observación Electoral (MOE). Independent validation source.

**Local files**: `data/2022-presidential-results/moe_vuelta1.csv`, `moe_vuelta2.csv`, `2022.11.09-LIBRO-RESULTADOS-ELECTORALES-PRESIDENCIALES-2022.pdf` (MOE libro oficial). Not tracked in git.

**Format**: UTF-8, comma-delimited. Municipal-level (~12K R1, ~5.5K R2).

**Key findings**: Two-way cross-validation between Registraduría and MOE produces zero warnings: total votes match within 0.1%, per-candidate shares within 0.5%. The PDF provides a third-way validation for the final canonical total.

### Participation Files

**Local files**: `data/2022-presidential-results/reg_participacion_vuelta1.csv`, `_2.csv`. Not tracked in git.

**Key findings**: Registered voters ~39M. R1 turnout ~54%, R2 turnout ~55%. Polling stations ~99K across both rounds. Used to compute `RoundResult.registered_voters` and `turnout()`.

---

## 2026 Election Results

### 2026 Presidential (First Round)

**Provenance**: Registraduría Nacional. Pre-count (preconteo) results, department-level MMV files.

**Local files**: `data/2026-elections/2026-presidential1/` — MMV files per department. Not tracked in git.

**Format**: Similar to 2022 MMV files (semicolon-delimited, ISO-8859-1). Available per department as separate CSVs.

**Key findings** (to be extracted):
- First round results available — enables 2026 R1 backtest
- Departmental granularity same as 2022
- Cross-validation against MOE pending

### 2026 Legislative Results (Cámara + Senado)

**Local files**: `data/2026-elections/2026-legislative/` — MMV files per department for legislative elections. Not tracked in git.

**Key findings** (to be extracted):
- Cámara de Representantes and Senado results available
- Enables SPEC-30 transfer model training on 2026 legislative data
- Department-level party vote shares for gamonalismo analysis

---

## Data Processing Pipeline

Both years follow the same ingestion path: `load_canonical_results()` → cross-validate Registraduría vs MOE → produce `RoundResult`. For 2026, the department-level MMV files must be concatenated before aggregation.

**Cross-validation logic** (`data_results.py:cross_validate()`):
```
if |total_votes_reg - total_votes_moe| / total_votes_reg > 0.001 → WARNING
if any |candidate_share_reg - candidate_share_moe| > 0.005 → WARNING
```

**References**: See `docs/reference/key-findings.md` §10 (Data Sources) for validation methodology. Features documented in `docs/architecture/features.md` §3 (Prior Features).
