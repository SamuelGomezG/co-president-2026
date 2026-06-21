# SPEC Status Tracker

> **Source of truth**: See `MVP_SPECS_GUIDE.md` (SPEC-01→11), `docs/plans/demographic_ingestion.md` (SPEC-15→30),
> `docs/plans/cne_2026_ingestion.md` (SPEC-31→36).

## MVP — National Polling Model (Shipped)

| SPEC | Title | Modules | PRs | Status |
|------|-------|---------|-----|--------|
| SPEC-01 | Project scaffolding | `paths.py`, `pyproject.toml` | #1, #155, #128 | ✅ Shipped |
| SPEC-02 | Configuration | `config.py` | #16, #30, #74, #92 | ✅ Shipped |
| SPEC-03 | Results consolidation | `data_results.py` | #32, #71, #77 | ✅ Shipped |
| SPEC-04 | Poll loading & cleaning | `data_polls.py` | #33, #71, #91, #93 | ✅ Shipped |
| SPEC-05 | Baseline aggregation | `aggregation.py` | #100 | ✅ Shipped |
| SPEC-06 | 1st-round Bayesian model | `model_round1.py` | #101, #102, #103, #104, #105 | ✅ Shipped |
| SPEC-07 | Runoff simple (K=3) | `model_runoff_simple.py` | #106 | ✅ Shipped |
| SPEC-08 | Runoff pairing matrix | `model_runoff_matrix.py` | #107 | ✅ Shipped |
| SPEC-09 | Validation & plotting | `validation.py`, `plotting.py` | #108, #109 | ✅ Shipped |
| SPEC-10 | CLI | `__main__.py` | #110 | ✅ Shipped |
| SPEC-11 | Data quality diagnostics | `_data_quality.py` | — | ✅ Shipped |

## Post-MVP — Municipal Hierarchical Model (Shipped)

| SPEC | Title | Modules | PRs | Status |
|------|-------|---------|-----|--------|
| SPEC-12.1 | DIVIPOLA master registry | `ingest_divipola.py` | #220 | ✅ Shipped |
| SPEC-12.2 | Historical election results | `ingest_historical.py`, `_download_cedae.py` | #220 | ✅ Shipped |
| SPEC-13a | Socioeconomic demographics | `ingest_socioeconomic.py` | — | ⚠️ Shipped (stub fallback) |
| SPEC-13.2 | Extended risk (MOE) | `ingest_risk.py` | — | ✅ Shipped |
| SPEC-14 | Feature matrix builder | `build_feature_matrix.py` | — | ✅ Shipped |
| SPEC-15 | Coverage report | `report.py` | #220 | ✅ Shipped |
| SPEC-16 | Sabaneta fixture loader | `ingest_sabaneta.py` | #227 | ✅ Shipped |
| SPEC-17 | CNPV 2018 census | `ingest_cnpv.py`, `scripts/download_cnpv_2018.py` | #228 | ✅ Shipped |
| SPEC-18 | NBI + IPM poverty | `ingest_nbi.py`, `ingest_ipm.py` | #229 | ✅ Shipped |
| SPEC-19 | Population projections | `ingest_population.py` | — | ✅ Shipped |
| SPEC-20 | Extended risk (conflict) | `ingest_risk.py` | #231 | ✅ Shipped |
| SPEC-21 | Municipal features API | `fundamentals/features.py` | — | ⚠️ Shipped (needs real-data smoke tests) |
| SPEC-21a | Fiscal autonomy | `ingest_fiscal.py` | #237 | ✅ Shipped |
| SPEC-21b | Bogotá disaggregation | `ingest_bogota.py` | #238 | ✅ Shipped |
| SPEC-21c | CLR/logit transforms | `fundamentals/features.py` | #240 | ✅ Shipped |
| SPEC-22 | Municipal hierarchical model | `model_municipal.py` | #241 | ✅ Shipped |

## Extension Shipments (Shipped)

| SPEC | Title | Plan doc | Status |
|------|-------|----------|--------|
| SPEC-23 | MOE PDF cross-validation | `docs/plans/demographic_ingestion.md` | ✅ Shipped |
| SPEC-24 | ECP cultural attitudinal signal | `docs/plans/demographic_ingestion.md` | ✅ Shipped |
| SPEC-25 | AS/COA third poll source | `docs/plans/demographic_ingestion.md` | ✅ Shipped |
| SPEC-26 | OOS validation framework | `docs/plans/demographic_ingestion.md` | ✅ Shipped |
| SPEC-27 | Regularization (horseshoe) | `docs/plans/demographic_ingestion.md` | ✅ Shipped |
| SPEC-28 | 2026 forecast mode | `docs/plans/demographic_ingestion.md` | ✅ Shipped |
| SPEC-29 | Generalization audit | `docs/plans/demographic_ingestion.md` | ✅ Shipped |

## Digital Signals (Shipped)

| SPEC | Title | Modules | PRs | Status |
|------|-------|---------|-----|--------|
| SPEC-38 | Google Trends ingestion | `ingest_trends.py`, `trends_keywords.py`, `aggregation.py`, `__main__.py` | #254 | ✅ Shipped |
| SPEC-39 | Twitter sentiment pipeline | `twitter_preprocess.py`, `twitter_denoise.py`, `bert_sentiment.py`, `llm_sentiment.py`, `sentiment_series.py` | #260 | ✅ Shipped |

## Compositional & ML Benchmarks (Shipped)

| SPEC | Title | Modules | PRs | Status |
|------|-------|---------|-----|--------|
| SPEC-37 | Compositional data transforms (ALR/ILR) | `benchmarks/transforms.py` | #253 | ✅ Shipped |
| SPEC-40 | Multi-election model refactor | `model_round1.py`, `model_municipal.py` | #256 | ✅ Shipped |
| SPEC-41 | FNN+CLR ML benchmark | `benchmarks/fnn_clr.py`, `benchmarks/runner.py` | #257 | ✅ Shipped |
| SPEC-41a | SVR baseline | `benchmarks/runner.py` | #257 | ✅ Shipped |
| SPEC-41b | Random Forest baseline | `benchmarks/runner.py` | #257 | ✅ Shipped |
| SPEC-41c | Gradient Boosting + KNN baselines | `benchmarks/runner.py` | #257 | ✅ Shipped |

## Current Work

| SPEC | Title | Plan doc | Status |
|------|-------|----------|--------|
| SPEC-30 | Transfer rate estimation | `docs/plans/demographic_ingestion.md` | ✅ Shipped |

## Planned (not yet implemented)

| SPEC | Title | Plan doc | Status |
|------|-------|----------|--------|
| SPEC-31 | CNE ingest & normalization | `docs/plans/cne_2026_ingestion.md` | 📋 Planned |
| SPEC-32 | CNE cross-validation & calibration | `docs/plans/cne_2026_ingestion.md` | 📋 Planned |
| SPEC-33 | Backfill pipeline | `docs/plans/cne_2026_ingestion.md` | 📋 Planned |
| SPEC-34 | Empirical runoff matrix | `docs/plans/cne_2026_ingestion.md` | 📋 Planned |
| SPEC-35 | 2026 validation & forecasting | `docs/plans/cne_2026_ingestion.md` | 📋 Planned |
| SPEC-36 | Stretch: demographics | `docs/plans/cne_2026_ingestion.md` | 📋 Planned |
