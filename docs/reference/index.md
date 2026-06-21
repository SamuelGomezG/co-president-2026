# Reference Catalog

Complete catalog of every file in this directory, organized by topic. Each entry includes the file name, a synopsis, and its relationship to the co-president-2026 model.

## Colombian Academic Papers

These PDFs form the core academic foundation for the project's methodology. They are cited throughout `docs/architecture/` and `docs/reference/key-findings.md`.

| File | Synopsis | Model relation |
|------|----------|----------------|
| `scielo_search_patterns_prediction_colombia_2022.pdf` | Google Trends T-1 has 1.86pp error (vs. polls' 6.56pp). Single-name queries, T-1 peak, internet weighting. | **Adopted**: digital signal decay peaks at T-1, single-name query maps, internet rate weighting. Disabled for 2022 runoff (bias documented). |
| `usantotomas_comparacion_modelos_machine_learning.pdf` | FNN+CLR achieves R² 0.94 for ideological vote share. ML benchmark comparison. Municipal outperforms national. | **Adopted**: municipal hierarchical model, CLR for poverty/historical features, non-centered mu_m_raw. |
| `modelos_mixtos_datos_composicionales_aplicacion_electoral.pdf` | Mixed-effects regression on ILR-transformed votes (R² 0.93). Department-level random intercepts. | **Adopted**: shared β coefficients across municipalities, zero-sum house effects. |
| `plebiscito_paz_colombia_analisis_datos_composicionales.pdf` | Compositional data analysis applied to 2016 Colombian peace plebiscite. Aitchison distance metrics. | **Adopted**: Dirichlet-Multinomial likelihood validates non-raw-share approach. |
| `sentiment_analysis_model_of_spanish_tweets_2014_elections.pdf` | Twitter sentiment as Bayesian prior, not poll substitute. Spanish-specific tooling required. | **Future work**: sentiment pipelines exist but are not wired into model. |
| `arxiv_emotions_twitter_colombia_2022.pdf` | Emotion-labeled Twitter corpus (1,200 tweets, 4 emotions). RoBERTuito F1 0.729, GPT-4 F1 0.752. | **Future work**: emotion series can feed Bayesian prior in 2026 mode. |
| `ockham_rodolfo_sentiment_analysis_2022.pdf` | Sentiment analysis of Rodolfo Hernández tweets. 60% positive, "trust" dominant emotion. | **Reference**: confirms sentiment captures candidate-specific dynamics. |
| `prediccion_electoral_modelo_hibrido_analisis_sentimental_seguimiento_encuestas.pdf` | LSTM hybrid (LSVM + Prophet + LSTM): 2.47% RMSE, beats best pollster (Invamer 4.66%). | **Reference**: validates hybrid signal-polls approach for future implementation. |
| `Aprendizaje_de_Maquinas_para_la_Predicci.pdf` | Machine learning for electoral prediction (general). Holt-Winters imputation. | **Reference**: imputation methodology for missing municipality data. |
| `DANE_IPM_2018.pdf` | DANE's Multidimensional Poverty Index methodology for 2018 census. | **Reference**: secondary poverty feature (NBI is primary). |

## Colombian Data Methodology

| File | Synopsis | Model relation |
|------|----------|----------------|
| `data/2026-polls/silla_vacia_ponderador/Nota-tecnica-ponderacion-encuestas-Presidenciales-2026-V-Mar27.docx-1.pdf` | La Silla Vacía 2026 pollster weighting methodology: in-person 1.0, phone 0.8, online 0.75. Weekly ponderación + Predicción 30d. | **Adopted**: weighting formula extends 2022 POLLSTER_RATINGS methodology. Cross-validation workflow for 2026 CNE integration. |

## International Tech Guides

These describe the methodology of established election forecasting models. They informed the model architecture, particularly the Dirichlet-Multinomial, house effects, and polling aggregation.

| File | Source | Synopsis | Model relation |
|------|--------|----------|----------------|
| `538model-tech_guide.md` | Silver, N. / FiveThirtyEight | House effects (PIE), time decay, state correlation, pollster weights. | **Adopted**: time decay (0.5^(days/30)), pollster quality weighting, state-level effects pattern. |
| `co_elect-tech_guide.md` | Nelson Amaya / recetas-electorales | Ajiaco model: Dirichlet-Multinomial + reverse-time RW for Colombian elections. | **Adopted**: core model architecture (DirMult + RW + house effects), consultation prior, undecided redistribution formula. |
| `pollsposition_models-tech_guide.md` | Various | Dirichlet-Multinomial with per-candidate φ_poll, GP time dynamics, ZeroSumNormal. | **Adopted**: single φ_poll with sample-size scaling. **Not adopted**: multi-lengthscale GP, per-candidate φ_poll. |
| `elex_model-tech_guide.md` | Various (BootstrapElectionModel) | Stratified residual bootstrap, joint vote + turnout model. | **Reference**: bootstrap methodology for live election night (future). |
| `forecast_2016-tech_guide.md` | 2016 US forecast methodology | Power prior `a_0`, logit-Normal MH, poll decay auto-tuning. | **Adopted**: poll-decay concept. **Not adopted**: auto-tuned a_0 (we use fixed 30-day half-life). |
| `2024_potus-tech_guide.md` / `us_potus_tech_guide.md` / `us_election2024-tech_guide.md` | US presidential election forecasting | Various: copula transforms, spatial GP, multi-lengthscale kernels. | **Reference**: future feature engineering (copula for bounded proportions, GP for spatial effects). |

## Research Compilations

| File | Synopsis | Model relation |
|------|----------|----------------|
| `Colombia_Presidential_Predictor_Research_Reference.md` | Master reference document: 14 sections covering all data categories, sources, methodology notes, and academic benchmarks for Colombian presidential election prediction. | **Adopted**: cross-validation strategy, data source mapping, turnout modeling framework, legislative feature design (Cámara + Senado). |
| `colombia_predictor_urls.txt` | 189 URLs for Colombian election data sources. | **Reference**: planned data acquisition for 2026. |

## Derived Analysis

| File | Location | Synopsis |
|------|----------|----------|
| `key-findings.md` | `docs/reference/key-findings.md` (this directory) | Narrative synthesis of key findings from all reference sources, organized by topic area, with adoption status. |
| `bibliography.md` | `docs/reference/bibliography.md` (this directory) | Full academic citations with citation keys for every referenced work. |
| `adopted.md` | `docs/reference/adopted.md` (this directory) | Quick-reference table: design decision → supporting paper(s) → rationale. |
| `gap_analysis_pdfs_2026-06-12.md` | `docs/reference/gap_analysis_pdfs_2026-06-12.md` | P1-P68 gap analysis: 68 techniques from 8 PDFs with implementation status. Derived from scanning all PDFs in this directory. |

## Directory Index

- `reference/` (this directory): 11 PDFs, 6 markdown tech guides, 1 research reference, 1 URL list, 3 summary documents (index, key-findings, bibliography, adopted), 1 `.keep` placeholder.
- `docs/architecture/`: Architectural rationale docs that cite these references.
- `docs/reference/`: `gap_analysis_pdfs_2026-06-12.md` (derived gap analysis).
