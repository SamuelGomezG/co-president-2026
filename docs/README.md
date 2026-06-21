# docs/ — Project Documentation

## Architecture (why & how)

| Document | Content |
|----------|---------|
| `architecture/overview.md` | Pipeline, data flow, model layers diagram |
| `architecture/model-design.md` | Why Dirichlet-Multinomial, RW, K=3, drift, non-centered parameterization |
| `architecture/honest-forecast.md` | No-R2-leakage design, transfer prior, drift, rest override, blanco override |
| `architecture/features.md` | Complete catalog of every model feature by module |
| `architecture/sampling-strategy.md` | numpyro, non-centered RW, φ_elec divergence fix, convergence diagnostics |
| `architecture/packages.md` | Package organization rationale, test structure |

## Results (what we achieved)

| Document | Content |
|----------|---------|
| `results/2022-backtest.md` | Full backtest results (R1 MAE 0.14pp, runoff MAE 0.68pp, P(Petro)=57.7%) |
| `results/2022-forward.md` | True forward forecast (R1 MAE 2.18pp, runoff MAE 0.77pp, P(Petro)=56.9%) |

## Specs & Status

| Document | Content |
|----------|---------|
| `../MVP_SPECS_GUIDE.md` | SPEC-01 through SPEC-11 (MVP): authoritative design specification |
| `specs/STATUS.md` | All SPECs (01-41) with implementation status |

## Plans (future work, NOT yet applied)

| Document | Content |
|----------|---------|
| `plans/cne_2026_ingestion.md` | SPEC-31→36: CNE microdata integration for 2026 forecast |
| `plans/demographic_ingestion.md` | SPEC-15→30: Demographic data ingestion plan (mostly shipped) |

## Data Reports

| Document | Content |
|----------|---------|
| `data/README.md` | Index of all data source reports |
| `data/polls.md` | Poll data: 2022 CSV, 2026 CNE microdata, La Silla Vacía |
| `data/election-results.md` | Official results: Registraduría, MOE, participation |
| `data/census-demographics.md` | DANE census data: CNPV 2018, NBI, IPM, population |
| `data/historical-results.md` | Historical: CEDAE 2002-2018, MOE legislative, Sabaneta |
| `data/risk-and-conflict.md` | MOE risk, PDET, INDEPAZ, UNODC |
| `data/digital-signals.md` | Google Trends via pytrends |
| `data/geographic-reference.md` | DIVIPOLA registry, Bogotá localidades |
| `data/2026-candidate-config.md` | 2026 timeline, pollster ratings, differences from 2022 |

## Reference (research & background)

| Document | Content |
|----------|---------|
| `docs/reference/index.md` | Complete catalog of every file in the reference/ directory. |
| `docs/reference/key-findings.md` | Narrative synthesis of key findings from all reference literature. |
| `docs/reference/bibliography.md` | Full academic citations with citation keys for every cited work. |
| `docs/reference/adopted.md` | Quick-reference: design decision → supporting paper(s). |

**Note**: `reference/` (root) contains the actual PDFs and tech guides — those files are gitignored (external material). Our derived analytical documents are in `docs/reference/`.

## Archive (superseded or historical)

| Document | Superseded by |
|----------|---------------|
| `archive/ARCHITECTURE_PLAN.md` | `architecture/` docs |
| `archive/ENHANCEMENT_ROADMAP.md` | `specs/STATUS.md` + `plans/` |
| `archive/2026-05-CNE-data-plan.md` | `plans/cne_2026_ingestion.md` |

## Other

- `data/processed/feature_dictionary.md` — auto-generated data dictionary
- `data/2022-polls/as_coa/README.md` — AS/COA poll source documentation
