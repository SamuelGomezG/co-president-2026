# Design Decisions and Literature Support

This document maps every key design decision in the co-president-2026 model to the specific paper(s) that establish or support it. For narrative detail, see `key-findings.md` (topic-area synthesis). For full citations, see `bibliography.md`.

## Compositional Data & Likelihood

| Decision | Paper(s) | Rationale | Applied in |
|----------|----------|-----------|------------|
| Dirichlet-Multinomial likelihood | [Pollsposition], [CoElect], [ElexModel] | Standard for compositional count data. Handles overdispersion. | `model_round1.py`, `model_runoff_simple.py` |
| Single φ_poll with sample-size scaling | [Aitchison1982], [Pollsposition] | Shared concentration across candidates; scaled by `log(n+1)` for sample-size effects. | `model_round1.py:259-278` |
| Softmax logit → probability | [Aitchison1986] | Inverse of ALR, maps ℝ^{K-1} to simplex. | `model_round1.py:305` |
| `apply_zero_floor` for zero shares | [SmithsonVerkuilen2006], [ModelosMixtos] | Prevents log(0) = -∞ in Dirichlet-Multinomial. Redistributes from largest column. | `fundamentals/compositional.py` |
| CLR transform for municipal features | [USANTOMAS2025], [Aitchison1982] | Eliminates spurious correlations. R² improves ~10-15pp vs raw shares. | `fundamentals/features.py`, `model_municipal.py` |

## Random Walk & Time Dynamics

| Decision | Paper(s) | Rationale | Applied in |
|----------|----------|-----------|------------|
| Reverse-time RW (anchor at election) | [CoElect], [Forecast2016] | Uncertainty grows backward from election day. Matches forecasting intuition. | `model_round1.py:337-371`, `model_runoff_simple.py:280-302` |
| Non-centered parameterization | [Betancourt2017], [Papaspiliopoulos2007] | Breaks posterior correlation in RW. Enabled R-hat < 1.01. | `model_round1.py:339-356`, `model_runoff_simple.py:280-302` |
| RW drift for trend extrapolation | Standard time-series [Hamilton1994] | Campaigns are non-stationary. Drift extrapolates observed poll trend (Petro 0.095pp/day). | `model_runoff_simple.py:292` |
| Informative prior at oldest time point | [CoElect], [GelmanHill2007] | Consultation votes + municipal model provide real-voter-behavior prior. | `model_round1.py:297-329` |

## House Effects

| Decision | Paper(s) | Rationale | Applied in |
|----------|----------|-----------|------------|
| Hierarchical per-pollster bias | [538Model], [Pollsposition], [CoElect] | Each pollster has systematic error. Shared σ_house learns typical bias scale. | `model_round1.py:360-374`, `model_runoff_simple.py:307-320` |
| Zero-sum constraint | [Pollsposition], [CoElect] | Identifies house effects: Σ_p effects[p] = 0 per candidate. | `model_round1.py:366` |

## Election Likelihood

| Decision | Paper(s) | Rationale | Applied in |
|----------|----------|-----------|------------|
| Fixed φ_elec = 5,000 | [Gelman2013], [Pollsposition] | Moderate constraint (~0.7pp SE). No funnel geometry (0 divergences vs Gamma's 1,106). | `model_round1.py:428-458` |
| Fixed-path threshold (φ > 0) | Experience from Gamma(0.75) divergences | Shape < 1 creates density spike → funnel. Fixed path eliminates. | `model_round1.py:42`, `model_runoff_simple.py:41` |
| Backtest only; none in forward mode | [GelmanHill2007] | Forward prediction has no election results. Likelihood added only when `results is not None`. | `model_round1.py:416`, `model_runoff_simple.py:425` |

## Digital Signals

| Decision | Paper(s) | Rationale | Applied in |
|----------|----------|-----------|------------|
| T-1 peak decay: exp(-abs(days-1)/3) | [SciELO2023] | T-1 has 1.86pp error, election day has 6.56pp. Peak at T-1, not election day. | `model_round1.py:462-498`, `model_runoff_simple.py:367-384` |
| Single-name query maps | [SciELO2023] | "Petro" outperforms "Gustavo Petro" in search frequency and predictiveness. | `trends_keywords.py` |
| Internet rate weighting (Option C) | [SciELO2023], [DANECNPV2018] | Digital signals only measure online population. Weight = 1 - α(1-rate). | `model_round1.py:475` helper, `model_runoff_simple.py:356-358` |
| Digital signals disabled in runoff | Empirical (2022 bias) | Rodolfo bias: T-1 signal showed Rodolfo +1.7-4.2pp, actual was Petro +3.07pp. | `model_runoff_matrix.py:534-536` |
| `concentration_t1_boost = 0.0` | Empirical (2022 bias) | Separate T-1 Beta likelihood was double-counting biased T-1 signal. | `config.py:105` |

## Runoff Model

| Decision | Paper(s) | Rationale | Applied in |
|----------|----------|-----------|------------|
| K=3 (A, B, rest+blanco) | [CoElect], empirical | Rest captures blank+null votes (~2.3%). Prevents both candidates' shares from inflating. | `model_runoff_simple.py:173-174` |
| Rest_blanco prior override | Empirical (historical R2 rest rate) | R1-implied rest (~32%) massively overestimates runoff rest (~2.3%). Override with 2.3% prior. | `model_runoff_simple.py:251-258` |
| Transfer-implied prior (not R1 shares) | [King1997], [ASCOA] | R1 shares are wrong for runoff. Use R1 posterior + transfer model for realistic prior. | `model_runoff_matrix.py:395-405` |
| `round2_result = None` (honest) | [GelmanHill2007] (Bayesian principles) | No R2 data enters the model. R2 is evaluation-only. | `model_runoff_matrix.py:540` |

## Transfer Model

| Decision | Paper(s) | Rationale | Applied in |
|----------|----------|-----------|------------|
| Bayesian ecological inference | [King1997] | Individual voter transitions inferred from aggregate municipal data. | `model_transfer.py` |
| Exclude 2022 from training | Standard CV | Target year is held out. Trained on 2010/2014/2018. | `model_transfer.py:321,364` |
| Cámara + Senado legislative features | [USANTOMAS2025], Colombian PRR | Cámara = local machine influence, Senado = national ideology. | `model_transfer.py:58-64` |
| Convergence guardrail (R-hat < 1.10) | [Pollsposition], reviewer feedback | Falls back to Dirichlet-Categorical if transfer model fails to converge. | `model_transfer.py:491-493` (concept), external validation |

## Sampling & Diagnostics

| Decision | Paper(s) | Rationale | Applied in |
|----------|----------|-----------|------------|
| numpyro JAX backend | [numpyro], [HoffmanGelman2014] | 500+ it/s vs 54 it/s for PyMC C NUTS. Faster convergence at 5K draws. | `config.py:112` |
| target_accept = 0.95 | [HoffmanGelman2014], [StanRef] | Higher acceptance reduces divergences while maintaining efficiency. | `config.py:111` |
| target_accept = 0.9 (municipal) | Experience | Horseshoe prior on mu_m_raw (1,155 params) makes adapt_full infeasible. | `model_round1.py:109` |
| R-hat < 1.01, ESS > 400, 0 divergences | [StanRef], [Gelman2013] | Standard convergence criteria. All verified per model run. | `sampling-strategy.md` |

## Polling Aggregation (Baseline)

| Decision | Paper(s) | Rationale | Applied in |
|----------|----------|-----------|------------|
| Time decay 0.5^(days/30) | [538Model], [CoElect] | 30-day half-life, exponential decay. | `aggregation.py` |
| Sample size weight log(n+1) | [538Model], [Pollsposition] | Diminishing returns on sample size. | `aggregation.py` |
| Pollster weight = 0.02·rating + 0.8 | [538Model], [LaSillaVacia], [LaSillaVacia2026] | Constrained to [0.82, 1.0]. Prevents single pollster dominance. 2026 methodology: in-person 1.0, phone 0.8, online 0.75. | `config.py:375-399`, `data/2026-polls/silla_vacia_ponderador/` |

## Municipal Model

| Decision | Paper(s) | Rationale | Applied in |
|----------|----------|-----------|------------|
| Shared β across 1,122 municipalities | [ModelosMixtos], [USANTOMAS2025] | Learns country-wide patterns. Non-centered mu_m_raw for local deviation. | `model_municipal.py` |
| NBI as primary poverty feature | [DANENBI2018], reviewer feedback | Complete municipal coverage, 0% imputation vs IPM's ~80% imputation. | `fundamentals/features.py`, `model_municipal.py` |
| 11 demographic + 1 historical features | [USANTOMAS2025], [BaqueroRosero2019] | Proven set from academic benchmark. | `model_municipal.py:57-71` |
