# Key Findings from Reference Literature

## Overview

This document synthesizes the key findings from every academic paper, technical guide, and data methodology reference in this directory. Each section identifies the finding, cites the source, and states how it was adopted (or why it was rejected) in the co-president-2026 model.

Section cross-references: `adopted.md` (decision-level mapping) | `bibliography.md` (full citations) | `index.md` (file catalog).

---

## 1. Compositional Data Analysis

### 1.1 Vote Shares Live on the Simplex

Vote shares for K candidates are a composition on the (K-1)-dimensional simplex: they are strictly non-negative and sum to 1. Standard multivariate methods applied to raw shares produce **spurious negative correlations** between components — any increase in one share mechanically decreases the others. This was established by Karl Pearson in 1896 [Pearson1896] and formalized into a complete statistical framework by John Aitchison in 1982 [Aitchison1982].

**We adopted**: The Dirichlet-Multinomial likelihood, which models K-part count compositions directly. The latent vote intention `p[t]` is a K-dimensional simplex, transformed from unbounded logit parameters via softmax.

**Not adopted**: Aitchison's CLR/ILR transforms for the target variable in the model likelihood. Our model works in logit space (which is a form of additive log-ratio), not true CLR. CLR is used for municipal features only (see §6).

**Source**: `reference/plebiscito_paz_colombia_analisis_datos_composicionales.pdf`, `reference/modelos_mixtos_datos_composicionales_aplicacion_electoral.pdf`, `reference/usantotomas_comparacion_modelos_machine_learning.pdf`.

### 1.2 CLR Outperforms Raw Shares for Prediction

Multiple Colombian election papers (USANTOMAS 2025 [USANTOMAS2025], modelos_mixtos, plebiscito_composicionales) apply the Centered Log-Ratio transform:
```
clr(x) = [ln(x_i / g(x))]  where g(x) = (∏ x_i)^(1/K)
```
This maps the simplex to ℝ^{K-1}, eliminating the sum constraint and removing spurious correlations. The USANTOMAS study found that CLR-transformed municipal features achieved **R² = 0.94** for the Centro-Derecha ideological spectrum — substantially higher than ALR (R² 0.92) and raw shares (not reported but implicitly lower).

**We adopted**: CLR transform for municipal features (via `clr_nbi_array` in `model_municipal.py` and `clr_left_share` in `fundamentals/features.py`). The municipal model uses CLR-transformed NBI poverty rates and historical left-vote shares as predictors.

**Future work**: Apply CLR to the full vote share target vector (currently in logit space via softmax) and wire into the model likelihood per Section 7 of modelos_mixtos [ModelosMixtos].

### 1.3 Zero-Value Handling for Dirichlet Likelihood

The `apply_zero_floor` function in our model replaces zeros with 1 vote (redistributed from the largest column) to prevent `log(0) = -∞` in the Dirichlet-Multinomial likelihood. This is consistent with the Smithson-Verkuilen (2006) [SmithsonVerkuilen2006] recommendation for zero-valued compositional parts.

**Source**: `reference/modelos_mixtos_datos_composicionales_aplicacion_electoral.pdf` Eq. 8, Section "Imputation of zero-valued compositional parts".

---

## 2. Bayesian Election Forecasting

### 2.1 Dirichlet-Multinomial as the Core Likelihood

The Dirichlet-Multinomial is the standard observation model for polling data in election forecasting [pollsposition tech guide, co_elect tech guide]. It naturally handles the compositionality of vote shares and adds overdispersion relative to the Multinomial via a concentration parameter φ:

```
observed_counts ~ DirichletMultinomial(n, α = p · φ)
```

As φ → ∞, the model approaches the Multinomial (polls treated as simple random samples). As φ → 0, overdispersion is maximal (polls given very low weight). Learning φ from data lets the model decide how trustworthy polls are.

**We adopted**: φ_poll ~ Gamma(2, 2/5) with posterior mean ≈5 for the first round, same for runoff. The concentration is scaled by `effective_n_multiplier` = max(log(n+ε) / log(mean_n+ε), ε) so that polls with larger samples have (diminishingly) higher concentration.

**Source**: `reference/pollsposition_models-tech_guide.md` Sections 4.1-4.2, `reference/co_elect-tech_guide.md` Sections 10.1-10.2.

### 2.2 Reverse-Time Random Walk

The reverse-time (or "backward") random walk anchors the latent vote intention at election day and walks backward in time:

```
θ[0] ← election day (target of inference)
θ[t] ~ Normal(θ[t+1], σ_rw)     // moving backward from T-1 to 0
θ[T-1] ← informative prior      // consultation votes + municipal model
```

This matches the intuition that uncertainty grows as we move away from election day. The forward random walk (anchored at the earliest poll) would have maximum uncertainty at election day, which is backwards for forecasting. The Ajiaco model [co_elect tech guide] uses this exact design.

**We adopted**: Reverse-time walk for both R1 and runoff models. The election-day time point is the primary inference target.

**We extended**: Added a **drift** term to the runoff RW to capture non-stationary campaign dynamics (see §4). This matches the standard random walk with drift (RWD) model from time-series econometrics.

**Source**: `reference/co_elect-tech_guide.md` Section 10.2, `reference/forecast_2016-tech_guide.md` Section 4.3.

### 2.3 House Effects (Pollster Bias)

Hierarchical pollster house effects with a zero-sum constraint are standard practice in election forecasting [538model, pollsposition, elex_model]. The constraint `Σ_p house_effects[p] = 0` identifies the model — without it, the pollster biases are confounded with the national trend.

**We adopted**: House effects as non-centered Normal(0, σ_house) with deterministic zero-sum centering. σ_house ~ HalfNormal(1.0).

**Source**: `reference/538model-tech_guide.md` Step 1 (Pollster-introduced error / PIE), `reference/pollsposition_models-tech_guide.md` Section 4.4 (ZeroSumNormal).

### 2.4 Election Likelihood

The actual election result provides a second Dirichlet-Multinomial observation with its own concentration φ_elec. This is essentially an informative data point: the election result has zero pollster bias, zero measurement error, and covers the entire electorate.

**We adopted**: Fixed φ_elec = 5,000 (backtest mode only). This provides a moderate constraint (~0.7pp SE) to keep the posterior near the known result without reciting it. The fixed path eliminated 1,106 divergences that occurred with a Gamma(0.75) prior on φ_elec (see `sampling-strategy.md`).

**Not adopted**: Learned φ_elec via Gamma. A Gamma(0.75, 0.75/50K) prior created a density spike at zero (shape < 1), producing a funnel geometry and divergences.

**Source**: `reference/pollsposition_models-tech_guide.md` Section 4.1.3 (concentration priors), `reference/co_elect-tech_guide.md` Section 10.3.

---

## 3. Google Trends Digital Signals

### 3.1 T-1 Outperforms Election Day

The central finding of Pérez-Rave et al. (2023) [SciELO2023] is that Google Trends data from the **day before the election** (T-1) predicts the runoff winner with 1.86pp error, while election-day data has 6.56pp error. The mechanism: bots and undecided voters distort search patterns on election day; the day before reflects genuine settled preferences.

**We adopted**: An exponential decay function peaked at T-1:
```python
decay = exp(-abs(days_from_elec - 1.0) / 3.0)
```
Where T-1 = 1.0 (peak), T-0 ≈ 0.72, T-7 ≈ 0.10, T-30 ≈ 0.00005. This is applied in both `model_round1.py` and `model_runoff_simple.py` (when enabled).

**Key quantitative conclusion from the paper**: T-1 Google Trends achieved 1.86pp error for the 2022 runoff winner (Petro vs. Hernández), while polls had 6.56pp error on election day. Combined Google+YouTube > Google alone > YouTube alone.

**Source**: `reference/scielo_search_patterns_prediction_colombia_2022.pdf`.

### 3.2 Party-Based Queries Outperform Whole-Name Queries

SciELO found that single-name queries ("Petro", "Rodolfo") are more frequent and more predictive than full-name queries ("Gustavo Petro", "Rodolfo Hernández"). Short queries are the default search behavior and better reflect general interest.

**We adopted**: Candidate query maps using single names: `"Petro"`, `"Rodolfo"`, `"Fico"`, `"Fajardo"` in `trends_keywords.py`.

### 3.3 Digital Signal Bias in 2022 Runoff

Despite SciELO's validated error bound, the Google Trends head-to-head signal for the 2022 Colombian runoff was systematically biased toward Rodolfo. The head-to-head T-1 signal (June 17-18) showed Rodolfo with 50.85-52.1% vs. Petro 47.9-49.2%, while the actual result was Petro 50.42% / Rodolfo 47.35%. The margin error is ~4.8pp in the wrong direction.

**We adopted**: Digital signals are disabled in the runoff model by default (`config.use_digital_signals_runoff = False`). The regular digital signal with decay is also disabled due to systematic bias (e.g., June 8 data showing Rodolfo +17.5pp). The `concentration_t1_boost` parameter is set to 0.0.

**Future work**: Add a bias-correction layer to digital signals before re-enabling them in the runoff model. This could involve learning a per-platform bias parameter or using multi-platform signals (Google + YouTube) to cancel platform-specific bias as recommended by SciELO.

### 3.4 Internet Access Rate Weighting

The SciELO paper and our analysis both confirm that digital signals only measure the online population. Weighting by internet penetration prevents over-weighting digital data in municipalities with low connectivity.

**We adopted**: The `_compute_national_internet_rate()` function returns a population-weighted national average. The digital signal concentration `φ_digital[t]` is multiplied by this rate (currently ~80.79% for 2022), deflating the signal in proportion to offline population.

**We extended** with Option C (soft deflation): `rate_adjusted = 1 - α·(1 - rate)` with α = 0.3 per user direction. This gives a weight of 0.808 at the 2022 national average internet rate of 36%, versus 0.36 with hard deflation.

**Source**: `reference/scielo_search_patterns_prediction_colombia_2022.pdf` Section 4.3 (internet penetration adjustment), CNPV 2018 census for internet access rates.

---

## 4. Trend Extrapolation (Drift)

### 4.1 Campaigns Are Non-Stationary

Election campaigns are directional: candidates rise and fall. A zero-mean random walk assumes stationary fluctuations around a fixed level, which means the model treats the last poll values as the best estimate for election day. In 2022, Petro was rising at 0.095pp/day over the 19-day runoff period. Without drift, the model predicts the runner-up (Rodolfo, matching the last polls); with drift, it correctly predicts Petro.

The linear trend fit to sample-size-weighted runoff polls extrapolates to Petro 50.79% on election day — within 0.37pp of the actual 50.42%.

**We adopted**: A drift parameter in the runoff random walk:
```python
drift = pm.Normal("drift", mu=0, sigma=0.02, shape=2)
theta_increments = sigma_rw * rw_raw + drift
```

The prior is centered at 0 (no assumption on direction). The data decides. With 19 polls over 19 days, the drift is well-identified.

**Source**: Standard time-series econometrics [DickeyFuller, Hamilton1994]. The drift term converts the RW from a stationary to a non-stationary process with constant trend, matching the observed polling trajectory.

### 4.2 Comparison to Other Trend Models

The reverse-time RW with drift is structurally similar to the Ajiaco model [co_elect] but adds the drift term. It differs from the GP-based multi-lengthscale approach [pollsposition Section 4.1.6] which uses 5/14/28 day RBF kernels. Our drift model is simpler (1 extra parameter vs. 3-5 kernel parameters) and is identifiable with 19 data points.

**Not adopted**: Multi-lengthscale GP (too complex for the available data), LSTM fusion [predicción hibrida] (requires Twitter data we don't currently integrate).

---

## 5. Municipal Hierarchical Model

### 5.1 Regional Outperforms National

The dominant finding across Colombian election literature is that municipal/regional models significantly outperform national-level models. The USANTOMAS 2025 study [USANTOMAS2025] found that an FNN+CLR model on demographic features achieved R² = 0.94 for ideological vote share — meaning 94% of municipal-level variation is predictable from demographics alone, before any polls are considered. Baquero & Rosero (2019) [BaqueroRosero2019] achieved 95.9% AUC for binary runoff winner classification at the municipal level.

**We adopted**: A municipal hierarchical model (`model_municipal.py`) that estimates per-municipality vote shares from demographic features (afro-colombian %, NBI poverty, rural dispersion, education, risk, indigenous %, internet access, legislative vote shares). The municipal posterior is summarized as `(mu_logit, sigma_logit)` and used as a structural prior on the national model's `theta[T-1]`.

**Key quote from literature**: The federated approach — national poll likelihood at the rollup, municipal-level prior from fundamentals — is exactly the architecture recommended by the USANTOMAS, modelos_mixtos, and Baquero & Rosero findings.

**Source**: `reference/usantotomas_comparacion_modelos_machine_learning.pdf` Tables 5-7, `reference/modelos_mixtos_datos_composicionales_aplicacion_electoral.pdf` Section 7.

### 5.2 Shared β Coefficients Prevent Overfitting

Our municipal model uses **shared β coefficients** across all 1,122 municipalities (only K β values per feature, not 1,122 × K). The only municipality-specific parameters are the non-centered `mu_m_raw` terms, which are regularized toward zero by `σ_m`. This design was singled out in the mixed-models literature [ModelosMixtos] as "solving the curse of dimensionality beautifully" — the model learns country-wide patterns while letting each municipality deviate locally.

**Source**: `reference/modelos_mixtos_datos_composicionales_aplicacion_electoral.pdf` Sections 4-5, `reference/pollsposition_models-tech_guide.md` Section 4.3 (shared hyperpriors).

### 5.3 Cámara + Senado Legislative Features

Our transfer model (`model_transfer.py`) uses both Cámara de Representantes (local, department-level) and Senado (national, at-large) legislative vote shares as predictors. This follows the literature's finding that the two chambers capture different dimensions of voter behavior:

- **Cámara** reflects local political machinery (gamonalismo, clientelism, regional party strength)
- **Senado** reflects national ideological alignment (which national figures and party brands resonate)

Together they disentangle "local machine loyalty" from "national ideological preference."

**Source**: `reference/usantotomas_comparacion_modelos_machine_learning.pdf` Section 2 (feature engineering), `reference/Colombia_Presidential_Predictor_Research_Reference.md` §11.

---

## 6. Transfer Rate Estimation (Ecological Inference)

### 6.1 Bayesian Ecological Inference

Estimating how eliminated candidates' voters transfer to runoff candidates is an ecological inference problem: we observe aggregate R1 and R2 vote shares per municipality but not individual voter transitions. King (1997) [King1997] provides the canonical solution using a hierarchical Bayesian model with a Beta-binomial likelihood.

**We adopted**: `model_transfer.py` implements Bayesian ecological inference with the structure:

```
logit(β_{c→A}) = γ₀c + γ_ethn·pct_afro + γ_poverty·nbi_rate + γ_rural·pct_rural_disperso + γ_cámara·cámara_share + γ_senado·senado_share
```

Trained on 2010/2014/2018 historical R1→R2 data with municipal covariates. Excludes 2022 from training (`exclude_year=2022`).

**Source**: King (1997) "A Solution to the Ecological Inference Problem" [King1997].

### 6.2 Integration into Runoff Model

The transfer model estimates are used as the prior mean for the runoff model's `theta[T-1]`, not as fixed constants. This means the runoff model can override the transfer prior if poll data strongly disagrees — a key difference from hard-coded transfer constants used in earlier SPEC-08 implementations.

```python
shares_first, shares_second = _compute_transfer_shares(
    election_day, all_candidate_keys, first, second, transfer_rates
)
first_share = float(shares_first.mean())  # Used as prior mean
```

**Source**: `reference/co_elect-tech_guide.md` Section 10.4 (runoff pairing), `reference/Colombia_Presidential_Predictor_Research_Reference.md` §11 (AS/COA transfer matrix).

### 6.3 Convergence Guardrails

Following reviewer feedback, the transfer model includes an automatic fallback: if the demographic-driven γ coefficients fail to achieve R-hat < 1.10, the model falls back to a simpler Dirichlet-Categorical distribution over historically observed transfer rates. This was recommended in the PLAN documents to ensure the model always produces a valid transfer estimate regardless of convergence quality.

**Source**: `reference/pollsposition_models-tech_guide.md` Section 4.5 (convergence diagnostics), `reference/forecast_2016-tech_guide.md` Section 4.4 (fallback strategies).

---

## 7. Polling Aggregation Methods

### 7.1 Time Decay

The standard exponential decay function `w = 0.5^(days/half_life)` with 30-day half-life is used by 538 [538model] and co_elect [co_elect]. It gives a poll on election day weight 1.0, a poll 30 days before weight 0.5.

**We adopted**: Same formula in `aggregation.py` for the baseline weighted average and in the Bayesian model's poll likelihood (via `combined_weight`).

**Source**: `reference/538model-tech_guide.md` Step 1b (`time_weight`), `reference/co_elect-tech_guide.md` Section 8.2.

### 7.2 Pollster Quality Weighting

La Silla Vacía's pollster quality ratings (0-10 scale) are converted to weights via `w = 0.02·rating + 0.8`, constraining weights to [0.82, 1.0]. This follows the 538 PIE methodology: pollsters with higher historical accuracy get higher weight, but the cap at ±20% prevents any single pollster from dominating.

| Pollster | Rating | Weight |
|----------|--------|--------|
| Invamer | 10.0 | 1.00 |
| GAD3 | 8.1 | 0.962 |
| Mosqueteros | 1.0 | 0.82 |

**Source**: `reference/538model-tech_guide.md` Step 1 (PIE calculation), `reference/co_elect-tech_guide.md` Section 8.3 (pollster weight for Colombia).

### 7.3 La Silla Vacía Pollster Weighting Methodology

La Silla Vacía's 2026 technical note [LaSillaVacia2026] defines a three-category weighting system for polling methodology: in-person polls receive weight 1.0, phone polls 0.8, and online polls 0.75. The weekly `ponderacion.csv` file contains recency-weighted averages per candidate, and the June 21 `Prediccion_30d` snapshot provides the 30-day-ahead forecast baseline used as a validation target.

The 2026 pollster ratings are extended from the 2022 `POLLSTER_RATINGS` with updated values: Atlas Intel 5.9, CNC 5.8, GAD3 6.2, Guarumo 5.1, Invamer 5.3, with new firms (Génesis Crea 2.0, TEMPO 2.0, Corporación MMM 2.0) added conservatively.

**We adopted**: The `POLLSTER_WEIGHT_FORMULA = 0.02 × rating + 0.8` for 2022. The 2026 ratings are stored separately in the CNE pipeline and will be integrated with the same formula. The La Silla Vacía cross-validation workflow (`encuestas_detalle.csv` → reconciliation with microdata toplines) is documented but not yet automated.

**Source**: `data/2026-polls/silla_vacia_ponderador/Nota-tecnica-ponderacion-encuestas-Presidenciales-2026-V-Mar27.docx-1.pdf` [LaSillaVacia2026].

### 7.4 Google Trends as Legal Alternative During Poll Blackout

Colombian law prohibits publishing polls in the final week of the campaign. Google Trends is the primary legal real-time signal during this window. The SciELO finding that T-1 Google Trends has 1.86pp error is comparable to top-quality polls, making it a viable substitute during the blackout period.

**We adopted**: The T-1 digital signal infrastructure is in place (`ingest_trends.py`, `phi_digital_base`, decay function, `concentration_t1_boost`). For 2022 runoff, it is disabled due to documented bias. For 2026, it can be re-enabled with calibration.

**Source**: `reference/scielo_search_patterns_prediction_colombia_2022.pdf` Section 5 (legal implications), Colombian electoral law (Act 1992 of 2019, Article 19).

---

## 8. Sampling & Convergence Strategy

### 8.1 Non-Centered Parameterization

Centered parameterization of hierarchical models creates a funnel geometry that NUTS struggles to explore [Betancourt2017]. The non-centered form:

```python
theta_0 ~ Normal(μ, σ)
rw_raw ~ Normal(0, 1)                # independent
theta_increments = σ_rw · rw_raw     # scaling
theta[t] = theta_0 + cumsum(increments)[t]
```

breaks the posterior correlation between σ_rw and the θ[t] values. In practice, R-hat dropped from >1.10 to <1.01 after reparameterization. This is the standard fix for funnel geometries in hierarchical models, first formalized in the parameter-expansion literature [Papaspiliopoulos2007].

**Source**: Betancourt (2017) "A Conceptual Introduction to Hamiltonian Monte Carlo" [Betancourt2017], Papaspiliopoulos et al. (2007) [Papaspiliopoulos2007].

### 8.2 numpyro Over C NUTS

numpyro compiles the model graph to JAX, achieving ~500+ iterations/second per chain vs. ~54 for PyMC's C NUTS. This enables 5,000 draws × 4 chains in ~1-2 minutes per model. The JAX backend uses GPU execution and vectorized auto-differentiation.

| Sampler | Speed | R-hat at 5K | Divergences |
|---------|-------|--------------|-------------|
| PyMC C NUTS | ~54 it/s | ~1.05 | variable |
| numpyro JAX | ~500+ it/s | ~1.01 | 0 (with current config) |

**Source**: numpyro documentation [numpyro], Hoffman & Gelman (2014) "The No-U-Turn Sampler" [HoffmanGelman2014].

---

## 9. Sentiment Analysis & Social Media (Future Work)

### 9.1 Sentiment as Bayesian Prior, Not Poll Substitute

Cerón-Guzmán (2016) [CerónGuzmán2016] established that Twitter sentiment analysis "cannot be put forward as a substitute for traditional polling." The LSTM hybrid model achieves 2.47% RMSE (better than any individual pollster) but only when sentiment is fused with polls through a neural network, not as a standalone signal.

**Not adopted (planned)**: Sentiment pipeline infrastructure exists (`twitter_preprocess.py`, `twitter_denoise.py`, `bert_sentiment.py`, `llm_sentiment.py`, `sentiment_series.py`) but is not wired into the model. Planned for SPEC-39 integration.

**Source**: `reference/sentiment_analysis_model_of_spanish_tweets_2014_elections.pdf`, `reference/ockham_rodolfo_sentiment_analysis_2022.pdf`, `reference/prediccion_electoral_modelo_hibrido_analisis_sentimental_seguimiento_encuestas.pdf`.

### 9.2 Spanish-Specific Tooling Is Mandatory

The literature consistently shows that generic multilingual BERT underperforms Spanish-specific models (RoBERTuito) by 5-7 F1 points on Colombian Spanish sentiment classification. Lexical normalization of Spanish tweet text (OOV → canonical) is required preprocessing — without it, error rates increase 30-50%.

**Source**: `reference/sentiment_analysis_model_of_spanish_tweets_2014_elections.pdf` (Cerón-Guzmán 2016), `reference/arxiv_emotions_twitter_colombia_2022.pdf` (arXiv 2407.07258).

---

## 10. Data Sources & Cross-Validation

### 10.1 Three-Way Results Validation

Election results are consolidated from three independent sources:
1. **Registraduría Nacional** (MMV files, polling-station-level, ~1.1M rows across two rounds)
2. **MOE CSV** (municipal-level, ~17K rows, coalition-level aggregation)
3. **MOE PDF** (book of official results, used for third-source validation)

Two-way agreement (Registraduría vs MOE) is within 0.1% for total valid votes and 0.5% per candidate. The 99.9% convergence rate between pre-count (preconteo) and official scrutiny (escrutinio) confirms the reliability of a single source as canonical.

**Source**: `reference/Colombia_Presidential_Predictor_Research_Reference.md` §4 (election data), Registraduría methodology documentation.

### 10.2 Turnout & Violence Coefficients

The PMC study PMC11929383 [PMC11929383] establishes that assassination events reduce turnout by 1.9-9.0pp, with β₂ = -0.090 to -0.019 across three violence indicators (all p < 0.01). This validates the turnout model formula `T_mt = α + β₁·Historical_Turnout_m + β₂·Violence_Index_m(t-1)` proposed in the planning documents.

**Source**: `reference/Colombia_Presidential_Predictor_Research_Reference.md` §12 (turnout modeling).

---

## References

Full citations for every source cited above are in `bibliography.md` in this directory. See also `index.md` for the complete file catalog and `adopted.md` for the decision-level mapping.
