# Model Features — co-president 2026

## 1. Poll-Level Features

Used by R1 model (`model_round1.py`), runoff model (`model_runoff_simple.py`), and municipal model (`model_municipal.py`).

### 1.1 Raw poll share columns (from `encuestas_2022.csv`)

| Column | Source | Models | Notes |
|---|---|---|---|
| `fecha` | `encuestas_2022.csv` | R1, Runoff, Municipal | Parsed via `pd.to_datetime`; `days_before = ELECTION_DATE - fecha` drives time index |
| `encuestadora` | `encuestas_2022.csv` | R1, Runoff, Municipal | Mapped to integer `pollster_idx` for house effects |
| `muestra` | `encuestas_2022.csv` | R1, Runoff, Municipal | Sample size → trial count in `DirichletMultinomial(n=muestra)` |
| `muestra_int_voto` | `encuestas_2022.csv` | Aggregation only | Voting-intent sample; preferred over `muestra` for weighting |
| `gustavo_petro` | `encuestas_2022.csv` | R1, Runoff | Share (%) → normalized → rounded to observed counts |
| `federico_gutierrez` | `encuestas_2022.csv` | R1 | Share (%) → normalized → rounded |
| `rodolfo_hernandez` | `encuestas_2022.csv` | R1, Runoff | Share (%) → normalized → rounded |
| `sergio_fajardo` | `encuestas_2022.csv` | R1 | Share (%) → normalized → rounded |
| `ingrid_betancourt` | `encuestas_2022.csv` | R1 | Share (%) → normalized → rounded |
| `rest` | Derived from `"otros"` | R1, Runoff | Sum of remaining minor candidates; renamed from `"otros"` during cleaning |
| `blanco` | `encuestas_2022.csv` | R1 | Blank vote share; merged into `rest_blanco` for runoff (K=3) |
| `ns_nr` | `encuestas_2022.csv` | Preprocessing only | Undecided/no-response; redistributed proportionally to all shares during `normalize_undecided()` |
| `round_number` | Inferred | R1, Runoff | 1 or 2; inferred via `infer_round_number()` from candidate presence + date threshold |

### 1.2 Derived poll columns (created during model build)

| Column | Model | How Computed |
|---|---|---|
| `days_before` | R1, Runoff, Municipal | `(ELECTION_DATE - fecha).days` → integer time index |
| `time_idx` | R1, Runoff, Municipal | Maps `days_before` to sequential integer `[0..T-1]` |
| `pollster_idx` | R1, Runoff, Municipal | Maps `encuestadora` to integer `[0..P-1]` |
| `rest_blanco` | Runoff only | `sum(other_keys)` where `other_keys` = all R1 candidates except the two runoff finalists |

### 1.3 Runoff-specific columns (K=3 structure)

Created in `build_runoff_simple_model()` from the top-two candidates identified in `results.top_two()`:

| Column | Content |
|---|---|
| `{cand_a_key}` | Winner of R1 (e.g. `gustavo_petro`) |
| `{cand_b_key}` | Runner-up of R1 (e.g. `rodolfo_hernandez`) |
| `rest_blanco` | All eliminated candidates + `blanco` collapsed into one category |

---

## 2. Digital Signal Features

Optional layer from Google Trends (`ingest_trends.py` → `merge_digital_signals()`).

Merged into R1, runoff, and municipal models. Applied using `merge_asof` (backward-fill) from signal dates to poll dates.

| Column | Type | Computation |
|---|---|---|
| `prop_fav` (per candidate) | float [0,1] | `interest(A) / sum(interest(all))` per day via `compute_prop_fav()` |
| Digital signal decay | float | `exp(-abs(days_from_elec - 1.0) / 3.0)` — peaks at T-1 per SciELO 2023 |
| Internet rate for decay | float | Population-weighted national `internet_access_rate` used to deflate digital signal concentration |

---

## 3. Prior Features

### 3.1 Consultation prior (R1 only)

From `config.CONSULTATION_VOTES` and `consultation_log_share_prior()`:

| Feature | Source | Computation |
|---|---|---|
| `consultation_log_share_prior` | `CONSULTATION_VOTES` dict | `log(votes / sum(votes))` per candidate |
| `consultation_prior_strength` | `config.consultation_prior_strength` | Default 0.5; overridable per candidate |

Consultation votes:
| Candidate | Votes |
|---|---|
| `gustavo_petro` | 5,500,000 |
| `federico_gutierrez` | 3,800,000 |
| `sergio_fajardo` | 1,800,000 |
| `rodolfo_hernandez` | 0 |
| `ingrid_betancourt` | 0 |

### 3.2 Runoff prior (from R1 posterior or results)

Used in `build_runoff_simple_model()`:

| Feature | Source |
|---|---|
| `prior_mean_a` | `log(p_A / p_rest_runoff)` from R1 posterior `p_time` or `results.get_share()` |
| `prior_mean_b` | `log(p_B / p_rest_runoff)` from R1 posterior `p_time` or `results.get_share()` |
| `p_rest_runoff` | `1.0 - p_A - p_B` (clamped above epsilon) |

### 3.3 Municipal structural prior (optional, R1)

When `config.fundamentals_mode != "off"`, the municipal model runs in prior-only mode to produce:

| Feature | Source | Shape |
|---|---|---|
| `mu_logit` (CLR-space mean) | `_extract_municipal_prior()` | (K,) |
| `sigma_logit` (CLR-space std) | `_extract_municipal_prior()` | (K,) |

Overrides the consultation prior for `theta[T-1]` in the R1 model.

---

## 4. Municipal Hierarchical Model Features (SPEC-22)

### 4.1 Required columns (validated at `model_municipal.py:198-211`)

| Column | Type | Source | How Used |
|---|---|---|---|
| `codigo_municipio` | str | DIVIPOLA | Index key |
| `pop_2022` | int | DANE PPED | Population weight for turnout-weighted rollup |
| `historical_turnout_m` | float | CEDAE | Mean turnout per municipality across years → effective pop |
| `pct_afro_colombian` | float | CNPV 2018 | Z-scored → `beta_ethnicity` |
| `nbi_rate` | float | NBI 2018 | CLR-transformed → Z-scored → `beta_poverty` |
| `pct_rural_disperso` | float | CNPV 2018 | Z-scored → `beta_rural` |
| `years_schooling_promedio` | float | CNPV 2018 | Z-scored → `beta_education` |
| `high_risk_flag` | bool | MOE Risk | Binary → `beta_risk` |
| `is_pdet` | bool | MOE Risk | Binary → `beta_pdet` |
| `pct_indigenous` | float | CNPV 2018 | Z-scored → `beta_indigenous` |
| `internet_access_rate` | float | CNPV 2018 | Z-scored → `beta_internet` |
| `historical` | tuple[HistoricalRecord] | CEDAE + MOE | `clr_left_share` extracted → Z-scored → `beta_historical` |

### 4.2 Intermediate derived variables (model_municipal.py)

| Variable | Computation |
|---|---|
| `effective_pop` | `pop * turnout / mean(turnout)` via `compute_effective_pop()` |
| `pop_weights` | `effective_pop / effective_pop.sum()` |
| `clr_left` | `_extract_clr_left_share()` — CLR of historical left-share from target year R1 |
| `clr_nbi` | `_clr_nbi_array(nbi_rate)` — CLR of `(nbi_rate, 1 - nbi_rate)` |
| `*_z` (e.g. `pct_afro_z`) | `(x - mean(x)) / (std(x))` — zero-mean unit-variance standardization |

### 4.3 Schema-only columns (available in `MunicipalFeatures` but not used in regression)

| Column | Source |
|---|---|
| `poblacion_total` | CNPV 2018 |
| `poblacion_afrocolombiana` | CNPV 2018 |
| `poblacion_indigena` | CNPV 2018 |
| `poblacion_rural_dispersa` | CNPV 2018 |
| `pct_school_attendance` | CNPV 2018 |
| `labor_force_participation_rate` | CNPV 2018 |
| `pct_female` | CNPV 2018 |
| `rooms_per_household` | CNPV 2018 |
| `persons_per_household` | CNPV 2018 |
| `pct_age_18_29` | CNPV 2018 |
| `pct_age_30_54` | CNPV 2018 |
| `pct_age_55_plus` | CNPV 2018 |
| `nbi_urban` | NBI 2018 |
| `nbi_rural` | NBI 2018 |
| `ipm_2018` | IPM 2018 |
| `ipm_2018_imputed` | IPM pipeline |
| `ipm_2022` | IPM 2022 |
| `ipm_2022_imputed` | IPM pipeline |
| `pop_2018`–`pop_2026` | DANE PPED |
| `pct_ingresos_propios` | TerriData fiscal |
| `gastos_totales_per_capita` | TerriData fiscal |
| `transferencias_per_capita` | TerriData fiscal |
| `ingresos_tributarios_per_capita` | TerriData fiscal |
| `risk_level` | MOE Risk |
| `armed_group_presence` | MOE Risk |
| `coca_hectares` | UNODC |
| `comuna_nombre` | DIVIPOLA (Bogotá localidades) |
| `departamento` | DIVIPOLA |
| `region` | DIVIPOLA |

### 4.4 Linear predictor (municipal logit model)

```
logit(p_mk) = alpha_k
    + beta_historical[k] * clr_left_z[m]
    + beta_ethnicity[k]  * pct_afro_z[m]
    + beta_poverty[k]    * clr_nbi_z[m]
    + beta_rural[k]      * pct_rural_z[m]
    + beta_education[k]  * years_schooling_z[m]
    + beta_risk[k]       * high_risk[m]
    + beta_pdet[k]       * is_pdet[m]
    + beta_indigenous[k] * pct_indigenous_z[m]
    + beta_internet[k]   * internet_access_z[m]
    + sigma_m[k]         * mu_m_raw[m, k]
```

---

## 5. Transfer Model Features (SPEC-30)

### 5.1 Target variables

| Variable | Source | Description |
|---|---|---|
| `left_r1` | `historical_results.csv` | Left candidate R1 vote share per municipality |
| `left_r2` | `historical_results.csv` | Left candidate R2 vote share per municipality |
| `elim_r1` | `historical_results.csv` | Eliminated candidate R1 vote share per municipality |

### 5.2 Covariates (feature cols defined at `model_transfer.py:58-64`)

| Feature | Source | Transformation |
|---|---|---|
| `pct_afro_colombian` | CNPV 2018 | Z-scored per observation group |
| `nbi_rate` | NBI 2018 | Z-scored per observation group |
| `pct_rural_disperso` | CNPV 2018 | Z-scored per observation group |
| `camara_left_share` | Historical legislative | Z-scored per observation group |
| `senado_left_share` | Historical legislative | Z-scored per observation group |

### 5.3 Transfer constants (SPEC-08 fallback, `config.py`)

| Constant | Value | Meaning |
|---|---|---|
| `TRANSFER_FAJARDO_PETRO` | 0.40 | Fraction of Fajardo votes → Petro |
| `TRANSFER_FAJARDO_HERNANDEZ` | 0.60 | Fraction of Fajardo votes → Hernández |
| `TRANSFER_GUTIERREZ_HERNANDEZ` | 0.87 | Fraction of Gutiérrez votes → Hernández |
| `TRANSFER_GUTIERREZ_PETRO` | 0.13 | Fraction of Gutiérrez votes → Petro |
| `TRANSFER_BLANCO_SPLIT` | 0.50 | Fraction of blanco votes → candidate A |

---

## 6. Aggregation Baseline Features (SPEC-05)

Used by `aggregation.py` for weighted-average baseline (non-Bayesian).

| Feature | Formula |
|---|---|
| `time_weight` | `0.5^(days_from_election / half_life)` |
| `sample_size_weight` | `log(max(muestra_int_voto, 0) + 1)` |
| `pollster_weight_map` | `rating × 0.02 + 0.8` from `POLLSTER_RATINGS` |
| `combined_weight` | `w_time × w_sample × w_pollster`, normalized to sum=1 |
| `weighted_average` | `sum(w_i × share_i) / sum(w_i)` per candidate |

---

## 7. Data Sources Summary

| Source | Files | Ingestion Module | Provides |
|---|---|---|---|
| Encuestas 2022 | `encuestas_2022.csv` | `data_polls.py` | 46 national polls, 27 columns |
| Consultas | `consultas.csv` | `data_polls.py` | 65 coalition internal polls |
| Google Trends | (live fetch) | `ingest_trends.py` | Per-candidate prop_fav |
| AS/COA | `as_coa/round1.csv`, `runoff.csv` | `data_polls.py` | 3rd-party aggregate tracker |
| DIVIPOLA | `divipola_master.csv` | `ingest_divipola.py` | Municipality codes, names, dept, region |
| CNPV 2018 | `cnpv_2018.csv` | `ingest_cnpv.py` | Demographics (age, ethnicity, education, internet, housing) |
| NBI 2018 | `nbi_2018.csv` | `ingest_nbi.py` | Poverty rates (total, urban, rural) |
| IPM | `ipm_2018.csv` | `ingest_ipm.py` | Multidimensional Poverty Index |
| DANE Population | `population_2018_2026.csv` | `ingest_population.py` | Population projections 2018-2026 |
| Historical (CEDAE) | `historical_results.csv` | `ingest_historical.py` | Per-municipality election results 2002-2022 |
| MOE Risk | `risk_factors.csv` | `ingest_risk.py` | Electoral risk, PDET, armed groups, coca |
| Fiscal (TerriData) | `fiscal.csv` | `ingest_fiscal.py` | Fiscal autonomy indicators |
| Registraduría MMV | `MMV_NACIONAL_PRESIDENTE_2022_1v.csv`, `_2v.csv` | `data_results.py` | Official R1/R2 vote counts |
| MOE results | `moe_vuelta1.csv`, `moe_vuelta2.csv` | `data_results.py` | Validation source for election results |
| Participation | `reg_participacion_vuelta1.csv`, `_2.csv` | `data_results.py` | Registered voters, polling stations |

---

## 8. Model-Generated Variables (Output Features)

### 8.1 R1 model (`model_round1.py`)

| PyMC Variable | Shape | Description |
|---|---|---|
| `sigma_rw` | scalar | Random walk step size (HalfNormal) |
| `sigma_house` | scalar | House effect spread (HalfNormal) |
| `phi_poll` | scalar | Poll Dirichlet concentration (Gamma) |
| `phi_poll_n` | (P,) | Sample-size-scaled concentration |
| `p_time` | (T, K) | Softmax(theta) — latent vote shares per time point |
| `p_adj` | (P, K) | Poll-adjusted probabilities (theta + house effects, softmax) |
| `raw_house` | (P, K) | Raw per-pollster-candidate effects (Normal) |
| `house_effects` | (P, K) | Zero-sum constrained house effects |
| `p_elec` | (K,) | Election-day vote share probabilities |
| `phi_digital_base` | scalar | Digital signal concentration (Gamma) |
| `theta_r_*` | (K-1,) | Reverse-time random walk states at each time point |

### 8.2 Runoff model (`model_runoff_simple.py`)

| PyMC Variable | Shape | Description |
|---|---|---|
| `sigma_rw` | scalar | Random walk step size |
| `sigma_house` | scalar | House effect spread |
| `phi_poll` | scalar | Poll Dirichlet concentration |
| `phi_poll_n` | (P, 1) | Sample-size-scaled concentration |
| `theta_r_*` | (2,) | Reverse-time RW states (2 free dimensions, rest as reference) |
| `p_time` | (T, 3) | K=3 latent shares: cand_A, cand_B, rest_blanco |
| `p_adj` | (P, 3) | Poll-adjusted probabilities |
| `raw_house` | (P, 3) | Raw per-pollster-candidate effects |
| `house_effects` | (P, 3) | Zero-sum constrained |
| `p_elec` | (3,) | Election-day shares |
| `phi_digital_base` | scalar | Digital signal concentration |

### 8.3 Municipal model (`model_municipal.py`)

| PyMC Variable | Shape | Description |
|---|---|---|
| `alpha` | (K,) | Intercept per candidate |
| `beta_historical` | (K,) | Historical left-vote CLR coefficient |
| `beta_ethnicity` | (K,) | Afro-Colombian % coefficient |
| `beta_poverty` | (K,) | NBI poverty CLR coefficient |
| `beta_rural` | (K,) | Rural dispersion % coefficient |
| `beta_education` | (K,) | Education level coefficient |
| `beta_risk` | (K,) | High risk flag coefficient |
| `beta_pdet` | (K,) | PDET flag coefficient |
| `beta_indigenous` | (K,) | Indigenous % coefficient |
| `beta_internet` | (K,) | Internet access rate coefficient |
| `sigma_m` | (K,) | Municipal random-effect scale |
| `mu_m_raw` | (M, K) | Non-centered municipal random effects |
| `p_municipal` | (M, K) | Per-municipality vote probabilities |
| `p_natl` | (K,) | Turnout-weighted national rollup |
| `p_municipal_clr` (optional) | (M, K) | CLR of p_municipal when `config.clr_target=True` |
