# Model Design Decisions

## Why Dirichlet-Multinomial?

Vote shares are **compositional data** — they sum to 100% (or 1.0 on the simplex). The natural likelihood for compositional observations is the Dirichlet-Multinomial:

- **Multinomial**: Assumes independent draws. Ignores overdispersion. Unrealistic for polls (polls are not simple random samples).
- **Dirichlet-Multinomial**: Adds a concentration parameter φ that controls overdispersion. As φ → ∞, approaches Multinomial. As φ → 0, the data is maximally overdispersed.

The Dirichlet-Multinomial likelihood:
```
observed_counts ~ DirichletMultinomial(n, α = p · φ)
```
where `n` is the poll sample size, `p` is the latent vote share vector, and `φ` is the concentration (learned from data).

### Why not alternatives?

| Model | Limitation |
|-------|-----------|
| Beta regression | Only handles 2 categories. Need multiple binary models that don't sum to 1. |
| Gaussian on CLR-transformed shares | Assumes homoscedasticity and ignores sample-size uncertainty. |
| Dirichlet distribution | Doesn't model the count-generation process — loses sample-size information in polls. |
| Multinomial | Assumes polls are simple random samples (they're not — they have design effects, weighting, and correlated errors). |

## Why Reverse-Time Random Walk?

The latent vote intention θ[t] evolves backward from election day:

```
θ[0] ← free (election day)
θ[t] ~ Normal(θ[t+1], σ_rw)   // walking backward
θ[T-1] ← informative prior from consultation results + municipal model
```

**Rationale**:
- Uncertainty grows as we move away from election day — natural for forecasting.
- Election day is the anchor (we know the most about it).
- The oldest time point receives an informative prior from real voter behavior (consultation votes + municipal demographics).
- A forward RW (anchored at uncertain past) would be less informative.

## Why Non-Centered Parameterization?

A centered RW:
```
θ[0] ~ Normal(θ_prior, ...)    // already widens as we go back
θ[t] ~ Normal(θ[t-1], σ_rw)    // forward
```

Creates strong posterior correlation between adjacent θ[t] values — NUTS struggles, R-hat is high, ESS is low.

The non-centered reparameterization:
```
theta_0 ~ Normal(μ, σ)              // prior at oldest time point
rw_raw ~ Normal(0, 1)               // standard normal innovations
theta_increments = σ_rw · rw_raw    // scale by RW step size
theta[t] = theta_0[None] + cumsum(theta_increments)[t]  // build forward
theta_free = reverse(theta)         // reverse so index 0 = closest to election
```

This breaks the posterior correlation. Sigma_rw and theta become independent in the prior, which makes NUTS mixing much faster. In practice, R-hat dropped from >1.10 to <1.01 after reparameterization.

## Why K=3 Runoff Model?

The actual runoff has only two candidates, but blank+null votes absorb ~2.3% of valid votes. A K=3 model (candidate A, candidate B, rest+blanco) captures this:

- **Accurate absolute shares**: Without the third category, both candidates' shares would be inflated by ~2.3pp.
- **Rest_blanco prior override**: Historical runoff blank+null rate is ~2.3%. The prior is anchored there with tight sigma (0.3 on logit scale).
- **Preserves the margin**: The margin (A - B) is preserved regardless of the third category's size.

## Why Drift in the Runoff RW?

Campaigns are non-stationary. Candidates rise and fall. A zero-mean RW assumes the process is stationary — it treats the last poll value as the best estimate for election day.

In 2022, Petro was rising at 0.095pp/day in the polls. Without drift, the model predicts Rodolfo wins (matching the last polls: Petro 49.96%). With drift, it extrapolates the trend to election day and correctly predicts Petro (50.79%).

```
theta_increments = σ_rw · rw_raw + drift
```

The drift is estimated from the poll data (19 polls over 19 days). The prior is N(0, 0.02) in logit space (~0.5pp/day in probability), centered at zero — the data decides the direction.

## Why Fixed φ_elec (Not Gamma)?

**History**:
1. Initially: φ_elec = 500,000 (fixed, near-deterministic) — model recited election results perfectly but with no uncertainty.
2. Changed to Gamma(shape=0.75, rate=0.75/50000) — introduced honest uncertainty, but the Gamma's density spike at zero created a funnel geometry → **1106 divergences**.
3. Current: fixed φ_elec = 5,000 via the fixed-path threshold — eliminates the funnel, 0 divergences, ESS > 500.

The fixed concentration path gives φ_elec = 5,000 effective observations of the election result. This provides a moderate constraint (~0.7pp SE) — enough to anchor the posterior without reciting.

For forward prediction (2026), the election likelihood is not added (`results=None`), so φ_elec is irrelevant.

## References

1. [Aitchison1982] — Compositional data analysis. The simplex geometry of vote shares. Foundation for Dirichlet-Multinomial.
2. [Pearson1896] — Spurious negative correlations in compositional data. Why raw share regression is invalid.
3. [Aitchison1986] — Log-ratio transforms (ALR, CLR, ILR). Framework for compositional predictors.
4. [Pollsposition] — Dirichlet-Multinomial for polling data. Single φ_poll with sample-size scaling.
5. [CoElect] — Reverse-time RW for Colombian elections. Ajiaco model architecture.
6. [Betancourt2017] — Non-centered vs centered parameterization. Funnel geometry diagnosis.
7. [SciELO2023] — T-1 Google Trends methodology. Motivation for drift (non-stationary campaign dynamics).
8. [GelmanHill2007] — Hierarchical Bayes, data combination across sources.
9. [King1997] — Ecological inference for transfer rates.
10. [USANTOMAS2025] — Municipal prior rationale. CLR for features.

Full citations: `docs/reference/bibliography.md`. For detailed synthesis: `docs/reference/key-findings.md`.
