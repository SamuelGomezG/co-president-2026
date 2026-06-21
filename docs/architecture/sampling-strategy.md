# Sampling Strategy

## Sampler: numpyro

| Metric | PyMC (C NUTS) | numpyro (JAX NUTS) |
|--------|---------------|-------------------|
| Speed per chain | ~54 it/s | ~500+ it/s |
| R-hat convergence | ~1.05 at 10K | ~1.01 at 5K |
| Memory | ~500 MB | ~2 GB (JAX pre-allocation) |

numpyro is the default sampler (`config.nuts_sampler = "numpyro"`). It compiles the model graph to JAX, enabling GPU execution and vectorized auto-differentiation.

### Configuration

```python
ModelConfig(
    mcmc_draws=5000,     # Posterior draws per chain
    mcmc_tune=5000,      # Tuning/warmup draws per chain
    mcmc_chains=4,       # Independent chains
    mcmc_cores=4,        # Parallel chains
    target_accept=0.95,  # NUTS target acceptance rate
    nuts_sampler="numpyro",
)
```

## Non-Centered Random Walk

The most impactful change for sampling efficiency was reparameterizing the random walk:

**Centered (old)** — creates strong posterior correlation between adjacent θ[t]:
```
θ[t] ~ Normal(θ[t-1], σ_rw)   // for each time point t
```

**Non-centered (current)** — breaks the correlation:
```
theta_0 ~ Normal(μ, σ)           // prior at oldest time point
rw_raw ~ Normal(0, 1)            // independent innovations
theta_increments = σ_rw · rw_raw
theta_cumulative = cumsum(theta_increments, axis=0)
theta_forward = concat([theta_0[None], theta_0[None] + theta_cumulative])
theta_free = reverse(theta_forward)  // index 0 = closest to election
```

The non-centered form makes σ_rw and θ independent in the prior. NUTS can efficiently sample both without the funnel geometry of the centered form.

## φ_elec: Divergence Elimination

The election likelihood concentration parameter went through three iterations:

| Version | Configuration | Divergences | ESS bulk | R-hat |
|---------|--------------|-------------|----------|-------|
| Fixed 500K | `phi_elec = 500000` | 0 | 40 | 1.69 |
| **Gamma(0.75)** | `phi_elec ~ Gamma(0.75, 0.75/50000)` | **1106** ❌ | **303** | **1.01** |
| **Fixed 5K (current)** | `phi_elec = 5000` | **0** ✅ | **595** | **1.000** |

The Gamma(0.75) failed because shape < 1 creates a density spike at zero. The posterior becomes bimodal (either trust the election result or ignore it). NUTS explores the funnel geometry and diverges.

The fix: use the fixed-path branch (`_PHI_ELEC_FIXED_THRESHOLD = 0` ensures any positive value uses the fixed path). With φ_elec = 5,000, the model has 5,000 effective observations of the election result — a moderate constraint that eliminates the funnel while keeping R1 MAE at 0.14pp (not reciting).

## Municipal Model Compatibility

The municipal model (`model_municipal.py`) uses `target_accept=0.9` instead of 0.95. The horseshoe prior on `mu_m_raw` (1,122 municipalities × K candidates) creates a diagonal-dominant posterior mass matrix. `adapt_full` attempts to compute a dense Hessian (O(N³)) which is infeasible for 1,155+ parameters. `adapt_diag` with lower target_accept works around this.

The municipal model is run **outside** the main `pm.Model()` context to prevent parameter leakage (which previously caused 4 OOM crashes from 2.5GB NetCDF files).

## Convergence Diagnostics

All models are verified against:

| Criterion | Target | R1 | Runoff |
|-----------|--------|----|--------|
| R-hat (all params) | < 1.05 | 1.000 | 1.000 |
| ESS bulk (min) | > 100 | 595 | 8050 |
| ESS tail (min) | > 100 | 485 | 8696 |
| Divergences | 0 | **0** | **0** |

## References

1. [HoffmanGelman2014] — The No-U-Turn Sampler (NUTS). Adaptive path length for Hamiltonian Monte Carlo.
2. [Betancourt2017] — Non-centered vs centered parameterization. Funnel geometry diagnosis. Motivation for reparameterizing the RW.
3. [Papaspiliopoulos2007] — Parameter expansion for hierarchical models. General framework for non-centered parameterization.
4. [StanRef] — Convergence diagnostics: R-hat < 1.05, ESS > 100, 0 divergences. Standard reporting.
5. [numpyro] — JAX-based NUTS implementation. GPU-accelerated auto-differentiation.
6. [Gelman2013] — Bayesian Data Analysis (3rd ed.). Effective sample size, Monte Carlo standard error.

Full citations: `docs/reference/bibliography.md`. For detailed synthesis: `docs/reference/key-findings.md`.
