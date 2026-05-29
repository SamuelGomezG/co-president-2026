---
name: pymc-bayesian
description: Bayesian election forecasting with PyMC. Covers model building, sampling, shape handling, reproducibility, and the four-phase MCMC test strategy specific to this project.
---

# PyMC Bayesian Modeling

Skill for building and sampling Bayesian election forecasting models in this project using PyMC ≥6.0.

## When to Use This Skill

- Writing or modifying PyMC models (`model_round1.py`, `model_runoff_simple.py`, `model_runoff_matrix.py`)
- Implementing MCMC sampling or posterior predictive checks
- Adding new likelihood terms or random walk components
- Verifying model correctness or convergence
- Interpreting `arviz.InferenceData` outputs

## Project Architecture

```
src/co_president/
  model_round1.py        — Dirichlet-Multinomial 1st round (K=N candidates)
  model_runoff_simple.py  — K=3 runoff Dirichlet model (A vs B vs rest+blanco)
  model_runoff_matrix.py  — Full probabilistic pairing matrix
```

All models follow the same structure: build graph with `pm.Model()`, sample with `pm.sample()`, return `az.InferenceData`.

---

## Core Patterns

### Model Building

```python
import pymc as pm

def build_model(polls: pd.DataFrame, config: ModelConfig) -> pm.Model:
    with pm.Model() as model:
        # Define priors
        theta = pm.Dirichlet("theta", a=prior_alpha)
        # Observe data
        pm.Multinomial("obs", n=polls["muestra"].values, p=theta, observed=counts)
    return model
```

### Sampling with Reproducibility

**Always pass `random_seed` to `pm.sample()` and `pm.sample_prior_predictive()`:**

```python
with model:
    idata = pm.sample(
        draws=config.mcmc_draws,
        tune=config.mcmc_tune,
        chains=config.mcmc_chains,
        cores=config.mcmc_cores,
        target_accept=config.target_accept,
        random_seed=config.seed,  # REQUIRED for reproducibility
    )
```

```python
# Prior predictive sampling also needs seed
prior_pred = pm.sample_prior_predictive(draws=100, random_seed=config.seed)
```

### Shape Handling (PyMC ≥6.0)

PyMC 6.0+ changed shape semantics. When defining variables with batch dimensions (e.g., time × candidate), use explicit shape notation:

```python
with pm.Model() as model:
    # PyMC 6.0+ prefers explicit shape
    n_polls = len(polls)
    n_candidates = len(candidate_keys)

    theta = pm.Normal("theta", mu=0, sigma=1, shape=(n_polls, n_candidates))
```

### Dirichlet-Multinomial

The project's core observation model:

```python
# Vote shares as Dirichlet prior (one parameter per candidate)
alpha = np.ones(n_candidates) * 0.1
theta = pm.Dirichlet("theta", a=alpha, shape=(n_polls,))

# Multinomial likelihood per poll
for i, (_, poll) in enumerate(polls.iterrows()):
    pm.Multinomial(
        "obs",
        n=int(poll["muestra"]),
        p=theta[i],
        observed=[poll[c] for c in candidate_keys],
    )
```

### K=3 Runoff Pattern

`model_runoff_simple.py` uses K=3 (candidate A, candidate B, rest+blanco). The third category is always the reference:

```python
# K=3: A, B, rest+blanco
alpha = np.ones(3)
theta = pm.Dirichlet("theta", a=alpha, shape=(n_polls, 3))
# Index 0 = candidate A, 1 = candidate B, 2 = rest+blanco
```

---

## Four-Phase Test Strategy

Every new model or significant change must pass four test phases. See `tests/test_model.py` for implementation.

### Phase 1: Graph Test (fast)

Verify the model builds with correct RV and deterministic counts:

```python
def test_model_graph_builds():
    model = build_model(polls, config)
    assert len(model.value_vars) == expected_rvs
    assert sum(1 for var in model.value_vars if var in model.deterministics) == expected_deterministics
```

### Phase 2: Prior Predictive Test (fast)

Samples must fall in valid probability range [0, 1]:

```python
def test_prior_predictive_in_unit_interval():
    prior_pred = pm.sample_prior_predictive(draws=100, random_seed=seed)
    for key in prior_pred.prior:
        assert (prior_pred.prior[key] >= 0).all()
        assert (prior_pred.prior[key] <= 1).all()
```

### Phase 3: Convergence Test (slow, `@pytest.mark.slow`)

R-hat must be < 1.10 on minimal data:

```python
@pytest.mark.slow
def test_convergence_rhat_below_threshold():
    idata = pm.sample(..., random_seed=seed)
    rhat = az.rhat(idata)
    for var in rhat:
        assert rhat[var] < 1.10
```

### Phase 4: Sanity Test (slow)

Posterior mean within ±5pp of known truth on synthetic polls:

```python
@pytest.mark.slow
def test_posterior_mean_within_5pp_of_truth():
    # 3 synthetic polls with known truth
    idata = pm.sample(..., random_seed=seed)
    posterior_mean = idata.posterior["theta"].mean()
    assert abs(posterior_mean - known_truth) < 0.05
```

---

## PyMC ≥6.0 Notes

This project uses PyMC ≥6.0. API may differ from 5.x documentation.

### Verified API Signatures

```python
# pm.sample() — confirmed PyMC 6.x
pm.sample(draws, tune, chains, cores, target_accept, random_seed)

# pm.sample_prior_predictive() — confirmed
pm.sample_prior_predictive(draws, random_seed)

# pm.sample_posterior_predictive() — confirmed
pm.sample_posterior_predictive(idata, random_seed)

# pm.Dirichlet — confirmed
pm.Dirichlet("name", a=alpha, shape=...)

# pm.Normal — confirmed
pm.Normal("name", mu, sigma, shape=...)
```

### Common PyMC 6.x Changes

- Shape handling is stricter; always use explicit `shape=` argument
- `pm.math.stack()` behavior may differ from 5.x
- `dim` argument in distributions replaced by `shape`

### API Docs

When in doubt, verify signatures at: https://docs.pymc.io/

---

## Backtesting and Posterior Analysis

Use `arviz` for posterior analysis:

```python
import arviz as az

# Posterior summary
az.summary(idata, var_names=["theta"])

# R-hat convergence check
rhat = az.rhat(idata)

# Posterior predictive check
az.plot_ppc(idata, num_pp_samples=100)
```

---

## Common Issues

| Issue | Solution |
|-------|----------|
| `SettingWithCopyWarning` | Use `.copy()` when slicing DataFrames |
| Shape mismatch in `pm.Dirichlet` | Ensure `a` (concentration) and `p` (probability) have matching shape |
| `random_seed` not set | Always pass `config.seed` to all `pm.sample*` functions |
| Slow sampling | Reduce `draws`, increase `target_accept`, use `cores` for parallel chains |
| `pm.Mutable` warning | Use immutable dataclasses for model outputs |
