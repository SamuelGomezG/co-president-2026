# PyMC Project Patterns

Project-specific PyMC modeling patterns extracted from `src/co_president/`.

## Random Walk Construction

### Reverse-Time Random Walk

Models use reverse-time RW so that earlier polls are more uncertain:

```python
with model:
    # Random walk precision (inverse variance)
    sigma = pm.HalfNormal("sigma", sigma=0.05)

    # Build RW in reverse time
    delta = pm.Normal("delta", mu=0, sigma=sigma, shape=(n_polls, n_candidates))
    delta_rev = delta[::-1]  # Reverse to forward time

    # Cumulative sum for forward-time trajectory
    cumsum = pm.math.cumsum(delta_rev, axis=0)
    theta = pm.math.softmax(cumsum, axis=1)
```

## House Effects (Pollster Bias)

Hierarchical pollster effects:

```python
with model:
    # Pollster-level house effect
    house = pm.Normal("house", mu=0, sigma=0.1, shape=(n_pollsters,))

    # Apply house effect per poll
    for i, pollster_idx in enumerate(pollster_indices):
        theta_adjusted[i] = softmax(cumsum[i] + house[pollster_idx])
```

## Election Day Likelihood

When official results are available:

```python
if results is not None:
    with model:
        # Election day observation
        pm.Multinomial(
            "election_day",
            n=total_votes,
            p=theta[-1],  # Last time point
            observed=official_counts,
        )
```

## Config Parameters

| Parameter | Default | Purpose |
|----------|---------|---------|
| `mcmc_draws` | 500 | Post-warmup samples per chain |
| `mcmc_tune` | 500 | Warmup iterations |
| `mcmc_chains` | 4 | Number of MCMC chains |
| `mcmc_cores` | 1 | CPU cores for parallel chains |
| `target_accept` | 0.9 | NUTS acceptance rate target |
| `seed` | 42 | Random seed for reproducibility |
