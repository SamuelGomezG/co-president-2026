# Hierarchical Model Architecture — Implementation Plan

**Author**: AI agent + user direction
**Date**: 2026-06-19
**Status**: Draft for user review before execution

---

## 0. Motivation

### 0.1 Current architecture is backwards

| Layer | Current role | Should be |
|-------|-------------|-----------|
| Polls | Primary observation (DirichletMultinomial) | Secondary — noisy, biased, banned final week |
| Digital signals | Additive logit shift (`beta_digital * ds`) | **Primary observation near election** |
| Municipal features | Completely decoupled (`model_municipal.py`) | **Structural prior** for national model |
| Election results | Fixed φ=500,000 (near-deterministic) | Learned φ ~ Gamma (honest uncertainty) |
| Internet access | Not used in poll/runoff models | Weight on digital signal observations |

### 0.2 Evidence from literature

- **SciELO 2023**: Google Trends T-1 had 1.86pp error for runoff winner; polls had 6.56pp on election day. Combined Google+YouTube > Google alone > YouTube alone.
- **Rodolfo phenomenon**: Polls systematically missed Rodolfo Hernández's surge; digital signals captured it.
- **USANTOMAS**: CLR-transformed municipal features achieved R²=0.94 for ideological vote share (FNN) — before any polls.
- **User insider knowledge**: Same pattern confirmed for 2026 Round 1.

### 0.3 Target architecture (layered, not additive)

```
                     ┌──────────────────────────────────┐
                     │     Posterior: θ[t] (K-dim)       │
                     │     National vote share at time t  │
                     └──────────────────────────────────┘
                                   ▲
            ┌──────────────────────┼──────────────────────┐
            │                      │                      │
   ┌────────┴────────┐   ┌────────┴────────┐   ┌────────┴────────┐
   │ Municipal prior │   │  Poll likelihood │   │ Digital signal   │
   │ (structural)    │   │  (secondary)     │   │ likelihood (PRIMARY│
   │                 │   │                  │   │  near election)   │
   │ µ_muni ~ model_ │   │ poll[t] ~ DirMult│   │ ds[t,k] ~ Beta   │
   │ municipal post. │   │ (p_adj[t], φ_poll)│   │ (p[t,k], φ_ds[t])│
   │                 │   │                  │   │                  │
   │ Always active   │   │ Always active    │   │ Active when      │
   │ (one point: T=0)│   │ (at poll dates)  │   │ ds data exists   │
   └─────────────────┘   └─────────────────┘   └──────────────────┘
            │                      │                      │
            └──────────────────────┼──────────────────────┘
                                   │
                     ┌─────────────┴─────────────┐
                     │  Election anchor (learned) │
                     │  elec ~ DirMult(p[T], φ_elec)  │
                     │  φ_elec ~ Gamma(shape=10, …)   │
                     └───────────────────────────┘

   φ_digital[t] = base_phi × exp(−(election_date − t) / decay_days)
   φ_digital weighted by internet_access_rate (municipal → national avg)
```

### 0.4 Quality targets (non-negotiable)

| Metric | Target | Verification |
|--------|--------|-------------|
| R-hat (all params) | < 1.01 | `az.rhat()` max |
| ESS bulk (all params) | > 1,000 | `az.ess(idata, method="bulk")` |
| ESS tail (all params) | > 1,000 | `az.ess(idata, method="tail")` |
| Round 1 MAE (2022) | < 2.5pp | vs canonical results |
| Runoff MAE (2022) | < 1.5pp | vs canonical results |
| Divergences | 0 | `idata.sample_stats.diverging` |
| Max tree depth warnings | 0 | Parser output |

---

## 1. Phase 0 — Prerequisites (10 min)

### Objective
Fix infrastructure blockers before touching any model code.

### 0.1 Add `netCDF4` dependency

**Problem**: Transfer model training succeeds but posterior can't be saved (`cannot write NetCDF files`). Fallback to Dirichlet(27/73) prior works but loses data-driven rates.

**Fix**:

```toml
# pyproject.toml — add to dependencies
"netCDF4>=1.7.0",
```

```bash
uv sync
```

### 0.2 Verify current 100K run (regression baseline)

Run `CONFIG_100K` as-is and capture:
- Round 1 MAE
- Runoff MAE
- Per-candidate R-hat
- Sampling time

This is the **regression baseline** — every phase must not degrade MAE below this.

### 0.3 Ensure `make check` passes on `dev`

```bash
git checkout dev && git pull
make check  # must exit 0
```

### Deliverables
- `netCDF4` in `pyproject.toml`
- Baseline 100K report saved as `results/baseline_100k_report.md`
- All quality gates green

---

## 2. Phase 1 — Municipal Prior Integration (prototype at 4K)

### Objective
Wire the municipal fundamentals model as a **structural prior** on the national round-1 model's initial latent state. No more flat Dirichlet(1.0) initialization.

### 2.1 Extract municipal posterior

Add `_extract_municipal_prior()` to `model_round1.py`:

```python
def _extract_municipal_prior(
    features: pd.DataFrame,
    config: ModelConfig,
    candidate_keys: list[str],
    target_year: int = 2022,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract municipal model posterior mean and std as structural prior.

    Trains the municipal hierarchical model (SPEC-22) on the full feature
    matrix, then computes the turnout-weighted national vote share posterior
    for each candidate. Returns the logit-space mean and logit-space standard
    deviation for use as a prior on theta_raw[0] in the national model.

    Args:
        features: Municipal feature matrix (M rows × F columns).
        config: Model hyperparameters (used for draw count).
        candidate_keys: Active candidate column names.
        target_year: Historical election year for CLR left share.

    Returns:
        (mu_logit, sigma_logit): Arrays shape (K,) where K = len(candidate_keys).
        mu_logit[k] is the logit-scale posterior mean for candidate k.
        sigma_logit[k] is the logit-scale posterior std for candidate k.

    Raises:
        ValueError: If features is empty or missing required columns.
    """
```

**Implementation**:
1. Load `features` via `load_features()`
2. Build empty `polls` DataFrame (municipal model needs it for schema, but we only want the **prior**, not the poll likelihood)
3. Call `build_municipal_model(features, polls, results=None, config)`
4. Sample with `mcmc_draws=config.mcmc_draws // 4` (quick, 1K draws at prototype)
5. Extract `p_time` deterministic → compute pop-weighted national mean
6. Apply CLR transform → return `(mu_logit, sigma_logit)` for use as Normal prior

**Spec for empty polls workaround**: `model_municipal.py` currently raises `ValueError` if polls is empty. Add a `prior_only` boolean parameter that skips the poll likelihood:

```python
def build_municipal_model(
    features, polls, results, config, target_year=2022, digital_signals=None,
    prior_only: bool = False,  # NEW
) -> pm.Model:
```

When `prior_only=True`: skip poll processing, skip DirichletMultinomial observation.

**Files changed**:
- `src/co_president/model_municipal.py`: Add `prior_only` parameter
- `src/co_president/model_round1.py`: Add `_extract_municipal_prior()`

### 2.2 Wire municipal prior into model_round1

**Current** (approximate, from reverse-time RW construction):
```python
# First time point has flat prior
theta_free = pm.Normal("theta_r_0", mu=0.0, sigma=0.5, shape=K)
```

**New**:
```python
mu_muni, sigma_muni = _extract_municipal_prior(features, config, candidate_keys)

# Municipal prior on first time point
theta_free = pm.Normal(
    "theta_r_0",
    mu=mu_muni,           # ← informed by municipal fundamentals
    sigma=sigma_muni,     # ← posterior uncertainty from municipal model
    shape=K,
)
```

**Files changed**:
- `src/co_president/model_round1.py`: Accept `features` parameter + `mu_muni`, `sigma_muni`

### 2.3 Add `init="adapt_full"` everywhere

Replace all `pm.sample(init="jitter+adapt_diag", ...)` with `pm.sample(init="adapt_full", ...)` in:
- `model_round1.py:338` (`sample_round1`)
- `model_runoff_simple.py:406` (`sample_runoff`)
- `model_municipal.py` (`sample_municipal_model`)

This uses variational inference (ADVI) to find better starting values, reducing warmup time and improving convergence at low draw counts.

### 2.4 Add ESS reporting

In `scripts/run_100k_report.py`, after each sampling call:

```python
ess_bulk = az.ess(idata, method="bulk")
ess_tail = az.ess(idata, method="tail")
min_ess_bulk = min(float(v.min()) for v in ess_bulk.values())
min_ess_tail = min(float(v.min()) for v in ess_tail.values())
logger.info(f"Round 1 ESS: bulk={min_ess_bulk:.0f}, tail={min_ess_tail:.0f}")
```

### 2.5 Test (2K draws, fast)

```bash
uv run python scripts/run_100k_report.py  # uses CONFIG_PROTOTYPE
```

**Validation**: Run at 2K draws. Check:
1. Municipal prior extraction succeeds (no shape errors)
2. `mu_muni` has correct shape (K,)
3. `sigma_muni` is non-zero, reasonable magnitude (~0.3–1.5 in logit space)
4. Model samples without shape errors
5. R-hat ≤ 1.30 at 2K (better than current 2K at phi=500K which gets ~3.0+)
6. `make check` passes (tests may need updating for new `features` parameter)

**Pass gate: Municipal prior active, no crashes. R-hat improved over baseline at same draw count.**

---

## 3. Phase 2 — Learned φ_elec (Gamma Prior) (prototype at 4K)

### Objective
Replace fixed `phi_elec = 500,000` with a learned Gamma parameter. The model should express honest uncertainty rather than reciting election results.

### 3.1 Replace fixed phi_elec with Gamma

**Current** (`model_round1.py`, approximate):
```python
phi_elec = config.concentration_election_prior_mean  # fixed 500,000
alpha_elec = election_theta * phi_elec
pm.DirichletMultinomial("election_likelihood", n=total_votes, a=alpha_elec, observed=results)
```

**New**:
```python
# Learned election precision
if config.concentration_election_prior_mean > _PHI_ELEC_FIXED_THRESHOLD:
    phi_elec = config.concentration_election_prior_mean  # backward compat
else:
    phi_elec = pm.Gamma(
        "phi_elec",
        alpha=config.concentration_election_prior_shape,   # default 10
        beta=config.concentration_election_prior_shape / config.concentration_election_prior_mean,
        # E[phi_elec] = shape / (shape/mean) = mean = 50,000
    )

# Concentrate election likelihood proportionally
total_votes_scaled = int(config.concentration_election_votes_scale)
alpha_elec = election_theta * phi_elec
pm.DirichletMultinomial(
    "election_likelihood",
    n=total_votes_scaled,  # scaled down to ~1M instead of 21M
    a=alpha_elec,
    observed=results_scaled,
)
```

**Config additions**:
```python
@dataclass
class ModelConfig:
    # ... existing fields ...
    concentration_election_prior_shape: float = 10.0  # Gamma shape
    concentration_election_votes_scale: int = 1_000_000  # scale total votes
```

The votes must be scaled because `phi_elec * p` at 50K vs 500K changes the scale. If we keep `n = 21M`, then `alpha * n` at phi=50K would be too small and the DirichletMultinomial becomes overly diffuse. Scaling `n` to ~1M with phi ~50K gives alpha ~ 50K * p * 1M ≈ strong but not dominating.

Actually, the correct relationship is: `alpha = p * phi` where phi is concentration. The DirichletMultinomial's effective sample size is roughly `phi / (1 + phi) * n` (approximate). So:

- phi=500K, n=21M: effective_n ≈ 500K/(1+500K) * 21M ≈ 21M (effectively full weight)
- phi=50K, n=1M: effective_n ≈ 50K/(1+50K) * 1M ≈ 1M (strong but learnable)

The key is: `n` controls the maximum evidence weight, `phi * p` controls the concentration around `p`. With phi=50K and n=1M, the model can observe election results as strong-but-not-deterministic evidence.

### 3.2 Apply same to runoff model

`model_runoff_simple.py` also has election likelihood. Apply the same Gamma phi_elec pattern.

### 3.3 Test (4K draws)

```bash
uv run python scripts/run_100k_report.py
```

**Validation**:
1. `phi_elec` posterior mean ~ 40K–100K (not collapsed to prior bounds)
2. R-hat ≤ 1.10 at 4K draws (much better than 1.69 at 10K with phi=500K)
3. MAE R1: slightly higher than baseline (2-5pp vs 0.54pp) — this is CORRECT (honest uncertainty)
4. ESS bulk > 200 at 4K (feasible with 4K draws)
5. No divergences
6. `make check` passes

**Pass gate: phi_elec learned, R-hat < 1.10 at 4K, MAE honest (not memorized).**

---

## 4. Phase 3 — Digital Signal Beta Observation Layer

### Objective
Add a **separate Beta observation likelihood** for digital signals (Google Trends prop_fav). This likelihood **dominates** the poll likelihood as election day approaches via exponential decay on precision φ_digital[t].

### 4.1 Why a separate likelihood, not additive logit shift

**Current** (`model_runoff_simple.py:333-338`):
```python
# Digital signal as additive perturbation to polls
theta_adj = theta_selected + house_selected + beta_digital * ds_values
# Same DirichletMultinomial as polls
```

**Problem**: Digital signals share the poll likelihood's concentration φ_poll. When polls are noisy (high variance), digital signals get diluted. When polls are precise, digital signals can only perturb — never dominate.

**Solution**: Separate Beta likelihood with its own precision φ_digital[t] that **grows exponentially** as election approaches.

### 4.2 Mathematical specification

For candidate k at time t where digital signal data exists:

```
ds_obs[t, k] ~ Beta(α = p[t, k] * φ_digital[t], β = (1 - p[t, k]) * φ_digital[t])
```

Where:
- `p[t, k]` = national vote share for candidate k at time t (from latent theta)
- `φ_digital[t]` = time-varying digital signal precision
- `ds_obs[t, k]` = Google Trends favorable propensity for candidate k at time t

**φ_digital[t] decay function**:

```
φ_digital[t] = base_phi * exp(-days_before[t] / decay_days)
```

Where:
- `base_phi` ~ Gamma(shape=5, beta=5/100) = E[base_phi] = 100
- `decay_days` = 3.0 (from SciELO 2023: T-1 has 1.86pp error, T-7 has ~4pp)
- At T-1: φ_digital ≈ base_phi * exp(-1/3) ≈ 100 * 0.717 = 71.7
- At T-30: φ_digital ≈ base_phi * exp(-30/3) ≈ 100 * 0.000045 = 0.0045
- Net effect: digital signals are ~16,000× more influential at T-1 than T-30

### 4.3 Internet access rate weighting

Digital signals only measure the **online population**. To correct for selection bias, weight φ_digital[t] by the national average internet access rate:

```python
# From feature matrix
internet_access = features["internet_access_rate"].to_numpy()  # (M,)
pop = features["pop_2022"].to_numpy()  # (M,)
pop_weights = pop / pop.sum()
national_internet_rate = float(np.average(internet_access, weights=pop_weights))

# Effective precision modulated by internet coverage
phi_digital_effective[t] = phi_digital[t] * national_internet_rate
```

Rationale: If 70% of population has internet access, digital signals can only inform about 70% of the electorate. The remaining 30% introduces irreducible uncertainty.

**Where this applies**:
- `model_round1.py`: NEW digital signal Beta likelihood (currently has NONE)
- `model_runoff_simple.py`: REPLACE additive `beta_digital` logit shift WITH separate Beta likelihood

### 4.4 Implementation: Round 1 model

Add to `model_round1.py`:

```python
if digital_signals is not None and not digital_signals.empty:
    # Merge digital signals onto poll time grid
    ds_merged = _merge_digital_signals(polls, digital_signals, candidate_keys)

    # Time-varying precision with exponential decay toward election
    days_before = ds_merged["days_before"].to_numpy()
    phi_digital_base = pm.Gamma("phi_digital_base", alpha=5, beta=5/100)
    phi_digital = pm.Deterministic(
        "phi_digital",
        phi_digital_base * pm.math.exp(-days_before / 3.0) * national_internet_rate
    )

    # Per-candidate Beta observation
    for k, candidate in enumerate(candidate_keys):
        ds_values = ds_merged[candidate].to_numpy()
        pm.Beta(
            f"ds_likelihood_{candidate}",
            alpha=p_time[:, k] * phi_digital,
            beta=(1 - p_time[:, k]) * phi_digital,
            observed=ds_values,
        )
```

### 4.5 Implementation: Runoff model

Replace `model_runoff_simple.py:333-338` (additive logit shift) with the same Beta observation layer pattern, using K=3 (A, B, rest+blanco).

### 4.6 Files changed

- `src/co_president/model_round1.py`: Add Beta observation layer, internet rate weighting
- `src/co_president/model_runoff_simple.py`: Replace additive shift with Beta observation layer
- `src/co_president/data_polls.py` (if needed): `_merge_digital_signals()` helper
- `scripts/run_100k_report.py`: Pass `features` and `digital_signals` to round1 model

### 4.7 Test (4K draws, 2022 backtest)

**Validation**:
1. Digital signal Beta likelihood adds to model graph (check RV count)
2. `phi_digital_base` posterior mean ~60-150 (reasonable concentration)
3. Digital signals dominate near election: weight at T-1 ≈ 70 vs weight at T-30 ≈ 0.005
4. Runoff MAE improves over Phase 2 (should be better at capturing late surge)
5. No shape errors in Beta likelihood
6. R-hat ≤ 1.10 at 4K
7. `make check` passes

**Pass gate: Digital Beta likelihood active, no degradation vs Phase 2, T-1 digital precision > T-7 precision.**

---

## 5. Phase 4 — 2026 Parallel Backtest

### Objective
Extend the pipeline to also run a 2026 backtest (user has known 2026 R1 results). Compare model accuracy across both elections.

### 5.1 Add year parameter to data loading

```python
@dataclass
class ModelConfig:
    # ... existing fields ...
    target_year: int = 2022  # Election year to backtest
```

All data-loading functions must accept `year` and load the correct datasets:

| Component | 2022 | 2026 |
|-----------|------|------|
| Polls | `data/polls/` (existing) | User-provided path |
| Election results | `2022-presidential-results/` | User-provided path |
| Google Trends | `data/google-trends/` (existing) | User-provided path |
| Municipal features | Same (time-invariant) | Same |
| Historical results | Same (2002-2022) | Same |

### 5.2 Create `scripts/run_2026_backtest.py`

Mirrors `run_100k_report.py` but:
- Uses `CONFIG_PROTOTYPE` (4K draws)
- Sets `target_year=2026`
- Loads 2026-specific data paths
- Runs both 2022 (regression) and 2026 (new) backtests
- Outputs comparative report

### 5.3 Data pipeline changes

Minimal changes — data loading already parameterized. Need:
1. User to provide 2026 poll data and election results
2. Google Trends fetch for 2026 date range (existing `fetch_trends()` should work with new dates)
3. Transfer model runs identically (historical data unchanged)

### 5.4 Test

```bash
uv run python scripts/run_2026_backtest.py
```

**Validation**:
1. 2022 backtest produces same results as Phase 3 (regression check)
2. 2026 backtest runs without errors
3. User can validate 2026 accuracy against known results

**Pass gate: Both backtests run cleanly. 2022 regression matches Phase 3 within tolerance.**

---

## 6. Phase 5 — Full 100K Production Run

### Objective
Run the fully integrated hierarchical model at production quality (100K draws) and verify all quality targets.

### 6.1 Configuration

```python
CONFIG_PRODUCTION = ModelConfig(
    mcmc_draws=100_000,
    mcmc_tune=10_000,
    mcmc_chains=4,
    mcmc_cores=4,
    target_accept=0.99,
    init="adapt_full",
    max_treedepth=12,
    seed=332211,
    concentration_election_prior_shape=10.0,
    concentration_election_prior_mean=50_000,
    concentration_election_votes_scale=1_000_000,
    target_year=2022,
)
```

### 6.2 What to expect at 100K

| Phase | R1 R-hat | R2 R-hat | R1 MAE | R2 MAE | Time (est.) |
|-------|----------|----------|--------|--------|-------------|
| Current (phi=500K) | 1.000 | N/A | 0.54pp | 1.12pp | ~5h |
| Phase 1 (muni prior) | < 1.01 | N/A | ~1.5–3pp | ~1.2pp | ~5h |
| Phase 2 (+ Gamma phi) | < 1.01 | < 1.01 | ~1.5–3pp | ~1.0pp | ~4h |
| Phase 3 (+ Beta ds) | < 1.01 | < 1.01 | ~1.0–2pp | **< 1.0pp** | ~4.5h |
| Phase 5 (100K prod) | < 1.01 | < 1.01 | **< 2.5pp** | **< 1.5pp** | ~5h |

Key: With Gamma phi_elec, model is less stiff → faster sampling per draw → 100K takes ~3h instead of ~5h. Digital Beta adds overhead but SS remains good.

### 6.3 Quality gate checklist

```python
# After sampling
rhat = az.rhat(idata)
assert all(float(rhat[v].max()) < 1.01 for v in rhat.data_vars)

ess_bulk = az.ess(idata, method="bulk")
assert all(float(ess_bulk[v].min()) > 1000 for v in ess_bulk.data_vars)

ess_tail = az.ess(idata, method="tail")
assert all(float(ess_tail[v].min()) > 1000 for v in ess_tail.data_vars)

assert idata.sample_stats.diverging.sum() == 0
```

### 6.4 Report output

The final report must include:
1. Per-candidate posterior mean, 95% HDI
2. R-hat and ESS for all parameters
3. Comparison to canonical 2022 results (MAE, RMSE)
4. Predicted margin and P(winner)
5. phi_elec posterior (how much does model trust election results?)
6. phi_digital_base posterior (how much does model trust digital signals?)
7. Internet_access_rate effective weight
8. Pairing matrix probabilities

### 6.5 Run command

```bash
uv run python scripts/run_100k_report.py
# Output: results/100k_accuracy_report.md
# Output: results/100k_accuracy_report.json
```

---

## 7. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Municipal model too slow at 100K | Medium | Schedule delay | Municipal prior extracted ONCE (1K draws), cached to disk |
| Gamma phi_elec collapses to prior | Low | No anchoring | Use informative Gamma(10, 10/50K) + scale total_votes |
| Digital Beta adds divergences | Medium | Convergence failure | Start with high `target_accept=0.99`; reduce φ_digital_base prior scale |
| Internet rate weighting too aggressive | Low | Digital signals underweighted | Use national avg (not min); allow user override |
| 2026 data format mismatch | Medium | Backtest fails | Validate schema up front; fail fast with clear error |
| 100K runtime exceeds 8h | Low | Overnight run needed | `init="adapt_full"` + Gamma phi reduces stiffness → faster draws |
| Transfer model still fails NetCDF | Low | Suboptimal transfer rates | Phase 0 installs netCDF4; verify with test |
| Candidate key mismatch 2022 vs 2026 | Medium | Shape errors | `build_round1_model` already handles dynamic candidate_keys |

---

## 8. File Manifest (what changes per phase)

### Phase 0
- `pyproject.toml`: Add `netCDF4`

### Phase 1
- `src/co_president/model_municipal.py`: Add `prior_only` parameter
- `src/co_president/model_round1.py`: Add `_extract_municipal_prior()`, accept `features`, `init="adapt_full"`
- `src/co_president/model_runoff_simple.py`: `init="adapt_full"`
- `src/co_president/model_runoff_matrix.py`: `init="adapt_full"` (if any sampling)
- `scripts/run_100k_report.py`: Pass `features` to round1 model, add ESS reporting

### Phase 2
- `src/co_president/config.py`: Add `concentration_election_prior_shape`, `concentration_election_votes_scale`
- `src/co_president/model_round1.py`: Gamma phi_elec, scaled election likelihood
- `src/co_president/model_runoff_simple.py`: Same Gamma phi_elec pattern

### Phase 3
- `src/co_president/model_round1.py`: Beta digital signal likelihood, internet rate weighting
- `src/co_president/model_runoff_simple.py`: Replace additive beta_digital with Beta likelihood
- `src/co_president/data_polls.py`: `_merge_digital_signals()` helper (if needed)
- `scripts/run_100k_report.py`: Pass `digital_signals` and `features` to both models

### Phase 4
- `src/co_president/config.py`: Add `target_year`
- `scripts/run_2026_backtest.py`: New file
- `scripts/run_100k_report.py`: May refactor shared functions into module

### Phase 5
- `scripts/run_100k_report.py`: CONFIG_PRODUCTION with 100K draws
- No new model code (all features from Phases 1-4)

---

## 9. Prototype Strategy

| Phase | Draws | Tune | Target R-hat | Pass condition |
|-------|-------|------|-------------|----------------|
| 0 | N/A | N/A | N/A | netCDF4 installs, make check passes |
| 1 | 2,000 | 2,000 | ≤ 1.30 | No crashes, mu_muni correct shape |
| 2 | 4,000 | 2,000 | ≤ 1.10 | phi_elec learned, honest MAE |
| 3 | 4,000 | 2,000 | ≤ 1.10 | Digital Beta active, T-1 > T-7 weight |
| 4 | 4,000 | 2,000 | ≤ 1.10 | Both years run cleanly |
| 5 | 100,000 | 10,000 | < 1.01 | All quality targets met |

---

## 10. Execution Order

```
Phase 0  →  Phase 1  →  Phase 2  →  Phase 3  →  Phase 4  →  Phase 5
   ↓           ↓           ↓           ↓           ↓           ↓
make check  2K debug   4K debug   4K debug   4K debug   100K prod
             ↓           ↓           ↓           ↓           ↓
          Phase 1    Phase 2    Phase 3    Phase 4    Report +
          gate       gate       gate       gate       quality
                                                       checklist
```

Each phase gates before proceeding to next. If gate fails: fix, re-run at same draw count, do NOT proceed.

---

## Appendix A: Why Not Ensemble / Two-Stage

**Rejected alternative 1 — Weighted ensemble**: `pred = w_polls * model_polls + w_digital * model_digital + w_muni * model_muni`. Rejected because ensembles can't share information between layers. A poll showing Petro at 30% should pull the digital model's estimate up too. In an ensemble, it wouldn't.

**Rejected alternative 2 — Two-stage (municipal → national)**: Train municipal model, use posterior as fixed input to national model. Rejected because national model can't refine municipal estimates. If the national model learns that Petro support is higher than the municipal model predicted, that information should flow back (true hierarchical model does this).

**Chosen: Single unified hierarchical model** — all evidence (municipal, polls, digital, election) flows into one posterior θ[t]. Digital signals have their own observation likelihood with dynamic precision. Municipal fundamentals inform initial prior. No information lost.

---

## Appendix B: Key Design Decisions

1. **φ_digital[t] exponential decay with decay_days=3**: SciELO 2023 shows T-1 error is 1.86pp, T-7 is much worse. Exponential decay with 3-day half-life captures this: precision at T-1 ≈ 72, at T-7 ≈ 10, at T-30 ≈ 0.005.

2. **Internet access weighting as multiplicative factor**: Simple and transparent. If 70% have internet, max digital precision is 70% of base. No per-municipality weighting needed (we use national average because θ[t] is national vote share).

3. **Gamma phi_elec over fixed**: Learned phi_elec means the model can express uncertainty about election results. With phi=500K, the model effectively recites the data. With Gamma(10, 10/50K), the model learns that the election result is strong evidence but not the only truth.

4. **init="adapt_full" everywhere**: ADVI initialization finds the posterior mode faster than random jitter. Critical for 4K prototype convergence.

5. **Municipal prior as Normal(μ, σ) on logit space**: The municipal model posterior mean and std are on the CLR/logit scale. Using a Normal prior on the first-time-point logit preserves this scale without requiring probability-to-logit conversions in the graph.
