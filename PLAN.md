# PLAN — Issue #140

## Summary
**Title:** fix(SPEC-06): use Highest Density Interval instead of percentile-based credible intervals  
**SPEC:** SPEC-06 — §9.2.2 item 6, §15 (glossary: HDI)  
**Classification:** CORE | **Isolation:** ISOLATED  
**Stream:** Phase 2 / Stream A

---

## Objective

SPEC-06 mandates HDI (Highest Density Interval) for credible intervals, but the implementation uses symmetric percentile-based (equal-tailed) intervals via `np.percentile()`. For skewed posteriors, HDI produces narrower, more informative intervals. Replace `np.percentile()` with ArviZ's `az.hdi()` throughout `model_round1.py` and `model_runoff_simple.py`.

---

## Files to Modify

| File | Change |
|------|--------|
| `src/co_president/model_round1.py` | Replace `np.percentile()` calls in `forecast_round1()` with `az.hdi()` |
| `src/co_president/model_runoff_simple.py` | Replace `np.percentile()` calls in `forecast_runoff_simple()` with `az.hdi()` |
| `tests/test_model.py` | Update assertions that check for specific CI values |

---

## Implementation Plan

### Step 1: Understand the current implementation

In `forecast_round1()` (model_round1.py), the code likely does:

```python
# Current: percentile-based equal-tailed intervals
ci_50_lower = np.percentile(draws, 25, axis=0)
ci_50_upper = np.percentile(draws, 75, axis=0)
ci_95_lower = np.percentile(draws, 2.5, axis=0)
ci_95_upper = np.percentile(draws, 97.5, axis=0)
```

### Step 2: Replace with `az.hdi()`

```python
import arviz as az

# HDI-based intervals
ci_50 = az.hdi(draws, hdi_prob=0.5)   # returns (lower, upper) tuple per candidate
ci_95 = az.hdi(draws, hdi_prob=0.95)
```

`az.hdi()` returns a NumPy array with shape `(2, K)` where `[0, :]` is lower bounds and `[1, :]` is upper bounds.

### Step 3: Update the CandidateForecast construction

The HDI result needs to be unpacked correctly:

```python
ci_50 = az.hdi(draws, hdi_prob=0.5)
ci_95 = az.hdi(draws, hdi_prob=0.95)

candidate = CandidateForecast(
    candidate_key=key,
    mean_share=float(np.mean(draws)),
    median_share=float(np.median(draws)),
    ci_50=(float(ci_50[0, i]), float(ci_50[1, i])),
    ci_95=(float(ci_95[0, i]), float(ci_95[1, i])),
    ...
)
```

### Step 4: Apply the same change in `model_runoff_simple.py`

In `forecast_runoff_simple()`, replace any `np.percentile()` calls with `az.hdi()` for `ci_95_a` and `ci_95_b` (and median/50% CI if they will be added later).

### Step 5: Update tests

Tests in `test_model.py` may have assertions on CI values. Since HDI can differ from percentile-based intervals (especially for skewed distributions), update test expectations:
- For synthetic symmetric posteriors: HDI ≈ percentile intervals (verify within tolerance)
- For real/synthetic skewed posteriors: HDI may be narrower — update expected ranges

### Step 6: Add `arviz` import

Ensure `import arviz as az` is present at the top of both files (it likely already is, since both files use `az.summary` and `az.plot_trace`).

---

## Quality Gates

| Gate | Command |
|------|---------|
| Lint | `ruff check src/co_president/model_round1.py src/co_president/model_runoff_simple.py` |
| Format | `ruff format src/co_president/model_round1.py src/co_president/model_runoff_simple.py` |
| Typecheck | `pyright src/co_president/model_round1.py src/co_president/model_runoff_simple.py` |
| Tests (fast) | `uv run pytest tests/test_model.py -v -m "not slow"` |
| Tests (all model) | `uv run pytest tests/test_model.py -v` |

---

## Notes

- `az.hdi()` is already available since ArviZ ≥0.18 is a dependency
- HDI is a strictly more correct credible interval per Bayesian literature
- The difference from percentile intervals is minimal for well-behaved, roughly symmetric posteriors (like a balanced two-candidate race), but can be significant for skewed posteriors (like minor candidates with low vote shares near 0%)

---

## Commit

```
fix(SPEC-06): use Highest Density Interval instead of percentile-based credible intervals

Replaces np.percentile() with az.hdi() in forecast_round1 and
forecast_runoff_simple. HDI produces narrower, more informative
intervals for skewed posteriors as mandated by SPEC-06.
```
