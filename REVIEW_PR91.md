# Senior Dev Review: PR #91 — Issues #79 & #80

> **Reviewer:** Senior Dev (strict mode activated)
> **Reviewee:** Junior Dev
> **Branch:** `feat/track-b-data`
> **Scope:** Changes for Issue #79 (exclude MassiveCaller forced-choice R2 polls) and Issue #80 (fix YanHaas 103% sum anomaly)
> **Date:** 2026-05-22

---

## Executive Summary

The core logic for both issues is **directionally correct**, but the execution is careless and unprofessional. I found **8 critical issues**, **6 improvements**, and **3 nits**. The most egregious problems:

1. **A merge artifact duplicated the entire `load_and_clean_all` pipeline** — requiring a follow-up fix commit.
2. **A committed placeholder test (`assert True`)** giving false confidence.
3. **Dead warning code** in `CleanPolls.__post_init__` that will never fire.
4. **Tests deleted and misplaced** to make room for new code, causing coverage regression.
5. **Docstrings that lie** about the behavior of the code.

This is exactly the kind of sloppy, AI-assisted work I expected. You clearly did not review your own diff before committing. Do better.

---

## Merge Process Notes

Before this review could even begin, the branch needed manual intervention:

- **Conflict:** `tests/test_data.py` import block collision between PR imports (`_detect_forced_choice`) and dev imports (`_validate_normalized_rows`, helper functions).
- **Merge artifact:** `load_and_clean_all` contained a **complete duplicate** of steps 4–5 (`retain_active_candidates` + `infer_round_number` + `_detect_forced_choice`), with `all_polls = polls.copy()` overwriting the earlier snapshot and destroying the `all_polls` semantics.
- **Test bug:** `test_forced_choice_detection_identifies_massivecaller` had misaligned test data where index 31 contained `blanco=5.0, ns_nr=0.0` but the test asserted it should be flagged. The data lists were shifted relative to the index.

All three issues were resolved and committed before this review was conducted.

---

## Critical Issues (MUST FIX before merge)

### 1. Merge Artifacts in `load_and_clean_all` — ALREADY FIXED IN FOLLOW-UP
**File:** `src/co_president/data_polls.py` (original commit `597959d`)  
**Status:** Fixed in commit `7bc6642`

The original PR commit introduced a **complete duplicate** of steps 4–5 in `load_and_clean_all`:
- `retain_active_candidates` called **twice** on `polls`
- `infer_round_number` called **twice**
- `_detect_forced_choice` called **twice**
- `all_polls = polls.copy()` overwritten **after** filtering, destroying the snapshot semantics

The fact that a separate commit was needed to repair merge damage is evidence you **did not inspect your own diff before committing**. You should have caught this in `git diff --cached`.

### 2. Dead Warning Code in `CleanPolls.__post_init__`
**File:** `src/co_president/data_polls.py`, lines 228–237

```python
if "forced_choice" in self.round2.columns:
    fc_mask = self.round2["forced_choice"]
    if fc_mask.any():
        ...
```

This warning **will never fire** in normal operation. `load_and_clean_all` filters forced-choice rows **before** constructing `CleanPolls`:

```python
mask_r2 = (polls["round_number"] == _ROUND_TWO) & (~polls["forced_choice"])
round2_df = polls[mask_r2].copy()
```

Even though the `forced_choice` *column* may exist in `round2_df`, all values are `False`, so `fc_mask.any()` is always `False`. This is performative "defensive coding" that defends against nothing. **Remove this block or move the filtering logic into `CleanPolls` itself** so the warning actually has a purpose.

### 3. `fix_yanhaas_20220611` Docstring Misrepresents Threshold
**File:** `src/co_president/data_polls.py`, lines 500–504

Docstring claims: *"If total > 100% ± 1%, renormalizes rows."*

Code actually uses `_RENORMALIZE_THRESHOLD = 0.01` (1 **hundredth** of a percent, not 1 percent). The docstring says "±1%" so a reader expects no renormalization at 100.5%, but the code **will** renormalize because `0.5 > 0.01`. Conversely, at 100.009%, the reader expects renormalization (since 0.009 < 1), but the code **skips** it because `0.009 < 0.01`. The point is: **the docstring and the code disagree**. Fix the docstring to match the code, or change the threshold if 1% was intended. Do not lie to the reader.

### 4. Placeholder Test Committed: `assert True`
**File:** `tests/test_data.py`, line 1472

```python
def test_yanhaas_20220611_preserves_ratios(self) -> None:
    """Verify proportional redistribution."""
    # ... implementation ...
    assert True
```

This is a **committed TODO**. There is no implementation. The test passes vacuously and gives false confidence. This is exactly the kind of AI slop I expect from a junior who generates code without understanding it. **Complete this test or delete it** — do not leave `assert True` in the test suite.

### 5. Test Coverage Regression: `test_empty_dataframe_returns_empty_copy` Deleted
**File:** `tests/test_data.py`

The diff shows this existing test was **removed** from `TestDeduplicatePolls` to make room for the new classes. It was never restored. This is a real coverage regression. **Restore this test** in `TestDeduplicatePolls`.

### 6. Tests Misplaced in Wrong Class
**File:** `tests/test_data.py`, lines 1474–1500

`test_logs_deduplication_summary` and `test_no_log_when_no_duplicates` are inside `TestYanHaasAnomaly`. They test `deduplicate_polls`, not YanHaas. This is sloppy copy-paste. **Move them back to `TestDeduplicatePolls`**.

### 7. Missing Tests That Were Claimed to Exist
The PR description lists these tests, but they are **NOT in the code**:

- `test_massivecaller_r2_excluded` — should verify 0 MassiveCaller rows in `CleanPolls.round2`
- `test_normalize_undecided_yearly_yanhaas` — should verify June 5 YanHaas (101%) handled by existing normalization

**Write these tests.** Without them, the integration is unverified.

### 8. AI Slop Comments in Original Commit
**File:** `tests/test_data.py` (commit `597959d`)

Original commit contained:
```python
# Add index 31, 36, 42 - but dataframe is small.
# The prompt says "identifies_massivecaller (flags rows 31, 36, 42)".
# I will create a df with those indices.
```

These are **AI self-referential comments** that leaked into production code. They were removed in a follow-up commit, but the fact they were committed at all reveals you are copy-pasting from LLM output without review.

---

## Improvements (SHOULD FIX)

### 9. `fix_yanhaas_20220611` Should Be Private
**File:** `src/co_president/data_polls.py`, line 500

This is a one-off data fix for a specific pollster and date. It is **not** part of the public API. Per project conventions: *"Prefix internal helpers with `_`"*. Rename to `_fix_yanhaas_20220611`.

### 10. `_detect_forced_choice` Docstring Incomplete
**File:** `src/co_president/data_polls.py`, lines 485–489

Missing `Args` and `Returns` sections. Per project conventions (Google-style docstrings required on every public function), this needs:

```python
def _detect_forced_choice(df: pd.DataFrame) -> pd.Series:
    """Detect forced-choice R2 polls.

    Flags rows where blanco and ns_nr are NA, and petro+hernandez sum to 100±1%.

    Args:
        df: Poll DataFrame with candidate share columns.

    Returns:
        Boolean Series indexed like ``df``, True for forced-choice polls.
    """
```

### 11. `_detect_forced_choice` Edge Case: NA Petro/Hernandez
**File:** `src/co_president/data_polls.py`, lines 490–491

```python
petro = df["gustavo_petro"].fillna(0)
hernandez = df["rodolfo_hernandez"].fillna(0)
```

If `petro` is NA and `hernandez` is 100.0, with `blanco` and `ns_nr` both NA, this row gets flagged as forced-choice even though Petro data is missing. While `infer_round_number` normally prevents NA values in round 2, this function is not robust if called independently. Add an explicit check:

```python
both_present = df["gustavo_petro"].notna() & df["rodolfo_hernandez"].notna()
return blanco_na & ns_nr_na & sum_check & both_present
```

### 12. `all_polls` Test Weakened
**File:** `tests/test_data.py`, `TestLoadAndCleanAll`

The diff changed:
```python
- assert "alejandro_gaviria" in self.clean_polls.all_polls.columns
+ assert "gustavo_petro" in self.clean_polls.all_polls.columns
```

This weakens the test. `gustavo_petro` exists in both `all_polls` and filtered DataFrames. `alejandro_gaviria` is a pre-consultation column that ONLY survives in `all_polls`. The original assertion verified the snapshot semantics. **Restore the original assertion** (or add both).

### 13. `TestMassiveCallerR2` Fixture Is Unused
**File:** `tests/test_data.py`, lines 1402–1414

The `r2_polls` fixture creates a DataFrame that **no test in the class uses**. Both tests create their own DataFrames. Remove the dead fixture or use it.

### 14. `TODO` Comment at EOF
**File:** `src/co_president/data_polls.py`, line 925

```python
# TODO(SPEC-07): Implement R-hat guard. # noqa: FIX002, TD003
```

A TODO with `# noqa` suppressions in production code. Either implement it, open an issue, or delete the comment. The `# noqa` is a confession that you know it's wrong.

---

## Nits (COULD FIX)

### 15. `fix_yanhaas_20220611` Log Message Uses `idx` Which Is a pandas Index Label
After `result.loc[idx, "ns_nr"] = 0.0`, the log uses `idx`. If the DataFrame has a non-integer index, `idx` might not be a row number. The message format `%s` handles this fine, but `%d` would crash. Not a bug, just noting.

### 16. `_detect_forced_choice` Return Type Could Be Narrower
`-> pd.Series` is correct but vague. `-> pd.Series[bool]` would be more precise. (Minor pyright hint.)

---

## Action Plan

| # | Priority | Task | File | Line(s) |
|---|----------|------|------|---------|
| 1 | **Critical** | Delete dead warning block in `CleanPolls.__post_init__` or move R2 filtering into `CleanPolls` so the warning is meaningful | `data_polls.py` | 228–237 |
| 2 | **Critical** | Fix `fix_yanhaas_20220611` docstring: change "± 1%" to match actual `_RENORMALIZE_THRESHOLD` of 0.01 | `data_polls.py` | 500–504 |
| 3 | **Critical** | Complete or remove `test_yanhaas_20220611_preserves_ratios` — no `assert True` in tests | `test_data.py` | ~1472 |
| 4 | **Critical** | Restore `test_empty_dataframe_returns_empty_copy` to `TestDeduplicatePolls` | `test_data.py` | `TestDeduplicatePolls` |
| 5 | **Critical** | Move `test_logs_deduplication_summary` and `test_no_log_when_no_duplicates` back to `TestDeduplicatePolls` | `test_data.py` | 1474–1500 |
| 6 | **Critical** | Add missing tests: `test_massivecaller_r2_excluded` and `test_normalize_undecided_yearly_yanhaas` | `test_data.py` | New |
| 7 | **Improvement** | Rename `fix_yanhaas_20220611` → `_fix_yanhaas_20220611` to follow internal-helper convention | `data_polls.py` | 500 |
| 8 | **Improvement** | Harden `_detect_forced_choice` with `both_present` check | `data_polls.py` | 490–497 |
| 9 | **Improvement** | Complete `_detect_forced_choice` docstring with `Args`/`Returns` | `data_polls.py` | 485–489 |
| 10 | **Improvement** | Restore `assert "alejandro_gaviria" in self.clean_polls.all_polls.columns` in `TestLoadAndCleanAll` | `test_data.py` | `TestLoadAndCleanAll` |
| 11 | **Improvement** | Remove unused `r2_polls` fixture from `TestMassiveCallerR2` | `test_data.py` | 1402–1414 |
| 12 | **Improvement** | Remove the `TODO` comment at EOF or move to an issue | `data_polls.py` | 925 |
| 13 | **Nit** | Consider narrowing `_detect_forced_choice` return type to `pd.Series[bool]` | `data_polls.py` | 485 |
| 14 | **Process** | **Run `make check`** and ensure `ruff`, `pyright`, and `pytest` all pass before re-requesting review | All |

---

## Closing Thoughts

You got the logic right, but you packaged it like a student rushing a deadline. The merge damage, the placeholder test, the misplaced tests, the dead code, and the lying docstring are all symptoms of the same disease: **you are not reading your own code before you commit it.**

I do not care that you use AI. I care that you use it as a replacement for thinking instead of a tool to augment it. An AI can generate code; only a human can verify it. Start acting like the second half of that sentence.

Fix the critical issues first. Then the improvements. Then run `make check`. Only then ask me for a re-review.
