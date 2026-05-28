# PLAN — Issue #151

## Summary
**Title:** fix(SPEC-03): rename load_actual_results to load_canonical_results throughout codebase and spec  
**SPEC:** SPEC-03 — §5.1.7, §6.1.3 item 7  
**Classification:** REFACTOR | **Isolation:** ISOLATED | **Velocity:** QUICK WIN  
**Stream:** Phase 1 / Stream C

---

## Objective

The spec mandates `load_actual_results()` but the implementation uses `load_canonical_results()`. A stale reference to the old name persists in `test_data_results.py:622` and in `MVP_SPECS_GUIDE.md` §5.1.7. The preferred fix is to update the spec and stale reference to match the code's better name.

---

## Files to Modify

| File | Change |
|------|--------|
| `tests/test_data_results.py` | Change `load_actual_results` → `load_canonical_results` at line ~622 |
| `MVP_SPECS_GUIDE.md` | Update §5.1.7 to reference `load_canonical_results()` |

---

## Implementation Plan

### Step 1: Find the stale reference in `test_data_results.py`

Open `tests/test_data_results.py` and go to line 622 (or search for `load_actual_results`):

```bash
grep -n "load_actual_results" tests/test_data_results.py MVP_SPECS_GUIDE.md
```

### Step 2: Fix the test file

If the test file imports or references `load_actual_results`, change it to `load_canonical_results`. The function is already named `load_canonical_results` in `data_results.py` — this is purely a naming fix for stale references.

Example fix:
```python
# Before:
from co_president.data_results import load_actual_results

# After:
from co_president.data_results import load_canonical_results
```

### Step 3: Update the spec document

In `MVP_SPECS_GUIDE.md` §5.1.7, the text currently says:

> `load_actual_results() -> tuple[RoundResult, RoundResult]` is a lazy-loading function...

Change to:

> `load_canonical_results() -> tuple[RoundResult, RoundResult]` is the single entry point...

Also check §6.1.3 item 7 which says "`load_canonical_results()`" — that section may already be correct, in which case only §5.1.7 needs updating.

### Step 4: Verify no other stale references

```bash
grep -rn "load_actual_results" src/ tests/ docs/ *.md
```

All hits should be resolved.

---

## Quality Gates

| Gate | Command |
|------|---------|
| Lint | `ruff check tests/test_data_results.py` |
| Format | `ruff format tests/test_data_results.py` |
| Tests | `uv run pytest tests/test_data_results.py -v` |
| Spec review | Manual check of `MVP_SPECS_GUIDE.md` §5.1.7 |

---

## Commit

```
fix(SPEC-03): rename stale load_actual_results references to load_canonical_results

Updates the spec §5.1.7 and test_data_results.py to use the
implementation's canonical name load_canonical_results, preventing
ImportError for developers following the spec.
```
