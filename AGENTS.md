# AGENTS.md — Agent Instructions for co-president-2026

This file instructs AI agents (including this one) on how to work with this codebase. It covers development workflow, documentation standards, package verification, quality gates, and the pre-push CodeRabbit procedure.

---

## 1. Project Context

`co-president-2026` is a Python 3.12 package that ingests Colombian presidential poll data, cleans and normalizes it, consolidates official election results from two independent sources (Registraduría Nacional and MOE), and fits a Bayesian Dirichlet-Multinomial model with a reverse-time random walk to forecast vote shares. The MVP backtests against the 2022 election.

The authoritative design document is **`MVP_SPECS_GUIDE.md`**. Every SPEC-XX in that document defines what must happen. If a behavior is not in a spec, it is out of scope. No speculative features.

**Key libraries:** pandas, numpy, PyMC, ArviZ, matplotlib, pyarrow (runtime); pytest, ruff, pyright (dev).

---

## 2. Development Workflow (TDD)

This project follows strict TDD. Every change must go through this cycle:

```
  STEP 1: RED    — Write a failing test that defines the expected behavior.
  STEP 2: GREEN  — Write the minimum code to make the test pass.
  STEP 3: REFACTOR — Extract helpers, improve names, reduce duplication.
  STEP 4: TYPE-CHECK — pyright src/ must pass with zero errors.
  STEP 5: LINT   — ruff check src/ tests/ must pass with zero errors.
  STEP 6: COMMIT — Only when all gates pass.
```

**One spec at a time, sequentially.** SPEC-03 does not begin until SPEC-02 is fully tested and green. Specs that depend on earlier specs inherit all prior tests and must keep them green.

---

## 3. Code Documentation Standards

All generated code must be thoroughly and professionally documented following industry standards.

### 3.1 Docstrings

Use **Google-style docstrings** for every public function, class, and module.

**Module docstring** — at the top of every `.py` file:

```python
"""SPEC-XX: Short description of the module's purpose.

Longer description of what this module provides, key classes/functions,
and how it fits into the overall pipeline.
"""
```

**Class docstring:**

```python
class Candidate:
    """A presidential candidate across the election cycle.

    Attributes:
        key: Internal key matching column names in polls (e.g., "gustavo_petro").
        display_name: Human-readable name (e.g., "Gustavo Petro").
        coalition: Coalition the candidate runs under, or None if independent.
        first_round: Whether the candidate ran in the first round.
        runoff: Whether the candidate made the runoff.
    """
```

**Function docstring:**

```python
def normalize_undecided(df: DataFrame) -> DataFrame:
    """Redistribute undecided/NS/NR vote shares proportionally across all candidates.

    For every row where ``ns_nr > 0``, rescale all non-NS/NR columns so their
    shares sum to 100%. This avoids discarding the undecided fraction and
    assumes undecided voters will eventually distribute proportionally to
    declared preferences.

    Args:
        df: Poll DataFrame with raw vote share columns and an ``ns_nr`` column.

    Returns:
        A new DataFrame with normalized candidate shares. Rows where
        ``ns_nr == 0`` are returned unchanged.

    Raises:
        ValueError: If any row has total shares exceeding 100% by more than
            a 1% rounding tolerance after normalization.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"gustavo_petro": [30.0], "ns_nr": [10.0], "blanco": [5.0]})
        >>> normalize_undecided(df)["gustavo_petro"].iloc[0]
        33.33  # 30.0 / (100 - 10) * 100
    """
```

### 3.2 Type Annotations

Every function parameter and return type must be annotated. Every class attribute must be annotated. No `Any` unless mathematically justified.

```python
def load_raw_polls(path: str | None = None) -> pd.DataFrame: ...
```

### 3.3 Commenting Philosophy

- **Comments explain "why", not "what".** The code itself should be self-documenting about what it does.
- Reserve comments for: design decisions, algorithm trade-offs, non-obvious edge cases, references to external sources or spec sections.
- Example:
  ```python
  # The Invamer row 26 has fecha=2022-04-19 but the correct date is
  # 2022-05-19. This is a known data-entry error documented in co_elect
  # (1a_v.R:9-12). The correction makes the poll contemporaneous with the
  # actual campaign period.
  ```

### 3.4 Private vs Public

- Prefix internal/helper functions with `_` (e.g., `_validate_pollster_ratings`).
- Only expose public functions that are tested and called from other modules.
- Document private helpers with a one-line docstring or a `#` comment.

---

## 4. Package Documentation Verification

Before writing any code that uses a package from the dependency list, the agent must verify the API at the documented URL. This ensures the code uses the correct function signatures, parameter names, and import paths for the pinned version.

### 4.1 Runtime Dependencies

| Package | Version | Documentation URL |
|---------|---------|------------------|
| pandas | ≥2.2 | https://pandas.pydata.org/docs/ |
| numpy | ≥1.26 | https://numpy.org/doc/stable/ |
| PyMC | ≥5.15 | https://www.pymc.io/welcome.html |
| ArviZ | ≥0.18 | https://python.arviz.org/ |
| matplotlib | ≥3.8 | https://matplotlib.org/stable/ |
| pyarrow | ≥15.0 | https://arrow.apache.org/docs/python/ |

### 4.2 Dev Dependencies

| Package | Version | Documentation URL |
|---------|---------|------------------|
| pytest | ≥8.0 | https://docs.pytest.org/ |
| ruff | ≥0.4 | https://docs.astral.sh/ruff/ |
| pyright | ≥1.1 | https://microsoft.github.io/pyright/ |

### 4.3 Tooling

| Tool | Documentation URL |
|------|------------------|
| uv | https://docs.astral.sh/uv/ |
| pyproject.toml (PEP 621) | https://packaging.python.org/en/latest/guides/writing-pyproject-toml/ |

### 4.4 Key PyMC API References (SPEC-06–08)

Before using any of these, verify the signature at the linked page:

| API | URL |
|-----|-----|
| `pm.Model` | https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.Model.html |
| `pm.Normal` | https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.Normal.html |
| `pm.HalfNormal` | https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.HalfNormal.html |
| `pm.DirichletMultinomial` | https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.DirichletMultinomial.html |
| `pm.Gamma` | https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.Gamma.html |
| `pm.Deterministic` | https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.Deterministic.html |
| `pm.sample` | https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample.html |
| `pm.Data` | https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.Data.html |
| `pm.sample_prior_predictive` | https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample_prior_predictive.html |
| `pm.sample_posterior_predictive` | https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample_posterior_predictive.html |
| `pm.math.softmax` | https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.math.softmax.html |
| `az.summary` | https://python.arviz.org/en/stable/api/generated/arviz.summary.html |
| `az.plot_trace` | https://python.arviz.org/en/stable/api/generated/arviz.plot_trace.html |

### 4.5 Verification Procedure

1. Look up the URL for the package you need to use.
2. Verify the import path and function signature match the pinned version.
3. If the documentation describes a newer API that is incompatible, use the version-appropriate API (refer to the pinned version range).
4. Do not assume API compatibility from memory alone — always check.

---

## 5. Pre-Commit Quality Gates

Before every commit, the agent must run all three quality gates. The `make check` target runs them in sequence.

```bash
make check
```

This is equivalent to:

```bash
uv run ruff check src/ tests/       # Zero lint errors (ALL rules)
uv run pyright src/                  # Zero type errors (strict mode)
uv run pytest tests/ -v             # All tests pass
```

All three must return exit code 0. If any fails, fix the issue before committing.

---

## 6. CodeRabbit Pre-Push Workflow

Before pushing any changes to remote, the agent must execute this CodeRabbit review loop.

### 6.1 Procedure

```
  STEP 1: RUN
    cr --agent --type uncommitted
    Run in the background. Let it take as long as it needs.
    Check on it periodically.

  STEP 2: EVALUATE
    Review all findings.
    - Fix only MAJOR and CRITICAL issues.
    - Ignore all nits (style preferences, minor suggestions,
      non-blocking comments, documentation nits).

    If zero critical/major issues → skip to STEP 4.

  STEP 3: FIX
    Implement fixes for all critical and major issues.
    Run `make check` to confirm no regressions after each fix.

  STEP 4: RE-RUN (second and final run)
    cr --agent --type uncommitted
    - If this run finds NO critical/major issues:
        Ignore any remaining nits. The review is complete.
    - If this run finds NEW critical/major issues:
        Fix them. STOP. Do NOT run the loop a third time.

  NEVER run the loop more than twice.
```

### 6.2 Summary

After completing the workflow, produce a summary stating:

1. What critical/major issues were found (if any).
2. Which were fixed and why.
3. Which were deferred and why (e.g., "deferred to SPEC-09 as out of scope").
4. The result of the second CodeRabbit run.

---

## 7. Test Conventions

- **Test files mirror `src/` one-to-one:**
  ```
  src/co_president/config.py          →  tests/test_config.py
  src/co_president/data.py            →  tests/test_data.py
  src/co_president/aggregation.py     →  tests/test_aggregation.py
  src/co_president/model_round1.py    →  tests/test_model.py
  src/co_president/model_runoff_simple.py →  tests/test_model.py
  src/co_president/model_runoff_matrix.py →  tests/test_model.py
  src/co_president/validation.py      →  tests/test_validation.py
  src/co_president/plotting.py        →  tests/test_validation.py
  src/co_president/__main__.py        →  tests/test_cli.py
  ```
- **Test data is deterministic and minimal.** Use hardcoded 3–5 row DataFrames for unit tests. Never load full CSV files in unit tests. Full datasets are only used in SPEC-09 (integration validation) and SPEC-10 (CLI commands).
- **MCMC test strategy** (from SPEC §3.2.6):
  1. **Graph test:** Model builds without error; expected RV/deterministic/observed counts are asserted.
  2. **Prior predictive test:** `pm.sample_prior_predictive()` returns valid ranges (e.g., vote shares in [0, 1]).
  3. **Convergence test** (slow, marked `@pytest.mark.slow`): Fit on minimal data; R-hat < 1.10.
  4. **Sanity test:** 3 artificial polls with known ground truth; posterior mean within 5pp.
- Mark slow MCMC tests with `@pytest.mark.slow`. Fast tests (graph, prior predictive) have no marker.

---

## 8. Code & Tooling Style

### 8.1 Python Conventions

| Setting | Value |
|---------|-------|
| Line length | 100 |
| Quotes | Double |
| Line endings | LF |
| isort first-party | `co_president` |
| isort force-sort-within-type-sections | true |

All configured in `[tool.ruff]` in `pyproject.toml`.

### 8.2 Ruff Configuration

- **Rule set:** `ALL` (all rules enabled by default).
- **Globally ignored:**
  - `D100` (missing docstring in public module — waived because `MVP_SPECS_GUIDE.md` is the documentation)
  - `D104` (missing docstring in public package)
  - `S101` (use of assert — allowed in tests via per-file ignores)
- **Per-file ignores for tests:**
  - `tests/**/*.py` → `S101`, `PLR2004`
- **Docstrings:** D100/D104 are waived at module/package level, but ALL other docstring rules still apply (functions need docstrings).

### 8.3 Pyright Configuration

- **Mode:** `strict`
- **Extra checks:** `reportMissingTypeStubs = true`, `reportMissingImports = true`, `reportUnusedImport = true`

---

## 9. Commit Conventions

- **Format:** `feat(SPEC-XX): <description>` for feature implementations.
  - Example: `feat(SPEC-02): add Candidate dataclass and pollster ratings`
- **Format:** `test(SPEC-XX): <description>` for test-only commits.
  - Example: `test(SPEC-02): add model config default value tests`
- **Format:** `fix(SPEC-XX): <description>` for bug fixes.
  - Example: `fix(SPEC-04): correct Invamer date normalization edge case`
- **Format:** `chore: <description>` for infrastructure, tooling, or dependency updates.
- A commit must only happen after all three quality gates pass.
- Each spec commit should group the test and implementation together (the full RED→GREEN→REFACTOR cycle).
- Use `make check` before every commit to verify gates.

---

## 10. Additional Agent Directives

### 10.1 Under no circumstances should the agent:

- Run `git commit` without running the quality gates first.
- Use a package without verifying its API documentation as specified in §4.
- Add code without a corresponding test.
- Write code for a SPEC that has not been started yet.
- Use `Any` in type annotations unless mathematically justified and documented with a comment.
- Skip docstrings on public functions, classes, or modules.

### 10.2 The agent should always:

- Keep `MVP_SPECS_GUIDE.md` open as reference when implementing a SPEC.
- Read the full spec section (including TDD Steps and Acceptance Criteria) before writing any code.
- Run the full test suite after implementing changes, not just the newly added tests.
- Notify the user when a spec's acceptance criteria are met.
