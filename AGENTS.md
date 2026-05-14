# AGENTS.md — Agent Instructions for co-president-2026

This file tells AI agents how to work with this codebase without making avoidable mistakes.

---

## 1. Source of Truth

**`MVP_SPECS_GUIDE.md`** is the authoritative design document. Every `SPEC-XX` defines what must happen and nothing else. If a behavior is not in a spec, it is out of scope. No speculative features.

---

## 2. Branch Strategy

```text
main                    ← Only merged when the full pipeline is working
  └── dev               ← Long-lived development branch; all feature branches fork from here
        └── feat/*      ← One feature branch per SPEC (e.g., feat/project-scaffolding)
```

- Create feature branches off `dev`, NOT `main`.
- Merge feature branches into `dev` via PR.
- Do NOT merge into `main` until the entire MVP is validated.

---

## 3. Developer Commands (do NOT guess these)

### First-time setup

```bash
uv sync --extra dev          # Install runtime + dev dependencies (does NOT install local package)
uv pip install -e .          # Install co-president in editable mode (REQUIRED after uv sync)
```

**`uv sync` alone will NOT install dev tools (pytest, ruff, pyright) or the local package.** You need both commands above, in that order.

### Quality gates — required before EVERY commit

```bash
make check
```

This runs `fmt` → `lint` → `typecheck` → `test` in sequence. **The formatter (`ruff format`) runs first and may modify files in place.** Re-inspect after `make check` before committing.

Individual gates:
```bash
make fmt          # ruff format src/ tests/
make lint         # ruff check src/ tests/        (ALL rules, zero tolerance)
make typecheck    # pyright src/                   (strict mode, zero errors)
make test         # pytest tests/ -v               (all tests must pass)
make test-fast    # Skip slow MCMC model tests
make test-model   # Fast model tests only (graph + prior predictive)
make test-model-slow  # Slow MCMC tests only (convergence + sanity)
```

### Security scanning
```bash
make sec          # Runs pip-audit + bandit
```
Also available: `pre-commit run --all-files` for pre-push validation.

### One-test shortcuts
```bash
uv run pytest tests/test_config.py -v                # Single file
uv run pytest tests/test_config.py::test_project_exists -v  # Single test
```

---

## 4. TDD Cycle (non-negotiable)

```text
RED        →  Write a failing test
GREEN      →  Write minimum code to pass
REFACTOR   →  Clean up, add docstrings, annotate types
make check →  fmt + lint + typecheck + test  (ALL must return exit 0)
COMMIT     →  Only after all gates pass
```

---

## 5. Key Code Conventions

- **Google-style docstrings** on every public function, class, and module. Module docstrings start with `"""SPEC-XX: ..."""`.
- **Full type annotations** on all parameters, returns, and class attributes. No `Any` unless mathematically justified.
- **ruff rules**: `ALL` enabled. Globally ignored: `D100`, `D104`, `S101`. Tests additionally ignore `PLR2004`.
- **Line length**: 100. **Quotes**: double. **Line endings**: LF.
- **Imports**: isort with `known-first-party = ["co_president"]`, `force-sort-within-sections = true`.
- Prefix internal helpers with `_`.
- Comments explain **why**, not what.

---

## 6. Test Conventions

- **Test files mirror `src/co_president/` one-to-one**. Model tests (`model_round1.py`, `model_runoff_simple.py`, `model_runoff_matrix.py`) all share `tests/test_model.py`.
- **Test data is minimal**: hardcoded 3–5 row DataFrames. Never load full CSV files in unit tests.
- **MCMC tests use a four-phase strategy**:
  1. **Graph test** (fast) — model builds, correct RV/deterministic counts
  2. **Prior predictive test** (fast) — samples fall in [0,1]
  3. **Convergence test** (slow, `@pytest.mark.slow`) — R-hat < 1.10 on minimal data
  4. **Sanity test** (slow) — 3 synthetic polls, posterior mean within ±5pp of known truth
- The `data_dir` fixture in `tests/conftest.py` provides the project data directory path.

---

## 7. Commit Format

```text
feat(SPEC-02): add Candidate dataclass and pollster ratings
test(SPEC-02): add model config default value tests
fix(SPEC-04): correct Invamer date normalization edge case
chore: update ruff configuration
```

---

## 8. GitHub Project Management

Each SPEC has a corresponding GitHub issue with `spec:XX` label, assigned to a milestone (Phase 1–5). Check the [project board](https://github.com/users/SamuelGomezG/projects/2) for current status. The `spec:06a` through `spec:06d` sub-labels cover the four sub-issues of SPEC-06.

---

## 9. CodeRabbit

`.coderabbit.yaml` is configured for `dev` branch reviews. Before pushing, run:

```bash
cr --agent --type uncommitted
```

Fix MAJOR and CRITICAL issues (not nits). Re-run once. NEVER loop more than twice.

---

## 10. Data Files (one-time setup)

Large Registraduría MMV files are `.csv.gz` and must be decompressed before use:

```bash
gunzip data/2022-presidential-results/MMV_NACIONAL_PRESIDENTE_2022_*.csv.gz
```

The `.gitignore` ignores uncompressed `.csv` copies to prevent accidental re-commits.

---

## 11. Operational Gotchas

- **`uv run python` may fail with `ModuleNotFoundError`** if the virtual environment was created but the local package wasn't installed. Run `uv pip install -e .` to fix.
- **`ruff format` auto-modifies files** during `make check`. Check `git status` after to stage any formatting changes before committing.
- **PyMC ≥6.0 is installed** (pyproject allows ≥5.15). API version differences from PyMC 5.x docs are possible — verify signatures at the linked docs before using any PyMC class.
- **`.python-version` pins 3.12.13**. `uv` will use whatever 3.12 interpreter it finds; ensure 3.12.13 is available via pyenv or uv.
- **Large CSV files** (`MMV_NACIONAL_PRESIDENTE_2022_1v.csv.gz`) are ~95 MB each. Unit tests must never load them.
- **Package verification**: Always check the current API docs for a library before writing code. Do not trust memory. Key reference URLs are in `MVP_SPECS_GUIDE.md` §2.
