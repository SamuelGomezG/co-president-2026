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
make sync          # Install runtime + dev dependencies (does NOT install local package)
make install       # Install co-president in editable mode (REQUIRED after make sync)
```

**`make sync` installs runtime + dev extras (pytest, ruff, pyright) but NOT the local package.** You need both commands above, in that order.

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
- **ruff rules**: `ALL` enabled. Globally ignored: `D100`, `D104`. Tests additionally ignore `S101`, `PLR2004`.
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

## 9. CodeRabbit (Manual User Review)

**The CodeRabbit workflow MUST be executed BEFORE every commit — no exceptions. Never commit without a successful CodeRabbit review first.**

The agent MUST NOT run CodeRabbit automatically. The **user** executes the review manually using the `cr` CLI.

Before committing, the agent is responsible for halting development and printing the exact command the user needs to run. The agent must infer the correct context (current branch, staged/unstaged changes, etc.) and produce a command tailored to the situation, for example:

```bash
# Review all uncommitted changes (default)
cr

# Review against a non-main base branch
cr --base develop

# Run with agent-mode JSON output for structured results
cr --agent

# Interactive terminal UI
cr --interactive
```

The agent MUST then wait for the user to confirm:
1. That the review was completed successfully, **AND**
2. That any critical/major issues found have been addressed (the user decides what is critical).

Only after the user gives explicit approval to proceed should the agent move to the commit phase. Do not skip, shorten, or automate this step.

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

---

## 12. Custom Agents & Skills

This project has access to a rich ecosystem of custom sub-agents (via the `task` tool) and skills (via the `skill` tool). Use them proactively based on the task at hand.

### Sub-agents (delegation via `task` tool)

| Sub-agent | When to use |
|---|---|
| `explore` | Codebase discovery — understanding project layout, finding relevant specs, or searching for patterns before implementing a feature |
| `git-smart-commit` | After CodeRabbit review is approved and changes are staged — generates granular semantic commits matching project conventions |
| `github-issue-writer` | Drafting detailed bug reports, spec tickets, or feature requests with correct labels, milestones, and project board coordination |
| `github-pr-writer` | Reviewing branch diffs and building professional PR bodies with architectural context and changelog entries |
| `post-merge-cleanup` | After a `feat/*` branch merges into `dev` — purges stale local/remote branches, cleans git worktrees, advances project board status |
| `coderabbit-assessment` | Processing CodeRabbit diagnostic output — audits findings for validity and compiles remediation paths or rebuttals (only after user runs the manual review) |

### Skills (context-loading via `skill` tool)

| Skill | When to load |
|---|---|
| `pandas-pro`, `Pandas Data Analysis` | Data manipulation, DataFrame cleaning, aggregation, merge operations on polling/MMV data |
| `python-testing-patterns` | Writing or debugging pytest fixtures, mocks, parametrize, or following the TDD cycle |
| `github-workflow-expert` | Creating issues, PRs, labels, milestones — broader project management that sub-agents don't cover |
| `python-executor` | Running sandboxed Python scripts for data processing, web scraping, or prototyping |
| `Machine Learning` | Working directly with PyMC models, scikit-learn, or PyTorch |
| `code-review` | Only as a **reference** — the actual review is run manually by the user per §9 |
| `autofix` | Safely applying CodeRabbit PR review-thread feedback from GitHub with per-change approval |

### Invocation pattern

When a task matches one of the above descriptions, load the skill or delegate to the sub-agent. Sub-agents run in their own context — provide them with a clear, self-contained prompt specifying what to return and how to verify their work.
