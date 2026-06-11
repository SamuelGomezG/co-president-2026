# Contributing to co-president-2026

Thank you for your interest! This project uses a strict TDD workflow with CodeRabbit review gates.

## Quick Start

```bash
make setup              # Install dependencies + local package
make check              # fmt → lint → typecheck → test (all must pass before commit)
```

## Development Workflow

1. **Branch**: Create feature branches off `dev` as `feat/spec-XX-description`.
2. **Specs**: Every change must tie to a SPEC from `MVP_SPECS_GUIDE.md` or `ENHANCEMENT_ROADMAP.md`. If behavior isn't in a SPEC, it's out of scope.
3. **TDD cycle**: Write a failing test → minimum code to pass → refactor → `make check` → commit.
4. **CodeRabbit**: Run `cr` before every commit. Do not commit without a passing review.
5. **Commit format**: `feat(SPEC-XX): description`, `test(SPEC-XX): description`, `fix(SPEC-XX): description`, or `chore: description` (non-scope changes). No other commit types are allowed. See `AGENTS.md` §7.

## Code Conventions

- Google-style docstrings on every public function/class/module
- Full type annotations (no `Any` unless mathematically justified)
- ruff ALL rules enabled (line length 100, double quotes, LF endings)
- pyright strict mode on `src/` (zero errors required)
- Prefix internal helpers with `_`

## Testing

- Test files mirror `src/co_president/` one-to-one
- Use hardcoded 3–5 row DataFrames (never load real CSV files)
- Model tests use 4-phase strategy: graph → prior predictive → convergence → sanity
- Run `make test` for the full suite, `make test-fast` to skip slow MCMC tests

## Questions?

See `AGENTS.md` for the full agent operating manual, or the contributing guide on the [project board](https://github.com/users/SamuelGomezG/projects/2).
