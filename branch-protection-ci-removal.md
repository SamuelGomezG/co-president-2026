# Branch Protection Rule Update — ci.yml Removal

## What changed

The legacy `ci.yml` workflow has been deleted. The three parallel workflows (`ci-fast.yml`, `ci-model.yml`, `ci-quality.yml`) now cover all checks independently.

## Required GitHub Branch Protection Updates

Navigate to **Settings → Branches → Branch protection rules** for each protected branch (`dev`, `main`).

### 1. Remove old required check

Uncheck the following requirement:

- ❌ `CI / test` (ci.yml was a single-job workflow named `CI`)
- ❌ `CI / model-tests` (the old `model-tests` job that ran after `test`)

### 2. Add new required checks

Check the following three (they run in parallel):

- ✅ `CI - Fast / lint-typecheck` — ruff format/check + pyright (from `ci-fast.yml`)
- ✅ `CI - Fast / unit-tests` — pytest fast (from `ci-fast.yml`)
- ✅ `CI - Model / model-tests` — MCMC graph + prior predictive (from `ci-model.yml`)
- ✅ `CI - Quality (Security & Code Quality) / security` — pip-audit + bandit (from `ci-quality.yml`)
- ✅ `CI - Quality (Security & Code Quality) / changes` — path filter job (from `ci-quality.yml`, allows the security job to be skipped)

> **Note**: `CI - Quality` has two jobs (`changes` and `security`). Consider making only `security` required, since `changes` is a metadata-only job.

### 3. Verify PR title check

- ✅ `PR Title` should already be configured if you had it — no change needed.

## Summary table

| Old (removed) | New (required) |
|---|---|
| `CI / test` | `CI - Fast / lint-typecheck` |
| `CI / test` | `CI - Fast / unit-tests` |
| `CI / model-tests` | `CI - Model / model-tests` |
| *(not present)* | `CI - Quality (Security & Code Quality) / security` |

After updating, merge this PR and verify that the next PR shows the new checks as required.
