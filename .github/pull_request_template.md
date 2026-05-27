<!--
  ============================================================================
  PR Title — semantic conventional commits (enforced by CI):
    type(SCOPE): description

  Allowed types: feat | fix | chore | test | docs | refactor | data | ci | sec
  Scope:         SPEC-XX for spec work; omit for cross-cutting changes
  Examples:      feat(SPEC-04): add forced-choice poll detection
                 fix(SPEC-02): correct Invamer date normalization edge case
                 chore: update ruff configuration
  ============================================================================

  Branch naming: type/brief-description
    → feat/spec-04-polls | fix/date-bug | refactor/split-data | chore/ci-cleanup
    → Fork off dev, NOT main (see AGENTS.md §2)

  Labels (apply before merging):
    type:*      → feat, fix, chore, test, docs, refactor, data, ci, sec
    spec:*      → spec:XX  (matching the spec being implemented)
    priority:*  → critical | high | medium | low
    status:*    → ready | in-review
  ============================================================================

  ─── Single-issue PRs ───
  Delete the "## Issue #XX — ..." blocks below. Move their content directly
  into the top-level ## Problem / ## Solution / ## Tests Added sections.

  ─── Multi-issue PRs ───
  Fill out one "## Issue #XX — ..." block per issue below. Each block has its
  own Problem / Solution / Tests Added / Completion Verification subsections.
  ============================================================================
-->

## Summary

- **#XX** — brief description of what this PR does
<!-- add more bullets for multi-issue PRs -->

Closes #XX

---

## Problem

<!-- What is wrong, missing, or needed? Why is this important? For multi-issue
     PRs, move per-issue details into the ## Issue #XX blocks below. -->

...

## Solution

<!-- What approach did you take? Key design decisions and trade-offs. -->

...

## Tests Added

<!-- New or modified tests. Delete if no test changes. -->

- `test_...` — what it verifies

---

<!--
  ════════════════════════════════════════════════════════════════════════════
  MULTI-ISSUE SECTION: repeat the block below for each issue in this PR.
  Delete this entire section for single-issue PRs.
  ════════════════════════════════════════════════════════════════════════════
-->

## Issue #XX — Title

### Problem

<!-- Issue-specific problem description. -->

...

### Solution

<!-- Issue-specific solution and rationale. -->

...

### Tests Added

<!-- Issue-specific tests. Delete if no test changes. -->

- `test_...` — what it verifies

### Completion Verification

<!-- Manual checks to confirm this specific issue is resolved. -->

- [ ] ...

---

## Changes

| File | Change |
|------|--------|
| `src/co_president/...` | description |
| `tests/...` | description |

## Verification

- [ ] `make check` exits 0
- [ ] All tests pass
- [ ] ...
