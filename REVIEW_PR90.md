# REVIEW_PR90.md — Code Review: PR #90

**Reviewer**: Senior Dev (AI Agent)  
**Reviewee**: Junior Dev (Author)  
**PR**: `#90` — `docs(SPEC-07): document GAD3 runoff polling gap in Appendix 16.4`  
**Commits**: `04d55c8`, `ca7a29f`  
**Branch**: `feat/track-c-docs`  
**Date**: 2026-05-22

> **Agent Note**: The review agent initially claimed the PR changes were "missing from HEAD." This was incorrect. Manual verification confirmed the changes are present in `HEAD` on both `MVP_SPECS_GUIDE.md` and `src/co_president/data_polls.py`. The review below preserves all other findings, which remain valid.

---

## 1. MVP_SPECS_GUIDE.md — Appendix 16.4

### 1.1 Data Accuracy — Wave 10 and Wave 11 are Identical

In your table:

| Wave | Date Range | Hernández | Petro | Blanco | Sample |
|---|---|---|---|---|---|
| 10 | May 30–Jun 10 | 47.9% | 47.1% | 5.0% | 5,236 |
| 11 | May 30–Jun 11 | 47.9% | 47.1% | 5.0% | 5,236 |

Wave 10 and Wave 11 have **identical** percentages and the **identical** sample size. That is either a copy-paste error, a data transcription error, or the underlying source is garbage. A daily tracking poll with the exact same numbers two days in a row and no additional sample is suspicious. You need to either:
- Explain why they are identical (e.g., "Wave 11 is a recapitulation of Wave 10 published by GAD3 on June 11"), or
- Correct the values if they are wrong.

Do not dump unverified data into our authoritative design document. The SPEC guide is the source of truth.

### 1.2 Date Inconsistency — "May 31" vs "May 30"

Your prose says:
> "The 10 'missing' GAD3 tracking poll waves (May 31 – June 10) were found..."

Your table says:
> "May 30–31", "May 30–Jun 1", etc.

Which is it? May 30 or May 31? A one-day shift matters for time-series alignment and model training. Pick one, justify it with a source citation, and fix the other.

### 1.3 Source Quality — Wikipedia is NOT a Primary Source

You wrote:
> "found documented on the Spanish Wikipedia page [...] with RCN Radio as the primary source."

Then two lines later:
> "Verification: 7 of 10 waves were cross-checked against Wayback Machine archives of the original RCN Radio articles."

If RCN Radio is the primary source, why did you cite Wikipedia at all? Wikipedia is a **secondary** source. You should lead with the primary source (RCN Radio / Wayback Machine) and mention Wikipedia only as a discovery aid. More importantly, **which 3 waves were NOT cross-checked?** This is a glaring omission. If you are going to claim verification, you verify all of them or you explicitly list the exceptions and explain why. "7 of 10" reads like you got lazy.

### 1.4 Stale Row Reference — "Row 41"

> "Wave 11 (June 11) was already present as row 41."

Row numbers are not stable. If waves 1–10 were inserted into `encuestas_2022.csv`, Wave 11 is no longer row 41. This is brittle documentation that will rot immediately. Remove the row number and use a unique identifier instead (e.g., `encuestadora == "GAD3" and fecha == "2022-06-11"`).

### 1.5 Recommendation is Self-Contradictory

> "Status: Waves 1–10 have been added to `encuestas_2022.csv`."
> "Recommendation: Before SPEC-07 (runoff model), ensure `encuestas_2022.csv` includes these GAD3 waves."

If they have already been added, the recommendation is pointless. Rewrite the recommendation to something actionable, e.g.:
> "Recommendation: Validate that the GAD3 waves load correctly through `load_and_clean_all()` and that `CleanPolls.round2` contains at least `N` GAD3 rows with `blanco` shares in [2.6, 5.8]."

### 1.6 Missing `otros` Column

Your table shows Hernández, Petro, Blanco, and Sample. It does **not** show `otros`. If `otros = 0` for all these polls, you must state that explicitly. If `otros` is missing from the source, you must state that too. The K=3 model (Petro, Hernández, Blanco) only works if `otros` is zero or folded into one of the three categories. Do not leave ambiguity in the spec.

### 1.7 Table Formatting

The table header has 6 columns and 6 separators — that part is correct. But "Sample" is vague. Use "Sample size" or `muestra` to match the terminology used everywhere else in the document.

---

## 2. src/co_president/data_polls.py — Module Docstring

### 2.1 Inappropriate Location for a Module-Level NOTE

A module docstring is the wrong place for this. AGENTS.md §5:

> "Google-style docstrings on every public function, class, and module. Module docstrings start with `"""SPEC-XX: ..."""`."

A module docstring should describe the module's API and purpose. Operational notes about data provenance belong in:
- `data/README.md`
- `docs/DATA_PROVENANCE.md`
- A `# TODO` or `# NOTE` comment directly above `load_raw_polls()` where the CSV is actually read

If you must keep it in the docstring, it belongs in `load_raw_polls()`'s docstring under a "Notes" section, not at module scope.

### 2.2 Same Factual Errors as the SPEC Guide

> "The 10 GAD3 tracking poll waves (May 31 - Jun 10) were recovered from Wikipedia and RCN Radio primary sources."

Same problems:
- "May 31" contradicts your own table ("May 30").
- Wikipedia is not a primary source. Say "recovered from RCN Radio primary sources (discovered via Wikipedia)" or simply "recovered from RCN Radio archives."

### 2.3 Unclear Scope

The NOTE says:
> "See Issue #88 and MVP_SPECS_GUIDE.md Appendix 16.4."

Issue references in code rot quickly. If Issue #88 is closed, this line becomes a dead pointer. The SPEC guide reference is acceptable because it is version-controlled, but the Issue reference should be removed or qualified (e.g., "GitHub Issue #88 (closed)").

---

## 3. Alignment with Issue #88

Issue #88's purpose, based on the commit message and PR title, is to **document** the GAD3 runoff polling gap. The first commit (`04d55c8`) correctly documented the gap as a missing-data problem. The second commit (`ca7a29f`) pivoted to "the polls are now recovered." That is fine if the recovery is real, but:

1. The recovery claims are under-verified (only 7/10 cross-checked).
2. There is no automated test asserting that the recovered polls are actually in the CSV.

A documentation issue that turns into a data-recovery claim needs a corresponding test. Add a test to `tests/test_data.py`:

```python
def test_gad3_runoff_polls_present(data_dir: Path) -> None:
    cp = load_and_clean_all(data_dir=data_dir)
    gad3_r2 = cp.round2[cp.round2["encuestadora"].str.strip() == "GAD3"]
    assert len(gad3_r2) >= 10, f"Expected >=10 GAD3 R2 polls, got {len(gad3_r2)}"
```

Without this test, your "Waves 1–10 have been added" claim is unverified and will regress silently.

---

## 4. Grammar, Spelling, and Style

### 4.1 Spelling

- "cross-checked" vs "cross checked" — pick one. The spec guide uses "cross-checked" (correct).
- "renormalizes" vs "renormalises" — British vs American. The codebase uses both, but you should not introduce new inconsistency. (This is a pre-existing issue; do not make it worse.)

### 4.2 Punctuation

- "May 31 - Jun 10" should use an en-dash (`–`) and no spaces, or spaced hyphens. The SPEC guide table uses en-dashes. Be consistent.
- "primary sources." — the period inside the quotes is American style. The rest of the codebase uses logical punctuation. Pick one.

### 4.3 Line Length

The Wikipedia URL in `MVP_SPECS_GUIDE.md` is extremely long:
```
https://es.wikipedia.org/wiki/Anexo:Sondeos_de_intenci%C3%B3n_de_voto_para_las_elecciones_presidenciales_de_Colombia_de_2022
```

While markdown does not strictly enforce line length, AGENTS.md specifies **100 characters** for code. For markdown, we should still wrap long lines when possible. Use a reference-style link:

```markdown
[Spanish Wikipedia page][wiki-sondeos] with RCN Radio as the primary source.

[wiki-sondeos]: https://es.wikipedia.org/wiki/Anexo:Sondeos_de_intenci%C3%B3n_de_voto_para_las_elecciones_presidenciales_de_Colombia_de_2022
```

---

## 5. Action Plan

### Phase 1 — Fix the Data Accuracy Issues

1. **Clarify Wave 10 vs Wave 11**: Explain the identical numbers or correct them.
2. **Fix dates**: Align "May 31" in prose with "May 30" in table, or vice versa, with a source citation.
3. **Complete verification**: Cross-check the remaining 3 waves, or explicitly list which ones were not verified and why.
4. **Add `otros` column or explicit note** in the table or prose.
5. **Remove stale row number** ("row 41").

### Phase 2 — Fix the Documentation Structure

6. **Move the module-level NOTE** from `data_polls.py` docstring to either:
   - A `data/README.md` file, or
   - A comment block above `load_raw_polls()`, or
   - `load_raw_polls()`'s docstring under a "Notes" section.
7. **Fix source description**: "RCN Radio primary sources (discovered via Wikipedia)". Remove "Wikipedia" as a primary source.
8. **Fix recommendation** in Appendix 16.4 to be actionable and not self-contradictory.
9. **Use reference-style links** for long URLs in markdown.

### Phase 3 — Add a Regression Test

10. **Add `test_gad3_runoff_polls_present`** to `tests/test_data.py` to ensure the recovered polls are actually loaded by the pipeline.

### Phase 4 — Run Quality Gates

11. `make check` (fmt, lint, typecheck, test).
12. Re-read both files out loud. If a sentence sounds like AI slop, rewrite it.

---

## 6. Conclusion

This PR contains data accuracy issues (identical waves, unverified sources), structural problems (module-level NOTE in a docstring), and missing regression tests. The SPEC guide is a contract, not a scratchpad. Fix the data, fix the structure, add the test, and come back.

**Status**: 🔴 **REQUEST CHANGES**
