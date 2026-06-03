# AS/COA Poll Tracker — 2022 Colombia Presidential Election

Data extracted from the **AS/COA (Americas Society / Council of the Americas)** poll
trackers for the 2022 Colombian presidential election. AS/COA curated and
published a regularly-updated visualization of polling aggregates ahead of both
rounds.

## Sources

| Round | Article | Underlying Infogram |
|---|---|---|
| First round | https://www.as-coa.org/articles/poll-tracker-colombias-2022-presidential-election | https://infogram.com/colombia-first-round-poll-tracker-1h8n6m35qnn9z4x |
| Runoff | https://www.as-coa.org/articles/poll-tracker-colombias-2022-presidential-runoff | https://infogram.com/colombia-runoff-poll-tracker-1h7v4pw0zxxm86k |

Authors: Chase Harrison, Jon Orbach. First-round tracker first published
2022‑04‑13, last updated 2022‑05‑20. Runoff tracker last updated 2022‑06‑13.

## Extraction method

Infogram inlines a complete `window.infographicData` JSON object in the public
viewer HTML (no API auth required — we confirmed the JSON endpoints require a
session we cannot reproduce). The full payload is preserved in
`round1_raw.json` and `runoff_raw.json` (≈ 53 KB and 58 KB respectively) and
contains every chart entity, with the data sheets under
`elements.content.content.entities[uuid].data`.

`extract.py` parses those JSONs and emits the cleaned CSVs. Re-run with:

```bash
uv run python data/2022-polls/as_coa/extract.py
```

## Files

| File | Contents |
|---|---|
| `round1_raw.json` | Raw Infogram payload for the first-round tracker |
| `runoff_raw.json` | Raw Infogram payload for the runoff tracker |
| `round1.csv` | Cleaned first-round timeline (6 published aggregated values) |
| `runoff.csv` | Cleaned runoff timeline (3 published aggregated values) |
| `transfer_matrix.csv` | First-round-vote → runoff-candidate transfer (SPEC-08 input) |
| `extract.py` | Reproducible extractor |

### Schema

`round1.csv`, `runoff.csv`:

```
fecha,fuente,encuestadora,ambito,<candidato_1>,<candidato_2>,...,blanco,ninguno,ns_nr
```

- `fecha`: ISO date (YYYY-MM-DD) of the publication/aggregation date.
- `fuente`: always `AS/COA`.
- `encuestadora`: `AGREGADO` because AS/COA presents one summary number per
  time point, not per underlying pollster.
- `ambito`: `nacional` (national scope; regional cuts are available in the
  raw JSON but out of MVP scope).
- Candidate columns hold integer or float percentages.

`transfer_matrix.csv`:

```
candidato_primera_vuelta,pct_primera_vuelta,petro_runoff,hernandez_runoff,blanco_runoff,ns_nr_runoff,fuente
```

- The `pct_primera_vuelta` column carries the official first-round share.
- The four transfer columns show where those voters landed in the runoff.

## Important caveats

1. **This is curated, aggregated data.** AS/COA's tracker does not preserve
   the original pollster, sample size, margin of error, field date, or source
   URL of the underlying polls. The numbers in `round1.csv` / `runoff.csv` are
   AS/COA's *public-aggregate view* at each time point, not raw survey
   records. For raw survey records use `data/2022-polls/encuestas_2022.csv`
   (the project's master CSV, fed by SPEC-04).
2. **Date label interpretation.** "March 18" is treated as the publication
   date of an aggregated value, not a field date. The two are often close but
   not identical.
3. **Single-snapshot charts ignored for the timeline CSVs.** The raw JSON
   contains several "headline" charts (one value each, dated to the article's
   last update). They are preserved in the raw JSON and can be inspected
   manually; the timeline CSVs only include date-keyed rows.
4. **MassiveCaller / Centro Esperanza / Betancourt withdrawal** corrections
   (see `MVP_SPECS_GUIDE.md` §11) do **not** apply to this dataset because
   AS/COA already published post-correction values; the editorial article
   explicitly notes Betancourt dropped out on May 20.
5. **Out of MVP scope** (but available in raw JSON): regional cuts for round
   1, age/gender cuts, departmental map for the runoff, favorability
   statistics, scenario simulations. These could feed future extensions of
   SPEC‑06/07/08 (demographic model, geo model) but are not used by the
   current MVP.

## Cross-check procedure

After SPEC-04 ingests `encuestas_2022.csv`, run a quick diff:

```python
import pandas as pd
master = pd.read_csv("data/2022-polls/encuestas_2022.csv", parse_dates=["fecha"])
ascoa = pd.read_csv("data/2022-polls/as_coa/round1.csv", parse_dates=["fecha"])
# For each AS/COA date, take the mean of master polls whose field date is
# within ±7 days and compare. Discrepancies > 3pp warrant investigation.
```

## Reproducibility

The raw JSONs are committed (small, deterministic) so any future re-parse is
guaranteed to produce the same CSVs without re-hitting the network. If
AS/COA publishes a new update, re-run the curl commands in this commit
message history (or the script's docstring) and re-run `extract.py`.
