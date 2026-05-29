# Colombia Polling Data Patterns

Project-specific data handling for Colombian election polling data.

## File Encoding

**Always use `latin-1` (ISO-8859-1) encoding** when reading MMV and consultas files:

```python
df = pd.read_csv(path, encoding="latin-1", low_memory=False)
```

Failing to specify encoding will cause UnicodeDecodeError on these files.

---

## Known Data Anomalies

### 1. Invamer Date Error

**Problem**: Invamer poll dated `2022-04-19` should be `2022-05-19` (data-entry typo).

**Detection and fix**:

```python
def fix_invamer_date(df: pd.DataFrame) -> pd.DataFrame:
    """Correct the known Invamer data-entry error."""
    result = df.copy()
    mask = (
        (result["encuestadora"].str.strip() == "Invamer") &
        (result["fecha"] == pd.Timestamp("2022-04-19"))
    )
    result.loc[mask, "fecha"] = pd.Timestamp("2022-05-19")
    return result
```

**Test**: `TestFixInvamerDate` in `tests/test_data_polls.py`.

---

### 2. MassiveCaller Forced-Choice Format

**Problem**: Round 2 polls from MassiveCaller use IVR forced-choice format — respondents forced to pick A or B, no `rest+blanco`. These polls should be excluded from Round 2 aggregation.

**Detection**: Pollster name contains `MassiveCaller` in Round 2 context.

**Fix**: Filter out before modeling:

```python
df_r2 = df_r2[df_r2["encuestadora"].str.strip() != "MassiveCaller"]
```

**Test**: `TestMassiveCallerR2` in `tests/integration/test_data_polls_integration.py`.

---

### 3. Centro Esperanza Split

**Problem**: Fajardo and Betancourt tracked separately in polls. In official results, Centro Esperanza = Fajardo.

**Fix**: Map Betancourt → Fajardo in results after official results are loaded:

```python
result.loc[result["coalition"] == "Centro Esperanza", "candidate"] = "Fajardo"
```

---

### 4. GAD3 Runoff Tracking Waves (Otros = NA)

**Problem**: GAD3 Round 2 has 11 tracking waves. `otros` is not reported — raw value is NA, but it means 0 in K=3 model.

**Fix**: Treat `NA` as 0 for `otros` column in GAD3 runoff data:

```python
df["otros"] = df["otros"].fillna(0)
```

---

### 5. Mosqueteros Massive Sample

**Problem**: `muestra=6000`, `muestra_int_voto=NA` — sample size is 6000 but intermediate vote intention is NA.

**Fix**: Fall back to `muestra` when `muestra_int_voto` is NA:

```python
df["muestra_int_voto"] = df["muestra_int_voto"].fillna(df["muestra"])
```

---

## Common Column Transformations

### Fecha Parsing

```python
df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
```

### Numeric Conversion (Latin American decimal comma)

```python
df["int_voto"] = pd.to_numeric(
    df["int_voto"].astype(str).str.replace(",", ".", regex=False),
    errors="coerce",
)
```

### Muestra (Sample Size)

```python
df["muestra"] = pd.to_numeric(df["muestra"], errors="coerce").astype("Int64")
```

---

## Poll Schema

Required columns for poll DataFrames:

| Column | Type | Description |
|--------|------|-------------|
| `fecha` | datetime | Poll date |
| `encuestadora` | str | Pollster name |
| `muestra` | int | Sample size |
| `<candidate_key>` | float | Vote share (%) per candidate |
| `otros` | float | Other/abstain share (may be NA) |

---

## Polling Sources

| Pollster | Rating | Notes |
|----------|--------|-------|
| Invamer | 10.0 | Highest quality; dates corrected |
| GAD3 | 8.10 | Round 1 + 11 tracking waves Round 2 |
| MassiveCaller | 5.10 | Excluded from Round 2 (forced-choice) |
| Mosqueteros | 1.00 | Large sample fallback |
| Guarumo | — | — |
| CELAG | — | — |
| CNC | — | — |

Pollster ratings used for weighted aggregation in `aggregation.py`.
