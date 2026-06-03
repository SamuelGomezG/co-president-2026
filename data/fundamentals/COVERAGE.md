# Fundamentals Coverage

Generated: 2026-06-03 01:48 UTC

| Source | File | Rows Expected | Rows Actual | Status | Notes |
|---|---|---|---|---|---|
| DIVIPOLA | divipola_master.csv | 1122 | 1123 | ✅ FULL | 1123 unique municipality codes |
| Historical results | historical_results.csv | 13464 | 25708 | ✅ FULL | 6 election years; CONTAINS 000NA PLACEHOLDER |
| Socioeconomic | socioeconomic.csv | 1122 | 3 | ⚠️ STUB | Hardcoded fallback (3 rows only) |
| Risk | risk_factors.csv | 1122 | 10 | ⚠️ STUB | Hardcoded fallback (10 rows only) |
| CEDAE (raw) | data/raw/cedae/ | 18 files (2002-2018) | 18 files | ✅ READY | Legislative + presidential; primary source for historical results |
| MOE 2022 (legislative) | data/raw/MOE-2022-legislativas/ | Camara + Senado | 0 files | ❌ MISSING | Camara 13K rows, Senado 20K rows |
| DANE NBI (2018) | data/raw/DANE-NBI/ | Complete municipal NBI | 0 files | ❌ MISSING | Primary poverty feature (0% imputation needed) |

---

## Legend

| Status | Meaning |
|---|---|
| ✅ FULL | Complete coverage (>= expected rows) |
| ⚠️ PARTIAL | Partial coverage (< expected rows) |
| ⚠️ STUB | Placeholder data (< threshold rows) |
| ✅ READY | Files present on disk, not yet ingested |
| ❌ MISSING | Source not found |
| ⚠️ PLACEHOLDER | Contains 000NA sentinel rows (in notes) |

---

## Legislative Schema Compatibility

- **CEDAE Cámara columns**: ``['id_electoral', 'ano', 'tipo_eleccion', 'fecha_eleccion', 'coddpto', 'departamento', 'codmpio', 'municipio', 'circunscripcion', 'codigo_partido', 'codigo_lista', 'en_lista', 'primer_apellido', 'segundo_apellido', 'nombres', 'votos', 'curules']``
- **MOE Cámara columns**: ``[]``
- **CEDAE Senado columns**: ``['ano', 'tipo_eleccion', 'fecha_eleccion', 'coddpto', 'departamento', 'codmpio', 'municipio', 'circunscripcion', 'codigo_partido', 'codigo_lista', 'primer_apellido', 'segundo_apellido', 'nombres', 'votos', 'curules']``
- **MOE Senado columns**: ``[]``
- **Schemas compatible**: No (see notes)
- **Notes**: Could not read one or both Cámara data sources
