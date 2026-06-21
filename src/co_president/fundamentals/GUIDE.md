# fundamentals/

**Purpose**: Municipal features API — loading, validation, CLR/logit transforms.
**Category**: source

## Contents

| Name | Type | Description | SPEC |
|------|------|-------------|------|
| `__init__.py` | module | Re-exports `load_features`, `MunicipalFeatures` | SPEC-21 |
| `features.py` | module | `MunicipalFeatures` dataclass, `load_features()`, CLR transforms | SPEC-21 |
| `compositional.py` | module | `_apply_zero_floor`, `impute_zero_shares` for compositional data | — |

## Cross-References

- `data/fundamentals/` for the output CSV files.
- `docs/architecture/features.md` §4 for feature documentation.
- `model_municipal.py` for the model that consumes these features.
