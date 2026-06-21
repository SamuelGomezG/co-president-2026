# architecture/

**Purpose**: Architectural rationale for the co-president-2026 model. Explains why each design decision was made, with citations to academic literature.
**Category**: docs

## Contents

| Name | Description |
|------|-------------|
| `overview.md` | Pipeline diagram, model layers, data flow, key design principles |
| `model-design.md` | Why Dirichlet-Multinomial, reverse-time RW, K=3, drift, non-centered |
| `honest-forecast.md` | No R2 leakage, transfer prior, drift, rest/blanco override |
| `features.md` | Complete feature catalog by module |
| `sampling-strategy.md` | numpyro, non-centered RW, φ_elec divergence fix, diagnostics |
| `packages.md` | Package organization rationale, test structure |

## Cross-References

- `docs/reference/bibliography.md` for full citations.
- `docs/reference/key-findings.md` for literature synthesis.
- `docs/reference/adopted.md` for decision-level mapping to papers.
