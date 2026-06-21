# Geographic Reference Data

## DIVIPOLA Master Registry

**Provenance**: DANE / datos.gov.co. Socrata API dataset `mv2e-prx5`. [https://www.datos.gov.co/](https://www.datos.gov.co/)

**Local files**: `data/fundamentals/divipola_master.csv` (our processed output). Not tracked in git (derived from public API).

**Key findings**:
- 1,122 municipalities in Colombia (pre-Bogotá-disaggregation)
- 1,143 rows with Bogotá localidad disaggregation (21 localidades replacing Bogotá D.C.)
- Each municipality has: 5-digit DIVIPOLA code (zero-padded), name, department, region
- DIVIPOLA code is the primary join key across ALL datasets

**Region classification** (5 regions):
| Region | Description | Municipalities |
|--------|-------------|---------------|
| Caribe | Caribbean coast | Atlántico, Bolívar, Cesar, Córdoba, La Guajira, Magdalena, Sucre |
| Pacífica | Pacific coast | Cauca, Chocó, Nariño, Valle del Cauca |
| Andina | Andes mountains | Antioquia, Boyacá, Caldas, Cundinamarca, Huila, etc. |
| Orinoquía | Eastern plains | Arauca, Casanare, Meta, Vichada |
| Amazonía | Amazon rainforest | Amazonas, Caquetá, Guainía, Guaviare, Putumayo, Vaupés |

**Features extracted**: `codigo_municipio` (join key), `departamento` (33 departments), `region` (5 regions).

## Bogotá Localidad Disaggregation

**Provenance**: DANE DIVIPOLA — 20 official localidades + 1 catch-all (SIN COMUNA).

**Key findings**: Bogotá D.C. (code 11001) is disaggregated into 21 localidad codes (01-20 + 99). This increases the effective feature matrix from 1,122 to 1,143 rows. Localidades have different demographics (e.g., Chapinero high-income, Usme low-income) that are lost when Bogotá is treated as a single entity.

**Feature extracted**: `comuna_nombre` (localidad name).

---

## References

All geographic features documented in `docs/architecture/features.md` §4. DIVIPOLA join key is the primary index for `MunicipalFeatures` dataclass.
