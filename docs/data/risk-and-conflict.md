# Risk & Conflict Data Sources

## MOE Electoral Risk Maps

**Provenance**: Misión de Observación Electoral. [https://moe.org.co/datos-electorales/mapas-de-riesgo-electoral/](https://moe.org.co/datos-electorales/mapas-de-riesgo-electoral/)

**Local files**: `data/fundamentals/risk_factors.csv` (our processed output). Raw source not tracked in git.

**Key findings**:
- 4 risk levels: extreme (49 municipalities), high (65), medium (17), low (all others)
- Extreme + high risk municipalities concentrated in: Pacific coast (Chocó, Cauca, Nariño), Catatumbo (Norte de Santander), Amazonía (Putumayo, Caquetá, Guaviare)
- These regions correlate with high Petro support and low turnout — the risk features capture the structural disadvantage in these areas

**Features extracted**: `risk_level` → `high_risk_flag` (binary: extreme+high = 1).

## PDET Municipalities

**Provenance**: Agencia de Renovación del Territorio. [https://centralpdet.renovacionterritorio.gov.co/](https://centralpdet.renovacionterritorio.gov.co/)

**Key findings**: 170 prioritized municipalities for post-conflict development. These are the municipalities most affected by armed conflict — used as a binary flag in the model.

**Feature extracted**: `is_pdet` (bool).

## INDEPAZ Armed Groups

**Provenance**: INDEPAZ. [https://indepaz.org.co/](https://indepaz.org.co/)

**Key findings**: Armed group presence in 216+ municipalities. The presence of illegal armed groups correlates with:
- Lower turnout (violence suppresses voting)
- Higher blank/null vote rates
- Lower polling accuracy (insecure areas have fewer reliable polls)

**Feature extracted**: `armed_group_presence` (bool).

## UNODC Coca Cultivation

**Provenance**: UNODC (United Nations Office on Drugs and Crime). [https://unodc.org/colombia](https://unodc.org/colombia)

**Key findings**: Coca cultivation concentrated in Putumayo, Cauca, Nariño, Norte de Santander. Heavy overlap with PDET and extreme-risk municipalities. Continuous (hectares) rather than binary.

**Feature extracted**: `coca_hectares` (float).

---

## Features Used

| Feature | Source | Model role |
|---------|--------|-----------|
| high_risk_flag | MOE Risk | β_risk in municipal model |
| is_pdet | PDET registry | β_pdet in municipal model |
| armed_group_presence | INDEPAZ | Reserved for turnout model |
| coca_hectares | UNODC | Reserved for turnout model |

**References**: Features documented in `docs/architecture/features.md` §4. Literature: `docs/reference/key-findings.md` §5 (Municipal Model) and §10 (Data Sources).
