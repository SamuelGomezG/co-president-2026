# 2026 Candidate Configuration

## Election Timeline

| Event | Date |
|-------|------|
| Inter-party consultations | March 8, 2026 |
| First Round | **May 31, 2026** |
| Runoff | **June 21, 2026** |

## Differences from 2022

| Aspect | 2022 | 2026 |
|--------|------|------|
| First Round date | May 29 | May 31 |
| Runoff date | June 19 | June 21 |
| Candidates | 7 + rest + blanco | To be determined |
| Poll source | recetas-electorales (46 rows) | CNE microdata (23 ZIPs, 9 firms) |
| Poll format | Single CSV | Mixed CSV/Excel/SPSS/PDF |
| Consultation votes | 5 coalitions, known values | To be determined |
| Pollster ratings | 13 rating values (0.5-10.0) | New ratings: Atlas 5.9, CNC 5.8, GAD3 6.2, etc. |

## Pollster Ratings (2026, from La Silla Vacía)

| Pollster | Rating (0-10) |
|----------|--------------|
| Atlas Intel | 5.9 |
| CNC | 5.8 |
| GAD3 | 6.2 |
| Guarumo Ecoanalítica | 5.1 |
| Invamer | 5.3 |
| Génesis Crea | 2.0 |
| TEMPO Consultoría | 2.0 |
| Corporación MMM | 2.0 |

These are stored separately from the 2022 `POLLSTER_RATINGS` in the planned `config.py` extension to prevent test breakage.

## Key Differences for the Model

1. **Transfer model**: 2026 candidates need mapping to historical coalitions. The candidate-agnostic feature space (demographics, legislative shares) remains the same — only the candidate labels change.

2. **Drift parameter**: Works the same way — needs runoff polls to estimate the trend. With fewer polls than 2022's 19, the drift prior sigma may need tightening.

3. **Digital signals**: Should be re-enabled for 2026 (no known candidate-specific bias yet). The `concentration_t1_boost` and `use_digital_signals_runoff` config flags provide control.

4. **Consultation prior**: Requires updated `CONSULTATION_VOTES` once the 2026 inter-party consultation results are known.

5. **Validation target**: La Silla Vacía's `Prediccion_30d` (June 21 snapshot) provides a published forecast to compare against.

**References**: `docs/PLAN_cne_2026_ingestion.md` for the full integration plan. `docs/reference/key-findings.md` §7.3 for La Silla Vacía methodology.
