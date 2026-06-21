# Digital Signal Data Sources

## Google Trends (pytrends API)

**Provenance**: Google Trends via `pytrends` (unofficial API, no authentication required). [https://github.com/GeneralMills/pytrends](https://github.com/GeneralMills/pytrends)

**Local files**: `results/trends_cache_2022.parquet` (our cached output, regeneratable via `ingest_trends.py`). Not tracked in git.

**2022 Query Maps** (`trends_keywords.py:17-21`):
| Candidate | 2022 query | Type |
|-----------|-----------|------|
| Gustavo Petro | `"Petro"` | Single-name per SciELO |
| Rodolfo Hernández | `"Rodolfo"` | Single-name |
| Federico Gutiérrez | `"Fico"` | Nickname |
| Sergio Fajardo | `"Fajardo"` | Single-name |

**2026 Query Maps** (planned, different candidates — `trends_keywords.py` will be extended):
To be defined when 2026 candidates are finalized. Will follow SciELO single-name convention.

**Key findings** (from SciELO 2023 paper, Pérez-Rave et al.):
- **T-1 is best time point**: 1.86pp error on T-1 (June 18, 2022) vs 6.56pp on election day (June 19)
- **Party-based queries outperform name queries**: "Petro" beats "Gustavo Petro"
- **Single-name is most frequent**: short queries dominate search volume
- **Google+YouTube > Google alone > YouTube alone**: cross-platform cancellation of platform-specific bias
- **Documented bias in 2022 runoff**: T-1 signal showed Rodolfo 50.85% vs actual 50.42% Petro. Estimated margin error ~4.8pp in wrong direction

**Prop-Fav computation** (`ingest_trends.py:prop_fav`):
```python
prop_fav(A, t) = interest(A, t) / sum(interest(all), t)
```
Normalized by total interest across all tracked candidates for that day.

**Decay function** (model application):
```python
decay = exp(-abs(days_from_elec - 1.0) / 3.0)
phi_digital[t] = phi_digital_base * decay * internet_rate * concentration_t1_boost
```

**Features extracted**:
| Feature | Computation | Model role |
|---------|-------------|-----------|
| prop_fav per candidate | `interest(A)/sum(interest)` | Beta observation likelihood |
| phi_digital_base | Gamma(5, 5/100) | Learned digital signal concentration |
| phi_digital[t] | Base × decay × internet rate | Time-varying observation precision |
| concentration_t1_boost | Config param (default 0.0) | T-1 boost factor (disabled for 2022 runoff) |

**Usage per model**:
| Model | Digital signals? | Reason |
|-------|-----------------|--------|
| R1 (backtest) | ✅ Enabled | Good signal quality for 2022 R1 |
| R1 (forward) | ✅ Enabled | Digital signal available before R1 |
| Runoff | ❌ Disabled (default) | Systematic Rodolfo bias in 2022. Re-enabled via `config.use_digital_signals_runoff = True` |

**Internet rate weighting** (`_compute_national_internet_rate()`):
Population-weighted average of municipal internet access rates from CNPV 2018. For 2022: ~80.79% with Option C soft deflation. Digital signal concentration is multiplied by this rate to account for the offline population.

**References**: Full discussion in `docs/reference/key-findings.md` §3 (Google Trends Digital Signals). SciELO paper cited as `[SciELO2023]` in `docs/reference/bibliography.md`.
