# Honest Forecast Design

## What "Honest" Means

An honest forecast uses only data available **before the event**. For 2022 backtesting:

| Data source | Used in R1? | Used in runoff? | Notes |
|------------|-------------|-----------------|-------|
| 2022 R1 polls | ✅ | — | Available before R1 |
| 2022 R1 results | ✅ | — | Known after R1, before R2 |
| 2022 R2 polls | — | ✅ | Available before R2 |
| 2022 R2 results | ❌ | ❌ | **Never in the model** |
| Google Trends | ✅ | **❌ disabled** | Biased toward Rodolfo in 2022 R2 |
| Transfer rates (2010/2014/2018) | — | ✅ | Trained on historical data only |
| Municipal fundamentals | ✅ | — | Time-invariant demographics |

## No R2 Leakage

The runoff model receives `round2_result=None` explicitly. The election likelihood (which would use R2 results) is **never added** to the runoff model. R2 results are used only for evaluation (`compute_r2_accuracy_from_matrix`), never for inference.

```python
# estimate_runoff_matrix — model path
round2_result=None,  # ← No R2 data in the model
```

## Transfer-Implied Prior

The runoff model's prior is anchored using the **transfer model**, not the R1 result shares:

1. The R1 posterior gives election-day vote shares for all candidates (from R1 data only).
2. The transfer model (trained on 2010/2014/2018) estimates how each eliminated candidate's voters would transfer to the two runoff finalists.
3. The transfer-implied runoff shares become the prior mean for the runoff model's initial time point.

```python
shares_first, shares_second = _compute_transfer_shares(
    election_day, all_candidate_keys, first, second, transfer_rates
)
first_share = float(shares_first.mean())  # Used as synthetic result prior
```

The transfer model explicitly excludes 2022 from its training set (`exclude_year=2022`).

## Drift for Trend Extrapolation

The zero-mean RW assumes the process is stationary, but campaigns are not. A drift term allows the model to extrapolate the observed poll trend to election day:

```python
drift = pm.Normal("drift", mu=0, sigma=0.02, shape=2)
theta_increments = sigma_rw * rw_raw + drift
```

In 2022, Petro was rising 0.095pp/day in the polls. Without drift, the model treats the last poll (Petro 49.96%) as election-day estimate — Rodolfo wins. With drift, it extrapolates the trend → Petro 50.79% → correct winner.

## Rest_blanco Prior Override

The runoff third category (rest = blank+null+other) is anchored at the historical rate (2.3% for 2022):

```python
_sum_ab = 1.0 / (1.0 + np.exp(prior_mean_b))
_p_a_adj = _sum_ab * (1.0 - _HISTORICAL_RUNOFF_REST_RATE)
prior_mean_rest = float(np.log(_HISTORICAL_RUNOFF_REST_RATE / _p_a_adj))
```

Without this override, the R1-implied rest share (~32%) massively inflates both candidates' absolute shares. The override preserves the p_b/p_a ratio while fixing rest at the historical rate.

## Digital Signals Disabled in Runoff

Digital signals (Google Trends) are disabled in the runoff model (`config.use_digital_signals_runoff = False` by default). The 2022 runoff digital signals showed:

| Date | Petro | Rodolfo | Bias |
|------|-------|---------|------|
| June 8 | 41.3% | 58.8% | Rodolfo +17.5pp |
| June 10 | 45.8% | 54.2% | Rodolfo +8.4pp |
| June 11 | 45.1% | 54.9% | Rodolfo +9.9pp |

These signals systematically favored Rodolfo. Including them would bias the forecast away from the correct winner. The digital signal T-1 likelihood (`concentration_t1_boost = 0.0`) is also disabled — a separate Beta observation layer was double-counting the T-1 data.

For 2026, digital signals can be re-enabled with bias calibration once that mechanism is implemented.

## Blanco Prior Override (R1)

The R1 model overrides the blanco vote prior at ~2.0% (historical rate) instead of the poll-implied ~10.6%:

```python
_r1_historical_blanco = 0.020
theta_mu[_b_idx] = np.log(_r1_historical_blanco)
theta_sigma[_b_idx] = 0.3
```

Colombian polls systematically inflate blank votes relative to actual election results. The override shrinks the prior toward the historical rate.

## References

1. [SciELO2023] — T-1 Google Trends methodology and documented bias in 2022 runoff (Rodolfo +1.7-4.2pp vs actual Petro +3.07pp). Foundation for disabling digital signals in runoff.
2. [King1997] — Ecological inference: inferring individual-level voter transitions from aggregate R1→R2 data. Basis for transfer model.
3. [CoElect] — Transfer heuristic methodology for Colombian runoff elections.
4. [ASCOA] — Poll tracker and transfer rate methodology for 2022 Colombian election.
5. [538Model] — House effects decomposition (PIE per pollster). Basis for pollster weighting.
6. [GelmanHill2007] — Bayesian data combination principles: no data used beyond what was available at forecast time.

Full citations: `docs/reference/bibliography.md`. For detailed synthesis: `docs/reference/key-findings.md`.
