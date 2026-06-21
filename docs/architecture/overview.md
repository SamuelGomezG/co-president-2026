# Architecture Overview

## Pipeline

```
encuestas_2022.csv ─── polls ──┬─── load_and_clean_all() ─── CleanPolls
                                │         │
registraduria MMV ──┬── results ├─── load_canonical_results() ── RoundResult
moe csv ────────────┘           │
                                │
google trends ──────── trends ──┴─── ingest_trends() ── digital_signals
```

## Model Layers

The model predicts the 2022 Colombian presidential election using **three model stages**, connected via their posterior distributions:

### Stage 1: First-Round Model (`model_round1.py`)

**Purpose**: Predict all 6 candidate vote shares in the first round.

**Observations**: 30 national polls + Google Trends digital signals

**Structure**:
- K=6 candidates (Petro, Hernandez, Gutierrez, Fajardo, Betancourt, rest+blanco)
- Non-centered reverse-time random walk on logit scale
- Hierarchical pollster house effects (zero-sum constrained)
- Municipal structural prior (informed by demographics + historical results)
- Election-day anchor via fixed-concentration DirichletMultinomial likelihood

**Output**: Posterior over `p_time` (vote shares at each time point), used as prior for the runoff model.

### Stage 2: Transfer Model (`model_transfer.py`)

**Purpose**: Estimate how eliminated candidates' voters transfer to runoff candidates.

**Data**: Historical R1→R2 from 2010, 2014, 2018 + municipal demographics + legislative vote shares.

**Structure**: Bayesian ecological inference model, estimated per-candidate transfer rates.

**Output**: Per-draw transfer rates used to compute the runoff prior.

### Stage 3: Runoff Model (`model_runoff_simple.py`)

**Purpose**: Predict the head-to-head runoff between the top two candidates.

**Observations**: 19 runoff polls (no digital signals — they were biased in 2022).

**Structure**:
- K=3 (Candidate A, Candidate B, rest+blanco)
- Non-centered reverse-time random walk with **drift** (extrapolates poll trends)
- Transfer-implied prior from R1 posterior (no R2 data)
- Rest_blanco prior override at historical ~2.3% rate
- Hierarchical pollster house effects

**Output**: Win probability, vote share distribution, margin distribution.

## Data Flow

```
R1 polls ──► R1 model ──► R1 posterior (p_time)
                                      │
                                      ▼
                              Transfer model
                                      │
      ┌────────────────────────────────┴────────────┐
      ▼                                             ▼
Runoff prior (transfer-implied)            Runoff polls
      │                                             │
      └─────────────────┬───────────────────────────┘
                        ▼
                  Runoff model (K=3)
                  (RW with drift + rest override)
                        │
                        ▼
               RunoffForecast (P(winner), margin, CI)
```

## Key Design Principles

1. **Honesty**: No R2 election data enters any model. R2 is used only for evaluation.
2. **Compositional data**: Vote shares sum to 100% — Dirichlet-Multinomial is the natural likelihood.
3. **Non-centered parameterization**: Breaks posterior correlation in the random walk, enabling R-hat < 1.01.
4. **Independent likelihoods**: Polls, digital signals, and election result each have their own likelihood with separate concentration parameters.
5. **Transfer prior for runoff**: Uses R1 posterior + historical transfer patterns, not R2 data.
6. **Drift for trend extrapolation**: Standard time-series technique. Captures non-stationary campaign dynamics.

## References

1. [Aitchison1982] — Compositional data analysis foundation (Dirichlet-Multinomial, simplex geometry).
2. [SciELO2023] — Google Trends T-1 methodology: 1.86pp error, T-1 peak decay.
3. [USANTOMAS2025] — Municipal outperforms national: FNN+CLR R² 0.94 for ideological vote share.
4. [CoElect] — Ajiaco model: reverse-time RW, Dirichlet-Multinomial for Colombian elections.
5. [Betancourt2017] — Non-centered parameterization for hierarchical models (enabled R-hat < 1.01).
6. [GelmanHill2007] — Hierarchical Bayesian data combination; principles for honest forecasting.

Full citations: `docs/reference/bibliography.md`. For detailed synthesis: `docs/reference/key-findings.md`. For decision-level mapping: `docs/reference/adopted.md`. For file-level index: `docs/reference/index.md`.
