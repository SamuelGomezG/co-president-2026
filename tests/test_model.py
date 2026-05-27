"""Tests for the first-round Bayesian model (SPEC-06)."""

import numpy as np
import pandas as pd
import pymc as pm  # type: ignore[reportMissingTypeStubs]

from co_president.config import ModelConfig
from co_president.data_results import CandidateResult, RoundResult
from co_president.model_round1 import build_round1_model


def _make_3row_polls_7candidates() -> pd.DataFrame:
    """Return a 3-row synthetic DataFrame with all 7 candidate columns."""
    return pd.DataFrame(
        {
            "fecha": ["2022-05-29", "2022-05-29", "2022-05-29"],
            "encuestadora": ["PollsterA", "PollsterB", "PollsterC"],
            "muestra": [1000, 1000, 1000],
            "gustavo_petro": [50.0, 51.0, 49.0],
            "federico_gutierrez": [10.0, 10.0, 11.0],
            "rodolfo_hernandez": [20.0, 20.0, 21.0],
            "sergio_fajardo": [10.0, 10.0, 9.0],
            "ingrid_betancourt": [2.0, 2.0, 3.0],
            "rest": [5.0, 5.0, 5.0],
            "blanco": [3.0, 2.0, 2.0],
            "round_number": [1, 1, 1],
        }
    )


def _make_3row_polls_3candidates() -> pd.DataFrame:
    """Return a 3-row synthetic DataFrame with 3 candidate columns."""
    return pd.DataFrame(
        {
            "fecha": ["2022-05-29", "2022-05-01", "2022-05-01"],
            "encuestadora": ["PollsterA", "PollsterB", "PollsterC"],
            "muestra": [1000, 1000, 1000],
            "gustavo_petro": [50.0, 48.0, 52.0],
            "rodolfo_hernandez": [40.0, 42.0, 38.0],
            "blanco": [10.0, 10.0, 10.0],
            "round_number": [1, 1, 1],
        }
    )


def _make_round1_result() -> RoundResult:
    """Return a minimal RoundResult for backtest testing."""
    candidates = (
        CandidateResult("gustavo_petro", 8_000_000, 0.40),
        CandidateResult("rodolfo_hernandez", 5_600_000, 0.28),
        CandidateResult("federico_gutierrez", 4_800_000, 0.24),
        CandidateResult("sergio_fajardo", 880_000, 0.044),
        CandidateResult("ingrid_betancourt", 80_000, 0.004),
        CandidateResult("rest", 240_000, 0.012),
        CandidateResult("blanco", 400_000, 0.020),
    )
    total = sum(c.votes for c in candidates)
    return RoundResult(
        round_number=1,
        date=pd.Timestamp("2022-05-29"),
        total_valid_votes=total - 400_000,
        total_votes_incl_blank=total,
        registered_voters=39_000_000,
        polling_stations=12_500,
        candidates=candidates,
        blank_votes=400_000,
        null_votes=200_000,
        unmarked_votes=30_000,
    )


def test_build_round1_model_house_effects() -> None:
    """Test model graph with house effects and observation model."""
    polls = _make_3row_polls_7candidates()
    config = ModelConfig(random_walk_sigma_prior=0.5, house_effect_sigma_prior=1.0)
    model = build_round1_model(polls, None, config)

    # Free RVs: sigma_rw, sigma_house, phi_poll, theta_0, raw_house
    assert len(model.free_RVs) == 5
    # Deterministics: p_time, house_effects, p_adj
    assert len(model.deterministics) == 3
    # Observed RVs: poll_likelihood
    assert len(model.observed_RVs) == 1


def test_build_round1_model_prior_predictive() -> None:
    """Test that prior predictive samples produce valid shares.

    Verifies that both per-poll adjusted shares (``p_adj``) and per-time-point
    latent shares (``p_time``) fall in [0, 1] and that ``p_time`` sums to 1.0
    for each time point (simplex constraint).
    """
    polls = _make_3row_polls_3candidates()
    config = ModelConfig(random_walk_sigma_prior=0.5, house_effect_sigma_prior=1.0)
    model = build_round1_model(polls, None, config)

    with model:
        prior_pred = pm.sample_prior_predictive(draws=100, random_seed=config.seed)

    # Per-poll adjusted shares (includes house effects)
    p_adj = prior_pred.prior["p_adj"]
    assert p_adj.min() >= 0.0
    assert p_adj.max() <= 1.0

    # Per-time-point latent shares (no house effects)
    p_time = prior_pred.prior["p_time"]
    assert p_time.min() >= 0.0
    assert p_time.max() <= 1.0
    # Sum to 1 across candidates for every draw, chain, and time point
    np.testing.assert_allclose(p_time.sum(axis=-1), 1.0, atol=1e-6)


def test_build_round1_model_forecast_mode() -> None:
    """Test that forecast mode (results=None) has no election likelihood."""
    polls = _make_3row_polls_3candidates()
    config = ModelConfig()
    model = build_round1_model(polls, None, config)

    assert len(model.observed_RVs) == 1  # Only poll_likelihood


def test_build_round1_model_backtest_mode() -> None:
    """Test that backtest mode (results=RoundResult) includes election likelihood."""
    polls = _make_3row_polls_3candidates()
    results = _make_round1_result()
    config = ModelConfig()
    model = build_round1_model(polls, results, config)

    assert len(model.observed_RVs) == 2  # poll_likelihood + election_likelihood
    # Verify deterministics includes p_elec
    det_names = {d.name for d in model.deterministics}
    assert "p_elec" in det_names
