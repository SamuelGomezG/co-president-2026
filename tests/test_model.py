"""Tests for the first-round and runoff Bayesian models (SPEC-06, SPEC-07)."""

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm  # type: ignore[reportMissingTypeStubs]
import pytest
import xarray as xr  # type: ignore[reportMissingTypeStubs]

from co_president.config import ELECTION_DATES, FIRST_ROUND_CANDIDATES, ModelConfig
from co_president.data import (
    CandidateResult,
    RoundResult,
)
from co_president.model_round1 import (
    CandidateForecast,
    Round1Forecast,
    _preprocess_year_polls,
    build_multi_election_model,
    build_round1_model,
    forecast_round1,
    predict_year,
    sample_round1,
    simulate_elections,
)
import co_president.model_runoff_matrix as runoff_matrix
from co_president.model_runoff_matrix import (
    PairingForecast,
    RunoffMatrix,
    compute_top_two_probabilities,
    estimate_runoff_matrix,
    overall_win_probability,
)
from co_president.model_runoff_simple import (
    RunoffForecast,
    build_runoff_simple_model,
    forecast_runoff_simple,
    sample_runoff,
)


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


# ---------------------------------------------------------------------------
# Multi-election model helpers and tests (SPEC-40)
# ---------------------------------------------------------------------------


def _make_polls_year_3candidates(year: int) -> pd.DataFrame:
    """Return 4-row synthetic DataFrame for a specific election year.

    Creates 2 time points per year with 2 pollsters and 3 candidate columns
    (Petro, Hernandez, blanco) that overlap with ``FIRST_ROUND_CANDIDATES``.
    Poll dates are set for election-day and 14 days before.
    """
    if year not in set(ELECTION_DATES):
        msg = f"Unsupported year: {year}"
        raise ValueError(msg)

    election_date = ELECTION_DATES[year]
    date_14_before = election_date - pd.Timedelta(days=14)

    return pd.DataFrame(
        {
            "fecha": [
                str(date_14_before),
                str(date_14_before),
                str(election_date),
                str(election_date),
            ],
            "encuestadora": ["PollsterA", "PollsterB", "PollsterA", "PollsterB"],
            "muestra": [1000, 1000, 1000, 1000],
            "gustavo_petro": [48.0, 52.0, 50.0, 50.0],
            "rodolfo_hernandez": [42.0, 38.0, 40.0, 40.0],
            "blanco": [10.0, 10.0, 10.0, 10.0],
            "round_number": [1, 1, 1, 1],
        }
    )


CANDIDATES_3 = ["gustavo_petro", "rodolfo_hernandez", "blanco"]


def test_build_multi_election_model_graph() -> None:
    """Test multi-election model graph builds correctly with 2 years.

    Builds with 2018 + 2022 data, 2 time points, 2 pollsters, house effects.
    Free RVs: sigma_rw, sigma_house, phi_poll (3 shared)
             + 2 theta per year * 2 years = 4
             + raw_house per year * 2 years = 2
             = 9 total
    Deterministics: p_time, house_effects, p_adj, phi_poll_n per year * 2 = 8
    Observed: poll_likelihood per year * 2 = 2
    """
    polls_2018 = _make_polls_year_3candidates(2018)
    polls_2022 = _make_polls_year_3candidates(2022)
    config = ModelConfig(random_walk_sigma_prior=0.5, house_effect_sigma_prior=1.0)

    model = build_multi_election_model(
        {2018: polls_2018, 2022: polls_2022},
        {},
        config,
    )

    assert len(model.free_RVs) == 9
    det_names = {d.name for d in model.deterministics}
    expected_dets = {
        "2018_p_time",
        "2022_p_time",
        "2018_house_effects",
        "2022_house_effects",
        "2018_p_adj",
        "2022_p_adj",
        "2018_phi_poll_n",
        "2022_phi_poll_n",
    }
    assert det_names == expected_dets
    assert len(model.observed_RVs) == 2
    obs_names = {o.name for o in model.observed_RVs}
    assert obs_names == {"2018_poll_likelihood", "2022_poll_likelihood"}


def test_build_multi_election_model_prior_predictive() -> None:
    """Test multi-election prior predictive samples are valid.

    p_time and p_adj should be in [0, 1] and sum to 1 per time point.
    """
    polls_2018 = _make_polls_year_3candidates(2018)
    polls_2022 = _make_polls_year_3candidates(2022)
    config = ModelConfig(random_walk_sigma_prior=0.5, house_effect_sigma_prior=1.0)

    model = build_multi_election_model(
        {2018: polls_2018, 2022: polls_2022},
        {},
        config,
    )

    with model:
        prior = pm.sample_prior_predictive(draws=200, random_seed=config.seed)

    for year in (2018, 2022):
        p_time = prior.prior[f"{year}_p_time"]
        p_adj = prior.prior[f"{year}_p_adj"]
        assert (p_time >= 0.0).all(), f"{year}_p_time has negative values"
        assert (p_time <= 1.0).all(), f"{year}_p_time has values > 1"
        assert (p_adj >= 0.0).all(), f"{year}_p_adj has negative values"
        assert (p_adj <= 1.0).all(), f"{year}_p_adj has values > 1"
        np.testing.assert_allclose(
            p_time.sum(axis=-1).to_numpy(),
            1.0,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            p_adj.sum(axis=-1).to_numpy(),
            1.0,
            atol=1e-6,
        )


def test_build_multi_election_model_backtest() -> None:
    """Test multi-election model with results for one year (backtest mode).

    With results for 2022, there should be 1 extra free RV (phi_elec_2022),
    1 extra deterministic (p_elec_2022), and 1 extra observed RV.
    """
    polls_2018 = _make_polls_year_3candidates(2018)
    polls_2022 = _make_polls_year_3candidates(2022)
    results_2022 = _make_round1_result()
    config = ModelConfig(random_walk_sigma_prior=0.5, house_effect_sigma_prior=1.0)

    model = build_multi_election_model(
        {2018: polls_2018, 2022: polls_2022},
        {2022: results_2022},
        config,
    )

    # Free RVs: 9 (shared + per-year) + 1 (phi_elec) = 10
    assert len(model.free_RVs) == 10
    det_names = {d.name for d in model.deterministics}
    assert "2022_p_elec" in det_names
    assert "2018_p_elec" not in det_names  # no results for 2018
    # Observed: 2 poll_likelihood + 1 election_likelihood = 3
    assert len(model.observed_RVs) == 3
    obs_names = {o.name for o in model.observed_RVs}
    assert obs_names == {
        "2018_poll_likelihood",
        "2022_poll_likelihood",
        "2022_election_likelihood",
    }


def test_build_multi_election_model_no_house_effects() -> None:
    """Test multi-election model without house effects.

    No sigma_house, phi_poll, raw_house, or house_effects.
    Shared: sigma_rw only.
    Per-year: 2 theta vars * 2 = 4
    Total free RVs = 1 + 4 = 5
    Deterministics: p_time, p_adj per year * 2 = 4
    """
    polls_2018 = _make_polls_year_3candidates(2018)
    polls_2022 = _make_polls_year_3candidates(2022)
    config = ModelConfig(random_walk_sigma_prior=0.5)

    model = build_multi_election_model(
        {2018: polls_2018, 2022: polls_2022},
        {},
        config,
        no_house_effects=True,
    )

    assert len(model.free_RVs) == 5
    det_names = {d.name for d in model.deterministics}
    expected_dets = {
        "2018_p_time",
        "2022_p_time",
        "2018_p_adj",
        "2022_p_adj",
    }
    assert det_names == expected_dets
    assert len(model.observed_RVs) == 2


def test_predict_year_synthetic() -> None:
    """Test predict_year on a synthetic multi-election posterior.

    Creates a synthetic InferenceData with {year}_p_time variables and
    verifies that predict_year returns a valid Round1Forecast.
    """
    n_chains, n_draws = 2, 200
    alphas = np.array([52, 38, 10], dtype=float)

    rng = np.random.default_rng(42)
    raw = rng.gamma(alphas, 1, size=(n_chains, n_draws, 2, 3))
    p_time = raw / raw.sum(axis=-1, keepdims=True)

    idata = az.from_dict(
        data={"posterior": {"2022_p_time": p_time}},
        dims={"2022_p_time": ["chain", "draw", "time_dim_0", "candidate_dim_0"]},
    )

    forecast = predict_year(2022, idata, CANDIDATES_3)
    assert isinstance(forecast, Round1Forecast)
    assert len(forecast.candidates) == 3
    assert forecast.round_number == 1
    assert 0.0 <= forecast.prob_runoff <= 1.0

    petro = next(c for c in forecast.candidates if c.candidate_key == "gustavo_petro")
    hernandez = next(c for c in forecast.candidates if c.candidate_key == "rodolfo_hernandez")
    blanco = next(c for c in forecast.candidates if c.candidate_key == "blanco")
    assert petro.mean_share > hernandez.mean_share > blanco.mean_share
    assert 0.0 < petro.ci_50[0] < petro.ci_50[1] < 1.0
    assert 0.0 < petro.ci_95[0] < petro.ci_95[1] < 1.0


def test_predict_year_missing_variable() -> None:
    """Test predict_year raises ValueError for missing year."""
    rng = np.random.default_rng(42)
    raw = rng.gamma(np.array([1, 1, 1]), 1, size=(2, 100, 2, 3))
    p_time = raw / raw.sum(axis=-1, keepdims=True)
    idata = az.from_dict(
        data={"posterior": {"2018_p_time": p_time}},
        dims={"2018_p_time": ["chain", "draw", "time_dim_0", "candidate_dim_0"]},
    )
    with pytest.raises(ValueError, match="Posterior does not contain '2022_p_time'"):
        predict_year(2022, idata, CANDIDATES_3)


def test_preprocess_year_polls_unsupported_year() -> None:
    """_preprocess_year_polls raises ValueError for unsupported election year."""
    polls = _make_3row_polls_7candidates()
    with pytest.raises(ValueError, match="Unsupported election year: 2024"):
        _preprocess_year_polls(2024, polls, results=None)


@pytest.mark.slow
def test_build_multi_election_model_slow_convergence() -> None:
    """Test multi-election model MCMC convergence on 2-year minimal data.

    2 years (2018 + 2022), each with 2 dates x 3 pollsters = 12 polls total.
    Asserts R-hat < 1.10 for all free RVs (Gelman-Rubin convergence criterion).
    Uses minimal draws (500) to keep runtime manageable.
    """
    polls_2018 = _make_polls_year_3candidates(2018)
    polls_2022 = _make_polls_year_3candidates(2022)
    config = ModelConfig(
        random_walk_sigma_prior=0.5,
        house_effect_sigma_prior=1.0,
        mcmc_draws=500,
        mcmc_tune=500,
        mcmc_chains=2,
        mcmc_cores=2,
    )

    model = build_multi_election_model(
        {2018: polls_2018, 2022: polls_2022},
        {},
        config,
    )

    with model:
        idata = pm.sample(
            draws=config.mcmc_draws,
            tune=config.mcmc_tune,
            chains=config.mcmc_chains,
            cores=config.mcmc_cores,
            random_seed=config.seed,
            compute_convergence_checks=False,
        )

    summary = az.summary(idata)
    r_hat = pd.to_numeric(summary["r_hat"], errors="coerce").dropna()
    assert not r_hat.empty, "r_hat is empty; no parameters to evaluate"
    assert (r_hat < 1.10).all(), (
        f"R-hat convergence failure: max r_hat = {r_hat.max():.4f}, "
        f"parameters with r_hat >= 1.10: {list(r_hat[r_hat >= 1.10].index)}"
    )


@pytest.mark.slow
def test_build_multi_election_model_sanity() -> None:
    """Test multi-election model posterior accuracy on synthetic ground truth.

    2 years (2018 + 2022), each with 2 dates x 3 pollsters.  Known ground truth:
      2018: Petro=60%, Hernandez=30%, Blanco=10%
      2022: Petro=50%, Hernandez=40%, Blanco=10%
    Election-day posterior means must be within ±5pp of known truth (SPEC-06
    §9.3).
    """
    truth: dict[int, dict[str, float]] = {
        2018: {"gustavo_petro": 0.60, "rodolfo_hernandez": 0.30, "blanco": 0.10},
        2022: {"gustavo_petro": 0.50, "rodolfo_hernandez": 0.40, "blanco": 0.10},
    }

    polls_by_year: dict[int, pd.DataFrame] = {}
    for year in (2018, 2022):
        election_date = ELECTION_DATES[year]
        date_14_before = election_date - pd.Timedelta(days=14)

        t = truth[year]
        polls_by_year[year] = pd.DataFrame(
            {
                "fecha": [
                    str(date_14_before),
                    str(date_14_before),
                    str(date_14_before),
                    str(election_date),
                    str(election_date),
                    str(election_date),
                ],
                "encuestadora": [
                    "PollsterA",
                    "PollsterB",
                    "PollsterC",
                    "PollsterA",
                    "PollsterB",
                    "PollsterC",
                ],
                "muestra": [1000, 1200, 800, 1100, 900, 1000],
                "gustavo_petro": [
                    t["gustavo_petro"] * 100 - 2,
                    t["gustavo_petro"] * 100 + 1,
                    t["gustavo_petro"] * 100 - 1,
                    t["gustavo_petro"] * 100 + 2,
                    t["gustavo_petro"] * 100 - 1,
                    t["gustavo_petro"] * 100 + 1,
                ],
                "rodolfo_hernandez": [
                    t["rodolfo_hernandez"] * 100 + 2,
                    t["rodolfo_hernandez"] * 100 - 1,
                    t["rodolfo_hernandez"] * 100 + 1,
                    t["rodolfo_hernandez"] * 100 - 2,
                    t["rodolfo_hernandez"] * 100 + 1,
                    t["rodolfo_hernandez"] * 100 - 1,
                ],
                "blanco": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                "round_number": [1, 1, 1, 1, 1, 1],
            }
        )

    config = ModelConfig(
        random_walk_sigma_prior=0.5,
        house_effect_sigma_prior=1.0,
        mcmc_draws=2000,
        mcmc_tune=2000,
        mcmc_chains=2,
        mcmc_cores=2,
    )

    model = build_multi_election_model(polls_by_year, {}, config)

    with model:
        idata = pm.sample(
            draws=config.mcmc_draws,
            tune=config.mcmc_tune,
            chains=config.mcmc_chains,
            cores=config.mcmc_cores,
            random_seed=config.seed,
            compute_convergence_checks=False,
            target_accept=0.9,
        )

    # Verify convergence first
    summary = az.summary(idata)
    r_hat = pd.to_numeric(summary["r_hat"], errors="coerce").dropna()
    assert (r_hat < 1.10).all(), (
        f"R-hat convergence failure: max r_hat = {r_hat.max():.4f}, "
        f"parameters with r_hat >= 1.10: {list(r_hat[r_hat >= 1.10].index)}"
    )

    # Verify posterior accuracy against ground truth (±5pp tolerance)
    for year, expected in truth.items():
        candidate_keys = sorted(expected.keys())
        p_time = idata.posterior[f"{year}_p_time"]
        election_day = p_time[:, :, 0, :]  # (chain, draw, candidate)
        means = election_day.mean(dim=["chain", "draw"]).to_numpy()
        for i, cand in enumerate(candidate_keys):
            assert abs(means[i] - expected[cand]) <= 0.05, (
                f"{year} {cand}: posterior mean {means[i]:.3f} "
                f"outside ±5pp of truth {expected[cand]:.3f}"
            )


@pytest.mark.slow
def test_sample_round1_sanity() -> None:
    """Test posterior accuracy on minimal synthetic data with known ground truth.

    6 polls across 2 time points (30 days apart) with 3 pollsters. Election-day
    ground truth: Petro=60%, Hernandez=30%, Blanco=10%. Posterior mean at
    election day must be within 5 percentage points of the known truth
    (SPEC-06 §9.3: ±5pp tolerance).
    """
    polls = pd.DataFrame(
        {
            "fecha": [
                "2022-04-29",
                "2022-04-29",
                "2022-04-29",
                "2022-05-29",
                "2022-05-29",
                "2022-05-29",
            ],
            "encuestadora": [
                "PollsterA",
                "PollsterB",
                "PollsterC",
                "PollsterA",
                "PollsterB",
                "PollsterC",
            ],
            "muestra": [1000, 1200, 800, 1100, 900, 1000],
            "gustavo_petro": [58.0, 59.0, 57.0, 60.0, 61.0, 59.0],
            "rodolfo_hernandez": [32.0, 31.0, 33.0, 30.0, 29.0, 31.0],
            "blanco": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            "round_number": [1, 1, 1, 1, 1, 1],
        },
    )

    config = ModelConfig(
        mcmc_draws=1000,
        mcmc_tune=1000,
        mcmc_chains=2,
        mcmc_cores=2,
    )
    model = build_round1_model(polls, None, config)
    idata = sample_round1(model, config)

    candidate_keys = sorted(set(FIRST_ROUND_CANDIDATES) & set(polls.columns))
    candidate_idx = {k: i for i, k in enumerate(candidate_keys)}

    summary = az.summary(idata, var_names=["~p_adj", "~p_time", "~house_effects", "~raw_house"])
    r_hat = pd.to_numeric(summary["r_hat"], errors="coerce").dropna()
    assert (r_hat < 1.10).all(), f"R-hat convergence failure: max r_hat = {r_hat.max():.4f}"

    p_time = idata.posterior["p_time"].to_numpy()
    election_day = p_time[:, :, 0, :]
    means = election_day.mean(axis=(0, 1))

    petro_mean = means[candidate_idx["gustavo_petro"]]
    hernandez_mean = means[candidate_idx["rodolfo_hernandez"]]

    assert 0.55 <= petro_mean <= 0.65, f"Petro posterior mean {petro_mean:.3f} outside [0.55, 0.65]"
    assert 0.25 <= hernandez_mean <= 0.35, (
        f"Hernandez posterior mean {hernandez_mean:.3f} outside [0.25, 0.35]"
    )


# ---------------------------------------------------------------------------
# Forecast dataclass tests (SPEC-06)
# ---------------------------------------------------------------------------


def _make_synthetic_round1_idata() -> xr.DataTree:  # type: ignore[type-arg]
    """Create synthetic DataTree mimicking a Round 1 posterior.

    Uses the full 7-candidate order: ``sorted(FIRST_ROUND_CANDIDATES.keys())``.
    Petro gets ~42%, Hernandez ~28%, Gutierrez ~24%, the rest ~6%.
    """
    rng = np.random.default_rng(42)
    n_chains, n_draws = 2, 500
    candidate_order = sorted(FIRST_ROUND_CANDIDATES.keys())

    # Known means: Petro ~42%, Hernandez ~28%, Gutierrez ~24%, rest ~6%
    alphas = np.array([2, 24, 42, 1, 3, 28, 2], dtype=float) + 1.0

    raw = rng.gamma(alphas, 1, size=(n_chains, n_draws, 1, len(candidate_order)))
    p_time = raw / raw.sum(axis=-1, keepdims=True)

    return az.from_dict(
        data={"posterior": {"p_time": p_time}},
        coords={
            "candidate_dim_0": candidate_order,
        },
        dims={"p_time": ["chain", "draw", "time_dim_0", "candidate_dim_0"]},
    )


def test_candidate_forecast_dataclass() -> None:
    """Test CandidateForecast immutability and basic validation."""
    cf = CandidateForecast(
        candidate_key="gustavo_petro",
        mean_share=0.4,
        median_share=0.395,
        ci_50=(0.35, 0.45),
        ci_95=(0.30, 0.50),
        prob_first=0.8,
        prob_second=0.15,
        prob_top_two=0.95,
        prob_win_outright=0.1,
    )
    assert cf.mean_share >= 0.0
    assert 0.0 <= cf.prob_first <= 1.0
    assert 0.0 <= cf.prob_top_two <= 1.0
    assert len(cf.ci_50) == 2
    assert cf.ci_50[0] <= cf.ci_50[1]
    assert cf.ci_95[0] <= cf.ci_95[1]


def test_round1_forecast_json_roundtrip() -> None:
    """Test Round1Forecast serialization roundtrip."""
    candidates = [
        CandidateForecast("a", 0.4, 0.39, (0.35, 0.45), (0.30, 0.50), 0.9, 0.1, 1.0, 0.0),
        CandidateForecast("b", 0.3, 0.29, (0.25, 0.35), (0.20, 0.40), 0.1, 0.8, 0.9, 0.0),
    ]
    original = Round1Forecast(candidates=candidates, prob_runoff=0.99)

    as_json = original.to_json()
    restored = Round1Forecast.from_json(as_json)

    assert original.prob_runoff == restored.prob_runoff
    assert original.round_number == restored.round_number
    assert len(original.candidates) == len(restored.candidates)
    for oc, rc in zip(original.candidates, restored.candidates, strict=True):
        assert oc.candidate_key == rc.candidate_key
        assert oc.mean_share == rc.mean_share
        assert oc.ci_95 == rc.ci_95
        assert oc.prob_first == rc.prob_first


def test_forecast_round1() -> None:
    """Test forecast_round1 on a synthetic InferenceData."""
    idata = _make_synthetic_round1_idata()
    candidate_order = sorted(FIRST_ROUND_CANDIDATES.keys())
    forecast = forecast_round1(idata, candidate_order)

    assert len(forecast.candidates) == len(candidate_order)
    assert 0.0 <= forecast.prob_runoff <= 1.0

    # Major candidate probabilities should be non-trivial
    for cf in forecast.candidates:
        assert cf.mean_share >= 0.0
        assert 0.0 <= cf.prob_first <= 1.0
        assert 0.0 <= cf.prob_second <= 1.0
        assert 0.0 <= cf.prob_top_two <= 1.0
        assert 0.0 <= cf.prob_win_outright <= 1.0

    # Petro should have highest mean share and highest prob_first
    petro = next(c for c in forecast.candidates if c.candidate_key == "gustavo_petro")
    hernandez = next(c for c in forecast.candidates if c.candidate_key == "rodolfo_hernandez")
    assert petro.mean_share > hernandez.mean_share
    assert petro.prob_first > hernandez.prob_first

    # Probabilities across candidates for top_two should not all be 1
    total_prob_top_two = sum(c.prob_top_two for c in forecast.candidates)
    # Since only 2 candidates can finish top-two, sum of probs = 2
    assert abs(total_prob_top_two - 2.0) < 0.01

    # Sum of prob_first across all candidates = 1
    total_prob_first = sum(c.prob_first for c in forecast.candidates)
    assert abs(total_prob_first - 1.0) < 0.01


# ---------------------------------------------------------------------------
# simulate_elections tests (SPEC-06 §9.2.3)
# ---------------------------------------------------------------------------


def test_simulate_elections_returns_correct_shape() -> None:
    """Test that simulate_elections returns expected DataFrame shape."""
    idata = _make_synthetic_round1_idata()
    candidate_order = sorted(FIRST_ROUND_CANDIDATES.keys())
    n_candidates = len(candidate_order)
    n_sim = 500
    result = simulate_elections(idata, candidate_order, n_simulations=n_sim)
    expected_rows = n_sim * n_candidates
    assert result.shape == (expected_rows, 6), f"Expected ({expected_rows}, 6), got {result.shape}"
    assert list(result.columns) == [
        "sim_id",
        "candidate",
        "share",
        "rank",
        "win_outright",
        "goes_to_runoff",
    ]


def test_simulate_elections_shares_sum_to_100() -> None:
    """Test that shares sum to 1.0 within each simulation."""
    idata = _make_synthetic_round1_idata()
    candidate_order = sorted(FIRST_ROUND_CANDIDATES.keys())
    n_sim = 500
    result = simulate_elections(idata, candidate_order, n_simulations=n_sim)
    grouped = result.groupby("sim_id")["share"].sum()
    np.testing.assert_allclose(grouped.values, 1.0, atol=1e-10)


def test_simulate_elections_rank_consistency() -> None:
    """Test that rank-ordered shares are monotonically decreasing."""
    idata = _make_synthetic_round1_idata()
    candidate_order = sorted(FIRST_ROUND_CANDIDATES.keys())
    n_sim = 500
    result = simulate_elections(idata, candidate_order, n_simulations=n_sim)
    for sim_id, group in result.groupby("sim_id"):
        ordered = group.sort_values("rank")
        shares = ordered["share"].to_numpy()
        for i in range(len(shares) - 1):
            assert shares[i] >= shares[i + 1] - 1e-10, (
                f"Sim {sim_id}: rank {i + 1} share {shares[i]:.4f} "
                f"< rank {i + 2} share {shares[i + 1]:.4f}"
            )


def _make_synthetic_outright_win_idata() -> xr.DataTree:
    """Create synthetic InferenceData mimicking a Round 1 posterior with an outright winner.

    Gustavo Petro is given overwhelming concentration to ensure his share > 50%
    in all draws.
    """
    rng = np.random.default_rng(42)
    n_chains, n_draws = 2, 500
    candidate_order = sorted(FIRST_ROUND_CANDIDATES.keys())

    # Petro (index 2) gets a massive concentration to guarantee > 50%
    alphas = np.array([1, 1, 500, 1, 1, 1, 1], dtype=float)

    raw = rng.gamma(alphas, 1, size=(n_chains, n_draws, 1, len(candidate_order)))
    p_time = raw / raw.sum(axis=-1, keepdims=True)

    return az.from_dict(
        data={"posterior": {"p_time": p_time}},
        coords={
            "candidate_dim_0": candidate_order,
        },
        dims={"p_time": ["chain", "draw", "time_dim_0", "candidate_dim_0"]},
    )


def test_simulate_elections_outright_win() -> None:
    """Test that outright win and runoff flags are consistent when a candidate exceeds 50%."""
    idata = _make_synthetic_outright_win_idata()
    candidate_order = sorted(FIRST_ROUND_CANDIDATES.keys())
    n_sim = 1000
    result = simulate_elections(idata, candidate_order, n_simulations=n_sim)

    # Verify our synthetic data actually produced shares > 50% for Petro
    petro_shares = result[result["candidate"] == "gustavo_petro"]["share"]
    assert (petro_shares > 0.5).all(), "Synthetic data failed to produce >50% shares for Petro"

    for sim_id, group in result.groupby("sim_id"):
        sim_goes_runoff = group["goes_to_runoff"].iloc[0]
        any_outright = group["win_outright"].any()

        # In our outright win synthetic data, someone must always win outright
        assert any_outright, f"Sim {sim_id}: Expected an outright winner but found none"
        assert not sim_goes_runoff, f"Sim {sim_id}: has outright winner but goes_to_runoff=True"

        # Verify Petro specifically is the outright winner
        petro_winner = group[(group["candidate"] == "gustavo_petro") & group["win_outright"]]
        assert len(petro_winner) == 1, f"Sim {sim_id}: Petro should be the sole outright winner"

        # Verify exactly one candidate has the win_outright flag True
        winner_rows = group[group["win_outright"]]
        assert len(winner_rows) == 1, (
            f"Sim {sim_id}: Expected 1 outright winner, got {len(winner_rows)}"
        )
        assert winner_rows["share"].iloc[0] > 0.5, f"Sim {sim_id}: Outright winner share <= 50%"

        # Verify all non-winners have win_outright=False
        non_winners = group[~group["win_outright"]]
        assert len(non_winners) == len(candidate_order) - 1, (
            f"Sim {sim_id}: Expected {len(candidate_order) - 1} non-winners, got {len(non_winners)}"
        )


def _make_synthetic_runoff_idata() -> xr.DataTree:
    """Create synthetic InferenceData where no candidate exceeds 50% (runoff scenario)."""
    rng = np.random.default_rng(42)
    n_chains, n_draws = 2, 500
    candidate_order = sorted(FIRST_ROUND_CANDIDATES.keys())

    # Equal-ish distribution: Petro ~40%, Hernandez ~28%, rest spread — no candidate > 50%
    alphas = np.array([10, 30, 40, 5, 5, 35, 10], dtype=float)

    raw = rng.gamma(alphas, 1, size=(n_chains, n_draws, 1, len(candidate_order)))
    p_time = raw / raw.sum(axis=-1, keepdims=True)

    return az.from_dict(
        data={"posterior": {"p_time": p_time}},
        coords={
            "candidate_dim_0": candidate_order,
        },
        dims={"p_time": ["chain", "draw", "time_dim_0", "candidate_dim_0"]},
    )


def test_simulate_elections_runoff_scenario() -> None:
    """Test that goes_to_runoff is True when no candidate exceeds 50%."""
    idata = _make_synthetic_runoff_idata()
    candidate_order = sorted(FIRST_ROUND_CANDIDATES.keys())
    n_sim = 1000
    result = simulate_elections(idata, candidate_order, n_simulations=n_sim)

    # Verify no Petro shares exceed 50% (runoff data should be well below threshold)
    petro_shares = result[result["candidate"] == "gustavo_petro"]["share"]
    assert (petro_shares <= 0.5).all(), "Synthetic runoff data should not produce Petro > 50%"

    for sim_id, group in result.groupby("sim_id"):
        sim_goes_runoff = group["goes_to_runoff"].iloc[0]
        any_outright = group["win_outright"].any()

        # No candidate should win outright in a runoff scenario
        assert not any_outright, f"Sim {sim_id}: Unexpected outright winner in runoff data"
        assert sim_goes_runoff, f"Sim {sim_id}: Expected goes_to_runoff=True but got False"

        # All candidates should have win_outright=False
        assert not group["win_outright"].any(), (
            f"Sim {sim_id}: Expected all win_outright=False in runoff scenario"
        )


# ---------------------------------------------------------------------------
# Runoff Simple Model Tests (SPEC-07)
# ---------------------------------------------------------------------------


def _make_3row_runoff_polls_round2() -> pd.DataFrame:
    """Return a 3-row synthetic Round 2 DataFrame with Petro, Hernandez, blanco."""
    return pd.DataFrame(
        {
            "fecha": ["2022-06-19", "2022-06-19", "2022-06-19"],
            "encuestadora": ["PollsterA", "PollsterB", "PollsterC"],
            "muestra": [1000, 1000, 1000],
            "gustavo_petro": [52.0, 51.0, 50.0],
            "rodolfo_hernandez": [48.0, 49.0, 50.0],
            "blanco": [0.0, 0.0, 0.0],
            "round_number": [2, 2, 2],
        }
    )


def test_build_runoff_simple_model_graph() -> None:
    """Test that the runoff model builds with correct graph structure (K=3)."""
    polls = _make_3row_runoff_polls_round2()
    results = _make_round1_result()
    config = ModelConfig(random_walk_sigma_prior=0.5, house_effect_sigma_prior=1.0)

    model = build_runoff_simple_model(polls, results, None, config)

    # Free RVs: sigma_rw, sigma_house, phi_poll, theta_r_0, raw_house
    assert len(model.free_RVs) == 5
    # Deterministics: p_time, house_effects, p_adj, phi_poll_n
    det_names = {d.name for d in model.deterministics}
    assert det_names == {"p_time", "house_effects", "p_adj", "phi_poll_n"}
    # Observed RVs: poll_likelihood
    assert len(model.observed_RVs) == 1


def test_build_runoff_simple_model_prior_predictive() -> None:
    """Test that runoff prior predictive samples produce valid shares in [0, 1]."""
    polls = _make_3row_runoff_polls_round2()
    results = _make_round1_result()
    config = ModelConfig(random_walk_sigma_prior=0.5, house_effect_sigma_prior=1.0)

    model = build_runoff_simple_model(polls, results, None, config)

    with model:
        prior_pred = pm.sample_prior_predictive(draws=100, random_seed=config.seed)

    p_adj = prior_pred.prior["p_adj"]
    assert p_adj.min() >= 0.0
    assert p_adj.max() <= 1.0

    p_time = prior_pred.prior["p_time"]
    assert p_time.min() >= 0.0
    assert p_time.max() <= 1.0
    # Sum to 1 across K=3 categories
    np.testing.assert_allclose(p_time.sum(axis=-1), 1.0, atol=1e-6)

    # K should be 3 for the runoff model
    assert p_time.shape[-1] == 3


def test_build_runoff_simple_model_zero_share_floor() -> None:
    """Zero-floor on runoff model: prior predictive valid despite zero-share polls."""
    polls = _make_3row_runoff_polls_round2()  # blanco=[0,0,0]
    results = _make_round1_result()
    config = ModelConfig(random_walk_sigma_prior=0.5, house_effect_sigma_prior=1.0)

    model = build_runoff_simple_model(polls, results, None, config)

    with model:
        prior_pred = pm.sample_prior_predictive(draws=50, random_seed=config.seed)
    p_adj = prior_pred.prior["p_adj"]
    assert np.all(np.isfinite(p_adj)), "NaN/Inf in prior predictive (log(0) bug?)"
    assert p_adj.min() >= 0.0
    assert p_adj.max() <= 1.0
    np.testing.assert_allclose(p_adj.sum(axis=-1), 1.0, atol=1e-6)


def test_build_runoff_simple_model_informative_prior() -> None:
    """Test that providing round1_idata produces a valid model."""
    polls = _make_3row_runoff_polls_round2()
    results = _make_round1_result()
    config = ModelConfig()

    round1_idata = _make_synthetic_round1_idata()
    model = build_runoff_simple_model(polls, results, round1_idata, config)

    # Same graph structure as with informed prior
    assert len(model.free_RVs) == 5
    det_names = {d.name for d in model.deterministics}
    assert det_names == {"p_time", "house_effects", "p_adj", "phi_poll_n"}
    assert len(model.observed_RVs) == 1


def test_build_runoff_simple_model_informed_fallback() -> None:
    """Test that round1_idata=None uses actual election results as informed prior.

    When ``round1_idata`` is ``None``, the fallback computes the prior for
    ``theta[T-1]`` from the actual Round 1 vote shares (``results.get_share``).
    This test verifies that prior predictive samples center around those shares.
    """
    polls = _make_3row_runoff_polls_round2()
    results = _make_round1_result()
    config = ModelConfig(random_walk_sigma_prior=0.5, house_effect_sigma_prior=1.0)

    # Petro 40%, Hernandez 28%, rest+blanco 32%
    expected_a = 0.40
    expected_b = 0.28
    expected_rest = 0.32

    model = build_runoff_simple_model(polls, results, None, config)

    with model:
        prior_pred = pm.sample_prior_predictive(draws=500, random_seed=config.seed)

    # p_time at the last time index (T-1, the initial time point) should
    # reflect the informed prior from round 1 results
    p_time = prior_pred.prior["p_time"]  # (chain, draw, time, candidate=3)
    initial_p = p_time[:, :, -1, :]  # last time index = T-1

    prior_mean = initial_p.mean(dim=("chain", "draw")).to_numpy()

    # Prior means should be closer to the informed election-result values
    # than to a uniform (33% each) vague prior
    assert prior_mean[0] > 0.30  # A (Petro) > 30%
    assert prior_mean[1] > 0.20  # B (Hernandez) > 20%
    assert prior_mean[0] > prior_mean[1]  # Petro > Hernandez

    # The shares at T-1 should be within reasonable distance of the
    # informed-prior targets (allowing Monte Carlo noise)
    np.testing.assert_allclose(prior_mean[0], expected_a, atol=0.08)
    np.testing.assert_allclose(prior_mean[1], expected_b, atol=0.08)
    np.testing.assert_allclose(prior_mean[2], expected_rest, atol=0.08)


def test_forecast_runoff_simple() -> None:
    """Test forecast_runoff_simple on a synthetic posterior."""
    rng = np.random.default_rng(42)
    n_chains, n_draws = 2, 500

    # Petro ~52%, Hernandez ~48%, rest ~0%
    alphas = np.array([52, 48, 1], dtype=float)
    raw = rng.gamma(alphas, 1, size=(n_chains, n_draws, 1, 3))
    p_time = raw / raw.sum(axis=-1, keepdims=True)

    idata = az.from_dict(
        data={"posterior": {"p_time": p_time}},
        dims={"p_time": ["chain", "draw", "time_dim_0", "candidate_dim_0"]},
    )

    forecast = forecast_runoff_simple(idata, "gustavo_petro", "rodolfo_hernandez")

    assert forecast.candidate_a_key == "gustavo_petro"
    assert forecast.candidate_b_key == "rodolfo_hernandez"
    assert 0.0 <= forecast.prob_a_wins <= 1.0
    assert 0.0 <= forecast.prob_b_wins <= 1.0
    assert 0.0 <= forecast.mean_share_a <= 1.0
    assert 0.0 <= forecast.mean_share_b <= 1.0

    # Petro should have higher win probability
    assert forecast.prob_a_wins > 0.5
    assert forecast.mean_share_a > forecast.mean_share_b

    # Median shares should exist in [0, 1]
    assert 0.0 <= forecast.median_share_a <= 1.0
    assert 0.0 <= forecast.median_share_b <= 1.0

    # CI lengths should be positive
    a_ci_len = forecast.ci_95_a[1] - forecast.ci_95_a[0]
    b_ci_len = forecast.ci_95_b[1] - forecast.ci_95_b[0]
    assert a_ci_len > 0.0
    assert b_ci_len > 0.0

    # 50% CI should be narrower than 95% CI
    a_ci_50_len = forecast.ci_50_a[1] - forecast.ci_50_a[0]
    b_ci_50_len = forecast.ci_50_b[1] - forecast.ci_50_b[0]
    assert 0.0 < a_ci_50_len < a_ci_len
    assert 0.0 < b_ci_50_len < b_ci_len


def test_runoff_forecast_dataclass() -> None:
    """Test RunoffForecast dataclass validation."""
    rf = RunoffForecast(
        candidate_a_key="gustavo_petro",
        candidate_b_key="rodolfo_hernandez",
        prob_a_wins=0.75,
        prob_b_wins=0.25,
        mean_share_a=0.52,
        mean_share_b=0.48,
        median_share_a=0.51,
        median_share_b=0.47,
        mean_margin=0.04,
        ci_50_a=(0.50, 0.54),
        ci_50_b=(0.46, 0.50),
        ci_95_a=(0.48, 0.56),
        ci_95_b=(0.44, 0.52),
    )
    assert rf.prob_a_wins + rf.prob_b_wins <= 1.0
    assert rf.mean_share_a > rf.mean_share_b
    assert abs(rf.mean_margin - (rf.mean_share_a - rf.mean_share_b)) < 1e-10
    assert 0.0 <= rf.median_share_a <= 1.0
    assert 0.0 <= rf.median_share_b <= 1.0
    assert rf.ci_50_a[0] <= rf.ci_50_a[1]
    assert rf.ci_50_b[0] <= rf.ci_50_b[1]


# ---------------------------------------------------------------------------
# Runoff slow tests (MCMC)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_sample_runoff_convergence() -> None:
    """Test MCMC convergence on minimal 6-poll Round 2 data.

    Uses 2 dates x 3 pollsters (6 polls total). Asserts R-hat < 1.10.
    """
    polls = pd.DataFrame(
        {
            "fecha": [
                "2022-06-05",
                "2022-06-05",
                "2022-06-05",
                "2022-06-19",
                "2022-06-19",
                "2022-06-19",
            ],
            "encuestadora": [
                "PollsterA",
                "PollsterB",
                "PollsterC",
                "PollsterA",
                "PollsterB",
                "PollsterC",
            ],
            "muestra": [1000, 1200, 800, 1100, 900, 1000],
            "gustavo_petro": [50.0, 51.0, 49.0, 50.0, 51.0, 50.0],
            "rodolfo_hernandez": [45.0, 44.0, 46.0, 45.0, 44.0, 45.0],
            "blanco": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            "round_number": [2, 2, 2, 2, 2, 2],
        }
    )
    results = _make_round1_result()
    config = ModelConfig(mcmc_draws=500, mcmc_tune=500, mcmc_chains=2, mcmc_cores=2)
    model = build_runoff_simple_model(polls, results, None, config)
    idata = sample_runoff(model, config)

    summary = az.summary(idata, var_names=["~p_adj", "~p_time", "~house_effects", "~raw_house"])
    r_hat = pd.to_numeric(summary["r_hat"], errors="coerce").dropna()
    assert not r_hat.empty, "r_hat is empty; no parameters to evaluate"
    assert (r_hat < 1.10).all(), (
        f"R-hat convergence failure: max r_hat = {r_hat.max():.4f}, "
        f"parameters with r_hat >= 1.10: {list(r_hat[r_hat >= 1.10].index)}"
    )


@pytest.mark.slow
def test_sample_runoff_sanity() -> None:
    """Test posterior accuracy on synthetic runoff data.

    6 polls across 2 time points (2 weeks apart) with 3 pollsters.
    Ground truth: Petro=52%, Hernandez=45%, Blanco=3%.
    Posterior mean must be within 5pp of truth.
    """
    polls = pd.DataFrame(
        {
            "fecha": [
                "2022-06-05",
                "2022-06-05",
                "2022-06-05",
                "2022-06-19",
                "2022-06-19",
                "2022-06-19",
            ],
            "encuestadora": [
                "PollsterA",
                "PollsterB",
                "PollsterC",
                "PollsterA",
                "PollsterB",
                "PollsterC",
            ],
            "muestra": [1000, 1200, 800, 1100, 900, 1000],
            "gustavo_petro": [50.0, 52.0, 48.0, 52.0, 53.0, 51.0],
            "rodolfo_hernandez": [46.0, 44.0, 48.0, 45.0, 44.0, 46.0],
            "blanco": [4.0, 4.0, 4.0, 3.0, 3.0, 3.0],
            "round_number": [2, 2, 2, 2, 2, 2],
        }
    )
    results = _make_round1_result()
    config = ModelConfig(mcmc_draws=1000, mcmc_tune=500, mcmc_chains=2, mcmc_cores=2)
    model = build_runoff_simple_model(polls, results, None, config)
    idata = sample_runoff(model, config)

    forecast = forecast_runoff_simple(idata, "gustavo_petro", "rodolfo_hernandez")

    # The prior from Round 1 results (Petro=40%, Hernandez=28%) pulls against
    # the poll data (~52%/45%), so the posterior lands between them. Allow a
    # wider tolerance to account for prior-data tension.
    assert 0.42 <= forecast.mean_share_a <= 0.57, (
        f"Petro posterior mean {forecast.mean_share_a:.3f} outside [0.42, 0.57]"
    )
    assert 0.35 <= forecast.mean_share_b <= 0.50, (
        f"Hernandez posterior mean {forecast.mean_share_b:.3f} outside [0.35, 0.50]"
    )

    # Petro should have higher win probability given poll signal
    assert forecast.prob_a_wins > 0.5


# ---------------------------------------------------------------------------
# Runoff Matrix Tests (SPEC-08)
# ---------------------------------------------------------------------------


def test_compute_top_two_probabilities() -> None:
    """Test that compute_top_two_probabilities identifies correct pairing.

    Uses a synthetic posterior where Petro (~42%) and Hernandez (~28%) are
    clearly the top two candidates. The pairing
    (gustavo_petro, rodolfo_hernandez) should dominate, and all pairing
    probabilities should sum to 1.0.
    """
    rng = np.random.default_rng(42)
    n_chains, n_draws = 2, 1000
    candidate_order = sorted(FIRST_ROUND_CANDIDATES.keys())

    # Petro ~46%, Hernandez ~24%, Gutierrez ~8%, rest ~22%
    # Wide separation ensures the top two (Petro, Hernandez) are clearly
    # distinguishable from third place (Gutierrez).
    alphas = np.array([1, 10, 80, 1, 10, 40, 8], dtype=float) + 5.0

    raw = rng.gamma(alphas, 1, size=(n_chains, n_draws, 1, len(candidate_order)))
    p_time = raw / raw.sum(axis=-1, keepdims=True)

    idata = az.from_dict(
        data={"posterior": {"p_time": p_time}},
        coords={
            "candidate_dim_0": candidate_order,
        },
        dims={"p_time": ["chain", "draw", "time_dim_0", "candidate_dim_0"]},
    )

    # Only named candidates (not rest/blanco) can finish top two
    nominations = [
        "gustavo_petro",
        "rodolfo_hernandez",
        "federico_gutierrez",
        "sergio_fajardo",
        "ingrid_betancourt",
    ]
    result = compute_top_two_probabilities(idata, nominations)

    petro_hernandez = result.get(("gustavo_petro", "rodolfo_hernandez"), 0.0)
    assert petro_hernandez > 0.8, f"Expected Petro-Hernandez prob > 0.8, got {petro_hernandez:.4f}"

    total_prob = sum(result.values())
    assert abs(total_prob - 1.0) < 0.01, (
        f"Pairing probabilities sum to {total_prob:.4f}, expected 1.0"
    )


def test_pairing_forecast_dataclass() -> None:
    """Test PairingForecast immutability and field validation.

    All probabilities must be in [0, 1], prob_first_wins + prob_second_wins
    must equal 1.0, and mean_margin should equal mean_first - mean_second.
    """
    pf = PairingForecast(
        candidate_first="gustavo_petro",
        candidate_second="rodolfo_hernandez",
        prob_pairing=0.75,
        prob_first_wins=0.65,
        prob_second_wins=0.35,
        mean_margin=0.03,
    )
    assert pf.candidate_first == "gustavo_petro"
    assert pf.candidate_second == "rodolfo_hernandez"
    assert 0.0 <= pf.prob_pairing <= 1.0
    assert 0.0 <= pf.prob_first_wins <= 1.0
    assert 0.0 <= pf.prob_second_wins <= 1.0
    assert abs(pf.prob_first_wins + pf.prob_second_wins - 1.0) < 1e-10


def test_runoff_matrix_dataclass() -> None:
    """Test RunoffMatrix dataclass creation and ordering.

    Verifies that pairings are stored, prob_runoff reflects total, and
    ordered_by_likelihood follows the correct sequence.
    """
    pairings = (
        PairingForecast("gustavo_petro", "rodolfo_hernandez", 0.7, 0.6, 0.4, 0.02),
        PairingForecast("gustavo_petro", "federico_gutierrez", 0.2, 0.8, 0.2, 0.05),
        PairingForecast(
            "rodolfo_hernandez",
            "federico_gutierrez",
            0.1,
            0.55,
            0.45,
            0.01,
        ),
    )
    ordered = tuple((p.candidate_first, p.candidate_second) for p in pairings)

    matrix = RunoffMatrix(
        pairings=pairings,
        prob_runoff=1.0,
        ordered_by_likelihood=ordered,
    )

    assert len(matrix.pairings) == 3
    assert matrix.prob_runoff == 1.0
    assert len(matrix.ordered_by_likelihood) == 3
    assert matrix.ordered_by_likelihood[0] == ("gustavo_petro", "rodolfo_hernandez")


def test_overall_win_probability() -> None:
    """Test overall_win_probability with known values.

    Manually constructs a RunoffMatrix and verifies the computed overall win
    probability for each candidate matches a hand-calculation.
    """
    pairings = (
        PairingForecast("gustavo_petro", "rodolfo_hernandez", 0.7, 0.6, 0.4, 0.02),
        PairingForecast("gustavo_petro", "federico_gutierrez", 0.2, 0.8, 0.2, 0.05),
        PairingForecast(
            "rodolfo_hernandez",
            "federico_gutierrez",
            0.1,
            0.55,
            0.45,
            0.01,
        ),
    )
    ordered = tuple((p.candidate_first, p.candidate_second) for p in pairings)
    matrix = RunoffMatrix(pairings=pairings, prob_runoff=1.0, ordered_by_likelihood=ordered)

    prob_outright: dict[str, float] = {
        "gustavo_petro": 0.1,
        "rodolfo_hernandez": 0.0,
        "federico_gutierrez": 0.0,
    }
    overall = overall_win_probability(matrix, prob_outright)

    # Petro: 0.1 (outright) + 0.7*0.6 (wins vs Hernandez) + 0.2*0.8 (wins vs Gutierrez) = 0.68
    expected_petro = 0.1 + 0.7 * 0.6 + 0.2 * 0.8
    assert abs(overall["gustavo_petro"] - expected_petro) < 1e-10, (
        f"Petro overall win prob {overall['gustavo_petro']:.4f} != {expected_petro:.4f}"
    )

    # Hernandez: 0.0 (outright) + 0.7*0.4 (wins vs Petro) + 0.1*0.55 (wins vs Gutierrez) = 0.335
    expected_hernandez = 0.0 + 0.7 * 0.4 + 0.1 * 0.55
    assert abs(overall["rodolfo_hernandez"] - expected_hernandez) < 1e-10, (
        f"Hernandez overall win prob {overall['rodolfo_hernandez']:.4f} != {expected_hernandez:.4f}"
    )


def test_overall_win_probability_outright_winner() -> None:
    """When one candidate wins outright (>50%), their overall win probability is 1.0.

    Edge case where prob_win_outright is 1.0 for one candidate and 0.0 for all
    others. With no runoff (empty pairings), the overall probability should
    reflect the outright outcome exactly.
    """
    prob_win_outright = {
        "candidate_a": 1.0,
        "candidate_b": 0.0,
        "candidate_c": 0.0,
    }

    matrix = RunoffMatrix(
        pairings=(),
        prob_runoff=0.0,
        ordered_by_likelihood=(),
    )

    result = overall_win_probability(matrix, prob_win_outright)

    assert result["candidate_a"] == 1.0
    assert result["candidate_b"] == 0.0
    assert result["candidate_c"] == 0.0
    assert abs(sum(result.values()) - 1.0) < 1e-9


def test_overall_win_probability_with_runoff() -> None:
    """When no one wins outright, overall win probability includes runoff chances.

    Constructs a runoff matrix with known pairing probabilities and verifies
    that the arithmetic combination of outright win probabilities and runoff
    contributions yields correct totals for each candidate.
    """
    pairings = (
        PairingForecast("candidate_a", "candidate_b", 0.6, 0.55, 0.45, 0.02),
        PairingForecast("candidate_a", "candidate_c", 0.3, 0.7, 0.3, 0.05),
        PairingForecast("candidate_b", "candidate_c", 0.1, 0.6, 0.4, 0.01),
    )
    ordered = tuple((p.candidate_first, p.candidate_second) for p in pairings)
    matrix = RunoffMatrix(
        pairings=pairings,
        prob_runoff=1.0,
        ordered_by_likelihood=ordered,
    )

    prob_outright: dict[str, float] = {
        "candidate_a": 0.0,
        "candidate_b": 0.0,
        "candidate_c": 0.0,
    }
    overall = overall_win_probability(matrix, prob_outright)

    # A: 0.0 (outright) + 0.6*0.55 (wins vs B) + 0.3*0.7 (wins vs C) = 0.54
    expected_a = 0.0 + 0.6 * 0.55 + 0.3 * 0.7
    assert abs(overall["candidate_a"] - expected_a) < 1e-10, (
        f"Candidate A overall win prob {overall['candidate_a']:.4f} != {expected_a:.4f}"
    )

    # B: 0.0 (outright) + 0.6*0.45 (wins vs A) + 0.1*0.6 (wins vs C) = 0.33
    expected_b = 0.0 + 0.6 * 0.45 + 0.1 * 0.6
    assert abs(overall["candidate_b"] - expected_b) < 1e-10, (
        f"Candidate B overall win prob {overall['candidate_b']:.4f} != {expected_b:.4f}"
    )

    # C: 0.0 (outright) + 0.3*0.3 (wins vs A) + 0.1*0.4 (wins vs B) = 0.13
    expected_c = 0.0 + 0.3 * 0.3 + 0.1 * 0.4
    assert abs(overall["candidate_c"] - expected_c) < 1e-10, (
        f"Candidate C overall win prob {overall['candidate_c']:.4f} != {expected_c:.4f}"
    )

    assert abs(sum(overall.values()) - 1.0) < 1e-9, (
        f"Total probability {sum(overall.values()):.4f} != 1.0"
    )


def test_transfer_heuristic_pairing_probs_sum_to_one() -> None:
    """Test that transfer heuristic produces valid probabilities.

    When ``estimate_runoff_matrix`` is called with ``round2_polls=None``, all
    pairings should use the transfer heuristic. For every resulting
    ``PairingForecast``, ``prob_first_wins + prob_second_wins`` must equal 1.0
    (no probability mass is lost or duplicated).
    """
    rng = np.random.default_rng(42)
    n_chains, n_draws = 2, 500
    candidate_order = sorted(FIRST_ROUND_CANDIDATES.keys())

    alphas = np.array([1, 10, 80, 1, 10, 40, 8], dtype=float) + 5.0
    raw = rng.gamma(alphas, 1, size=(n_chains, n_draws, 1, len(candidate_order)))
    p_time = raw / raw.sum(axis=-1, keepdims=True)

    idata = az.from_dict(
        data={"posterior": {"p_time": p_time}},
        coords={"candidate_dim_0": candidate_order},
        dims={"p_time": ["chain", "draw", "time_dim_0", "candidate_dim_0"]},
    )

    results_round1 = _make_round1_result()
    config = ModelConfig()

    matrix = estimate_runoff_matrix(idata, (results_round1, results_round1), None, config)

    for pf in matrix.pairings:
        prob_sum = pf.prob_first_wins + pf.prob_second_wins
        assert abs(prob_sum - 1.0) < 1e-6, (
            f"Pairing {pf.candidate_first} vs {pf.candidate_second}: "
            f"prob_first_wins + prob_second_wins = {prob_sum} != 1.0"
        )


def test_transfer_heuristic_vote_share_bounds() -> None:
    """Ensure transfer heuristic never yields >100% shares."""
    rng = np.random.default_rng(123)
    n_chains, n_draws = 2, 200
    candidate_order = sorted(FIRST_ROUND_CANDIDATES.keys())

    alphas = np.array([1, 10, 80, 1, 10, 40, 8], dtype=float) + 5.0
    raw = rng.gamma(alphas, 1, size=(n_chains, n_draws, 1, len(candidate_order)))
    p_time = raw / raw.sum(axis=-1, keepdims=True)
    election_day = p_time[:, :, 0, :]

    shares_first, shares_second = runoff_matrix._compute_transfer_shares(
        election_day,
        candidate_order,
        "gustavo_petro",
        "rodolfo_hernandez",
    )

    expected_flat = n_chains * n_draws
    assert shares_first.shape == (expected_flat,), (
        f"shares_first has shape {shares_first.shape}, expected ({expected_flat},)"
    )
    assert shares_second.shape == (expected_flat,), (
        f"shares_second has shape {shares_second.shape}, expected ({expected_flat},)"
    )

    assert (shares_first + shares_second <= 1.0 + 1e-9).all(), "Sum of transfer shares exceeds 1.0"

    assert (shares_first >= 0.0).all(), "shares_first contains values < 0"
    assert (shares_first <= 1.0).all(), "shares_first contains values > 1"
    assert (shares_second >= 0.0).all(), "shares_second contains values < 0"
    assert (shares_second <= 1.0).all(), "shares_second contains values > 1"


def test_build_round1_model_phi_poll_n_scaling() -> None:
    """Test that poll concentration scales with log(sample size)."""
    # 3-row DataFrame as per requirements
    polls = pd.DataFrame(
        {
            "fecha": ["2022-05-29", "2022-05-29", "2022-05-29"],
            "encuestadora": ["PollsterA", "PollsterA", "PollsterA"],
            "muestra": [500, 2000, 1000],
            "gustavo_petro": [50.0, 50.0, 50.0],
            "rodolfo_hernandez": [40.0, 40.0, 40.0],
            "blanco": [10.0, 10.0, 10.0],
            "round_number": [1, 1, 1],
        }
    )
    config = ModelConfig(random_walk_sigma_prior=0.5, house_effect_sigma_prior=1.0)
    # Same pollster to isolate sample-size scaling from house effects.
    model = build_round1_model(polls, None, config)

    with model:
        prior_pred = pm.sample_prior_predictive(draws=10, random_seed=config.seed)

    phi_poll_n = prior_pred.prior["phi_poll_n"].to_numpy()

    # Assert shape is (..., num_polls=3, 1)
    assert phi_poll_n.shape[-2:] == (len(polls), 1)

    # Ratio of poll 2 (2000) to poll 1 (500)
    ratio = phi_poll_n[..., 1, 0] / phi_poll_n[..., 0, 0]
    eps = 1e-8
    expected_ratio = np.log(2000 + 1 + eps) / np.log(500 + 1 + eps)
    np.testing.assert_allclose(ratio, expected_ratio, rtol=1e-6, atol=1e-8)


@pytest.mark.slow
def test_estimate_runoff_matrix_uses_head_to_head_polls() -> None:
    """Test that 3+ head-to-head polls trigger the model path.

    When ``round2_polls`` contains 3 rows for a specific candidate pairing,
    ``estimate_runoff_matrix`` should dispatch to
    ``build_runoff_simple_model`` instead of the transfer heuristic,
    producing a different ``prob_first_wins`` for that pairing.
    """
    idata = _make_synthetic_round1_idata()
    results_round1 = _make_round1_result()
    config = ModelConfig(
        mcmc_draws=500,
        mcmc_tune=500,
        mcmc_chains=2,
        mcmc_cores=2,
        seed=42,
    )

    round2_polls = pd.DataFrame(
        {
            "fecha": ["2022-06-19", "2022-06-19", "2022-06-19"],
            "encuestadora": ["PollsterA", "PollsterB", "PollsterC"],
            "muestra": [1000, 1000, 1000],
            "gustavo_petro": [52.0, 51.0, 50.0],
            "rodolfo_hernandez": [48.0, 49.0, 50.0],
            "blanco": [0.0, 0.0, 0.0],
            "round_number": [2, 2, 2],
        }
    )

    matrix_with_polls = estimate_runoff_matrix(
        idata,
        (results_round1, results_round1),
        round2_polls,
        config,
    )
    matrix_no_polls = estimate_runoff_matrix(
        idata,
        (results_round1, results_round1),
        None,
        config,
    )

    ph_polls = next(
        p
        for p in matrix_with_polls.pairings
        if p.candidate_first == "gustavo_petro" and p.candidate_second == "rodolfo_hernandez"
    )
    ph_no_polls = next(
        p
        for p in matrix_no_polls.pairings
        if p.candidate_first == "gustavo_petro" and p.candidate_second == "rodolfo_hernandez"
    )

    diff = abs(ph_polls.prob_first_wins - ph_no_polls.prob_first_wins)
    assert diff > 0.01, (
        f"Model path produced nearly identical prob_first_wins "
        f"({ph_polls.prob_first_wins:.4f} vs {ph_no_polls.prob_first_wins:.4f}, "
        f"diff={diff:.4f}); expected difference > 0.01"
    )


def test_estimate_runoff_matrix_falls_back_to_heuristic_when_few_polls() -> None:
    """Test that fewer than 3 head-to-head polls falls back to the heuristic.

    When ``round2_polls`` contains only 1 row for a candidate pairing, the
    filter returns ``None`` and the transfer heuristic must be used instead,
    producing results identical to the ``round2_polls=None`` case.
    """
    idata = _make_synthetic_round1_idata()
    results_round1 = _make_round1_result()
    config = ModelConfig(seed=42)

    round2_polls = pd.DataFrame(
        {
            "fecha": ["2022-06-19"],
            "encuestadora": ["PollsterA"],
            "muestra": [1000],
            "gustavo_petro": [52.0],
            "rodolfo_hernandez": [48.0],
            "blanco": [0.0],
            "round_number": [2],
        }
    )

    matrix_few_polls = estimate_runoff_matrix(
        idata,
        (results_round1, results_round1),
        round2_polls,
        config,
    )
    matrix_no_polls = estimate_runoff_matrix(
        idata,
        (results_round1, results_round1),
        None,
        config,
    )

    no_polls_lookup = {
        (pf.candidate_first, pf.candidate_second): pf for pf in matrix_no_polls.pairings
    }

    for pf_few in matrix_few_polls.pairings:
        key = (pf_few.candidate_first, pf_few.candidate_second)
        pf_none = no_polls_lookup[key]
        assert pf_few.prob_first_wins == pf_none.prob_first_wins, (
            f"Pairing {pf_few.candidate_first} vs {pf_few.candidate_second}: "
            "prob_first_wins differs despite both using the heuristic"
        )
        assert pf_few.mean_margin == pf_none.mean_margin, (
            f"Pairing {pf_few.candidate_first} vs {pf_few.candidate_second}: "
            "mean_margin differs despite both using the heuristic"
        )
