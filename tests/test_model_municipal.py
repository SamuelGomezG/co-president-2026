"""Tests for the municipal hierarchical model (SPEC-22)."""

from __future__ import annotations

import arviz as az  # type: ignore[reportMissingTypeStubs]
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

from co_president.config import (
    FIRST_ROUND_CANDIDATES,
    ModelConfig,
)
from co_president.data import CandidateResult, RoundResult
from co_president.fundamentals.features import HistoricalRecord
from co_president.model_municipal import (
    build_municipal_model,
    compute_effective_num_municipalities,
    compute_effective_pop,
    sample_municipal_model,
)

__all__: list[str] = []


# ═══════════════════════════════════════════════════════════════════════
# Synthetic data helpers
# ═══════════════════════════════════════════════════════════════════════


def _make_synthetic_features(n_municipalities: int = 3) -> pd.DataFrame:
    """Return a synthetic municipal feature matrix.

    Creates *n_municipalities* rows with plausible but synthetic values.
    All columns required by ``build_municipal_model`` are present.
    """
    rng = np.random.default_rng(42)
    rows: list[dict] = []
    for i in range(n_municipalities):
        code = f"{10000 + i:05d}"
        rows.append(
            {
                "codigo_municipio": code,
                "nombre_municipio": f"Municipio_{i}",
                "comuna_nombre": None,
                "departamento": "Test",
                "region": "Test",
                "poblacion_total": int(rng.integers(50_000, 500_000)),
                "poblacion_afrocolombiana": int(rng.integers(1_000, 50_000)),
                "poblacion_indigena": int(rng.integers(500, 10_000)),
                "poblacion_rural_dispersa": int(rng.integers(1_000, 100_000)),
                "pct_afro_colombian": rng.uniform(0.01, 0.30),
                "pct_indigenous": rng.uniform(0.005, 0.15),
                "pct_rural_disperso": rng.uniform(0.05, 0.50),
                "years_schooling_promedio": rng.uniform(5.0, 12.0),
                "pct_school_attendance": rng.uniform(0.70, 0.95),
                "internet_access_rate": rng.uniform(0.20, 0.85),
                "labor_force_participation_rate": rng.uniform(0.50, 0.80),
                "pct_female": rng.uniform(0.48, 0.53),
                "rooms_per_household": rng.uniform(2.0, 5.0),
                "persons_per_household": rng.uniform(3.0, 5.0),
                "pct_age_18_29": rng.uniform(0.20, 0.35),
                "pct_age_30_54": rng.uniform(0.30, 0.45),
                "pct_age_55_plus": rng.uniform(0.15, 0.35),
                "nbi_rate": rng.uniform(0.05, 0.40),
                "nbi_urban": rng.uniform(0.03, 0.30),
                "nbi_rural": rng.uniform(0.10, 0.50),
                "ipm_2018": rng.uniform(0.05, 0.40),
                "ipm_2018_imputed": True,
                "ipm_2022": rng.uniform(0.05, 0.40),
                "ipm_2022_imputed": True,
                "pop_2018": int(rng.integers(50_000, 500_000)),
                "pop_2019": int(rng.integers(50_000, 500_000)),
                "pop_2020": int(rng.integers(50_000, 500_000)),
                "pop_2021": int(rng.integers(50_000, 500_000)),
                "pop_2022": int(rng.integers(50_000, 500_000)),
                "pop_2023": int(rng.integers(50_000, 500_000)),
                "pop_2024": int(rng.integers(50_000, 500_000)),
                "pop_2025": int(rng.integers(50_000, 500_000)),
                "pop_2026": int(rng.integers(50_000, 500_000)),
                "pct_ingresos_propios": rng.uniform(0.10, 0.80),
                "gastos_totales_per_capita": rng.uniform(1.0, 5.0),
                "transferencias_per_capita": rng.uniform(0.5, 3.0),
                "ingresos_tributarios_per_capita": rng.uniform(0.2, 2.0),
                "risk_level": rng.choice(["low", "medium", "high", "extreme"]),
                "is_pdet": bool(rng.integers(0, 2)),
                "armed_group_presence": bool(rng.integers(0, 2)),
                "coca_hectares": rng.uniform(0.0, 100.0),
                "high_risk_flag": bool(rng.integers(0, 2)),
                "historical": (
                    HistoricalRecord(
                        year=2022,
                        round=1,
                        left_candidate="gustavo_petro",
                        right_candidate="federico_gutierrez",
                        left_share=rng.uniform(0.30, 0.50),
                        right_share=rng.uniform(0.15, 0.35),
                        abstention_rate=rng.uniform(0.30, 0.55),
                    ),
                ),
                "historical_turnout_m": rng.uniform(0.40, 0.75),
            }
        )
    return pd.DataFrame(rows)


def _make_3row_polls() -> pd.DataFrame:
    """Return a 3-row synthetic poll DataFrame with 3 candidate columns."""
    return pd.DataFrame(
        {
            "fecha": ["2022-05-29", "2022-05-29", "2022-05-29"],
            "encuestadora": ["PollsterA", "PollsterB", "PollsterC"],
            "muestra": [1000, 1000, 1000],
            "gustavo_petro": [50.0, 51.0, 49.0],
            "rodolfo_hernandez": [30.0, 29.0, 31.0],
            "blanco": [20.0, 20.0, 20.0],
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


# ═══════════════════════════════════════════════════════════════════════
# compute_effective_pop tests
# ═══════════════════════════════════════════════════════════════════════


class TestComputeEffectivePop:
    """Tests for the turnout-weighting function."""

    def test_basic_weighting(self) -> None:
        """Verify turnout weighting adjusts populations correctly."""
        pop = np.array([100.0, 200.0])
        turnout = np.array([0.5, 0.8])
        result = compute_effective_pop(pop, turnout)
        expected = np.array([100 * 0.5 / 0.65, 200 * 0.8 / 0.65])
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_uniform_turnout(self) -> None:
        """When all turnout is equal, effective equals raw population."""
        pop = np.array([100.0, 200.0, 300.0])
        turnout = np.array([0.6, 0.6, 0.6])
        result = compute_effective_pop(pop, turnout)
        np.testing.assert_allclose(result, pop, rtol=1e-6)

    def test_zero_turnout_fallback(self) -> None:
        """When all turnout is zero, return raw population (no division by zero)."""
        pop = np.array([100.0, 200.0])
        turnout = np.array([0.0, 0.0])
        result = compute_effective_pop(pop, turnout)
        np.testing.assert_allclose(result, pop, rtol=1e-6)

    def test_handles_integer_input(self) -> None:
        """Integer population input produces float output."""
        pop = np.array([100, 200], dtype=int)
        turnout = np.array([0.5, 0.8])
        result = compute_effective_pop(pop, turnout)
        assert result.dtype == float


# ═══════════════════════════════════════════════════════════════════════
# Graph test (Phase 1 of 4-phase strategy)
# ═══════════════════════════════════════════════════════════════════════


def test_build_municipal_model_graph() -> None:
    """Test that the municipal model builds with correct graph structure.

    Expected free RVs: alpha, beta_historical, beta_ethnicity, beta_poverty,
        beta_rural, beta_education, beta_risk, sigma_m, mu_m_raw,
        sigma_house, phi_poll, raw_house (12 total).
    Expected deterministics: p_municipal, p_natl, phi_poll_n,
        house_effects, p_poll (5 total).
    Expected observed RVs: poll_likelihood (1 total).
    """
    features = _make_synthetic_features(n_municipalities=3)
    polls = _make_3row_polls()
    config = ModelConfig(
        beta_coefficient_prior_sigma=0.5,
        sigma_m_prior=0.3,
        house_effect_sigma_prior=1.0,
    )
    model = build_municipal_model(features, polls, None, config)

    # Expected free RV names
    expected_free_rv_names = {
        "alpha",
        "beta_historical",
        "beta_ethnicity",
        "beta_poverty",
        "beta_rural",
        "beta_education",
        "beta_risk",
        "sigma_m",
        "mu_m_raw",
        "sigma_house",
        "phi_poll",
        "raw_house",
    }
    free_rv_names = {rv.name for rv in model.free_RVs}
    assert free_rv_names == expected_free_rv_names, (
        f"Free RV mismatch.\nExpected: {expected_free_rv_names}\nGot:      {free_rv_names}"
    )

    # Expected deterministic names
    expected_det_names = {
        "p_municipal",
        "p_natl",
        "phi_poll_n",
        "house_effects",
        "p_poll",
    }
    det_names = {d.name for d in model.deterministics}
    assert det_names == expected_det_names, (
        f"Deterministic mismatch.\nExpected: {expected_det_names}\nGot:      {det_names}"
    )

    # Expected observed RVs
    assert len(model.observed_RVs) == 1, f"Expected 1 observed RV, got {len(model.observed_RVs)}"
    assert model.observed_RVs[0].name == "poll_likelihood"


def test_build_municipal_model_backtest_mode() -> None:
    """Test that backtest mode (results=RoundResult) includes election likelihood."""
    features = _make_synthetic_features(n_municipalities=3)
    polls = _make_3row_polls()
    results = _make_round1_result()
    config = ModelConfig()
    model = build_municipal_model(features, polls, results, config)

    assert len(model.observed_RVs) == 2
    assert {rv.name for rv in model.observed_RVs} == {
        "poll_likelihood",
        "election_likelihood",
    }
    det_names = {d.name for d in model.deterministics}
    assert "p_elec" in det_names


def test_build_municipal_model_requires_features() -> None:
    """Test that empty or insufficient features raise ValueError."""
    polls = _make_3row_polls()
    config = ModelConfig()

    with pytest.raises(ValueError, match="at least 2 municipalities"):
        build_municipal_model(
            _make_synthetic_features(n_municipalities=1),
            polls,
            None,
            config,
        )


# ═══════════════════════════════════════════════════════════════════════
# Horseshoe prior graph test
# ═══════════════════════════════════════════════════════════════════════


def test_build_municipal_model_graph_horseshoe() -> None:
    """Test that the municipal model builds with Horseshoe priors enabled.

    Expected free RVs must include ``tau_horseshoe`` and the ``_lam`` /
    ``_z`` variables from the Horseshoe parameterization, while still
    producing the same deterministic and observed structure.
    """
    features = _make_synthetic_features(n_municipalities=3)
    polls = _make_3row_polls()
    config = ModelConfig(
        use_horseshoe_prior=True,
        beta_coefficient_prior_sigma=0.5,
        sigma_m_prior=0.3,
        house_effect_sigma_prior=1.0,
    )
    model = build_municipal_model(features, polls, None, config)

    # Horseshoe introduces: tau_horseshoe, beta_*_lam (6 groups), beta_*_z (6 groups)
    # This replaces the 6 plain Normal(0, 0.5) betas with 13 new RVs.
    # Expected free RV names (Normal baseline + Horseshoe additions)
    baseline_names = {
        "alpha",
        "sigma_m",
        "mu_m_raw",
        "sigma_house",
        "phi_poll",
        "raw_house",
    }
    horseshoe_names = {
        "tau_horseshoe",
        "beta_historical_lam",
        "beta_historical_z",
        "beta_ethnicity_lam",
        "beta_ethnicity_z",
        "beta_poverty_lam",
        "beta_poverty_z",
        "beta_rural_lam",
        "beta_rural_z",
        "beta_education_lam",
        "beta_education_z",
        "beta_risk_lam",
        "beta_risk_z",
    }
    free_rv_names = {rv.name for rv in model.free_RVs}
    expected_free = baseline_names | horseshoe_names
    assert free_rv_names == expected_free, (
        f"Free RV mismatch.\nExpected: {expected_free}\nGot:      {free_rv_names}"
    )

    # Horseshoe adds 6 Deterministic betas (beta_historical, etc.)
    expected_det_names = {
        "p_municipal",
        "p_natl",
        "phi_poll_n",
        "house_effects",
        "p_poll",
    }
    horseshoe_det_names = {
        "beta_historical",
        "beta_ethnicity",
        "beta_poverty",
        "beta_rural",
        "beta_education",
        "beta_risk",
    }
    det_names = {d.name for d in model.deterministics}
    assert det_names == expected_det_names | horseshoe_det_names, (
        f"Deterministic mismatch.\nExpected: {expected_det_names | horseshoe_det_names}\n"
        f"Got:      {det_names}"
    )

    assert len(model.observed_RVs) == 1
    assert model.observed_RVs[0].name == "poll_likelihood"


def test_build_municipal_model_graph_clr_target() -> None:
    """Test that clr_target=True adds p_municipal_clr deterministic."""
    features = _make_synthetic_features(n_municipalities=3)
    polls = _make_3row_polls()
    config = ModelConfig(
        beta_coefficient_prior_sigma=0.5,
        sigma_m_prior=0.3,
        house_effect_sigma_prior=1.0,
        clr_target=True,
    )
    model = build_municipal_model(features, polls, None, config)

    det_names = {d.name for d in model.deterministics}
    assert "p_municipal_clr" in det_names, "p_municipal_clr not found with clr_target=True"

    expected_free_rv_names = {
        "alpha",
        "beta_historical",
        "beta_ethnicity",
        "beta_poverty",
        "beta_rural",
        "beta_education",
        "beta_risk",
        "sigma_m",
        "mu_m_raw",
        "sigma_house",
        "phi_poll",
        "raw_house",
    }
    free_rv_names = {rv.name for rv in model.free_RVs}
    assert free_rv_names == expected_free_rv_names, (
        f"Free RV mismatch.\nExpected: {expected_free_rv_names}\nGot:      {free_rv_names}"
    )

    expected_det_names = {
        "p_municipal",
        "p_municipal_clr",
        "p_natl",
        "phi_poll_n",
        "house_effects",
        "p_poll",
    }
    det_names = {d.name for d in model.deterministics}
    assert det_names == expected_det_names, (
        f"Deterministic mismatch.\nExpected: {expected_det_names}\nGot:      {det_names}"
    )

    assert len(model.observed_RVs) == 1
    assert model.observed_RVs[0].name == "poll_likelihood"


# ═══════════════════════════════════════════════════════════════════════
# Effective municipalities test
# ═══════════════════════════════════════════════════════════════════════


class TestComputeEffectiveNumMunicipalities:
    """Tests for the shrinkage-based effective municipality count."""

    def test_perfect_shrinkage(self) -> None:
        """When posterior variance is zero, all 3 municipalities are effective."""
        idata = _make_dummy_idata(n_muni=3)
        eff = compute_effective_num_municipalities(idata, pool_alpha=0.95)
        assert eff == 3

    def test_no_shrinkage(self) -> None:
        """When posterior equals prior variance, effective count is 0."""
        idata = _make_dummy_idata(n_muni=3, var_mult=0.95**2)
        eff = compute_effective_num_municipalities(idata, pool_alpha=0.95)
        assert eff == 0

    def test_partial_shrinkage(self) -> None:
        """When posterior variance is half of prior, effective count is ~1.5 (rounded 1 or 2)."""
        idata = _make_dummy_idata(n_muni=3, var_mult=(0.95**2) * 0.5)
        eff = compute_effective_num_municipalities(idata, pool_alpha=0.95)
        # shrinkage = 0.5 → sum = 1.5 → np.round(1.5) = 2 (banker's rounding)
        # Sampling noise in variance estimate can shift result to 1 or 2
        assert eff in (1, 2)


def _make_dummy_idata(
    n_muni: int = 3,
    var_mult: float = 0.0,
    n_chains: int = 2,
    n_draws: int = 100,
) -> xr.DataTree:
    """Create a dummy ``xr.DataTree`` with a ``mu_m_raw`` posterior.

    Sets ``mu_m_raw`` to zero-centered noise with variance
    *var_mult* (relative to pool_alpha^2).  When *var_mult* = 0 (the
    default), the posterior variance is 0 and shrinkage is perfect.

    Args:
        n_muni: Number of municipalities.
        var_mult: Multiplier for the posterior variance relative to
            ``pool_alpha**2``.
        n_chains: Number of MCMC chains.
        n_draws: Number of draws per chain.

    Returns:
        Dummy ``xr.DataTree`` with only ``mu_m_raw`` in the posterior.

    """
    rng = np.random.default_rng(42)
    data = rng.normal(
        0,
        np.sqrt(var_mult),
        size=(n_chains, n_draws, n_muni, 3),
    ).astype(np.float32)

    posterior = xr.Dataset(
        {
            "mu_m_raw": xr.DataArray(
                data,
                dims=("chain", "draw", "mu_m_raw_dim_0", "mu_m_raw_dim_1"),
            ),
        }
    )
    dt = xr.DataTree()
    dt["posterior"] = posterior
    return dt


# ═══════════════════════════════════════════════════════════════════════
# Prior predictive test (Phase 2 of 4-phase strategy)
# ═══════════════════════════════════════════════════════════════════════


def test_build_municipal_model_prior_predictive() -> None:
    """Test that prior predictive samples produce valid shares in [0, 1].

    Verifies that:
    - ``p_municipal`` is in [0, 1] for all municipalities and candidates.
    - ``p_natl`` (national rollup) is in [0, 1] and sums to 1.
    - ``p_poll`` (house-effect-adjusted) is in [0, 1].
    """
    features = _make_synthetic_features(n_municipalities=3)
    polls = _make_3row_polls()
    config = ModelConfig(
        beta_coefficient_prior_sigma=0.5,
        sigma_m_prior=0.3,
        house_effect_sigma_prior=1.0,
    )
    model = build_municipal_model(features, polls, None, config)

    with model:
        prior_pred = pm.sample_prior_predictive(draws=100, random_seed=config.seed)

    # Municipal-level vote shares (chain, draw, municipality, candidate)
    p_municipal = prior_pred.prior["p_municipal"]
    assert p_municipal.min() >= 0.0
    assert p_municipal.max() <= 1.0
    # Sum to 1 across candidates for each municipality, draw, chain
    np.testing.assert_allclose(
        p_municipal.sum(axis=-1),
        1.0,
        atol=1e-6,
    )

    # National rollup (chain, draw, candidate)
    p_natl = prior_pred.prior["p_natl"]
    assert p_natl.min() >= 0.0
    assert p_natl.max() <= 1.0
    np.testing.assert_allclose(
        p_natl.sum(axis=-1),
        1.0,
        atol=1e-6,
    )

    # House-effect-adjusted poll shares (chain, draw, poll, candidate)
    p_poll = prior_pred.prior["p_poll"]
    assert p_poll.min() >= 0.0
    assert p_poll.max() <= 1.0
    np.testing.assert_allclose(
        p_poll.sum(axis=-1),
        1.0,
        atol=1e-6,
    )


def test_build_municipal_model_clr_target_prior_predictive() -> None:
    """Test that clr_target=True produces valid CLR-transformed shares.

    Verifies that p_municipal_clr rows sum to ~0 (CLR property).
    """
    features = _make_synthetic_features(n_municipalities=3)
    polls = _make_3row_polls()
    config = ModelConfig(
        beta_coefficient_prior_sigma=0.5,
        sigma_m_prior=0.3,
        house_effect_sigma_prior=1.0,
        clr_target=True,
    )
    model = build_municipal_model(features, polls, None, config)

    with model:
        prior_pred = pm.sample_prior_predictive(draws=50, random_seed=config.seed)

    p_muni_clr = prior_pred.prior["p_municipal_clr"]
    np.testing.assert_allclose(
        p_muni_clr.sum(axis=-1),
        0.0,
        atol=1e-6,
    )

    p_municipal = prior_pred.prior["p_municipal"]
    assert p_municipal.min() >= 0.0
    assert p_municipal.max() <= 1.0
    np.testing.assert_allclose(
        p_municipal.sum(axis=-1),
        1.0,
        atol=1e-6,
    )


# ═══════════════════════════════════════════════════════════════════════
# Convergence test (Phase 3 of 4-phase strategy, slow)
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.slow
def test_sample_municipal_convergence() -> None:
    """Test MCMC convergence on minimal 3-municipality, 3-poll data.

    Asserts R-hat < 1.10 for all sampled parameters (Gelman-Rubin
    convergence criterion).
    """
    features = _make_synthetic_features(n_municipalities=3)
    polls = pd.DataFrame(
        {
            "fecha": [
                "2022-05-29",
                "2022-05-29",
                "2022-05-29",
            ],
            "encuestadora": [
                "PollsterA",
                "PollsterB",
                "PollsterC",
            ],
            "muestra": [1000, 1200, 800],
            "gustavo_petro": [50.0, 51.0, 49.0],
            "rodolfo_hernandez": [30.0, 29.0, 31.0],
            "blanco": [20.0, 20.0, 20.0],
            "round_number": [1, 1, 1],
        }
    )
    config = ModelConfig(
        beta_coefficient_prior_sigma=0.3,
        sigma_m_prior=0.2,
        house_effect_sigma_prior=0.5,
        mcmc_draws=500,
        mcmc_tune=500,
        mcmc_chains=2,
        mcmc_cores=2,
    )
    model = build_municipal_model(features, polls, None, config)
    idata = sample_municipal_model(model, config)

    summary = az.summary(
        idata,
        var_names=[
            "~p_municipal",
            "~p_natl",
            "~p_poll",
            "~house_effects",
            "~raw_house",
            "~phi_poll_n",
        ],
    )
    r_hat = pd.to_numeric(summary["r_hat"], errors="coerce").dropna()
    assert not r_hat.empty, "r_hat is empty; no parameters to evaluate"
    assert (r_hat < 1.10).all(), (
        f"R-hat convergence failure: max r_hat = {r_hat.max():.4f}, "
        f"parameters with r_hat >= 1.10: {list(r_hat[r_hat >= 1.10].index)}"
    )


# ═══════════════════════════════════════════════════════════════════════
# Sanity test (Phase 4 of 4-phase strategy, slow)
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.slow
def test_sample_municipal_sanity() -> None:
    """Test posterior accuracy on minimal synthetic data with known ground truth.

    3 polls at election day (3 pollsters).  Ground truth at national level:
    Petro=50%, Hernandez=30%, Blanco=20%.  Posterior national mean at
    election day must be within ±5 percentage points of the known truth
    (SPEC-22 tolerance, matching SPEC-06 §9.3).
    """
    features = _make_synthetic_features(n_municipalities=3)
    polls = pd.DataFrame(
        {
            "fecha": [
                "2022-05-29",
                "2022-05-29",
                "2022-05-29",
            ],
            "encuestadora": [
                "PollsterA",
                "PollsterB",
                "PollsterC",
            ],
            "muestra": [1200, 1000, 800],
            "gustavo_petro": [50.0, 51.0, 49.0],
            "rodolfo_hernandez": [30.0, 29.0, 31.0],
            "blanco": [20.0, 20.0, 20.0],
            "round_number": [1, 1, 1],
        }
    )
    config = ModelConfig(
        beta_coefficient_prior_sigma=0.3,
        sigma_m_prior=0.2,
        house_effect_sigma_prior=0.5,
        mcmc_draws=1000,
        mcmc_tune=1000,
        mcmc_chains=2,
        mcmc_cores=2,
    )
    model = build_municipal_model(features, polls, None, config)
    idata = sample_municipal_model(model, config)

    candidate_keys = sorted(set(FIRST_ROUND_CANDIDATES) & set(polls.columns))
    candidate_idx = {k: i for i, k in enumerate(candidate_keys)}

    summary = az.summary(
        idata,
        var_names=[
            "~p_municipal",
            "~p_natl",
            "~p_poll",
            "~house_effects",
            "~raw_house",
            "~phi_poll_n",
        ],
    )
    r_hat = pd.to_numeric(summary["r_hat"], errors="coerce").dropna()
    assert (r_hat < 1.10).all(), f"R-hat convergence failure: max r_hat = {r_hat.max():.4f}"

    p_natl = idata.posterior["p_natl"].to_numpy()
    means = p_natl.mean(axis=(0, 1))

    petro_mean = means[candidate_idx["gustavo_petro"]]
    hernandez_mean = means[candidate_idx["rodolfo_hernandez"]]
    blanco_mean = means[candidate_idx["blanco"]]

    assert 0.45 <= petro_mean <= 0.55, f"Petro posterior mean {petro_mean:.3f} outside [0.45, 0.55]"
    assert 0.25 <= hernandez_mean <= 0.35, (
        f"Hernandez posterior mean {hernandez_mean:.3f} outside [0.25, 0.35]"
    )
    assert 0.15 <= blanco_mean <= 0.25, (
        f"Blanco posterior mean {blanco_mean:.3f} outside [0.15, 0.25]"
    )
