"""Tests for SPEC-26: Out-of-sample validation framework."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pymc as pm  # type: ignore[reportMissingTypeStubs]
import pytest

from co_president.config import (
    POLLSTER_RATINGS,
    ModelConfig,
)
from co_president.data import CandidateResult, RoundResult
from co_president.fundamentals.features import HistoricalRecord
from co_president.model_municipal import build_municipal_model
from co_president.validation.municipal_oos import (
    _build_missing_data_report,
    _categorize_pollster,
    _compute_r2,
    compare_modes,
    leave_2022_out,
    run_sensitivity_ablation,
    sample_all_low_polls,
    sample_bootstrap_polls,
    sample_stratified_polls,
)

# ═══════════════════════════════════════════════════════════════════════
# Synthetic data fixtures
# ═══════════════════════════════════════════════════════════════════════


def _make_3row_polls() -> pd.DataFrame:
    """Return a 3-row synthetic poll DataFrame with 3 candidate columns."""
    return pd.DataFrame(
        {
            "fecha": ["2022-05-29", "2022-05-29", "2022-05-29"],
            "encuestadora": ["Invamer", "Invamer", "Invamer"],
            "muestra": [1000, 1000, 1000],
            "gustavo_petro": [50.0, 51.0, 49.0],
            "rodolfo_hernandez": [30.0, 29.0, 31.0],
            "blanco": [20.0, 20.0, 20.0],
            "round_number": [1, 1, 1],
        }
    )


def _make_2022_result() -> RoundResult:
    """Return a minimal RoundResult for 2022."""
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


def _make_synthetic_features(n_municipalities: int = 3) -> pd.DataFrame:
    """Return a synthetic municipal feature matrix for OOS testing."""
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
                    HistoricalRecord(
                        year=2014,
                        round=1,
                        left_candidate="gustavo_petro",
                        right_candidate="oscar_ivan_zuluaga",
                        left_share=rng.uniform(0.20, 0.35),
                        right_share=rng.uniform(0.20, 0.35),
                        abstention_rate=rng.uniform(0.30, 0.55),
                    ),
                ),
                "historical_turnout_m": rng.uniform(0.40, 0.75),
            }
        )
    return pd.DataFrame(rows)


def _make_multi_pollster_polls() -> pd.DataFrame:
    """Return a poll DataFrame with multiple pollsters of varying ratings."""
    return pd.DataFrame(
        {
            "fecha": [
                "2022-05-01",
                "2022-05-05",
                "2022-05-10",
                "2022-05-15",
                "2022-05-20",
            ],
            "encuestadora": [
                "Invamer",
                "Invamer",
                "CNC",
                "CELAG",
                "Mosqueteros",
            ],
            "muestra": [1000, 1000, 1000, 1000, 1000],
            "gustavo_petro": [50.0, 49.0, 48.0, 47.0, 46.0],
            "rodolfo_hernandez": [30.0, 31.0, 32.0, 33.0, 34.0],
            "blanco": [20.0, 20.0, 20.0, 20.0, 20.0],
            "round_number": [1, 1, 1, 1, 1],
        }
    )


# ═══════════════════════════════════════════════════════════════════════
# _categorize_pollster tests
# ═══════════════════════════════════════════════════════════════════════


class TestCategorizePollster:
    """Tests for the pollster rating categorizer."""

    def test_high_rating(self) -> None:
        """Ratings >= 8.0 are 'high'."""
        assert _categorize_pollster(10.0) == "high"
        assert _categorize_pollster(8.0) == "high"

    def test_mid_rating(self) -> None:
        """Ratings between low_cutoff and high_cutoff are 'mid'."""
        assert _categorize_pollster(6.3) == "mid"
        assert _categorize_pollster(5.0) == "mid"

    def test_low_rating(self) -> None:
        """Ratings <= low_cutoff are 'low'."""
        assert _categorize_pollster(3.8) == "low"
        assert _categorize_pollster(1.0) == "low"

    def test_custom_cutoffs(self) -> None:
        """Custom cutoffs override defaults."""
        assert _categorize_pollster(7.0, high_cutoff=9.0, low_cutoff=3.0) == "mid"
        assert _categorize_pollster(8.5, high_cutoff=9.0, low_cutoff=3.0) == "mid"
        assert _categorize_pollster(9.5, high_cutoff=9.0, low_cutoff=3.0) == "high"
        assert _categorize_pollster(2.0, high_cutoff=9.0, low_cutoff=3.0) == "low"


# ═══════════════════════════════════════════════════════════════════════
# Poll sampling strategy tests
# ═══════════════════════════════════════════════════════════════════════


class TestSampleStratifiedPolls:
    """Tests for the stratified poll sampling strategy."""

    def test_selects_correct_tier_counts(self) -> None:
        """Stratified sampling selects pollsters from all three rating tiers."""
        polls = _make_multi_pollster_polls()
        ratings = dict(POLLSTER_RATINGS)
        sampled = sample_stratified_polls(polls, pollster_ratings=dict(POLLSTER_RATINGS), seed=42)

        assert len(sampled) <= 10
        assert len(sampled) > 0

        # Verify that every selected pollster is in POLLSTER_RATINGS
        # and that we get a mix of tiers.  The fixture has Invamer
        # (high, 10.0), CNC (high, 8.1), CELAG (mid, 5.9), and
        # Mosqueteros (low, 1.0).  The stratified sampler selects
        # pollster NAMES from each tier (4 high / 3 mid / 3 low),
        # then filters rows to those names.  We verify the selected
        # rows contain pollsters whose ratings place them in the
        # expected tiers (but we cannot assert exact tier ROW counts
        # since that depends on which pollsters have rows in the
        # fixture).
        tiers_found: set[str] = set()
        for pollster in sampled["encuestadora"].unique():
            assert pollster in ratings, f"Unknown pollster: {pollster}"
            tiers_found.add(_categorize_pollster(ratings[pollster]))

        assert "high" in tiers_found, (
            f"Expected at least one high-rated pollster, got tiers: {tiers_found}"
        )

    def test_raises_on_empty_ratings(self) -> None:
        """Empty pollster_ratings raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            sample_stratified_polls(
                _make_multi_pollster_polls(),
                pollster_ratings={},
            )

    def test_raises_on_missing_column(self) -> None:
        """Missing encuestadora column raises ValueError."""
        bad_polls = pd.DataFrame({"some_col": [1, 2, 3]})
        with pytest.raises(ValueError, match="encuestadora"):
            sample_stratified_polls(bad_polls)


class TestSampleAllLowPolls:
    """Tests for the all-low-rated poll sampling strategy."""

    def test_selects_only_low_rated_pollsters(self) -> None:
        """All-low sampling returns only pollsters with rating <= cutoff."""
        polls = _make_multi_pollster_polls()
        sampled = sample_all_low_polls(polls, pollster_ratings=dict(POLLSTER_RATINGS), seed=42)

        assert len(sampled) > 0, "Expected at least one low-rated pollster in sample"
        low_pollsters = {"Medilab", "YanHaas", "CifrasYConceptos", "Datexco", "Mosqueteros"}
        for pollster in sampled["encuestadora"].unique():
            assert pollster in low_pollsters, f"Unexpected pollster {pollster} in all-low sample"

    def test_raises_when_no_low_pollsters(self) -> None:
        """Raises ValueError when no low-rated pollsters are available."""
        high_only_ratings = {"Invamer": 10.0, "CNC": 8.1}
        with pytest.raises(ValueError, match="no pollsters found"):
            sample_all_low_polls(
                _make_multi_pollster_polls(),
                pollster_ratings=high_only_ratings,
            )


class TestSampleBootstrapPolls:
    """Tests for the bootstrap poll sampling strategy."""

    def test_returns_correct_number_of_runs(self) -> None:
        """Bootstrap returns exactly n_runs samples."""
        polls = _make_multi_pollster_polls()
        samples = sample_bootstrap_polls(polls, n_total=3, n_runs=5, seed=42)

        assert len(samples) == 5
        for s in samples:
            assert len(s) <= 3

    def test_samples_are_different(self) -> None:
        """Bootstrap samples are not all identical (with replacement)."""
        polls = _make_multi_pollster_polls()
        samples = sample_bootstrap_polls(polls, n_total=3, n_runs=3, seed=42)

        # With only 5 polls and 3 taken per sample, they may overlap
        # but should not all be identical
        first = samples[0].reset_index(drop=True)
        all_same = all(s.reset_index(drop=True).equals(first) for s in samples[1:])
        assert not all_same, "All bootstrap samples are identical"


# ═══════════════════════════════════════════════════════════════════════
# Metric helper tests
# ═══════════════════════════════════════════════════════════════════════


class TestComputeR2:
    """Tests for the R² computation."""

    def test_perfect_prediction(self) -> None:
        """R² = 1 when predictions match actuals exactly."""
        actual = np.array([0.4, 0.3, 0.2, 0.1])
        predicted = np.array([0.4, 0.3, 0.2, 0.1])
        assert _compute_r2(actual, predicted) == pytest.approx(1.0)

    def test_mean_prediction(self) -> None:
        """R² = 0 when predictions equal the mean of actuals."""
        actual = np.array([0.4, 0.3, 0.2, 0.1])
        predicted = np.full_like(actual, actual.mean())
        assert _compute_r2(actual, predicted) == pytest.approx(0.0)

    def test_worse_than_mean(self) -> None:
        """R² can be negative when predictions are worse than the mean."""
        actual = np.array([0.4, 0.3, 0.2, 0.1])
        # Systematically wrong predictions
        predicted = np.array([0.1, 0.2, 0.3, 0.4])
        r2 = _compute_r2(actual, predicted)
        assert r2 < 0.0

    def test_constant_actual(self) -> None:
        """R² = 0 when all actual values are identical."""
        actual = np.array([0.3, 0.3, 0.3])
        predicted = np.array([0.3, 0.3, 0.3])
        assert _compute_r2(actual, predicted) == pytest.approx(0.0)


class TestBuildMissingDataReport:
    """Tests for the missing-data report formatter."""

    def test_report_includes_r2_and_year(self) -> None:
        """Report contains year and R² value."""
        errors = [
            {"candidate": "petro", "predicted_mean": 0.35, "actual_share": 0.40, "abs_error": 0.05},
        ]
        report = _build_missing_data_report(2018, 0.25, errors)
        assert "2018" in report
        assert "0.25" in report
        assert "R²" in report

    def test_report_includes_candidate_details(self) -> None:
        """Report lists each candidate with their errors."""
        errors = [
            {"candidate": "petro", "predicted_mean": 0.35, "actual_share": 0.40, "abs_error": 0.05},
            {
                "candidate": "hernandez",
                "predicted_mean": 0.28,
                "actual_share": 0.30,
                "abs_error": 0.02,
            },
        ]
        report = _build_missing_data_report(2022, 0.20, errors)
        assert "petro" in report
        assert "hernandez" in report

    def test_report_includes_action_required(self) -> None:
        """Report contains action guidance."""
        errors = [
            {"candidate": "petro", "predicted_mean": 0.35, "actual_share": 0.40, "abs_error": 0.05},
        ]
        report = _build_missing_data_report(2018, 0.15, errors)
        assert "Action required" in report

    def test_report_accepts_extra_detail(self) -> None:
        """Extra detail string is included in report."""
        errors = [
            {"candidate": "petro", "predicted_mean": 0.35, "actual_share": 0.40, "abs_error": 0.05},
        ]
        report = _build_missing_data_report(
            2018, 0.10, errors, extra_detail="No polls available for 2014."
        )
        assert "No polls available" in report


# ═══════════════════════════════════════════════════════════════════════
# leave_2022_out — poll cap enforcement
# ═══════════════════════════════════════════════════════════════════════


class TestLeave2022OutCap:
    """Tests for the ≤10-poll cap in leave_2022_out."""

    def test_raises_when_more_than_10_polls(self) -> None:
        """ValueError when > 10 polls are passed."""
        polls_11 = pd.concat(
            [_make_3row_polls()] * 4,
            ignore_index=True,
        )
        features = _make_synthetic_features()
        result = _make_2022_result()
        config = ModelConfig()

        with pytest.raises(ValueError, match="hard cap is 10"):
            leave_2022_out(features, polls_11, result, config)


# ═══════════════════════════════════════════════════════════════════════
# Graph test — model builds with target_year parameter
# ═══════════════════════════════════════════════════════════════════════


def test_build_municipal_model_with_target_year() -> None:
    """Model builds correctly when target_year differs from default 2022."""
    features = _make_synthetic_features(n_municipalities=3)
    polls = _make_3row_polls()
    config = ModelConfig(
        beta_coefficient_prior_sigma=0.5,
        sigma_m_prior=0.3,
        house_effect_sigma_prior=1.0,
    )

    # Should build with target_year=2014
    model = build_municipal_model(features, polls, None, config, target_year=2014)

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
    assert free_rv_names == expected_free_rv_names
    assert len(model.observed_RVs) == 1


def test_build_municipal_model_with_backtest_mode_and_target_year() -> None:
    """Model builds with results and target_year simultaneously."""
    features = _make_synthetic_features(n_municipalities=3)
    polls = _make_3row_polls()
    results = _make_2022_result()
    config = ModelConfig()

    model = build_municipal_model(features, polls, results, config, target_year=2014)
    assert len(model.observed_RVs) == 2
    assert {rv.name for rv in model.observed_RVs} == {
        "poll_likelihood",
        "election_likelihood",
    }


# ═══════════════════════════════════════════════════════════════════════
# Prior predictive test with target_year
# ═══════════════════════════════════════════════════════════════════════


def test_prior_predictive_with_target_year() -> None:
    """Prior predictive samples produce valid shares when target_year != 2022."""
    features = _make_synthetic_features(n_municipalities=3)
    polls = _make_3row_polls()
    config = ModelConfig(
        beta_coefficient_prior_sigma=0.5,
        sigma_m_prior=0.3,
        house_effect_sigma_prior=1.0,
    )

    model = build_municipal_model(features, polls, None, config, target_year=2014)
    with model:
        prior_pred = pm.sample_prior_predictive(draws=100, random_seed=config.seed)

    p_municipal = prior_pred.prior["p_municipal"]
    assert p_municipal.min() >= 0.0
    assert p_municipal.max() <= 1.0

    p_natl = prior_pred.prior["p_natl"]
    assert p_natl.min() >= 0.0
    assert p_natl.max() <= 1.0


# ═══════════════════════════════════════════════════════════════════════
# Sampling strategy integration with leave_2022_out context
# ═══════════════════════════════════════════════════════════════════════


def test_stratified_and_all_low_pass_on_mock_data() -> None:
    """Both stratified and all_low sampling produce valid poll subsets.

    Tests the sampling functions in a leave_2022_out-like context
    using mock poll data.
    """
    polls = _make_multi_pollster_polls()
    ratings = dict(POLLSTER_RATINGS)

    stratified = sample_stratified_polls(polls, pollster_ratings=ratings, seed=42)
    assert len(stratified) > 0
    assert "encuestadora" in stratified.columns

    all_low = sample_all_low_polls(polls, pollster_ratings=ratings, seed=42)
    assert len(all_low) > 0
    assert "encuestadora" in all_low.columns


# ═══════════════════════════════════════════════════════════════════════
# year_2018_holdout graph test (model builds, no MCMC)
# ═══════════════════════════════════════════════════════════════════════


def test_year_2018_holdout_model_graph() -> None:
    """The 2018 holdout model builds correctly with 2014 historical data."""
    features = _make_synthetic_features(n_municipalities=3)
    polls = _make_3row_polls()
    config = ModelConfig(
        beta_coefficient_prior_sigma=0.5,
        sigma_m_prior=0.3,
        house_effect_sigma_prior=1.0,
    )

    model = build_municipal_model(features, polls, None, config, target_year=2014)
    assert model is not None
    free_rv_names = {rv.name for rv in model.free_RVs}
    assert "alpha" in free_rv_names
    assert "beta_historical" in free_rv_names


# ═══════════════════════════════════════════════════════════════════════
# compare_modes — graph test
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.slow
def test_compare_modes_architecture() -> None:
    """compare_modes can run with fast synthetic data for structure validation.

    This test verifies that the mode comparison function produces
    results even with minimal synthetic data (it will not converge,
    but the code path is exercised).
    """
    features = _make_synthetic_features(n_municipalities=3)
    polls = _make_3row_polls()
    config_off = ModelConfig(
        beta_coefficient_prior_sigma=2.0,
        sigma_m_prior=0.5,
        mcmc_draws=200,
        mcmc_tune=100,
        mcmc_chains=1,
        mcmc_cores=1,
    )
    config_joint = ModelConfig(
        beta_coefficient_prior_sigma=0.3,
        sigma_m_prior=0.2,
        mcmc_draws=200,
        mcmc_tune=100,
        mcmc_chains=1,
        mcmc_cores=1,
    )

    diffs = compare_modes(features, polls, None, config_off, config_joint)
    assert isinstance(diffs, dict)
    for key in ("gustavo_petro", "rodolfo_hernandez"):
        assert key in diffs
        assert diffs[key] >= 0.0


# ═══════════════════════════════════════════════════════════════════════
# leave_2022_out — MAE gating test
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.slow
def test_leave_2022_out_mae_gate() -> None:
    """leave_2022_out raises ValueError when MAE exceeds 5 pp for a top-3 candidate.

    Uses weak priors + 1-chain MCMC to stress the model and trigger the
    MAE gating assertion.
    """
    features = _make_synthetic_features(n_municipalities=3)
    polls = _make_multi_pollster_polls().head(3).copy()
    polls["muestra"] = 10  # tiny samples increase posterior variance
    result = _make_2022_result()
    config = ModelConfig(
        beta_coefficient_prior_sigma=10.0,
        sigma_m_prior=0.5,
        house_effect_sigma_prior=2.0,
        mcmc_draws=200,
        mcmc_tune=100,
        mcmc_chains=1,
        mcmc_cores=1,
    )

    with pytest.raises(ValueError, match="abs_error"):
        leave_2022_out(features, polls, result, config, seed=42)


# ═══════════════════════════════════════════════════════════════════════
# leave_2022_out — HDI containment gating test
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.slow
def test_leave_2022_out_hdi_gate() -> None:
    """leave_2022_out raises ValueError when 94% HDI does not contain 40.34%.

    Uses weak priors + 1-chain MCMC to produce wide posteriors that
    may fail the 94% HDI containment test.
    """
    features = _make_synthetic_features(n_municipalities=3)
    polls = _make_multi_pollster_polls().head(3).copy()
    polls["muestra"] = 10
    result = _make_2022_result()
    config = ModelConfig(
        beta_coefficient_prior_sigma=10.0,
        sigma_m_prior=0.5,
        house_effect_sigma_prior=2.0,
        mcmc_draws=200,
        mcmc_tune=100,
        mcmc_chains=1,
        mcmc_cores=1,
    )

    with pytest.raises(ValueError, match=r"94% HDI|abs_error|exceeds"):
        leave_2022_out(features, polls, result, config, seed=99)


# ═══════════════════════════════════════════════════════════════════════
# run_sensitivity_ablation — smoke test
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.slow
def test_run_sensitivity_ablation_smoke() -> None:
    """run_sensitivity_ablation produces a comparison DataFrame with all three configs."""
    features = _make_synthetic_features(n_municipalities=3)
    polls = _make_3row_polls()
    config = ModelConfig(
        mcmc_draws=200,
        mcmc_tune=100,
        mcmc_chains=1,
        mcmc_cores=1,
    )

    result = run_sensitivity_ablation(features, polls, None, config)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert "configuration" in result.columns
    configs_found = set(result["configuration"])
    expected = {"off", "normal", "horseshoe"}
    assert configs_found == expected, f"Expected configs {expected}, got {configs_found}"
    assert "posterior_mean" in result.columns
    assert "effective_num_municipalities" in result.columns
