"""Tests for SPEC-37 benchmark comparison logic."""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock, patch

from benchmarks.compositional_lme4 import run_2018_comparison
import numpy as np
import pandas as pd
import pytest

from co_president.config import (
    FIRST_ROUND_CANDIDATES,
    ModelConfig,
)
from co_president.data import CandidateResult, RoundResult
from co_president.fundamentals.features import HistoricalRecord
from co_president.model_municipal import build_municipal_model

# ═══════════════════════════════════════════════════════════════════════
# Synthetic data fixtures
# ═══════════════════════════════════════════════════════════════════════


def _make_synthetic_features(n_municipalities: int = 3) -> pd.DataFrame:
    """Return a synthetic feature DataFrame with *n_municipalities* rows."""
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
                        year=2014,
                        round=1,
                        left_candidate="gustavo_petro",
                        right_candidate="rodolfo_hernandez",
                        left_share=0.45,
                        right_share=0.30,
                        abstention_rate=0.40,
                    ),
                ),
                "historical_turnout_m": rng.uniform(0.40, 0.75),
            },
        )
    return pd.DataFrame(rows)


def _make_polls_to_2014() -> pd.DataFrame:
    """Return a 3-row synthetic poll DataFrame for pre-2014 polls."""
    return pd.DataFrame(
        {
            "fecha": ["2014-05-01", "2014-05-01", "2014-05-01"],
            "encuestadora": ["Invamer", "Invamer", "Invamer"],
            "muestra": [1000, 1000, 1000],
            "gustavo_petro": [46.0, 47.0, 45.0],
            "rodolfo_hernandez": [32.0, 31.0, 33.0],
            "blanco": [22.0, 22.0, 22.0],
            "round_number": [1, 1, 1],
        },
    )


def _make_2018_result() -> RoundResult:
    """Return a minimal RoundResult for 2018."""
    candidates = (
        CandidateResult("gustavo_petro", 8_000_000, 0.50),
        CandidateResult("rodolfo_hernandez", 6_000_000, 0.30),
        CandidateResult("blanco", 1_000_000, 0.20),
    )
    total = sum(c.votes for c in candidates)
    return RoundResult(
        round_number=1,
        date=pd.Timestamp("2018-06-17").date(),
        total_valid_votes=total,
        total_votes_incl_blank=total,
        registered_voters=39_000_000,
        polling_stations=12_500,
        candidates=candidates,
        blank_votes=1_000_000,
        null_votes=200_000,
        unmarked_votes=30_000,
    )


def _get_candidate_keys(polls: pd.DataFrame) -> list[str]:
    """Return sorted intersection of FIRST_ROUND_CANDIDATES and poll columns."""
    return sorted(set(FIRST_ROUND_CANDIDATES) & set(polls.columns))


# ═══════════════════════════════════════════════════════════════════════
# Tests for run_2018_comparison (model build only, no MCMC)
# ═══════════════════════════════════════════════════════════════════════


class TestRun2018Comparison:
    """Tests for run_2018_comparison."""

    @pytest.fixture
    def base_config(self) -> ModelConfig:
        """Return a base ModelConfig for benchmarking."""
        return ModelConfig(
            fundamentals_mode="prior_only",
            beta_coefficient_prior_sigma=0.5,
            sigma_m_prior=0.3,
            mcmc_draws=100,
            mcmc_tune=50,
            mcmc_chains=1,
            mcmc_cores=1,
        )

    @pytest.fixture
    def features(self) -> pd.DataFrame:
        """Return synthetic features."""
        return _make_synthetic_features()

    @pytest.fixture
    def polls(self) -> pd.DataFrame:
        """Return synthetic polls to 2014."""
        return _make_polls_to_2014()

    @pytest.fixture
    def results(self) -> RoundResult:
        """Return synthetic 2018 results."""
        return _make_2018_result()

    def test_models_build_with_2018_data(
        self,
        features: pd.DataFrame,
        polls: pd.DataFrame,
        base_config: ModelConfig,
    ) -> None:
        """Both baseline and CLR-target models build without error."""
        cfg_baseline = dataclasses.replace(base_config, clr_target=False)
        cfg_clr = dataclasses.replace(base_config, clr_target=True)

        model_baseline = build_municipal_model(
            features,
            polls,
            None,
            cfg_baseline,
            target_year=2014,
        )
        model_clr = build_municipal_model(
            features,
            polls,
            None,
            cfg_clr,
            target_year=2014,
        )

        assert model_baseline is not None
        assert model_clr is not None
        baseline_rvs = {rv.name for rv in model_baseline.free_RVs}
        clr_rvs = {rv.name for rv in model_clr.free_RVs}
        assert baseline_rvs == clr_rvs

    def test_clr_model_has_clr_deterministic(
        self,
        features: pd.DataFrame,
        polls: pd.DataFrame,
        base_config: ModelConfig,
    ) -> None:
        """CLR-target model includes ``p_municipal_clr`` deterministic."""
        cfg_clr = dataclasses.replace(base_config, clr_target=True)
        model_clr = build_municipal_model(
            features,
            polls,
            None,
            cfg_clr,
            target_year=2014,
        )

        det_names = {det.name for det in model_clr.deterministics}
        assert "p_municipal_clr" in det_names

    @patch("benchmarks.compositional_lme4.sample_municipal_model")
    def test_run_2018_comparison_returns_expected_keys(
        self,
        mock_sample: MagicMock,
        features: pd.DataFrame,
        polls: pd.DataFrame,
        results: RoundResult,
        base_config: ModelConfig,
    ) -> None:
        """Returned dict contains all expected metric keys."""
        n_candidates = 3
        candidate_keys = _get_candidate_keys(polls)
        actual_shares = np.array(
            [results.get_share(k) for k in candidate_keys],
        )
        mock_idata = _make_mock_idata(n_candidates, actual_shares)
        mock_sample.return_value = mock_idata

        result = run_2018_comparison(
            features,
            polls,
            results,
            base_config,
            candidate_keys,
        )

        expected_keys = {
            "standard_r2_baseline",
            "standard_r2_clr",
            "composite_r2_baseline",
            "composite_r2_clr",
            "standard_r2_delta",
            "composite_r2_delta",
        }
        assert set(result.keys()) == expected_keys
        for key in expected_keys:
            assert isinstance(result[key], float)

    @patch("benchmarks.compositional_lme4.sample_municipal_model")
    def test_identical_data_returns_zero_delta(
        self,
        mock_sample: MagicMock,
        features: pd.DataFrame,
        polls: pd.DataFrame,
        results: RoundResult,
        base_config: ModelConfig,
    ) -> None:
        """Both models return same predictions -> delta is 0."""
        n_candidates = 3
        candidate_keys = _get_candidate_keys(polls)
        actual_shares = np.array(
            [results.get_share(k) for k in candidate_keys],
        )
        mock_idata = _make_mock_idata(n_candidates, actual_shares)
        mock_sample.return_value = mock_idata

        result = run_2018_comparison(
            features,
            polls,
            results,
            base_config,
            candidate_keys,
        )

        assert result["standard_r2_delta"] == pytest.approx(0.0, abs=1e-10)
        # Composite R² is undefined for single-row (n=1): always NaN
        assert np.isnan(result["composite_r2_delta"])


# ═══════════════════════════════════════════════════════════════════════
# Mock helpers
# ═══════════════════════════════════════════════════════════════════════


def _make_mock_idata(
    n_candidates: int,
    actual_shares: np.ndarray | None = None,
) -> MagicMock:
    """Create a mock InferenceData with ``p_natl`` posterior.

    Args:
        n_candidates: Number of candidate categories.
        actual_shares: If provided, ``p_natl`` posterior mean matches
            these shares so that R² ≈ 1.0 (above gating threshold).
            If None, uniform ``1/n_candidates`` shares are used.

    """
    if actual_shares is not None:
        p_natl = np.full((1, 100, n_candidates), actual_shares)
    else:
        p_natl = np.full((1, 100, n_candidates), 1.0 / n_candidates)
    p_natl_dataarray = MagicMock()
    p_natl_dataarray.to_numpy.return_value = p_natl

    posterior_mock = MagicMock()
    posterior_mock.__getitem__.return_value = p_natl_dataarray

    idata_mock = MagicMock()
    idata_mock.posterior = posterior_mock
    return idata_mock
