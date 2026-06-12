"""SPEC-02: Tests for configuration module — candidates, coalitions, hyperparameters."""

from __future__ import annotations

import importlib
import math
from typing import ClassVar

import pytest

from co_president.config import (
    COALITION_TO_CANDIDATE,
    CONSULTATION_KEY_MAP,
    CONSULTATION_VOTES,
    FIRST_ROUND_CANDIDATES,
    POLLSTER_RATINGS,
    TRANSFER_BLANCO_SPLIT,
    TRANSFER_FAJARDO_HERNANDEZ,
    TRANSFER_FAJARDO_PETRO,
    TRANSFER_GUTIERREZ_HERNANDEZ,
    TRANSFER_GUTIERREZ_PETRO,
    Candidate,
    ModelConfig,
    consultation_log_share_prior,
    get_active_candidates,
    get_candidate_column_map,
    get_default_pollster_weight,
    pollster_weight_formula,
)


def test_project_exists() -> None:
    """Verify the package can be imported and exposes a version string."""
    module = importlib.import_module("co_president")
    assert isinstance(getattr(module, "__version__", ""), str)
    assert module.__version__


class TestCandidate:
    """Tests for the Candidate frozen dataclass."""

    def test_instantiation(self) -> None:
        """Verify Candidate creates with all fields."""
        candidate = Candidate(
            key="gustavo_petro",
            display_name="Gustavo Petro",
            coalition="Pacto Histórico",
            first_round=True,
            runoff=True,
        )
        assert candidate.key == "gustavo_petro"
        assert candidate.display_name == "Gustavo Petro"
        assert candidate.coalition == "Pacto Histórico"
        assert candidate.first_round is True
        assert candidate.runoff is True

    def test_defaults(self) -> None:
        """Verify Candidate defaults for first_round and runoff."""
        candidate = Candidate(key="test", display_name="Test", coalition=None)
        assert candidate.first_round is True
        assert candidate.runoff is False

    def test_immutability(self) -> None:
        """Verify frozen dataclass rejects attribute assignment."""
        candidate = Candidate(key="test", display_name="Test", coalition=None)
        with pytest.raises(AttributeError):
            candidate.key = "changed"  # type: ignore[misc]


class TestModelConfig:
    """Tests for the ModelConfig frozen dataclass."""

    def test_default_values(self) -> None:
        """Verify all ModelConfig defaults match the spec."""
        cfg = ModelConfig()
        assert cfg.random_walk_sigma_prior == 0.5
        assert cfg.concentration_poll_prior_mean == 5.0
        assert cfg.concentration_election_prior_mean == 50.0
        assert cfg.house_effect_sigma_prior == 1.0
        assert cfg.mcmc_draws == 4000
        assert cfg.mcmc_tune == 1000
        assert cfg.mcmc_chains == 4
        assert cfg.mcmc_cores == 4
        assert cfg.target_accept == 0.95
        assert cfg.seed == 332211
        assert cfg.time_decay_half_life_days == 30.0
        assert cfg.consultation_prior_strength == 0.5
        assert cfg.fundamentals_mode == "prior_only"
        assert cfg.beta_coefficient_prior_sigma == 0.5
        assert cfg.sigma_m_prior == 0.3
        assert cfg.pool_alpha == 0.95
        assert cfg.enable_population_weighting is True

    def test_custom_values(self) -> None:
        """Verify ModelConfig accepts overrides for specific fields."""
        cfg = ModelConfig(mcmc_draws=1000, mcmc_chains=2)
        assert cfg.mcmc_draws == 1000
        assert cfg.mcmc_chains == 2

    def test_fundamentals_mode_off(self) -> None:
        """Verify fundamentals_mode can be set to 'off'."""
        cfg = ModelConfig(fundamentals_mode="off")
        assert cfg.fundamentals_mode == "off"

    def test_fundamentals_mode_joint(self) -> None:
        """Verify fundamentals_mode can be set to 'joint'."""
        cfg = ModelConfig(fundamentals_mode="joint")
        assert cfg.fundamentals_mode == "joint"

    def test_fundamentals_mode_prior_only_default(self) -> None:
        """Verify fundamentals_mode defaults to 'prior_only'."""
        cfg = ModelConfig()
        assert cfg.fundamentals_mode == "prior_only"

    def test_beta_coefficient_prior_sigma_custom(self) -> None:
        """Verify custom beta_coefficient_prior_sigma."""
        cfg = ModelConfig(beta_coefficient_prior_sigma=1.0)
        assert cfg.beta_coefficient_prior_sigma == 1.0

    def test_sigma_m_prior_custom(self) -> None:
        """Verify custom sigma_m_prior."""
        cfg = ModelConfig(sigma_m_prior=0.5)
        assert cfg.sigma_m_prior == 0.5

    def test_pool_alpha_custom(self) -> None:
        """Verify custom pool_alpha."""
        cfg = ModelConfig(pool_alpha=0.8)
        assert cfg.pool_alpha == 0.8

    def test_use_horseshoe_prior_default(self) -> None:
        """Verify use_horseshoe_prior defaults to False."""
        cfg = ModelConfig()
        assert cfg.use_horseshoe_prior is False

    def test_use_horseshoe_prior_enabled(self) -> None:
        """Verify use_horseshoe_prior can be set to True."""
        cfg = ModelConfig(use_horseshoe_prior=True)
        assert cfg.use_horseshoe_prior is True

    def test_enable_population_weighting_disabled(self) -> None:
        """Verify population weighting can be disabled."""
        cfg = ModelConfig(enable_population_weighting=False)
        assert cfg.enable_population_weighting is False

    def test_immutability(self) -> None:
        """Verify frozen dataclass rejects attribute assignment."""
        cfg = ModelConfig()
        with pytest.raises(AttributeError):
            cfg.mcmc_draws = 9999  # type: ignore[misc]

    def test_consultation_prior_strength_override_default(self) -> None:
        """Verify consultation_prior_strength_override defaults to None."""
        cfg = ModelConfig()
        assert cfg.consultation_prior_strength_override is None

    def test_consultation_prior_strength_override_custom(self) -> None:
        """Verify consultation_prior_strength_override accepts a dict."""
        overrides = {"gustavo_petro": 0.05}
        cfg = ModelConfig(consultation_prior_strength_override=overrides)
        assert cfg.consultation_prior_strength_override == overrides


class TestFirstRoundCandidates:
    """Tests for the FIRST_ROUND_CANDIDATES registry."""

    EXPECTED_COUNT: ClassVar[int] = 7  # 5 named + rest + blanco
    EXPECTED_KEYS: ClassVar[set[str]] = {
        "gustavo_petro",
        "federico_gutierrez",
        "rodolfo_hernandez",
        "sergio_fajardo",
        "ingrid_betancourt",
        "rest",
        "blanco",
    }

    def test_count(self) -> None:
        """Verify FIRST_ROUND_CANDIDATES has 7 entries."""
        assert len(FIRST_ROUND_CANDIDATES) == self.EXPECTED_COUNT

    def test_keys(self) -> None:
        """Verify FIRST_ROUND_CANDIDATES contains all expected keys."""
        assert set(FIRST_ROUND_CANDIDATES) == self.EXPECTED_KEYS

    def test_petro_fields(self) -> None:
        """Verify Gustavo Petro entry has correct coalition and runoff flag."""
        petro = FIRST_ROUND_CANDIDATES["gustavo_petro"]
        assert petro.display_name == "Gustavo Petro"
        assert petro.coalition == "Pacto Histórico"
        assert petro.first_round is True
        assert petro.runoff is True

    def test_gutierrez_not_runoff(self) -> None:
        """Verify Federico Gutiérrez is not in runoff."""
        gz = FIRST_ROUND_CANDIDATES["federico_gutierrez"]
        assert gz.runoff is False

    def test_hernandez_is_runoff(self) -> None:
        """Verify Rodolfo Hernández is flagged for runoff."""
        rodolfo = FIRST_ROUND_CANDIDATES["rodolfo_hernandez"]
        assert rodolfo.runoff is True

    def test_rest_is_sentinel(self) -> None:
        """Verify the rest sentinel has no coalition."""
        rest = FIRST_ROUND_CANDIDATES["rest"]
        assert rest.coalition is None


class TestCoalitionToCandidate:
    """Tests for the COALITION_TO_CANDIDATE mapping."""

    def test_major_coalitions(self) -> None:
        """Verify all 4 major coalitions map to correct candidates."""
        assert COALITION_TO_CANDIDATE["COALICION PACTO HISTORICO"] == "gustavo_petro"
        assert COALITION_TO_CANDIDATE["COALICION EQUIPO POR COLOMBIA"] == "federico_gutierrez"
        assert COALITION_TO_CANDIDATE["LIGA DE GOBERNANTES ANTICORRUPCION"] == "rodolfo_hernandez"
        assert COALITION_TO_CANDIDATE["COALICION CENTRO ESPERANZA"] == "sergio_fajardo"

    def test_blank_null_unmarked(self) -> None:
        """Verify blank, null, and unmarked vote mappings."""
        assert COALITION_TO_CANDIDATE["VOTOS EN BLANCO"] == "blanco"
        assert COALITION_TO_CANDIDATE["VOTOS NULOS"] == "nulos"
        assert COALITION_TO_CANDIDATE["VOTOS NO MARCADOS"] == "no_marcados"

    def test_minor_parties_map_to_rest(self) -> None:
        """Verify all minor parties map to the rest sentinel."""
        assert COALITION_TO_CANDIDATE["COLOMBIA JUSTA LIBRES"] == "rest"
        assert COALITION_TO_CANDIDATE["PARTIDO MOVIMIENTO DE SALVACION NACIONAL"] == "rest"
        assert COALITION_TO_CANDIDATE["COLOMBIA PIENSA EN GRANDE"] == "rest"
        assert COALITION_TO_CANDIDATE["PARTIDO VERDE OXIGENO"] == "rest"


class TestPollsterWeightFormula:
    """Tests for the pollster_weight_formula function."""

    def test_max_rating(self) -> None:
        """Verify rating 10.0 maps to weight 1.0."""
        assert pollster_weight_formula(10.0) == 1.0

    def test_zero_rating(self) -> None:
        """Verify rating 0.0 maps to weight 0.8."""
        assert pollster_weight_formula(0.0) == 0.8

    def test_mid_rating(self) -> None:
        """Verify rating 5.0 maps to weight 0.9."""
        assert pollster_weight_formula(5.0) == 0.9

    def test_invamer_rating(self) -> None:
        """Verify Invamer (top-rated) produces weight 1.0."""
        assert pollster_weight_formula(POLLSTER_RATINGS["Invamer"]) == 1.0

    def test_negative_rating_clamped_to_min(self) -> None:
        """Verify negative rating clamps to 0.8."""
        assert pollster_weight_formula(-5.0) == 0.8

    def test_above_ten_rating_clamped_to_max(self) -> None:
        """Verify rating > 10 clamps to 1.0."""
        assert pollster_weight_formula(15.0) == 1.0


class TestGetDefaultPollsterWeight:
    """Tests for the get_default_pollster_weight function."""

    def test_in_range(self) -> None:
        """Verify result is between 0.8 and 1.0."""
        w = get_default_pollster_weight()
        assert 0.8 <= w <= 1.0

    def test_deterministic(self) -> None:
        """Verify two calls return the same value."""
        assert get_default_pollster_weight() == get_default_pollster_weight()

    def test_formula_result(self) -> None:
        """Verify get_default_pollster_weight uses median of ratings.

        Median of 13 ratings is 5.4 (AtlasIntel).
        Formula: 5.4 * 0.02 + 0.8 = 0.908.
        """
        assert get_default_pollster_weight() == 0.908

    def test_empty_ratings_raises_valueerror(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify ValueError is raised if POLLSTER_RATINGS is empty."""
        monkeypatch.setattr("co_president.config.POLLSTER_RATINGS", {})
        with pytest.raises(ValueError, match="POLLSTER_RATINGS cannot be empty"):
            get_default_pollster_weight()


class TestGetActiveCandidates:
    """Tests for the get_active_candidates helper."""

    def test_round1_count(self) -> None:
        """Verify round 1 returns 7 candidates."""
        active = get_active_candidates(1)
        assert len(active) == 7

    def test_round1_contains_petro(self) -> None:
        """Verify Petro is active in round 1."""
        keys = {c.key for c in get_active_candidates(1)}
        assert "gustavo_petro" in keys

    def test_round1_excludes_nulos(self) -> None:
        """Verify nulos is not in round 1 active candidates."""
        keys = {c.key for c in get_active_candidates(1)}
        assert "nulos" not in keys

    def test_round2_count(self) -> None:
        """Verify round 2 returns 3 candidates (petro, hernandez, rest which aggregates blanco)."""
        active = get_active_candidates(2)
        assert len(active) == 3

    def test_round2_contains_only_runoff(self) -> None:
        """Verify only runoff candidates appear in round 2."""
        keys = {c.key for c in get_active_candidates(2)}
        assert keys == {"gustavo_petro", "rodolfo_hernandez", "rest"}

    def test_round_three_raises_valueerror(self) -> None:
        """Verify invalid round_number raises ValueError."""
        with pytest.raises(ValueError, match="round_number"):
            get_active_candidates(3)  # type: ignore[arg-type]

    def test_round1_has_correct_keys(self) -> None:
        """Verify round 1 returns the expected 7 candidate keys."""
        keys = {c.key for c in get_active_candidates(1)}
        expected = {
            "gustavo_petro",
            "federico_gutierrez",
            "rodolfo_hernandez",
            "sergio_fajardo",
            "ingrid_betancourt",
            "rest",
            "blanco",
        }
        assert keys == expected


class TestGetCandidateColumnMap:
    """Tests for the get_candidate_column_map helper."""

    def test_returns_dict(self) -> None:
        """Verify the function returns a dict."""
        mapping = get_candidate_column_map()
        assert isinstance(mapping, dict)

    def test_maps_petro(self) -> None:
        """Verify petro maps to itself (identity mapping)."""
        mapping = get_candidate_column_map()
        assert mapping["gustavo_petro"] == "gustavo_petro"

    def test_all_candidates_present(self) -> None:
        """Verify all 7 candidate keys appear in the mapping."""
        mapping = get_candidate_column_map()
        expected = {
            "gustavo_petro",
            "federico_gutierrez",
            "rodolfo_hernandez",
            "sergio_fajardo",
            "ingrid_betancourt",
            "rest",
            "blanco",
        }
        assert set(mapping) == expected


class TestConsultationVotes:
    """Tests for the CONSULTATION_VOTES constant."""

    EXPECTED_KEYS: ClassVar[set[str]] = {
        "gustavo_petro",
        "federico_gutierrez",
        "sergio_fajardo",
        "rodolfo_hernandez",
        "ingrid_betancourt",
    }

    def test_keys(self) -> None:
        """Verify CONSULTATION_VOTES has all 5 candidate keys."""
        assert set(CONSULTATION_VOTES) == self.EXPECTED_KEYS

    def test_non_negative(self) -> None:
        """Verify all consultation vote values are non-negative."""
        for v in CONSULTATION_VOTES.values():
            assert v >= 0

    def test_petro_positive(self) -> None:
        """Verify Petro has a positive consultation vote count."""
        assert CONSULTATION_VOTES["gustavo_petro"] > 0

    def test_hernandez_zero(self) -> None:
        """Verify Hernández has zero (independent, no consultation)."""
        assert CONSULTATION_VOTES["rodolfo_hernandez"] == 0


class TestConsultationLogSharePrior:
    """Tests for the consultation_log_share_prior function."""

    def test_returns_dict(self) -> None:
        """Verify the function returns a dict."""
        shares = consultation_log_share_prior()
        assert isinstance(shares, dict)

    def test_all_candidates_present(self) -> None:
        """Verify all 5 candidates have a log-share entry."""
        shares = consultation_log_share_prior()
        expected_keys = {
            "gustavo_petro",
            "federico_gutierrez",
            "sergio_fajardo",
            "rodolfo_hernandez",
            "ingrid_betancourt",
        }
        assert set(shares) == expected_keys

    def test_all_finite(self) -> None:
        """Verify all log-shares are finite (no -inf for zero-vote candidates)."""
        shares = consultation_log_share_prior()
        for k, v in shares.items():
            assert math.isfinite(v), f"Log-share for {k} is not finite: {v}"

    def test_non_zero_candidate_larger_than_zero(self) -> None:
        """Verify Petro (non-zero votes) has a higher log-share than Hernández (zero)."""
        shares = consultation_log_share_prior()
        assert shares["gustavo_petro"] > shares["rodolfo_hernandez"]

    def test_zero_vote_candidates_not_inf(self) -> None:
        """Verify zero-vote candidates do not get -inf log-shares."""
        shares = consultation_log_share_prior()
        assert shares["rodolfo_hernandez"] != -math.inf
        assert shares["ingrid_betancourt"] != -math.inf

    def test_deterministic(self) -> None:
        """Verify calling the function twice returns identical results."""
        a = consultation_log_share_prior()
        b = consultation_log_share_prior()
        assert a == b

    def test_all_zero_consultation_votes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify all-zero votes returns uniform -1.0 for every candidate."""
        monkeypatch.setattr(
            "co_president.config.CONSULTATION_VOTES",
            {"gustavo_petro": 0, "rodolfo_hernandez": 0},
        )
        shares = consultation_log_share_prior()
        assert shares == {"gustavo_petro": -1.0, "rodolfo_hernandez": -1.0}


class TestTransferConstants:
    """Tests for the TRANSFER_* runoff constants."""

    def test_fajardo_transfers_sum_to_one(self) -> None:
        """Verify Fajardo voter transfers partition correctly."""
        assert TRANSFER_FAJARDO_PETRO + TRANSFER_FAJARDO_HERNANDEZ == 1.0

    def test_gutierrez_transfers_sum_to_one(self) -> None:
        """Verify Gutiérrez voter transfers partition correctly."""
        assert TRANSFER_GUTIERREZ_HERNANDEZ + TRANSFER_GUTIERREZ_PETRO == 1.0

    def test_all_in_unit_interval(self) -> None:
        """Verify every transfer constant is in [0.0, 1.0]."""
        for name, val in [
            ("FAJARDO_PETRO", TRANSFER_FAJARDO_PETRO),
            ("FAJARDO_HERNANDEZ", TRANSFER_FAJARDO_HERNANDEZ),
            ("GUTIERREZ_HERNANDEZ", TRANSFER_GUTIERREZ_HERNANDEZ),
            ("GUTIERREZ_PETRO", TRANSFER_GUTIERREZ_PETRO),
            ("BLANCO_SPLIT", TRANSFER_BLANCO_SPLIT),
        ]:
            assert 0.0 <= val <= 1.0, f"TRANSFER_{name} out of range: {val}"

    def test_transfer_aggregate_split_approximates_observed(self) -> None:
        """Verify the aggregate transfer split approximates the observed 73/27 within ±5pp.

        Note: The simple average assumes equal electorate sizes for Fajardo
        and Gutiérrez. This is a calibration sanity check, not the true
        weighted aggregate (which would use each candidate's vote share).
        """
        flow_petro = (TRANSFER_FAJARDO_PETRO + TRANSFER_GUTIERREZ_PETRO) / 2
        flow_hernandez = (TRANSFER_FAJARDO_HERNANDEZ + TRANSFER_GUTIERREZ_HERNANDEZ) / 2

        assert math.isclose(flow_petro, 0.27, abs_tol=0.05)
        assert math.isclose(flow_hernandez, 0.73, abs_tol=0.05)

    def test_directional_constraints(self) -> None:
        """Verify directional constraints for transfer heuristics.

        Gutierrez voters should flow more to Hernandez than Petro; Fajardo
        voters should flow more to Hernandez than Petro per the empirical
        calibration.
        """
        assert TRANSFER_GUTIERREZ_HERNANDEZ > TRANSFER_GUTIERREZ_PETRO
        assert TRANSFER_FAJARDO_HERNANDEZ > TRANSFER_FAJARDO_PETRO


class TestConsultationKeyMap:
    """Tests for the CONSULTATION_KEY_MAP constant."""

    def test_has_expected_names(self) -> None:
        """Verify 5 expected human-readable names are present."""
        expected = {
            "Gustavo Petro",
            "Federico Gutierrez",
            "Sergio Fajardo",
            "Ingrid Betancourt",
            "Rodolfo Hernández",
        }
        assert set(CONSULTATION_KEY_MAP) == expected

    def test_values_are_valid_candidate_keys(self) -> None:
        """Verify every mapped value exists in FIRST_ROUND_CANDIDATES."""
        for key in CONSULTATION_KEY_MAP.values():
            assert key in FIRST_ROUND_CANDIDATES, f"{key} not in FIRST_ROUND_CANDIDATES"
