"""SPEC-02: Tests for configuration module — candidates, coalitions, hyperparameters."""

from __future__ import annotations

import importlib
import math
from typing import ClassVar

import pytest

from co_president.config import (
    COALITION_TO_CANDIDATE,
    CONSULTATION_VOTES,
    FIRST_ROUND_CANDIDATES,
    POLLSTER_RATINGS,
    Candidate,
    ModelConfig,
    consultation_prior_logits,
    get_active_candidates,
    get_candidate_column_map,
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
        """Verify all 12 ModelConfig defaults match the spec."""
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

    def test_custom_values(self) -> None:
        """Verify ModelConfig accepts overrides for specific fields."""
        cfg = ModelConfig(mcmc_draws=1000, mcmc_chains=2)
        assert cfg.mcmc_draws == 1000
        assert cfg.mcmc_chains == 2

    def test_immutability(self) -> None:
        """Verify frozen dataclass rejects attribute assignment."""
        cfg = ModelConfig()
        with pytest.raises(AttributeError):
            cfg.mcmc_draws = 9999  # type: ignore[misc]


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
        """Verify round 2 returns 3 candidates (petro, hernandez, blanco)."""
        active = get_active_candidates(2)
        assert len(active) == 3

    def test_round2_contains_only_runoff(self) -> None:
        """Verify only runoff candidates appear in round 2."""
        keys = {c.key for c in get_active_candidates(2)}
        assert keys == {"gustavo_petro", "rodolfo_hernandez", "blanco"}

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


class TestConsultationPriorLogits:
    """Tests for the consultation_prior_logits function."""

    def test_returns_dict(self) -> None:
        """Verify the function returns a dict."""
        logits = consultation_prior_logits()
        assert isinstance(logits, dict)

    def test_all_candidates_present(self) -> None:
        """Verify all 5 candidates have a logit entry."""
        logits = consultation_prior_logits()
        expected_keys = {
            "gustavo_petro",
            "federico_gutierrez",
            "sergio_fajardo",
            "rodolfo_hernandez",
            "ingrid_betancourt",
        }
        assert set(logits) == expected_keys

    def test_all_finite(self) -> None:
        """Verify all logits are finite (no -inf for zero-vote candidates)."""
        logits = consultation_prior_logits()
        for k, v in logits.items():
            assert math.isfinite(v), f"Logit for {k} is not finite: {v}"

    def test_non_zero_candidate_larger_than_zero(self) -> None:
        """Verify Petro (non-zero votes) has a higher logit than Hernández (zero)."""
        logits = consultation_prior_logits()
        assert logits["gustavo_petro"] > logits["rodolfo_hernandez"]

    def test_zero_vote_candidates_not_inf(self) -> None:
        """Verify zero-vote candidates do not get -inf logits."""
        logits = consultation_prior_logits()
        assert logits["rodolfo_hernandez"] != -math.inf
        assert logits["ingrid_betancourt"] != -math.inf
