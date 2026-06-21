"""SPEC-02: Tests for configuration module — candidates, coalitions, hyperparameters."""

from __future__ import annotations

from datetime import date
import importlib
import math
from typing import ClassVar

import pytest

from co_president.config import (
    CANDIDATES_BY_YEAR,
    COALITION_TO_CANDIDATE,
    COALITION_TO_CANONICAL_WEIGHTS,
    CONSULTATION_KEY_MAP,
    CONSULTATION_VOTES,
    FIRST_ROUND_CANDIDATES,
    FIRST_ROUND_CANDIDATES_2026,
    HISTORICAL_CANDIDATE_IDEOLOGY,
    HISTORICAL_CANDIDATE_IDEOLOGY_5CLASS,
    HISTORICAL_ROUND2_IDEOLOGY,
    POLLSTER_RATINGS,
    Candidate,
    ModelConfig,
    consultation_log_share_prior,
    get_active_candidates,
    get_candidate_column_map,
    get_coalition_to_candidate,
    get_coalition_weights,
    get_consultation_key_map,
    get_consultation_votes,
    get_default_pollster_weight,
    get_election_date,
    get_historical_candidate_ideology,
    get_pollster_ratings,
    get_runoff_ideology,
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
        assert cfg.concentration_election_prior_mean == 5000.0
        assert cfg.concentration_election_votes_scale == 1000000
        assert cfg.concentration_election_prior_shape == 0.75
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
        assert cfg.nuts_sampler is None
        assert cfg.target_year == 2022

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


class TestFirstRoundCandidates2026:
    """Tests for the FIRST_ROUND_CANDIDATES_2026 registry."""

    EXPECTED_KEYS_2026: ClassVar[set[str]] = {
        "cepeda",
        "de_la_espriella",
        "valencia",
        "fajardo",
        "claudia_lopez",
        "rest",
        "blanco",
    }

    def test_populated(self) -> None:
        """Verify 2026 candidate dict is populated with canonical keys."""
        assert set(FIRST_ROUND_CANDIDATES_2026) == self.EXPECTED_KEYS_2026

    def test_is_dict(self) -> None:
        """Verify type is dict."""
        assert isinstance(FIRST_ROUND_CANDIDATES_2026, dict)


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

    def test_year_2026_returns_candidates(self) -> None:
        """Verify year=2026 returns the populated candidate registry."""
        active = get_active_candidates(1, year=2026)
        keys = {c.key for c in active}
        assert keys == TestFirstRoundCandidates2026.EXPECTED_KEYS_2026


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

    def test_year_2026_returns_mapping(self) -> None:
        """Verify year=2026 returns an identity mapping for populated candidates."""
        mapping = get_candidate_column_map(year=2026)
        assert set(mapping) == TestFirstRoundCandidates2026.EXPECTED_KEYS_2026
        assert mapping["cepeda"] == "cepeda"


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


class TestCoalitionCrosswalk:
    """Tests for the COALITION_TO_CANONICAL_WEIGHTS crosswalk dict."""

    def test_pacto_historico_has_national_entry(self) -> None:
        """Pacto Histórico has national-level weights."""
        entry = COALITION_TO_CANONICAL_WEIGHTS.get("COALICION PACTO HISTORICO")
        assert entry is not None
        national = entry.get("__national__")
        assert isinstance(national, dict)
        assert len(national) >= 5
        assert "Colombia Humana (15)" in national

    def test_equipo_colombia_has_national_entry(self) -> None:
        """Equipo por Colombia has national-level weights."""
        entry = COALITION_TO_CANONICAL_WEIGHTS.get("COALICION EQUIPO POR COLOMBIA")
        assert entry is not None
        national = entry.get("__national__")
        assert isinstance(national, dict)
        assert len(national) >= 4

    def test_centro_esperanza_has_national_entry(self) -> None:
        """Centro Esperanza has national-level weights."""
        entry = COALITION_TO_CANONICAL_WEIGHTS.get("COALICION CENTRO ESPERANZA")
        assert entry is not None
        national = entry.get("__national__")
        assert isinstance(national, dict)
        assert len(national) >= 4

    def test_single_party_coalitions_weigh_one(self) -> None:
        """Single-party coalitions have weight 1.0 for their party."""
        entry = COALITION_TO_CANONICAL_WEIGHTS["LIGA DE GOBERNANTES ANTICORRUPCION"]
        national = entry["__national__"]
        assert isinstance(national, dict)
        assert list(national.values()) == [1.0]

    def test_all_weights_sum_to_one(self) -> None:
        """Every national entry's weights sum to 1.0 within tolerance."""
        for coalition, entry in COALITION_TO_CANONICAL_WEIGHTS.items():
            national = entry.get("__national__")
            if isinstance(national, dict):
                total = sum(national.values())
                assert abs(total - 1.0) < 0.01, (
                    f"{coalition} national weights sum to {total:.3f} != 1.0"
                )


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


class TestHistoricalCandidateIdeology5Class:
    """Validation tests for the 5-class ideology master map."""

    _VALID_CLASSES: ClassVar[frozenset[str]] = frozenset(
        {
            "Izquierda",
            "Centro_Izquierda",
            "Centro",
            "Centro_Derecha",
            "Derecha",
        }
    )

    def test_all_entries_have_valid_class(self) -> None:
        """Every candidate in the 5-class map has a valid ideology class."""
        for (year, round_num), round_map in HISTORICAL_CANDIDATE_IDEOLOGY_5CLASS.items():
            for candidate, cls in round_map.items():
                assert cls in self._VALID_CLASSES, (
                    f"{candidate} in ({year}, {round_num}) has invalid class {cls!r}"
                )

    def test_all_years_covered(self) -> None:
        """All presidential election years 2002-2022 are covered."""
        years = sorted({k[0] for k in HISTORICAL_CANDIDATE_IDEOLOGY_5CLASS})
        assert years == [2002, 2006, 2010, 2014, 2018, 2022]


class TestHistoricalCandidateIdeology:
    """Auto-derivation validation for 2-class ideology from 5-class master map."""

    _EXPECTED: ClassVar[dict[int, dict[str, str]]] = {
        2002: {"left": "luis_eduardo_garzon", "right": "alvaro_uribe"},
        2006: {"left": "carlos_gaviria", "right": "alvaro_uribe"},
        2010: {"left": "gustavo_petro", "right": "juan_manuel_santos"},
        2014: {"left": "clara_lopez", "right": "juan_manuel_santos"},
        2018: {"left": "gustavo_petro", "right": "ivan_duque"},
        2022: {"left": "gustavo_petro", "right": "rodolfo_hernandez"},
    }

    def test_derived_values_match_expected(self) -> None:
        """Auto-derived map matches Option A expected values."""
        assert HISTORICAL_CANDIDATE_IDEOLOGY == self._EXPECTED

    def test_all_years_covered(self) -> None:
        """All 6 presidential election years 2002-2022 have an entry."""
        assert sorted(HISTORICAL_CANDIDATE_IDEOLOGY) == [2002, 2006, 2010, 2014, 2018, 2022]

    def test_left_and_right_differ(self) -> None:
        """Left and right candidates are distinct for every year."""
        for year, ideology in HISTORICAL_CANDIDATE_IDEOLOGY.items():
            assert ideology["left"] != ideology["right"], f"{year} left equals right"

    def test_canonical_names_only(self) -> None:
        """All left/right values are canonical (lowercase_snake_case)."""
        for year, ideology in HISTORICAL_CANDIDATE_IDEOLOGY.items():
            for side, cand in ideology.items():
                assert "_" in cand, f"{year} {side}={cand!r} has no underscore"
                assert cand.islower(), f"{year} {side}={cand!r} is not lowercase"

    def test_round2_ideology_consistent(self) -> None:
        """Round-2 overrides reference valid candidates from the 5-class R2 map."""
        for year, ideology in HISTORICAL_ROUND2_IDEOLOGY.items():
            round_map = HISTORICAL_CANDIDATE_IDEOLOGY_5CLASS.get((year, 2), {})
            for side, candidate in ideology.items():
                assert candidate in round_map, (
                    f"{year} R2 {side}={candidate!r} not in 5-class R2 map"
                )


class TestYearAgnosticHelpers:
    """Tests for the year-indexed helper functions."""

    def test_get_election_date_2022(self) -> None:
        """Verify election date lookup for 2022."""
        assert get_election_date(2022, 1) == date(2022, 5, 29)
        assert get_election_date(2022, 2) == date(2022, 6, 19)
        assert get_election_date(2022) == date(2022, 3, 13)

    def test_get_election_date_unknown_year(self) -> None:
        """Verify ValueError for unknown year/round combinations."""
        with pytest.raises(ValueError, match="No first-round election date"):
            get_election_date(1990, 1)
        with pytest.raises(ValueError, match="No runoff election date"):
            get_election_date(1990, 2)
        with pytest.raises(ValueError, match="No consultation date"):
            get_election_date(1990)

    def test_get_pollster_ratings_2022(self) -> None:
        """Verify 2022 pollster ratings are returned."""
        ratings = get_pollster_ratings(2022)
        assert ratings == POLLSTER_RATINGS

    def test_get_pollster_ratings_2026(self) -> None:
        """Verify 2026 pollster ratings are returned."""
        ratings = get_pollster_ratings(2026)
        assert isinstance(ratings, dict)
        assert len(ratings) > 0
        assert "CNC" in ratings
        assert ratings["CNC"] == 5.8

    def test_get_consultation_votes_2022(self) -> None:
        """Verify 2022 consultation votes are returned."""
        assert get_consultation_votes(2022) == CONSULTATION_VOTES

    def test_get_consultation_key_map_2022(self) -> None:
        """Verify 2022 consultation key map is returned."""
        assert get_consultation_key_map(2022) == CONSULTATION_KEY_MAP

    def test_get_coalition_to_candidate_2022(self) -> None:
        """Verify 2022 coalition-to-candidate map is returned."""
        assert get_coalition_to_candidate(2022) == COALITION_TO_CANDIDATE

    def test_get_coalition_weights_2022(self) -> None:
        """Verify 2022 coalition weights are returned."""
        assert get_coalition_weights(2022) == COALITION_TO_CANONICAL_WEIGHTS

    def test_get_historical_candidate_ideology(self) -> None:
        """Verify 5-class ideology lookup by (year, round)."""
        r1_map = get_historical_candidate_ideology(2022, 1)
        assert r1_map["gustavo_petro"] == "Izquierda"
        assert get_historical_candidate_ideology(1990, 1) == {}

    def test_get_runoff_ideology_2022(self) -> None:
        """Verify runoff ideology override for 2022."""
        assert get_runoff_ideology(2022) == {"left": "gustavo_petro", "right": "rodolfo_hernandez"}

    def test_get_runoff_ideology_unknown(self) -> None:
        """Verify empty dict for unknown year."""
        assert get_runoff_ideology(1990) == {}

    def test_candidates_by_year_2022(self) -> None:
        """Verify 2022 candidate registry matches FIRST_ROUND_CANDIDATES."""
        assert CANDIDATES_BY_YEAR[2022] == FIRST_ROUND_CANDIDATES
        assert CANDIDATES_BY_YEAR[2026] == FIRST_ROUND_CANDIDATES_2026

    def test_consultation_log_share_prior_year_default(self) -> None:
        """Verify default year for consultation log prior is 2022."""
        default = consultation_log_share_prior()
        explicit = consultation_log_share_prior(2022)
        assert default == explicit

    def test_get_default_pollster_weight_year(self) -> None:
        """Verify default weight uses the requested year's ratings."""
        assert get_default_pollster_weight(2022) == get_default_pollster_weight()
