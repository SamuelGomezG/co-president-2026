"""SPEC-03: Tests for election results data consolidation.

Tests cover CandidateResult/RoundResult dataclasses, data loaders,
cross-validation, consolidation, and end-to-end load_actual_results().
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from co_president.config import (
    ELECTION_DATE_ROUND1,
    ELECTION_DATE_ROUND2,
    get_active_candidates,
)
from co_president.data import (
    CandidateResult,
    RoundResult,
    _build_round_result,
    consolidate_round,
    cross_validate,
    load_actual_results,
    load_moe_round1,
    load_moe_round2,
    load_participation_round1,
    load_participation_round2,
    load_registraduria_round1,
    load_registraduria_round2,
)

# ═══════════════════════════════════════════════════════════════════
# CandidateResult unit tests
# ═══════════════════════════════════════════════════════════════════


class TestCandidateResult:
    """Tests for the CandidateResult frozen dataclass."""

    def test_instantiation(self) -> None:
        """Verify CandidateResult creates with all fields."""
        cr = CandidateResult(candidate_key="gustavo_petro", votes=8_542_020, vote_share=0.4034)
        assert cr.candidate_key == "gustavo_petro"
        assert cr.votes == 8_542_020
        assert cr.vote_share == 0.4034

    def test_immutability(self) -> None:
        """Verify frozen dataclass rejects attribute assignment."""
        cr = CandidateResult(candidate_key="test", votes=100, vote_share=0.5)
        with pytest.raises(FrozenInstanceError):
            cr.candidate_key = "changed"  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════
# RoundResult unit tests
# ═══════════════════════════════════════════════════════════════════


class TestRoundResult:
    """Tests for the RoundResult frozen dataclass and its methods."""

    @pytest.fixture
    def sample_candidates(self) -> tuple[CandidateResult, ...]:
        """Provide a canonical 3-candidate round-1-like result."""
        return (
            CandidateResult("gustavo_petro", 8_542_020, 0.4034),
            CandidateResult("rodolfo_hernandez", 5_965_531, 0.2815),
            CandidateResult("federico_gutierrez", 5_069_526, 0.2389),
        )

    @pytest.fixture
    def sample_result(self, sample_candidates: tuple[CandidateResult, ...]) -> RoundResult:
        """Provide a RoundResult instance for method tests."""
        return RoundResult(
            round_number=1,
            date=ELECTION_DATE_ROUND1,
            total_valid_votes=19_577_077,  # sum of 3 candidate votes
            total_votes_incl_blank=19_942_854,  # total_valid + blank
            registered_voters=38_971_664,
            polling_stations=12_505,
            candidates=sample_candidates,
            blank_votes=365_777,
            null_votes=241_826,
            unmarked_votes=26_632,
        )

    def test_instantiation(self, sample_result: RoundResult) -> None:
        """Verify RoundResult stores all fields correctly."""
        assert sample_result.round_number == 1
        assert sample_result.date == ELECTION_DATE_ROUND1
        assert sample_result.total_valid_votes == 19_577_077
        assert sample_result.total_votes_incl_blank == 19_942_854
        assert sample_result.total_votes_incl_blank > sample_result.total_valid_votes
        assert sample_result.registered_voters == 38_971_664
        assert sample_result.polling_stations == 12_505
        assert len(sample_result.candidates) == 3
        assert sample_result.blank_votes == 365_777
        assert sample_result.null_votes == 241_826
        assert sample_result.unmarked_votes == 26_632

    def test_immutability(self, sample_result: RoundResult) -> None:
        """Verify frozen dataclass rejects attribute assignment."""
        with pytest.raises(FrozenInstanceError):
            sample_result.round_number = 2  # type: ignore[misc]

    # ── get_share ──

    def test_get_share_found(self, sample_result: RoundResult) -> None:
        """Verify get_share returns the correct vote share for an existing key."""
        assert sample_result.get_share("gustavo_petro") == 0.4034

    def test_get_share_not_found(self, sample_result: RoundResult) -> None:
        """Verify get_share raises KeyError for a missing candidate."""
        with pytest.raises(KeyError):
            sample_result.get_share("nonexistent")

    # ── get_candidates_above ──

    def test_get_candidates_above_some(self, sample_result: RoundResult) -> None:
        """Verify candidates exceeding the threshold are returned."""
        above = sample_result.get_candidates_above(25.0)
        keys = [c.candidate_key for c in above]
        assert keys == ["gustavo_petro", "rodolfo_hernandez"]

    def test_get_candidates_above_none(self, sample_result: RoundResult) -> None:
        """Verify empty list when no candidate meets the threshold."""
        above = sample_result.get_candidates_above(99.0)
        assert above == []

    def test_get_candidates_above_all(self, sample_result: RoundResult) -> None:
        """Verify all candidates returned when threshold is 0."""
        above = sample_result.get_candidates_above(0.0)
        assert len(above) == 3

    # ── top_two ──

    def test_top_two_round1(self, sample_result: RoundResult) -> None:
        """Verify top_two returns the two highest-vote candidates in order."""
        first, second = sample_result.top_two()
        assert first.candidate_key == "gustavo_petro"
        assert second.candidate_key == "rodolfo_hernandez"

    def test_top_two_not_enough_candidates(self) -> None:
        """Verify top_two raises ValueError when fewer than 2 candidates exist."""
        result = RoundResult(
            round_number=1,
            date=ELECTION_DATE_ROUND1,
            total_valid_votes=1000,
            total_votes_incl_blank=1000,
            registered_voters=2000,
            polling_stations=50,
            candidates=(CandidateResult("gustavo_petro", 1000, 1.0),),
            blank_votes=0,
            null_votes=0,
            unmarked_votes=0,
        )
        with pytest.raises(ValueError, match="fewer than 2"):
            result.top_two()

    # ── turnout ──

    def test_turnout_normal(self, sample_result: RoundResult) -> None:
        """Verify turnout is correctly computed as total_votes_incl_blank / registered."""
        expected = 19_942_854 / 38_971_664
        assert sample_result.turnout() == pytest.approx(expected)

    def test_turnout_zero_registered(self) -> None:
        """Verify turnout raises ZeroDivisionError when registered_voters is 0."""
        result = RoundResult(
            round_number=1,
            date=ELECTION_DATE_ROUND1,
            total_valid_votes=1000,
            total_votes_incl_blank=1000,
            registered_voters=0,
            polling_stations=0,
            candidates=(CandidateResult("gustavo_petro", 1000, 1.0),),
            blank_votes=0,
            null_votes=0,
            unmarked_votes=0,
        )
        with pytest.raises(ZeroDivisionError):
            result.turnout()


# ═══════════════════════════════════════════════════════════════════
# cross_validate unit tests
# ═══════════════════════════════════════════════════════════════════


class TestCrossValidate:
    """Tests for the cross_validate function."""

    @pytest.fixture
    def base_result(self) -> RoundResult:
        """Return a minimal RoundResult for cross-validation tests."""
        candidates = (
            CandidateResult("gustavo_petro", 1_000_000, 0.40),
            CandidateResult("rodolfo_hernandez", 750_000, 0.30),
        )
        return RoundResult(
            round_number=1,
            date=ELECTION_DATE_ROUND1,
            total_valid_votes=1_750_000,
            total_votes_incl_blank=2_560_000,  # total_valid + blank + null + unmarked
            registered_voters=5_000_000,
            polling_stations=100,
            candidates=candidates,
            blank_votes=500_000,
            null_votes=50_000,
            unmarked_votes=10_000,
        )

    def test_identical_results_returns_empty(self, base_result: RoundResult) -> None:
        """Verify identical RoundResults produce zero warnings."""
        warnings = cross_validate(base_result, base_result)
        assert warnings == []

    def test_differing_total_votes_returns_warning(self, base_result: RoundResult) -> None:
        """Verify total valid vote mismatch beyond 0.10% produces a warning."""
        moe = RoundResult(
            round_number=1,
            date=ELECTION_DATE_ROUND1,
            total_valid_votes=1_746_500,  # 0.20% less, exceeds 0.10% tolerance
            total_votes_incl_blank=2_556_500,
            registered_voters=5_000_000,
            polling_stations=100,
            candidates=base_result.candidates,
            blank_votes=250_000,
            null_votes=50_000,
            unmarked_votes=10_000,
        )
        warnings = cross_validate(base_result, moe)
        assert len(warnings) >= 1
        assert any("total valid votes" in w.lower() for w in warnings)

    def test_differing_shares_returns_warnings(self, base_result: RoundResult) -> None:
        """Verify candidate share mismatch beyond 0.50% produces warnings (0.60pp diff)."""
        diff_candidates = (
            CandidateResult(
                "gustavo_petro", 1_000_000, 0.394
            ),  # 0.60pp lower (unambiguously > 0.50pp)
            CandidateResult("rodolfo_hernandez", 750_000, 0.306),  # 0.60pp higher
        )
        moe = RoundResult(
            round_number=1,
            date=ELECTION_DATE_ROUND1,
            total_valid_votes=1_750_000,
            total_votes_incl_blank=2_560_000,
            registered_voters=5_000_000,
            polling_stations=100,
            candidates=diff_candidates,
            blank_votes=250_000,
            null_votes=50_000,
            unmarked_votes=10_000,
        )
        warnings = cross_validate(base_result, moe)
        assert len(warnings) >= 2  # one per differing candidate


# ═══════════════════════════════════════════════════════════════════
# consolidate_round unit tests
# ═══════════════════════════════════════════════════════════════════


class TestConsolidateRound:
    """Tests for the consolidate_round function."""

    @pytest.fixture
    def reg_result(self) -> RoundResult:
        """Return a canonical Registraduría result."""
        candidates = (
            CandidateResult("gustavo_petro", 1_000_000, 0.40),
            CandidateResult("rodolfo_hernandez", 750_000, 0.30),
        )
        return RoundResult(
            round_number=1,
            date=ELECTION_DATE_ROUND1,
            total_valid_votes=1_750_000,
            total_votes_incl_blank=2_560_000,
            registered_voters=5_000_000,
            polling_stations=100,
            candidates=candidates,
            blank_votes=250_000,
            null_votes=50_000,
            unmarked_votes=10_000,
        )

    def test_matching_returns_registraduria_unchanged(
        self,
        reg_result: RoundResult,
    ) -> None:
        """Verify consolidate_round returns Registraduría unchanged when MOE matches."""
        result = consolidate_round(reg_result, reg_result, 1)
        assert result is reg_result  # same object returned

    def test_discrepancy_exceeds_tolerance_raises_valueerror(
        self,
        reg_result: RoundResult,
    ) -> None:
        """Verify consolidate_round raises ValueError when diff exceeds tolerance."""
        bad_moe = RoundResult(
            round_number=1,
            date=ELECTION_DATE_ROUND1,
            total_valid_votes=200_000,  # massively different
            total_votes_incl_blank=200_000,
            registered_voters=5_000_000,
            polling_stations=100,
            candidates=(
                CandidateResult("gustavo_petro", 100_000, 0.50),
                CandidateResult("rodolfo_hernandez", 100_000, 0.50),
            ),
            blank_votes=0,
            null_votes=0,
            unmarked_votes=0,
        )
        with pytest.raises(ValueError, match=r"[Cc]onsolidation"):
            consolidate_round(reg_result, bad_moe, 1)


# ═══════════════════════════════════════════════════════════════════
# Loader integration tests (read actual data files)
# ═══════════════════════════════════════════════════════════════════


class TestRegistraduriaLoaders:
    """Tests for Registraduría MMV data loaders."""

    def test_load_registraduria_round1_has_expected_keys(self, data_dir: Path) -> None:
        """Verify round 1 contains all major candidate keys after mapping."""
        df = load_registraduria_round1(data_dir)
        assert set(df.index) >= {"gustavo_petro", "rodolfo_hernandez", "federico_gutierrez"}
        assert "blanco" in df.index
        # Betancourt's votes go to "rest" per spec (she ran under PARTIDO VERDE OXIGENO)
        assert "ingrid_betancourt" not in df.index

    def test_load_registraduria_round1_total_votes(self, data_dir: Path) -> None:
        """Verify round 1 total votes are reasonable (nulos excluded downstream)."""
        df = load_registraduria_round1(data_dir)
        total = df["votes"].sum()
        # Total should be in the 20-22 million range including nulos/no_marcados
        assert 20_000_000 < total < 22_000_000

    def test_load_registraduria_round2_has_runoff_keys(self, data_dir: Path) -> None:
        """Verify round 2 contains only runoff-relevant candidate keys."""
        df = load_registraduria_round2(data_dir)
        assert "gustavo_petro" in df.index
        assert "rodolfo_hernandez" in df.index
        assert "blanco" in df.index  # raw data has blanco (merged to rest in _build)
        assert "federico_gutierrez" not in df.index
        assert "sergio_fajardo" not in df.index

    def test_load_registraduria_round2_rest_includes_blanco(self, data_dir: Path) -> None:
        """Verify final round 2 result merges blanco into rest."""
        raw = load_registraduria_round2(data_dir)
        part = load_participation_round1(data_dir)
        r2 = _build_round_result(
            raw,
            2,
            registered_voters=int(part["Total censo"].sum()),
            polling_stations=int(part["Código Puesto"].nunique()),
        )
        assert "blanco" not in {c.candidate_key for c in r2.candidates}
        rest = next(c for c in r2.candidates if c.candidate_key == "rest")
        assert rest.votes > 400_000


class TestMOELoaders:
    """Tests for MOE data loaders."""

    def test_load_moe_round1_has_coalition_names(self, data_dir: Path) -> None:
        """Verify MOE round 1 contains the expected coalitions after mapping."""
        df = load_moe_round1(data_dir)
        assert set(df.index) >= {"gustavo_petro", "rodolfo_hernandez", "federico_gutierrez"}
        assert "blanco" in df.index

    def test_load_moe_round1_total_matches_registraduria(self, data_dir: Path) -> None:
        """Verify MOE round 1 total votes match Registraduría within 1%."""
        reg = load_registraduria_round1(data_dir)
        moe = load_moe_round1(data_dir)
        reg_total = reg["votes"].sum()
        moe_total = moe["votes"].sum()
        assert abs(reg_total - moe_total) / reg_total < 0.01

    def test_load_moe_round2_has_runoff_coalitions(self, data_dir: Path) -> None:
        """Verify MOE round 2 contains only runoff coalitions."""
        df = load_moe_round2(data_dir)
        assert "gustavo_petro" in df.index
        assert "rodolfo_hernandez" in df.index
        assert "blanco" in df.index


class TestParticipationLoaders:
    """Tests for participation data loaders."""

    def test_load_participation_round1_has_required_columns(self, data_dir: Path) -> None:
        """Verify participation round 1 has Total censo and Código Puesto."""
        df = load_participation_round1(data_dir)
        assert "Total censo" in df.columns
        assert "Código Puesto" in df.columns

    def test_load_participation_round1_total_censo(self, data_dir: Path) -> None:
        """Verify total registered voters is within ±1% of official ~39M."""
        df = load_participation_round1(data_dir)
        total_censo = df["Total censo"].sum()
        assert 38_500_000 < total_censo < 39_500_000

    def test_load_participation_round1_unique_puestos(self, data_dir: Path) -> None:
        """Verify polling station count is reasonable."""
        df = load_participation_round1(data_dir)
        n_puestos = df["Código Puesto"].nunique()
        assert 10_000 < n_puestos < 15_000

    def test_load_participation_round2_has_required_columns(self, data_dir: Path) -> None:
        """Verify participation round 2 has Total censo and Código Puesto."""
        df = load_participation_round2(data_dir)
        assert "Total censo" in df.columns
        assert "Código Puesto" in df.columns

    def test_load_participation_round2_total_censo_matches_round1(self, data_dir: Path) -> None:
        """Verify round 2 total censo matches round 1 (same electorate)."""
        df1 = load_participation_round1(data_dir)
        df2 = load_participation_round2(data_dir)
        assert df1["Total censo"].sum() == df2["Total censo"].sum()


# ═══════════════════════════════════════════════════════════════════
# End-to-end integration tests: load_actual_results()
# ═══════════════════════════════════════════════════════════════════


class TestLoadActualResults:
    """Tests for the full load_actual_results() pipeline."""

    @pytest.fixture(autouse=True, scope="class")
    def _results(self, request: pytest.FixtureRequest, data_dir: Path) -> None:
        """Load canonical results once per test class."""
        request.cls.round1, request.cls.round2 = load_actual_results(data_dir)

    # ── Round structure ──

    def test_round1_candidate_keys(self) -> None:
        """Verify round 1 candidates match get_active_candidates(1) minus Betancourt.

        Betancourt's votes are under PARTIDO VERDE OXIGENO which maps to ``rest``
        in the consolidated results (she withdrew; tracked separately in polls only).
        """
        expected_keys = {c.key for c in get_active_candidates(1)}
        # ingrid_betancourt is tracked in polls only, not in consolidated results
        expected_keys.discard("ingrid_betancourt")
        actual_keys = {c.candidate_key for c in self.round1.candidates}
        assert actual_keys == expected_keys

    def test_round2_candidate_keys(self) -> None:
        """Verify round 2 candidates match get_active_candidates(2)."""
        expected_keys = {c.key for c in get_active_candidates(2)}
        actual_keys = {c.candidate_key for c in self.round2.candidates}
        assert actual_keys == expected_keys

    def test_round_number_fields(self) -> None:
        """Verify round_number and date are correctly assigned."""
        assert self.round1.round_number == 1
        assert self.round1.date == ELECTION_DATE_ROUND1
        assert self.round2.round_number == 2
        assert self.round2.date == ELECTION_DATE_ROUND2

    # ── Vote share acceptance criteria ──

    def test_round1_petro_share(self) -> None:
        """Verify Petro round 1 vote share ≈ 40.34% (within 0.5pp)."""
        share = self.round1.get_share("gustavo_petro")
        assert share == pytest.approx(0.4034, abs=0.005)

    def test_round1_hernandez_share(self) -> None:
        """Verify Hernández round 1 vote share ≈ 28.15% (within 0.5pp)."""
        share = self.round1.get_share("rodolfo_hernandez")
        assert share == pytest.approx(0.2815, abs=0.005)

    def test_round1_gutierrez_share(self) -> None:
        """Verify Gutiérrez round 1 vote share ≈ 23.89% (within 0.5pp)."""
        share = self.round1.get_share("federico_gutierrez")
        assert share == pytest.approx(0.2389, abs=0.005)

    def test_round1_fajardo_share(self) -> None:
        """Verify Fajardo round 1 vote share ≈ 4.39% (within 0.5pp)."""
        share = self.round1.get_share("sergio_fajardo")
        assert share == pytest.approx(0.0439, abs=0.005)

    def test_round2_petro_share(self) -> None:
        """Verify Petro round 2 vote share ≈ 50.44% (within 0.5pp)."""
        share = self.round2.get_share("gustavo_petro")
        assert share == pytest.approx(0.5044, abs=0.005)

    def test_round2_hernandez_share(self) -> None:
        """Verify Hernández round 2 vote share ≈ 47.26% (within 0.5pp)."""
        share = self.round2.get_share("rodolfo_hernandez")
        assert share == pytest.approx(0.4726, abs=0.005)

    # ── RoundResult methods ──

    def test_round1_top_two(self) -> None:
        """Verify top_two for round 1 returns Petro then Hernández."""
        first, second = self.round1.top_two()
        assert first.candidate_key == "gustavo_petro"
        assert second.candidate_key == "rodolfo_hernandez"

    def test_round1_candidates_above_20(self) -> None:
        """Verify candidates above 20% in round 1."""
        above = self.round1.get_candidates_above(20.0)
        keys = [c.candidate_key for c in above]
        assert keys == ["gustavo_petro", "rodolfo_hernandez", "federico_gutierrez"]

    def test_round2_top_two(self) -> None:
        """Verify top_two for round 2 returns Petro then Hernández."""
        first, second = self.round2.top_two()
        assert first.candidate_key == "gustavo_petro"
        assert second.candidate_key == "rodolfo_hernandez"

    # ── Census and turnout ──

    def test_registered_voters_reasonable(self) -> None:
        """Verify registered voters count is within ±1% of official ~39M."""
        assert 38_500_000 < self.round1.registered_voters < 39_500_000
        assert self.round1.registered_voters == self.round2.registered_voters

    def test_polling_stations_reasonable(self) -> None:
        """Verify polling station count is reasonable."""
        assert 10_000 < self.round1.polling_stations < 15_000
        assert self.round1.polling_stations == self.round2.polling_stations

    # ── Cross-validation ──

    def test_cross_validation_zero_warnings(self, data_dir: Path) -> None:
        """Verify cross-validation between Reg and MOE produces zero warnings."""
        moe1 = load_moe_round1(data_dir)
        part1 = load_participation_round1(data_dir)

        # Build MOE RoundResult
        moe_r1 = _build_round_result(
            moe1,
            round_number=1,
            registered_voters=int(part1["Total censo"].sum()),
            polling_stations=int(part1["Código Puesto"].nunique()),
        )
        warnings = cross_validate(self.round1, moe_r1)
        assert warnings == [], f"Cross-validation warnings: {warnings}"

    # ── Vote share sum consistency ──

    def test_round1_vote_shares_sum_to_approx_one(self) -> None:
        """Verify round 1 candidate vote shares sum to ~1.0 (excluding blanco from shares)."""
        total = sum(c.vote_share for c in self.round1.candidates)
        assert total == pytest.approx(1.0, abs=0.01)

    def test_round2_vote_shares_sum_to_approx_one(self) -> None:
        """Verify round 2 candidate vote shares sum to ~1.0 (rest includes blanco)."""
        total = sum(c.vote_share for c in self.round2.candidates)
        assert total == pytest.approx(1.0, abs=0.01)
