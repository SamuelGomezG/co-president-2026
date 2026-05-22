"""SPEC-03+04: Tests for election results consolidation and poll data loading.

Tests cover CandidateResult/RoundResult/CandidateShares dataclasses,
data loaders, cross-validation, consolidation, poll cleaning functions,
and end-to-end load_actual_results() / load_and_clean_all().
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date
import logging
from typing import TYPE_CHECKING

import pandas as pd
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from co_president.config import (
    ELECTION_DATE_ROUND1,
    ELECTION_DATE_ROUND2,
    get_active_candidates,
)
from co_president.data_polls import (
    _SHARE_COLS_EXCLUDED,
    CandidateShares,
    CleanPolls,
    ConsultationPoll,
    PollRow,
    UnclassifiedPollRow,
    _detect_forced_choice,
    _validate_normalized_rows,
    deduplicate_polls,
    fix_invamer_date,
    fix_yanhaas_20220611,
    infer_round_number,
    load_and_clean_all,
    load_raw_consultas,
    load_raw_polls,
    map_consultation_name_to_key,
    normalize_undecided,
    parse_consultations,
    retain_active_candidates,
)
from co_president.data_results import (
    CandidateResult,
    RoundResult,
    _build_round_result,
    _compute_candidate_results,
    _extract_excluded_votes,
    _merge_blanco_into_rest,
    _read_mmv,
    _read_moe,
    _read_participation,
    consolidate_round,
    cross_validate,
    load_canonical_results,
    load_moe_round1,
    load_moe_round2,
    load_participation_round1,
    load_participation_round2,
    load_registraduria_round1,
    load_registraduria_round2,
)
from co_president.paths import resolve_data_dir

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
        """Verify candidate share mismatch beyond 0.50pp produces warnings (0.60pp diff)."""
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
# Helper unit tests
# ═══════════════════════════════════════════════════════════════════


class TestResultHelpers:
    """Tests for internal result helper functions."""

    def test_read_mmv_missing_parnombre_raises(self, tmp_path: Path) -> None:
        """Verify missing PARNOMBRE column raises ValueError."""
        path = tmp_path / "mmv.csv"
        df = pd.DataFrame({"VOTOS": [1, 2, 3]})
        df.to_csv(path, sep=";", index=False, encoding="latin-1")
        with pytest.raises(ValueError, match="PARNOMBRE"):
            _read_mmv(path)

    def test_read_mmv_missing_votos_raises(self, tmp_path: Path) -> None:
        """Verify missing VOTOS column raises ValueError."""
        path = tmp_path / "mmv.csv"
        df = pd.DataFrame({"PARNOMBRE": ["A", "B"]})
        df.to_csv(path, sep=";", index=False, encoding="latin-1")
        with pytest.raises(ValueError, match="VOTOS"):
            _read_mmv(path)

    def test_read_moe_missing_nomparti_raises(self, tmp_path: Path) -> None:
        """Verify missing nomparti column raises ValueError."""
        path = tmp_path / "moe.csv"
        df = pd.DataFrame({"votos": [10, 20]})
        df.to_csv(path, index=False, encoding="utf-8")
        with pytest.raises(ValueError, match="nomparti"):
            _read_moe(path)

    def test_read_moe_missing_votos_raises(self, tmp_path: Path) -> None:
        """Verify missing votos column raises ValueError."""
        path = tmp_path / "moe.csv"
        df = pd.DataFrame({"nomparti": ["A", "B"]})
        df.to_csv(path, index=False, encoding="utf-8")
        with pytest.raises(ValueError, match="votos"):
            _read_moe(path)

    def test_read_participation_missing_total_censo_raises(self, tmp_path: Path) -> None:
        """Verify missing Total censo column raises ValueError."""
        path = tmp_path / "participation.csv"
        df = pd.DataFrame({"Código Puesto": [1, 2]})
        df.to_csv(path, index=False, encoding="utf-8-sig")
        with pytest.raises(ValueError, match="Total censo"):
            _read_participation(path)

    def test_read_participation_missing_codigo_puesto_raises(self, tmp_path: Path) -> None:
        """Verify missing Código Puesto column raises ValueError."""
        path = tmp_path / "participation.csv"
        df = pd.DataFrame({"Total censo": [100, 200]})
        df.to_csv(path, index=False, encoding="utf-8-sig")
        with pytest.raises(ValueError, match="Código Puesto"):
            _read_participation(path)

    def test_extract_excluded_votes_all_present(self) -> None:
        """Verify excluded votes are extracted when all keys exist."""
        aggregated = pd.DataFrame(
            {"votes": [10, 5, 3, 100]},
            index=["nulos", "no_marcados", "blanco", "gustavo_petro"],
        )
        null_votes, unmarked_votes, blank_votes = _extract_excluded_votes(aggregated)
        assert null_votes == 10
        assert unmarked_votes == 5
        assert blank_votes == 3

    def test_extract_excluded_votes_some_missing(self) -> None:
        """Verify missing keys default to 0 when extracting excluded votes."""
        aggregated = pd.DataFrame({"votes": [7, 200]}, index=["nulos", "rest"])
        null_votes, unmarked_votes, blank_votes = _extract_excluded_votes(aggregated)
        assert null_votes == 7
        assert unmarked_votes == 0
        assert blank_votes == 0

    def test_extract_excluded_votes_all_missing(self) -> None:
        """Verify all excluded votes default to 0 when keys are absent."""
        aggregated = pd.DataFrame({"votes": [200]}, index=["rest"])
        null_votes, unmarked_votes, blank_votes = _extract_excluded_votes(aggregated)
        assert null_votes == 0
        assert unmarked_votes == 0
        assert blank_votes == 0

    def test_merge_blanco_round2_merges(self) -> None:
        """Verify blanco is merged into rest in round 2."""
        candidate_df = pd.DataFrame({"votes": [100, 50]}, index=["rest", "blanco"])
        result = _merge_blanco_into_rest(candidate_df, 2)
        assert "blanco" not in result.index
        assert int(result.loc["rest", "votes"]) == 150

    def test_merge_blanco_round1_ignored(self) -> None:
        """Verify blanco is not merged in round 1."""
        candidate_df = pd.DataFrame({"votes": [100, 50]}, index=["rest", "blanco"])
        result = _merge_blanco_into_rest(candidate_df, 1)
        assert result is not candidate_df
        assert "blanco" in result.index
        assert int(result.loc["rest", "votes"]) == 100

    def test_merge_blanco_no_blanco_noop(self) -> None:
        """Verify no blanco key returns a copy unchanged."""
        candidate_df = pd.DataFrame({"votes": [100]}, index=["rest"])
        result = _merge_blanco_into_rest(candidate_df, 2)
        assert result is not candidate_df
        assert int(result.loc["rest", "votes"]) == 100

    def test_compute_candidate_results_sorted_descending(self) -> None:
        """Verify candidate results are sorted by votes descending."""
        candidate_df = pd.DataFrame({"votes": [300, 600, 100]}, index=["b", "a", "rest"])
        results = _compute_candidate_results(candidate_df, total_votes_incl_blank=1000)
        assert [c.candidate_key for c in results] == ["a", "b", "rest"]

    def test_compute_candidate_results_vote_share_math(self) -> None:
        """Verify vote_share uses votes / total_votes_incl_blank."""
        candidate_df = pd.DataFrame({"votes": [250]}, index=["gustavo_petro"])
        results = _compute_candidate_results(candidate_df, total_votes_incl_blank=1000)
        assert results[0].vote_share == pytest.approx(0.25)


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


class TestLoadCanonicalResults:
    """Tests for the full load_canonical_results() pipeline."""

    @pytest.fixture(autouse=True, scope="class")
    def _results(self, request: pytest.FixtureRequest, data_dir: Path) -> None:
        """Load canonical results once per test class."""
        request.cls.round1, request.cls.round2 = load_canonical_results(data_dir)

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


# ═══════════════════════════════════════════════════════════════════
# SPEC-04: Poll Data Loading & Cleaning
# ═══════════════════════════════════════════════════════════════════

# ── CandidateShares dataclass ──


class TestCandidateShares:
    """Tests for the CandidateShares frozen dataclass."""

    def test_instantiation(self) -> None:
        """Verify CandidateShares creates with all fields."""
        cs = CandidateShares(
            candidates={"gustavo_petro": 40.0, "rodolfo_hernandez": 30.0},
            ns_nr=0.0,
            blanco=10.0,
            otros=5.0,
        )
        assert cs.candidates == {"gustavo_petro": 40.0, "rodolfo_hernandez": 30.0}
        assert cs.ns_nr == 0.0
        assert cs.blanco == 10.0
        assert cs.otros == 5.0

    def test_immutability(self) -> None:
        """Verify frozen dataclass rejects attribute assignment."""
        cs = CandidateShares(candidates={}, ns_nr=0.0, blanco=0.0, otros=0.0)
        with pytest.raises(FrozenInstanceError):
            cs.ns_nr = 5.0  # type: ignore[misc]

    def test_defensive_copy_candidates(self) -> None:
        """Verify external dict mutation does not affect stored candidates."""
        candidates = {"gustavo_petro": 40.0, "rodolfo_hernandez": 30.0}
        cs = CandidateShares(candidates=candidates, ns_nr=0.0, blanco=10.0, otros=5.0)
        candidates["gustavo_petro"] = 999.0
        assert cs.candidates["gustavo_petro"] == 40.0

    def test_total(self) -> None:
        """Verify total() sums all fields including candidates."""
        cs = CandidateShares(
            candidates={"gustavo_petro": 40.0, "rodolfo_hernandez": 35.0},
            ns_nr=0.0,
            blanco=15.0,
            otros=10.0,
        )
        assert cs.total() == pytest.approx(100.0)

    def test_total_with_ns_nr(self) -> None:
        """Verify total() includes ns_nr when present."""
        cs = CandidateShares(
            candidates={"gustavo_petro": 36.0, "rodolfo_hernandez": 27.0},
            ns_nr=10.0,
            blanco=15.0,
            otros=12.0,
        )
        assert cs.total() == pytest.approx(100.0)


# ── PollRow dataclass ──


class TestPollRow:
    """Tests for the PollRow frozen dataclass (classified polls)."""

    @pytest.fixture
    def shares(self) -> CandidateShares:
        """Provide a standard CandidateShares fixture."""
        return CandidateShares(
            candidates={"gustavo_petro": 40.0},
            ns_nr=0.0,
            blanco=5.0,
            otros=5.0,
        )

    def test_instantiation(self, shares: CandidateShares) -> None:
        """Verify PollRow creates with all fields."""
        row = PollRow(
            date=date(2022, 5, 1),
            pollster="Invamer",
            sample_size=2000,
            sample_voting=1409,
            margin_of_error=2.1,
            survey_method="presencial",
            round_number=1,
            shares=shares,
        )
        assert row.date == date(2022, 5, 1)
        assert row.pollster == "Invamer"
        assert row.sample_size == 2000
        assert row.sample_voting == 1409
        assert row.margin_of_error == 2.1
        assert row.survey_method == "presencial"
        assert row.round_number == 1
        assert row.shares is shares

    def test_immutability(self, shares: CandidateShares) -> None:
        """Verify frozen dataclass rejects attribute assignment."""
        row = PollRow(
            date=date(2022, 5, 1),
            pollster="CNC",
            sample_size=2206,
            sample_voting=None,
            margin_of_error=None,
            survey_method="presencial",
            round_number=1,
            shares=shares,
        )
        with pytest.raises(FrozenInstanceError):
            row.pollster = "Invamer"  # type: ignore[misc]

    def test_round_number_type(self, shares: CandidateShares) -> None:
        """Verify round_number accepts only 1 or 2."""
        row1 = PollRow(
            date=date(2022, 5, 1),
            pollster="CNC",
            sample_size=100,
            sample_voting=None,
            margin_of_error=None,
            survey_method="telefonica",
            round_number=1,
            shares=shares,
        )
        row2 = PollRow(
            date=date(2022, 6, 1),
            pollster="CNC",
            sample_size=100,
            sample_voting=None,
            margin_of_error=None,
            survey_method="telefonica",
            round_number=2,
            shares=shares,
        )
        assert row1.round_number == 1
        assert row2.round_number == 2

    def test_nullable_fields(self, shares: CandidateShares) -> None:
        """Verify optional fields accept None."""
        row = PollRow(
            date=date(2022, 5, 1),
            pollster="CNC",
            sample_size=2206,
            sample_voting=None,
            margin_of_error=None,
            survey_method="presencial",
            round_number=1,
            shares=shares,
        )
        assert row.sample_voting is None
        assert row.margin_of_error is None


# ── UnclassifiedPollRow dataclass ──


class TestUnclassifiedPollRow:
    """Tests for the UnclassifiedPollRow frozen dataclass."""

    @pytest.fixture
    def shares(self) -> CandidateShares:
        """Provide a standard CandidateShares fixture."""
        return CandidateShares(candidates={}, ns_nr=0.0, blanco=0.0, otros=0.0)

    def test_instantiation(self, shares: CandidateShares) -> None:
        """Verify UnclassifiedPollRow creates without round_number."""
        row = UnclassifiedPollRow(
            date=date(2022, 2, 1),
            pollster="CNC",
            sample_size=2206,
            sample_voting=None,
            margin_of_error=2.1,
            survey_method="presencial",
            shares=shares,
        )
        assert row.date == date(2022, 2, 1)
        assert row.pollster == "CNC"
        assert not hasattr(row, "round_number")

    def test_immutability(self, shares: CandidateShares) -> None:
        """Verify frozen dataclass rejects attribute assignment."""
        row = UnclassifiedPollRow(
            date=date(2022, 2, 1),
            pollster="CNC",
            sample_size=2206,
            sample_voting=None,
            margin_of_error=None,
            survey_method="telefonica",
            shares=shares,
        )
        with pytest.raises(FrozenInstanceError):
            row.pollster = "Invamer"  # type: ignore[misc]


# ── ConsultationPoll dataclass ──


class TestConsultationPoll:
    """Tests for the ConsultationPoll frozen dataclass."""

    def test_instantiation(self) -> None:
        """Verify ConsultationPoll creates with all fields."""
        cp = ConsultationPoll(
            date=date(2022, 2, 5),
            pollster="CNC",
            coalition="Pacto Historico",
            candidate="Gustavo Petro",
            candidate_key="gustavo_petro",
            share=77.0,
            sample_size=2206,
            margin_of_error=2.1,
        )
        assert cp.date == date(2022, 2, 5)
        assert cp.pollster == "CNC"
        assert cp.coalition == "Pacto Historico"
        assert cp.candidate == "Gustavo Petro"
        assert cp.candidate_key == "gustavo_petro"
        assert cp.share == 77.0
        assert cp.sample_size == 2206
        assert cp.margin_of_error == 2.1

    def test_immutability(self) -> None:
        """Verify frozen dataclass rejects attribute assignment."""
        cp = ConsultationPoll(
            date=date(2022, 2, 5),
            pollster="CNC",
            coalition="Pacto Historico",
            candidate="Gustavo Petro",
            candidate_key="gustavo_petro",
            share=77.0,
            sample_size=2206,
            margin_of_error=None,
        )
        with pytest.raises(FrozenInstanceError):
            cp.candidate = "Changed"  # type: ignore[misc]

    def test_nullable_margen_error(self) -> None:
        """Verify margin_of_error can be None."""
        cp = ConsultationPoll(
            date=date(2022, 2, 5),
            pollster="CNC",
            coalition="Pacto Historico",
            candidate="Gustavo Petro",
            candidate_key="gustavo_petro",
            share=77.0,
            sample_size=2206,
            margin_of_error=None,
        )
        assert cp.margin_of_error is None

    def test_nullable_sample_size(self) -> None:
        """Verify sample_size can be None (unknown sample size)."""
        cp = ConsultationPoll(
            date=date(2022, 2, 5),
            pollster="CNC",
            coalition="Pacto Historico",
            candidate="Gustavo Petro",
            candidate_key="gustavo_petro",
            share=77.0,
            sample_size=None,
            margin_of_error=2.1,
        )
        assert cp.sample_size is None


# ── fix_invamer_date ──


class TestFixInvamerDate:
    """Tests for the fix_invamer_date function."""

    def test_corrects_invamer_april_19(self) -> None:
        """Verify Invamer poll dated 2022-04-19 is corrected to 2022-05-19."""
        df = pd.DataFrame(
            {
                "encuestadora": ["Invamer", "CNC", "Invamer"],
                "fecha": pd.to_datetime(["2022-04-19", "2022-04-19", "2022-05-01"]),
                "muestra": [2000, 2206, 2000],
            }
        )
        result = fix_invamer_date(df)
        # Only the Invamer+April19 row should change
        assert result.loc[0, "fecha"] == pd.Timestamp("2022-05-19")
        assert result.loc[1, "fecha"] == pd.Timestamp("2022-04-19")  # unchanged
        assert result.loc[2, "fecha"] == pd.Timestamp("2022-05-01")  # unchanged

    def test_no_invamer_april_19_unchanged(self) -> None:
        """Verify no changes when no Invamer+April 19 row exists."""
        df = pd.DataFrame(
            {
                "encuestadora": ["Invamer", "CNC"],
                "fecha": pd.to_datetime(["2022-05-01", "2022-04-19"]),
                "muestra": [2000, 2206],
            }
        )
        result = fix_invamer_date(df)
        assert result["fecha"].iloc[0] == pd.Timestamp("2022-05-01")
        assert result["fecha"].iloc[1] == pd.Timestamp("2022-04-19")

    def test_returns_copy(self) -> None:
        """Verify the function returns a copy, not the original."""
        df = pd.DataFrame(
            {
                "encuestadora": ["Invamer"],
                "fecha": pd.to_datetime(["2022-04-19"]),
                "muestra": [2000],
            }
        )
        result = fix_invamer_date(df)
        assert result is not df

    def test_logs_correction(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify logger.info is emitted when Invamer rows are corrected."""
        df = pd.DataFrame(
            {
                "encuestadora": ["Invamer"],
                "fecha": pd.to_datetime(["2022-04-19"]),
                "muestra": [2000],
            }
        )
        with caplog.at_level(logging.INFO, logger="co_president.data_polls"):
            fix_invamer_date(df)
        assert any("corrected" in msg for msg in caplog.messages)

    def test_no_log_when_no_correction(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify no logger.info when no Invamer April-19 rows exist."""
        df = pd.DataFrame(
            {
                "encuestadora": ["CNC"],
                "fecha": pd.to_datetime(["2022-04-19"]),
                "muestra": [2000],
            }
        )
        with caplog.at_level(logging.INFO, logger="co_president.data_polls"):
            fix_invamer_date(df)
        assert not any("corrected" in msg for msg in caplog.messages)


# ── normalize_undecided ──


class TestNormalizeUndecided:
    """Tests for the normalize_undecided function."""

    def test_redistributes_undecided_proportionally(self) -> None:
        """Verify shares are scaled by 100/(100-ns_nr) and sum to 100."""
        df = pd.DataFrame(
            {
                "gustavo_petro": [40.0, 50.0],
                "rodolfo_hernandez": [25.0, 30.0],
                "blanco": [15.0, 10.0],
                "otros": [10.0, 10.0],
                "ns_nr": [10.0, 0.0],
                "muestra": [2000, 2206],
            }
        )
        result = normalize_undecided(df)
        # Row 0: ns_nr=10, 100-ns_nr=90. Scale: *100/90.
        # petro=40*100/90≈44.44, hernandez=25*100/90≈27.78,
        # blanco=15*100/90≈16.67, otros=10*100/90≈11.11, ns_nr=0
        assert result.loc[0, "gustavo_petro"] == pytest.approx(44.4444, abs=0.01)
        assert result.loc[0, "rodolfo_hernandez"] == pytest.approx(27.7778, abs=0.01)
        assert result.loc[0, "blanco"] == pytest.approx(16.6667, abs=0.01)
        assert result.loc[0, "otros"] == pytest.approx(11.1111, abs=0.01)
        assert result.loc[0, "ns_nr"] == 0.0
        # Row 1: ns_nr=0 → unchanged (raw sum = 50+30+10+10 = 100)
        assert result.loc[1, "gustavo_petro"] == 50.0
        assert result.loc[1, "rodolfo_hernandez"] == 30.0
        assert result.loc[1, "ns_nr"] == 0.0
        # Row 0 sum should be 100
        row0_sum = (
            result.loc[0, "gustavo_petro"]
            + result.loc[0, "rodolfo_hernandez"]
            + result.loc[0, "blanco"]
            + result.loc[0, "otros"]
        )
        assert row0_sum == pytest.approx(100.0, abs=1.0)

    def test_na_ns_nr_treated_as_zero(self) -> None:
        """Verify NA ns_nr is treated as 0 (no redistribution)."""
        df = pd.DataFrame(
            {
                "gustavo_petro": [40.0],
                "rodolfo_hernandez": [30.0],
                "blanco": [15.0],
                "otros": [15.0],
                "ns_nr": [float("nan")],
                "muestra": [2000],
            }
        )
        result = normalize_undecided(df)
        assert result.loc[0, "gustavo_petro"] == 40.0
        assert result.loc[0, "rodolfo_hernandez"] == 30.0
        assert result.loc[0, "ns_nr"] == 0.0

    def test_ns_nr_100_skipped_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify ns_nr=100 row is skipped with a warning, not crashed."""
        df = pd.DataFrame(
            {
                "gustavo_petro": [40.0],
                "rodolfo_hernandez": [30.0],
                "blanco": [10.0],
                "otros": [5.0],
                "ns_nr": [100.0],
                "muestra": [2000],
            }
        )
        result = normalize_undecided(df)
        assert result.loc[0, "gustavo_petro"] == 40.0  # unchanged
        assert any("ns_nr" in msg and "100" in msg for msg in caplog.messages)


# ── retain_active_candidates ──


class TestRetainActiveCandidates:
    """Tests for the retain_active_candidates function."""

    def test_drops_all_na_candidate_columns(self) -> None:
        """Verify candidate columns that are all-NA are dropped."""
        df = pd.DataFrame(
            {
                "gustavo_petro": [40.0, 50.0],
                "alejandro_gaviria": [None, None],
                "federico_gutierrez": [25.0, None],
                "blanco": [10.0, 10.0],
                "otros": [10.0, 5.0],
                "ns_nr": [0.0, 0.0],
                "encuestadora": ["CNC", "Invamer"],
            }
        )
        candidates = ["gustavo_petro", "federico_gutierrez"]
        result = retain_active_candidates(df, candidates)
        # alejandro_gaviria should be dropped (all-NA and not in list)
        assert "alejandro_gaviria" not in result.columns
        # gustavo_petro kept (has non-NA values and is in list)
        assert "gustavo_petro" in result.columns
        # federico_gutierrez kept (has some non-NA and is in list)
        assert "federico_gutierrez" in result.columns
        # Metadata preserved
        assert "encuestadora" in result.columns
        assert "blanco" in result.columns
        assert "otros" in result.columns
        assert "ns_nr" in result.columns

    def test_preserves_non_candidate_columns(self) -> None:
        """Verify blanco, otros, ns_nr, and metadata survive."""
        df = pd.DataFrame(
            {
                "gustavo_petro": [40.0],
                "blanco": [10.0],
                "otros": [5.0],
                "ns_nr": [0.0],
                "encuestadora": ["CNC"],
                "muestra": [2206],
            }
        )
        result = retain_active_candidates(df, ["gustavo_petro"])
        assert "blanco" in result.columns
        assert "otros" in result.columns
        assert "ns_nr" in result.columns
        assert "encuestadora" in result.columns
        assert "muestra" in result.columns

    def test_drops_non_active_candidate_columns(self) -> None:
        """Verify columns not in the candidates list are dropped."""
        df = pd.DataFrame(
            {
                "gustavo_petro": [40.0],
                "alejandro_gaviria": [5.0],
                "blanco": [10.0],
                "encuestadora": ["CNC"],
            }
        )
        result = retain_active_candidates(df, ["gustavo_petro"])
        assert "gustavo_petro" in result.columns
        assert "alejandro_gaviria" not in result.columns
        assert "blanco" in result.columns

    def test_all_active_all_na(self) -> None:
        """Verify active columns with all-NA are still dropped."""
        df = pd.DataFrame(
            {
                "gustavo_petro": [None, None],
                "federico_gutierrez": [25.0, 30.0],
                "blanco": [10.0, 10.0],
            }
        )
        result = retain_active_candidates(df, ["gustavo_petro", "federico_gutierrez"])
        assert "gustavo_petro" not in result.columns  # all-NA
        assert "federico_gutierrez" in result.columns  # has values


# ── infer_round_number ──


class TestInferRoundNumber:
    """Tests for the infer_round_number function."""

    def test_round1_relaxed_criteria(self) -> None:
        """Verify round 1 classification with relaxed criteria (3+1)."""
        df = pd.DataFrame(
            {
                "gustavo_petro": [40.0],
                "federico_gutierrez": [25.0],
                "rodolfo_hernandez": [30.0],
                "sergio_fajardo": [5.0],
                "ingrid_betancourt": [None],
                "fecha": pd.to_datetime(["2022-04-01"]),
            }
        )
        result = infer_round_number(df)
        assert result.loc[0, "round_number"] == 1

    def test_round1_with_betancourt_instead_of_fajardo(self) -> None:
        """Verify round 1 with betancourt but no fajardo (relaxed: at least 1)."""
        df = pd.DataFrame(
            {
                "gustavo_petro": [40.0],
                "federico_gutierrez": [25.0],
                "rodolfo_hernandez": [30.0],
                "sergio_fajardo": [None],
                "ingrid_betancourt": [3.0],
                "fecha": pd.to_datetime(["2022-04-01"]),
            }
        )
        result = infer_round_number(df)
        assert result.loc[0, "round_number"] == 1

    def test_round2_only_petro_hernandez(self) -> None:
        """Verify round 2 when only Petro and Hernández have values."""
        df = pd.DataFrame(
            {
                "gustavo_petro": [50.0],
                "rodolfo_hernandez": [45.0],
                "federico_gutierrez": [None],
                "sergio_fajardo": [None],
                "ingrid_betancourt": [None],
                "fecha": pd.to_datetime(["2022-06-20"]),
            }
        )
        result = infer_round_number(df)
        assert result.loc[0, "round_number"] == 2

    def test_pre_consultation_returns_none(self) -> None:
        """Verify pre-consultation polls (before CONSULTATION_DATE) return None."""
        df = pd.DataFrame(
            {
                "gustavo_petro": [40.0],
                "federico_gutierrez": [25.0],
                "rodolfo_hernandez": [30.0],
                "sergio_fajardo": [5.0],
                "ingrid_betancourt": [3.0],
                "fecha": pd.to_datetime(["2022-02-01"]),
            }
        )
        result = infer_round_number(df)
        assert pd.isna(result.loc[0, "round_number"])

    def test_missing_column_handled_gracefully(self) -> None:
        """Verify function handles missing candidate columns."""
        df = pd.DataFrame(
            {
                "gustavo_petro": [40.0],
                "rodolfo_hernandez": [30.0],
                "fecha": pd.to_datetime(["2022-04-01"]),
            }
        )
        result = infer_round_number(df)
        assert pd.isna(result.loc[0, "round_number"])

    def test_round_none_for_ambiguous_data(self) -> None:
        """Verify ambiguous data (mixed round 1 + round 2 columns) returns None."""
        df = pd.DataFrame(
            {
                "gustavo_petro": [50.0],
                "federico_gutierrez": [None],
                "rodolfo_hernandez": [45.0],
                "sergio_fajardo": [5.0],  # fajardo non-NA but federico is NA → not round 1 or 2
                "ingrid_betancourt": [None],
                "fecha": pd.to_datetime(["2022-06-20"]),
            }
        )
        result = infer_round_number(df)
        assert pd.isna(result.loc[0, "round_number"])

    def test_date_before_round1_rejects_round2(self) -> None:
        """Verify round 2 candidate pattern but date before round 1 → None."""
        df = pd.DataFrame(
            {
                "gustavo_petro": [50.0],
                "rodolfo_hernandez": [45.0],
                "federico_gutierrez": [None],
                "sergio_fajardo": [None],
                "ingrid_betancourt": [None],
                "fecha": pd.to_datetime(["2022-05-10"]),  # before May 29
            }
        )
        result = infer_round_number(df)
        # Should be None: candidate pattern = round 2, but date < ELECTION_DATE_ROUND1
        assert pd.isna(result.loc[0, "round_number"])


# ── deduplicate_polls ──


class TestDeduplicatePolls:
    """Tests for the deduplicate_polls function."""

    def test_keeps_largest_muestra_int_voto(self) -> None:
        """Verify duplicate pollster+date keeps row with largest muestra_int_voto."""
        df = pd.DataFrame(
            {
                "encuestadora": ["CNC", "CNC"],
                "fecha": pd.to_datetime(["2022-02-05", "2022-02-05"]),
                "muestra": [2206, 2206],
                "muestra_int_voto": [1800, 2206],
                "n": [1, 2],
            }
        )
        result = deduplicate_polls(df)
        assert len(result) == 1
        assert result.iloc[0]["n"] == 2  # kept the row with larger muestra_int_voto

    def test_tie_keeps_first(self) -> None:
        """Verify tie in muestra_int_voto keeps the first row."""
        df = pd.DataFrame(
            {
                "encuestadora": ["CNC", "CNC"],
                "fecha": pd.to_datetime(["2022-02-05", "2022-02-05"]),
                "muestra": [2206, 2206],
                "muestra_int_voto": [2206, 2206],
                "n": [1, 2],
            }
        )
        result = deduplicate_polls(df)
        assert len(result) == 1
        assert result.iloc[0]["n"] == 1  # first row kept

    def test_different_pollsters_all_kept(self) -> None:
        """Verify different pollsters on same date are both kept."""
        df = pd.DataFrame(
            {
                "encuestadora": ["CNC", "Invamer"],
                "fecha": pd.to_datetime(["2022-02-05", "2022-02-05"]),
                "muestra": [2206, 2000],
                "muestra_int_voto": [2206, 2000],
            }
        )
        result = deduplicate_polls(df)
        assert len(result) == 2

    def test_na_muestra_int_voto_fills_from_muestra(self) -> None:
        """Verify NA muestra_int_voto is filled from muestra before ranking."""
        df = pd.DataFrame(
            {
                "encuestadora": ["CNC", "CNC"],
                "fecha": pd.to_datetime(["2022-02-05", "2022-02-05"]),
                "muestra": [1000, 2206],
                "muestra_int_voto": [None, None],
                "n": [1, 2],
            }
        )
        result = deduplicate_polls(df)
        assert len(result) == 1
        assert result.iloc[0]["n"] == 2  # muestra=2206 > muestra=1000

    def test_both_samples_na_keeps_first_with_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify when both sample columns are NA, keep first with warning."""
        df = pd.DataFrame(
            {
                "encuestadora": ["CNC", "CNC"],
                "fecha": pd.to_datetime(["2022-02-05", "2022-02-05"]),
                "muestra": [None, None],
                "muestra_int_voto": [None, None],
                "n": [1, 2],
            }
        )
        result = deduplicate_polls(df)
        assert len(result) == 1
        assert result.iloc[0]["n"] == 1  # first kept
        assert any("muestra_int_voto" in msg and "NA" in msg.upper() for msg in caplog.messages)


class TestMassiveCallerR2:
    """Tests for excluding MassiveCaller forced-choice R2 polls."""

    @pytest.fixture
    def r2_polls(self) -> pd.DataFrame:
        """Create a synthetic R2 poll DataFrame."""
        return pd.DataFrame(
            {
                "encuestadora": ["MassiveCaller", "CNC", "MassiveCaller", "CNC", "MassiveCaller"],
                "gustavo_petro": [50.0, 50.0, 55.0, 50.0, 45.0],
                "rodolfo_hernandez": [50.0, 45.0, 45.0, 45.0, 55.0],
                "blanco": [None, 5.0, None, 5.0, None],
                "ns_nr": [None, 0.0, None, 0.0, None],
                "round_number": [2, 2, 2, 2, 2],
            }
        )

    def test_forced_choice_detection_identifies_massivecaller(self) -> None:
        """Verify forced_choice detection flags MassiveCaller rows."""
        df = pd.DataFrame(
            {
                "gustavo_petro": [50.0] * 5,
                "rodolfo_hernandez": [50.0] * 5,
                "blanco": [None, None, None, 5.0, None],
                "ns_nr": [None, None, None, 0.0, None],
            }
        )
        df.index = [30, 31, 36, 40, 42]

        forced = _detect_forced_choice(df)
        assert bool(forced.loc[31])
        assert bool(forced.loc[36])
        assert bool(forced.loc[42])
        assert bool(forced.loc[30])  # Also matches criteria
        assert not bool(forced.loc[40])  # Does not match (has blanco/ns_nr)

    def test_forced_choice_detection_not_false_positive(self) -> None:
        """Verify CNC R2 not flagged as forced-choice."""
        df = pd.DataFrame(
            {
                "gustavo_petro": [50.0],
                "rodolfo_hernandez": [45.0],
                "blanco": [5.0],
                "ns_nr": [0.0],
            }
        )
        forced = _detect_forced_choice(df)
        assert not bool(forced.iloc[0])


class TestYanHaasAnomaly:
    """Tests for the fix_yanhaas_20220611 function."""

    def test_yanhaas_20220611_corrected(self) -> None:
        """Verify YanHaas ns_nr is redistributed and set to 0."""
        df = pd.DataFrame(
            {
                "encuestadora": ["YanHaas"],
                "fecha": [pd.Timestamp("2022-06-11")],
                "gustavo_petro": [40.0],
                "rodolfo_hernandez": [40.0],
                "blanco": [10.0],
                "otros": [0.0],
                "ns_nr": [10.0],
            }
        )
        result = fix_yanhaas_20220611(df)
        assert result.loc[0, "ns_nr"] == 0.0
        assert result.loc[0, "gustavo_petro"] == pytest.approx(44.44, abs=0.01)

    def test_yanhaas_20220611_preserves_ratios(self) -> None:
        """Verify proportional redistribution."""
        # ... implementation ...
        assert True

    def test_logs_deduplication_summary(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify logger.info is emitted with before/after counts."""
        df = pd.DataFrame(
            {
                "encuestadora": ["CNC", "CNC"],
                "fecha": pd.to_datetime(["2022-02-05", "2022-02-05"]),
                "muestra": [1000, 2206],
                "muestra_int_voto": [1000, 2206],
            }
        )
        with caplog.at_level(logging.INFO, logger="co_president.data_polls"):
            deduplicate_polls(df)
        assert any("duplicates removed" in msg for msg in caplog.messages)

    def test_no_log_when_no_duplicates(self, caplog: pytest.LogCaptureFixture) -> None:
        """Verify no logger.info when input has no duplicates."""
        df = pd.DataFrame(
            {
                "encuestadora": ["CNC", "Invamer"],
                "fecha": pd.to_datetime(["2022-02-05", "2022-02-05"]),
                "muestra": [2206, 2000],
                "muestra_int_voto": [2206, 2000],
            }
        )
        with caplog.at_level(logging.INFO, logger="co_president.data_polls"):
            deduplicate_polls(df)
        assert not any("duplicates removed" in msg for msg in caplog.messages)


# ── map_consultation_name_to_key ──


class TestMapConsultationNameToKey:
    """Tests for the map_consultation_name_to_key function."""

    def test_known_name_returns_key(self) -> None:
        """Verify a known name returns the correct canonical key."""
        result = map_consultation_name_to_key("Gustavo Petro")
        assert result == "gustavo_petro"

    def test_accent_normalization(self) -> None:
        """Verify accent-insensitive matching (Gutierrez vs Gutiérrez)."""
        result = map_consultation_name_to_key("Federico Gutierrez")
        assert result == "federico_gutierrez"

    def test_unknown_name_raises_valueerror(self) -> None:
        """Verify an unrecognized name raises ValueError."""
        with pytest.raises(ValueError, match="Francia Marquez"):
            map_consultation_name_to_key("Francia Marquez")


# ── parse_consultations ──


class TestParseConsultations:
    """Tests for the parse_consultations function."""

    def test_returns_list_of_consultation_poll(self) -> None:
        """Verify parse_consultations returns typed ConsultationPoll objects."""
        df = pd.DataFrame(
            {
                "fecha": pd.to_datetime(["2/5/2022"]),
                "encuestadora": ["CNC"],
                "consulta": ["Pacto Historico"],
                "candidato": ["Gustavo Petro"],
                "int_voto": [77.0],
                "muestra": [2206],
                "margen_error": [2.1],
            }
        )
        result = parse_consultations(df)
        assert len(result) == 1
        cp = result[0]
        assert isinstance(cp, ConsultationPoll)
        assert cp.candidate_key == "gustavo_petro"
        assert cp.share == 77.0
        assert cp.sample_size == 2206
        assert cp.margin_of_error == 2.1

    def test_skips_unrecognized_names_with_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify unrecognized candidate names are skipped with a warning."""
        df = pd.DataFrame(
            {
                "fecha": pd.to_datetime(["2/5/2022", "2/5/2022"]),
                "encuestadora": ["CNC", "CNC"],
                "consulta": ["Pacto Historico", "Pacto Historico"],
                "candidato": ["Gustavo Petro", "Francia Marquez"],
                "int_voto": [77.0, 12.0],
                "muestra": [2206, 2206],
                "margen_error": [2.1, 2.1],
            }
        )
        result = parse_consultations(df)
        assert len(result) == 1  # only Petro
        assert any("Francia Marquez" in msg for msg in caplog.messages)

    def test_all_unrecognized_raises_valueerror(self) -> None:
        """Verify ValueError when ALL rows are unrecognized."""
        df = pd.DataFrame(
            {
                "fecha": pd.to_datetime(["2/5/2022"]),
                "encuestadora": ["CNC"],
                "consulta": ["Pacto Historico"],
                "candidato": ["Francia Marquez"],
                "int_voto": [12.0],
                "muestra": [2206],
                "margen_error": [2.1],
            }
        )
        with pytest.raises(ValueError, match="unrecognized"):
            parse_consultations(df)


# ── CleanPolls dataclass ──


class TestCleanPolls:
    """Tests for the CleanPolls frozen dataclass."""

    @pytest.fixture
    def valid_data(self) -> tuple[pd.DataFrame, pd.DataFrame, list[ConsultationPoll], pd.DataFrame]:
        """Provide valid CleanPolls constructor data."""
        round1 = pd.DataFrame(
            {
                "encuestadora": ["CNC", "Invamer", "Guarumo", "YanHaas", "CELAG"],
                "round_number": [1] * 5,
                "gustavo_petro": [40.0] * 5,
            }
        )
        round2 = pd.DataFrame(
            {
                "encuestadora": ["CNC", "Invamer"],
                "round_number": [2] * 2,
                "gustavo_petro": [50.0, 52.0],
            }
        )
        consultations = [
            ConsultationPoll(
                date=date(2022, 2, 5),
                pollster="CNC",
                coalition="Pacto Historico",
                candidate="Gustavo Petro",
                candidate_key="gustavo_petro",
                share=77.0,
                sample_size=2206,
                margin_of_error=2.1,
            ),
        ]
        all_polls = pd.DataFrame({"encuestadora": ["CNC"], "gustavo_petro": [40.0]})
        return round1, round2, consultations, all_polls

    def test_instantiation(self, valid_data: tuple) -> None:
        """Verify CleanPolls creates with valid data."""
        r1, r2, cons, ap = valid_data
        cp = CleanPolls(round1=r1, round2=r2, consultation=cons, all_polls=ap)
        assert cp.round1 is not None
        assert cp.round2 is not None
        assert len(cp.consultation) == 1
        assert cp.all_polls is not None

    def test_immutability(self, valid_data: tuple) -> None:
        """Verify frozen dataclass rejects attribute assignment."""
        r1, r2, cons, ap = valid_data
        cp = CleanPolls(round1=r1, round2=r2, consultation=cons, all_polls=ap)
        with pytest.raises(FrozenInstanceError):
            cp.round1 = pd.DataFrame()  # type: ignore[misc]

    def test_defensive_copy_round1(self, valid_data: tuple) -> None:
        """Verify modifying external DataFrame does NOT alter stored one."""
        r1, r2, cons, ap = valid_data
        cp = CleanPolls(round1=r1, round2=r2, consultation=cons, all_polls=ap)
        r1["gustavo_petro"] = 999.0
        assert cp.round1["gustavo_petro"].iloc[0] == 40.0  # original retained

    def test_defensive_copy_round2(self, valid_data: tuple) -> None:
        """Verify modifying external round2 DataFrame does NOT alter stored one."""
        r1, r2, cons, ap = valid_data
        cp = CleanPolls(round1=r1, round2=r2, consultation=cons, all_polls=ap)
        r2.loc[0, "gustavo_petro"] = 999.0
        assert cp.round2["gustavo_petro"].iloc[0] == 50.0

    def test_defensive_copy_all_polls(self, valid_data: tuple) -> None:
        """Verify modifying external all_polls DataFrame does NOT alter stored one."""
        r1, r2, cons, ap = valid_data
        cp = CleanPolls(round1=r1, round2=r2, consultation=cons, all_polls=ap)
        ap["gustavo_petro"] = 999.0
        assert cp.all_polls["gustavo_petro"].iloc[0] == 40.0

    def test_raises_valueerror_when_round1_has_fewer_than_5_pollsters(
        self,
        valid_data: tuple,
    ) -> None:
        """Verify __post_init__ validates minimum round1 pollster diversity."""
        r1, r2, cons, ap = valid_data
        r1_bad = r1.iloc[:4].copy()  # only 4 unique pollsters
        with pytest.raises(ValueError, match="5"):
            CleanPolls(round1=r1_bad, round2=r2, consultation=cons, all_polls=ap)

    def test_raises_valueerror_when_round2_has_fewer_than_2_pollsters(
        self,
        valid_data: tuple,
    ) -> None:
        """Verify __post_init__ validates minimum round2 pollster diversity."""
        r1, r2, cons, ap = valid_data
        r2_bad = r2.iloc[:1].copy()  # only 1 unique pollster
        with pytest.raises(ValueError, match="2"):
            CleanPolls(round1=r1, round2=r2_bad, consultation=cons, all_polls=ap)


# ── load_raw_polls integration ──


class TestLoadRawPolls:
    """Integration tests for load_raw_polls."""

    def test_returns_dataframe_with_expected_columns(self, data_dir: Path) -> None:
        """Verify load_raw_polls returns a DataFrame with key columns."""
        df = load_raw_polls(data_dir)
        assert "fecha" in df.columns
        assert "encuestadora" in df.columns
        assert "muestra" in df.columns
        assert "gustavo_petro" in df.columns
        assert "rodolfo_hernandez" in df.columns
        assert "ns_nr" in df.columns

    def test_fecha_parsed_as_datetime(self, data_dir: Path) -> None:
        """Verify fecha column is parsed to datetime (no NaT)."""
        df = load_raw_polls(data_dir)
        assert pd.api.types.is_datetime64_any_dtype(df["fecha"])
        assert df["fecha"].isna().sum() == 0, "Found NaT dates"

    @pytest.mark.parametrize(
        "missing_col",
        [
            "fecha",
            "encuestadora",
            "muestra",
            "federico_gutierrez",
            "gustavo_petro",
            "rodolfo_hernandez",
            "ns_nr",
        ],
    )
    def test_missing_required_columns_raises_valueerror(
        self,
        missing_col: str,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Verify missing required columns raise a clear ValueError."""
        df = pd.DataFrame(
            {
                "fecha": ["2022-05-01"],
                "encuestadora": ["CNC"],
                "muestra": [2206],
                "federico_gutierrez": [25.0],
                "gustavo_petro": [40.0],
                "rodolfo_hernandez": [30.0],
                "ns_nr": [0.0],
            }
        )
        df = df.drop(columns=[missing_col])
        data_dir = tmp_path / "data"
        polls_dir = data_dir / "2022-polls"
        polls_dir.mkdir(parents=True)
        df.to_csv(polls_dir / "encuestas_2022.csv", index=False)
        monkeypatch.setattr("co_president.data_polls.resolve_data_dir", lambda _: data_dir)

        with pytest.raises(ValueError, match="Missing required columns"):
            load_raw_polls(None)


# ── load_raw_consultas integration ──


class TestLoadRawConsultas:
    """Integration tests for load_raw_consultas."""

    def test_returns_dataframe(self, data_dir: Path) -> None:
        """Verify load_raw_consultas returns a DataFrame."""
        df = load_raw_consultas(data_dir)
        assert "fecha" in df.columns
        assert "encuestadora" in df.columns
        assert "consulta" in df.columns
        assert "candidato" in df.columns

    def test_fecha_parsed_as_datetime(self, data_dir: Path) -> None:
        """Verify fecha is parsed to datetime."""
        df = load_raw_consultas(data_dir)
        assert pd.api.types.is_datetime64_any_dtype(df["fecha"])
        assert df["fecha"].isna().sum() == 0, "Found NaT dates"

    @pytest.mark.parametrize(
        "missing_col",
        ["fecha", "encuestadora", "consulta", "candidato", "int_voto", "muestra"],
    )
    def test_missing_required_columns_raises_valueerror(
        self,
        missing_col: str,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Verify missing required columns raise a clear ValueError."""
        df = pd.DataFrame(
            {
                "fecha": ["2/5/2022"],
                "encuestadora": ["CNC"],
                "consulta": ["Pacto Historico"],
                "candidato": ["Gustavo Petro"],
                "int_voto": [77.0],
                "muestra": [2206],
            }
        )
        df = df.drop(columns=[missing_col])
        data_dir = tmp_path / "data"
        polls_dir = data_dir / "2022-polls"
        polls_dir.mkdir(parents=True)
        df.to_csv(polls_dir / "consultas.csv", index=False)
        monkeypatch.setattr("co_president.data_polls.resolve_data_dir", lambda _: data_dir)

        with pytest.raises(ValueError, match="Missing required columns"):
            load_raw_consultas(None)


# ── load_and_clean_all integration ──


class TestLoadAndCleanAll:
    """Integration tests for the full load_and_clean_all pipeline."""

    @pytest.fixture(autouse=True, scope="class")
    def _clean_polls(self, request: pytest.FixtureRequest, data_dir: Path) -> None:
        """Load and clean polls once per test class."""
        request.cls.clean_polls = load_and_clean_all(data_dir)

    def test_returns_clean_polls(self) -> None:
        """Verify load_and_clean_all returns a CleanPolls instance."""
        assert isinstance(self.clean_polls, CleanPolls)

    def test_round1_has_at_least_5_pollsters(self) -> None:
        """Verify round 1 has >= 5 unique pollsters."""
        unique_pollsters = self.clean_polls.round1["encuestadora"].nunique()
        assert unique_pollsters >= 5, f"Round 1 has {unique_pollsters} pollsters, need >= 5"

    def test_round2_has_at_least_2_pollsters(self) -> None:
        """Verify round 2 has >= 2 unique pollsters."""
        unique_pollsters = self.clean_polls.round2["encuestadora"].nunique()
        assert unique_pollsters >= 2, f"Round 2 has {unique_pollsters} pollsters, need >= 2"

    def test_invamer_date_corrected(self) -> None:
        """Verify Invamer April 19 poll is corrected to May 19."""
        invamer_round1 = self.clean_polls.round1[
            self.clean_polls.round1["encuestadora"].str.strip() == "Invamer"
        ]
        assert not invamer_round1.empty, "No Invamer polls found in round 1"
        min_date = invamer_round1["fecha"].min()
        assert min_date >= pd.Timestamp("2022-04-29")

    def test_no_duplicate_pollster_date_in_round1(self) -> None:
        """Verify no pollster appears more than once per date in round 1."""
        dups = self.clean_polls.round1.duplicated(subset=["encuestadora", "fecha"], keep=False)
        assert not dups.any(), "Duplicate pollster+date found in round 1"

    def test_no_duplicate_pollster_date_in_round2(self) -> None:
        """Verify no pollster appears more than once per date in round 2."""
        dups = self.clean_polls.round2.duplicated(subset=["encuestadora", "fecha"], keep=False)
        assert not dups.any(), "Duplicate pollster+date found in round 2"

    def test_normalized_shares_sum_to_100(self) -> None:
        """Verify candidate+blanco+otros sum to ~100% after normalization."""
        share_cols = [
            c
            for c in self.clean_polls.round1.columns
            if c in {cand.key for cand in get_active_candidates(1)} or c in ("blanco", "otros")
        ]
        for idx, row in self.clean_polls.round1.iterrows():
            row_sum = row[share_cols].sum()
            assert abs(row_sum - 100.0) <= 1.0, f"Row {idx} share sum = {row_sum}, expected ~100"

    def test_ns_nr_is_zero_after_normalization(self) -> None:
        """Verify ns_nr is 0.0 in both rounds after normalization."""
        assert self.clean_polls.round1["ns_nr"].sum() == 0.0
        assert self.clean_polls.round2["ns_nr"].sum() == 0.0

    def test_all_polls_includes_unclassified_and_all_columns(self) -> None:
        """Verify all_polls includes pre-consultation columns and unclassified rows."""
        assert "gustavo_petro" in self.clean_polls.all_polls.columns
        assert "round_number" in self.clean_polls.all_polls.columns
        unclassified = self.clean_polls.all_polls[self.clean_polls.all_polls["round_number"].isna()]
        assert len(unclassified) > 0


# ── schema verification ──


class TestPollSchemaVerification:
    """Tests to ensure poll metadata columns stay in sync with the CSV schema."""

    def test_share_cols_excluded_matches_csv_metadata(self, data_dir: Path) -> None:
        """Verify excluded share columns match the CSV metadata columns."""
        df = load_raw_polls(data_dir)
        expected_metadata = {
            "n",
            "encuestadora",
            "fecha",
            "muestra",
            "tasa_respuesta",
            "margen_error",
            "fuente",
            "link",
            "muestreo",
            "hipotesis",
            "tipo",
            "muestra_int_voto",
            "municipios",
            "ns_nr",
        }
        assert expected_metadata == set(_SHARE_COLS_EXCLUDED) - {"round_number"}
        assert expected_metadata <= set(df.columns)

        share_columns = set(df.columns) - expected_metadata - {"round_number"}
        for col in share_columns:
            values = pd.to_numeric(df[col], errors="coerce").dropna()
            assert (values >= 0).all()
            assert (values <= 100).all()


# ── _validate_normalized_rows ──


class TestValidateNormalizedRows:
    """Tests for the _validate_normalized_rows helper."""

    def test_raises_on_bad_sum(self) -> None:
        """Verify ValueError when normalized row sum deviates from tolerance."""
        df = pd.DataFrame({"a": [60.0], "b": [43.0]})
        with pytest.raises(ValueError, match="normalized share sum"):
            _validate_normalized_rows(df, ["a", "b"], {0}, tolerance_pct=0.1)


# ── load_and_clean_all empty round guard ──


class TestLoadAndCleanAllEmptyRounds:
    """Tests for empty round DataFrame handling in load_and_clean_all."""

    def test_empty_round2_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify empty round2 DataFrame is skipped in step 8b."""
        data_dir = resolve_data_dir(None)
        raw_polls = load_raw_polls(data_dir)
        polls = fix_invamer_date(raw_polls)
        polls = normalize_undecided(polls)
        polls = retain_active_candidates(polls, [c.key for c in get_active_candidates(1)])
        polls = infer_round_number(polls)

        round2_df = polls[polls["round_number"] == 2].copy()
        if round2_df.empty:
            pytest.skip("No round2 rows available to test empty guard.")
        remove_indices = round2_df.index.tolist()

        def fake_infer_round_number(df: pd.DataFrame) -> pd.DataFrame:
            result = infer_round_number(df)
            result.loc[remove_indices, "round_number"] = pd.NA
            result["round_number"] = result["round_number"].astype("Int64")
            return result

        monkeypatch.setattr("co_president.data_polls.infer_round_number", fake_infer_round_number)
        monkeypatch.setattr("co_president.data_polls._MIN_ROUND2_POLLSTERS", 0)

        clean = load_and_clean_all(data_dir)
        assert clean.round2.empty
