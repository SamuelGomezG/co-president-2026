"""SPEC-03: Tests for election results consolidation."""

from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
import logging
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from co_president.config import (
    COALITION_TO_CANDIDATE,
    ELECTION_DATE_ROUND1,
)
from co_president.data_results import (
    PDF_RESULTS_2022,
    CandidateResult,
    RoundResult,
    _aggregate_and_map,
    _build_round_result,
    _compute_candidate_results,
    _extract_excluded_votes,
    _extract_tables_with_fallback,
    _merge_blanco_into_rest,
    _read_mmv,
    _read_moe,
    _read_participation,
    _resolve_mmv_path,
    consolidate_round,
    cross_validate,
    cross_validate_against_moe_pdf,
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

    def test_read_mmv_missing_cannombre_raises(self, tmp_path: Path) -> None:
        """Verify missing CANNOMBRE column raises ValueError."""
        path = tmp_path / "mmv.csv"
        df = pd.DataFrame({"PARNOMBRE": ["A", "B"], "VOTOS": [1, 2]})
        df.to_csv(path, sep=";", index=False, encoding="latin-1")
        with pytest.raises(ValueError, match="CANNOMBRE"):
            _read_mmv(path)

    def test_read_mmv_missing_votos_raises(self, tmp_path: Path) -> None:
        """Verify missing VOTOS column raises ValueError."""
        path = tmp_path / "mmv.csv"
        df = pd.DataFrame({"PARNOMBRE": ["A", "B"]})
        df.to_csv(path, sep=";", index=False, encoding="latin-1")
        with pytest.raises(ValueError, match="VOTOS"):
            _read_mmv(path)

    def test_read_mmv_gz_supported(self, tmp_path: Path) -> None:
        """Verify gzip-compressed MMV files are readable."""
        path = tmp_path / "mmv.csv.gz"
        df = pd.DataFrame({"CANNOMBRE": ["CAND A"], "PARNOMBRE": ["A"], "VOTOS": [1]})
        df.to_csv(path, sep=";", index=False, encoding="latin-1", compression="gzip")
        result = _read_mmv(path)
        assert "PARNOMBRE" in result.columns
        assert "VOTOS" in result.columns

    def test_resolve_mmv_path_prefers_gz_when_csv_missing(self, tmp_path: Path) -> None:
        """Verify MMV path resolution falls back to .csv.gz when .csv is absent."""
        stem = "MMV_TEST"
        gz_path = tmp_path / f"{stem}.csv.gz"
        df = pd.DataFrame({"PARNOMBRE": ["A"], "VOTOS": [1]})
        df.to_csv(gz_path, sep=";", index=False, encoding="latin-1", compression="gzip")
        resolved = _resolve_mmv_path(tmp_path, stem)
        assert resolved == gz_path

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

    def test_read_participation_skips_bad_lines(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify malformed rows are skipped and logged explicitly."""
        path = tmp_path / "participation.csv"
        content = "Código Puesto,Total censo\n1,1000\n999,bad,school\n2,500\n"
        path.write_text(content, encoding="utf-8-sig")
        with caplog.at_level(logging.WARNING, logger="co_president.data_results"):
            result = _read_participation(path)
        assert len(result) == 2
        assert any("malformed" in msg for msg in caplog.messages)
        assert any("skipped" in msg for msg in caplog.messages)

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

    def test_build_round_result_zero_valid_votes_raises(self) -> None:
        """Verify ValueError when total valid votes is zero after aggregation."""
        aggregated = pd.DataFrame({"votes": [100]}, index=["blanco"])
        with pytest.raises(ValueError, match="zero total valid votes"):
            _build_round_result(
                aggregated,
                1,
                registered_voters=1000,
                polling_stations=10,
            )

    def test_unmapped_coalition_accumulates_rest_with_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify unmapped coalitions accumulate into rest and log a warning."""
        mapped_name = next(iter(COALITION_TO_CANDIDATE))
        mapped_key = COALITION_TO_CANDIDATE[mapped_name]
        df = pd.DataFrame({"name": [mapped_name, "UNKNOWN COALITION"], "votes": [100, 25]})
        with caplog.at_level(logging.WARNING, logger="co_president.data_results"):
            result = _aggregate_and_map(df, "name", "votes")
        assert int(result.loc[mapped_key, "votes"]) == 100
        assert int(result.loc["rest", "votes"]) == 25
        assert any("Unmapped coalition name" in msg for msg in caplog.messages)


# ═══════════════════════════════════════════════════════════════════
# Loader integration tests (read actual data files)
# ═══════════════════════════════════════════════════════════════════


# NOTE: TestRegistraduriaLoaders, TestMOELoaders, TestParticipationLoaders,
# TestLoadCanonicalResults have been moved to tests/integration/test_data_results_integration.py


# ═══════════════════════════════════════════════════════════════════
# MOE PDF cross-validation tests (SPEC-23)
# ═══════════════════════════════════════════════════════════════════


_SYNTHETIC_PDF_TABLE: list[list[str]] = [
    ["PARTIDO - MOVIMIENTO", "VOTOS", "%"],
    ["COALICIÓN PACTO HISTÓRICO", "8.000.000", "40,00%"],
    ["LIGA DE GOBERNANTES ANTICORRUPCIÓN", "6.000.000", "30,00%"],
    ["COALICIÓN EQUIPO POR COLOMBIA", "4.000.000", "20,00%"],
    ["COALICIÓN CENTRO ESPERANZA", "1.000.000", "5,00%"],
    ["COLOMBIA PIENSA EN GRANDE", "500.000", "2,50%"],
    ["VOTOS EN BLANCO", "500.000", "2,50%"],
    ["VOTOS NULOS", "200.000", "1,00%"],
    ["VOTOS NO MARCADOS", "50.000", "0,25%"],
]

# Flat list of extracted tables as returned by _extract_tables_with_fallback
# when page_range restricts extraction to the results pages.
# The extracted tables are a simple flat sequence — no per-page structure.
_SYNTHETIC_PDF_TABLES: list[list[list[str]]] = [_SYNTHETIC_PDF_TABLE]


def _make_pdf_test_round1_results() -> tuple[RoundResult, RoundResult]:
    """Create Registraduría and MOE CSV RoundResults matching synthetic PDF data."""
    candidates = (
        CandidateResult("gustavo_petro", 8_000_000, 0.40),
        CandidateResult("rodolfo_hernandez", 6_000_000, 0.30),
        CandidateResult("federico_gutierrez", 4_000_000, 0.20),
        CandidateResult("sergio_fajardo", 1_000_000, 0.05),
        CandidateResult("rest", 500_000, 0.025),
    )
    result = RoundResult(
        round_number=1,
        date=ELECTION_DATE_ROUND1,
        total_valid_votes=19_500_000,
        total_votes_incl_blank=20_000_000,
        registered_voters=38_971_664,
        polling_stations=12_505,
        candidates=candidates,
        blank_votes=500_000,
        null_votes=200_000,
        unmarked_votes=50_000,
    )
    return result, result


class TestMOEPDFCrossValidation:
    """Tests for MOE PDF cross-validation (SPEC-23)."""

    def test_constants_defined(self) -> None:
        """Verify PDF_RESULTS_2022 constant is defined."""
        assert isinstance(PDF_RESULTS_2022, str)
        assert PDF_RESULTS_2022.endswith(".pdf")

    def test_extract_tables_fallback_uses_pymupdf(self, tmp_path: Path) -> None:
        """Verify fallback to pymupdf when pdfplumber returns no data."""
        pdf_file = tmp_path / "test_empty.pdf"
        pdf_file.write_text("dummy")
        with (
            patch("co_president.data_results._try_extract_pdfplumber", return_value=None),
            patch("co_president.data_results._try_extract_pymupdf") as mock_fallback,
        ):
            mock_fallback.return_value = [[["coalicion", "100"]]]
            result = _extract_tables_with_fallback(pdf_file)
        assert len(result) >= 1
        assert len(result[0]) >= 1

    def test_extract_tables_fallback_both_fail_return_empty(self, tmp_path: Path) -> None:
        """Verify empty list when both engines fail."""
        pdf_file = tmp_path / "test_corrupt.pdf"
        pdf_file.write_text("not a pdf")
        with (
            patch("co_president.data_results._try_extract_pdfplumber", return_value=None),
            patch("co_president.data_results._try_extract_pymupdf", return_value=None),
        ):
            result = _extract_tables_with_fallback(pdf_file)
        assert result == []

    def test_cross_validate_moe_pdf_missing_file_raises(self) -> None:
        """Verify FileNotFoundError when PDF does not exist."""
        reg, moe = _make_pdf_test_round1_results()
        with pytest.raises(FileNotFoundError, match="MOE PDF not found"):
            cross_validate_against_moe_pdf(reg, moe, Path("/nonexistent/moe.pdf"))

    def test_cross_validate_moe_pdf_petro_share_within_tolerance(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify no Petro warning when PDF data matches Registraduría closely."""
        pdf_file = tmp_path / "moe.pdf"
        pdf_file.write_text("dummy")
        reg, moe = _make_pdf_test_round1_results()
        with patch(
            "co_president.data_results._extract_tables_with_fallback",
            return_value=_SYNTHETIC_PDF_TABLES,
        ):
            warnings = cross_validate_against_moe_pdf(reg, moe, pdf_file)
        petro_warnings = [w for w in warnings if "gustavo_petro" in w]
        assert not petro_warnings, f"Unexpected Petro warnings: {petro_warnings}"

    def test_cross_validate_moe_pdf_hernandez_share_within_tolerance(
        self,
        tmp_path: Path,
    ) -> None:
        """Verify no Hernandez warning when PDF data matches Registraduría closely."""
        pdf_file = tmp_path / "moe.pdf"
        pdf_file.write_text("dummy")
        reg, moe = _make_pdf_test_round1_results()
        with patch(
            "co_president.data_results._extract_tables_with_fallback",
            return_value=_SYNTHETIC_PDF_TABLES,
        ):
            warnings = cross_validate_against_moe_pdf(reg, moe, pdf_file)
        hernandez_warnings = [w for w in warnings if "rodolfo_hernandez" in w]
        assert not hernandez_warnings, f"Unexpected Hernandez warnings: {hernandez_warnings}"

    def test_cross_validate_moe_pdf_tolerance_boundary(self, tmp_path: Path) -> None:
        """Verify 0.50pp tolerance boundary: no warning at +0.49pp, warning at +0.51pp."""
        pdf_file = tmp_path / "moe.pdf"
        pdf_file.write_text("dummy")
        reg, moe = _make_pdf_test_round1_results()

        # To shift Petro's share by +0.49pp (from 0.4000 to 0.4049):
        # X / (12_000_000 + X) = 0.4049 => X = 8_164_678
        modified_49 = copy.deepcopy(_SYNTHETIC_PDF_TABLES)
        modified_49[0][1][1] = "8.164.678"

        with patch(
            "co_president.data_results._extract_tables_with_fallback",
            return_value=modified_49,
        ):
            warnings_49 = cross_validate_against_moe_pdf(reg, moe, pdf_file)
        petro_warnings_49 = [w for w in warnings_49 if "gustavo_petro" in w]
        assert not petro_warnings_49, f"Unexpected Petro warning at +0.49pp: {petro_warnings_49}"

        # To shift Petro's share by +0.51pp (from 0.4000 to 0.4051):
        # X / (12_000_000 + X) = 0.4051 => X = 8_171_457
        modified_51 = copy.deepcopy(_SYNTHETIC_PDF_TABLES)
        modified_51[0][1][1] = "8.171.457"

        with patch(
            "co_president.data_results._extract_tables_with_fallback",
            return_value=modified_51,
        ):
            warnings_51 = cross_validate_against_moe_pdf(reg, moe, pdf_file)
        petro_warnings_51 = [w for w in warnings_51 if "gustavo_petro" in w]
        assert petro_warnings_51, "Expected Petro warning at +0.51pp"
