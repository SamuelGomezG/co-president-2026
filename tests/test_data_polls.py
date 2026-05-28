"""SPEC-04: Poll data loading & cleaning tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date
import logging
from typing import TYPE_CHECKING

import pandas as pd
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from co_president.config import get_active_candidates
from co_president.data_polls import (
    _SHARE_COLS_EXCLUDED,
    CandidateShares,
    CleanPolls,
    ConsultationPoll,
    PollRow,
    _detect_forced_choice,
    _fix_yanhaas_20220611,
    _validate_normalized_rows,
    deduplicate_polls,
    fix_invamer_date,
    infer_round_number,
    load_and_clean_all,
    load_raw_consultas,
    load_raw_polls,
    map_consultation_name_to_key,
    normalize_undecided,
    parse_consultations,
    retain_active_candidates,
    validate_cross_pollster_consistency,
)
from co_president.paths import resolve_data_dir


@pytest.fixture(scope="module")
def clean_polls_fixture(data_dir: Path) -> CleanPolls:
    """Load and clean polls once per module."""
    return load_and_clean_all(data_dir)


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

    def test_normalize_undecided_june5_yanhaas(self) -> None:
        """Verify June 5 YanHaas (101%) handled by existing normalization."""
        # YanHaas June 5: 101% total shares
        df = pd.DataFrame(
            {
                "gustavo_petro": [50.5],
                "rodolfo_hernandez": [50.5],
                "blanco": [0.0],
                "otros": [0.0],
                "ns_nr": [0.0],
                "muestra": [2000],
            }
        )
        result = normalize_undecided(df)
        assert result.loc[0, "gustavo_petro"] == pytest.approx(50.0)
        assert result.loc[0, "rodolfo_hernandez"] == pytest.approx(50.0)


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

    def test_empty_dataframe_returns_empty_copy(self) -> None:
        """Verify empty DataFrame returns an empty copy."""
        df = pd.DataFrame(columns=["encuestadora", "fecha", "muestra"])
        result = deduplicate_polls(df)
        assert result.empty
        assert result is not df

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

    def test_missing_muestra_int_voto_uses_muestra(self) -> None:
        """Verify missing muestra_int_voto column is filled from muestra."""
        df = pd.DataFrame(
            {
                "encuestadora": ["CNC", "CNC"],
                "fecha": pd.to_datetime(["2022-02-05", "2022-02-05"]),
                "muestra": [1000, 2206],
                "n": [1, 2],
            }
        )
        result = deduplicate_polls(df)
        assert len(result) == 1
        assert result.iloc[0]["n"] == 2
        assert "muestra_int_voto" in result.columns


class TestCrossPollsterConsistency:
    """Tests for validate_cross_pollster_consistency diagnostics."""

    def test_same_day_agreement_no_flags(self) -> None:
        """Verify identical polls on same day produce no flags."""
        df = pd.DataFrame(
            {
                "encuestadora": ["PollsterA", "PollsterB"],
                "fecha": pd.to_datetime(["2022-05-15", "2022-05-15"]),
                "gustavo_petro": [40.0, 40.0],
                "rodolfo_hernandez": [28.0, 28.0],
                "margen_error": [2.0, 2.0],
            }
        )
        result = validate_cross_pollster_consistency(df)
        assert result.empty

    def test_three_pollster_disagreement_flagged(self) -> None:
        """Verify outlier pollsters exceeding 2x MoE are flagged."""
        df = pd.DataFrame(
            {
                "encuestadora": ["A", "B", "C"],
                "fecha": pd.to_datetime(["2022-05-19", "2022-05-19", "2022-05-19"]),
                "gustavo_petro": [40.0, 50.0, 41.0],
                "rodolfo_hernandez": [28.0, 18.0, 29.0],
                "margen_error": [1.5, 1.5, 1.5],
            }
        )
        result = validate_cross_pollster_consistency(df)
        assert not result.empty
        assert "gustavo_petro" in result["candidate"].to_numpy()

    def test_single_pollster_date_empty(self) -> None:
        """Verify single-pollster dates produce empty diagnostics."""
        df = pd.DataFrame(
            {
                "encuestadora": ["Solo"],
                "fecha": pd.to_datetime(["2022-05-01"]),
                "gustavo_petro": [40.0],
                "rodolfo_hernandez": [28.0],
                "margen_error": [2.0],
            }
        )
        result = validate_cross_pollster_consistency(df)
        assert result.empty

    def test_within_moe_not_flagged(self) -> None:
        """Verify small differences within MoE are not flagged."""
        df = pd.DataFrame(
            {
                "encuestadora": ["A", "B"],
                "fecha": pd.to_datetime(["2022-05-15", "2022-05-15"]),
                "gustavo_petro": [40.0, 41.0],
                "rodolfo_hernandez": [28.0, 27.0],
                "margen_error": [2.0, 2.0],
            }
        )
        result = validate_cross_pollster_consistency(df)
        assert result.empty


class TestMassiveCallerR2:
    """Tests for excluding MassiveCaller forced-choice R2 polls."""

    def test_massivecaller_r2_excluded(self, clean_polls_fixture: CleanPolls) -> None:
        """Verify forced-choice MassiveCaller rows excluded from round2."""
        clean = clean_polls_fixture
        # 3 forced-choice MC rows exist in raw CSV (original indices 33, 38, 44)
        # where blanco+ns_nr are absent and petro+hernandez sum to ~100%.
        forced_choice_in_all = clean.all_polls[
            (clean.all_polls["encuestadora"].str.strip() == "MassiveCaller")
            & clean.all_polls["forced_choice"]
        ]
        assert len(forced_choice_in_all) == 3, (
            f"Expected 3 forced-choice MassiveCaller rows in all_polls, "
            f"found {len(forced_choice_in_all)}"
        )
        mc_in_round2 = clean.round2[clean.round2["encuestadora"].str.strip() == "MassiveCaller"]
        assert mc_in_round2.empty, (
            f"Expected 0 MassiveCaller rows in round2, found {len(mc_in_round2)}"
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
        result = _fix_yanhaas_20220611(df)
        assert result.loc[0, "ns_nr"] == 0.0
        assert result.loc[0, "gustavo_petro"] == pytest.approx(44.44, abs=0.01)
        total = result.loc[
            0, ["gustavo_petro", "rodolfo_hernandez", "blanco", "otros", "ns_nr"]
        ].sum()
        assert total == pytest.approx(100.0, abs=0.1)


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
        all_polls = pd.DataFrame(
            {"encuestadora": ["CNC"], "gustavo_petro": [40.0], "alejandro_gaviria": [5.0]}
        )
        return round1, round2, consultations, all_polls

    def test_instantiation(self, valid_data: tuple) -> None:
        """Verify CleanPolls creates with valid data."""
        r1, r2, cons, ap = valid_data
        cp = CleanPolls(round1=r1, round2=r2, consultation=cons, all_polls=ap)
        assert cp.round1 is not None
        assert cp.round2 is not None
        assert len(cp.consultation) == 1
        assert cp.all_polls is not None
        assert "alejandro_gaviria" in cp.all_polls.columns

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

    def test_invalid_fecha_raises_valueerror(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Verify invalid fecha values raise ValueError with row indices."""
        df = pd.DataFrame(
            {
                "fecha": ["not-a-date"],
                "encuestadora": ["CNC"],
                "muestra": [2206],
                "federico_gutierrez": [25.0],
                "gustavo_petro": [40.0],
                "rodolfo_hernandez": [30.0],
                "ns_nr": [0.0],
            }
        )
        data_dir = tmp_path / "data"
        polls_dir = data_dir / "2022-polls"
        polls_dir.mkdir(parents=True)
        df.to_csv(polls_dir / "encuestas_2022.csv", index=False)
        monkeypatch.setattr("co_president.data_polls.resolve_data_dir", lambda _: data_dir)

        with pytest.raises(ValueError, match="Failed to parse fecha"):
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

    def test_returns_clean_polls(self, clean_polls_fixture: CleanPolls) -> None:
        """Verify load_and_clean_all returns a CleanPolls instance."""
        assert isinstance(clean_polls_fixture, CleanPolls)

    def test_round1_has_at_least_5_pollsters(self, clean_polls_fixture: CleanPolls) -> None:
        """Verify round 1 has >= 5 unique pollsters."""
        unique_pollsters = clean_polls_fixture.round1["encuestadora"].nunique()
        assert unique_pollsters >= 5, f"Round 1 has {unique_pollsters} pollsters, need >= 5"

    def test_round2_has_at_least_2_pollsters(self, clean_polls_fixture: CleanPolls) -> None:
        """Verify round 2 has >= 2 unique pollsters."""
        unique_pollsters = clean_polls_fixture.round2["encuestadora"].nunique()
        assert unique_pollsters >= 2, f"Round 2 has {unique_pollsters} pollsters, need >= 2"

    def test_invamer_date_corrected(self, clean_polls_fixture: CleanPolls) -> None:
        """Verify Invamer April 19 poll is corrected to May 19."""
        invamer_round1 = clean_polls_fixture.round1[
            clean_polls_fixture.round1["encuestadora"].str.strip() == "Invamer"
        ]
        assert not invamer_round1.empty, "No Invamer polls found in round 1"
        min_date = invamer_round1["fecha"].min()
        assert min_date >= pd.Timestamp("2022-04-29")

    def test_no_duplicate_pollster_date_in_round1(self, clean_polls_fixture: CleanPolls) -> None:
        """Verify no pollster appears more than once per date in round 1."""
        dups = clean_polls_fixture.round1.duplicated(subset=["encuestadora", "fecha"], keep=False)
        assert not dups.any(), "Duplicate pollster+date found in round 1"

    def test_no_duplicate_pollster_date_in_round2(self, clean_polls_fixture: CleanPolls) -> None:
        """Verify no pollster appears more than once per date in round 2."""
        dups = clean_polls_fixture.round2.duplicated(subset=["encuestadora", "fecha"], keep=False)
        assert not dups.any(), "Duplicate pollster+date found in round 2"

    def test_normalized_shares_sum_to_100(self, clean_polls_fixture: CleanPolls) -> None:
        """Verify candidate+blanco+otros sum to ~100% after normalization."""
        share_cols = [
            c
            for c in clean_polls_fixture.round1.columns
            if c in {cand.key for cand in get_active_candidates(1)} or c in ("blanco", "otros")
        ]
        for idx, row in clean_polls_fixture.round1.iterrows():
            row_sum = row[share_cols].sum()
            assert abs(row_sum - 100.0) <= 1.0, f"Row {idx} share sum = {row_sum}, expected ~100"

    def test_round2_shares_sum_to_100(self, clean_polls_fixture: CleanPolls) -> None:
        """Verify round 2 candidate+blanco+otros sum to ~100% after normalization."""
        share_cols = [
            c
            for c in clean_polls_fixture.round2.columns
            if c in {cand.key for cand in get_active_candidates(2)} or c in ("blanco", "otros")
        ]
        for idx, row in clean_polls_fixture.round2.iterrows():
            row_sum = row[share_cols].sum()
            assert abs(row_sum - 100.0) <= 1.0, f"Row {idx} share sum = {row_sum}, expected ~100"

    def test_gad3_round2_count(self, clean_polls_fixture: CleanPolls) -> None:
        """Verify the number of GAD3 runoff tracking polls in round 2.

        10 waves are present (May 31-Jun 10). Wave 11 (Jun 11) was a
        duplicate of wave 10 (identical sample and shares) and was removed.
        """
        gad3_round2 = clean_polls_fixture.round2[
            clean_polls_fixture.round2["encuestadora"].str.strip() == "GAD3"
        ]
        assert len(gad3_round2) == 10

    def test_gad3_final_wave_not_duplicated(self, clean_polls_fixture: CleanPolls) -> None:
        """Verify the final GAD3 tracking wave appears only once in round 2.

        The final wave (2022-06-10) was corrected from an erroneous 2022-06-11
        date to avoid duplication.
        """
        gad3_round2 = clean_polls_fixture.round2[
            clean_polls_fixture.round2["encuestadora"].str.strip() == "GAD3"
        ]
        final_wave = gad3_round2[gad3_round2["fecha"] == pd.Timestamp("2022-06-10")]
        assert len(final_wave) == 1

    def test_ns_nr_is_zero_after_normalization(self, clean_polls_fixture: CleanPolls) -> None:
        """Verify ns_nr is 0.0 in both rounds after normalization."""
        assert clean_polls_fixture.round1["ns_nr"].sum() == 0.0
        assert clean_polls_fixture.round2["ns_nr"].sum() == 0.0

    def test_all_polls_includes_unclassified_and_all_columns(
        self,
        clean_polls_fixture: CleanPolls,
    ) -> None:
        """Verify all_polls includes pre-consultation columns and unclassified rows."""
        assert "gustavo_petro" in clean_polls_fixture.all_polls.columns
        assert "round_number" in clean_polls_fixture.all_polls.columns
        unclassified = clean_polls_fixture.all_polls[
            clean_polls_fixture.all_polls["round_number"].isna()
        ]
        assert len(unclassified) > 0

    def test_all_polls_dates_monotonic(self, clean_polls_fixture: CleanPolls) -> None:
        """Verify dates are non-decreasing in the cleaned output.

        fix_invamer_date corrects the known Invamer anomaly before all_polls
        is constructed, so no exception is needed here.
        """
        assert clean_polls_fixture.all_polls["fecha"].is_monotonic_increasing


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
