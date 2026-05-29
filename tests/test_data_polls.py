"""SPEC-04: Poll data loading & cleaning tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date
import importlib
import logging
import math
import statistics
from typing import TYPE_CHECKING

import pandas as pd
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from co_president.config import CONSULTATION_VOTES
from co_president.data_polls import (
    CandidateShares,
    CleanPolls,
    ConsultationPoll,
    PollRow,
    _detect_forced_choice,
    _fix_yanhaas_20220611,
    _validate_normalized_rows,
    compute_consultation_prior_strength,
    deduplicate_polls,
    fix_invamer_date,
    get_computed_consultation_prior_strengths,
    infer_round_number,
    load_raw_consultas,
    load_raw_polls,
    map_consultation_name_to_key,
    normalize_undecided,
    parse_consultations,
    retain_active_candidates,
    validate_consultation_prior_means,
)

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


class TestForcedChoiceDetection:
    """Tests for forced-choice poll detection (synthetic data)."""

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
        assert bool(forced.loc[30])
        assert not bool(forced.loc[40])

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


# ── load_raw_polls validation (synthetic data tests) ──
# Full CSV integration tests moved to tests/integration/test_data_polls_integration.py


class TestLoadRawPollsValidation:
    """Validation tests for load_raw_polls using synthetic data."""

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


# ── load_raw_consultas validation (synthetic data tests) ──


class TestLoadRawConsultasValidation:
    """Validation tests for load_raw_consultas using synthetic data."""

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


# NOTE: TestLoadAndCleanAll, TestMassiveCallerR2 full-CSV tests, TestPollSchemaVerification,
# TestLoadAndCleanAllEmptyRounds, and TestComputedPriorStrengthsLazyEval have been moved
# to tests/integration/test_data_polls_integration.py


# ── _validate_normalized_rows ──


class TestValidateNormalizedRows:
    """Tests for the _validate_normalized_rows helper."""

    def test_raises_on_bad_sum(self) -> None:
        """Verify ValueError when normalized row sum deviates from tolerance."""
        df = pd.DataFrame({"a": [60.0], "b": [43.0]})
        with pytest.raises(ValueError, match="normalized share sum"):
            _validate_normalized_rows(df, ["a", "b"], {0}, tolerance_pct=0.1)


# NOTE: TestLoadAndCleanAllEmptyRounds has been moved to
# tests/integration/test_data_polls_integration.py


# ═══════════════════════════════════════════════════════════════════
# Consultation Prior Strength (relocated from test_config.py)
# ═══════════════════════════════════════════════════════════════════


class TestConsultationPriorStrength:
    """Tests for compute_consultation_prior_strength with tmp_path CSV data."""

    CSV_HEADER = "candidato,int_voto\n"
    CSV_MULTI = CSV_HEADER + (
        # Wide spread so stdev > 0.10 after /100
        "Gustavo Petro,90.0\n"
        "Gustavo Petro,50.0\n"
        "Gustavo Petro,70.0\n"
        "Federico Gutierrez,81.0\n"
        "Sergio Fajardo,84.0\n"
    )

    CSV_SINGLE = CSV_HEADER + ("Gustavo Petro,77.0\nFederico Gutierrez,81.0\nSergio Fajardo,84.0\n")

    CSV_EMPTY = CSV_HEADER + "\n"

    def _write_csv(
        self,
        tmp_path: Path,
        csv_text: str,
        filename: str = "consultas.csv",
    ) -> Path:
        """Write CSV content to tmp_path/data/2022-polls/ and return data dir."""
        data_dir = tmp_path / "data"
        polls_dir = data_dir / "2022-polls"
        polls_dir.mkdir(parents=True)
        path = polls_dir / filename
        path.write_text(csv_text, encoding="utf-8")
        return data_dir

    def _setup(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        csv_text: str,
    ) -> None:
        """Write CSV and patch resolve_data_dir to point at tmp_path."""
        data_dir = self._write_csv(tmp_path, csv_text)
        monkeypatch.setattr(
            "co_president.data_polls.resolve_data_dir",
            lambda _: data_dir,
        )

    def test_multiple_rows_returns_correct_stdev(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Verify stdev of multiple rows is computed on [0,1] values."""
        self._setup(monkeypatch, tmp_path, self.CSV_MULTI)
        result = compute_consultation_prior_strength()
        expected_stdev = statistics.stdev([0.9, 0.5, 0.7])
        assert math.isclose(result["gustavo_petro"], expected_stdev)

    def test_single_row_returns_floor(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Verify a candidate with one data row gets the 0.10 floor."""
        self._setup(monkeypatch, tmp_path, self.CSV_SINGLE)
        result = compute_consultation_prior_strength()
        assert result["gustavo_petro"] == 0.10

    def test_hernandez_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Verify Hernández (absent from data) gets mean * 1.5 fallback."""
        self._setup(monkeypatch, tmp_path, self.CSV_MULTI)
        result = compute_consultation_prior_strength()
        assert "rodolfo_hernandez" in result
        mean_base = statistics.mean(
            [v for k, v in result.items() if CONSULTATION_VOTES.get(k, 0) > 0]
        )
        assert math.isclose(result["rodolfo_hernandez"], mean_base * 1.5)

    def test_contains_all_expected_keys(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Verify candidates with non-zero CONSULTATION_VOTES have results."""
        self._setup(monkeypatch, tmp_path, self.CSV_MULTI)
        result = compute_consultation_prior_strength()
        expected = [k for k, v in CONSULTATION_VOTES.items() if v > 0] + ["rodolfo_hernandez"]
        for key in expected:
            assert key in result, f"{key} missing from results"

    def test_values_are_positive(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Verify all values are positive."""
        self._setup(monkeypatch, tmp_path, self.CSV_MULTI)
        for val in compute_consultation_prior_strength().values():
            assert val > 0

    def test_floor_is_at_least_0_10(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Verify values >= 0.10."""
        self._setup(monkeypatch, tmp_path, self.CSV_MULTI)
        for val in compute_consultation_prior_strength().values():
            assert val >= 0.10

    def test_not_empty(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Verify the function returns non-empty dict."""
        self._setup(monkeypatch, tmp_path, self.CSV_MULTI)
        assert len(compute_consultation_prior_strength()) > 0

    def test_deterministic(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Verify two calls with same data return identical results."""
        self._setup(monkeypatch, tmp_path, self.CSV_MULTI)
        a = compute_consultation_prior_strength()
        b = compute_consultation_prior_strength()
        assert a == b

    def test_missing_key_raises_valueerror(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Verify a missing CSV name for a mapped key raises ValueError."""
        csv_missing_fajardo = self.CSV_HEADER + "Gustavo Petro,77.0\n" + "Federico Gutierrez,81.0\n"
        self._setup(monkeypatch, tmp_path, csv_missing_fajardo)
        with pytest.raises(ValueError, match="No consultation data found"):
            compute_consultation_prior_strength()


class TestConsultationPriorMeans:
    """Tests for the validate_consultation_prior_means function."""

    CSV_DATA = (
        "candidato,int_voto\nGustavo Petro,77.0\nFederico Gutierrez,81.0\nSergio Fajardo,84.0\n"
    )

    def _write_csv(self, tmp_path: Path) -> Path:
        path = tmp_path / "consultas.csv"
        path.write_text(self.CSV_DATA, encoding="utf-8")
        return path

    def test_positive_means_passes(self, tmp_path: Path) -> None:
        """Verify a mean within polling range passes."""
        csv_path = self._write_csv(tmp_path)
        validate_consultation_prior_means(
            {"gustavo_petro": 0.77},
            consultas_path=str(csv_path),
        )

    def test_negative_mean_raises_valueerror(self, tmp_path: Path) -> None:
        """Verify negative mean raises ValueError when outside polling range."""
        csv_path = tmp_path / "consultas.csv"
        csv_path.write_text("candidato,int_voto\nGustavo Petro,77.0\n", encoding="utf-8")
        with pytest.raises(ValueError, match=r"Prior mean for 'gustavo_petro' \(-0.5"):
            validate_consultation_prior_means(
                {"gustavo_petro": -0.5},
                consultas_path=str(csv_path),
            )

    def test_empty_dict_passes(self, tmp_path: Path) -> None:
        """Verify an empty dict passes validation trivially."""
        csv_path = self._write_csv(tmp_path)
        validate_consultation_prior_means({}, consultas_path=str(csv_path))

    def test_zero_mean(self, tmp_path: Path) -> None:
        """Verify a mean of 0.0 raises (outside polling range)."""
        csv_path = self._write_csv(tmp_path)
        with pytest.raises(
            ValueError,
            match=r"Prior mean for 'gustavo_petro' \(0.0000\) is outside polling range",
        ):
            validate_consultation_prior_means(
                {"gustavo_petro": 0.0},
                consultas_path=str(csv_path),
            )

    def test_multiple_keys(self, tmp_path: Path) -> None:
        """Verify multiple valid keys all pass."""
        csv_path = self._write_csv(tmp_path)
        validate_consultation_prior_means(
            {
                "gustavo_petro": 0.77,
                "federico_gutierrez": 0.81,
                "sergio_fajardo": 0.84,
            },
            consultas_path=str(csv_path),
        )

    def test_out_of_range_raises(self, tmp_path: Path) -> None:
        """Verify a mean above the polling max raises ValueError."""
        csv_path = self._write_csv(tmp_path)
        with pytest.raises(ValueError, match=r"Prior mean for 'gustavo_petro' \(0.99"):
            validate_consultation_prior_means(
                {"gustavo_petro": 0.99},
                consultas_path=str(csv_path),
            )

    def test_no_data_for_key_raises(self, tmp_path: Path) -> None:
        """Verify a key with no polling data raises ValueError."""
        csv_path = self._write_csv(tmp_path)
        with pytest.raises(ValueError, match="No consultation data found for candidate"):
            validate_consultation_prior_means(
                {"rodolfo_hernandez": 0.50},
                consultas_path=str(csv_path),
            )


# ═══════════════════════════════════════════════════════════════════
# Lazy Evaluation Tests (SPEC-02: Issue #133)
# ═══════════════════════════════════════════════════════════════════


def test_computed_prior_strengths_not_called_during_import() -> None:
    """Verify the function is not called during module import."""
    get_computed_consultation_prior_strengths.cache_clear()

    mod = importlib.import_module("co_president.config")
    try:
        importlib.reload(mod)
        info = get_computed_consultation_prior_strengths.cache_info()
        assert info.hits == 0
        assert info.misses == 0
    finally:
        importlib.reload(mod)


# NOTE: TestComputedPriorStrengthsLazyEval has been moved to
# tests/integration/test_data_polls_integration.py
