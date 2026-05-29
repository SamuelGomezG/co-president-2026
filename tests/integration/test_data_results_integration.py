"""Integration tests for SPEC-03: Election results consolidation.

These tests load full production CSV datasets and are marked with
``@pytest.mark.integration``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from co_president.config import ELECTION_DATE_ROUND1, ELECTION_DATE_ROUND2, get_active_candidates
from co_president.data import (
    RoundResult,
    load_canonical_results,
    load_moe_round1,
    load_moe_round2,
    load_participation_round1,
    load_participation_round2,
    load_registraduria_round1,
    load_registraduria_round2,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.integration
class TestRegistraduriaLoaders:
    """Tests for Registraduría MMV data loaders."""

    def test_load_registraduria_round1_has_expected_keys(self, data_dir: Path) -> None:
        """Verify round 1 contains all major candidate keys after mapping."""
        df = load_registraduria_round1(data_dir)
        assert set(df.index) >= {"gustavo_petro", "rodolfo_hernandez", "federico_gutierrez"}
        assert "blanco" in df.index
        assert "ingrid_betancourt" not in df.index

    def test_load_registraduria_round1_total_votes(self, data_dir: Path) -> None:
        """Verify round 1 total votes are reasonable (nulos excluded downstream)."""
        df = load_registraduria_round1(data_dir)
        total = df["votes"].sum()
        assert 20_000_000 < total < 22_000_000

    def test_load_registraduria_round2_has_runoff_keys(self, data_dir: Path) -> None:
        """Verify round 2 contains only runoff-relevant candidate keys."""
        df = load_registraduria_round2(data_dir)
        assert "gustavo_petro" in df.index
        assert "rodolfo_hernandez" in df.index
        assert "blanco" in df.index
        assert "federico_gutierrez" not in df.index
        assert "sergio_fajardo" not in df.index

    def test_load_registraduria_round2_rest_includes_blanco(self, data_dir: Path) -> None:
        """Verify final round 2 result merges blanco into rest (via public API)."""
        _, round2 = load_canonical_results(data_dir)
        assert "blanco" not in {c.candidate_key for c in round2.candidates}
        rest = next(c for c in round2.candidates if c.candidate_key == "rest")
        assert rest.votes > 400_000


@pytest.mark.integration
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


@pytest.mark.integration
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


@pytest.mark.integration
class TestLoadCanonicalResults:
    """Tests for the full load_canonical_results() pipeline."""

    # ── Round structure ──

    def test_round1_candidate_keys(
        self,
        canonical_results: tuple[RoundResult, RoundResult],
    ) -> None:
        """Verify round 1 candidates match get_active_candidates(1) minus Betancourt."""
        round1, _ = canonical_results
        expected_keys = {c.key for c in get_active_candidates(1)}
        expected_keys.discard("ingrid_betancourt")
        actual_keys = {c.candidate_key for c in round1.candidates}
        assert actual_keys == expected_keys

    def test_round2_candidate_keys(
        self,
        canonical_results: tuple[RoundResult, RoundResult],
    ) -> None:
        """Verify round 2 candidates match get_active_candidates(2)."""
        _, round2 = canonical_results
        expected_keys = {c.key for c in get_active_candidates(2)}
        actual_keys = {c.candidate_key for c in round2.candidates}
        assert actual_keys == expected_keys

    def test_round_number_fields(self, canonical_results: tuple[RoundResult, RoundResult]) -> None:
        """Verify round_number and date are correctly assigned."""
        round1, round2 = canonical_results
        assert round1.round_number == 1
        assert round1.date == ELECTION_DATE_ROUND1
        assert round2.round_number == 2
        assert round2.date == ELECTION_DATE_ROUND2

    # ── Vote share acceptance criteria ──

    def test_round1_petro_share(self, canonical_results: tuple[RoundResult, RoundResult]) -> None:
        """Verify Petro round 1 vote share ≈ 40.34% (within 0.5pp)."""
        round1, _ = canonical_results
        share = round1.get_share("gustavo_petro")
        assert share == pytest.approx(0.4034, abs=0.005)

    def test_round1_hernandez_share(
        self,
        canonical_results: tuple[RoundResult, RoundResult],
    ) -> None:
        """Verify Hernández round 1 vote share ≈ 28.15% (within 0.5pp)."""
        round1, _ = canonical_results
        share = round1.get_share("rodolfo_hernandez")
        assert share == pytest.approx(0.2815, abs=0.005)

    def test_round1_gutierrez_share(
        self,
        canonical_results: tuple[RoundResult, RoundResult],
    ) -> None:
        """Verify Gutiérrez round 1 vote share ≈ 23.89% (within 0.5pp)."""
        round1, _ = canonical_results
        share = round1.get_share("federico_gutierrez")
        assert share == pytest.approx(0.2389, abs=0.005)

    def test_round1_fajardo_share(
        self,
        canonical_results: tuple[RoundResult, RoundResult],
    ) -> None:
        """Verify Fajardo round 1 vote share ≈ 4.39% (within 0.5pp)."""
        round1, _ = canonical_results
        share = round1.get_share("sergio_fajardo")
        assert share == pytest.approx(0.0439, abs=0.005)

    def test_round2_petro_share(self, canonical_results: tuple[RoundResult, RoundResult]) -> None:
        """Verify Petro round 2 vote share ≈ 50.44% (within 0.5pp)."""
        _, round2 = canonical_results
        share = round2.get_share("gustavo_petro")
        assert share == pytest.approx(0.5044, abs=0.005)

    def test_round2_hernandez_share(
        self,
        canonical_results: tuple[RoundResult, RoundResult],
    ) -> None:
        """Verify Hernández round 2 vote share ≈ 47.26% (within 0.5pp)."""
        _, round2 = canonical_results
        share = round2.get_share("rodolfo_hernandez")
        assert share == pytest.approx(0.4726, abs=0.005)

    # ── RoundResult methods ──

    def test_round1_top_two(self, canonical_results: tuple[RoundResult, RoundResult]) -> None:
        """Verify top_two for round 1 returns Petro then Hernández."""
        round1, _ = canonical_results
        first, second = round1.top_two()
        assert first.candidate_key == "gustavo_petro"
        assert second.candidate_key == "rodolfo_hernandez"

    def test_round1_candidates_above_20(
        self,
        canonical_results: tuple[RoundResult, RoundResult],
    ) -> None:
        """Verify candidates above 20% in round 1."""
        round1, _ = canonical_results
        above = round1.get_candidates_above(20.0)
        keys = [c.candidate_key for c in above]
        assert keys == ["gustavo_petro", "rodolfo_hernandez", "federico_gutierrez"]

    def test_round2_top_two(self, canonical_results: tuple[RoundResult, RoundResult]) -> None:
        """Verify top_two for round 2 returns Petro then Hernández."""
        _, round2 = canonical_results
        first, second = round2.top_two()
        assert first.candidate_key == "gustavo_petro"
        assert second.candidate_key == "rodolfo_hernandez"

    # ── Census and turnout ──

    def test_registered_voters_reasonable(
        self,
        canonical_results: tuple[RoundResult, RoundResult],
    ) -> None:
        """Verify registered voters count is within ±1% of official ~39M for both rounds."""
        round1, round2 = canonical_results
        assert 38_500_000 < round1.registered_voters < 39_500_000
        assert 38_500_000 < round2.registered_voters < 39_500_000

    def test_polling_stations_reasonable(
        self,
        canonical_results: tuple[RoundResult, RoundResult],
    ) -> None:
        """Verify polling station count is reasonable for both rounds."""
        round1, round2 = canonical_results
        assert 10_000 < round1.polling_stations < 15_000
        assert 10_000 < round2.polling_stations < 15_000

    # ── Cross-validation ──

    def test_cross_validation_round1_passes_consolidation(
        self,
        data_dir: Path,
    ) -> None:
        """Verify load_canonical_results round 1 passes internal consolidation (Reg vs MOE)."""
        round1, _ = load_canonical_results(data_dir)
        assert round1.round_number == 1
        assert round1.total_valid_votes > 0

    def test_cross_validation_round2_passes_consolidation(
        self,
        data_dir: Path,
    ) -> None:
        """Verify load_canonical_results round 2 passes internal consolidation (Reg vs MOE)."""
        _, round2 = load_canonical_results(data_dir)
        assert round2.round_number == 2
        assert round2.total_valid_votes > 0

    # ── Vote share sum consistency ──

    def test_round1_vote_shares_sum_to_approx_one(
        self,
        canonical_results: tuple[RoundResult, RoundResult],
    ) -> None:
        """Verify round 1 candidate vote shares sum to ~1.0."""
        round1, _ = canonical_results
        total = sum(c.vote_share for c in round1.candidates)
        assert total == pytest.approx(1.0, abs=0.01)

    def test_round2_vote_shares_sum_to_approx_one(
        self,
        canonical_results: tuple[RoundResult, RoundResult],
    ) -> None:
        """Verify round 2 candidate vote shares sum to ~1.0."""
        _, round2 = canonical_results
        total = sum(c.vote_share for c in round2.candidates)
        assert total == pytest.approx(1.0, abs=0.01)

    # ── Participation data correctness ──

    def test_round2_loads_correct_participation_data(
        self,
        data_dir: Path,
    ) -> None:
        """Verify round 2 RoundResult uses round 2 participation data, not round 1's."""
        _, round2 = load_canonical_results(data_dir)
        part2 = load_participation_round2(data_dir)
        expected_voters = int(part2["Total censo"].sum())
        expected_puestos = int(part2["Código Puesto"].nunique())
        assert round2.registered_voters == expected_voters
        assert round2.polling_stations == expected_puestos

    def test_canonical_results_calls_round2_loader(self, data_dir: Path) -> None:
        """Verify load_canonical_results() actually calls load_participation_round2()."""
        with patch(
            "co_president.data_results.load_participation_round2",
            wraps=load_participation_round2,
        ) as mock_loader:
            load_canonical_results(data_dir)
        mock_loader.assert_called_once()
