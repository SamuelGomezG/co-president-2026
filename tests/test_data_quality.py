"""Tests for co_president.data_quality."""

from __future__ import annotations

from datetime import date
import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from co_president.config import POLLSTER_RATINGS
from co_president.data_polls import load_and_clean_all
from co_president.data_quality import (
    quantify_methodology_effect,
    validate_pollster_ratings,
    validate_time_decay,
)
from co_president.data_results import CandidateResult, RoundResult, load_canonical_results

if TYPE_CHECKING:
    from pathlib import Path

_MAE_CANDIDATES = (
    "gustavo_petro",
    "federico_gutierrez",
    "rodolfo_hernandez",
    "sergio_fajardo",
    "ingrid_betancourt",
)


@pytest.fixture
def sample_r1_polls() -> pd.DataFrame:
    """Minimal R1 poll DataFrame for unit testing."""
    return pd.DataFrame(
        {
            "encuestadora": ["Invamer", "YanHaas"],
            "fecha": [pd.Timestamp("2022-05-20"), pd.Timestamp("2022-05-10")],
            "gustavo_petro": [41.0, 44.0],
            "rodolfo_hernandez": [27.0, 20.0],
            "federico_gutierrez": [24.0, 21.0],
            "sergio_fajardo": [4.5, 7.0],
            "ingrid_betancourt": [0.3, 1.5],
            "blanco": [2.0, 4.0],
            "otros": [1.2, 2.5],
            "ns_nr": [0, 0],
        },
    )


@pytest.fixture
def sample_r1_polls_multi_pollster() -> pd.DataFrame:
    """R1 poll DataFrame with multiple pollsters for presence checking."""
    return pd.DataFrame(
        {
            "encuestadora": [
                "TYSE",
                "CNC",
                "Mosqueteros",
                "CELAG",
                "AtlasIntel",
                "Guarumo",
                "Invamer",
                "YanHaas",
                "MassiveCaller",
            ],
            "fecha": [
                pd.Timestamp("2022-05-17"),
                pd.Timestamp("2022-05-15"),
                pd.Timestamp("2022-05-19"),
                pd.Timestamp("2022-05-20"),
                pd.Timestamp("2022-05-20"),
                pd.Timestamp("2022-05-20"),
                pd.Timestamp("2022-05-20"),
                pd.Timestamp("2022-05-10"),
                pd.Timestamp("2022-05-18"),
            ],
            "gustavo_petro": [43.3, 41.3, 44.7, 47.3, 41.2, 38.8, 43.6, 40.0, 36.0],
            "rodolfo_hernandez": [26.9, 25.9, 29.2, 27.4, 28.0, 30.5, 13.9, 12.0, 35.2],
            "federico_gutierrez": [20.0, 22.0, 16.3, 16.9, 14.4, 16.7, 26.7, 21.0, 13.8],
            "sergio_fajardo": [5.9, 5.6, 5.6, 4.3, 6.0, 5.5, 6.5, 7.0, 4.5],
            "ingrid_betancourt": [0.5, 0.4, 0.3, 0.7, 7.5, 0.6, 0.5, 1.0, 0.3],
            "blanco": [2.0, 3.4, 2.8, 2.1, 1.8, 6.0, 5.7, 13.0, 8.5],
            "otros": [1.4, 1.4, 1.1, 1.3, 1.1, 1.9, 3.1, 0.6, 1.7],
            "ns_nr": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0],
        },
    )


@pytest.fixture
def sample_r1_polls_time_decay() -> pd.DataFrame:
    """R1 poll DataFrame with varying dates for time decay validation.

    Earlier polls (more days before election) have progressively larger MAE
    to simulate the expected relationship between recency and accuracy.
    """
    return pd.DataFrame(
        {
            "encuestadora": [
                "Invamer",
                "TYSE",
                "CNC",
                "Guarumo",
                "CELAG",
                "Mosqueteros",
            ],
            "fecha": [
                pd.Timestamp("2022-05-20"),
                pd.Timestamp("2022-05-17"),
                pd.Timestamp("2022-04-22"),
                pd.Timestamp("2022-04-01"),
                pd.Timestamp("2022-03-25"),
                pd.Timestamp("2022-03-20"),
            ],
            # Closer to actuals for recent polls, more deviation for older
            "gustavo_petro": [41.0, 43.3, 43.4, 37.0, 45.6, 35.0],
            "rodolfo_hernandez": [27.0, 26.9, 25.9, 23.0, 18.0, 22.0],
            "federico_gutierrez": [24.0, 20.0, 22.0, 27.5, 25.0, 29.0],
            "sergio_fajardo": [4.5, 5.9, 5.6, 6.0, 7.0, 6.5],
            "ingrid_betancourt": [0.3, 0.5, 0.4, 0.8, 0.6, 0.7],
            "blanco": [2.0, 2.0, 3.4, 4.5, 2.8, 5.0],
            "otros": [1.2, 1.4, 1.4, 1.2, 1.0, 1.8],
            "ns_nr": [0, 0, 0, 0, 0, 0],
        },
    )


@pytest.fixture
def sample_r1_polls_methodology() -> pd.DataFrame:
    """R1 poll DataFrame with survey methodology column."""
    return pd.DataFrame(
        {
            "encuestadora": [
                "Invamer",
                "YanHaas",
                "CNC",
                "Guarumo",
                "AtlasIntel",
                "MassiveCaller",
            ],
            "fecha": [
                pd.Timestamp("2022-05-20"),
                pd.Timestamp("2022-05-10"),
                pd.Timestamp("2022-05-15"),
                pd.Timestamp("2022-05-19"),
                pd.Timestamp("2022-05-20"),
                pd.Timestamp("2022-05-18"),
            ],
            "tipo": [
                "presencial",
                "presencial",
                "telefonico y presencial",
                "telefonica",
                "digital",
                "telefonica",
            ],
            "gustavo_petro": [41.0, 40.0, 41.3, 38.8, 41.2, 36.0],
            "rodolfo_hernandez": [27.0, 28.0, 25.9, 30.5, 28.0, 35.2],
            "federico_gutierrez": [24.0, 22.0, 22.0, 20.0, 14.4, 13.8],
            "sergio_fajardo": [4.5, 5.0, 5.6, 5.5, 6.0, 4.5],
            "ingrid_betancourt": [0.3, 0.5, 0.4, 0.6, 7.5, 0.3],
            "blanco": [2.0, 3.0, 3.4, 3.5, 1.8, 8.5],
            "otros": [1.2, 1.5, 1.4, 1.1, 1.1, 1.7],
            "ns_nr": [0, 0, 0, 0, 0, 0],
        },
    )


@pytest.fixture
def r1_results() -> RoundResult:
    """Minimal R1 RoundResult for unit testing."""
    candidates = (
        CandidateResult("gustavo_petro", 8_500_000, 0.4034),
        CandidateResult("rodolfo_hernandez", 5_900_000, 0.2817),
        CandidateResult("federico_gutierrez", 4_000_000, 0.2394),
        CandidateResult("sergio_fajardo", 900_000, 0.0418),
        CandidateResult("blanco", 400_000, 0.0173),
        CandidateResult("rest", 350_000, 0.0163),
    )
    return RoundResult(
        round_number=1,
        date=date(2022, 5, 29),
        total_valid_votes=20_000_000,
        total_votes_incl_blank=22_000_000,
        registered_voters=35_000_000,
        polling_stations=100_000,
        candidates=candidates,
        blank_votes=500_000,
        null_votes=300_000,
        unmarked_votes=200_000,
    )


# ═══════════════════════════════════════════════════════════════════
# validate_pollster_ratings
# ═══════════════════════════════════════════════════════════════════


class TestValidatePollsterRatings:
    """Tests for validate_pollster_ratings()."""

    def test_returns_dataframe(
        self,
        sample_r1_polls: pd.DataFrame,
        r1_results: RoundResult,
    ) -> None:
        """Verify the function returns a DataFrame with expected columns."""
        result = validate_pollster_ratings(sample_r1_polls, r1_results)
        assert isinstance(result, pd.DataFrame)
        expected_cols = [
            "pollster",
            "la_silla_rating",
            "empirical_mae",
            "empirical_score",
            "deviation",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_invamer_high_score(
        self,
        sample_r1_polls: pd.DataFrame,
        r1_results: RoundResult,
    ) -> None:
        """Verify Invamer's empirical score > 5.0 (high-quality pollster)."""
        result = validate_pollster_ratings(sample_r1_polls, r1_results)
        invamer_row = result[result["pollster"] == "Invamer"]
        assert not invamer_row.empty
        assert invamer_row["empirical_score"].iloc[0] > 5.0

    def test_all_r1_pollsters_present(
        self,
        sample_r1_polls_multi_pollster: pd.DataFrame,
        r1_results: RoundResult,
    ) -> None:
        """Verify all R1 pollsters appear in the output."""
        result = validate_pollster_ratings(
            sample_r1_polls_multi_pollster,
            r1_results,
        )
        input_pollsters = set(
            sample_r1_polls_multi_pollster["encuestadora"].unique(),
        )
        output_pollsters = set(result["pollster"].unique())
        missing = input_pollsters - output_pollsters
        assert not missing, f"Missing pollsters: {missing}"

    def test_scores_in_bounds(
        self,
        sample_r1_polls_multi_pollster: pd.DataFrame,
        r1_results: RoundResult,
    ) -> None:
        """Verify all empirical scores are within [0, 10]."""
        result = validate_pollster_ratings(
            sample_r1_polls_multi_pollster,
            r1_results,
        )
        scores = result["empirical_score"]
        assert scores.between(0, 10).all(), (
            f"Scores outside [0, 10]: {scores[~scores.between(0, 10)].tolist()}"
        )

    def test_last_poll_per_pollster_used(
        self,
        r1_results: RoundResult,
    ) -> None:
        """Verify that only the last poll per pollster is used."""
        df = pd.DataFrame(
            {
                "encuestadora": ["Invamer", "Invamer", "YanHaas"],
                "fecha": [
                    pd.Timestamp("2022-05-01"),
                    pd.Timestamp("2022-05-20"),
                    pd.Timestamp("2022-05-10"),
                ],
                "gustavo_petro": [42.0, 43.6, 40.0],
                "rodolfo_hernandez": [10.0, 13.9, 12.0],
                "federico_gutierrez": [25.0, 26.7, 21.0],
                "sergio_fajardo": [8.0, 6.5, 7.0],
                "ingrid_betancourt": [1.5, 0.5, 1.0],
                "blanco": [8.5, 5.7, 13.0],
                "otros": [5.0, 3.1, 0.6],
                "ns_nr": [0, 0, 6.0],
            },
        )
        result = validate_pollster_ratings(df, r1_results)
        invamer = result[result["pollster"] == "Invamer"]
        assert len(invamer) == 1
        # The empirical MAE should reflect the 2022-05-20 poll (closer to actuals)
        assert invamer["empirical_mae"].iloc[0] < 5.0

    def test_gad3_excluded_no_r1_polls(
        self,
        r1_results: RoundResult,
    ) -> None:
        """Verify pollsters with no R1 polls are excluded."""
        df = pd.DataFrame(
            {
                "encuestadora": ["Invamer", "GAD3"],
                "fecha": [
                    pd.Timestamp("2022-05-20"),
                    pd.Timestamp("2022-06-10"),
                ],
                "gustavo_petro": [43.6, 45.0],
                "rodolfo_hernandez": [13.9, 48.0],
                "federico_gutierrez": [26.7, float("nan")],
                "sergio_fajardo": [6.5, float("nan")],
                "ingrid_betancourt": [0.5, float("nan")],
                "blanco": [5.7, 2.0],
                "otros": [3.1, float("nan")],
                "ns_nr": [0, 0],
            },
        )
        result = validate_pollster_ratings(df, r1_results)
        assert "GAD3" not in result["pollster"].to_numpy()

    def test_unrated_pollster_uses_median(
        self,
        r1_results: RoundResult,
    ) -> None:
        """Verify an unrated pollster gets median rating as fallback."""
        df = pd.DataFrame(
            {
                "encuestadora": ["NewPollster", "Invamer"],
                "fecha": [
                    pd.Timestamp("2022-05-15"),
                    pd.Timestamp("2022-05-20"),
                ],
                "gustavo_petro": [41.0, 43.6],
                "rodolfo_hernandez": [20.0, 13.9],
                "federico_gutierrez": [22.0, 26.7],
                "sergio_fajardo": [6.0, 6.5],
                "ingrid_betancourt": [1.0, 0.5],
                "blanco": [7.0, 5.7],
                "otros": [3.0, 3.1],
                "ns_nr": [0, 0],
            },
        )
        median_rating = float(np.median(list(POLLSTER_RATINGS.values())))
        result = validate_pollster_ratings(df, r1_results)
        new_row = result[result["pollster"] == "NewPollster"]
        assert not new_row.empty
        assert new_row["la_silla_rating"].iloc[0] == median_rating

    def test_large_deviation_logs_warning(
        self,
        sample_r1_polls: pd.DataFrame,
        r1_results: RoundResult,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verify large deviations trigger a warning log."""
        with caplog.at_level(logging.WARNING, logger="co_president.data_quality"):
            validate_pollster_ratings(sample_r1_polls, r1_results)
        assert any("deviation" in msg.lower() for msg in caplog.messages)


# ═══════════════════════════════════════════════════════════════════
# validate_time_decay
# ═══════════════════════════════════════════════════════════════════


class TestValidateTimeDecay:
    """Tests for validate_time_decay()."""

    def test_returns_dict(
        self,
        sample_r1_polls_time_decay: pd.DataFrame,
        r1_results: RoundResult,
    ) -> None:
        """Verify the function returns a dict with expected keys."""
        result = validate_time_decay(sample_r1_polls_time_decay, r1_results)
        assert isinstance(result, dict)
        expected_keys = {
            "optimal_half_life_days",
            "current_half_life_days",
            "r_squared",
            "recommendation",
        }
        assert set(result.keys()) == expected_keys, (
            f"Expected keys {expected_keys}, got {set(result.keys())}"
        )

    def test_optimal_positive(
        self,
        sample_r1_polls_time_decay: pd.DataFrame,
        r1_results: RoundResult,
    ) -> None:
        """Verify optimal half-life is positive."""
        result = validate_time_decay(sample_r1_polls_time_decay, r1_results)
        assert result["optimal_half_life_days"] > 0

    def test_r_squared_in_range(
        self,
        sample_r1_polls_time_decay: pd.DataFrame,
        r1_results: RoundResult,
    ) -> None:
        """Verify R² is in [0, 1]."""
        result = validate_time_decay(sample_r1_polls_time_decay, r1_results)
        assert 0 <= result["r_squared"] <= 1

    def test_insufficient_data_returns_inconclusive(
        self,
        r1_results: RoundResult,
    ) -> None:
        """Verify < 5 polls returns inconclusive recommendation."""
        df = pd.DataFrame(
            {
                "encuestadora": ["Invamer", "YanHaas", "CNC"],
                "fecha": [
                    pd.Timestamp("2022-05-20"),
                    pd.Timestamp("2022-05-10"),
                    pd.Timestamp("2022-05-15"),
                ],
                "gustavo_petro": [43.6, 40.0, 41.3],
                "rodolfo_hernandez": [13.9, 12.0, 25.9],
                "federico_gutierrez": [26.7, 21.0, 22.0],
                "sergio_fajardo": [6.5, 7.0, 5.6],
                "ingrid_betancourt": [0.5, 1.0, 0.4],
                "blanco": [5.7, 13.0, 3.4],
                "otros": [3.1, 0.6, 1.4],
                "ns_nr": [0, 6.0, 0],
            },
        )
        result = validate_time_decay(df, r1_results)
        assert "insufficient" in result["recommendation"].lower()

    def test_current_half_life_is_30(
        self,
        sample_r1_polls_time_decay: pd.DataFrame,
        r1_results: RoundResult,
    ) -> None:
        """Verify current_half_life_days is 30.0."""
        result = validate_time_decay(sample_r1_polls_time_decay, r1_results)
        assert result["current_half_life_days"] == 30.0

    def test_all_types_match(
        self,
        sample_r1_polls_time_decay: pd.DataFrame,
        r1_results: RoundResult,
    ) -> None:
        """Verify return values have correct types."""
        result = validate_time_decay(sample_r1_polls_time_decay, r1_results)
        assert isinstance(result["optimal_half_life_days"], float)
        assert isinstance(result["current_half_life_days"], float)
        assert isinstance(result["r_squared"], float)
        assert isinstance(result["recommendation"], str)


# ═══════════════════════════════════════════════════════════════════
# quantify_methodology_effect
# ═══════════════════════════════════════════════════════════════════


class TestQuantifyMethodologyEffect:
    """Tests for quantify_methodology_effect()."""

    def test_returns_dict(
        self,
        sample_r1_polls_methodology: pd.DataFrame,
    ) -> None:
        """Verify the function returns a dict with expected top-level keys."""
        result = quantify_methodology_effect(sample_r1_polls_methodology)
        assert isinstance(result, dict)
        expected_keys = {
            "per_candidate",
            "methodology_counts",
            "flag_threshold_pp",
            "any_flagged",
        }
        assert set(result.keys()) == expected_keys, (
            f"Expected keys {expected_keys}, got {set(result.keys())}"
        )

    def test_all_candidate_keys_present(
        self,
        sample_r1_polls_methodology: pd.DataFrame,
    ) -> None:
        """Verify per_candidate keys match the 5 named candidates."""
        result = quantify_methodology_effect(sample_r1_polls_methodology)
        expected_candidates = {
            "gustavo_petro",
            "federico_gutierrez",
            "rodolfo_hernandez",
            "sergio_fajardo",
            "ingrid_betancourt",
        }
        assert set(result["per_candidate"].keys()) == expected_candidates

    def test_diffs_in_range(
        self,
        sample_r1_polls_methodology: pd.DataFrame,
    ) -> None:
        """Verify all differences are within [-15, +15] pp."""
        result = quantify_methodology_effect(sample_r1_polls_methodology)
        for candidate_key, data in result["per_candidate"].items():
            assert -15 <= data["diff"] <= 15, (
                f"Diff for {candidate_key} ({data['diff']}) outside [-15, 15]"
            )

    def test_digital_flagged(
        self,
        sample_r1_polls_methodology: pd.DataFrame,
    ) -> None:
        """Verify digital is included in methodology_counts but flagged."""
        result = quantify_methodology_effect(sample_r1_polls_methodology)
        assert "digital" in result["methodology_counts"]
        assert result["methodology_counts"]["digital"] >= 1

    def test_per_candidate_structure(
        self,
        sample_r1_polls_methodology: pd.DataFrame,
    ) -> None:
        """Verify each per_candidate entry has the expected sub-keys."""
        result = quantify_methodology_effect(sample_r1_polls_methodology)
        for data in result["per_candidate"].values():
            assert "presencial_mean" in data
            assert "telefonica_mean" in data
            assert "diff" in data
            assert "flag" in data
            assert isinstance(data["flag"], bool)

    def test_any_flagged_boolean(
        self,
        sample_r1_polls_methodology: pd.DataFrame,
    ) -> None:
        """Verify any_flagged is a boolean."""
        result = quantify_methodology_effect(sample_r1_polls_methodology)
        assert isinstance(result["any_flagged"], bool)

    def test_missing_tipo_excluded(
        self,
    ) -> None:
        """Verify polls with missing tipo are excluded from methodology analysis."""
        df = pd.DataFrame(
            {
                "encuestadora": ["Invamer", "YanHaas"],
                "fecha": [
                    pd.Timestamp("2022-05-20"),
                    pd.Timestamp("2022-05-10"),
                ],
                "tipo": [None, "presencial"],
                "gustavo_petro": [43.6, 40.0],
                "rodolfo_hernandez": [13.9, 12.0],
                "federico_gutierrez": [26.7, 21.0],
                "sergio_fajardo": [6.5, 7.0],
                "ingrid_betancourt": [0.5, 1.0],
                "blanco": [5.7, 13.0],
                "otros": [3.1, 0.6],
                "ns_nr": [0, 6.0],
            },
        )
        result = quantify_methodology_effect(df)
        assert result["methodology_counts"].get("presencial", 0) == 1


# ═══════════════════════════════════════════════════════════════════
# Integration tests (real 2022 data)
# ═══════════════════════════════════════════════════════════════════


class TestDataQualityIntegration:
    """Integration tests for data_quality functions on real 2022 data."""

    @pytest.fixture(autouse=True, scope="class")
    def _load_data(self, request: pytest.FixtureRequest, data_dir: Path) -> None:
        """Load real 2022 polls and results once per test class."""
        request.cls.clean_polls = load_and_clean_all(data_dir)
        request.cls.r1_results, request.cls.r2_results = load_canonical_results(data_dir)

    def test_pollster_ratings_on_real_data(self) -> None:
        """Verify validate_pollster_ratings runs on real 2022 data."""
        result = validate_pollster_ratings(
            self.clean_polls.round1,  # type: ignore[attr-defined]
            self.r1_results,  # type: ignore[attr-defined]
        )
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert "Invamer" in result["pollster"].to_numpy()

    def test_pollster_ratings_all_scores_in_bounds_real(self) -> None:
        """Verify all empirical scores in [0, 10] on real data."""
        result = validate_pollster_ratings(
            self.clean_polls.round1,  # type: ignore[attr-defined]
            self.r1_results,  # type: ignore[attr-defined]
        )
        assert result["empirical_score"].between(0, 10).all()

    def test_time_decay_on_real_data(self) -> None:
        """Verify validate_time_decay runs on real 2022 data and returns positive half-life."""
        result = validate_time_decay(
            self.clean_polls.round1,  # type: ignore[attr-defined]
            self.r1_results,  # type: ignore[attr-defined]
        )
        assert result["optimal_half_life_days"] > 0
        assert 0 <= result["r_squared"] <= 1

    def test_methodology_on_real_data(self) -> None:
        """Verify quantify_methodology_effect runs on real 2022 data."""
        result = quantify_methodology_effect(
            self.clean_polls.round1,  # type: ignore[attr-defined]
        )
        assert isinstance(result, dict)
        assert len(result["per_candidate"]) >= 1
        assert result["methodology_counts"].get("presencial", 0) >= 1

    def test_methodology_digital_has_one_poll_real(self) -> None:
        """Verify digital methodology has exactly 1 poll on real data."""
        result = quantify_methodology_effect(
            self.clean_polls.round1,  # type: ignore[attr-defined]
        )
        assert result["methodology_counts"].get("digital", 0) == 1

    def test_methodology_any_flagged_is_bool_real(self) -> None:
        """Verify any_flagged is a boolean on real data."""
        result = quantify_methodology_effect(
            self.clean_polls.round1,  # type: ignore[attr-defined]
        )
        assert isinstance(result["any_flagged"], bool)
