"""Tests for co_president._data_quality."""

from __future__ import annotations

from datetime import date
import logging

import numpy as np
import pandas as pd
import pytest

from co_president._data_quality import (
    _MAE_CANDIDATES,
    _compute_mae,
    quantify_methodology_effect,
    validate_pollster_ratings,
    validate_time_decay,
)
from co_president.config import POLLSTER_RATINGS, ModelConfig
from co_president.data import CandidateResult, RoundResult


@pytest.fixture
def sample_r1_polls() -> pd.DataFrame:
    """Minimal R1 poll DataFrame for unit testing (3 rows per AGENTS.md)."""
    return pd.DataFrame(
        {
            "encuestadora": ["Invamer", "YanHaas", "CNC"],
            "fecha": [
                pd.Timestamp("2022-05-20"),
                pd.Timestamp("2022-05-10"),
                pd.Timestamp("2022-05-15"),
            ],
            "gustavo_petro": [41.0, 44.0, 39.0],
            "rodolfo_hernandez": [27.0, 20.0, 30.0],
            "federico_gutierrez": [24.0, 21.0, 23.0],
            "sergio_fajardo": [4.5, 7.0, 4.0],
            "ingrid_betancourt": [0.3, 1.5, 0.5],
            "blanco": [2.0, 4.0, 2.5],
            "otros": [1.2, 2.5, 1.0],
            "ns_nr": [0, 0, 0],
        },
    )


@pytest.fixture
def sample_r1_polls_multi_pollster() -> pd.DataFrame:
    """R1 poll DataFrame with multiple pollsters for presence checking.

    Kept to 5 rows per project conventions (AGENTS.md §6).
    """
    return pd.DataFrame(
        {
            "encuestadora": [
                "TYSE",
                "CNC",
                "Mosqueteros",
                "Invamer",
                "YanHaas",
            ],
            "fecha": [
                pd.Timestamp("2022-05-17"),
                pd.Timestamp("2022-05-15"),
                pd.Timestamp("2022-05-19"),
                pd.Timestamp("2022-05-20"),
                pd.Timestamp("2022-05-10"),
            ],
            "gustavo_petro": [43.3, 41.3, 44.7, 43.6, 40.0],
            "rodolfo_hernandez": [26.9, 25.9, 29.2, 13.9, 12.0],
            "federico_gutierrez": [20.0, 22.0, 16.3, 26.7, 21.0],
            "sergio_fajardo": [5.9, 5.6, 5.6, 6.5, 7.0],
            "ingrid_betancourt": [0.5, 0.4, 0.3, 0.5, 1.0],
            "blanco": [2.0, 3.4, 2.8, 5.7, 13.0],
            "otros": [1.4, 1.4, 1.1, 3.1, 0.6],
            "ns_nr": [0.0, 0.0, 0.0, 0.0, 6.0],
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
    """Minimal R1 RoundResult for unit testing.

    Vote counts and shares are internally consistent:
    share = votes / total_votes_incl_blank.
    """
    blank_votes = 400_000
    candidates = (
        CandidateResult("gustavo_petro", 8_500_000, 0.422360),
        CandidateResult("rodolfo_hernandez", 5_900_000, 0.293168),
        CandidateResult("federico_gutierrez", 4_000_000, 0.198758),
        CandidateResult("sergio_fajardo", 900_000, 0.044720),
        CandidateResult("ingrid_betancourt", 75_000, 0.003727),
        CandidateResult("blanco", blank_votes, 0.019876),
        CandidateResult("rest", 350_000, 0.017391),
    )
    total_valid_votes = sum(c.votes for c in candidates if c.candidate_key != "blanco")
    total_votes_incl_blank = total_valid_votes + blank_votes
    return RoundResult(
        round_number=1,
        date=date(2022, 5, 29),
        total_valid_votes=total_valid_votes,
        total_votes_incl_blank=total_votes_incl_blank,
        registered_voters=35_000_000,
        polling_stations=100_000,
        candidates=candidates,
        blank_votes=blank_votes,
        null_votes=300_000,
        unmarked_votes=200_000,
    )


def test_compute_mae_all_nan(r1_results: RoundResult) -> None:
    """Verify MAE is NaN when all candidate shares are NaN."""
    row = pd.Series({key: float("nan") for key in _MAE_CANDIDATES}, dtype=float)
    assert np.isnan(_compute_mae(row, r1_results))


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
        # The empirical MAE should reflect the 2022-05-20 poll, not the 2022-05-01 poll
        assert invamer["empirical_mae"].iloc[0] > 0

    def test_post_election_pollsters_excluded(
        self,
        r1_results: RoundResult,
    ) -> None:
        """Verify post-election pollsters are excluded."""
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
        assert any(
            "YanHaas" in record.message and "deviation=" in record.message
            for record in caplog.records
        )


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
        """Verify current_half_life_days matches ModelConfig."""
        result = validate_time_decay(sample_r1_polls_time_decay, r1_results)
        assert result["current_half_life_days"] == ModelConfig().time_decay_half_life_days

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

    def test_beta_non_positive_returns_relationship_not_detected(
        self,
        r1_results: RoundResult,
    ) -> None:
        """Verify non-positive beta returns relationship_not_detected."""
        df = pd.DataFrame(
            {
                "encuestadora": [
                    "Invamer",
                    "YanHaas",
                    "CNC",
                    "Guarumo",
                    "TYSE",
                ],
                "fecha": [
                    pd.Timestamp("2022-05-28"),
                    pd.Timestamp("2022-05-20"),
                    pd.Timestamp("2022-05-10"),
                    pd.Timestamp("2022-04-20"),
                    pd.Timestamp("2022-04-01"),
                ],
                "gustavo_petro": [45.0, 44.0, 43.0, 41.0, 40.5],
                "rodolfo_hernandez": [22.0, 23.0, 24.0, 26.5, 27.5],
                "federico_gutierrez": [27.0, 26.0, 25.0, 23.8, 24.0],
                "sergio_fajardo": [6.0, 5.5, 5.0, 4.3, 4.2],
                "ingrid_betancourt": [1.0, 0.8, 0.6, 0.4, 0.35],
                "blanco": [2.0, 2.0, 2.0, 2.0, 2.0],
                "otros": [1.0, 1.0, 1.0, 1.0, 1.0],
                "ns_nr": [0, 0, 0, 0, 0],
            },
        )
        result = validate_time_decay(df, r1_results)
        assert "relationship_not_detected" in result["recommendation"].lower()

    def test_four_polls_is_insufficient(self, r1_results: RoundResult) -> None:
        """Verify exactly four polls returns insufficient_data."""
        df = pd.DataFrame(
            {
                "encuestadora": ["Invamer", "YanHaas", "CNC", "Guarumo"],
                "fecha": [
                    pd.Timestamp("2022-05-20"),
                    pd.Timestamp("2022-05-10"),
                    pd.Timestamp("2022-05-15"),
                    pd.Timestamp("2022-05-18"),
                ],
                "gustavo_petro": [43.6, 40.0, 41.3, 42.0],
                "rodolfo_hernandez": [13.9, 12.0, 25.9, 20.0],
                "federico_gutierrez": [26.7, 21.0, 22.0, 22.5],
                "sergio_fajardo": [6.5, 7.0, 5.6, 5.2],
                "ingrid_betancourt": [0.5, 1.0, 0.4, 0.6],
                "blanco": [5.7, 13.0, 3.4, 3.0],
                "otros": [3.1, 0.6, 1.4, 1.0],
                "ns_nr": [0, 6.0, 0, 0],
            },
        )
        result = validate_time_decay(df, r1_results)
        assert result["recommendation"] == "insufficient_data"


# ═══════════════════════════════════════════════════════════════════
# quantify_methodology_effect
# ═══════════════════════════════════════════════════════════════════


class TestQuantifyMethodologyEffect:
    """Tests for quantify_methodology_effect()."""

    def test_returns_dict(
        self,
        sample_r1_polls_methodology: pd.DataFrame,
        r1_results: RoundResult,
    ) -> None:
        """Verify the function returns a dict with expected top-level keys."""
        result = quantify_methodology_effect(sample_r1_polls_methodology, r1_results)
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
        r1_results: RoundResult,
    ) -> None:
        """Verify per_candidate keys match the 5 named candidates."""
        result = quantify_methodology_effect(sample_r1_polls_methodology, r1_results)
        assert set(result["per_candidate"].keys()) == set(_MAE_CANDIDATES)

    def test_diffs_in_range(
        self,
        sample_r1_polls_methodology: pd.DataFrame,
        r1_results: RoundResult,
    ) -> None:
        """Verify all differences are within [-15, +15] pp."""
        result = quantify_methodology_effect(sample_r1_polls_methodology, r1_results)
        for candidate_key, data in result["per_candidate"].items():
            if np.isnan(data["max_diff_pp"]):
                continue
            assert -15 <= data["max_diff_pp"] <= 15, (
                f"Diff for {candidate_key} ({data['max_diff_pp']}) outside [-15, 15]"
            )

    def test_digital_in_methodology_counts(
        self,
        sample_r1_polls_methodology: pd.DataFrame,
        r1_results: RoundResult,
    ) -> None:
        """Verify digital is included in methodology_counts."""
        result = quantify_methodology_effect(sample_r1_polls_methodology, r1_results)
        assert "digital" in result["methodology_counts"]
        assert result["methodology_counts"]["digital"] >= 1

    def test_per_candidate_structure(
        self,
        sample_r1_polls_methodology: pd.DataFrame,
        r1_results: RoundResult,
    ) -> None:
        """Verify each per_candidate entry has the expected sub-keys."""
        result = quantify_methodology_effect(sample_r1_polls_methodology, r1_results)
        for data in result["per_candidate"].values():
            assert "methodology_mae" in data
            assert "max_diff_pp" in data
            assert "flagged" in data
            assert isinstance(data["flagged"], bool)

    def test_any_flagged_boolean(
        self,
        sample_r1_polls_methodology: pd.DataFrame,
        r1_results: RoundResult,
    ) -> None:
        """Verify any_flagged is a boolean."""
        result = quantify_methodology_effect(sample_r1_polls_methodology, r1_results)
        assert isinstance(result["any_flagged"], bool)

    def test_missing_tipo_excluded(
        self,
        r1_results: RoundResult,
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
        result = quantify_methodology_effect(df, r1_results)
        assert result["methodology_counts"].get("presencial", 0) == 1

    def test_empty_dataframe_returns_nan_diffs(
        self,
        r1_results: RoundResult,
    ) -> None:
        """Verify empty inputs yield NaN diffs and no flags."""
        df = pd.DataFrame(
            columns=[
                "encuestadora",
                "fecha",
                "tipo",
                *_MAE_CANDIDATES,
                "blanco",
                "otros",
                "ns_nr",
            ],
        )
        result = quantify_methodology_effect(df, r1_results)
        assert result["methodology_counts"] == {}
        assert result["any_flagged"] is False
        sample = result["per_candidate"]["gustavo_petro"]
        assert np.isnan(sample["max_diff_pp"])
