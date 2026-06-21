"""Tests for SPEC-05: Poll Aggregation (Baseline)."""

from __future__ import annotations

from datetime import date
import logging

import numpy as np
import pandas as pd
import pytest

from co_president.aggregation import (
    aggregation_snapshot,
    combined_weight,
    evolution_series,
    pollster_weight_map,
    sample_size_weight,
    time_weight,
    weighted_average,
)
from co_president.config import (
    ELECTION_DATE_ROUND1,
    POLLSTER_RATINGS,
    pollster_weight_formula,
)


class TestTimeWeight:
    """Tests for ``time_weight``."""

    def test_election_day_returns_one(self) -> None:
        """A poll on election day gets weight 1.0."""
        dates = pd.Series([date(2022, 5, 29)])
        result = time_weight(dates, ELECTION_DATE_ROUND1)
        assert result.iloc[0] == pytest.approx(1.0)

    def test_thirty_days_before_returns_half(self) -> None:
        """A poll 30 days before election day gets weight 0.5."""
        dates = pd.Series([date(2022, 4, 29)])
        result = time_weight(dates, ELECTION_DATE_ROUND1)
        assert result.iloc[0] == pytest.approx(0.5, rel=1e-3)

    def test_half_life_custom(self) -> None:
        """Custom half-life does not affect weight for same-day poll."""
        dates = pd.Series([date(2022, 5, 29)])
        result = time_weight(dates, ELECTION_DATE_ROUND1, half_life=15.0)
        assert result.iloc[0] == pytest.approx(1.0)

    def test_multiple_dates(self) -> None:
        """Multiple dates return correct exponential decay values."""
        dates = pd.Series(
            [
                date(2022, 5, 29),  # 0 days -> 1.0
                date(2022, 4, 29),  # 30 days -> 0.5
                date(2022, 3, 30),  # 60 days -> 0.25
            ]
        )
        result = time_weight(dates, ELECTION_DATE_ROUND1)
        assert result.iloc[0] == pytest.approx(1.0)
        assert result.iloc[1] == pytest.approx(0.5, rel=1e-3)
        assert result.iloc[2] == pytest.approx(0.25, rel=1e-3)

    def test_poll_after_election_is_zero(self) -> None:
        """A poll after the election date receives zero weight."""
        dates = pd.Series([date(2022, 6, 19)])
        result = time_weight(dates, ELECTION_DATE_ROUND1)
        assert result.iloc[0] == pytest.approx(0.0)

    def test_time_weight_future_poll_is_zero(self) -> None:
        """A poll after election day gets zero weight (not decaying positive)."""
        election_date = date(2022, 5, 29)
        dates = pd.Series([date(2022, 5, 29), date(2022, 6, 1)])
        weights = time_weight(dates, election_date)
        assert weights.iloc[0] == pytest.approx(1.0)
        assert weights.iloc[1] == pytest.approx(0.0)

    def test_time_weight_future_poll_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """A future poll triggers a warning log."""
        election_date = date(2022, 5, 29)
        dates = pd.Series([date(2022, 5, 29), date(2022, 6, 10)])
        with caplog.at_level(logging.WARNING):
            time_weight(dates, election_date)
        assert "after election date" in caplog.text

    def test_returns_series(self) -> None:
        """Returns a pandas Series of the same length as input."""
        dates = pd.Series([date(2022, 5, 29), date(2022, 4, 29)])
        result = time_weight(dates, ELECTION_DATE_ROUND1)
        assert isinstance(result, pd.Series)
        assert len(result) == 2


class TestSampleSizeWeight:
    """Tests for ``sample_size_weight``."""

    def test_returns_series(self) -> None:
        """Returns a pandas Series of the same length as input."""
        sizes = pd.Series([100, 1000, 5000])
        result = sample_size_weight(sizes)
        assert isinstance(result, pd.Series)
        assert len(result) == 3

    def test_monotonic_increasing(self) -> None:
        """Larger sample sizes produce larger weights."""
        sizes = pd.Series([100, 500, 1000, 5000])
        result = sample_size_weight(sizes)
        for i in range(len(result) - 1):
            assert result.iloc[i] < result.iloc[i + 1]

    def test_diminishing_returns(self) -> None:
        """Rate of increase slows for equal-spaced increments (convex)."""
        sizes = pd.Series([100, 200, 300, 400])
        result = sample_size_weight(sizes)
        d1 = result.iloc[1] - result.iloc[0]
        d2 = result.iloc[2] - result.iloc[1]
        d3 = result.iloc[3] - result.iloc[2]
        assert d1 > d2 > d3

    def test_value_for_n_1000(self) -> None:
        """Weight for N=1000 equals log(1001)."""
        sizes = pd.Series([1000])
        result = sample_size_weight(sizes)
        expected = np.log(1001)
        assert result.iloc[0] == pytest.approx(expected)

    def test_value_for_n_100(self) -> None:
        """Weight for N=100 equals log(101)."""
        sizes = pd.Series([100])
        result = sample_size_weight(sizes)
        expected = np.log(101)
        assert result.iloc[0] == pytest.approx(expected)

    def test_larger_n_gives_larger_weight(self) -> None:
        """N=1000 produces larger weight than N=100."""
        sizes = pd.Series([100, 1000])
        result = sample_size_weight(sizes)
        assert result.iloc[0] < result.iloc[1]

    def test_zero_sample_size(self) -> None:
        """Zero sample size produces a finite weight (log(1)=0)."""
        sizes = pd.Series([0])
        result = sample_size_weight(sizes)
        assert np.isfinite(result.iloc[0])

    def test_negative_sample_size_clipped(self) -> None:
        """Negative sample sizes are clipped to 0 before computation."""
        sizes = pd.Series([-5])
        result = sample_size_weight(sizes)
        assert np.isfinite(result.iloc[0])


class TestPollsterWeightMap:
    """Tests for ``pollster_weight_map``."""

    def test_invamer_returns_one(self) -> None:
        """Invamer (rating 10) maps to weight 1.0."""
        pollsters = pd.Series(["Invamer"])
        result = pollster_weight_map(pollsters, POLLSTER_RATINGS)
        assert result.iloc[0] == pytest.approx(1.0)

    def test_mosqueteros_returns_0_82(self) -> None:
        """Mosqueteros (rating 1.0) maps to weight 0.82."""
        pollsters = pd.Series(["Mosqueteros"])
        result = pollster_weight_map(pollsters, POLLSTER_RATINGS)
        assert result.iloc[0] == pytest.approx(0.82)

    def test_unknown_pollster_uses_median_fallback(self) -> None:
        """Unknown pollster receives the median weight (0.908)."""
        pollsters = pd.Series(["UnknownPollster"])
        result = pollster_weight_map(pollsters, POLLSTER_RATINGS)
        assert 0.8 <= result.iloc[0] <= 1.0

    def test_multiple_pollsters(self) -> None:
        """Mixed known and unknown pollsters return correct weights."""
        pollsters = pd.Series(["Invamer", "Mosqueteros", "Unknown"])
        result = pollster_weight_map(pollsters, POLLSTER_RATINGS)
        assert len(result) == 3
        assert result.iloc[0] == pytest.approx(1.0)
        assert result.iloc[1] == pytest.approx(0.82)
        assert 0.8 <= result.iloc[2] <= 1.0

    def test_weights_derived_from_config_formula(self) -> None:
        """All pollster weights match calling ``pollster_weight_formula`` directly."""
        custom_ratings = {"A": 0.0, "B": 5.0, "C": 10.0, "D": 3.5}
        pollsters = pd.Series(list(custom_ratings))
        result = pollster_weight_map(pollsters, custom_ratings)
        for idx, (_, rating) in enumerate(custom_ratings.items()):
            expected = pollster_weight_formula(rating)
            assert result.iloc[idx] == pytest.approx(expected)

    def test_empty_ratings_raises(self) -> None:
        """Empty ratings dict raises ValueError."""
        pollsters = pd.Series(["Invamer"])
        with pytest.raises(ValueError, match="ratings"):
            pollster_weight_map(pollsters, {})


class TestCombinedWeight:
    """Tests for ``combined_weight``."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Three-row DataFrame with varied dates, pollsters, and sample sizes."""
        return pd.DataFrame(
            {
                "fecha": pd.to_datetime(
                    [
                        date(2022, 5, 29),
                        date(2022, 4, 29),
                        date(2022, 3, 30),
                    ]
                ),
                "encuestadora": ["Invamer", "Mosqueteros", "Invamer"],
                "muestra_int_voto": pd.array([1000, 500, 200], dtype="Int64"),
                "gustavo_petro": [40.0, 38.0, 35.0],
                "federico_gutierrez": [25.0, 24.0, 28.0],
            }
        )

    def test_all_weights_positive(self, sample_df: pd.DataFrame) -> None:
        """All combined weights are strictly positive."""
        result = combined_weight(sample_df, ELECTION_DATE_ROUND1, POLLSTER_RATINGS)
        assert (result > 0).all()

    def test_weights_sum_to_one(self, sample_df: pd.DataFrame) -> None:
        """Combined weights are globally normalized to sum to 1.0."""
        result = combined_weight(sample_df, ELECTION_DATE_ROUND1, POLLSTER_RATINGS)
        assert result.sum() == pytest.approx(1.0, rel=1e-10)

    def test_returns_series(self, sample_df: pd.DataFrame) -> None:
        """Returns a Series with same length as the input DataFrame."""
        result = combined_weight(sample_df, ELECTION_DATE_ROUND1, POLLSTER_RATINGS)
        assert isinstance(result, pd.Series)
        assert len(result) == 3

    def test_election_day_gets_highest_weight(self) -> None:
        """When pollster and sample size are equal, nearer poll gets higher weight."""
        df = pd.DataFrame(
            {
                "fecha": pd.to_datetime(
                    [
                        date(2022, 5, 29),
                        date(2022, 4, 29),
                    ]
                ),
                "encuestadora": ["Invamer", "Invamer"],
                "muestra_int_voto": pd.array([1000, 1000], dtype="Int64"),
                "gustavo_petro": [40.0, 38.0],
            }
        )
        result = combined_weight(df, ELECTION_DATE_ROUND1, POLLSTER_RATINGS)
        assert result.iloc[0] > result.iloc[1]

    def test_uses_muestra_int_voto_falling_back_to_muestra(self) -> None:
        """When muestra_int_voto is NA, falls back to muestra column."""
        df = pd.DataFrame(
            {
                "fecha": pd.to_datetime([date(2022, 5, 29)]),
                "encuestadora": ["Invamer"],
                "muestra_int_voto": pd.array([pd.NA], dtype="Int64"),
                "muestra": pd.array([1000], dtype="Int64"),
                "gustavo_petro": [40.0],
            }
        )
        result = combined_weight(df, ELECTION_DATE_ROUND1, POLLSTER_RATINGS)
        assert result.iloc[0] == pytest.approx(1.0)

    def test_uses_custom_half_life(self, sample_df: pd.DataFrame) -> None:
        """Custom half-life produces different weight distribution."""
        result_default = combined_weight(sample_df, ELECTION_DATE_ROUND1, POLLSTER_RATINGS)
        result_custom = combined_weight(
            sample_df, ELECTION_DATE_ROUND1, POLLSTER_RATINGS, half_life=60.0
        )
        assert not result_default.equals(result_custom)


class TestWeightedAverage:
    """Tests for ``weighted_average``."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Three-row DataFrame with multiple candidates."""
        return pd.DataFrame(
            {
                "fecha": pd.to_datetime(
                    [
                        date(2022, 5, 29),
                        date(2022, 5, 19),
                        date(2022, 4, 29),
                    ]
                ),
                "encuestadora": ["Invamer", "CNC", "Mosqueteros"],
                "muestra_int_voto": pd.array([1000, 800, 500], dtype="Int64"),
                "gustavo_petro": [42.0, 40.0, 38.0],
                "federico_gutierrez": [24.0, 25.0, 26.0],
                "rodolfo_hernandez": [28.0, 27.0, 30.0],
                "blanco": [3.0, 4.0, 3.0],
                "rest": [3.0, 4.0, 3.0],
            }
        )

    def test_returns_dict(self, sample_df: pd.DataFrame) -> None:
        """Returns a dictionary of candidate -> weighted average."""
        candidates = [
            "gustavo_petro",
            "federico_gutierrez",
            "rodolfo_hernandez",
            "blanco",
            "rest",
        ]
        result = weighted_average(sample_df, candidates, ELECTION_DATE_ROUND1, POLLSTER_RATINGS)
        assert isinstance(result, dict)

    def test_all_candidates_present(self, sample_df: pd.DataFrame) -> None:
        """Every candidate key appears in the result dict."""
        candidates = [
            "gustavo_petro",
            "federico_gutierrez",
            "rodolfo_hernandez",
            "blanco",
            "rest",
        ]
        result = weighted_average(sample_df, candidates, ELECTION_DATE_ROUND1, POLLSTER_RATINGS)
        for c in candidates:
            assert c in result

    def test_values_are_floats(self, sample_df: pd.DataFrame) -> None:
        """All weighted average values are Python floats."""
        candidates = ["gustavo_petro", "federico_gutierrez"]
        result = weighted_average(sample_df, candidates, ELECTION_DATE_ROUND1, POLLSTER_RATINGS)
        for v in result.values():
            assert isinstance(v, float)

    def test_values_in_plausible_range(self, sample_df: pd.DataFrame) -> None:
        """All weighted averages fall within [0, 100]."""
        candidates = ["gustavo_petro", "federico_gutierrez"]
        result = weighted_average(sample_df, candidates, ELECTION_DATE_ROUND1, POLLSTER_RATINGS)
        for v in result.values():
            assert 0.0 <= v <= 100.0

    def test_single_poll_returns_raw_value(self) -> None:
        """Single poll returns the raw vote share."""
        df = pd.DataFrame(
            {
                "fecha": pd.to_datetime([date(2022, 5, 29)]),
                "encuestadora": ["Invamer"],
                "muestra_int_voto": pd.array([1000], dtype="Int64"),
                "gustavo_petro": [42.0],
            }
        )
        result = weighted_average(df, ["gustavo_petro"], ELECTION_DATE_ROUND1, POLLSTER_RATINGS)
        assert result["gustavo_petro"] == pytest.approx(42.0)

    def test_arithmetic_correctness_with_known_weights(self) -> None:
        """Weighted average math is correct for a two-poll scenario."""
        df = pd.DataFrame(
            {
                "fecha": pd.to_datetime(
                    [
                        date(2022, 5, 29),
                        date(2022, 4, 29),
                    ]
                ),
                "encuestadora": ["Invamer", "Invamer"],
                "muestra_int_voto": pd.array([1000, 1000], dtype="Int64"),
                "candidate_a": [50.0, 30.0],
            }
        )
        result = weighted_average(df, ["candidate_a"], ELECTION_DATE_ROUND1, POLLSTER_RATINGS)
        expected = (50.0 * (1.0 / 1.5)) + (30.0 * (0.5 / 1.5))
        assert result["candidate_a"] == pytest.approx(expected, rel=1e-3)


class TestAggregationSnapshot:
    """Tests for ``aggregation_snapshot``."""

    @pytest.fixture
    def df(self) -> pd.DataFrame:
        """Three-row DataFrame with polls spanning April through May."""
        return pd.DataFrame(
            {
                "fecha": pd.to_datetime(
                    [
                        date(2022, 5, 29),
                        date(2022, 5, 15),
                        date(2022, 4, 1),
                    ]
                ),
                "encuestadora": ["Invamer", "CNC", "Mosqueteros"],
                "muestra_int_voto": pd.array([1000, 800, 500], dtype="Int64"),
                "gustavo_petro": [42.0, 40.0, 35.0],
            }
        )

    def test_returns_dict(self, df: pd.DataFrame) -> None:
        """Returns a dictionary mapping candidate keys to floats."""
        result = aggregation_snapshot(
            df,
            ["gustavo_petro"],
            date(2022, 5, 28),
            POLLSTER_RATINGS,
        )
        assert isinstance(result, dict)

    def test_excludes_polls_after_cut_date(self, df: pd.DataFrame) -> None:
        """Polls after the as_of_date are excluded from the average."""
        as_of = date(2022, 5, 15)
        result = aggregation_snapshot(df, ["gustavo_petro"], as_of, POLLSTER_RATINGS)
        # Manually filter to the expected pre-cutoff rows and re-compute.
        # Both calls should produce the same answer if the cutoff is correct.
        filtered = df[df["fecha"] <= pd.Timestamp(as_of)].copy()
        expected = aggregation_snapshot(filtered, ["gustavo_petro"], as_of, POLLSTER_RATINGS)
        assert result["gustavo_petro"] == pytest.approx(expected["gustavo_petro"])

    def test_cut_date_before_all_polls_returns_nan(self, df: pd.DataFrame) -> None:
        """When no polls are before as_of_date, returns NaN."""
        result = aggregation_snapshot(
            df,
            ["gustavo_petro"],
            date(2022, 3, 1),
            POLLSTER_RATINGS,
        )
        assert pd.isna(result["gustavo_petro"])

    def test_cut_date_after_all_polls_returns_full_average(self, df: pd.DataFrame) -> None:
        """When all polls are before as_of_date, matches full weighted average."""
        result = aggregation_snapshot(
            df,
            ["gustavo_petro"],
            date(2022, 6, 1),
            POLLSTER_RATINGS,
        )
        full = weighted_average(df, ["gustavo_petro"], ELECTION_DATE_ROUND1, POLLSTER_RATINGS)
        assert result["gustavo_petro"] == pytest.approx(full["gustavo_petro"])

    def test_all_candidates_present(self, df: pd.DataFrame) -> None:
        """Every requested candidate key appears in the result."""
        candidates = ["gustavo_petro"]
        result = aggregation_snapshot(
            df,
            candidates,
            date(2022, 5, 28),
            POLLSTER_RATINGS,
        )
        for c in candidates:
            assert c in result


class TestEvolutionSeries:
    """Tests for ``evolution_series``."""

    @pytest.fixture
    def df(self) -> pd.DataFrame:
        """Four-row DataFrame spanning April through May."""
        return pd.DataFrame(
            {
                "fecha": pd.to_datetime(
                    [
                        date(2022, 5, 29),
                        date(2022, 5, 15),
                        date(2022, 5, 1),
                        date(2022, 4, 1),
                    ]
                ),
                "encuestadora": ["Invamer", "CNC", "Mosqueteros", "Invamer"],
                "muestra_int_voto": pd.array([1000, 800, 500, 1200], dtype="Int64"),
                "gustavo_petro": [42.0, 40.0, 38.0, 35.0],
                "federico_gutierrez": [24.0, 25.0, 26.0, 28.0],
            }
        )

    def test_returns_dataframe(self, df: pd.DataFrame) -> None:
        """Returns a pandas DataFrame."""
        result = evolution_series(
            df,
            ["gustavo_petro", "federico_gutierrez"],
            ELECTION_DATE_ROUND1,
            POLLSTER_RATINGS,
            n_snapshots=5,
        )
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self, df: pd.DataFrame) -> None:
        """Result has ``as_of_date``, ``candidate``, and ``weighted_average`` columns."""
        result = evolution_series(
            df,
            ["gustavo_petro", "federico_gutierrez"],
            ELECTION_DATE_ROUND1,
            POLLSTER_RATINGS,
            n_snapshots=5,
        )
        assert "as_of_date" in result.columns
        assert "candidate" in result.columns
        assert "weighted_average" in result.columns

    def test_correct_number_of_rows(self, df: pd.DataFrame) -> None:
        """Row count equals n_snapshots * len(candidates)."""
        candidates = ["gustavo_petro", "federico_gutierrez"]
        n_snapshots = 5
        result = evolution_series(
            df, candidates, ELECTION_DATE_ROUND1, POLLSTER_RATINGS, n_snapshots=n_snapshots
        )
        assert len(result) == n_snapshots * len(candidates)

    def test_dates_in_chronological_order(self, df: pd.DataFrame) -> None:
        """Snapshot dates are sorted chronologically."""
        result = evolution_series(
            df, ["gustavo_petro"], ELECTION_DATE_ROUND1, POLLSTER_RATINGS, n_snapshots=5
        )
        dates = result["as_of_date"].unique()
        for i in range(len(dates) - 1):
            assert dates[i] <= dates[i + 1]

    def test_has_all_candidates(self, df: pd.DataFrame) -> None:
        """All candidate keys appear in the candidate column."""
        candidates = ["gustavo_petro", "federico_gutierrez"]
        result = evolution_series(
            df, candidates, ELECTION_DATE_ROUND1, POLLSTER_RATINGS, n_snapshots=3
        )
        found = result["candidate"].unique()
        for c in candidates:
            assert c in found

    def test_weighted_averages_are_float(self, df: pd.DataFrame) -> None:
        """Weighted average column has floating-point dtype."""
        result = evolution_series(
            df, ["gustavo_petro"], ELECTION_DATE_ROUND1, POLLSTER_RATINGS, n_snapshots=3
        )
        assert np.issubdtype(result["weighted_average"].dtype, np.floating)

    def test_includes_trends_column_when_trends_df_provided(self, df: pd.DataFrame) -> None:
        """When ``trends_df`` is provided, result includes a ``trends`` column."""
        candidates = ["gustavo_petro", "federico_gutierrez"]
        # trends_df must be wide-format (date index, query-string columns)
        # matching what fetch_trends would return.
        trends_df = pd.DataFrame(
            {
                "Petro": [60.0, 60.0],
                "Fico": [40.0, 40.0],
            },
            index=pd.to_datetime(["2022-05-15", "2022-05-28"]),
        )
        result = evolution_series(
            df,
            candidates,
            ELECTION_DATE_ROUND1,
            POLLSTER_RATINGS,
            n_snapshots=5,
            trends_df=trends_df,
        )
        assert "trends" in result.columns
        # Verify trends values were actually merged (not NaN fallback).
        trends_values = result.dropna(subset=["trends"])
        assert len(trends_values) > 0, "Expected non-NaN trends values after merge"
        # gustavo_petro should have prop_fav ≈ 0.6, federico ≈ 0.4.
        for _, row in trends_values.iterrows():
            if row["candidate"] == "gustavo_petro":
                assert float(row["trends"]) == pytest.approx(0.6, abs=0.01)
            elif row["candidate"] == "federico_gutierrez":
                assert float(row["trends"]) == pytest.approx(0.4, abs=0.01)

    def test_backward_alignment_matches_nearest_earlier_trend(self, df: pd.DataFrame) -> None:
        """``merge_asof`` backward matches the nearest trend date ≤ snapshot.

        Uses distinct proportions per trend date so the test fails if
        alignment direction is wrong (e.g., nearest instead of backward).
        """
        candidates = ["gustavo_petro", "federico_gutierrez"]
        trends_df = pd.DataFrame(
            {
                "Petro": [70.0, 50.0],
                "Fico": [30.0, 50.0],
            },
            index=pd.to_datetime(["2022-05-15", "2022-05-28"]),
        )
        result = evolution_series(
            df,
            candidates,
            ELECTION_DATE_ROUND1,
            POLLSTER_RATINGS,
            n_snapshots=5,
            trends_df=trends_df,
        )
        trends_values = result.dropna(subset=["trends"])
        # Only the 05-27 snapshot falls between the two trend dates (≥ 05-15, < 05-28).
        # Backward alignment must match 05-15, not 05-28.
        assert len(trends_values) > 0, "Expected non-NaN trend matches"
        for _, row in trends_values.iterrows():
            if row["candidate"] == "gustavo_petro":
                assert float(row["trends"]) == pytest.approx(0.7, abs=0.01)
            elif row["candidate"] == "federico_gutierrez":
                assert float(row["trends"]) == pytest.approx(0.3, abs=0.01)
