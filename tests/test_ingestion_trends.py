"""SPEC-38: Tests for Google Trends ingestion.

Tests cover ``fetch_trends`` mocking, ``compute_prop_fav``
computations, and the combined Google+YouTube mode.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from co_president.ingestion.ingest_trends import (
    _fetch_pytrends,
    compute_prop_fav,
    fetch_trends,
)


class TestFetchTrends:
    """Tests for ``fetch_trends`` (live API calls mocked)."""

    @patch("co_president.ingestion.ingest_trends._fetch_pytrends")
    def test_single_day_single_keyword(self, mock_fetch: MagicMock) -> None:
        """fetch_trends('Petro', '2022-06-18', '2022-06-18') returns DataFrame."""
        mock_fetch.return_value = pd.DataFrame(
            {"Gustavo Petro": [80.0]},
            index=pd.to_datetime(["2022-06-18"]),
        )
        result = fetch_trends(
            "Gustavo Petro",
            start_date="2022-06-18",
            end_date="2022-06-18",
        )
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert "Gustavo Petro" in result.columns

    @patch("co_president.ingestion.ingest_trends._fetch_pytrends")
    def test_combined_mode_averages(self, mock_fetch: MagicMock) -> None:
        """Combined Google+YouTube mode averages both sources."""
        mock_fetch.side_effect = [
            pd.DataFrame(
                {"Petro": [100.0, 50.0]},
                index=pd.to_datetime(["2022-06-17", "2022-06-18"]),
            ),
            pd.DataFrame(
                {"Petro": [60.0, 30.0]},
                index=pd.to_datetime(["2022-06-17", "2022-06-18"]),
            ),
        ]
        result = fetch_trends(
            "Petro",
            start_date="2022-06-17",
            end_date="2022-06-18",
        )
        expected = pd.DataFrame(
            {"Petro": [80.0, 40.0]},
            index=pd.to_datetime(["2022-06-17", "2022-06-18"]),
        )
        pd.testing.assert_frame_equal(result, expected)

    @patch("co_president.ingestion.ingest_trends._fetch_pytrends")
    def test_both_empty_returns_empty(self, mock_fetch: MagicMock) -> None:
        """When both web and YouTube return empty, result is empty DataFrame."""
        mock_fetch.return_value = pd.DataFrame()
        result = fetch_trends("Petro", start_date="2022-06-18", end_date="2022-06-18")
        assert result.empty

    @patch("co_president.ingestion.ingest_trends._fetch_pytrends")
    def test_default_window_t1(self, mock_fetch: MagicMock) -> None:
        """fetch_trends with default window='T-1' resolves correctly.

        The resolved timeframe should end at yesterday (not today) per
        SciELO 2023 T-1 design.
        """
        today = pd.Timestamp.today().normalize()
        yesterday = today - pd.Timedelta(days=1)
        mock_fetch.return_value = pd.DataFrame(
            {"Petro": [80.0]},
            index=pd.to_datetime([yesterday]),
        )
        result = fetch_trends("Petro")
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        # Verify _fetch_pytrends was called with a timeframe containing
        # yesterday's date (not today).
        timeframe_arg = mock_fetch.call_args_list[0][0][1]
        yesterday_str = yesterday.strftime("%Y-%m-%d")
        assert yesterday_str in timeframe_arg
        # The single expected row should be at yesterday's date.
        pd.testing.assert_frame_equal(result, mock_fetch.return_value)


class TestFetchPytrends:
    """Tests for the internal ``_fetch_pytrends`` helper."""

    @patch("co_president.ingestion.ingest_trends.TrendReq")
    def test_drops_is_partial_column(self, mock_trend_req: MagicMock) -> None:
        """_fetch_pytrends drops the 'isPartial' column from the response."""
        mock_instance = MagicMock()
        mock_trend_req.return_value = mock_instance
        mock_instance.interest_over_time.return_value = pd.DataFrame(
            {"Petro": [80.0], "isPartial": [False]},
            index=pd.to_datetime(["2022-06-18"]),
        )
        result = _fetch_pytrends(["Petro"], "2022-06-18 2022-06-18")
        assert "isPartial" not in result.columns
        assert "Petro" in result.columns


class TestComputePropFav:
    """Tests for ``compute_prop_fav``."""

    def test_two_candidates_equal_interest(self) -> None:
        """With equal interest, prop_fav is 0.5 for both."""
        trends_df = pd.DataFrame(
            {"Gustavo Petro": [50.0], "Federico Guti\u00e9rrez": [50.0]},
            index=pd.to_datetime(["2022-06-17"]),
        )
        query_map = {
            "gustavo_petro": "Gustavo Petro",
            "federico_gutierrez": "Federico Guti\u00e9rrez",
        }
        result = compute_prop_fav(trends_df, query_map)
        assert len(result) == 2
        assert set(result["candidate"]) == {"gustavo_petro", "federico_gutierrez"}
        assert (
            round(float(result.loc[result["candidate"] == "gustavo_petro", "prop_fav"].iloc[0]), 2)
            == 0.5
        )

    def test_unequal_interest(self) -> None:
        """prop_fav reflects interest proportion."""
        trends_df = pd.DataFrame(
            {"Gustavo Petro": [80.0], "Federico Guti\u00e9rrez": [20.0]},
            index=pd.to_datetime(["2022-06-17"]),
        )
        query_map = {
            "gustavo_petro": "Gustavo Petro",
            "federico_gutierrez": "Federico Guti\u00e9rrez",
        }
        result = compute_prop_fav(trends_df, query_map)
        petro_val = float(result.loc[result["candidate"] == "gustavo_petro", "prop_fav"].iloc[0])
        fed_val = float(result.loc[result["candidate"] == "federico_gutierrez", "prop_fav"].iloc[0])
        assert round(petro_val, 2) == 0.8
        assert round(fed_val, 2) == 0.2

    def test_unknown_columns_skipped(self) -> None:
        """Candidates not in trends_df columns are omitted."""
        trends_df = pd.DataFrame(
            {"Gustavo Petro": [100.0]},
            index=pd.to_datetime(["2022-06-17"]),
        )
        query_map = {
            "gustavo_petro": "Gustavo Petro",
            "federico_gutierrez": "Federico Guti\u00e9rrez",
        }
        result = compute_prop_fav(trends_df, query_map)
        assert len(result) == 1
        assert result["candidate"].iloc[0] == "gustavo_petro"

    def test_zero_interest_rows_dropped(self) -> None:
        """Rows where all candidates have zero interest are dropped."""
        trends_df = pd.DataFrame(
            {"Gustavo Petro": [0.0, 50.0], "Federico Guti\u00e9rrez": [0.0, 50.0]},
            index=pd.to_datetime(["2022-06-17", "2022-06-18"]),
        )
        query_map = {
            "gustavo_petro": "Gustavo Petro",
            "federico_gutierrez": "Federico Guti\u00e9rrez",
        }
        result = compute_prop_fav(trends_df, query_map)
        assert len(result) == 2
        assert all(result["as_of_date"] == pd.to_datetime("2022-06-18").date())

    def test_empty_trends_df(self) -> None:
        """Empty trends_df returns empty result."""
        trends_df = pd.DataFrame()
        result = compute_prop_fav(trends_df, {"gustavo_petro": "Gustavo Petro"})
        assert result.empty
        assert list(result.columns) == ["as_of_date", "candidate", "prop_fav"]
