"""SPEC-12.2: Tests for historical election results ingestion."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from co_president.ingestion.ingest_historical import (
    build_historical_matrix,
    compute_derived_features,
    fetch_cedae_results,
    fetch_datos_gov_results,
)

if TYPE_CHECKING:
    from pathlib import Path


__all__: list[str] = []


def _make_historical_input() -> pd.DataFrame:
    """Synthetic 3-municipality historical data for 4 election years.

    Columns match the expected CEDAE API output format.
    """
    data = {
        "codigo_municipio": [
            "05001",
            "05001",
            "05001",
            "05001",
            "05001",
            "05001",
            "05002",
            "05002",
            "05002",
            "05003",
            "05003",
            "05003",
        ],
        "year": [
            2014,
            2014,
            2018,
            2018,
            2022,
            2022,
            2014,
            2018,
            2022,
            2014,
            2018,
            2022,
        ],
        "round": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "candidate": [
            "oscar_ivan_zuluaga",
            "gustavo_petro",
            "ivan_duque",
            "gustavo_petro",
            "rodolfo_hernandez",
            "gustavo_petro",
            "oscar_ivan_zuluaga",
            "ivan_duque",
            "gustavo_petro",
            "oscar_ivan_zuluaga",
            "ivan_duque",
            "gustavo_petro",
        ],
        "votes": [
            5000,
            4000,
            6000,
            3000,
            5000,
            5500,
            3000,
            4000,
            3500,
            1000,
            2000,
            2500,
        ],
        "total_votes": [
            10000,
            10000,
            10000,
            10000,
            12000,
            12000,
            5000,
            6000,
            7000,
            3000,
            4000,
            5000,
        ],
        "registered_voters": [
            15000,
            15000,
            15000,
            15000,
            18000,
            18000,
            8000,
            9000,
            10000,
            5000,
            6000,
            7000,
        ],
    }
    return pd.DataFrame(data)


# ═══════════════════════════════════════════════════════════════════
# Fetch functions (structural tests — no real API calls)
# ═══════════════════════════════════════════════════════════════════


class TestFetchCedaeResults:
    """Contract and behavior tests for ``fetch_cedae_results``."""

    def test_is_callable(self) -> None:
        """Function is importable and callable."""
        assert callable(fetch_cedae_results)

    def test_signature_has_expected_params(self) -> None:
        """Accepts year (int) and round_num (int) parameters."""
        sig = inspect.signature(fetch_cedae_results)
        params = list(sig.parameters.keys())
        assert "year" in params, "year parameter missing"
        assert "round_num" in params, "round_num parameter missing"

    def test_return_annotation_is_dataframe(self) -> None:
        """Return type annotation is pd.DataFrame."""
        sig = inspect.signature(fetch_cedae_results)
        assert sig.return_annotation == "pd.DataFrame"


class TestFetchDatosGovResults:
    """Structural contract for ``fetch_datos_gov_results``."""

    def test_is_callable(self) -> None:
        """Function is importable and callable."""
        assert callable(fetch_datos_gov_results)

    def test_accepts_no_args(self) -> None:
        """Accepts zero arguments."""
        sig = inspect.signature(fetch_datos_gov_results)
        assert len(sig.parameters) == 0


# ═══════════════════════════════════════════════════════════════════
# Feature computation
# ═══════════════════════════════════════════════════════════════════


class TestComputeDerivedFeatures:
    """``compute_derived_features`` calculates vote shares and abstention."""

    def test_returns_dataframe(self) -> None:
        """Returns a DataFrame."""
        df = _make_historical_input()
        result = compute_derived_features(df)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self) -> None:
        """Output contains expected column groups."""
        df = _make_historical_input()
        result = compute_derived_features(df)
        cols = set(result.columns)
        assert "codigo_municipio" in cols
        assert "year" in cols
        assert "candidate" in cols
        assert "vote_share" in cols
        assert "abstention_rate" in cols

    def test_vote_share_range(self) -> None:
        """Vote shares are between 0 and 1."""
        df = _make_historical_input()
        result = compute_derived_features(df)
        assert result["vote_share"].between(0, 1).all()

    def test_abstention_rate_range(self) -> None:
        """Abstention rates are between 0 and 1."""
        df = _make_historical_input()
        result = compute_derived_features(df)
        assert result["abstention_rate"].between(0, 1).all()

    def test_abstention_calculation(self) -> None:
        """Abstention = 1 - (total_votes / registered_voters)."""
        df = _make_historical_input()
        result = compute_derived_features(df)
        row = result[result["candidate"] == "gustavo_petro"].iloc[0]
        expected_row = df[(df["candidate"] == "gustavo_petro")].iloc[0]
        expected_abstention = 1 - (expected_row["total_votes"] / expected_row["registered_voters"])
        assert row["abstention_rate"] == pytest.approx(expected_abstention, abs=0.01)

    def test_vote_share_calculation(self) -> None:
        """Vote share = votes / total_votes."""
        df = _make_historical_input()
        result = compute_derived_features(df)
        row = result[result["candidate"] == "gustavo_petro"].iloc[0]
        expected_row = df[(df["candidate"] == "gustavo_petro")].iloc[0]
        expected_share = expected_row["votes"] / expected_row["total_votes"]
        assert row["vote_share"] == pytest.approx(expected_share, abs=0.01)

    def test_zero_total_votes_yields_zero_share(self) -> None:
        """Rows with total_votes == 0 get vote_share == 0.0."""
        df = pd.DataFrame(
            {
                "codigo_municipio": ["05001"],
                "year": [2022],
                "round": [1],
                "candidate": ["test_candidate"],
                "votes": [0],
                "total_votes": [0],
                "registered_voters": [10000],
            }
        )
        result = compute_derived_features(df)
        assert result["vote_share"].iloc[0] == 0.0

    def test_zero_registered_yields_full_abstention(self) -> None:
        """Rows with registered_voters == 0 get abstention_rate == 1.0."""
        df = pd.DataFrame(
            {
                "codigo_municipio": ["05001"],
                "year": [2022],
                "round": [1],
                "candidate": ["test_candidate"],
                "votes": [500],
                "total_votes": [1000],
                "registered_voters": [0],
            }
        )
        result = compute_derived_features(df)
        assert result["abstention_rate"].iloc[0] == 1.0


# ═══════════════════════════════════════════════════════════════════
# Build pipeline
# ═══════════════════════════════════════════════════════════════════


class TestBuildHistoricalMatrix:
    """``build_historical_matrix`` writes a validated CSV."""

    def test_creates_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Produces ``historical_results.csv`` in the given data directory."""
        df = _make_historical_input()

        monkeypatch.setattr(
            "co_president.ingestion.ingest_historical._fetch_all_years",
            lambda: df,
        )

        build_historical_matrix(data_dir=tmp_path)
        target = tmp_path / "fundamentals" / "historical_results.csv"
        assert target.is_file(), f"Expected {target} to exist"

    def test_saved_file_non_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The saved CSV has rows and computed feature columns."""
        df = _make_historical_input()

        monkeypatch.setattr(
            "co_president.ingestion.ingest_historical._fetch_all_years",
            lambda: df,
        )

        build_historical_matrix(data_dir=tmp_path)
        saved = pd.read_csv(tmp_path / "fundamentals" / "historical_results.csv")
        assert len(saved) > 0, "Saved CSV should have at least one row"
        assert "vote_share" in saved.columns, "vote_share column missing"
        assert "abstention_rate" in saved.columns, "abstention_rate column missing"
        assert saved["vote_share"].notna().all(), "vote_share should have no nulls"
        assert saved["abstention_rate"].notna().all(), "abstention_rate should have no nulls"
