"""SPEC-12.1: Tests for DIVIPOLA master registry ingestion."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from co_president.ingestion.ingest_divipola import (
    build_divipola_master,
    fetch_divipola_github,
    fetch_divipola_socrata,
    validate_divipola,
)

if TYPE_CHECKING:
    from pathlib import Path

__all__: list[str] = []


def _build_valid_divipola(n: int = 1122) -> pd.DataFrame:
    """Build a synthetic DIVIPOLA DataFrame with ``n`` municipalities.

    Columns match the expected Socrata API output.
    """
    rng = np.random.default_rng(seed=42)
    return pd.DataFrame(
        {
            "codigo_municipio": [f"{i:05d}" for i in range(1, n + 1)],
            "nombre_municipio": [f"Municipio_{i}" for i in range(1, n + 1)],
            "departamento": ["Antioquia"] * n,
            "latitud": rng.uniform(0, 10, size=n).round(4),
            "longitud": rng.uniform(-80, -70, size=n).round(4),
        }
    )


# ═══════════════════════════════════════════════════════════════════
# Fetch functions (structural tests — no real API calls)
# ═══════════════════════════════════════════════════════════════════


class TestFetchDivipolaSocrata:
    """Structural contract for ``fetch_divipola_socrata``."""

    def test_is_callable(self) -> None:
        """Function is importable and callable (may be @retry-wrapped)."""
        func = getattr(fetch_divipola_socrata, "__wrapped__", fetch_divipola_socrata)
        assert callable(func)

    def test_return_annotation_is_dataframe(self) -> None:
        """Return type annotation is pd.DataFrame."""
        sig = inspect.signature(fetch_divipola_socrata)
        assert sig.return_annotation == "pd.DataFrame"


class TestFetchDivipolaGithub:
    """Structural contract for ``fetch_divipola_github``."""

    def test_is_callable(self) -> None:
        """Function is importable and callable."""
        assert callable(fetch_divipola_github)

    def test_return_annotation_is_dataframe(self) -> None:
        """Return type annotation is pd.DataFrame."""
        sig = inspect.signature(fetch_divipola_github)
        assert sig.return_annotation == "pd.DataFrame"


# ═══════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════


class TestValidateDivipola:
    """``validate_divipola`` enforces size, uniqueness, and non-null codes."""

    def test_valid_data(self) -> None:
        """A well-formed 1 122-row DataFrame passes without error."""
        df = _build_valid_divipola(1122)
        result = validate_divipola(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1122

    def test_valid_data_more_than_minimum(self) -> None:
        """A DataFrame with >1 122 rows also passes (threshold is >=)."""
        df = _build_valid_divipola(1200)
        result = validate_divipola(df)
        assert len(result) >= 1122

    def test_null_codes_raises(self) -> None:
        """Null ``codigo_municipio`` values raise ``ValueError``."""
        df = _build_valid_divipola(1122)
        df.loc[0, "codigo_municipio"] = None  # type: ignore[typeddict-item]
        with pytest.raises(ValueError, match="Null codes"):
            validate_divipola(df)

    def test_duplicate_codes_raises(self) -> None:
        """Duplicate ``codigo_municipio`` values raise ``ValueError``."""
        df = _build_valid_divipola(1122)
        df.loc[1, "codigo_municipio"] = df.loc[0, "codigo_municipio"]
        with pytest.raises(ValueError, match="Duplicate codes"):
            validate_divipola(df)

    def test_too_few_rows_raises(self) -> None:
        """Fewer than 1 122 rows raises ``ValueError``."""
        df = _build_valid_divipola(1121)
        with pytest.raises(ValueError, match="Expected >= 1122"):
            validate_divipola(df)

    def test_empty_dataframe_raises(self) -> None:
        """An empty DataFrame raises ``ValueError``."""
        df = _build_valid_divipola(0)
        with pytest.raises(ValueError, match="Expected >= 1122"):
            validate_divipola(df)


# ═══════════════════════════════════════════════════════════════════
# Build pipeline
# ═══════════════════════════════════════════════════════════════════


class TestBuildDivipolaMaster:
    """``build_divipola_master`` writes a validated CSV."""

    def test_creates_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Produces ``divipola_master.csv`` in the given data directory."""
        df = _build_valid_divipola(1122)

        def _mock_fetch() -> pd.DataFrame:
            return df

        monkeypatch.setattr(
            "co_president.ingestion.ingest_divipola.fetch_divipola_socrata",
            _mock_fetch,
        )

        build_divipola_master(data_dir=tmp_path)
        target = tmp_path / "fundamentals" / "divipola_master.csv"
        assert target.is_file(), f"Expected {target} to exist"

    def test_saved_file_columns(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The saved CSV contains the expected columns."""
        df = _build_valid_divipola(1122)

        def _mock_fetch() -> pd.DataFrame:
            return df

        monkeypatch.setattr(
            "co_president.ingestion.ingest_divipola.fetch_divipola_socrata",
            _mock_fetch,
        )

        build_divipola_master(data_dir=tmp_path)
        saved = pd.read_csv(tmp_path / "fundamentals" / "divipola_master.csv")
        expected = {"codigo_municipio", "nombre_municipio", "departamento", "latitud", "longitud"}
        assert expected.issubset(set(saved.columns))

    def test_fetch_socrata_fallback_to_github(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the Socrata fetch raises, the GitHub fallback is used."""
        df = _build_valid_divipola(1122)

        def _mock_fail() -> pd.DataFrame:
            msg = "API unavailable"
            raise ConnectionError(msg)

        def _mock_github() -> pd.DataFrame:
            return df

        monkeypatch.setattr(
            "co_president.ingestion.ingest_divipola.fetch_divipola_socrata",
            _mock_fail,
        )
        monkeypatch.setattr(
            "co_president.ingestion.ingest_divipola.fetch_divipola_github",
            _mock_github,
        )

        build_divipola_master(data_dir=tmp_path)
        target = tmp_path / "fundamentals" / "divipola_master.csv"
        assert target.is_file()
