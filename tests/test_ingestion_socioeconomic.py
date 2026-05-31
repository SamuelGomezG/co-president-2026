"""SPEC-13.1: Tests for socioeconomic data ingestion."""

from __future__ import annotations

import inspect
import typing
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from co_president.ingestion.ingest_socioeconomic import (
    build_socioeconomic_matrix,
    calculate_features,
    fetch_dane_csv,
    fetch_poverty_indicators,
    scrape_dane_portal_playwright,
)

if TYPE_CHECKING:
    from pathlib import Path

__all__: list[str] = []


def _make_census_input() -> pd.DataFrame:
    """Synthetic DANE census DataFrame with 3 municipalities."""
    return pd.DataFrame(
        {
            "codigo_municipio": ["11001", "05001", "76001"],
            "nombre_municipio": ["Bogota D.C.", "Medellin", "Cali"],
            "poblacion_total": [7_181_469, 2_427_129, 2_172_527],
            "poblacion_afrocolombiana": [95_000, 260_000, 580_000],
            "poblacion_indigena": [14_000, 3_500, 2_100],
            "poblacion_rural_dispersa": [0, 30_000, 2_000],
            "anos_escolaridad": [11.5, 10.2, 10.5],
            "hogares_con_internet": [2_100_000, 680_000, 580_000],
            "hogares_totales": [2_650_000, 850_000, 750_000],
        }
    )


def _make_poverty_input() -> pd.DataFrame:
    """Synthetic IPM/NBI DataFrame."""
    return pd.DataFrame(
        {
            "codigo_municipio": ["11001", "05001", "76001"],
            "ipm_score": [0.045, 0.125, 0.098],
            "nbi_rate": [0.032, 0.098, 0.071],
        }
    )


def _make_projections_input() -> pd.DataFrame:
    """Synthetic population projections DataFrame."""
    return pd.DataFrame(
        {
            "codigo_municipio": ["11001", "05001", "76001"],
            "proyeccion_2022": [7_900_000, 2_569_007, 2_300_000],
        }
    )


# ═══════════════════════════════════════════════════════════════════
# Fetch functions (structural tests — no real API calls)
# ═══════════════════════════════════════════════════════════════════


class TestFetchDaneCsv:
    """Structural contract for ``fetch_dane_csv``."""

    def test_is_callable(self) -> None:
        """Function is importable and callable."""
        assert callable(fetch_dane_csv)

    def test_signature_has_expected_params(self) -> None:
        """Accepts a single string URL parameter."""
        sig = inspect.signature(fetch_dane_csv)
        params = list(sig.parameters.keys())
        assert "dane_url" in params

    def test_return_annotation_is_optional_dataframe(self) -> None:
        """Return type annotation is ``pd.DataFrame | None``."""
        sig = inspect.signature(fetch_dane_csv)
        assert "pd.DataFrame | None" in sig.return_annotation


class TestScrapeDanePortalPlaywright:
    """Structural contract for ``scrape_dane_portal_playwright``."""

    def test_is_callable(self) -> None:
        """Function is importable and callable."""
        assert callable(scrape_dane_portal_playwright)

    def test_return_annotation_optional_dataframe(self) -> None:
        """Return type annotation is ``pd.DataFrame | None``."""
        sig = inspect.signature(scrape_dane_portal_playwright)
        assert "pd.DataFrame | None" in sig.return_annotation


class TestFetchPovertyIndicators:
    """Structural contract for ``fetch_poverty_indicators``."""

    def test_is_callable(self) -> None:
        """Function is importable and callable."""
        assert callable(fetch_poverty_indicators)

    def test_return_annotation_is_dataframe(self) -> None:
        """Return type annotation is ``pd.DataFrame``."""
        hints = typing.get_type_hints(fetch_poverty_indicators)
        assert hints["return"] is pd.DataFrame


# ═══════════════════════════════════════════════════════════════════
# Feature computation
# ═══════════════════════════════════════════════════════════════════


class TestCalculateFeatures:
    """``calculate_features`` normalises census variables and merges with poverty data."""

    def test_returns_dataframe(self) -> None:
        """Returns a DataFrame."""
        census = _make_census_input()
        poverty = _make_poverty_input()
        projections = _make_projections_input()
        result = calculate_features(census, poverty, projections)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self) -> None:
        """Output contains all expected socioeconomic columns."""
        census = _make_census_input()
        poverty = _make_poverty_input()
        projections = _make_projections_input()
        result = calculate_features(census, poverty, projections)
        expected = {
            "codigo_municipio",
            "pct_afro_colombian",
            "pct_indigenous",
            "pct_rural_disperso",
            "years_schooling",
            "internet_access_rate",
            "ipm_score",
            "nbi_rate",
            "population_2022",
        }
        assert expected.issubset(set(result.columns))

    def test_percentages_in_range(self) -> None:
        """All normalised percentages are between 0 and 1."""
        census = _make_census_input()
        poverty = _make_poverty_input()
        projections = _make_projections_input()
        result = calculate_features(census, poverty, projections)
        for col in (
            "pct_afro_colombian",
            "pct_indigenous",
            "pct_rural_disperso",
            "internet_access_rate",
        ):
            assert result[col].between(0, 1).all(), f"{col} out of [0, 1] range"

    def test_ipm_score_in_range(self) -> None:
        """IPM scores are between 0 and 1."""
        census = _make_census_input()
        poverty = _make_poverty_input()
        projections = _make_projections_input()
        result = calculate_features(census, poverty, projections)
        assert result["ipm_score"].between(0, 1).all()

    def test_population_2022_positive(self) -> None:
        """Population projections are positive integers."""
        census = _make_census_input()
        poverty = _make_poverty_input()
        projections = _make_projections_input()
        result = calculate_features(census, poverty, projections)
        assert (result["population_2022"] > 0).all()

    def test_pct_afro_calculation(self) -> None:
        """Afro-colombian percentage is computed as ratio of total population."""
        census = _make_census_input()
        poverty = _make_poverty_input()
        projections = _make_projections_input()
        result = calculate_features(census, poverty, projections)
        row = result[result["codigo_municipio"] == "11001"].iloc[0]
        expected = 95_000 / 7_181_469
        assert row["pct_afro_colombian"] == pytest.approx(expected, abs=0.001)

    def test_zero_denominator_guard(self) -> None:
        """Zero population total yields zero percentage without division error."""
        census = _make_census_input()
        census.loc[0, "poblacion_total"] = 0
        poverty = _make_poverty_input()
        projections = _make_projections_input()
        result = calculate_features(census, poverty, projections)
        assert result["pct_afro_colombian"].iloc[0] == 0.0

    def test_missing_column_handling(self) -> None:
        """Missing columns in census data are handled gracefully (zero fill)."""
        minimal = pd.DataFrame({"codigo_municipio": ["11001"]})
        poverty = _make_poverty_input()
        projections = _make_projections_input()
        result = calculate_features(minimal, poverty, projections)
        assert "pct_afro_colombian" in result.columns
        assert result["pct_afro_colombian"].iloc[0] == 0.0

    def test_null_projections_uses_pd_na(self) -> None:
        """When projections is empty, ``population_2022`` contains NA."""
        census = _make_census_input()
        poverty = _make_poverty_input()
        empty_projections = pd.DataFrame()
        result = calculate_features(census, poverty, empty_projections)
        assert result["population_2022"].isna().all()


# ═══════════════════════════════════════════════════════════════════
# Build pipeline
# ═══════════════════════════════════════════════════════════════════


class TestBuildSocioeconomicMatrix:
    """``build_socioeconomic_matrix`` writes a validated CSV."""

    def test_creates_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Produces ``socioeconomic.csv`` in the given data directory."""
        census = _make_census_input()
        poverty = _make_poverty_input()
        projections = _make_projections_input()

        monkeypatch.setattr(
            "co_president.ingestion.ingest_socioeconomic._fetch_census_fallback",
            lambda: census,
        )
        monkeypatch.setattr(
            "co_president.ingestion.ingest_socioeconomic.fetch_poverty_indicators",
            lambda: poverty,
        )
        monkeypatch.setattr(
            "co_president.ingestion.ingest_socioeconomic._fetch_population_projections",
            lambda: projections,
        )

        build_socioeconomic_matrix(data_dir=tmp_path)
        target = tmp_path / "fundamentals" / "socioeconomic.csv"
        assert target.is_file(), f"Expected {target} to exist"

    def test_saved_file_columns(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The saved CSV contains the expected columns."""
        census = _make_census_input()
        poverty = _make_poverty_input()
        projections = _make_projections_input()

        monkeypatch.setattr(
            "co_president.ingestion.ingest_socioeconomic._fetch_census_fallback",
            lambda: census,
        )
        monkeypatch.setattr(
            "co_president.ingestion.ingest_socioeconomic.fetch_poverty_indicators",
            lambda: poverty,
        )
        monkeypatch.setattr(
            "co_president.ingestion.ingest_socioeconomic._fetch_population_projections",
            lambda: projections,
        )

        build_socioeconomic_matrix(data_dir=tmp_path)
        saved = pd.read_csv(tmp_path / "fundamentals" / "socioeconomic.csv")
        expected = {
            "codigo_municipio",
            "pct_afro_colombian",
            "pct_indigenous",
            "pct_rural_disperso",
            "years_schooling",
            "internet_access_rate",
            "ipm_score",
            "nbi_rate",
            "population_2022",
        }
        assert expected.issubset(set(saved.columns))
