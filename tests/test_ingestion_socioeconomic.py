"""SPEC-13a: Tests for socioeconomic data ingestion."""

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
    scrape_dane_portal_playwright,
    validate_socioeconomic,
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
        hints = typing.get_type_hints(fetch_dane_csv)
        return_type = hints["return"]
        # In Python 3.10+, pd.DataFrame | None is a types.UnionType
        origin = typing.get_origin(return_type)
        args = typing.get_args(return_type)
        assert origin is not None  # it's a union
        assert pd.DataFrame in args
        assert type(None) in args


class TestScrapeDanePortalPlaywright:
    """Structural contract for ``scrape_dane_portal_playwright``."""

    def test_is_callable(self) -> None:
        """Function is importable and callable."""
        assert callable(scrape_dane_portal_playwright)

    def test_return_annotation_optional_dataframe(self) -> None:
        """Return type annotation is ``pd.DataFrame | None``."""
        sig = inspect.signature(scrape_dane_portal_playwright)
        assert "pd.DataFrame | None" in sig.return_annotation


# ═══════════════════════════════════════════════════════════════════
# Feature computation
# ═══════════════════════════════════════════════════════════════════


class TestCalculateFeatures:
    """``calculate_features`` normalises census variables (census-only, no poverty)."""

    def test_returns_dataframe(self) -> None:
        """Returns a DataFrame."""
        census = _make_census_input()
        poverty = _make_poverty_input()
        projections = _make_projections_input()
        result = calculate_features(census, poverty, projections)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self) -> None:
        """Output contains all expected socioeconomic columns (census-only, no poverty)."""
        census = _make_census_input()
        poverty = pd.DataFrame()
        projections = _make_projections_input()
        result = calculate_features(census, poverty, projections)
        expected = {
            "codigo_municipio",
            "pct_afro_colombian",
            "pct_indigenous",
            "pct_rural_disperso",
            "years_schooling",
            "internet_access_rate",
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


class TestValidateSocioeconomic:
    """``validate_socioeconomic`` checks output against acceptance criteria."""

    def _make_valid_features(self, n: int = 3) -> pd.DataFrame:
        """Build a valid socioeconomic feature DataFrame with ``n`` municipalities."""
        return pd.DataFrame(
            {
                "codigo_municipio": [f"{i:05d}" for i in range(1, n + 1)],
                "pct_afro_colombian": [0.1] * n,
                "pct_indigenous": [0.02] * n,
                "pct_rural_disperso": [0.3] * n,
                "years_schooling": [8.0] * n,
                "internet_access_rate": [0.5] * n,
                "population_2022": [100_000] * n,
            }
        )

    def test_valid_data_returns_empty_warnings(self) -> None:
        """Valid data with 1,122 municipalities yields no warnings."""
        df = self._make_valid_features(n=1122)
        result = validate_socioeconomic(df)
        assert result == []

    def test_missing_codigo_municipio_warns(self) -> None:
        """Missing codigo_municipio column triggers a warning."""
        df = self._make_valid_features().drop(columns=["codigo_municipio"])
        result = validate_socioeconomic(df)
        assert len(result) >= 1
        assert any("codigo_municipio" in w for w in result)

    def test_null_in_internet_access_rate_warns(self) -> None:
        """Null values in internet_access_rate trigger a warning."""
        df = self._make_valid_features()
        df.loc[0, "internet_access_rate"] = pd.NA
        result = validate_socioeconomic(df)
        assert any("internet_access_rate" in w for w in result)

    def test_null_in_population_2022_warns(self) -> None:
        """Null values in population_2022 trigger a warning."""
        df = self._make_valid_features()
        df.loc[0, "population_2022"] = pd.NA
        result = validate_socioeconomic(df)
        assert any("population_2022" in w for w in result)

    def test_negative_population_warns(self) -> None:
        """Negative population projection triggers a warning."""
        df = self._make_valid_features()
        df.loc[0, "population_2022"] = -100
        result = validate_socioeconomic(df)
        assert any("population_2022" in w for w in result)

    def test_zero_population_warns(self) -> None:
        """Zero population projection triggers a warning (positive required)."""
        df = self._make_valid_features()
        df.loc[0, "population_2022"] = 0
        result = validate_socioeconomic(df)
        assert any("population_2022" in w for w in result)

    def test_range_warnings_fire_on_raw_unclipped_values(self) -> None:
        """validate_socioeconomic catches out-of-range rates from calculate_features output."""
        census = _make_census_input()
        census.loc[census["codigo_municipio"] == "11001", "hogares_con_internet"] = 3_000_000

        poverty = pd.DataFrame()
        projections = _make_projections_input()

        features = calculate_features(census, poverty, projections)
        warnings = validate_socioeconomic(features)

        assert any("internet_access_rate" in w and "range" in w for w in warnings), (
            "Expected internet_access_rate out-of-range warning for ratio > 1.0"
        )

    def test_fewer_than_expected_municipalities_warns(self) -> None:
        """Fewer than 1,122 municipalities triggers a warning."""
        df = self._make_valid_features(n=100)
        result = validate_socioeconomic(df)
        assert any("1122" in w or "municipalities" in w.lower() for w in result)


# ═══════════════════════════════════════════════════════════════════
# Build pipeline
# ═══════════════════════════════════════════════════════════════════


class TestBuildSocioeconomicMatrix:
    """``build_socioeconomic_matrix`` writes a validated CSV."""

    def test_creates_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Produces ``socioeconomic.csv`` in the given data directory."""
        census = _make_census_input()
        projections = _make_projections_input()

        monkeypatch.setattr(
            "co_president.ingestion.ingest_socioeconomic._fetch_census_fallback",
            lambda: census,
        )
        monkeypatch.setattr(
            "co_president.ingestion.ingest_socioeconomic._fetch_population_projections",
            lambda: projections,
        )

        build_socioeconomic_matrix(data_dir=tmp_path)
        target = tmp_path / "fundamentals" / "socioeconomic.csv"
        assert target.is_file(), f"Expected {target} to exist"

    def test_saved_file_columns(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The saved CSV contains the expected columns (census-only, no poverty)."""
        census = _make_census_input()
        projections = _make_projections_input()

        monkeypatch.setattr(
            "co_president.ingestion.ingest_socioeconomic._fetch_census_fallback",
            lambda: census,
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
            "population_2022",
        }
        assert expected.issubset(set(saved.columns))
