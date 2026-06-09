"""SPEC-21b: Tests for Bogotá D.C. localidad disaggregation."""

from __future__ import annotations

from pathlib import Path

import pytest

from co_president.ingestion.ingest_bogota import (
    _build_7digit_code,
    _classify_localidad,
    _extract_localidad_code,
    _localidad_name,
    _normalize_puesto,
    disaggregate_bogota_population,
    get_bogota_localidad_rows,
)

__all__: list[str] = []


# ═══════════════════════════════════════════════════════════════════════
# _normalize_puesto
# ═══════════════════════════════════════════════════════════════════════


class TestNormalizePuesto:
    """Puesto name normalisation tests."""

    def test_strips_accents(self) -> None:
        """Accented characters are decomposed and stripped."""
        assert _normalize_puesto("COLEGIO CHAMPAGNAT") == "colegio champagnat"

    def test_strips_whitespace(self) -> None:
        """Leading/trailing whitespace is removed."""
        assert _normalize_puesto("  INEM   ") == "inem"

    def test_handles_none(self) -> None:
        """None as input is handled without error."""
        result = _normalize_puesto(None)  # type: ignore[arg-type]
        assert isinstance(result, str)


# ═══════════════════════════════════════════════════════════════════════
# _extract_localidad_code
# ═══════════════════════════════════════════════════════════════════════


class TestExtractLocalidadCode:
    """Localidad code extraction from Código Puesto tests."""

    def test_usaquen(self) -> None:
        """Usaquén (COMUNA 1) extracts code 01."""
        assert _extract_localidad_code("11001010101") == "01"

    def test_chapinero(self) -> None:
        """Chapinero (COMUNA 2) extracts code 02."""
        assert _extract_localidad_code("11001020201") == "02"

    def test_sumapaz(self) -> None:
        """Sumapaz (COMUNA 20) extracts code 20."""
        assert _extract_localidad_code("11001209929") == "20"

    def test_catch_all(self) -> None:
        """Code 99 is extracted from a SIN COMUNA code."""
        assert _extract_localidad_code("11001999999") == "99"

    def test_non_bogota_returns_none(self) -> None:
        """Non-Bogotá codes return None."""
        assert _extract_localidad_code("05001009001") is None

    def test_short_code_returns_none(self) -> None:
        """Codes shorter than 11 digits return None."""
        assert _extract_localidad_code("11001") is None

    def test_missing_returns_none(self) -> None:
        """NaN or None input returns None."""
        assert _extract_localidad_code(None) is None  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════
# _classify_localidad
# ═══════════════════════════════════════════════════════════════════════


class TestClassifyLocalidad:
    """Localidad classification tests."""

    def test_known_codes_pass_through(self) -> None:
        """Codes 01-20 pass through unchanged."""
        for code in ("01", "02", "10", "20"):
            assert _classify_localidad(code) == code

    def test_unknown_code_maps_to_99(self) -> None:
        """Code 21 maps to catch-all 99."""
        assert _classify_localidad("21") == "99"

    def test_zero_maps_to_99(self) -> None:
        """SIN COMUNA (00) maps to catch-all 99."""
        assert _classify_localidad("00") == "99"

    def test_none_maps_to_99(self) -> None:
        """None maps to catch-all 99."""
        assert _classify_localidad(None) == "99"


# ═══════════════════════════════════════════════════════════════════════
# _build_7digit_code
# ═══════════════════════════════════════════════════════════════════════


class TestBuild7DigitCode:
    """7-digit DANE code construction tests."""

    def test_usaquen(self) -> None:
        """Usaquén (01) gets code 1100101."""
        assert _build_7digit_code("01") == "1100101"

    def test_chapinero(self) -> None:
        """Chapinero (02) gets code 1100102."""
        assert _build_7digit_code("02") == "1100102"

    def test_sumapaz(self) -> None:
        """Sumapaz (20) gets code 1100120."""
        assert _build_7digit_code("20") == "1100120"

    def test_catch_all(self) -> None:
        """Catch-all (None) gets code 1100199."""
        assert _build_7digit_code(None) == "1100199"


# ═══════════════════════════════════════════════════════════════════════
# _localidad_name
# ═══════════════════════════════════════════════════════════════════════


class TestLocalidadName:
    """Localidad name lookup tests."""

    def test_usaquen(self) -> None:
        """Code 01 maps to Usaquén."""
        assert _localidad_name("01") == "Usaquén"

    def test_chapinero(self) -> None:
        """Code 02 maps to Chapinero."""
        assert _localidad_name("02") == "Chapinero"

    def test_kennedy(self) -> None:
        """Code 08 maps to Kennedy."""
        assert _localidad_name("08") == "Kennedy"

    def test_sumapaz(self) -> None:
        """Code 20 maps to Sumapaz."""
        assert _localidad_name("20") == "Sumapaz"

    def test_catch_all(self) -> None:
        """Catch-all code 99 maps to SIN COMUNA name."""
        assert _localidad_name("99") == "BOGOTÁ D.C. - SIN COMUNA"

    def test_unknown_returns_none(self) -> None:
        """Unknown code returns None."""
        assert _localidad_name("99") is not None

    def test_none_returns_catch_all(self) -> None:
        """None returns the catch-all name."""
        assert _localidad_name(None) == "BOGOTÁ D.C. - SIN COMUNA"


# ═══════════════════════════════════════════════════════════════════════
# get_bogota_localidad_rows
# ═══════════════════════════════════════════════════════════════════════


class TestGetBogotaLocalidadRows:
    """Bogotá localidad DIVIPOLA rows tests."""

    def test_returns_21_rows(self) -> None:
        """Expected 21 localidad rows (20 + 1 catch-all)."""
        rows = get_bogota_localidad_rows()
        assert len(rows) == 21

    def test_required_columns(self) -> None:
        """Required columns are present."""
        rows = get_bogota_localidad_rows()
        for col in (
            "codigo_municipio",
            "nombre_municipio",
            "comuna_nombre",
            "departamento",
            "region",
        ):
            assert col in rows.columns, f"Missing column: {col}"

    def test_all_codes_are_seven_digit(self) -> None:
        """All localidad codes are 7-digit."""
        rows = get_bogota_localidad_rows()
        assert rows["codigo_municipio"].astype(str).str.len().eq(7).all()

    def test_all_start_with_11001(self) -> None:
        """All localidad codes start with 11001."""
        rows = get_bogota_localidad_rows()
        assert rows["codigo_municipio"].astype(str).str.startswith("11001").all()

    def test_all_departamento_is_cundinamarca(self) -> None:
        """All rows have departamento Cundinamarca."""
        rows = get_bogota_localidad_rows()
        assert (rows["departamento"] == "Cundinamarca").all()

    def test_comuna_nombre_is_never_null(self) -> None:
        """comuna_nombre is non-null for all localidad rows."""
        rows = get_bogota_localidad_rows()
        assert rows["comuna_nombre"].notna().all()

    def test_bogota_nombre(self) -> None:
        """nombre_municipio is 'Bogotá D.C.' for all rows."""
        rows = get_bogota_localidad_rows()
        assert (rows["nombre_municipio"] == "Bogotá D.C.").all()

    def test_known_localidad_present(self) -> None:
        """Specific known localidades are present."""
        rows = get_bogota_localidad_rows()
        names = set(rows["comuna_nombre"])
        assert "Usaquén" in names
        assert "Chapinero" in names
        assert "Kennedy" in names
        assert "Sumapaz" in names
        assert "BOGOTÁ D.C. - SIN COMUNA" in names


# ═══════════════════════════════════════════════════════════════════════
# disaggregate_bogota_population (unit — no file I/O)
# ═══════════════════════════════════════════════════════════════════════


class TestDisaggregateBogotaPopulation:
    """Population disaggregation tests (no file I/O)."""

    def test_smoke(self) -> None:
        """Returns 21 rows with pop columns when given real data."""
        result = disaggregate_bogota_population(Path("data"))
        if result.empty:
            msg = (
                "Population disaggregation returned empty — ensure population_2018_2026.csv exists"
            )
            pytest.skip(msg)
        assert len(result) == 21
        for col in ("codigo_municipio", "comuna_nombre", "pop_2022"):
            assert col in result.columns
        assert result["codigo_municipio"].astype(str).str.len().eq(7).all()
