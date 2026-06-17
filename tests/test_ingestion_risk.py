"""SPEC-13.2: Tests for electoral risk and conflict data ingestion."""

from __future__ import annotations

import inspect
import typing
from typing import TYPE_CHECKING

import pandas as pd

from co_president.ingestion.ingest_risk import (
    _pdet_hardcoded_fallback,
    build_risk_matrix,
    calculate_risk_features,
    fetch_pdet_list,
    fetch_unodc_coca,
    load_historical_moe_risk,
    parse_indepaz_pdf,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


# All 33 DANE department codes for Colombia (32 departments + Bogotá D.C.)
# Source: DANE Divipola coding standard.
_KNOWN_DANE_DEPT_CODES: frozenset[str] = frozenset(
    {
        "05",  # Antioquia
        "08",  # Atlántico
        "11",  # Bogotá D.C.
        "13",  # Bolívar
        "15",  # Boyacá
        "17",  # Caldas
        "18",  # Caquetá
        "19",  # Cauca
        "20",  # Cesar
        "23",  # Córdoba
        "25",  # Cundinamarca
        "27",  # Chocó
        "41",  # Huila
        "44",  # La Guajira
        "47",  # Magdalena
        "50",  # Meta
        "52",  # Nariño
        "54",  # Norte de Santander
        "63",  # Quindío
        "66",  # Risaralda
        "68",  # Santander
        "70",  # Sucre
        "73",  # Tolima
        "76",  # Valle del Cauca
        "81",  # Arauca
        "85",  # Casanare
        "86",  # Putumayo
        "88",  # San Andrés y Providencia
        "91",  # Amazonas
        "94",  # Guainía
        "95",  # Guaviare
        "97",  # Vaupés
        "99",  # Vichada
    }
)

__all__: list[str] = []


def _make_moe_input_wide() -> pd.DataFrame:
    """Synthetic wide-format MOE risk classification DataFrame (2 test years)."""
    return pd.DataFrame(
        {
            "codigo_municipio": ["11001", "05001", "76001", "08001", "68001"],
            "moe_risk_2022": ["low", "medium", "low", "extreme", "high"],
            "moe_high_risk_2022": [0, 0, 0, 1, 1],
            "moe_risk_2023": ["low", "medium", "low", "extreme", "low"],
            "moe_high_risk_2023": [0, 0, 0, 1, 0],
        }
    )


def _make_indepaz_input() -> pd.DataFrame:
    """Synthetic INDEPAZ armed group presence DataFrame."""
    return pd.DataFrame(
        {
            "codigo_municipio": ["08001", "68001", "20001"],
            "armed_group_presence": [1, 1, 1],
        }
    )


def _make_pdet_input() -> pd.DataFrame:
    """Synthetic PDET municipality list (3-test subset)."""
    return pd.DataFrame(
        {
            "codigo_municipio": ["08001", "68001", "20001"],
            "is_pdet": [1, 1, 1],
        }
    )


def _make_coca_input() -> pd.DataFrame:
    """Synthetic UNODC coca cultivation DataFrame."""
    return pd.DataFrame(
        {
            "codigo_municipio": ["08001", "11001", "76001"],
            "coca_hectares": [4500, 0, 0],
        }
    )


# ═══════════════════════════════════════════════════════════════════
# Fetch functions (structural tests — no real API calls)
# ═══════════════════════════════════════════════════════════════════


class TestLoadHistoricalMoeRisk:
    """Structural contract for ``load_historical_moe_risk``."""

    def test_is_callable(self) -> None:
        """Function is importable and callable."""
        assert callable(load_historical_moe_risk)

    def test_return_annotation_is_dataframe(self) -> None:
        """Return type annotation is ``pd.DataFrame``."""
        hints = typing.get_type_hints(load_historical_moe_risk)
        assert hints["return"] is pd.DataFrame


class TestParseIndepazPdf:
    """Structural contract for ``parse_indepaz_pdf``."""

    def test_is_callable(self) -> None:
        """Function is importable and callable."""
        assert callable(parse_indepaz_pdf)

    def test_return_annotation_is_dataframe(self) -> None:
        """Return type annotation is ``pd.DataFrame``."""
        hints = typing.get_type_hints(parse_indepaz_pdf)
        assert hints["return"] is pd.DataFrame

    def test_accepts_optional_path(self) -> None:
        """Accepts an optional ``pdf_path`` parameter."""
        sig = inspect.signature(parse_indepaz_pdf)
        assert "pdf_path" in sig.parameters
        default = sig.parameters["pdf_path"].default
        assert default is None


class TestFetchPdetList:
    """Structural contract for ``fetch_pdet_list``."""

    def test_is_callable(self) -> None:
        """Function is importable and callable."""
        assert callable(fetch_pdet_list)

    def test_return_annotation_is_dataframe(self) -> None:
        """Return type annotation is ``pd.DataFrame``."""
        hints = typing.get_type_hints(fetch_pdet_list)
        assert hints["return"] is pd.DataFrame


class TestFetchUnodcCoca:
    """Structural contract for ``fetch_unodc_coca``."""

    def test_is_callable(self) -> None:
        """Function is importable and callable."""
        assert callable(fetch_unodc_coca)

    def test_return_annotation_is_dataframe(self) -> None:
        """Return type annotation is ``pd.DataFrame``."""
        hints = typing.get_type_hints(fetch_unodc_coca)
        assert hints["return"] is pd.DataFrame


# ═══════════════════════════════════════════════════════════════════
# Feature computation
# ═══════════════════════════════════════════════════════════════════


class TestCalculateRiskFeatures:
    """``calculate_risk_features`` combines all risk indicators."""

    def test_returns_dataframe(self) -> None:
        """Returns a DataFrame."""
        result = calculate_risk_features(
            _make_moe_input_wide(),
            _make_indepaz_input(),
            _make_pdet_input(),
            _make_coca_input(),
        )
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self) -> None:
        """Output contains all expected risk columns."""
        result = calculate_risk_features(
            _make_moe_input_wide(),
            _make_indepaz_input(),
            _make_pdet_input(),
            _make_coca_input(),
        )
        expected = {
            "codigo_municipio",
            "moe_risk_2022",
            "moe_high_risk_2022",
            "moe_risk_2023",
            "moe_high_risk_2023",
            "risk_level",
            "high_risk_flag",
            "armed_group_presence",
            "is_pdet",
            "coca_hectares",
        }
        assert expected.issubset(set(result.columns))

    def test_binary_flags_are_zero_or_one(self) -> None:
        """Binary flags are 0 or 1."""
        result = calculate_risk_features(
            _make_moe_input_wide(),
            _make_indepaz_input(),
            _make_pdet_input(),
            _make_coca_input(),
        )
        for col in (
            "moe_high_risk_2022",
            "moe_high_risk_2023",
            "high_risk_flag",
            "armed_group_presence",
            "is_pdet",
        ):
            assert result[col].isin([0, 1]).all(), f"{col} is not binary"

    def test_coca_non_negative(self) -> None:
        """Coca hectares are non-negative."""
        result = calculate_risk_features(
            _make_moe_input_wide(),
            _make_indepaz_input(),
            _make_pdet_input(),
            _make_coca_input(),
        )
        assert (result["coca_hectares"] >= 0).all()

    def test_high_risk_flag_matches_latest_year(self) -> None:
        """``high_risk_flag`` matches the most recent MOE year (2023)."""
        result = calculate_risk_features(
            _make_moe_input_wide(),
            _make_indepaz_input(),
            _make_pdet_input(),
            _make_coca_input(),
        )
        for _, row in result.iterrows():
            expected = 1 if row["moe_risk_2023"] in ("extreme", "high") else 0
            assert row["high_risk_flag"] == expected, (
                f"{row['codigo_municipio']}: risk_2023={row['moe_risk_2023']}, "
                f"expected high_risk={expected}, got {row['high_risk_flag']}"
            )

    def test_risk_level_matches_latest_year(self) -> None:
        """``risk_level`` matches the most recent MOE year (2023)."""
        result = calculate_risk_features(
            _make_moe_input_wide(),
            _make_indepaz_input(),
            _make_pdet_input(),
            _make_coca_input(),
        )
        assert (result["risk_level"] == result["moe_risk_2023"]).all()

    def test_missing_indicators_zero_filled(self) -> None:
        """Municipalities not in a non-MOE source get zero-filled."""
        moe_only = pd.DataFrame(
            {
                "codigo_municipio": ["99999"],
                "moe_risk_2022": ["low"],
                "moe_high_risk_2022": [0],
                "moe_risk_2023": ["low"],
                "moe_high_risk_2023": [0],
            }
        )
        result = calculate_risk_features(
            moe_only,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )
        assert result["armed_group_presence"].iloc[0] == 0
        assert result["is_pdet"].iloc[0] == 0
        assert result["coca_hectares"].iloc[0] == 0
        assert result["risk_level"].iloc[0] == "low"
        assert result["high_risk_flag"].iloc[0] == 0


# ═══════════════════════════════════════════════════════════════════
# Build pipeline
# ═══════════════════════════════════════════════════════════════════


class TestPdetHardcodedFallback:
    """Direct tests for the PDET fallback list."""

    def test_returns_170_municipalities(self) -> None:
        """Fallback returns exactly 170 PDET municipalities."""
        df = _pdet_hardcoded_fallback()
        assert len(df) == 170
        assert set(df.columns) == {"codigo_municipio", "is_pdet"}
        assert (df["is_pdet"] == 1).all()

    def test_no_duplicate_codes(self) -> None:
        """All municipality codes are unique."""
        df = _pdet_hardcoded_fallback()
        assert df["codigo_municipio"].is_unique

    def test_all_codes_are_valid_dane(self) -> None:
        """All codes are 5-digit numeric strings with valid DANE department prefixes."""
        df = _pdet_hardcoded_fallback()
        for code in df["codigo_municipio"]:
            assert len(code) == 5, f"Code {code!r} is not 5 digits"
            assert code.isdigit(), f"Code {code!r} is not numeric"
            prefix = code[:2]
            assert prefix in _KNOWN_DANE_DEPT_CODES, (
                f"Invalid DANE department prefix {prefix!r} in codigo_municipio {code!r}"
            )


class TestBuildRiskMatrix:
    """``build_risk_matrix`` writes a validated CSV."""

    def test_creates_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Produces ``risk_factors.csv`` in the given data directory."""
        moe = _make_moe_input_wide()
        indepaz = _make_indepaz_input()
        pdet = _make_pdet_input()
        coca = _make_coca_input()

        monkeypatch.setattr(
            "co_president.ingestion.ingest_risk.load_historical_moe_risk",
            lambda: moe,
        )
        monkeypatch.setattr(
            "co_president.ingestion.ingest_risk.parse_indepaz_pdf",
            lambda _: indepaz,
        )
        monkeypatch.setattr(
            "co_president.ingestion.ingest_risk.fetch_pdet_list",
            lambda: pdet,
        )
        monkeypatch.setattr(
            "co_president.ingestion.ingest_risk.fetch_unodc_coca",
            lambda: coca,
        )

        build_risk_matrix(data_dir=tmp_path)
        target = tmp_path / "fundamentals" / "risk_factors.csv"
        assert target.is_file(), f"Expected {target} to exist"

    def test_saved_file_columns(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The saved CSV contains the expected columns."""
        moe = _make_moe_input_wide()
        indepaz = _make_indepaz_input()
        pdet = _make_pdet_input()
        coca = _make_coca_input()

        monkeypatch.setattr(
            "co_president.ingestion.ingest_risk.load_historical_moe_risk",
            lambda: moe,
        )
        monkeypatch.setattr(
            "co_president.ingestion.ingest_risk.parse_indepaz_pdf",
            lambda _: indepaz,
        )
        monkeypatch.setattr(
            "co_president.ingestion.ingest_risk.fetch_pdet_list",
            lambda: pdet,
        )
        monkeypatch.setattr(
            "co_president.ingestion.ingest_risk.fetch_unodc_coca",
            lambda: coca,
        )

        build_risk_matrix(data_dir=tmp_path)
        saved = pd.read_csv(
            tmp_path / "fundamentals" / "risk_factors.csv",
            dtype={"codigo_municipio": str},
        )
        expected = {
            "codigo_municipio",
            "moe_risk_2022",
            "moe_high_risk_2022",
            "moe_risk_2023",
            "moe_high_risk_2023",
            "risk_level",
            "high_risk_flag",
            "armed_group_presence",
            "is_pdet",
            "coca_hectares",
        }
        assert expected.issubset(set(saved.columns))
