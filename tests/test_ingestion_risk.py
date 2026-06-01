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
    fetch_moe_risk_maps,
    fetch_pdet_list,
    fetch_unodc_coca,
    parse_indepaz_pdf,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

__all__: list[str] = []


def _make_moe_input() -> pd.DataFrame:
    """Synthetic MOE risk classification DataFrame."""
    return pd.DataFrame(
        {
            "codigo_municipio": ["11001", "05001", "76001", "08001", "68001"],
            "risk_level": ["low", "medium", "low", "extreme", "high"],
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


class TestFetchMoeRiskMaps:
    """Structural contract for ``fetch_moe_risk_maps``."""

    def test_is_callable(self) -> None:
        """Function is importable and callable."""
        assert callable(fetch_moe_risk_maps)

    def test_return_annotation_is_dataframe(self) -> None:
        """Return type annotation is ``pd.DataFrame``."""
        hints = typing.get_type_hints(fetch_moe_risk_maps)
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
            _make_moe_input(),
            _make_indepaz_input(),
            _make_pdet_input(),
            _make_coca_input(),
        )
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self) -> None:
        """Output contains all expected risk columns."""
        result = calculate_risk_features(
            _make_moe_input(),
            _make_indepaz_input(),
            _make_pdet_input(),
            _make_coca_input(),
        )
        expected = {
            "codigo_municipio",
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
            _make_moe_input(),
            _make_indepaz_input(),
            _make_pdet_input(),
            _make_coca_input(),
        )
        for col in ("high_risk_flag", "armed_group_presence", "is_pdet"):
            assert result[col].isin([0, 1]).all(), f"{col} is not binary"

    def test_high_risk_flag_matches_extreme_or_high(self) -> None:
        """``high_risk_flag`` is 1 for extreme/high, 0 otherwise."""
        result = calculate_risk_features(
            _make_moe_input(),
            _make_indepaz_input(),
            _make_pdet_input(),
            _make_coca_input(),
        )
        for _, row in result.iterrows():
            expected = 1 if row["risk_level"] in ("extreme", "high") else 0
            assert row["high_risk_flag"] == expected

    def test_coca_non_negative(self) -> None:
        """Coca hectares are non-negative."""
        result = calculate_risk_features(
            _make_moe_input(),
            _make_indepaz_input(),
            _make_pdet_input(),
            _make_coca_input(),
        )
        assert (result["coca_hectares"] >= 0).all()

    def test_missing_indicators_zero_filled(self) -> None:
        """Municipalities not in a non-MOE source get zero-filled."""
        moe_only = pd.DataFrame(
            {
                "codigo_municipio": ["99999"],
                "risk_level": ["low"],
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
        assert result["high_risk_flag"].iloc[0] == 0

    def test_risk_level_valid_values(self) -> None:
        """Risk levels are valid: extreme, high, medium, low."""
        result = calculate_risk_features(
            _make_moe_input(),
            _make_indepaz_input(),
            _make_pdet_input(),
            _make_coca_input(),
        )
        valid = {"extreme", "high", "medium", "low"}
        assert result["risk_level"].isin(valid).all()


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
        """All codes are 5-digit numeric strings."""
        df = _pdet_hardcoded_fallback()
        for code in df["codigo_municipio"]:
            assert len(code) == 5, f"Code {code!r} is not 5 digits"
            assert code.isdigit(), f"Code {code!r} is not numeric"


class TestBuildRiskMatrix:
    """``build_risk_matrix`` writes a validated CSV."""

    def test_creates_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Produces ``risk_factors.csv`` in the given data directory."""
        moe = _make_moe_input()
        indepaz = _make_indepaz_input()
        pdet = _make_pdet_input()
        coca = _make_coca_input()

        monkeypatch.setattr(
            "co_president.ingestion.ingest_risk.fetch_moe_risk_maps",
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
        moe = _make_moe_input()
        indepaz = _make_indepaz_input()
        pdet = _make_pdet_input()
        coca = _make_coca_input()

        monkeypatch.setattr(
            "co_president.ingestion.ingest_risk.fetch_moe_risk_maps",
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
            "risk_level",
            "high_risk_flag",
            "armed_group_presence",
            "is_pdet",
            "coca_hectares",
        }
        assert expected.issubset(set(saved.columns))
