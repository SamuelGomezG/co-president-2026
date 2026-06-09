"""SPEC-21a: Tests for TerriData fiscal autonomy ingestion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path
import pytest

from co_president.ingestion.ingest_fiscal import (
    _compute_fiscal_features,
    _parse_colombian_number,
    build_fiscal_features,
    load_fiscal_data,
    validate_fiscal,
)

__all__: list[str] = []


# ═══════════════════════════════════════════════════════════════════
# Unit tests: _parse_colombian_number
# ═══════════════════════════════════════════════════════════════════


class TestParseColombianNumber:
    """_parse_colombian_number handles Colombian numeric format."""

    def test_simple_integer_string(self) -> None:
        assert _parse_colombian_number("1234") == 1234.0

    def test_colombian_format(self) -> None:
        """4.977,37 → 4977.37."""
        assert _parse_colombian_number("4.977,37") == 4977.37

    def test_colombian_format_thousands(self) -> None:
        """12.345.678,90 → 12345678.9."""
        assert _parse_colombian_number("12.345.678,90") == 12345678.9

    def test_zero(self) -> None:
        assert _parse_colombian_number("0,00") == 0.0

    def test_none(self) -> None:
        assert _parse_colombian_number(None) is None

    def test_empty_string(self) -> None:
        assert _parse_colombian_number("") is None

    def test_numeric_input(self) -> None:
        assert _parse_colombian_number(4977.37) == 4977.37

    def test_int_input(self) -> None:
        assert _parse_colombian_number(1234) == 1234.0


# ═══════════════════════════════════════════════════════════════════
# Helpers: synthetic data
# ═══════════════════════════════════════════════════════════════════


def _make_synthetic_fiscal_records(n_mpios: int = 5) -> pd.DataFrame:
    """Build a synthetic records DataFrame matching the TerriData schema.

    Returns a long-format DataFrame with columns ``codigo_municipio``,
    ``year``, ``indicator``, and ``value`` — one row per
    municipality-year-indicator combination.

    """
    rng = np.random.default_rng(seed=42)
    rows: list[dict[str, object]] = []
    for i in range(1, n_mpios + 1):
        code = f"{i:05d}"
        for year in range(2018, 2025):
            trib = round(rng.uniform(1_000, 500_000), 2)
            no_trib = round(rng.uniform(100, 50_000), 2)
            corrientes = round(trib + no_trib + rng.uniform(1_000, 100_000), 2)
            for ind, val in [
                ("Ingresos tributarios", trib),
                ("Ingresos no tributarios", no_trib),
                ("Ingresos corrientes", corrientes),
                ("Gastos totales per cápita", round(rng.uniform(0.5, 5.0), 4)),
                (
                    "Transferencias per cápita de los ingresos corrientes",
                    round(rng.uniform(0.2, 3.0), 4),
                ),
                (
                    "Ingresos tributarios per cápita",
                    round(rng.uniform(0.1, 1.5), 4),
                ),
            ]:
                rows.append(
                    {
                        "codigo_municipio": code,
                        "year": year,
                        "indicator": ind,
                        "value": val,
                    }
                )
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# Synthetic-data tests (no real xlsx dependency)
# ═══════════════════════════════════════════════════════════════════


class TestComputeFiscalFeatures:
    """_compute_fiscal_features transforms long-format records to wide."""

    def test_expected_columns(self) -> None:
        """Output contains codigo_municipio and 4 fiscal columns."""
        records = _make_synthetic_fiscal_records(5)
        result = _compute_fiscal_features(records)
        expected = {
            "codigo_municipio",
            "pct_ingresos_propios",
            "gastos_totales_per_capita",
            "transferencias_per_capita",
            "ingresos_tributarios_per_capita",
        }
        assert expected.issubset(set(result.columns))
        assert len(result.columns) == 5

    def test_row_count(self) -> None:
        """One row per municipality."""
        records = _make_synthetic_fiscal_records(5)
        result = _compute_fiscal_features(records)
        assert len(result) == 5

    def test_pct_ingresos_propios_in_unit_interval(self) -> None:
        """Ratio is clipped to [0, 1]."""
        records = _make_synthetic_fiscal_records(10)
        result = _compute_fiscal_features(records)
        assert result["pct_ingresos_propios"].between(0.0, 1.0).all()

    def test_department_codes_excluded(self) -> None:
        """Codes ending in '000' are dropped."""
        records = _make_synthetic_fiscal_records(3)
        dept_row = {
            "codigo_municipio": "05000",
            "year": 2022,
            "indicator": "Ingresos tributarios",
            "value": 1000.0,
        }
        records = pd.concat([records, pd.DataFrame([dept_row])], ignore_index=True)
        # Add all indicators for the department code so it doesn't fail
        for ind in [
            "Ingresos no tributarios",
            "Ingresos corrientes",
            "Gastos totales per cápita",
            "Transferencias per cápita de los ingresos corrientes",
            "Ingresos tributarios per cápita",
        ]:
            records = pd.concat(
                [
                    records,
                    pd.DataFrame(
                        [
                            {
                                "codigo_municipio": "05000",
                                "year": 2022,
                                "indicator": ind,
                                "value": 100.0,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        result = _compute_fiscal_features(records)
        assert "05000" not in result["codigo_municipio"].to_numpy()
        assert len(result) == 3


class TestBuildFiscalFeatures:
    """build_fiscal_features writes the output CSV (synthetic data)."""

    def test_writes_csv_with_columns(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Output CSV has the four fiscal columns plus codigo_municipio."""
        monkeypatch.setattr(
            "co_president.ingestion.ingest_fiscal._read_fiscal_zip",
            lambda _: _make_synthetic_fiscal_records(5),
        )
        build_fiscal_features(data_dir=tmp_path)
        saved = pd.read_csv(
            tmp_path / "fundamentals" / "fiscal.csv",
            dtype={"codigo_municipio": str},
        )
        expected = {
            "codigo_municipio",
            "pct_ingresos_propios",
            "gastos_totales_per_capita",
            "transferencias_per_capita",
            "ingresos_tributarios_per_capita",
        }
        assert expected.issubset(set(saved.columns))
        assert len(saved.columns) == 5

    def test_pct_ingresos_propios_in_unit_interval(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """pct_ingresos_propios is in [0, 1] for all rows."""
        monkeypatch.setattr(
            "co_president.ingestion.ingest_fiscal._read_fiscal_zip",
            lambda _: _make_synthetic_fiscal_records(5),
        )
        build_fiscal_features(data_dir=tmp_path)
        saved = pd.read_csv(
            tmp_path / "fundamentals" / "fiscal.csv",
            dtype={"codigo_municipio": str},
        )
        assert saved["pct_ingresos_propios"].between(0.0, 1.0).all()

    def test_codigo_municipio_is_five_char_string(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """All codigo_municipio values are 5-character zero-padded strings."""
        monkeypatch.setattr(
            "co_president.ingestion.ingest_fiscal._read_fiscal_zip",
            lambda _: _make_synthetic_fiscal_records(5),
        )
        build_fiscal_features(data_dir=tmp_path)
        saved = pd.read_csv(
            tmp_path / "fundamentals" / "fiscal.csv",
            dtype={"codigo_municipio": str},
        )
        assert saved["codigo_municipio"].str.len().eq(5).all()
        assert saved["codigo_municipio"].str.isdigit().all()

    def test_bogota_has_positive_propio(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Bogotá (11001) has positive propio income rate > 0.3."""
        # Create a specific record for 11001 with high tax base
        records = _make_synthetic_fiscal_records(3)
        rows: list[dict[str, object]] = []
        code = "11001"
        for year in range(2018, 2025):
            trib = 500_000.0
            no_trib = 50_000.0
            corrientes = 700_000.0
            for ind, val in [
                ("Ingresos tributarios", trib),
                ("Ingresos no tributarios", no_trib),
                ("Ingresos corrientes", corrientes),
                ("Gastos totales per cápita", 3.5),
                ("Transferencias per cápita de los ingresos corrientes", 0.3),
                ("Ingresos tributarios per cápita", 2.0),
            ]:
                rows.append(
                    {
                        "codigo_municipio": code,
                        "year": year,
                        "indicator": ind,
                        "value": val,
                    }
                )
        bogota_df = pd.DataFrame(rows)
        combined = pd.concat([records, bogota_df], ignore_index=True)

        monkeypatch.setattr(
            "co_president.ingestion.ingest_fiscal._read_fiscal_zip",
            lambda _: combined,
        )
        build_fiscal_features(data_dir=tmp_path)
        saved = pd.read_csv(
            tmp_path / "fundamentals" / "fiscal.csv",
            dtype={"codigo_municipio": str},
        )
        bogota = saved[saved["codigo_municipio"] == "11001"]
        assert not bogota.empty
        assert float(bogota.iloc[0]["pct_ingresos_propios"]) > 0.3


# ═══════════════════════════════════════════════════════════════════
# Load and validate tests (fast, no xlsx)
# ═══════════════════════════════════════════════════════════════════


def _make_valid_fiscal(n: int = 5) -> pd.DataFrame:
    """Build a synthetically valid fiscal DataFrame with ``n`` municipalities."""
    rng = np.random.default_rng(seed=99)
    return pd.DataFrame(
        {
            "codigo_municipio": [f"{i:05d}" for i in range(1, n + 1)],
            "pct_ingresos_propios": [round(rng.uniform(0.1, 0.8), 4) for _ in range(n)],
            "gastos_totales_per_capita": [round(rng.uniform(0.5, 5.0), 4) for _ in range(n)],
            "transferencias_per_capita": [round(rng.uniform(0.2, 3.0), 4) for _ in range(n)],
            "ingresos_tributarios_per_capita": [round(rng.uniform(0.1, 1.5), 4) for _ in range(n)],
        }
    ).astype({"codigo_municipio": str})


class TestLoadFiscalData:
    """load_fiscal_data reads a previously-saved fiscal.csv."""

    def test_round_trip(self, tmp_path: Path) -> None:
        """Round-trip: write then read returns identical data."""
        df = _make_valid_fiscal(5)
        out_dir = tmp_path / "fundamentals"
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "fiscal.csv", index=False)
        loaded = load_fiscal_data(data_dir=tmp_path)
        pd.testing.assert_frame_equal(df, loaded)

    def test_raises_when_missing(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError when fiscal.csv does not exist."""
        with pytest.raises(FileNotFoundError, match="fiscal"):
            load_fiscal_data(data_dir=tmp_path)


class TestValidateFiscal:
    """validate_fiscal checks output against acceptance criteria."""

    def test_valid_data_returns_empty(self) -> None:
        """Valid fiscal DataFrame yields no warnings."""
        df = _make_valid_fiscal(1122)
        warnings = validate_fiscal(df)
        assert warnings == []

    def test_missing_column_warns(self) -> None:
        """Missing pct_ingresos_propios column triggers a warning."""
        df = _make_valid_fiscal(5).drop(columns=["pct_ingresos_propios"])
        warnings = validate_fiscal(df)
        assert any("pct_ingresos_propios" in w for w in warnings)

    def test_null_in_critical_column_warns(self) -> None:
        """Null in pct_ingresos_propios triggers a warning."""
        df = _make_valid_fiscal(5)
        df.loc[0, "pct_ingresos_propios"] = pd.NA
        warnings = validate_fiscal(df)
        assert any("pct_ingresos_propios" in w for w in warnings)

    def test_pct_ingresos_propios_out_of_range_warns(self) -> None:
        """pct_ingresos_propios outside [0, 1] triggers a warning."""
        df = _make_valid_fiscal(5)
        df.loc[0, "pct_ingresos_propios"] = 1.5
        warnings = validate_fiscal(df)
        assert any("pct_ingresos_propios" in w and "range" in w for w in warnings)

    def test_non_five_char_code_warns(self) -> None:
        """codigo_municipio with length != 5 triggers a warning."""
        df = _make_valid_fiscal(5)
        df.loc[0, "codigo_municipio"] = "123"
        warnings = validate_fiscal(df)
        assert any("5" in w for w in warnings)

    def test_too_few_municipalities_warns(self) -> None:
        """Fewer than expected municipalities triggers a warning."""
        df = _make_valid_fiscal(5)
        warnings = validate_fiscal(df)
        assert any("1122" in w for w in warnings)
