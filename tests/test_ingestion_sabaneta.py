"""SPEC-16: Tests for the Sabaneta Cámara de Representantes loader.

Covers parsing the raw CSV, gold fixture path integrity, and expected
data shape (periods, parties, vote totals).
"""

from __future__ import annotations

import shutil
import typing
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from co_president.ingestion.ingest_sabaneta import (
    _EXPECTED_COLUMNS,
    _EXPECTED_PERIODS,
    _parse_sabaneta_file,
    build_sabaneta_camara_matrix,
    load_sabaneta_camara,
    validate_sabaneta,
)

if TYPE_CHECKING:
    from pathlib import Path

# Total votes (incl. blank/null/unmarked) in the 2019/2022 period,
# precomputed from the source CSV to serve as a known-integrity anchor.
_KNOWN_PERIOD_TOTALS: dict[int, int] = {
    2002: 11187,
    2006: 13886,
    2010: 26247,
    2015: 27928,
    2019: 35853,
}


# ═══════════════════════════════════════════════════════════════════════
# Structural contract
# ═══════════════════════════════════════════════════════════════════════


class TestLoadSabanetaCamara:
    """Structural contract for ``load_sabaneta_camara``."""

    def test_is_callable(self) -> None:
        """Function is importable and callable."""
        assert callable(load_sabaneta_camara)

    def test_return_annotation_is_dataframe(self) -> None:
        """Return type annotation is ``pd.DataFrame``."""
        hints = typing.get_type_hints(load_sabaneta_camara)
        assert hints["return"] is pd.DataFrame


class TestParseSabanetaFile:
    """Structural contract for ``_parse_sabaneta_file``."""

    def test_is_callable(self) -> None:
        """Function is importable and callable."""
        assert callable(_parse_sabaneta_file)

    def test_return_annotation_is_dataframe(self) -> None:
        """Return type annotation is ``pd.DataFrame``."""
        hints = typing.get_type_hints(_parse_sabaneta_file)
        assert hints["return"] is pd.DataFrame


# ═══════════════════════════════════════════════════════════════════════
# Parsing behaviour
# ═══════════════════════════════════════════════════════════════════════


class TestParseRawData:
    """``_parse_sabaneta_file`` produces correct long-form output."""

    def test_returns_dataframe(self, sabaneta_fixture: Path) -> None:
        """Returns a DataFrame from the gold fixture directory."""
        result = _parse_sabaneta_file(sabaneta_fixture)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self, sabaneta_fixture: Path) -> None:
        """Output has ``municipio``, ``periodo``, ``partido``, ``total_votes``."""
        result = _parse_sabaneta_file(sabaneta_fixture)
        assert set(result.columns) == _EXPECTED_COLUMNS

    def test_expected_record_count(self, sabaneta_fixture: Path) -> None:
        """Returns 80 records (5 periods x 16 rows)."""
        result = _parse_sabaneta_file(sabaneta_fixture)
        assert len(result) == 80

    def test_periods_present(self, sabaneta_fixture: Path) -> None:
        """Unique periods are exactly {2002, 2006, 2010, 2015, 2019}."""
        result = _parse_sabaneta_file(sabaneta_fixture)
        assert set(result["periodo"].unique()) == _EXPECTED_PERIODS

    def test_each_period_has_16_rows(self, sabaneta_fixture: Path) -> None:
        """Every period has exactly 16 rows."""
        result = _parse_sabaneta_file(sabaneta_fixture)
        counts = result.groupby("periodo").size()
        assert (counts == 16).all(), f"Period row counts: {counts.to_dict()}"

    def test_total_votes_are_integers(self, sabaneta_fixture: Path) -> None:
        """``total_votes`` column contains integer values (comma stripping correct)."""
        result = _parse_sabaneta_file(sabaneta_fixture)
        assert result["total_votes"].dtype == int
        assert (result["total_votes"] >= 0).all()

    def test_municipio_is_constant(self, sabaneta_fixture: Path) -> None:
        """The ``municipio`` column is the same value for all rows."""
        result = _parse_sabaneta_file(sabaneta_fixture)
        assert (result["municipio"] == "Sabaneta").all()

    def test_period_totals_match_known_values(self, sabaneta_fixture: Path) -> None:
        """The sum of ``total_votes`` per period matches precomputed anchors."""
        result = _parse_sabaneta_file(sabaneta_fixture)
        for period_start, expected_sum in _KNOWN_PERIOD_TOTALS.items():
            actual = result.loc[result["periodo"] == period_start, "total_votes"].sum()
            assert actual == expected_sum, (
                f"Period {period_start}: expected total {expected_sum}, got {actual}"
            )

    def test_raises_when_no_csv_present(self, tmp_path: Path) -> None:
        """Raises ``FileNotFoundError`` when the directory has no CSV files."""
        with pytest.raises(FileNotFoundError, match="No CSV file found"):
            _parse_sabaneta_file(tmp_path)


# ═══════════════════════════════════════════════════════════════════════
# Gold fixture path integrity (no data loading, just path existence)
# ═══════════════════════════════════════════════════════════════════════


class TestGoldFixture:
    """Gold fixture directory is present and contains the expected CSV."""

    def test_fixture_directory_exists(self, sabaneta_fixture: Path) -> None:
        """The sabaneta fixture directory exists."""
        assert sabaneta_fixture.is_dir()

    def test_fixture_contains_csv(self, sabaneta_fixture: Path) -> None:
        """The fixture directory contains at least one CSV."""
        csvs = list(sabaneta_fixture.glob("*.csv"))
        assert len(csvs) >= 1

    def test_fixture_csv_loads_via_parser(self, sabaneta_fixture: Path) -> None:
        """The gold CSV can be parsed by ``_parse_sabaneta_file``."""
        df = _parse_sabaneta_file(sabaneta_fixture)
        assert len(df) == 80
        assert df["total_votes"].sum() > 0


# ═══════════════════════════════════════════════════════════════════════
# End-to-end loader
# ═══════════════════════════════════════════════════════════════════════


class TestLoadSabanetaCamaraEndToEnd:
    """``load_sabaneta_camara`` returns a valid DataFrame."""

    def test_returns_dataframe(self) -> None:
        """Returns a DataFrame with the expected structure."""
        df = load_sabaneta_camara()
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == _EXPECTED_COLUMNS
        assert len(df) == 80

    def test_all_votes_positive(self) -> None:
        """All vote totals are non-negative integers."""
        df = load_sabaneta_camara()
        assert df["total_votes"].dtype == int
        assert (df["total_votes"] >= 0).all()

    def test_with_explicit_data_dir(self, sabaneta_fixture: Path) -> None:
        """Returns correct DataFrame when ``data_dir`` is passed explicitly."""
        df = load_sabaneta_camara(data_dir=sabaneta_fixture.parent)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 80
        assert set(df.columns) == _EXPECTED_COLUMNS
        assert set(df["periodo"].unique()) == _EXPECTED_PERIODS


# ═══════════════════════════════════════════════════════════════════════
# Validation and error handling
# ═══════════════════════════════════════════════════════════════════════


class TestValidation:
    """``validate_sabaneta`` checks DataFrame integrity."""

    def test_valid_dataframe_returns_empty_warnings(self) -> None:
        """Returns empty list for a valid DataFrame."""
        df = pd.DataFrame(
            {
                "municipio": ["Sabaneta"] * 5,
                "periodo": [2002, 2006, 2010, 2015, 2019],
                "partido": ["A", "B", "C", "D", "E"],
                "total_votes": [100, 200, 300, 400, 500],
            }
        )
        warnings = validate_sabaneta(df)
        assert warnings == []

    def test_missing_columns_returns_warnings(self) -> None:
        """Returns warnings when expected columns are missing."""
        df = pd.DataFrame({"municipio": ["Sabaneta"]})
        warnings = validate_sabaneta(df)
        assert len(warnings) == 1
        assert "Missing columns" in warnings[0]

    def test_negative_votes_returns_warning(self) -> None:
        """Returns warning when negative vote totals are found."""
        df = pd.DataFrame(
            {
                "municipio": ["Sabaneta", "Sabaneta"],
                "periodo": [2002, 2006],
                "partido": ["Partido A", "Partido B"],
                "total_votes": [-100, 500],
            }
        )
        warnings = validate_sabaneta(df)
        assert any("negative vote totals" in w for w in warnings)

    def test_mismatched_periods_returns_warning(self) -> None:
        """Returns warning when periods don't match expected set."""
        df = pd.DataFrame(
            {
                "municipio": ["Sabaneta"] * 5,
                "periodo": [1999, 2006, 2010, 2015, 2019],
                "partido": ["A", "B", "C", "D", "E"],
                "total_votes": [100, 200, 300, 400, 500],
            }
        )
        warnings = validate_sabaneta(df)
        assert any("Expected periods" in w for w in warnings)


class TestErrorHandling:
    """Error paths for malformed input."""

    def test_raises_on_bad_total(self, tmp_path: Path) -> None:
        """Raises ValueError when Total contains non-numeric values."""
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text(
            '"Partido Político","Periodo","Tipo Candidatura","Total"\n'
            '"Foo","2002/2006","Camara","N/A"\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Non-numeric Total"):
            _parse_sabaneta_file(tmp_path)

    def test_raises_on_bad_periodo(self, tmp_path: Path) -> None:
        """Raises ValueError when Periodo cannot be parsed into a year."""
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text(
            '"Partido Político","Periodo","Tipo Candidatura","Total"\n'
            '"Foo","MALFORMED","Camara","100"\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Failed to extract year"):
            _parse_sabaneta_file(tmp_path)

    def test_raises_on_empty_csv(self, tmp_path: Path) -> None:
        """Raises ValueError when CSV has header but no data rows."""
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text(
            '"Partido Político","Periodo","Tipo Candidatura","Total"\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="empty"):
            _parse_sabaneta_file(tmp_path)


class TestBuildMatrix:
    """``build_sabaneta_camara_matrix`` persists output to disk."""

    def test_saves_to_fundamentals(self, tmp_path: Path, sabaneta_fixture: Path) -> None:
        """Saves CSV to data_dir/fundamentals/sabaneta_camara.csv."""
        sabaneta_dst = tmp_path / "sabaneta"
        shutil.copytree(sabaneta_fixture, sabaneta_dst)

        build_sabaneta_camara_matrix(tmp_path)
        target = tmp_path / "fundamentals" / "sabaneta_camara.csv"
        assert target.exists()
        df = pd.read_csv(target)
        assert set(df.columns) == _EXPECTED_COLUMNS
        assert len(df) == 80
