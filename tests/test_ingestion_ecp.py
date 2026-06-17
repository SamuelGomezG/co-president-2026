"""Tests for SPEC-24 ECP region-level panel ingestion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from co_president.ingestion.ingest_ecp import (
    _min_max_normalise,
    _read_csv_from_zip,
    _warn_ecp_metadata_codes,
    _weighted_groupby,
    build_ecp_features,
    load_ecp_data,
    validate_ecp,
)


class TestWeightedGroupby:
    """Tests for ``_weighted_groupby``."""

    def test_basic_weighted_mean(self) -> None:
        """Weighted mean should be computed correctly."""
        df = pd.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "value": [10.0, 20.0, 30.0, 40.0],
                "weight": [1.0, 4.0, 2.0, 3.0],
            }
        )
        result = _weighted_groupby(df, "group", "value", "weight")
        assert result["A"] == pytest.approx(18.0)
        assert result["B"] == pytest.approx(36.0)

    def test_unequal_weights(self) -> None:
        """All weight in one row should pull the mean to that value."""
        df = pd.DataFrame(
            {
                "group": ["X", "X"],
                "value": [10.0, 100.0],
                "weight": [0.0, 1.0],
            }
        )
        result = _weighted_groupby(df, "group", "value", "weight")
        assert result["X"] == pytest.approx(100.0)


class TestMinMaxNormalise:
    """Tests for ``_min_max_normalise``."""

    def test_basic_normalisation(self) -> None:
        """Column values should be mapped to [0, 1]."""
        df = pd.DataFrame({"x": [10.0, 20.0, 30.0, 40.0]})
        _min_max_normalise(df, "x")
        assert list(df["x"]) == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])

    def test_constant_column(self) -> None:
        """A constant column should become NaN."""
        df = pd.DataFrame({"x": [5.0, 5.0, 5.0]})
        _min_max_normalise(df, "x")
        assert df["x"].isna().all()

    def test_single_row(self) -> None:
        """A single non-NA value should become NaN."""
        df = pd.DataFrame({"x": [42.0]})
        _min_max_normalise(df, "x")
        assert df["x"].isna().all()


class TestValidateEcp:
    """Tests for ``validate_ecp``."""

    def test_valid_dataframe_passes(self) -> None:
        """A well-formed DataFrame should produce no warnings."""
        df = pd.DataFrame(
            {
                "year": [2011, 2023],
                "region_code": [1, 5],
                "trust_index": [0.5, 0.7],
                "participation_index": [0.4, 0.6],
                "engagement_index": [0.3, 0.5],
                "weighted_n": [100.0, 200.0],
            }
        )
        warnings = validate_ecp(df)
        assert len(warnings) == 0, f"Got warnings: {warnings}"

    def test_missing_column_fails(self) -> None:
        """Missing required columns should produce a warning."""
        df = pd.DataFrame({"year": [2011]})
        warnings = validate_ecp(df)
        assert any("Missing columns" in w for w in warnings)

    def test_bad_region_code(self) -> None:
        """Invalid region code should produce a warning."""
        df = pd.DataFrame(
            {
                "year": [2011],
                "region_code": [99],
                "trust_index": [0.5],
                "participation_index": [0.4],
                "engagement_index": [0.3],
                "weighted_n": [100.0],
            }
        )
        warnings = validate_ecp(df)
        assert any("outside" in w and "region_code" in w for w in warnings)

    def test_null_index_columns(self) -> None:
        """Null values in index columns should produce a warning."""
        df = pd.DataFrame(
            {
                "year": [2011],
                "region_code": [1],
                "trust_index": [float("nan")],
                "participation_index": [0.4],
                "engagement_index": [0.3],
                "weighted_n": [100.0],
            }
        )
        warnings = validate_ecp(df)
        assert any("trust_index" in w for w in warnings)

    def test_unknown_year(self) -> None:
        """Unexpected years should produce a warning."""
        df = pd.DataFrame(
            {
                "year": [1999],
                "region_code": [1],
                "trust_index": [0.5],
                "participation_index": [0.4],
                "engagement_index": [0.3],
                "weighted_n": [100.0],
            }
        )
        warnings = validate_ecp(df)
        assert any("1999" in w for w in warnings)


class TestModuleFunctions:
    """Integration-level tests for public functions."""

    def test_load_ecp_data_raises_on_missing(self) -> None:
        """``load_ecp_data`` should raise ``FileNotFoundError``."""
        with pytest.raises(FileNotFoundError):
            load_ecp_data(data_dir=Path("/nonexistent/path"))

    def test_build_ecp_features_raises_on_empty_dir(self) -> None:
        """``build_ecp_features`` should raise when no data is available."""
        with pytest.raises(FileNotFoundError):
            build_ecp_features(data_dir=Path("/nonexistent/path"))


class TestReadCsvFromZip:
    """Tests for ``_read_csv_from_zip`` error handling."""

    def test_corrupt_zip_logs_warning_and_returns_none(
        self,
        caplog: pytest.LogCaptureFixture,
        tmp_path: Path,
    ) -> None:
        """Corrupt zip file should log a warning and return ``None``."""
        corrupt_zip = tmp_path / "corrupt.zip"
        corrupt_zip.write_bytes(b"not a zip file content")

        failed_zips: list[str] = []
        result = _read_csv_from_zip(corrupt_zip, year=2023, failed_zips=failed_zips)

        assert result is None
        assert corrupt_zip.name in failed_zips
        assert any("failed to read" in record.message.lower() for record in caplog.records)


class TestWarnEcpMetadataCodes:
    """Tests for ``_warn_ecp_metadata_codes``."""

    def test_valid_data_no_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """All values ≤6 should produce no warnings."""
        df = pd.DataFrame(
            {
                "Q1": ["1", "2", "3"],
                "Q2": ["4", "5", "6"],
            }
        )
        _warn_ecp_metadata_codes(df, "test_construct", 2023)
        assert len(caplog.records) == 0

    def test_non_numeric_value_logs_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Non-numeric string in question column logs a warning."""
        df = pd.DataFrame(
            {
                "Q1": ["1", "2", "3"],
                "Q2": ["1", "not_a_number", "3"],
            }
        )
        _warn_ecp_metadata_codes(df, "test_construct", 2023)
        assert any("Q2" in r.message for r in caplog.records)
        assert any("non-numeric" in r.message for r in caplog.records)

    def test_value_above_six_logs_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Value >6 (Likert metadata code) logs a warning."""
        df = pd.DataFrame(
            {
                "Q1": ["1", "2", "3"],
                "Q2": ["1", "9", "3"],
            }
        )
        _warn_ecp_metadata_codes(df, "test_construct", 2023)
        assert any("Q2" in r.message for r in caplog.records)
        assert any("metadata" in r.message for r in caplog.records)
