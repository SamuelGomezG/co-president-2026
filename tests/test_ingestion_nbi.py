"""SPEC-18: Tests for NBI municipal poverty indicators."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path
import pytest

from co_president.ingestion.ingest_nbi import (
    build_nbi_features,
    load_nbi_data,
    validate_nbi,
)

__all__: list[str] = []


# ═══════════════════════════════════════════════════════════════════
# Real-data tests (uses the 443 KB DANE xlsx on disk)
# ═══════════════════════════════════════════════════════════════════


def _nbi_xlsx_path(data_dir: Path) -> Path:
    """Return the path to the DANE NBI Excel file."""
    return data_dir / "raw" / "DANE-NBI" / "CNPV-2018-NBI.xlsx"


class TestBuildNbiFeaturesRealData:
    """Integration-style tests using the real DANE NBI xlsx."""

    def test_returns_at_least_1100_rows(self, data_dir: Path) -> None:
        """build_nbi_features writes a CSV with >= 1100 rows."""
        build_nbi_features(data_dir=data_dir)
        saved = pd.read_csv(
            data_dir / "fundamentals" / "nbi_2018.csv", dtype={"codigo_municipio": str}
        )
        assert len(saved) >= 1100

    def test_nbi_rate_in_unit_interval(self, data_dir: Path) -> None:
        """nbi_rate is in [0, 1] for all rows."""
        build_nbi_features(data_dir=data_dir)
        saved = pd.read_csv(
            data_dir / "fundamentals" / "nbi_2018.csv", dtype={"codigo_municipio": str}
        )
        assert saved["nbi_rate"].between(0, 1).all()

    def test_nbi_urban_in_unit_interval(self, data_dir: Path) -> None:
        """nbi_urban is in [0, 1] for all non-null rows."""
        build_nbi_features(data_dir=data_dir)
        saved = pd.read_csv(
            data_dir / "fundamentals" / "nbi_2018.csv", dtype={"codigo_municipio": str}
        )
        urban = saved["nbi_urban"].dropna()
        assert len(urban) > 0
        assert urban.between(0, 1).all()

    def test_nbi_rural_in_unit_interval(self, data_dir: Path) -> None:
        """nbi_rural is in [0, 1] for all rows."""
        build_nbi_features(data_dir=data_dir)
        saved = pd.read_csv(
            data_dir / "fundamentals" / "nbi_2018.csv", dtype={"codigo_municipio": str}
        )
        assert saved["nbi_rural"].between(0, 1).all()

    def test_codigo_municipio_is_five_char_string(self, data_dir: Path) -> None:
        """All codigo_municipio values are 5-character zero-padded strings."""
        build_nbi_features(data_dir=data_dir)
        saved = pd.read_csv(
            data_dir / "fundamentals" / "nbi_2018.csv", dtype={"codigo_municipio": str}
        )
        assert saved["codigo_municipio"].str.len().eq(5).all()
        assert saved["codigo_municipio"].str.isdigit().all()

    def test_bogota_nbi_lower_than_choco(self, data_dir: Path) -> None:
        """Bogota NBI (~3%) is far lower than Choco (~70%)."""
        build_nbi_features(data_dir=data_dir)
        saved = pd.read_csv(
            data_dir / "fundamentals" / "nbi_2018.csv", dtype={"codigo_municipio": str}
        )
        bogota = float(saved.loc[saved["codigo_municipio"] == "11001", "nbi_rate"].iloc[0])
        quibdo = float(saved.loc[saved["codigo_municipio"] == "27001", "nbi_rate"].iloc[0])
        assert bogota < 0.10
        assert quibdo > 0.50
        assert bogota < quibdo

    def test_total_nacional_row_excluded(self, data_dir: Path) -> None:
        """The 'TOTAL NACIONAL' summary row is not present."""
        build_nbi_features(data_dir=data_dir)
        saved = pd.read_csv(
            data_dir / "fundamentals" / "nbi_2018.csv", dtype={"codigo_municipio": str}
        )
        assert "00000" not in saved["codigo_municipio"].to_numpy()

    def test_anm_rows_have_nan_urban(self, data_dir: Path) -> None:
        """Areas No Municipalizadas have NaN nbi_urban (no urban cabecera)."""
        build_nbi_features(data_dir=data_dir)
        saved = pd.read_csv(
            data_dir / "fundamentals" / "nbi_2018.csv", dtype={"codigo_municipio": str}
        )
        anm_count = int(saved["nbi_urban"].isna().sum())
        assert 15 <= anm_count <= 25, f"Expected ~20 ANM rows with NaN urban, got {anm_count}"

    def test_all_expected_columns_present(self, data_dir: Path) -> None:
        """Output CSV has the three NBI columns plus codigo_municipio."""
        build_nbi_features(data_dir=data_dir)
        saved = pd.read_csv(
            data_dir / "fundamentals" / "nbi_2018.csv", dtype={"codigo_municipio": str}
        )
        expected = {"codigo_municipio", "nbi_rate", "nbi_urban", "nbi_rural"}
        assert expected.issubset(set(saved.columns))
        assert len(saved.columns) == 4

    def test_deterministic_column_order(self, data_dir: Path) -> None:
        """codigo_municipio is the first column."""
        build_nbi_features(data_dir=data_dir)
        saved = pd.read_csv(
            data_dir / "fundamentals" / "nbi_2018.csv", dtype={"codigo_municipio": str}
        )
        assert next(iter(saved.columns)) == "codigo_municipio"


# ═══════════════════════════════════════════════════════════════════
# Load and validate tests (fast, no xlsx)
# ═══════════════════════════════════════════════════════════════════


def _make_valid_nbi(n: int = 5) -> pd.DataFrame:
    """Build a synthetically valid NBI DataFrame with ``n`` municipalities."""
    return pd.DataFrame(
        {
            "codigo_municipio": [f"{i:05d}" for i in range(1, n + 1)],
            "nbi_rate": [0.05 * (i % 3 + 1) for i in range(n)],
            "nbi_urban": [0.04 * (i % 3 + 1) for i in range(n)],
            "nbi_rural": [0.10 * (i % 3 + 1) for i in range(n)],
        }
    ).astype({"codigo_municipio": str})


class TestLoadNbiData:
    """load_nbi_data reads a previously-saved nbi_2018.csv."""

    def test_round_trip(self, tmp_path: Path) -> None:
        """Round-trip: write then read returns identical data."""
        df = _make_valid_nbi(5)
        out_dir = tmp_path / "fundamentals"
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "nbi_2018.csv", index=False)
        loaded = load_nbi_data(data_dir=tmp_path)
        pd.testing.assert_frame_equal(df, loaded)

    def test_raises_when_missing(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError when nbi_2018.csv does not exist."""
        with pytest.raises(FileNotFoundError, match="nbi_2018"):
            load_nbi_data(data_dir=tmp_path)


class TestValidateNbi:
    """validate_nbi checks output against acceptance criteria."""

    def test_valid_data_returns_empty(self) -> None:
        """Valid NBI DataFrame yields no warnings."""
        df = _make_valid_nbi(1122)
        warnings = validate_nbi(df)
        assert warnings == []

    def test_missing_column_warns(self) -> None:
        """Missing nbi_rate column triggers a warning."""
        df = _make_valid_nbi(5).drop(columns=["nbi_rate"])
        warnings = validate_nbi(df)
        assert any("nbi_rate" in w for w in warnings)

    def test_null_in_critical_column_warns(self) -> None:
        """Null in nbi_rate triggers a warning."""
        df = _make_valid_nbi(5)
        df.loc[0, "nbi_rate"] = pd.NA
        warnings = validate_nbi(df)
        assert any("nbi_rate" in w for w in warnings)

    def test_nbi_rate_out_of_range_warns(self) -> None:
        """nbi_rate outside [0, 1] triggers a warning."""
        df = _make_valid_nbi(5)
        df.loc[0, "nbi_rate"] = 1.5
        warnings = validate_nbi(df)
        assert any("nbi_rate" in w and "range" in w for w in warnings)

    def test_non_five_char_code_warns(self) -> None:
        """codigo_municipio with length != 5 triggers a warning."""
        df = _make_valid_nbi(5)
        df.loc[0, "codigo_municipio"] = "123"
        warnings = validate_nbi(df)
        assert any("5" in w for w in warnings)