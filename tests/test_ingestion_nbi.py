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


def _make_synthetic_raw() -> pd.DataFrame:
    """Build a synthetic raw DataFrame mimicking the DANE NBI xlsx structure.

    Columns are ordered 0..24 so that ``raw.iloc[:, i]`` positionally indexes
    the same column as label ``i``, matching the real xlsx behavior.
    """
    from collections import OrderedDict  # noqa: PLC0415

    cols: dict[int, list[str | None]] = OrderedDict()
    for i in range(25):
        if i == 0:
            cols[i] = ["05", "05", "27", "91", "00"]
        elif i == 2:
            cols[i] = ["001", "002", "001", "001", "000"]
        elif i == 4:
            cols[i] = ["3.2", "5.1", "70.3", "8.0", "0.0"]
        elif i == 11:
            cols[i] = ["2.5", "4.0", "65.0", None, "0.0"]
        elif i == 18:
            cols[i] = ["4.0", "6.0", "75.0", "9.0", "0.0"]
        else:
            cols[i] = [""] * 5
    return pd.DataFrame(cols)


# ═══════════════════════════════════════════════════════════════════
# Build tests (uses monkeypatched xlsx reading — no real file needed)
# ═══════════════════════════════════════════════════════════════════


class TestBuildNbiFeatures:
    """Tests for ``build_nbi_features`` using synthetic xlsx data."""

    @pytest.fixture(autouse=True)
    def _mock_nbi_xlsx(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Override ``_read_nbi_xlsx`` to return synthetic data."""
        monkeypatch.setattr(
            "co_president.ingestion.ingest_nbi._read_nbi_xlsx",
            lambda _: _make_synthetic_raw(),
        )

    def test_returns_rows(self, tmp_path: Path) -> None:
        """build_nbi_features writes a CSV with rows."""
        build_nbi_features(data_dir=tmp_path)
        saved = pd.read_csv(
            tmp_path / "fundamentals" / "nbi_2018.csv", dtype={"codigo_municipio": str}
        )
        assert len(saved) > 0

    def test_nbi_rate_in_unit_interval(self, tmp_path: Path) -> None:
        """nbi_rate is in [0, 1] for all rows."""
        build_nbi_features(data_dir=tmp_path)
        saved = pd.read_csv(
            tmp_path / "fundamentals" / "nbi_2018.csv", dtype={"codigo_municipio": str}
        )
        assert saved["nbi_rate"].between(0, 1).all()

    def test_nbi_urban_in_unit_interval(self, tmp_path: Path) -> None:
        """nbi_urban is in [0, 1] for all non-null rows."""
        build_nbi_features(data_dir=tmp_path)
        saved = pd.read_csv(
            tmp_path / "fundamentals" / "nbi_2018.csv", dtype={"codigo_municipio": str}
        )
        urban = saved["nbi_urban"].dropna()
        assert len(urban) > 0
        assert urban.between(0, 1).all()

    def test_nbi_rural_in_unit_interval(self, tmp_path: Path) -> None:
        """nbi_rural is in [0, 1] for all rows."""
        build_nbi_features(data_dir=tmp_path)
        saved = pd.read_csv(
            tmp_path / "fundamentals" / "nbi_2018.csv", dtype={"codigo_municipio": str}
        )
        assert saved["nbi_rural"].between(0, 1).all()

    def test_codigo_municipio_is_five_char_string(self, tmp_path: Path) -> None:
        """All codigo_municipio values are 5-character zero-padded strings."""
        build_nbi_features(data_dir=tmp_path)
        saved = pd.read_csv(
            tmp_path / "fundamentals" / "nbi_2018.csv", dtype={"codigo_municipio": str}
        )
        assert saved["codigo_municipio"].str.len().eq(5).all()
        assert saved["codigo_municipio"].str.isdigit().all()

    def test_higher_nbi_implies_higher_poverty(self, tmp_path: Path) -> None:
        """Dept 27 (Choco-like) has higher NBI than dept 05."""
        build_nbi_features(data_dir=tmp_path)
        saved = pd.read_csv(
            tmp_path / "fundamentals" / "nbi_2018.csv", dtype={"codigo_municipio": str}
        )
        dept_05 = float(saved.loc[saved["codigo_municipio"] == "05001", "nbi_rate"].iloc[0])
        dept_27 = float(saved.loc[saved["codigo_municipio"] == "27001", "nbi_rate"].iloc[0])
        assert dept_27 > dept_05

    def test_total_nacional_row_excluded(self, tmp_path: Path) -> None:
        """The 'TOTAL NACIONAL' summary row is not present."""
        build_nbi_features(data_dir=tmp_path)
        saved = pd.read_csv(
            tmp_path / "fundamentals" / "nbi_2018.csv", dtype={"codigo_municipio": str}
        )
        assert "00000" not in saved["codigo_municipio"].to_numpy()

    def test_anm_rows_have_nan_urban(self, tmp_path: Path) -> None:
        """Areas No Municipalizadas have NaN nbi_urban (no urban cabecera)."""
        build_nbi_features(data_dir=tmp_path)
        saved = pd.read_csv(
            tmp_path / "fundamentals" / "nbi_2018.csv", dtype={"codigo_municipio": str}
        )
        anm_row = saved[saved["codigo_municipio"] == "91001"]
        assert len(anm_row) == 1
        assert pd.isna(anm_row["nbi_urban"].iloc[0])

    def test_all_expected_columns_present(self, tmp_path: Path) -> None:
        """Output CSV has the three NBI columns plus codigo_municipio."""
        build_nbi_features(data_dir=tmp_path)
        saved = pd.read_csv(
            tmp_path / "fundamentals" / "nbi_2018.csv", dtype={"codigo_municipio": str}
        )
        expected = {"codigo_municipio", "nbi_rate", "nbi_urban", "nbi_rural"}
        assert expected.issubset(set(saved.columns))
        assert len(saved.columns) == 4

    def test_deterministic_column_order(self, tmp_path: Path) -> None:
        """codigo_municipio is the first column."""
        build_nbi_features(data_dir=tmp_path)
        saved = pd.read_csv(
            tmp_path / "fundamentals" / "nbi_2018.csv", dtype={"codigo_municipio": str}
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
