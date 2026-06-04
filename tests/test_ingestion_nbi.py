"""SPEC-18: Tests for NBI municipal poverty indicators."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
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
# Synthetic-data tests (no real xlsx dependency)
# ═══════════════════════════════════════════════════════════════════


def _make_synthetic_nbi_raw(n_mpios: int = 5) -> pd.DataFrame:
    """Build a synthetic raw DataFrame matching the NBI xlsx schema.

    ``_read_nbi_xlsx`` returns a headerless DataFrame with 0-indexed
    columns.  ``_clean_nbi_data`` uses columns 0 (dept code), 2 (mpio
    code), 4 (NBI total %), 11 (NBI urban %), and 18 (NBI rural %).
    """
    rows: list[dict[int, object]] = []
    for i in range(1, n_mpios + 1):
        dept = str(i).zfill(2)
        mpio = str(i).zfill(3)
        row: dict[int, object] = {
            0: dept,
            1: f"Dept {i}",
            2: mpio,
            3: f"Municipio {i}",
            4: str(round(np.random.default_rng(seed=i).uniform(5, 80), 2)),
            11: str(round(np.random.default_rng(seed=i + 1000).uniform(3, 75), 2)),
            18: str(round(np.random.default_rng(seed=i + 2000).uniform(8, 85), 2)),
        }
        rows.append(row)
    return pd.DataFrame(rows).astype(str)


class TestBuildNbiFeatures:
    """build_nbi_features writes the output CSV (synthetic xlsx)."""

    def test_writes_csv_with_columns(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Output CSV has the three NBI columns plus codigo_municipio."""
        monkeypatch.setattr(
            "co_president.ingestion.ingest_nbi._read_nbi_xlsx",
            lambda _: _make_synthetic_nbi_raw(5),
        )
        build_nbi_features(data_dir=tmp_path)
        saved = pd.read_csv(
            tmp_path / "fundamentals" / "nbi_2018.csv",
            dtype={"codigo_municipio": str},
        )
        expected = {"codigo_municipio", "nbi_rate", "nbi_urban", "nbi_rural"}
        assert expected.issubset(set(saved.columns))
        assert len(saved.columns) == 4
        assert len(saved) == 5

    def test_nbi_rate_in_unit_interval(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """nbi_rate is in [0, 1] for all rows."""
        monkeypatch.setattr(
            "co_president.ingestion.ingest_nbi._read_nbi_xlsx",
            lambda _: _make_synthetic_nbi_raw(5),
        )
        build_nbi_features(data_dir=tmp_path)
        saved = pd.read_csv(
            tmp_path / "fundamentals" / "nbi_2018.csv",
            dtype={"codigo_municipio": str},
        )
        assert saved["nbi_rate"].between(0, 1).all()
        assert saved["nbi_urban"].between(0, 1).all()
        assert saved["nbi_rural"].between(0, 1).all()

    def test_codigo_municipio_is_five_char_string(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """All codigo_municipio values are 5-character zero-padded strings."""
        monkeypatch.setattr(
            "co_president.ingestion.ingest_nbi._read_nbi_xlsx",
            lambda _: _make_synthetic_nbi_raw(5),
        )
        build_nbi_features(data_dir=tmp_path)
        saved = pd.read_csv(
            tmp_path / "fundamentals" / "nbi_2018.csv",
            dtype={"codigo_municipio": str},
        )
        assert saved["codigo_municipio"].str.len().eq(5).all()
        assert saved["codigo_municipio"].str.isdigit().all()

    def test_total_nacional_row_excluded(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The '00000' summary code is not present in output."""
        raw = _make_synthetic_nbi_raw(5)
        raw.loc[0, 0] = "00"
        raw.loc[0, 2] = "000"
        monkeypatch.setattr(
            "co_president.ingestion.ingest_nbi._read_nbi_xlsx",
            lambda _: raw,
        )
        build_nbi_features(data_dir=tmp_path)
        saved = pd.read_csv(
            tmp_path / "fundamentals" / "nbi_2018.csv",
            dtype={"codigo_municipio": str},
        )
        assert "00000" not in saved["codigo_municipio"].to_numpy()
        assert len(saved) == 4  # one 00000 row was dropped from 5

    def test_deterministic_column_order(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """codigo_municipio is the first column."""
        monkeypatch.setattr(
            "co_president.ingestion.ingest_nbi._read_nbi_xlsx",
            lambda _: _make_synthetic_nbi_raw(5),
        )
        build_nbi_features(data_dir=tmp_path)
        saved = pd.read_csv(
            tmp_path / "fundamentals" / "nbi_2018.csv",
            dtype={"codigo_municipio": str},
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
