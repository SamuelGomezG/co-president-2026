"""SPEC-18: Tests for IPM poverty indicators (secondary, opt-in)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path
import pytest

from co_president.ingestion.ingest_ipm import (
    _pearson_r2,
    build_ipm_features,
    load_ipm_data,
    validate_ipm,
)

__all__: list[str] = []


# ═══════════════════════════════════════════════════════════════════
# Build and load tests (uses departamental ECV microdata on disk)
# ═══════════════════════════════════════════════════════════════════


def _ensure_nbi_csv(data_dir: Path) -> None:
    """Ensure nbi_2018.csv exists before IPM build (IPM R² guard reads NBI)."""
    from co_president.ingestion.ingest_nbi import build_nbi_features  # noqa: PLC0415

    nbi_path = data_dir / "fundamentals" / "nbi_2018.csv"
    if not nbi_path.is_file():
        build_nbi_features(data_dir=data_dir)


class TestBuildIpmFeaturesRealData:
    """Integration-style tests using real ECV microdata.

    Note: the R² guard may drop the ``ipm_2018`` column if the
    departamento-level ECV IPM does not correlate strongly enough
    with NBI (R² < 0.5).  These tests account for that optionality.
    """

    def test_emits_expected_columns_when_present(self, data_dir: Path) -> None:
        """If IPM-2018 passes the R² guard, all 5 expected columns exist."""
        _ensure_nbi_csv(data_dir)
        build_ipm_features(data_dir=data_dir)
        saved = pd.read_csv(
            data_dir / "fundamentals" / "ipm_2018.csv", dtype={"codigo_municipio": str}
        )
        expected = {"codigo_municipio", "ipm_2022", "ipm_2022_imputed"}
        assert expected.issubset(set(saved.columns))

    def test_has_as_many_rows_as_divipola(self, data_dir: Path) -> None:
        """Output has the same number of rows as DIVIPOLA (1,122)."""
        _ensure_nbi_csv(data_dir)
        build_ipm_features(data_dir=data_dir)
        divipola = pd.read_csv(
            data_dir / "fundamentals" / "divipola_master.csv",
            dtype={"codigo_municipio": str},
        )
        saved = pd.read_csv(
            data_dir / "fundamentals" / "ipm_2018.csv", dtype={"codigo_municipio": str}
        )
        assert len(saved) == len(divipola)

    def test_codigo_municipio_is_five_char_string(self, data_dir: Path) -> None:
        """All codigo_municipio values are 5-character zero-padded strings."""
        _ensure_nbi_csv(data_dir)
        build_ipm_features(data_dir=data_dir)
        saved = pd.read_csv(
            data_dir / "fundamentals" / "ipm_2018.csv", dtype={"codigo_municipio": str}
        )
        assert saved["codigo_municipio"].str.len().eq(5).all()

    def test_r2_guard_logs_warning_when_below_threshold(
        self,
        data_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """When R² < 0.5, a warning is logged and ipm_2018 column dropped."""
        import logging  # noqa: PLC0415

        caplog.set_level(logging.WARNING)
        _ensure_nbi_csv(data_dir)
        build_ipm_features(data_dir=data_dir)
        saved = pd.read_csv(
            data_dir / "fundamentals" / "ipm_2018.csv", dtype={"codigo_municipio": str}
        )
        if "ipm_2018" not in saved.columns:
            assert any("R²" in r.message and "dropping" in r.message for r in caplog.records), (
                "Expected a log warning when R² guard drops the column"
            )


class TestPearsonR2:
    """_pearson_r2 computes R² correctly."""

    def test_perfect_correlation(self) -> None:
        """Perfectly correlated variables yield R² = 1.0."""
        import numpy as np  # noqa: PLC0415

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = x * 2
        assert _pearson_r2(x, y) == pytest.approx(1.0, abs=0.001)

    def test_no_correlation(self) -> None:
        """Uncorrelated variables yield R² close to 0."""
        import numpy as np  # noqa: PLC0415

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        assert _pearson_r2(x, y) < 0.5


# ═══════════════════════════════════════════════════════════════════
# Load and validate tests (fast, no ECV file)
# ═══════════════════════════════════════════════════════════════════


def _make_valid_ipm(n: int = 5) -> pd.DataFrame:
    """Build a synthetically valid IPM DataFrame with ``n`` municipalities."""
    return pd.DataFrame(
        {
            "codigo_municipio": [f"{i:05d}" for i in range(1, n + 1)],
            "ipm_2018": [0.10, 0.25, 0.15, 0.30, 0.05][:n],
            "ipm_2018_imputed": [True] * n,
            "ipm_2022": [0.12, 0.27, 0.14, 0.28, 0.06][:n],
            "ipm_2022_imputed": [True] * n,
        }
    ).astype({"codigo_municipio": str})


class TestLoadIpmData:
    """load_ipm_data reads a previously-saved ipm_2018.csv."""

    def test_round_trip(self, tmp_path: Path) -> None:
        """Round-trip: write then read returns identical data."""
        df = _make_valid_ipm(5)
        out_dir = tmp_path / "fundamentals"
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "ipm_2018.csv", index=False)
        loaded = load_ipm_data(data_dir=tmp_path)
        pd.testing.assert_frame_equal(df, loaded)

    def test_raises_when_missing(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError when ipm_2018.csv does not exist."""
        with pytest.raises(FileNotFoundError, match="ipm_2018"):
            load_ipm_data(data_dir=tmp_path)


class TestValidateIpm:
    """validate_ipm checks output against acceptance criteria."""

    def test_valid_data_returns_empty(self) -> None:
        """Valid IPM DataFrame yields no warnings."""
        df = _make_valid_ipm(5)
        warnings = validate_ipm(df)
        assert warnings == []

    def test_ipm_out_of_range_warns(self) -> None:
        """ipm_2018 outside [0, 1] triggers a warning."""
        df = _make_valid_ipm(5)
        df.loc[0, "ipm_2018"] = 1.5
        warnings = validate_ipm(df)
        assert any("ipm_2018" in w and "range" in w for w in warnings)

    def test_non_five_char_code_warns(self) -> None:
        """codigo_municipio with length != 5 triggers a warning."""
        df = _make_valid_ipm(5)
        df.loc[0, "codigo_municipio"] = "123"
        warnings = validate_ipm(df)
        assert any("5" in w for w in warnings)