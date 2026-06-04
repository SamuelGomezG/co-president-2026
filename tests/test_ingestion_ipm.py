"""SPEC-18: Tests for IPM poverty indicators (secondary, opt-in)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
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
# Build tests (synthetic ECV and NBI data, no real files)
# ═══════════════════════════════════════════════════════════════════


def _make_synthetic_ecv_hogares(n_departamentos: int = 5) -> pd.DataFrame:
    """Build a synthetic ECV hogares DataFrame matching the reader schema."""
    rng = np.random.default_rng(seed=42)
    rows: list[dict[str, object]] = []
    for i in range(1, n_departamentos + 1):
        rows.extend(
            {
                "COD_DEPTO": str(i).zfill(2),
                "FACTOR_EXP": float(rng.uniform(100, 1000)),
                "INGRESO": float(rng.uniform(500, 5000)),
                "POBREZA": int(rng.choice([0, 1], p=[0.7, 0.3])),
            }
            for _ in range(3)
        )
    return pd.DataFrame(rows)


def _prepare_nbi_csv(tmp_path: Path, n: int = 5) -> Path:
    """Write a synthetic NBI CSV so the IPM R² guard has data to read."""
    fundamentals = tmp_path / "fundamentals"
    fundamentals.mkdir(parents=True, exist_ok=True)
    nbi = pd.DataFrame(
        {
            "codigo_municipio": [f"{i:05d}" for i in range(1, n + 1)],
            "nbi_rate": [
                round(x, 4) for x in np.random.default_rng(seed=42).uniform(0.05, 0.80, n)
            ],
            "nbi_urban": [
                round(x, 4) for x in np.random.default_rng(seed=142).uniform(0.03, 0.75, n)
            ],
            "nbi_rural": [
                round(x, 4) for x in np.random.default_rng(seed=242).uniform(0.08, 0.85, n)
            ],
        }
    )
    path = fundamentals / "nbi_2018.csv"
    nbi.to_csv(path, index=False)
    return path


class TestBuildIpmFeatures:
    """build_ipm_features writes the output CSV (synthetic ECV data)."""

    def test_emits_expected_columns(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Output has codigo_municipio, ipm_2022, ipm_2022_imputed."""
        _prepare_nbi_csv(tmp_path, 5)
        monkeypatch.setattr(
            "co_president.ingestion.ingest_ipm._read_ecv_hogares",
            lambda _base, _year: _make_synthetic_ecv_hogares(5),
        )
        monkeypatch.setattr(
            "co_president.ingestion.ingest_ipm._load_divipola",
            lambda _base: pd.DataFrame({"codigo_municipio": [f"{i:05d}" for i in range(1, 6)]}),
        )
        build_ipm_features(data_dir=tmp_path)
        saved = pd.read_csv(
            tmp_path / "fundamentals" / "ipm_2018.csv",
            dtype={"codigo_municipio": str},
        )
        expected = {"codigo_municipio", "ipm_2022", "ipm_2022_imputed"}
        assert expected.issubset(set(saved.columns))

    def test_codigo_municipio_is_five_char_string(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """All codigo_municipio values are 5-character zero-padded strings."""
        _prepare_nbi_csv(tmp_path, 5)
        monkeypatch.setattr(
            "co_president.ingestion.ingest_ipm._read_ecv_hogares",
            lambda _base, _year: _make_synthetic_ecv_hogares(5),
        )
        monkeypatch.setattr(
            "co_president.ingestion.ingest_ipm._load_divipola",
            lambda _base: pd.DataFrame({"codigo_municipio": [f"{i:05d}" for i in range(1, 6)]}),
        )
        build_ipm_features(data_dir=tmp_path)
        saved = pd.read_csv(
            tmp_path / "fundamentals" / "ipm_2018.csv",
            dtype={"codigo_municipio": str},
        )
        assert saved["codigo_municipio"].str.len().eq(5).all()


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
