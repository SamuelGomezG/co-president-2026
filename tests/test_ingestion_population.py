"""SPEC-19: Tests for DANE population projections ingest."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from co_president.config import BELEN_DE_BAJIRA_CODE
from co_president.ingestion.ingest_population import (
    _fill_missing_population,
    _filter_total_rows,
    _pivot_population_wide,
    _read_population_xlsx,
    build_population_features,
    load_population_data,
    validate_population,
)

__all__: list[str] = []


# ═══════════════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════════════


def _make_synthetic_xlsx(n_mpios: int = 3, years: list[int] | None = None) -> pd.DataFrame:
    """Build a synthetic raw DataFrame matching the XLSX "PobMunicipalxArea" schema.

    Returns a long-format DataFrame with Cabecera, Rural, and Total rows for
    each (municipio, year) combination.
    """
    if years is None:
        years = [2018, 2020, 2022, 2024, 2026]
    area_options = ["Cabecera Municipal", "Centros Poblados y Rural Disperso", "Total"]
    rows: list[dict[str, object]] = []
    for i in range(1, n_mpios + 1):
        mpio = f"{i:05d}"
        for year in years:
            cabecera = i * 100_000 + (year - 2018) * 1_000
            rural = cabecera // 20
            total = cabecera + rural
            for area, value in zip(area_options, [cabecera, rural, total], strict=True):
                rows.append(
                    {
                        "DP": str(i).zfill(2),
                        "DPNOM": f"Dept_{i}",
                        "MPIO": mpio,
                        "DPMP": f"Municipio_{i}",
                        "AÑO": year,
                        "ÁREA GEOGRÁFICA": area,
                        "TOTAL": value,
                    }
                )
    return pd.DataFrame(rows)


def _make_valid_population(n: int = 5) -> pd.DataFrame:
    """Build a synthetically valid population features DataFrame with ``n`` municipalities."""
    rng = np.random.default_rng(seed=42)
    data: dict[str, list[object]] = {"codigo_municipio": [f"{i:05d}" for i in range(1, n + 1)]}
    for year in range(2018, 2027):
        data[f"pop_{year}"] = [round(x) for x in rng.uniform(5_000, 2_000_000, size=n)]
    return pd.DataFrame(data).astype({"codigo_municipio": str})


def _setup_mock_fundamentals(base_path: Path, codes: list[str]) -> None:
    """Create mock fundamentals files matching the given municipality codes.

    Writes minimal ``divipola_master.csv`` and ``cnpv_2018.csv`` to
    ``base_path/fundamentals/`` so that ``_fill_missing_population`` can
    read them without requiring real data files.

    Args:
        base_path: Root data directory (``fundamentals/`` is created under
            this path).
        codes: Municipality codes to include in the mock DIVIPOLA master.

    """
    fund_dir = base_path / "fundamentals"
    fund_dir.mkdir(parents=True, exist_ok=True)

    divipola = pd.DataFrame(
        {
            "REGION": ["Mock"] * len(codes),
            "CÓDIGO DANE DEL DEPARTAMENTO": [c[:2] for c in codes],
            "departamento": ["Mock"] * len(codes),
            "codigo_municipio": codes,
            "nombre_municipio": [f"Mock_{c}" for c in codes],
        }
    )
    divipola.to_csv(fund_dir / "divipola_master.csv", index=False)

    cnpv = pd.DataFrame(
        {
            "codigo_municipio": codes,
            "poblacion_total": [100_000 + i * 1_000 for i in range(len(codes))],
        }
    )
    cnpv.to_csv(fund_dir / "cnpv_2018.csv", index=False)


# ═══════════════════════════════════════════════════════════════════
# _read_population_xlsx
# ═══════════════════════════════════════════════════════════════════


class TestReadPopulationXlsx:
    """_read_population_xlsx reads the DANE PPED Excel file."""

    @pytest.fixture(scope="class")
    def _xlsx_df(self, data_dir: Path) -> pd.DataFrame:
        """Read the real XLSX once per test class."""
        xlsx_path = data_dir / "raw" / "PPED-AreaMun-2018-2042_VP.xlsx"
        if not xlsx_path.is_file():
            pytest.skip("Real XLSX file not available (CI)")
        return _read_population_xlsx(xlsx_path)

    def test_reads_real_file(self, _xlsx_df: pd.DataFrame) -> None:
        """Returns a non-empty DataFrame with expected columns from the real XLSX."""
        assert not _xlsx_df.empty
        expected = {"MPIO", "AÑO", "ÁREA GEOGRÁFICA", "TOTAL"}
        assert expected.issubset(set(_xlsx_df.columns))
        assert _xlsx_df["AÑO"].min() >= 2018
        assert _xlsx_df["AÑO"].max() <= 2042
        assert _xlsx_df["MPIO"].str.len().eq(5).all()

    def test_all_total_rows_have_positive_population(self, _xlsx_df: pd.DataFrame) -> None:
        """After _filter_total_rows, all remaining Total rows have positive population."""
        total_rows = _filter_total_rows(_xlsx_df)
        assert not total_rows.empty, "_filter_total_rows returned zero rows"
        assert (total_rows["TOTAL"] > 0).all()


# ═══════════════════════════════════════════════════════════════════
# _filter_total_rows
# ═══════════════════════════════════════════════════════════════════


class TestFilterTotalRows:
    """_filter_total_rows keeps only the "Total" rows."""

    def test_filters_to_total_only(self) -> None:
        """Only rows with ÁREA GEOGRÁFICA == 'Total' remain."""
        raw = _make_synthetic_xlsx(3)
        result = _filter_total_rows(raw)
        assert (result["ÁREA GEOGRÁFICA"] == "Total").all()

    def test_removes_two_thirds_of_rows(self) -> None:
        """With 3 area types, filtering keeps 1/3 of the original rows."""
        raw = _make_synthetic_xlsx(5, years=[2018, 2022])
        result = _filter_total_rows(raw)
        assert len(result) == len(raw) // 3

    def test_handles_empty_input(self) -> None:
        """Empty input returns an empty DataFrame with the same columns."""
        raw_columns = ["MPIO", "AÑO", "ÁREA GEOGRÁFICA", "TOTAL"]
        empty = pd.DataFrame(columns=raw_columns)
        result = _filter_total_rows(empty)
        assert result.empty
        assert list(result.columns) == raw_columns

    def test_total_equals_cabecera_plus_rural(self) -> None:
        """For each (mpio, year), Total == Cabecera + Rural."""
        raw = _make_synthetic_xlsx(2, years=[2018, 2022, 2026])
        result = _filter_total_rows(raw)
        for _, row in result.iterrows():
            mpio = row["MPIO"]
            year = row["AÑO"]
            cabecera = int(
                raw[
                    (raw["MPIO"] == mpio)
                    & (raw["AÑO"] == year)
                    & (raw["ÁREA GEOGRÁFICA"] == "Cabecera Municipal")
                ]["TOTAL"].iloc[0]
            )
            rural = int(
                raw[
                    (raw["MPIO"] == mpio)
                    & (raw["AÑO"] == year)
                    & (raw["ÁREA GEOGRÁFICA"] == "Centros Poblados y Rural Disperso")
                ]["TOTAL"].iloc[0]
            )
            assert int(row["TOTAL"]) == cabecera + rural


# ═══════════════════════════════════════════════════════════════════
# _pivot_population_wide
# ═══════════════════════════════════════════════════════════════════


class TestPivotPopulationWide:
    """_pivot_population_wide transforms long to wide format."""

    def test_has_correct_shape(self) -> None:
        """Result has N rows and 10 columns (codigo_municipio + 9 yearly pop_*)."""
        all_years = list(range(2018, 2027))
        filtered = _filter_total_rows(_make_synthetic_xlsx(5, years=all_years))
        result = _pivot_population_wide(filtered)
        assert len(result) == 5
        assert len(result.columns) == 10

    def test_column_names(self) -> None:
        """Columns are codigo_municipio and pop_2018..pop_2026 in order."""
        all_years = list(range(2018, 2027))
        filtered = _filter_total_rows(_make_synthetic_xlsx(3, years=all_years))
        result = _pivot_population_wide(filtered)
        expected = ["codigo_municipio"] + [f"pop_{y}" for y in range(2018, 2027)]
        assert list(result.columns) == expected

    def test_filters_years_beyond_2026(self) -> None:
        """Years > 2026 are excluded from the output."""
        all_years = [*list(range(2018, 2027)), 2030, 2042]
        filtered = _filter_total_rows(_make_synthetic_xlsx(2, years=all_years))
        result = _pivot_population_wide(filtered)
        year_cols = [c for c in result.columns if c.startswith("pop_")]
        years = [int(c.split("_")[1]) for c in year_cols]
        assert all(2018 <= y <= 2026 for y in years)
        assert 2030 not in years
        assert 2042 not in years

    def test_codigo_municipio_is_first_column(self) -> None:
        """codigo_municipio is the first column."""
        all_years = list(range(2018, 2027))
        filtered = _filter_total_rows(_make_synthetic_xlsx(3, years=all_years))
        result = _pivot_population_wide(filtered)
        assert next(iter(result.columns)) == "codigo_municipio"

    def test_national_2022_total_within_tolerance(self) -> None:
        """Assert national pop_2022 total is within ±5% of 50M (synthetic)."""
        rng = np.random.default_rng(seed=42)
        n = 100
        # Build a synthetic dataset with all years 2018-2026
        raw_dict: dict = {"MPIO": [], "AÑO": [], "ÁREA GEOGRÁFICA": [], "TOTAL": []}
        for i in range(1, n + 1):
            mpio = f"{i:05d}"
            for year in range(2018, 2027):
                raw_dict["MPIO"].extend([mpio] * 3)
                raw_dict["AÑO"].extend([year] * 3)
                raw_dict["ÁREA GEOGRÁFICA"].extend(
                    ["Cabecera Municipal", "Centros Poblados y Rural Disperso", "Total"]
                )
                pop = int(rng.uniform(300_000, 700_000))
                raw_dict["TOTAL"].extend([int(pop * 0.85), int(pop * 0.15), pop])
        raw = pd.DataFrame(raw_dict)
        filtered = _filter_total_rows(raw)
        result = _pivot_population_wide(filtered)
        total_2022 = int(result["pop_2022"].sum())
        assert 47_500_000 <= total_2022 <= 52_500_000

    def test_all_years_present(self) -> None:
        """All 9 years 2018-2026 are present as columns."""
        all_years = list(range(2018, 2027))
        filtered = _filter_total_rows(_make_synthetic_xlsx(3, years=all_years))
        result = _pivot_population_wide(filtered)
        for year in range(2018, 2027):
            assert f"pop_{year}" in result.columns


# ───────────────────────────────────────────────────────────────────
# Orchestrator tests — build_population_features writes the CSV
# ───────────────────────────────────────────────────────────────────


class TestBuildPopulationFeatures:
    """build_population_features writes the output CSV."""

    _ALL_YEARS: ClassVar[list[int]] = list(range(2018, 2027))

    def test_writes_csv(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """CSV is written to fundamentals/population_2018_2026.csv."""
        _setup_mock_fundamentals(tmp_path, [f"{i:05d}" for i in range(1, 6)])
        monkeypatch.setattr(
            "co_president.ingestion.ingest_population._read_population_xlsx",
            lambda _: _make_synthetic_xlsx(5, years=self._ALL_YEARS),
        )
        build_population_features(data_dir=tmp_path)
        assert (tmp_path / "fundamentals" / "population_2018_2026.csv").is_file()

    def test_output_shape(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Output CSV has 5 rows and 10 columns."""
        _setup_mock_fundamentals(tmp_path, [f"{i:05d}" for i in range(1, 6)])
        monkeypatch.setattr(
            "co_president.ingestion.ingest_population._read_population_xlsx",
            lambda _: _make_synthetic_xlsx(5, years=self._ALL_YEARS),
        )
        build_population_features(data_dir=tmp_path)
        saved = pd.read_csv(
            tmp_path / "fundamentals" / "population_2018_2026.csv",
            dtype={"codigo_municipio": str},
        )
        assert len(saved) == 5
        assert len(saved.columns) == 10

    def test_codigo_municipio_is_string(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Column codigo_municipio is a string (not int)."""
        _setup_mock_fundamentals(tmp_path, [f"{i:05d}" for i in range(1, 6)])
        monkeypatch.setattr(
            "co_president.ingestion.ingest_population._read_population_xlsx",
            lambda _: _make_synthetic_xlsx(5, years=self._ALL_YEARS),
        )
        build_population_features(data_dir=tmp_path)
        saved = pd.read_csv(
            tmp_path / "fundamentals" / "population_2018_2026.csv",
            dtype={"codigo_municipio": str},
        )
        is_string_dtype = saved["codigo_municipio"].dtype == np.dtype("object")
        is_pd_string = isinstance(saved["codigo_municipio"].dtype, pd.StringDtype)
        assert is_string_dtype or is_pd_string

    def test_creates_fundamentals_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The fundamentals/ directory is created if it does not exist."""
        _setup_mock_fundamentals(tmp_path, [f"{i:05d}" for i in range(1, 4)])
        monkeypatch.setattr(
            "co_president.ingestion.ingest_population._read_population_xlsx",
            lambda _: _make_synthetic_xlsx(3, years=self._ALL_YEARS),
        )
        build_population_features(data_dir=tmp_path)
        assert (tmp_path / "fundamentals").is_dir()

    def test_raises_on_missing_xlsx(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError when the XLSX does not exist."""
        with pytest.raises(FileNotFoundError, match="PPED-AreaMun"):
            build_population_features(data_dir=tmp_path)

    def test_population_values_are_integers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Population values are integers (not floats)."""
        _setup_mock_fundamentals(tmp_path, [f"{i:05d}" for i in range(1, 6)])
        monkeypatch.setattr(
            "co_president.ingestion.ingest_population._read_population_xlsx",
            lambda _: _make_synthetic_xlsx(5, years=self._ALL_YEARS),
        )
        build_population_features(data_dir=tmp_path)
        saved = pd.read_csv(
            tmp_path / "fundamentals" / "population_2018_2026.csv",
            dtype={"codigo_municipio": str},
        )
        year_cols = [c for c in saved.columns if c.startswith("pop_")]
        for col in year_cols:
            assert saved[col].dtype in (np.dtype("int64"), np.dtype("float64"))
            assert (saved[col].dropna() % 1 == 0).all()


# ═══════════════════════════════════════════════════════════════════
# load_population_data
# ═══════════════════════════════════════════════════════════════════


class TestLoadPopulationData:
    """load_population_data reads a previously-saved CSV."""

    def test_round_trip(self, tmp_path: Path) -> None:
        """Round-trip: write then read returns identical data."""
        df = _make_valid_population(5)
        out_dir = tmp_path / "fundamentals"
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "population_2018_2026.csv", index=False)
        loaded = load_population_data(data_dir=tmp_path)
        pd.testing.assert_frame_equal(df, loaded)

    def test_raises_when_missing(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError when file does not exist."""
        with pytest.raises(FileNotFoundError, match="population_2018_2026"):
            load_population_data(data_dir=tmp_path)


# ═══════════════════════════════════════════════════════════════════
# validate_population
# ═══════════════════════════════════════════════════════════════════


class TestValidatePopulation:
    """validate_population checks output against acceptance criteria."""

    def test_valid_data_returns_empty(self) -> None:
        """Valid population DataFrame yields no warnings."""
        df = _make_valid_population(1123)
        warnings = validate_population(df)
        assert warnings == []

    def test_missing_column_warns(self) -> None:
        """Missing pop_2018 column triggers a warning."""
        df = _make_valid_population(5).drop(columns=["pop_2018"])
        warnings = validate_population(df)
        assert any("pop_2018" in w for w in warnings)

    def test_null_in_critical_column_warns(self) -> None:
        """Null in pop_2022 triggers a warning."""
        df = _make_valid_population(5)
        df.loc[0, "pop_2022"] = pd.NA
        warnings = validate_population(df)
        assert any("pop_2022" in w for w in warnings)

    def test_non_five_char_code_warns(self) -> None:
        """codigo_municipio with length != 5 triggers a warning."""
        df = _make_valid_population(5)
        df.loc[0, "codigo_municipio"] = "123"
        warnings = validate_population(df)
        assert any("5" in w for w in warnings)

    def test_fewer_than_expected_rows_warns(self) -> None:
        """Fewer than 1123 rows triggers a row-count warning."""
        df = _make_valid_population(5)
        warnings = validate_population(df)
        assert any("rows" in w for w in warnings)

    def test_negative_population_warns(self) -> None:
        """A negative population value triggers a warning."""
        df = _make_valid_population(5)
        df.loc[0, "pop_2022"] = -1
        warnings = validate_population(df)
        assert any("negative" in w or "non-positive" in w for w in warnings)

    def test_population_values_are_integers(self) -> None:
        """Population columns are integer dtypes (not float)."""
        df = _make_valid_population(5)
        year_cols = [c for c in df.columns if c.startswith("pop_")]
        for col in year_cols:
            assert df[col].dtype in (np.dtype("int64"), np.dtype("uint64")), (
                f"{col}: expected integer dtype, got {df[col].dtype}"
            )

    def test_all_required_columns_present(self) -> None:
        """Output has codigo_municipio + 9 pop_* columns = 10 columns."""
        df = _make_valid_population(5)
        expected = {"codigo_municipio"} | {f"pop_{y}" for y in range(2018, 2027)}
        assert expected.issubset(set(df.columns))
        assert len(df.columns) == 10

    def test_population_not_float64(self) -> None:
        """Population columns should not be float64 (no decimals)."""
        df = _make_valid_population(5)
        year_cols = [c for c in df.columns if c.startswith("pop_")]
        for col in year_cols:
            assert df[col].dtype != np.dtype("float64"), (
                f"{col}: expected non-float dtype, got float64"
            )

    def test_float_population_values_warn(self) -> None:
        """Float population columns trigger a validation warning."""
        df = _make_valid_population(5)
        df["pop_2022"] = df["pop_2022"].astype(float)
        warnings = validate_population(df)
        assert any("float" in w for w in warnings), f"Expected float warning, got: {warnings}"

    def test_nan_population_values_warn(self) -> None:
        """NaN in population columns triggers a validation warning."""
        df = _make_valid_population(5)
        df.loc[0, "pop_2020"] = float("nan")
        warnings = validate_population(df)
        assert any("null" in w for w in warnings), f"Expected null warning, got: {warnings}"


# ═══════════════════════════════════════════════════════════════════
# _fill_missing_population
# ═══════════════════════════════════════════════════════════════════


class TestFillMissingPopulation:
    """_fill_missing_population adds projections for DIVIPOLA codes missing from DANE PPED."""

    _POP_COLS: ClassVar[list[str]] = [f"pop_{y}" for y in range(2018, 2027)]

    def _write_mock_files(
        self,
        base_path: Path,
        divipola_codes: list[str],
        cnpv_overrides: dict[str, int] | None = None,
    ) -> None:
        """Write mock divipola_master.csv and cnpv_2018.csv to ``base_path/fundamentals/``."""
        fund_dir = base_path / "fundamentals"
        fund_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            {
                "REGION": ["Mock"] * len(divipola_codes),
                "CÓDIGO DANE DEL DEPARTAMENTO": [c[:2] for c in divipola_codes],
                "departamento": ["Mock"] * len(divipola_codes),
                "codigo_municipio": divipola_codes,
                "nombre_municipio": [f"Mock_{c}" for c in divipola_codes],
            }
        ).to_csv(fund_dir / "divipola_master.csv", index=False)

        cnpv_data: dict[str, list[str | int]] = {
            "codigo_municipio": [],
            "poblacion_total": [],
        }
        for code in divipola_codes:
            if cnpv_overrides and code in cnpv_overrides:
                cnpv_data["codigo_municipio"].append(code)
                cnpv_data["poblacion_total"].append(cnpv_overrides[code])
        pd.DataFrame(cnpv_data).to_csv(fund_dir / "cnpv_2018.csv", index=False)

    # ── Existing codes ──────────────────────────────────────────

    def test_skips_existing_codes(self, tmp_path: Path) -> None:
        """When all DIVIPOLA codes are already present, return unchanged."""
        self._write_mock_files(tmp_path, ["27001", "27006"])
        wide = pd.DataFrame(
            {
                "codigo_municipio": ["27001", "27006"],
                **{col: [1000, 2000] for col in self._POP_COLS},
            }
        )
        result = _fill_missing_population(wide, tmp_path)
        assert len(result) == 2
        pd.testing.assert_frame_equal(result, wide)

    # ── CNPV baseline ───────────────────────────────────────────

    def test_fills_missing_code_with_cnpv_baseline(self, tmp_path: Path) -> None:
        """Missing DIVIPOLA code gets projected using CNPV baseline and dept-wide growth rate."""
        self._write_mock_files(tmp_path, ["27001", "27099"], cnpv_overrides={"27099": 50_000})

        wide = pd.DataFrame(
            {
                "codigo_municipio": ["27001"],
                **{col: [1000] for col in self._POP_COLS},
            }
        )
        result = _fill_missing_population(wide, tmp_path)
        assert len(result) == 2

        row = result[result["codigo_municipio"] == "27099"].iloc[0]
        assert row["pop_2018"] == 50_000
        for y in range(2019, 2027):
            assert row[f"pop_{y}"] == 50_000, f"{y}: expected 50,000 with 0% growth"

    # ── Fallback estimate (27086 Belén de Bajirá) ───────────────

    def test_fallback_estimate_for_27086(self, tmp_path: Path) -> None:
        """When CNPV has no data for 27086, fall back to 25,000 baseline."""
        self._write_mock_files(tmp_path, ["27001", BELEN_DE_BAJIRA_CODE], cnpv_overrides={})

        wide = pd.DataFrame(
            {
                "codigo_municipio": ["27001"],
                **{col: [1000] for col in self._POP_COLS},
            }
        )
        result = _fill_missing_population(wide, tmp_path)
        row = result[result["codigo_municipio"] == BELEN_DE_BAJIRA_CODE].iloc[0]
        assert row["pop_2018"] == 25_000
        for y in range(2019, 2027):
            assert row[f"pop_{y}"] == 25_000

    # ── Column completeness ─────────────────────────────────────

    def test_all_pop_columns_filled(self, tmp_path: Path) -> None:
        """The appended row has all 9 pop_2018..pop_2026 columns non-null."""
        self._write_mock_files(tmp_path, ["27001", "27999"], cnpv_overrides={"27999": 10_000})

        wide = pd.DataFrame(
            {
                "codigo_municipio": ["27001", "27006"],
                **{col: [1000, 2000] for col in self._POP_COLS},
            }
        )
        result = _fill_missing_population(wide, tmp_path)
        row = result[result["codigo_municipio"] == "27999"].iloc[0]
        for col in self._POP_COLS:
            assert pd.notna(row[col]), f"{col} is NaN"
            assert row[col] > 0, f"{col} is <= 0"

    # ── Growth rate projection ──────────────────────────────────

    def test_growth_rate_projection(self, tmp_path: Path) -> None:
        """Projected values follow geometric growth computed from existing dept munis."""
        self._write_mock_files(tmp_path, ["27001", "27999"], cnpv_overrides={"27999": 100_000})

        wide = pd.DataFrame(
            {
                "codigo_municipio": ["27001"],
                **{col: [1000] for col in self._POP_COLS},
            }
        )
        wide.loc[0, "pop_2026"] = 2000

        result = _fill_missing_population(wide, tmp_path)
        row = result[result["codigo_municipio"] == "27999"].iloc[0]

        rate = (2000 / 1000) ** (1 / 8) - 1
        for y in range(2018, 2027):
            expected = round(100_000 * (1 + rate) ** (y - 2018))
            assert row[f"pop_{y}"] == expected, (
                f"pop_{y}: expected {expected}, got {row[f'pop_{y}']}"
            )
