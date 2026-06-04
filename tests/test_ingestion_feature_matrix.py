"""SPEC-14: Tests for municipal feature matrix integration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from co_president.ingestion.build_feature_matrix import (
    build_feature_matrix,
    generate_data_dictionary,
    load_all_components,
    pivot_historical_wide,
    save_feature_matrix,
    validate_component_health,
)

__all__: list[str] = []


def _make_divipola(n: int = 5) -> pd.DataFrame:
    """Synthetic DIVIPOLA DataFrame with ``n`` municipalities."""
    rng = np.random.default_rng(seed=42)
    return pd.DataFrame(
        {
            "codigo_municipio": [f"{i:05d}" for i in range(1, n + 1)],
            "nombre_municipio": [f"Municipio_{i}" for i in range(1, n + 1)],
            "departamento": ["Antioquia"] * n,
            "latitud": rng.uniform(0, 10, size=n).round(4),
            "longitud": rng.uniform(-80, -70, size=n).round(4),
        }
    )


def _make_historical() -> pd.DataFrame:
    """Synthetic historical election results (3 municipalities, 2 years)."""
    return pd.DataFrame(
        {
            "codigo_municipio": [
                "00001",
                "00001",
                "00001",
                "00001",
                "00002",
                "00002",
                "00002",
                "00002",
                "00003",
                "00003",
            ],
            "year": [2022, 2022, 2018, 2018, 2022, 2022, 2018, 2018, 2022, 2018],
            "round": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "candidate": [
                "gustavo_petro",
                "rodolfo_hernandez",
                "gustavo_petro",
                "ivan_duque",
                "gustavo_petro",
                "rodolfo_hernandez",
                "gustavo_petro",
                "ivan_duque",
                "gustavo_petro",
                "gustavo_petro",
            ],
            "vote_share": [
                0.45,
                0.30,
                0.35,
                0.40,
                0.50,
                0.25,
                0.42,
                0.30,
                0.60,
                0.38,
            ],
            "votes": [4500, 3000, 3500, 4000, 5000, 2500, 4200, 3000, 6000, 3800],
            "total_votes": [10000] * 10,
            "registered_voters": [15000] * 10,
            "abstention_rate": [0.3333] * 10,
        }
    )


def _make_socioeconomic() -> pd.DataFrame:
    """Synthetic socioeconomic feature DataFrame (6 municipalities)."""
    return pd.DataFrame(
        {
            "codigo_municipio": [f"{i:05d}" for i in range(1, 7)],
            "pct_afro_colombian": [0.08, 0.45, 0.02, 0.12, 0.30, 0.05],
            "pct_indigenous": [0.001, 0.02, 0.15, 0.005, 0.01, 0.50],
            "pct_rural_disperso": [0.05, 0.85, 0.30, 0.10, 0.60, 0.80],
            "years_schooling": [10.2, 6.5, 8.0, 11.5, 7.2, 5.0],
            "internet_access_rate": [0.78, 0.25, 0.45, 0.85, 0.30, 0.10],
            "population_2022": [2_569_007, 3_456, 15_200, 7_900_000, 28_000, 5_000],
        }
    )


def _make_risk() -> pd.DataFrame:
    """Synthetic risk feature DataFrame (16 municipalities)."""
    risk_levels = [
        "low",
        "extreme",
        "medium",
        "low",
        "high",
        "medium",
        "low",
        "extreme",
        "high",
        "low",
        "medium",
        "low",
        "high",
        "extreme",
        "low",
        "medium",
    ]
    flags = [0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0]
    return pd.DataFrame(
        {
            "codigo_municipio": [f"{i:05d}" for i in range(1, 17)],
            "risk_level": risk_levels,
            "high_risk_flag": flags,
            "armed_group_presence": flags,
            "is_pdet": [0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
            "coca_hectares": [0, 1250, 0, 0, 450, 800, 0, 2000, 600, 0, 0, 0, 900, 1500, 0, 0],
        }
    )


def _make_cnpv() -> pd.DataFrame:
    """Synthetic CNPV 2018 feature DataFrame (5 municipalities)."""
    return pd.DataFrame(
        {
            "codigo_municipio": [f"{i:05d}" for i in range(1, 6)],
            "poblacion_total": [100_000, 50_000, 25_000, 10_000, 5_000],
            "poblacion_afrocolombiana": [5_000, 1_000, 500, 200, 100],
            "poblacion_indigena": [1_000, 500, 100, 50, 20],
            "poblacion_rural_dispersa": [500, 200, 100, 50, 20],
            "pct_afro_colombian": [0.05, 0.02, 0.02, 0.02, 0.02],
            "pct_indigenous": [0.01, 0.01, 0.004, 0.005, 0.004],
            "pct_rural_disperso": [0.005, 0.004, 0.004, 0.005, 0.004],
            "years_schooling_promedio": [9.5, 8.0, 7.0, 6.0, 5.0],
            "pct_school_attendance": [0.85, 0.80, 0.75, 0.70, 0.65],
            "internet_access_rate": [0.75, 0.60, 0.40, 0.30, 0.20],
            "labor_force_participation_rate": [0.70, 0.72, 0.68, 0.65, 0.60],
            "pct_female": [0.52, 0.51, 0.50, 0.52, 0.51],
            "rooms_per_household": [3.5, 3.0, 2.8, 2.5, 2.0],
            "persons_per_household": [3.8, 3.5, 3.2, 3.0, 2.8],
            "pct_age_18_29": [0.25, 0.28, 0.30, 0.32, 0.35],
            "pct_age_30_54": [0.40, 0.38, 0.35, 0.33, 0.30],
            "pct_age_55_plus": [0.35, 0.34, 0.35, 0.35, 0.35],
        }
    )


def _make_nbi(n: int = 5) -> pd.DataFrame:
    """Synthetic NBI feature DataFrame with ``n`` municipalities."""
    return pd.DataFrame(
        {
            "codigo_municipio": [f"{i:05d}" for i in range(1, n + 1)],
            "nbi_rate": [0.10, 0.72, 0.50, 0.08, 0.60][:n],
            "nbi_urban": [0.08, 0.68, 0.45, 0.06, 0.55][:n],
            "nbi_rural": [0.15, 0.78, 0.55, 0.10, 0.65][:n],
        }
    ).astype({"codigo_municipio": str})


def _make_ipm(n: int = 5) -> pd.DataFrame:
    """Synthetic IPM feature DataFrame with ``n`` municipalities."""
    return pd.DataFrame(
        {
            "codigo_municipio": [f"{i:05d}" for i in range(1, n + 1)],
            "ipm_2018": [0.12, 0.55, 0.35, 0.10, 0.45][:n],
            "ipm_2018_imputed": [True] * n,
            "ipm_2022": [0.14, 0.50, 0.32, 0.11, 0.42][:n],
            "ipm_2022_imputed": [True] * n,
        }
    ).astype({"codigo_municipio": str})


def _make_components(n_divipola: int = 5) -> dict[str, pd.DataFrame]:
    """Build a complete synthetic component dict for testing."""
    return {
        "divipola": _make_divipola(n_divipola),
        "historical": _make_historical(),
        "nbi": _make_nbi(n_divipola),
        "ipm": _make_ipm(n_divipola),
        "socioeconomic": _make_socioeconomic(),
        "risk": _make_risk(),
        "cnpv": _make_cnpv(),
    }


# ═══════════════════════════════════════════════════════════════════
# load_all_components
# ═══════════════════════════════════════════════════════════════════


class TestLoadAllComponents:
    """``load_all_components`` reads component CSVs from disk."""

    def test_returns_dict_with_expected_keys(self, tmp_path: Path) -> None:
        """Returns dict with all seven component keys."""
        _write_component_files(tmp_path, _make_components())
        result = load_all_components(tmp_path)
        assert set(result) == {
            "divipola",
            "historical",
            "nbi",
            "ipm",
            "socioeconomic",
            "risk",
            "cnpv",
        }

    def test_all_dataframes_when_files_exist(self, tmp_path: Path) -> None:
        """Each component DataFrame contains the expected data."""
        _write_component_files(tmp_path, _make_components())
        result = load_all_components(tmp_path)
        for name, df in result.items():
            assert isinstance(df, pd.DataFrame), f"{name}: not a DataFrame"
            assert len(df) > 0, f"{name}: empty DataFrame"

    def test_empty_dataframe_when_file_missing(self, tmp_path: Path) -> None:
        """Missing component files produce empty DataFrames."""
        result = load_all_components(tmp_path)
        assert all(df.empty for df in result.values())

    def test_divipola_has_expected_columns(self, tmp_path: Path) -> None:
        """The loaded DIVIPOLA DataFrame has the canonical columns."""
        _write_component_files(tmp_path, _make_components())
        result = load_all_components(tmp_path)
        expected = {"codigo_municipio", "nombre_municipio", "departamento"}
        assert expected.issubset(set(result["divipola"].columns))


# ═══════════════════════════════════════════════════════════════════
# validate_component_health
# ═══════════════════════════════════════════════════════════════════


class TestValidateComponentHealth:
    """``validate_component_health`` flags data quality issues."""

    def test_clean_components_return_no_warnings(self) -> None:
        """Valid components produce an empty warnings list."""
        components = _make_components()
        warnings = validate_component_health(components)
        assert warnings == []

    def test_empty_dataframe_warning(self) -> None:
        """An empty DataFrame produces a warning."""
        components = _make_components()
        components["risk"] = pd.DataFrame()
        warnings = validate_component_health(components)
        assert any("empty" in w for w in warnings)

    def test_missing_codigo_column_warning(self) -> None:
        """A DataFrame without ``codigo_municipio`` produces a warning."""
        components = _make_components()
        components["socioeconomic"] = components["socioeconomic"].drop(columns=["codigo_municipio"])
        warnings = validate_component_health(components)
        assert any("missing codigo_municipio" in w for w in warnings)

    def test_null_codes_warning(self) -> None:
        """Null values in ``codigo_municipio`` produce a warning."""
        components = _make_components()
        components["divipola"].loc[0, "codigo_municipio"] = None  # type: ignore[typeddict-item]
        warnings = validate_component_health(components)
        assert any("null" in w for w in warnings)

    def test_duplicate_codes_warning(self) -> None:
        """Duplicate ``codigo_municipio`` values produce a warning."""
        components = _make_components()
        components["divipola"].loc[1, "codigo_municipio"] = "00001"
        warnings = validate_component_health(components)
        assert any("duplicate" in w for w in warnings)

    def test_multiple_issues_reported(self) -> None:
        """Multiple issues across different components are all reported."""
        components = _make_components()
        components["divipola"] = pd.DataFrame()
        components["socioeconomic"] = components["socioeconomic"].drop(columns=["codigo_municipio"])
        components["risk"].loc[0, "codigo_municipio"] = None  # type: ignore[typeddict-item]
        warnings = validate_component_health(components)
        assert len(warnings) >= 3

    # --- Strict mode ---

    def test_strict_mode_raises_on_socioeconomic_stub(self) -> None:
        """Strict mode raises ``ValueError`` for a small socioeconomic DataFrame."""
        components = _make_components()
        components["socioeconomic"] = pd.DataFrame(
            {"codigo_municipio": ["00001", "00002", "00003"]}
        )
        with pytest.raises(ValueError, match="stub"):
            validate_component_health(components, strict=True)

    def test_strict_mode_raises_on_risk_stub(self) -> None:
        """Strict mode raises ``ValueError`` for a small risk DataFrame."""
        components = _make_components()
        components["risk"] = pd.DataFrame({"codigo_municipio": [f"{i:05d}" for i in range(1, 11)]})
        with pytest.raises(ValueError, match="stub"):
            validate_component_health(components, strict=True)

    def test_strict_mode_raises_on_historical_placeholder(self) -> None:
        """Strict mode raises ``ValueError`` when historical has ``000NA`` rows."""
        components = _make_components()
        components["historical"] = components["historical"].copy()
        na_rows = pd.DataFrame(
            {
                "codigo_municipio": ["000NA"],
                "year": [2022],
                "round": [1],
                "candidate": ["test"],
                "vote_share": [0.5],
                "votes": [100],
                "total_votes": [200],
                "registered_voters": [300],
                "abstention_rate": [0.0],
            }
        )
        components["historical"] = pd.concat([components["historical"], na_rows], ignore_index=True)
        with pytest.raises(ValueError, match="000NA"):
            validate_component_health(components, strict=True)

    def test_default_mode_does_not_raise_on_stub(self) -> None:
        """Default (non-strict) mode warns but does not raise on stubs."""
        components = _make_components()
        components["socioeconomic"] = pd.DataFrame({"codigo_municipio": ["00001"]})
        warnings = validate_component_health(components)
        assert any("stub" in w.lower() for w in warnings)

    def test_empty_ipm_does_not_warn(self) -> None:
        """Empty IPM DataFrame (optional component) produces no warning."""
        components = _make_components()
        components["ipm"] = pd.DataFrame()
        warnings = validate_component_health(components)
        assert not any("ipm" in w for w in warnings)


# ═══════════════════════════════════════════════════════════════════
# pivot_historical_wide
# ═══════════════════════════════════════════════════════════════════


class TestPivotHistoricalWide:
    """``pivot_historical_wide`` transforms long to wide format."""

    def test_returns_dataframe(self) -> None:
        """Returns a pandas DataFrame."""
        historical = _make_historical()
        result = pivot_historical_wide(historical)
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_municipality(self) -> None:
        """Result has one row per unique municipality."""
        historical = _make_historical()
        result = pivot_historical_wide(historical)
        assert len(result) == historical["codigo_municipio"].nunique()

    def test_has_codigo_municipio_column(self) -> None:
        """Result preserves ``codigo_municipio`` as a column."""
        result = pivot_historical_wide(_make_historical())
        assert "codigo_municipio" in result.columns

    def test_wide_column_naming(self) -> None:
        """Wide columns follow ``vote_share_{year}_r{round}_{cand}`` format."""
        result = pivot_historical_wide(_make_historical())
        vote_cols = [c for c in result.columns if c.startswith("vote_share_")]
        for col in vote_cols:
            parts = col.split("_")
            assert len(parts) >= 5, f"Unexpected column name: {col}"

    def test_values_preserved(self) -> None:
        """Pivoted vote share values match the original long format."""
        historical = _make_historical()
        result = pivot_historical_wide(historical)
        petro_2022 = result.loc[
            result["codigo_municipio"] == "00001", "vote_share_2022_r1_gustavo_petro"
        ]
        assert not petro_2022.empty
        assert petro_2022.iloc[0] == pytest.approx(0.45, abs=1e-6)

    def test_raises_on_missing_columns(self) -> None:
        """Raises ``ValueError`` when required columns are absent."""
        bad = pd.DataFrame({"codigo_municipio": ["00001"]})
        with pytest.raises(ValueError, match="required columns"):
            pivot_historical_wide(bad)

    def test_empty_historical_returns_minimal(self) -> None:
        """Empty input returns a DataFrame with only ``codigo_municipio`` column."""
        empty = pd.DataFrame(
            columns=["codigo_municipio", "year", "round", "candidate", "vote_share"]
        )
        result = pivot_historical_wide(empty)
        assert list(result.columns) == ["codigo_municipio"]
        assert len(result) == 0


# ═══════════════════════════════════════════════════════════════════
# build_feature_matrix
# ═══════════════════════════════════════════════════════════════════


class TestBuildFeatureMatrix:
    """``build_feature_matrix`` joins all components."""

    def test_returns_dataframe(self, tmp_path: Path) -> None:
        """Returns a pandas DataFrame."""
        _write_component_files(tmp_path, _make_components())
        result = build_feature_matrix(data_dir=tmp_path)
        assert isinstance(result, pd.DataFrame)

    def test_correct_row_count(self, tmp_path: Path) -> None:
        """Result has as many rows as the DIVIPOLA registry."""
        _write_component_files(tmp_path, _make_components(5))
        result = build_feature_matrix(data_dir=tmp_path)
        assert len(result) == 5

    def test_has_all_component_columns(self, tmp_path: Path) -> None:
        """Result contains columns from every component."""
        _write_component_files(tmp_path, _make_components(5))
        result = build_feature_matrix(data_dir=tmp_path)
        assert "nombre_municipio" in result.columns
        assert "vote_share_2022_r1_gustavo_petro" in result.columns
        assert "nbi_rate" in result.columns
        assert "ipm_2018" in result.columns
        assert "risk_level" in result.columns

    def test_empty_divipola_returns_empty(self, tmp_path: Path) -> None:
        """Empty DIVIPOLA returns an empty DataFrame."""
        comps = _make_components()
        comps["divipola"] = pd.DataFrame()
        _write_component_files(tmp_path, comps)
        result = build_feature_matrix(data_dir=tmp_path)
        assert result.empty

    def test_unique_codes(self, tmp_path: Path) -> None:
        """Result has unique ``codigo_municipio`` values."""
        _write_component_files(tmp_path, _make_components(5))
        result = build_feature_matrix(data_dir=tmp_path)
        assert result["codigo_municipio"].is_unique

    def test_handles_missing_historical(self, tmp_path: Path) -> None:
        """Build succeeds when historical component is missing."""
        comps = _make_components()
        comps["historical"] = pd.DataFrame()
        _write_component_files(tmp_path, comps)
        result = build_feature_matrix(data_dir=tmp_path)
        assert len(result) == 5
        assert not any(c.startswith("vote_share_") for c in result.columns)

    def test_cnpv_columns_merged(self, tmp_path: Path) -> None:
        """CNPV-specific columns are present and overlapping columns are suffixed."""
        _write_component_files(tmp_path, _make_components(5))
        result = build_feature_matrix(data_dir=tmp_path)

        # CNPV-specific columns (no overlap with socioeconomic).
        cnpv_specific = {"poblacion_total", "poblacion_afrocolombiana", "poblacion_indigena"}
        missing = cnpv_specific - set(result.columns)
        assert not missing, f"Missing CNPV columns: {missing}"

        # Overlapping columns should have both _x (socioeconomic) and _y (CNPV) suffixes.
        overlapping_bases = [
            "pct_afro_colombian",
            "pct_indigenous",
            "pct_rural_disperso",
            "internet_access_rate",
        ]
        for base in overlapping_bases:
            assert f"{base}_x" in result.columns, f"Missing {base}_x (socioeconomic)"
            assert f"{base}_y" in result.columns, f"Missing {base}_y (CNPV)"
            assert base not in result.columns, f"Unsuffixed {base} should not exist after merge"


# ═══════════════════════════════════════════════════════════════════
# generate_data_dictionary
# ═══════════════════════════════════════════════════════════════════


class TestGenerateDataDictionary:
    """``generate_data_dictionary`` writes a valid Markdown file."""

    def test_creates_file(self, tmp_path: Path) -> None:
        """Produces a file at the given path."""
        matrix = _make_divipola(3)
        output = str(tmp_path / "dictionary.md")
        generate_data_dictionary(matrix, output)
        assert Path(output).is_file()

    def test_contains_column_headers(self, tmp_path: Path) -> None:
        """Each column gets a Markdown section header."""
        matrix = _make_divipola(3)
        output = str(tmp_path / "dictionary.md")
        generate_data_dictionary(matrix, output)
        content = Path(output).read_text(encoding="utf-8")
        for col in matrix.columns:
            assert f"## {col}" in content

    def test_contains_type_nulls_sample(self, tmp_path: Path) -> None:
        """Each column entry includes type, null count, and sample value."""
        matrix = _make_divipola(3)
        output = str(tmp_path / "dictionary.md")
        generate_data_dictionary(matrix, output)
        content = Path(output).read_text(encoding="utf-8")
        assert "**Type**" in content
        assert "**Nulls**" in content
        assert "**Sample**" in content

    def test_empty_matrix(self, tmp_path: Path) -> None:
        """An empty DataFrame produces a dictionary with no column entries."""
        empty = pd.DataFrame()
        output = str(tmp_path / "empty_dict.md")
        generate_data_dictionary(empty, output)
        content = Path(output).read_text(encoding="utf-8")
        assert "Columns: 0" in content

    def test_all_null_column_shows_properly(self, tmp_path: Path) -> None:
        """A column with all nulls shows 'ALL NULL' as sample."""
        df = pd.DataFrame({"x": [None, None]})
        output = str(tmp_path / "null_dict.md")
        generate_data_dictionary(df, output)
        content = Path(output).read_text(encoding="utf-8")
        assert "ALL NULL" in content


# ═══════════════════════════════════════════════════════════════════
# save_feature_matrix
# ═══════════════════════════════════════════════════════════════════


class TestSaveFeatureMatrix:
    """``save_feature_matrix`` writes CSV, Parquet, dictionary, and summary."""

    def test_creates_processed_dir(self, tmp_path: Path) -> None:
        """Creates the ``processed/`` subdirectory under data_dir."""
        matrix = _make_divipola(3)
        save_feature_matrix(matrix, data_dir=tmp_path)
        assert (tmp_path / "processed").is_dir()

    def test_writes_csv(self, tmp_path: Path) -> None:
        """Writes a CSV file."""
        matrix = _make_divipola(3)
        save_feature_matrix(matrix, data_dir=tmp_path)
        assert (tmp_path / "processed" / "municipal_feature_matrix.csv").is_file()

    def test_writes_parquet(self, tmp_path: Path) -> None:
        """Writes a Parquet file."""
        matrix = _make_divipola(3)
        save_feature_matrix(matrix, data_dir=tmp_path)
        assert (tmp_path / "processed" / "municipal_feature_matrix.parquet").is_file()

    def test_writes_data_dictionary(self, tmp_path: Path) -> None:
        """Writes the feature dictionary Markdown file."""
        matrix = _make_divipola(3)
        save_feature_matrix(matrix, data_dir=tmp_path)
        assert (tmp_path / "processed" / "feature_dictionary.md").is_file()

    def test_writes_summary_json(self, tmp_path: Path) -> None:
        """Writes the build summary JSON file."""
        matrix = _make_divipola(3)
        save_feature_matrix(matrix, data_dir=tmp_path)
        assert (tmp_path / "processed" / "build_summary.json").is_file()

    def test_summary_has_expected_keys(self, tmp_path: Path) -> None:
        """The build summary JSON contains expected metadata fields."""
        matrix = _make_divipola(3)
        save_feature_matrix(matrix, data_dir=tmp_path)
        summary = json.loads(
            (tmp_path / "processed" / "build_summary.json").read_text(encoding="utf-8")
        )
        for key in (
            "municipalities",
            "columns",
            "null_count",
            "null_rate",
            "memory_bytes",
            "memory_mb",
        ):
            assert key in summary, f"Missing key: {key}"

    def test_roundtrip_csv_parquet_identical(self, tmp_path: Path) -> None:
        """CSV and Parquet contain the same data after save."""
        matrix = _make_divipola(3)
        save_feature_matrix(matrix, data_dir=tmp_path)
        csv_df = pd.read_csv(
            tmp_path / "processed" / "municipal_feature_matrix.csv",
            dtype={"codigo_municipio": str},
        )
        parquet_df = pd.read_parquet(
            tmp_path / "processed" / "municipal_feature_matrix.parquet",
        )
        pd.testing.assert_frame_equal(csv_df, parquet_df, check_dtype=False)

    def test_raises_on_empty_matrix(self, tmp_path: Path) -> None:
        """Raises ``ValueError`` when matrix is empty."""
        empty = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            save_feature_matrix(empty, data_dir=tmp_path)


# ═══════════════════════════════════════════════════════════════════
# Private test helpers
# ═══════════════════════════════════════════════════════════════════


def _write_component_files(data_dir: Path, components: dict[str, pd.DataFrame]) -> None:
    """Write synthetic component DataFrames to the fundamentals directory."""
    fundamentals_dir = data_dir / "fundamentals"
    fundamentals_dir.mkdir(parents=True, exist_ok=True)
    filenames = {
        "divipola": "divipola_master.csv",
        "historical": "historical_results.csv",
        "nbi": "nbi_2018.csv",
        "ipm": "ipm_2018.csv",
        "socioeconomic": "socioeconomic.csv",
        "risk": "risk_factors.csv",
        "cnpv": "cnpv_2018.csv",
    }
    for name, df in components.items():
        filename = filenames[name]
        df.to_csv(fundamentals_dir / filename, index=False)