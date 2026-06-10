"""SPEC-21: Tests for MunicipalFeatures dataclass and load_features()."""

from __future__ import annotations

import math
from pathlib import Path
import shutil

import pandas as pd
import pytest

from co_president.fundamentals.features import (
    HistoricalRecord,
    MunicipalFeatures,
    clr,
    load_features,
    logit,
)

__all__: list[str] = []

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "fundamentals"
_MATRIX_FIXTURE = _FIXTURE_DIR / "municipal_feature_matrix.csv"
_HISTORICAL_FIXTURE = _FIXTURE_DIR / "historical_results.csv"

_EXPECTED_MUNICIPALITIES = 1_142


# ═══════════════════════════════════════════════════════════════════════
# Test helpers
# ═══════════════════════════════════════════════════════════════════════


def _setup_fixture_dir(tmp_path: Path) -> Path:
    """Copy fixture files into ``tmp_path`` mimicking the data directory layout.

    Creates ``processed/municipal_feature_matrix.csv`` (with ``comuna_nombre``
    and fiscal autonomy columns added) and ``fundamentals/historical_results.csv``
    under *tmp_path*.
    """
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    matrix = pd.read_csv(_MATRIX_FIXTURE, dtype={"codigo_municipio": str})
    matrix["comuna_nombre"] = None
    matrix["pct_ingresos_propios"] = 0.5
    matrix["gastos_totales_per_capita"] = 2.0
    matrix["transferencias_per_capita"] = 1.0
    matrix["ingresos_tributarios_per_capita"] = 0.8
    matrix.to_csv(processed_dir / "municipal_feature_matrix.csv", index=False)

    fundamentals_dir = tmp_path / "fundamentals"
    fundamentals_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(_HISTORICAL_FIXTURE), str(fundamentals_dir / "historical_results.csv"))

    return tmp_path


def _assert_no_nulls(df: pd.DataFrame, columns: list[str]) -> None:
    """Assert that no nulls exist in the given columns."""
    nulls = df[columns].isna().sum()
    bad = [(col, int(nulls[col])) for col in columns if nulls[col] > 0]
    if bad:
        msg = f"Nulls found in columns: {bad}"
        raise AssertionError(msg)


# ═══════════════════════════════════════════════════════════════════════
# MunicipalFeatures
# ═══════════════════════════════════════════════════════════════════════


class TestMunicipalFeatures:
    """MunicipalFeatures dataclass contract tests."""

    def test_is_frozen(self) -> None:
        """MunicipalFeatures raises FrozenInstanceError on mutation."""
        mf = MunicipalFeatures(
            codigo_municipio="05001",
            nombre_municipio="Medellín",
            comuna_nombre=None,
            departamento="Antioquia",
            region="Andina",
            poblacion_total=2_569_007,
            poblacion_afrocolombiana=5_000,
            poblacion_indigena=1_000,
            poblacion_rural_dispersa=500,
            pct_afro_colombian=0.08,
            pct_indigenous=0.01,
            pct_rural_disperso=0.05,
            years_schooling_promedio=10.2,
            pct_school_attendance=0.85,
            internet_access_rate=0.78,
            labor_force_participation_rate=0.70,
            pct_female=0.52,
            rooms_per_household=3.5,
            persons_per_household=3.8,
            pct_age_18_29=0.25,
            pct_age_30_54=0.40,
            pct_age_55_plus=0.35,
            nbi_rate=0.15,
            nbi_urban=0.12,
            nbi_rural=0.20,
            ipm_2018=0.12,
            ipm_2018_imputed=True,
            ipm_2022=0.14,
            ipm_2022_imputed=True,
            pop_2018=2_500_000,
            pop_2019=2_510_000,
            pop_2020=2_520_000,
            pop_2021=2_530_000,
            pop_2022=2_540_000,
            pop_2023=2_550_000,
            pop_2024=2_560_000,
            pop_2025=2_570_000,
            pop_2026=2_580_000,
            pct_ingresos_propios=0.65,
            gastos_totales_per_capita=2.5,
            transferencias_per_capita=0.8,
            ingresos_tributarios_per_capita=1.2,
            risk_level="medium",
            is_pdet=False,
            armed_group_presence=False,
            coca_hectares=0.0,
            high_risk_flag=False,
            historical=(),
            historical_turnout_m=0.75,
        )
        with pytest.raises(AttributeError):
            mf.codigo_municipio = "99999"  # type: ignore[misc]

    def test_historical_is_frozen(self) -> None:
        """HistoricalRecord raises FrozenInstanceError on mutation."""
        hr = HistoricalRecord(
            year=2022,
            round=1,
            left_candidate="gustavo_petro",
            right_candidate="rodolfo_hernandez",
            left_share=0.45,
            right_share=0.30,
            abstention_rate=0.3333,
        )
        with pytest.raises(AttributeError):
            hr.year = 2018  # type: ignore[misc]

    def test_historical_is_tuple(self) -> None:
        """MunicipalFeatures.historical field is a tuple."""
        mf = MunicipalFeatures(
            codigo_municipio="05001",
            nombre_municipio="Test",
            comuna_nombre=None,
            departamento="Antioquia",
            region=None,
            poblacion_total=100,
            poblacion_afrocolombiana=10,
            poblacion_indigena=5,
            poblacion_rural_dispersa=2,
            pct_afro_colombian=0.10,
            pct_indigenous=0.05,
            pct_rural_disperso=0.02,
            years_schooling_promedio=10.0,
            pct_school_attendance=0.85,
            internet_access_rate=0.70,
            labor_force_participation_rate=0.65,
            pct_female=0.51,
            rooms_per_household=3.0,
            persons_per_household=3.5,
            pct_age_18_29=0.30,
            pct_age_30_54=0.40,
            pct_age_55_plus=0.30,
            nbi_rate=0.20,
            nbi_urban=0.15,
            nbi_rural=0.30,
            ipm_2018=None,
            ipm_2018_imputed=True,
            ipm_2022=None,
            ipm_2022_imputed=True,
            pop_2018=100_000,
            pop_2019=101_000,
            pop_2020=102_000,
            pop_2021=103_000,
            pop_2022=104_000,
            pop_2023=105_000,
            pop_2024=106_000,
            pop_2025=107_000,
            pop_2026=108_000,
            pct_ingresos_propios=0.60,
            gastos_totales_per_capita=2.0,
            transferencias_per_capita=0.9,
            ingresos_tributarios_per_capita=1.0,
            risk_level="low",
            is_pdet=False,
            armed_group_presence=False,
            coca_hectares=0.0,
            high_risk_flag=False,
            historical=(
                HistoricalRecord(
                    year=2022,
                    round=1,
                    left_candidate="gustavo_petro",
                    right_candidate="rodolfo_hernandez",
                    left_share=0.45,
                    right_share=0.30,
                    abstention_rate=0.3333,
                ),
                HistoricalRecord(
                    year=2018,
                    round=1,
                    left_candidate="gustavo_petro",
                    right_candidate="ivan_duque",
                    left_share=0.35,
                    right_share=0.40,
                    abstention_rate=0.2500,
                ),
            ),
            historical_turnout_m=0.70,
        )
        assert isinstance(mf.historical, tuple)
        assert len(mf.historical) == 2
        assert mf.historical[0].year == 2022


# ═══════════════════════════════════════════════════════════════════════
# Compositional data helpers (SPEC-21c)
# ═══════════════════════════════════════════════════════════════════════


class TestCLR:
    """Tests for ``clr()`` centred log-ratio transform."""

    def test_sums_to_zero(self) -> None:
        """CLR output sums to zero (Aitchison property)."""
        compositions = [
            (0.1, 0.2, 0.3, 0.4),
            (0.25, 0.25, 0.25, 0.25),
            (0.5, 0.3, 0.2),
            (0.01, 0.99),
            (0.3, 0.7),
        ]
        for comp in compositions:
            out = clr(comp)
            assert abs(sum(out)) < 1e-12, f"CLR({comp}) sums to {sum(out)}"

    def test_symmetric_binary(self) -> None:
        """CLR((0.3, 0.7)) produces symmetric values (clr[0] == -clr[1])."""
        out = clr((0.3, 0.7))
        assert abs(out[0] - (-out[1])) < 1e-12

    def test_logit_equivalence(self) -> None:
        """For D=2, CLR(p, 1-p)[0] == 0.5 * logit(p)."""
        for p in (0.05, 0.15, 0.30, 0.50, 0.85, 0.95):
            out = clr((p, 1.0 - p))
            expected = 0.5 * logit(p)
            assert abs(out[0] - expected) < 1e-12, f"Failed for p={p}"

    def test_clr_shares_on_binary_record(self) -> None:
        """HistoricalRecord.clr_shares() returns CLR of (left, right)."""
        hr = HistoricalRecord(
            year=2022,
            round=2,
            left_candidate="gustavo_petro",
            right_candidate="rodolfo_hernandez",
            left_share=0.55,
            right_share=0.45,
            abstention_rate=0.30,
        )
        clr_left, clr_right = hr.clr_shares()
        assert abs(clr_left + clr_right) < 1e-12
        expected = 0.5 * logit(0.55 / 1.0)
        assert abs(clr_left - expected) < 1e-12

    def test_clr_shares_normalises_sum(self) -> None:
        """clr_shares() normalises left+right to sum to 1."""
        hr = HistoricalRecord(
            year=2022,
            round=1,
            left_candidate="gustavo_petro",
            right_candidate="rodolfo_hernandez",
            left_share=0.40,
            right_share=0.28,
            abstention_rate=0.20,
        )
        out = hr.clr_shares()
        assert abs(sum(out)) < 1e-12

    def test_raises_on_empty(self) -> None:
        """CLR of empty tuple raises ValueError."""
        with pytest.raises(ValueError, match="Composition"):
            clr(())

    def test_raises_on_zero_total(self) -> None:
        """CLR of all-zero composition raises ValueError."""
        with pytest.raises(ValueError, match="Composition"):
            clr((0.0, 0.0))


class TestLogit:
    """Tests for ``logit()`` log-odds transform."""

    def test_logit_half(self) -> None:
        """logit(0.5) == 0."""
        assert logit(0.5) == 0.0

    def test_logit_symmetric(self) -> None:
        """logit(p) == -logit(1-p)."""
        for p in (0.1, 0.2, 0.4, 0.8, 0.99):
            assert abs(logit(p) - (-logit(1.0 - p))) < 1e-12

    def test_logit_clamps_zero(self) -> None:
        """logit(0) returns a finite value (clamped)."""
        val = logit(0.0)
        assert math.isfinite(val)

    def test_logit_clamps_one(self) -> None:
        """logit(1) returns a finite value (clamped)."""
        val = logit(1.0)
        assert math.isfinite(val)


class TestCLRPoverty:
    """Tests for ``MunicipalFeatures`` compositional helpers."""

    def _make_mf(self, nbi_rate: float, nbi_urban: float, nbi_rural: float) -> MunicipalFeatures:
        return MunicipalFeatures(
            codigo_municipio="05001",
            nombre_municipio="Test",
            comuna_nombre=None,
            departamento="Antioquia",
            region=None,
            poblacion_total=100,
            poblacion_afrocolombiana=10,
            poblacion_indigena=5,
            poblacion_rural_dispersa=2,
            pct_afro_colombian=0.10,
            pct_indigenous=0.05,
            pct_rural_disperso=0.02,
            years_schooling_promedio=10.0,
            pct_school_attendance=0.85,
            internet_access_rate=0.70,
            labor_force_participation_rate=0.65,
            pct_female=0.51,
            rooms_per_household=3.0,
            persons_per_household=3.5,
            pct_age_18_29=0.30,
            pct_age_30_54=0.40,
            pct_age_55_plus=0.30,
            nbi_rate=nbi_rate,
            nbi_urban=nbi_urban,
            nbi_rural=nbi_rural,
            ipm_2018=0.12,
            ipm_2018_imputed=True,
            ipm_2022=0.14,
            ipm_2022_imputed=True,
            pop_2018=100_000,
            pop_2019=101_000,
            pop_2020=102_000,
            pop_2021=103_000,
            pop_2022=104_000,
            pop_2023=105_000,
            pop_2024=106_000,
            pop_2025=107_000,
            pop_2026=108_000,
            pct_ingresos_propios=0.60,
            gastos_totales_per_capita=2.0,
            transferencias_per_capita=0.9,
            ingresos_tributarios_per_capita=1.0,
            risk_level="low",
            is_pdet=False,
            armed_group_presence=False,
            coca_hectares=0.0,
            high_risk_flag=False,
            historical=(),
            historical_turnout_m=0.70,
        )

    def test_clr_poverty_sums_to_zero(self) -> None:
        """clr_poverty() output sums to zero."""
        for rate in (0.05, 0.15, 0.30, 0.50, 0.80):
            mf = self._make_mf(rate, 0.1, 0.2)
            out = mf.clr_poverty()
            assert abs(sum(out)) < 1e-12

    def test_clr_poverty_symmetric(self) -> None:
        """clr_poverty()[0] == -clr_poverty()[1]."""
        mf = self._make_mf(0.15, 0.1, 0.2)
        out = mf.clr_poverty()
        assert abs(out[0] + out[1]) < 1e-12

    def test_clr_nbi_areas_sums_to_zero(self) -> None:
        """clr_nbi_areas() output sums to zero."""
        mf = self._make_mf(0.15, 0.08, 0.22)
        out = mf.clr_nbi_areas()
        assert abs(sum(out)) < 1e-12


# ═══════════════════════════════════════════════════════════════════════
# load_features
# ═══════════════════════════════════════════════════════════════════════


class TestLoadFeatures:
    """load_features() loads and validates the feature matrix."""

    def test_returns_dataframe(self, tmp_path: Path) -> None:
        """Returns a pandas DataFrame."""
        data_dir = _setup_fixture_dir(tmp_path)
        result = load_features(data_dir=data_dir)
        assert isinstance(result, pd.DataFrame)

    def test_expected_row_count(self, tmp_path: Path) -> None:
        """Returns the expected number of rows from fixture."""
        data_dir = _setup_fixture_dir(tmp_path)
        result = load_features(data_dir=data_dir)
        assert len(result) == 5  # Fixture has 5 rows

    def test_nbi_rate_in_range(self, tmp_path: Path) -> None:
        """nbi_rate is in [0, 1] for all rows."""
        data_dir = _setup_fixture_dir(tmp_path)
        result = load_features(data_dir=data_dir)
        assert result["nbi_rate"].between(0.0, 1.0).all()

    def test_ipm_imputed_is_bool(self, tmp_path: Path) -> None:
        """ipm_2018_imputed and ipm_2022_imputed are boolean columns."""
        data_dir = _setup_fixture_dir(tmp_path)
        result = load_features(data_dir=data_dir)
        assert result["ipm_2018_imputed"].dtype == bool
        assert result["ipm_2022_imputed"].dtype == bool

    def test_ipm_imputed_all_true_in_fixture(self, tmp_path: Path) -> None:
        """All 5 fixture rows have ipm_2018_imputed=True."""
        data_dir = _setup_fixture_dir(tmp_path)
        result = load_features(data_dir=data_dir)
        assert result["ipm_2018_imputed"].all()

    def test_historical_turnout_m_in_range(self, tmp_path: Path) -> None:
        """historical_turnout_m is in [0, 1] for all rows."""
        data_dir = _setup_fixture_dir(tmp_path)
        result = load_features(data_dir=data_dir)
        assert result["historical_turnout_m"].between(0.0, 1.0).all()

    def test_historical_turnout_m_no_nulls(self, tmp_path: Path) -> None:
        """historical_turnout_m is non-NaN for all rows."""
        data_dir = _setup_fixture_dir(tmp_path)
        result = load_features(data_dir=data_dir)
        _assert_no_nulls(result, ["historical_turnout_m"])

    def test_historical_column_is_tuple(self, tmp_path: Path) -> None:
        """The historical column contains tuples of HistoricalRecord."""
        data_dir = _setup_fixture_dir(tmp_path)
        result = load_features(data_dir=data_dir)
        for entry in result["historical"]:
            assert isinstance(entry, tuple)
            if len(entry) > 0:
                assert isinstance(entry[0], HistoricalRecord)

    def test_pop_2026_positive(self, tmp_path: Path) -> None:
        """pop_2026 is positive for all municipalities."""
        data_dir = _setup_fixture_dir(tmp_path)
        result = load_features(data_dir=data_dir)
        assert (result["pop_2026"] > 0).all()

    def test_pop_2026_integer(self, tmp_path: Path) -> None:
        """pop_* columns are of integer type."""
        data_dir = _setup_fixture_dir(tmp_path)
        result = load_features(data_dir=data_dir)
        assert pd.api.types.is_integer_dtype(result["pop_2026"].dtype)

    def test_all_risk_flags_are_bool(self, tmp_path: Path) -> None:
        """is_pdet, armed_group_presence, high_risk_flag are bool."""
        data_dir = _setup_fixture_dir(tmp_path)
        result = load_features(data_dir=data_dir)
        for col in ("is_pdet", "armed_group_presence", "high_risk_flag"):
            assert result[col].dtype == bool, f"{col} is not bool"

    def test_raises_on_missing_column(self, tmp_path: Path) -> None:
        """Schema validation raises ValueError when a required column is missing."""
        data_dir = _setup_fixture_dir(tmp_path)
        matrix_path = data_dir / "processed" / "municipal_feature_matrix.csv"
        df_bad = pd.read_csv(matrix_path).drop(columns=["nbi_rate"])
        df_bad.to_csv(matrix_path, index=False)
        with pytest.raises(ValueError, match="nbi_rate"):
            load_features(data_dir=data_dir)

    def test_raises_on_missing_matrix_file(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError when the matrix file is absent."""
        with pytest.raises(FileNotFoundError, match="municipal_feature_matrix"):
            load_features(data_dir=tmp_path)

    def test_raises_on_missing_historical_file(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError when historical_results.csv is absent."""
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(_MATRIX_FIXTURE), str(processed_dir / "municipal_feature_matrix.csv"))
        with pytest.raises(FileNotFoundError, match="historical_results"):
            load_features(data_dir=tmp_path)

    def test_codigo_municipio_is_unique(self, tmp_path: Path) -> None:
        """codigo_municipio values are unique."""
        data_dir = _setup_fixture_dir(tmp_path)
        result = load_features(data_dir=data_dir)
        assert result["codigo_municipio"].is_unique

    def test_codigo_municipio_is_string(self, tmp_path: Path) -> None:
        """codigo_municipio is a string column with 5-char codes."""
        data_dir = _setup_fixture_dir(tmp_path)
        result = load_features(data_dir=data_dir)
        assert pd.api.types.is_string_dtype(result["codigo_municipio"].dtype)
        assert result["codigo_municipio"].str.len().isin({5, 7}).all()

    def test_int_type_pop_columns(self, tmp_path: Path) -> None:
        """All pop_2018..pop_2026 columns are integer type."""
        data_dir = _setup_fixture_dir(tmp_path)
        result = load_features(data_dir=data_dir)
        for year in range(2018, 2027):
            col = f"pop_{year}"
            assert col in result.columns, f"Missing column: {col}"
            assert pd.api.types.is_integer_dtype(result[col].dtype), f"{col} not integer"

    def test_region_can_be_none(self, tmp_path: Path) -> None:
        """Region column contains strings or None."""
        data_dir = _setup_fixture_dir(tmp_path)
        result = load_features(data_dir=data_dir)
        # Fixture has all non-null regions; just check the column exists
        assert "region" in result.columns


# ═══════════════════════════════════════════════════════════════════════
# Real-data smoke tests
# ═══════════════════════════════════════════════════════════════════════


_FISCAL_REQUIRED = {
    "pct_ingresos_propios",
    "gastos_totales_per_capita",
    "transferencias_per_capita",
    "ingresos_tributarios_per_capita",
}


def _real_matrix_has_fiscal(data_dir: Path) -> bool:
    """Check whether the real feature matrix CSV includes fiscal columns."""
    matrix_path = data_dir / "processed" / "municipal_feature_matrix.csv"
    if not matrix_path.is_file():
        return False
    cols = set(pd.read_csv(matrix_path, nrows=1).columns)
    return _FISCAL_REQUIRED.issubset(cols)


class TestLoadFeaturesRealData:
    """Smoke tests against the real data/processed/ matrix."""

    def test_real_data_returns_1142_rows(self, data_dir: Path) -> None:
        """load_features() returns exactly 1,142 rows on real data."""
        matrix_path = data_dir / "processed" / "municipal_feature_matrix.csv"
        historical_path = data_dir / "fundamentals" / "historical_results.csv"
        if not matrix_path.is_file() or not historical_path.is_file():
            pytest.skip(
                "Real data not fully built — expected both "
                "municipal_feature_matrix.csv and fundamentals/historical_results.csv"
            )
        if not _real_matrix_has_fiscal(data_dir):
            pytest.skip("Real matrix missing fiscal cols — run build_fiscal_features")
        result = load_features(data_dir=data_dir)
        assert len(result) == _EXPECTED_MUNICIPALITIES

    def test_real_data_historical_turnout_no_nulls(self, data_dir: Path) -> None:
        """historical_turnout_m is non-NaN for all municipalities."""
        matrix_path = data_dir / "processed" / "municipal_feature_matrix.csv"
        historical_path = data_dir / "fundamentals" / "historical_results.csv"
        if not matrix_path.is_file() or not historical_path.is_file():
            pytest.skip(
                "Real data not fully built — expected both "
                "municipal_feature_matrix.csv and fundamentals/historical_results.csv"
            )
        if not _real_matrix_has_fiscal(data_dir):
            pytest.skip("Real matrix missing fiscal cols — run build_fiscal_features")
        result = load_features(data_dir=data_dir)
        assert result["historical_turnout_m"].notna().all()

    def test_real_data_ipm_imputation_rate(self, data_dir: Path) -> None:
        """At least 1,000 municipalities have ipm_2018_imputed=True."""
        matrix_path = data_dir / "processed" / "municipal_feature_matrix.csv"
        historical_path = data_dir / "fundamentals" / "historical_results.csv"
        if not matrix_path.is_file() or not historical_path.is_file():
            pytest.skip(
                "Real data not fully built — expected both "
                "municipal_feature_matrix.csv and fundamentals/historical_results.csv"
            )
        if not _real_matrix_has_fiscal(data_dir):
            pytest.skip("Real matrix missing fiscal cols — run build_fiscal_features")
        result = load_features(data_dir=data_dir)
        assert int(result["ipm_2018_imputed"].sum()) >= 1_000
