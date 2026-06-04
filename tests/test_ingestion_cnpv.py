"""SPEC-17: Tests for CNPV 2018 census microdata ingestion.

Covers the chunked aggregation kernel (F11/F8/F9), percentage computation,
validation, and a slow integration test against the full build.
"""

from __future__ import annotations

import dataclasses
from io import BytesIO
from pathlib import Path
import re
from typing import get_type_hints
import zipfile

import numpy as np
import pandas as pd
import pytest

from co_president.ingestion.ingest_cnpv import (
    _aggregate_chunk_f8,
    _aggregate_chunk_f9,
    _aggregate_chunk_f11,
    _compute_percentages,
    _zero_pad_dane_code,
    iter_cnpv_zip,
    validate_cnpv,
)


@dataclasses.dataclass(frozen=True)
class _F11Overrides:
    """Row-level override configuration for synthetic F11 test data.

    Each field maps to a census variable; ``None`` means use defaults.
    """

    pa1_grp_etnic: dict[int, int] | None = None
    ua_clase: dict[int, int] | None = None
    anosr: dict[int, int] | None = None
    asistencia: dict[int, int] | None = None
    trabajo: dict[int, int] | None = None
    sexo: dict[int, int] | None = None
    edadr: dict[int, int] | None = None
    mpio: dict[int, str] | None = None


# ═══════════════════════════════════════════════════════════════════════
# Structural contract
# ═══════════════════════════════════════════════════════════════════════


class TestAggregateChunkF11:
    """Structural contract for ``_aggregate_chunk_f11``."""

    def test_is_callable(self) -> None:
        """Function is importable and callable."""
        assert callable(_aggregate_chunk_f11)

    def test_return_annotation_is_dataframe(self) -> None:
        """Return type annotation is ``pd.DataFrame``."""
        hints = get_type_hints(_aggregate_chunk_f11)
        assert hints["return"] is pd.DataFrame


class TestAggregateChunkF8:
    """Structural contract for ``_aggregate_chunk_f8``."""

    def test_is_callable(self) -> None:
        """Function is importable and callable."""
        assert callable(_aggregate_chunk_f8)

    def test_return_annotation_is_dataframe(self) -> None:
        """Return type annotation is ``pd.DataFrame``."""
        hints = get_type_hints(_aggregate_chunk_f8)
        assert hints["return"] is pd.DataFrame


class TestAggregateChunkF9:
    """Structural contract for ``_aggregate_chunk_f9``."""

    def test_is_callable(self) -> None:
        """Function is importable and callable."""
        assert callable(_aggregate_chunk_f9)

    def test_return_annotation_is_dataframe(self) -> None:
        """Return type annotation is ``pd.DataFrame``."""
        hints = get_type_hints(_aggregate_chunk_f9)
        assert hints["return"] is pd.DataFrame


# ═══════════════════════════════════════════════════════════════════════
# Pure helper: _zero_pad_dane_code
# ═══════════════════════════════════════════════════════════════════════


class TestZeroPadDaneCode:
    """``_zero_pad_dane_code`` produces correct 5-char DANE codes."""

    def test_standard_two_digit_dept(self) -> None:
        """Standard 2-digit department + 3-digit municipality."""
        assert _zero_pad_dane_code("05", "001") == "05001"

    def test_single_digit_dept(self) -> None:
        """Single-digit department gets zero-padded."""
        assert _zero_pad_dane_code("5", "1") == "05001"

    def test_with_whitespace(self) -> None:
        """Leading/trailing whitespace is stripped."""
        assert _zero_pad_dane_code(" 05 ", " 001 ") == "05001"

    def test_integer_inputs(self) -> None:
        """Integer inputs are converted to strings."""
        assert _zero_pad_dane_code(5, 1) == "05001"  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════
# Aggregation kernel: F11 (personas)
# ═══════════════════════════════════════════════════════════════════════


def _reset_idx(out: pd.DataFrame) -> pd.DataFrame:
    """Reset index so tests can assert on columns."""
    return out.reset_index()


class TestF11Aggregation:
    """``_aggregate_chunk_f11`` produces correct counts from synthetic data."""

    def test_total_population_equals_row_count(self) -> None:
        """Total population equals input row count."""
        chunk = _synthetic_f11(rows=5)
        out = _reset_idx(_aggregate_chunk_f11(chunk))
        assert out["poblacion_total"].sum() == 5

    def test_known_ethnicity_counts(self) -> None:
        """Ethnicity counts are correct for indigenous and afro."""
        chunk = _synthetic_f11(
            rows=5,
            overrides=_F11Overrides(pa1_grp_etnic={0: 1, 1: 1, 2: 5, 3: 5, 4: 9}),
        )
        out = _reset_idx(_aggregate_chunk_f11(chunk))
        assert out["poblacion_indigena"].sum() == 2
        assert out["poblacion_afrocolombiana"].sum() == 2

    def test_unknown_ethnicity_excluded(self) -> None:
        """Ethnicity 9 (No Informa) is excluded from counts."""
        chunk = _synthetic_f11(
            rows=4,
            overrides=_F11Overrides(pa1_grp_etnic={0: 9, 1: 9, 2: 9, 3: 9}),
        )
        out = _reset_idx(_aggregate_chunk_f11(chunk))
        assert out["poblacion_indigena"].sum() == 0
        assert out["poblacion_afrocolombiana"].sum() == 0

    def test_rural_disperso_count(self) -> None:
        """Rural disperso count matches UA_CLASE==3 rows."""
        chunk = _synthetic_f11(rows=5, overrides=_F11Overrides(ua_clase={0: 3, 2: 3}))
        out = _reset_idx(_aggregate_chunk_f11(chunk))
        assert out["poblacion_rural_dispersa"].sum() == 2

    def test_years_schooling_sum(self) -> None:
        """Years schooling sum matches input values."""
        chunk = _synthetic_f11(
            rows=5,
            overrides=_F11Overrides(anosr={0: 10, 1: 10, 2: 10, 3: 10, 4: 10}),
        )
        out = _reset_idx(_aggregate_chunk_f11(chunk))
        assert out["years_schooling_sum"].sum() == 50
        assert out["years_schooling_count"].sum() == 5

    def test_excludes_years_schooling_99(self) -> None:
        """P_NIVEL_ANOSR == 99 is excluded from sum and count."""
        chunk = _synthetic_f11(
            rows=4,
            overrides=_F11Overrides(anosr={0: 99, 1: 99, 2: 99, 3: 99}),
        )
        out = _reset_idx(_aggregate_chunk_f11(chunk))
        assert out["years_schooling_sum"].sum() == 0.0
        assert out["years_schooling_count"].sum() == 0

    def test_school_attendance_excludes_no_aplica(self) -> None:
        """PA_ASISTENCIA 4 (No Aplica) and 9 (No Informa) are excluded from denominator."""
        chunk = _synthetic_f11(
            rows=5,
            overrides=_F11Overrides(asistencia={0: 1, 1: 1, 2: 2, 3: 4, 4: 9}),
        )
        out = _reset_idx(_aggregate_chunk_f11(chunk))
        # Valid values: 1 (yes) and 2 (no); 4 and 9 excluded
        assert out["school_attendance_yes"].sum() == 2
        assert out["school_attendance_denom"].sum() == 3

    def test_labor_force_active_count(self) -> None:
        """Labour force counts only codes 1-4 as active."""
        chunk = _synthetic_f11(
            rows=5,
            overrides=_F11Overrides(trabajo={0: 1, 1: 1, 2: 5, 3: 5, 4: 9}),
        )
        out = _reset_idx(_aggregate_chunk_f11(chunk))
        assert out["labor_force_active"].sum() == 2
        # Denom excludes code 9 (No Informa), so 4 valid rows remain.
        assert out["labor_force_denom"].sum() == 4

    def test_female_count(self) -> None:
        """Female count matches P_SEXO==2 rows."""
        chunk = _synthetic_f11(
            rows=5,
            overrides=_F11Overrides(sexo={0: 2, 1: 2, 2: 2, 3: 1, 4: 1}),
        )
        out = _reset_idx(_aggregate_chunk_f11(chunk))
        assert out["female_count"].sum() == 3
        assert out["poblacion_total"].sum() == 5

    def test_age_bins(self) -> None:
        """Age bins correctly categorise P_EDADR codes."""
        chunk = _synthetic_f11(
            rows=5,
            overrides=_F11Overrides(
                edadr={
                    0: 5,  # 20-24 → age_18_29
                    1: 5,  # 20-24 → age_18_29
                    2: 10,  # 45-49 → age_30_54
                    3: 15,  # 70-74 → age_55_plus
                    4: 15,  # 70-74 → age_55_plus
                }
            ),
        )
        out = _reset_idx(_aggregate_chunk_f11(chunk))
        assert out["age_18_29"].sum() == 2
        assert out["age_30_54"].sum() == 1
        assert out["age_55_plus"].sum() == 2

    def test_handles_zero_edadr(self) -> None:
        """P_EDADR == 0 is excluded from age bin counts."""
        chunk = _synthetic_f11(
            rows=4,
            overrides=_F11Overrides(edadr={0: 0, 1: 0, 2: 0, 3: 0}),
        )
        out = _reset_idx(_aggregate_chunk_f11(chunk))
        assert out["age_18_29"].sum() == 0
        assert out["age_30_54"].sum() == 0
        assert out["age_55_plus"].sum() == 0
        assert out["poblacion_total"].sum() == 4

    def test_two_municipalities(self) -> None:
        """Chunk with two mpio codes produces two aggregate rows."""
        chunk = _synthetic_f11(
            rows=5,
            overrides=_F11Overrides(mpio={0: "002", 1: "002", 2: "002"}),
        )
        out = _reset_idx(_aggregate_chunk_f11(chunk))
        assert len(out) == 2
        by_code = out.set_index("codigo_municipio")["poblacion_total"]
        assert by_code["05001"] == 2
        assert by_code["05002"] == 3


# ═══════════════════════════════════════════════════════════════════════
# Aggregation kernel: F8 (viviendas)
# ═══════════════════════════════════════════════════════════════════════


class TestF8Aggregation:
    """``_aggregate_chunk_f8`` produces correct internet counts."""

    def test_internet_yes_count(self) -> None:
        """Internet yes count matches VF_INTERNET==1 rows."""
        chunk = _synthetic_f8(
            rows=5,
            internet_overrides={0: 1, 1: 1, 2: 1},
        )
        out = _reset_idx(_aggregate_chunk_f8(chunk))
        assert out["internet_yes"].sum() == 3
        assert out["internet_total"].sum() == 5

    def test_all_no_internet(self) -> None:
        """All-no-internet chunk yields zero yes counts."""
        chunk = _synthetic_f8(
            rows=4,
            internet_overrides={0: 2, 1: 2, 2: 2, 3: 2},
        )
        out = _reset_idx(_aggregate_chunk_f8(chunk))
        assert out["internet_yes"].sum() == 0
        assert out["internet_total"].sum() == 4


# ═══════════════════════════════════════════════════════════════════════
# Aggregation kernel: F9 (hogares)
# ═══════════════════════════════════════════════════════════════════════


class TestF9Aggregation:
    """``_aggregate_chunk_f9`` produces correct room and hh-size counts."""

    def test_rooms_sum(self) -> None:
        """Rooms sum matches H_NRO_DORMIT values."""
        chunk = _synthetic_f9(
            rows=5,
            nro_dormit_overrides={0: 3, 1: 3, 2: 3, 3: 3, 4: 3},
        )
        out = _reset_idx(_aggregate_chunk_f9(chunk))
        assert out["rooms_sum"].sum() == 15
        assert out["rooms_count"].sum() == 5

    def test_rooms_excludes_99(self) -> None:
        """H_NRO_DORMIT == 99 is excluded."""
        chunk = _synthetic_f9(
            rows=4,
            nro_dormit_overrides={0: 99, 1: 99, 2: 99, 3: 99},
        )
        out = _reset_idx(_aggregate_chunk_f9(chunk))
        assert out["rooms_sum"].sum() == 0.0
        assert out["rooms_count"].sum() == 0

    def test_persons_per_hh_sum(self) -> None:
        """Persons-per-hh sum matches HA_TOT_PER values."""
        chunk = _synthetic_f9(
            rows=5,
            ha_tot_per_overrides={0: 4, 1: 4, 2: 4, 3: 4, 4: 4},
        )
        out = _reset_idx(_aggregate_chunk_f9(chunk))
        assert out["persons_per_hh_sum"].sum() == 20
        assert out["persons_per_hh_count"].sum() == 5


# ═══════════════════════════════════════════════════════════════════════
# Percentage computation
# ═══════════════════════════════════════════════════════════════════════


class TestComputePercentages:
    """``_compute_percentages`` adds correct derived columns."""

    def test_pct_columns_in_unit_interval(self) -> None:
        """All pct and rate columns are in [0, 1]."""
        merged = _synthetic_merged_raw(rows=5)
        out = _compute_percentages(merged.copy())
        pct_cols = [c for c in out.columns if c.endswith("_rate") or c.startswith("pct_")]
        assert len(pct_cols) > 0
        for c in pct_cols:
            vals = out[c].dropna()
            assert vals.between(0, 1).all()

    def test_ethnic_percentages_are_correct(self) -> None:
        """Ethnic percentages match input ratios."""
        f11_chunk = _synthetic_f11(
            rows=5,
            overrides=_F11Overrides(pa1_grp_etnic={0: 1, 1: 1, 2: 5, 3: 6, 4: 6}),
        )
        f11_raw = _aggregate_chunk_f11(f11_chunk).copy()
        f8_raw = _aggregate_chunk_f8(_synthetic_f8(rows=5)).copy()
        f9_raw = _aggregate_chunk_f9(_synthetic_f9(rows=5)).copy()
        merged = _join_raw_parts(f11_raw, f8_raw, f9_raw)
        out = _compute_percentages(merged)
        assert out["pct_indigenous"].iloc[0] == pytest.approx(2 / 5)
        assert out["pct_afro_colombian"].iloc[0] == pytest.approx(1 / 5)

    def test_internet_rate(self) -> None:
        """Internet access rate matches yes / total."""
        f8_chunk = _synthetic_f8(
            rows=5,
            internet_overrides={0: 1, 1: 1, 2: 1, 3: 1},
        )
        f8_raw = _aggregate_chunk_f8(f8_chunk).copy()
        f11_raw = _aggregate_chunk_f11(_synthetic_f11(rows=5)).copy()
        f9_raw = _aggregate_chunk_f9(_synthetic_f9(rows=5)).copy()
        merged = _join_raw_parts(f11_raw, f8_raw, f9_raw)
        out = _compute_percentages(merged)
        assert out["internet_access_rate"].iloc[0] == pytest.approx(0.8)

    def test_school_attendance_rate(self) -> None:
        """School attendance rate matches attending / attending+not."""
        f11_chunk = _synthetic_f11(
            rows=5,
            overrides=_F11Overrides(asistencia={0: 1, 1: 1, 2: 2, 3: 4, 4: 9}),
        )
        f11_raw = _aggregate_chunk_f11(f11_chunk).copy()
        f8_raw = _aggregate_chunk_f8(_synthetic_f8(rows=5)).copy()
        f9_raw = _aggregate_chunk_f9(_synthetic_f9(rows=5)).copy()
        merged = _join_raw_parts(f11_raw, f8_raw, f9_raw)
        out = _compute_percentages(merged)
        # Valid: 1 (yes) and 2 (no); 4 and 9 excluded → rate = 2/3
        assert out["pct_school_attendance"].iloc[0] == pytest.approx(2 / 3)

    def test_years_schooling_mean(self) -> None:
        """Mean years of schooling matches the input average."""
        f11_chunk = _synthetic_f11(
            rows=5,
            overrides=_F11Overrides(anosr={0: 0, 1: 2, 2: 4, 3: 6, 4: 8}),
        )
        f11_raw = _aggregate_chunk_f11(f11_chunk).copy()
        f8_raw = _aggregate_chunk_f8(_synthetic_f8(rows=5)).copy()
        f9_raw = _aggregate_chunk_f9(_synthetic_f9(rows=5)).copy()
        merged = _join_raw_parts(f11_raw, f8_raw, f9_raw)
        out = _compute_percentages(merged)
        expected = (0 + 2 + 4 + 6 + 8) / 5  # = 4.0
        assert out["years_schooling_promedio"].iloc[0] == pytest.approx(expected)

    def test_rooms_per_household(self) -> None:
        """Mean rooms per household matches input average."""
        f9_chunk = _synthetic_f9(
            rows=5,
            nro_dormit_overrides={0: 1, 1: 2, 2: 3, 3: 4, 4: 1},
        )
        f9_raw = _aggregate_chunk_f9(f9_chunk).copy()
        f11_raw = _aggregate_chunk_f11(_synthetic_f11(rows=5)).copy()
        f8_raw = _aggregate_chunk_f8(_synthetic_f8(rows=5)).copy()
        merged = _join_raw_parts(f11_raw, f8_raw, f9_raw)
        out = _compute_percentages(merged)
        expected = (1 + 2 + 3 + 4 + 1) / 5  # = 2.2
        assert out["rooms_per_household"].iloc[0] == pytest.approx(expected)

    def test_zero_division_returns_nan(self) -> None:
        """Division-by-zero yields NaN for percentage columns."""
        raw = pd.DataFrame(
            {
                "poblacion_total": [0],
                "poblacion_indigena": [0],
                "poblacion_afrocolombiana": [0],
                "poblacion_rural_dispersa": [0],
                "years_schooling_sum": [0.0],
                "years_schooling_count": [0],
                "school_attendance_yes": [0],
                "school_attendance_denom": [0],
                "labor_force_active": [0],
                "labor_force_denom": [0],
                "female_count": [0],
                "internet_yes": [0],
                "internet_total": [0],
                "rooms_sum": [0.0],
                "rooms_count": [0],
                "persons_per_hh_sum": [0.0],
                "persons_per_hh_count": [0],
                "age_18_29": [0],
                "age_30_54": [0],
                "age_55_plus": [0],
            }
        )
        out = _compute_percentages(raw)
        for col in out.columns:
            if col.endswith(("_rate", "_promedio")) or col.startswith("pct_"):
                assert pd.isna(out[col].iloc[0]), f"{col} should be NaN"


# ═══════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════


class TestValidation:
    """``validate_cnpv`` checks DataFrame integrity."""

    def test_valid_dataframe_returns_empty(self) -> None:
        """Valid full DataFrame returns empty warning list."""
        df = pd.DataFrame(
            {
                "codigo_municipio": ["05001"],
                "poblacion_total": [1000],
                "poblacion_afrocolombiana": [100],
                "poblacion_indigena": [100],
                "poblacion_rural_dispersa": [50],
                "pct_afro_colombian": [0.1],
                "pct_indigenous": [0.1],
                "pct_rural_disperso": [0.05],
                "years_schooling_promedio": [8.5],
                "pct_school_attendance": [0.8],
                "internet_access_rate": [0.6],
                "labor_force_participation_rate": [0.7],
                "pct_female": [0.52],
                "rooms_per_household": [3.2],
                "persons_per_household": [3.8],
                "pct_age_18_29": [0.25],
                "pct_age_30_54": [0.40],
                "pct_age_55_plus": [0.35],
            }
        )
        assert validate_cnpv(df) == []

    def test_missing_columns_returns_warning(self) -> None:
        """Missing columns produce a warning."""
        df = pd.DataFrame({"codigo_municipio": ["05001"]})
        warnings = validate_cnpv(df)
        assert any("Missing columns" in w for w in warnings)

    def test_null_codigo_municipio_returns_warning(self) -> None:
        """Null codigo_municipio values produce a warning."""
        df = pd.DataFrame(
            {
                "codigo_municipio": ["05001", None],
                "poblacion_total": [1000, 500],
                "poblacion_afrocolombiana": [0, 0],
                "poblacion_indigena": [0, 0],
                "poblacion_rural_dispersa": [0, 0],
                "pct_afro_colombian": [0.0, 0.0],
                "pct_indigenous": [0.0, 0.0],
                "pct_rural_disperso": [0.0, 0.0],
                "years_schooling_promedio": [8.0, np.nan],
                "pct_school_attendance": [0.8, 0.7],
                "internet_access_rate": [0.6, 0.5],
                "labor_force_participation_rate": [0.7, 0.6],
                "pct_female": [0.52, 0.50],
                "rooms_per_household": [3.2, 3.0],
                "persons_per_household": [3.8, 4.0],
                "pct_age_18_29": [0.25, 0.20],
                "pct_age_30_54": [0.40, 0.45],
                "pct_age_55_plus": [0.35, 0.35],
            }
        )
        warnings = validate_cnpv(df)
        assert any("null" in w for w in warnings)


# ═══════════════════════════════════════════════════════════════════════
# Iterator section
# ═══════════════════════════════════════════════════════════════════════


def _create_synthetic_zip(
    tmp_path: Path,
    dept_code: str = "99",
    dept_name: str = "TestDept",
    num_rows: int = 10,
) -> Path:
    """Create a synthetic nested zip structure mimicking CNPV departmental data.

    Creates an outer zip containing an inner ``*_CSV.zip`` with three CSVs
    (F8, F9, F11) with minimal synthetic data.
    """
    outer_zip_path = tmp_path / f"{dept_code}_{dept_name}.zip"
    inner_zip_name = f"{dept_code}_{dept_name}_CSV.zip"

    # F11 (personas) CSV
    f11_header = (
        "U_DPTO,U_MPIO,UA_CLASE,PA1_GRP_ETNIC,P_NIVEL_ANOSR,"
        "P_EDADR,PA_ASISTENCIA,P_TRABAJO,P_SEXO\n"
    )
    f11_rows = "".join(f"{dept_code},001,1,6,10,8,1,1,2\n" for _ in range(num_rows))

    # F8 (viviendas) CSV
    f8_header = "U_DPTO,U_MPIO,VF_INTERNET\n"
    f8_rows = "".join(f"{dept_code},001,1\n" for _ in range(num_rows))

    # F9 (hogares) CSV
    f9_header = "U_DPTO,U_MPIO,H_NRO_DORMIT,HA_TOT_PER\n"
    f9_rows = "".join(f"{dept_code},001,3,4\n" for _ in range(num_rows))

    # Create inner zip
    inner_buf = BytesIO()
    with zipfile.ZipFile(inner_buf, "w", zipfile.ZIP_DEFLATED) as inner_zf:
        inner_zf.writestr(f"CNPV2018_5PER_A2_{dept_code}.CSV", f11_header + f11_rows)
        inner_zf.writestr(f"CNPV2018_1VIV_A2_{dept_code}.CSV", f8_header + f8_rows)
        inner_zf.writestr(f"CNPV2018_2HOG_A2_{dept_code}.CSV", f9_header + f9_rows)

    # Create outer zip
    with zipfile.ZipFile(outer_zip_path, "w", zipfile.ZIP_DEFLATED) as outer_zf:
        outer_zf.writestr(inner_zip_name, inner_buf.getvalue())

    return outer_zip_path


class TestIterCnpvZip:
    """``iter_cnpv_zip`` streams aggregated chunks from nested zips."""

    def test_yields_aggregated_chunks(self, tmp_path: Path) -> None:
        """iter_cnpv_zip yields DataFrames with correct municipio index."""
        zip_path = _create_synthetic_zip(tmp_path, num_rows=10)
        chunks = list(iter_cnpv_zip(zip_path))
        # Three kinds (F11, F8, F9), each yields one chunk.
        assert len(chunks) == 3
        for chunk in chunks:
            assert "99001" in chunk.index

    def test_f11_chunk_has_population_count(self, tmp_path: Path) -> None:
        """F11 chunk includes poblacion_total matching row count."""
        zip_path = _create_synthetic_zip(tmp_path, num_rows=25)
        chunks = list(iter_cnpv_zip(zip_path))
        f11_chunk = chunks[0]  # F11 is yielded first
        assert f11_chunk.loc["99001", "poblacion_total"] == 25

    def test_raises_on_missing_inner_zip(self, tmp_path: Path) -> None:
        """Missing inner _CSV.zip raises FileNotFoundError."""
        bad_zip = tmp_path / "99_Bad.zip"
        with zipfile.ZipFile(bad_zip, "w") as zf:
            zf.writestr("not_a_csv.zip", b"dummy")
        with pytest.raises(FileNotFoundError, match=re.escape("No inner _CSV.zip")):
            list(iter_cnpv_zip(bad_zip))


# ═══════════════════════════════════════════════════════════════════════
# Issue #206 core tests
# ═══════════════════════════════════════════════════════════════════════


class TestIssue206Core:
    """Six core tests from Issue #206 (excluding slow Bogotá anchor)."""

    def test_column_extraction_is_lossless(self) -> None:
        """Total population sum equals input row count."""
        chunk = _synthetic_f11(rows=5)
        out = _aggregate_chunk_f11(chunk)
        assert out["poblacion_total"].sum() == 5

    def test_uses_zero_padded_dane_codes(self) -> None:
        """DANE codes are correctly zero-padded to 5 chars."""
        chunk = _synthetic_f11(rows=3, dpto="5", mpio="001")
        out = _aggregate_chunk_f11(chunk)
        assert list(out.index) == ["05001"]

    def test_handles_unknown_ethnicity_as_zero(self) -> None:
        """PA1_GRP_ETNIC=9 is not counted as indigenous or afro."""
        chunk = _synthetic_f11(
            rows=5,
            overrides=_F11Overrides(pa1_grp_etnic={0: 9, 1: 9, 2: 9, 3: 9, 4: 9}),
        )
        out = _aggregate_chunk_f11(chunk)
        assert out["poblacion_indigena"].sum() == 0
        assert out["poblacion_afrocolombiana"].sum() == 0

    def test_all_municipalities_have_codigo_municipio(self) -> None:
        """All output rows have non-null 5-char codigo_municipio."""
        chunk = _synthetic_f11(rows=4)
        out = _aggregate_chunk_f11(chunk)
        assert not out.index.isna().any()
        assert all(out.index.str.len() == 5)

    def test_pct_columns_in_unit_interval(self) -> None:
        """All pct columns in the merged output are in [0, 1]."""
        merged = _synthetic_merged_raw(rows=5)
        out = _compute_percentages(merged.copy())
        pct_cols = [c for c in out.columns if c.endswith("_rate") or c.startswith("pct_")]
        assert len(pct_cols) > 0
        for c in pct_cols:
            vals = out[c].dropna()
            assert vals.between(0, 1).all()


@pytest.mark.slow
@pytest.mark.integration
class TestBogotaKnownTotals:
    """``test_cnpv_known_municipality_totals`` — Bogotá anchored > 7M."""

    _BOGOTA_FIXTURE: str = "data/fundamentals/cnpv_2018.csv"

    @pytest.mark.skipif(
        not Path("data/fundamentals/cnpv_2018.csv").exists(),
        reason="cnpv_2018.csv not built — run `make cnpv` first",
    )
    def test_bogota_population_exceeds_7m(self) -> None:
        """Bogotá (11001) population is greater than 7 million."""
        df = pd.read_csv(self._BOGOTA_FIXTURE, dtype={"codigo_municipio": str})
        assert df.loc[df.codigo_municipio == "11001", "poblacion_total"].iloc[0] > 7_000_000


# ═══════════════════════════════════════════════════════════════════════
# Synthetic fixture helpers
# ═══════════════════════════════════════════════════════════════════════


def _synthetic_f11(
    rows: int = 5,
    *,
    dpto: str = "05",
    mpio: str = "001",
    overrides: _F11Overrides | None = None,
) -> pd.DataFrame:
    """Build a synthetic F11 (personas) DataFrame for unit testing.

    Default values produce a homogeneous population in a single
    municipality.  Override dicts provide row-level control.
    """
    if overrides is None:
        overrides = _F11Overrides()
    pa1_grp_etnic = _build_col(rows, 6, overrides.pa1_grp_etnic)
    ua_clase = _build_col(rows, 1, overrides.ua_clase)
    p_nivel_anosr = _build_col(rows, 10, overrides.anosr)
    pa_asistencia = _build_col(rows, 1, overrides.asistencia)
    p_trabajo = _build_col(rows, 1, overrides.trabajo)
    p_sexo = _build_col(rows, 1, overrides.sexo)
    p_edadr = _build_col(rows, 8, overrides.edadr)
    u_mpio = [_mpio_override(i, mpio, overrides.mpio) for i in range(rows)]

    return pd.DataFrame(
        {
            "U_DPTO": [dpto] * rows,
            "U_MPIO": u_mpio,
            "UA_CLASE": ua_clase,
            "PA1_GRP_ETNIC": pa1_grp_etnic,
            "P_NIVEL_ANOSR": p_nivel_anosr,
            "P_EDADR": p_edadr,
            "PA_ASISTENCIA": pa_asistencia,
            "P_TRABAJO": p_trabajo,
            "P_SEXO": p_sexo,
        }
    )


def _synthetic_f8(
    rows: int = 5,
    *,
    dpto: str = "05",
    mpio: str = "001",
    internet_overrides: dict[int, int] | None = None,
) -> pd.DataFrame:
    """Build a synthetic F8 (viviendas) DataFrame for unit testing.

    Default VF_INTERNET is 2 (No). Override with 1 (Si) for
    internet-connected dwellings.
    """
    vf_internet = _build_col(rows, 2, internet_overrides)
    return pd.DataFrame(
        {
            "U_DPTO": [dpto] * rows,
            "U_MPIO": [mpio] * rows,
            "VF_INTERNET": vf_internet,
        }
    )


def _synthetic_f9(
    rows: int = 5,
    *,
    dpto: str = "05",
    mpio: str = "001",
    nro_dormit_overrides: dict[int, int] | None = None,
    ha_tot_per_overrides: dict[int, int] | None = None,
) -> pd.DataFrame:
    """Build a synthetic F9 (hogares) DataFrame for unit testing."""
    h_nro_dormit = _build_col(rows, 3, nro_dormit_overrides)
    ha_tot_per = _build_col(rows, 4, ha_tot_per_overrides)
    return pd.DataFrame(
        {
            "U_DPTO": [dpto] * rows,
            "U_MPIO": [mpio] * rows,
            "H_NRO_DORMIT": h_nro_dormit,
            "HA_TOT_PER": ha_tot_per,
        }
    )


def _build_col(
    rows: int,
    default: int,
    overrides: dict[int, int] | None,
) -> list[int]:
    """Build a column with *rows* identical values and optional overrides."""
    col = [default] * rows
    if overrides:
        for idx, val in overrides.items():
            if 0 <= idx < rows:
                col[idx] = val
    return col


def _mpio_override(
    idx: int,
    default: str,
    overrides: dict[int, str] | None,
) -> str:
    """Return the override value for *idx* or the *default* municipality."""
    if overrides is None:
        return default
    return overrides.get(idx, default)


def _join_raw_parts(
    f11: pd.DataFrame,
    f8: pd.DataFrame,
    f9: pd.DataFrame,
) -> pd.DataFrame:
    """Outer-join three per-municipio aggregation outputs on their index."""
    return f11.join(f8, how="outer").join(f9, how="outer").fillna(0)


def _synthetic_merged_raw(
    rows: int = 5,
    *,
    dpto: str = "05",
    mpio: str = "001",
) -> pd.DataFrame:
    """Build a synthetic merged raw-counts DataFrame for percentage tests.

    Creates a single-municipality synthetic dataset from all three
    CNPV CSVs (F11, F8, F9) and returns the joined raw counts.
    """
    f11_raw = _aggregate_chunk_f11(
        _synthetic_f11(rows=rows, dpto=dpto, mpio=mpio),
    ).copy()
    f8_raw = _aggregate_chunk_f8(
        _synthetic_f8(rows=rows, dpto=dpto, mpio=mpio),
    ).copy()
    f9_raw = _aggregate_chunk_f9(
        _synthetic_f9(rows=rows, dpto=dpto, mpio=mpio),
    ).copy()
    return _join_raw_parts(f11_raw, f8_raw, f9_raw)
