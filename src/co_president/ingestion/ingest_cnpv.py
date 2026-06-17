"""SPEC-17: CNPV 2018 census microdata ingestion pipeline.

Reads 33 departmental zip files from ``data/cnpv-2018/raw/``, streaming
each as 200 000-row chunks from three inner CSVs (F8 viviendas, F9 hogares,
F11 personas), aggregates individual-level microdata to municipal-level
demographic features, and writes the result to
``data/fundamentals/cnpv_2018.csv``.

References:
    - ``docs/PLAN_demographic_ingestion.md`` §4.1 — full design
    - ``data/cnpv-2018/docs/CNPV-2018-data-dictionary.ddi.xml`` — variable defs

"""

from __future__ import annotations

from io import BytesIO, TextIOWrapper
import logging
from typing import TYPE_CHECKING
import zipfile

import numpy as np
import pandas as pd

from co_president.config import EXPECTED_MUNICIPALITIES
from co_president.paths import resolve_data_dir

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

__all__ = [
    "build_cnpv_features",
    "iter_cnpv_zip",
    "load_cnpv_data",
    "validate_cnpv",
]

logger = logging.getLogger(__name__)

# Column patterns for the three inner CSVs.  The department code (2-digit)
# is injected via ``.format(dept)``.
_F8_NAME: str = "CNPV2018_1VIV_A2_{}.CSV"
_F9_NAME: str = "CNPV2018_2HOG_A2_{}.CSV"
_F11_NAME: str = "CNPV2018_5PER_A2_{}.CSV"

# Columns to extract from each inner CSV (keeping geo-locators on all).
_F8_COLUMNS: tuple[str, ...] = ("U_DPTO", "U_MPIO", "VF_INTERNET")
_F9_COLUMNS: tuple[str, ...] = ("U_DPTO", "U_MPIO", "H_NRO_DORMIT", "HA_TOT_PER")
_F11_COLUMNS: tuple[str, ...] = (
    "U_DPTO",
    "U_MPIO",
    "UA_CLASE",
    "PA1_GRP_ETNIC",
    "P_NIVEL_ANOSR",
    "P_EDADR",
    "PA_ASISTENCIA",
    "P_TRABAJO",
    "P_SEXO",
)

# Column patterns for the three inner CSVs.  The department code (2-digit)
# is injected via ``.format(dept)``.
_FILE_PATTERNS: dict[str, str] = {
    "F8": _F8_NAME,
    "F9": _F9_NAME,
    "F11": _F11_NAME,
}

_FILE_COLUMNS: dict[str, tuple[str, ...]] = {
    "F8": _F8_COLUMNS,
    "F9": _F9_COLUMNS,
    "F11": _F11_COLUMNS,
}

# All 18 output columns after merge + percentage computation.
# NOTE: Issue #206 specifies 16 demographic columns; we include
# ``poblacion_total`` (17th) as the denominator for all percentages,
# and ``codigo_municipio`` (18th) as the key column.
_EXPECTED_COLUMNS: frozenset[str] = frozenset(
    {
        "codigo_municipio",
        "poblacion_total",
        "poblacion_afrocolombiana",
        "poblacion_indigena",
        "poblacion_rural_dispersa",
        "pct_afro_colombian",
        "pct_indigenous",
        "pct_rural_disperso",
        "years_schooling_promedio",
        "pct_school_attendance",
        "internet_access_rate",
        "labor_force_participation_rate",
        "pct_female",
        "rooms_per_household",
        "persons_per_household",
        "pct_age_18_29",
        "pct_age_30_54",
        "pct_age_55_plus",
    }
)

# P_EDADR codes for age groups (DDI lines 24922-24942).
# NOTE: "age_18_29" uses codes 5-6 (20-24, 25-29) and intentionally
# excludes code 4 (15-19) because the DDI does not provide a finer
# granularity for 18-19 specifically; including code 4 would pull in
# 15-17 year-olds.  The label reflects the closest census-aligned range.
_AGE_BINS: dict[str, frozenset[int]] = {
    "age_18_29": frozenset({5, 6}),  # 20-24, 25-29
    "age_30_54": frozenset({7, 8, 9, 10, 11}),  # 30-34 … 50-54
    "age_55_plus": frozenset(range(12, 22)),  # 55-59 … 100+
}

# Intermediate aggregation columns that must be dropped from the final CSV.
_INTERMEDIATE_COLS: tuple[str, ...] = (
    "years_schooling_sum",
    "years_schooling_count",
    "school_attendance_yes",
    "school_attendance_denom",
    "labor_force_active",
    "labor_force_denom",
    "internet_yes",
    "internet_total",
    "rooms_sum",
    "rooms_count",
    "persons_per_hh_sum",
    "persons_per_hh_count",
    "female_count",
    "age_18_29",
    "age_30_54",
    "age_55_plus",
)

_CHUNK_SIZE: int = 200_000

# DDI value codes.
_PA1_GRP_ETNIC_INDIGENA: int = 1
_PA1_GRP_ETNIC_AFRO: int = 5
_PA1_GRP_ETNIC_NO_INFORMA: int = 9
_P_SEXO_FEMALE: int = 2
_P_NIVEL_ANOSR_NO_INFORMA: int = 99
_H_NRO_DORMIT_NO_INFORMA: int = 99
_UA_CLASE_RURAL_DISPERSO: int = 3
# P_TRABAJO codes 0-8 are valid labor statuses; 9 = No Informa (excluded).
_P_TRABAJO_VALID_CODES: frozenset[float] = frozenset({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0})

# Full DDI code sets for validation (includes metadata codes like No Informa).
_PA1_GRP_ETNIC_DDI_CODES: frozenset[int] = frozenset({1, 2, 3, 4, 5, 6, 9})
_PA_ASISTENCIA_DDI_CODES: frozenset[int] = frozenset({1, 2, 4, 9})
_P_TRABAJO_DDI_CODES: frozenset[int] = frozenset({0, 1, 2, 3, 4, 5, 6, 7, 8, 9})


# ═══════════════════════════════════════════════════════════════════════
# Pure helpers
# ═══════════════════════════════════════════════════════════════════════


def _extract_dept_code(zip_path: Path) -> str:
    """Extract the 2-digit department code from the outer zip filename.

    The filename pattern is ``NN_TitleCase.zip``, e.g. ``05_Antioquia.zip``.

    Args:
        zip_path: Path to the outer departmental zip.

    Returns:
        2-digit department code string.

    """
    return zip_path.stem[:2]


def _format_dane_code_col(u_dpto: pd.Series, u_mpio: pd.Series) -> pd.Series:
    """Combine DANE department and municipality codes into a 5-character string.

    Args:
        u_dpto: Series of 1-2 digit department codes.
        u_mpio: Series of 1-3 digit municipality codes.

    Returns:
        Series of 5-character zero-padded DANE codes (e.g., ``"05001"``).

    """
    return u_dpto.astype(str).str.strip().str.zfill(2) + u_mpio.astype(str).str.strip().str.zfill(3)


# Module-level dedup tracker for unexpected code warnings (issue #27).
_seen_code_warnings: set[tuple[str, int]] = set()


def _warn_unexpected_codes(
    series: pd.Series,
    known_codes: set[int] | frozenset[int],
    column_name: str,
) -> None:
    """Log a warning (once per column+value) for codes outside *known_codes*.

    Args:
        series: The raw column values (string or numeric).
        known_codes: All codes expected by the DDI for this column.
        column_name: Column name for the warning message.

    """
    vals = set(
        pd.to_numeric(series.dropna(), errors="coerce").dropna().astype(int).unique(),
    )
    unexpected = sorted(vals - known_codes)
    global _seen_code_warnings  # noqa: PLW0602
    for v in unexpected:
        key = (column_name, v)
        if key not in _seen_code_warnings:
            _seen_code_warnings.add(key)
            logger.warning(
                "CNPV %s: unexpected DDI code %d — excluded from computation",
                column_name,
                v,
            )


def _csv_name_for_kind(inner_zf: zipfile.ZipFile, dept: str, kind: str) -> str:
    """Find the exact CSV name inside *inner_zf* for a given *kind*.

    Args:
        inner_zf: The opened inner zip.
        dept: 2-digit department code.
        kind: ``"F8"``, ``"F9"``, or ``"F11"``.

    Returns:
        The CSV filename matching the expected pattern.

    Raises:
        ValueError: If no matching CSV is found.

    """
    pattern = _FILE_PATTERNS[kind].format(dept)
    for name in inner_zf.namelist():
        if name.upper() == pattern.upper():
            return name
    msg = (
        f"CSV not found for {kind} (dept={dept}): expected pattern {pattern!r}; "
        f"available: {sorted(inner_zf.namelist())}"
    )
    raise ValueError(msg)


def _read_kind_chunks(
    inner_zf: zipfile.ZipFile,
    dept: str,
    kind: str,
) -> Iterator[pd.DataFrame]:
    """Stream chunks from one CSV kind inside the inner zip.

    Args:
        inner_zf: The opened inner ``ZipFile``.
        dept: 2-digit department code.
        kind: ``"F8"``, ``"F9"``, or ``"F11"``.

    Yields:
        One ``pd.DataFrame`` per chunk.

    """
    csv_name = _csv_name_for_kind(inner_zf, dept, kind)
    columns = list(_FILE_COLUMNS[kind])
    with inner_zf.open(csv_name) as csv_file:
        text_file = TextIOWrapper(csv_file, encoding="latin-1")
        yield from pd.read_csv(
            text_file,
            sep=",",
            usecols=columns,
            dtype=str,
            chunksize=_CHUNK_SIZE,
        )


# ═══════════════════════════════════════════════════════════════════════
# Aggregation kernels (pure, testable without I/O)
# ═══════════════════════════════════════════════════════════════════════


def _aggregate_chunk_f11(chunk: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one F11 (personas) chunk to per-municipio demographic counts.

    Args:
        chunk: A single chunk of F11 data with columns ``U_DPTO``,
            ``U_MPIO``, ``UA_CLASE``, ``PA1_GRP_ETNIC``,
            ``P_NIVEL_ANOSR``, ``P_EDADR``, ``PA_ASISTENCIA``,
            ``P_TRABAJO``, ``P_SEXO``.

    Returns:
        DataFrame indexed by 5-character ``codigo_municipio`` with raw
        count columns used to compute the final demographic profile.

    """
    chunk = chunk.copy()
    chunk["codigo_municipio"] = _format_dane_code_col(chunk["U_DPTO"], chunk["U_MPIO"])
    chunk["P_NIVEL_ANOSR"] = pd.to_numeric(chunk["P_NIVEL_ANOSR"], errors="coerce")

    # Total persons.
    grouped_total = chunk.groupby("codigo_municipio").size().to_frame("poblacion_total")

    # Ethnicity (exclude PA1_GRP_ETNIC == 9 = No Informa).
    ethnic = chunk["PA1_GRP_ETNIC"].copy().astype(float)
    _warn_unexpected_codes(chunk["PA1_GRP_ETNIC"], _PA1_GRP_ETNIC_DDI_CODES, "PA1_GRP_ETNIC")
    grouped_indigena = (
        chunk.loc[ethnic == _PA1_GRP_ETNIC_INDIGENA]
        .groupby("codigo_municipio")
        .size()
        .to_frame("poblacion_indigena")
    )
    grouped_afro = (
        chunk.loc[ethnic == _PA1_GRP_ETNIC_AFRO]
        .groupby("codigo_municipio")
        .size()
        .to_frame("poblacion_afrocolombiana")
    )

    # Rural disperso (UA_CLASE == 3).
    ua_clase = pd.to_numeric(chunk["UA_CLASE"], errors="coerce")
    rural = chunk.loc[ua_clase == _UA_CLASE_RURAL_DISPERSO]
    grouped_rural = rural.groupby("codigo_municipio").size().to_frame("poblacion_rural_dispersa")

    # Years of schooling (exclude 99 = No Informa).
    anosr = chunk["P_NIVEL_ANOSR"].copy().astype(float)
    anosr_valid = anosr != _P_NIVEL_ANOSR_NO_INFORMA
    anosr_sum = (
        chunk.loc[anosr_valid, ["codigo_municipio", "P_NIVEL_ANOSR"]]
        .groupby("codigo_municipio")["P_NIVEL_ANOSR"]
        .sum()
        .to_frame("years_schooling_sum")
    )
    anosr_count = (
        chunk.loc[anosr_valid].groupby("codigo_municipio").size().to_frame("years_schooling_count")
    )

    # School attendance (exclude 4=No Aplica, 9=No Informa).
    asist = chunk["PA_ASISTENCIA"].copy().astype(float)
    _warn_unexpected_codes(chunk["PA_ASISTENCIA"], _PA_ASISTENCIA_DDI_CODES, "PA_ASISTENCIA")
    asist_valid = asist.isin({1, 2})
    asist_yes = (
        chunk.loc[asist == 1].groupby("codigo_municipio").size().to_frame("school_attendance_yes")
    )
    asist_denom = (
        chunk.loc[asist_valid]
        .groupby("codigo_municipio")
        .size()
        .to_frame("school_attendance_denom")
    )

    # Labour force participation (codes 1-4 = active, 0-8 = valid denom).
    trabajo = chunk["P_TRABAJO"].copy().astype(float)
    _warn_unexpected_codes(chunk["P_TRABAJO"], _P_TRABAJO_DDI_CODES, "P_TRABAJO")
    trabajo_valid = trabajo.isin(_P_TRABAJO_VALID_CODES)
    trabajo_active = chunk.loc[trabajo.isin({1.0, 2.0, 3.0, 4.0})]
    active = trabajo_active.groupby("codigo_municipio").size().to_frame("labor_force_active")
    denom = (
        chunk.loc[trabajo_valid].groupby("codigo_municipio").size().to_frame("labor_force_denom")
    )

    # Sex (P_SEXO == 2 = female).
    sexo = chunk["P_SEXO"].copy().astype(float)
    female_grp = chunk.loc[sexo == _P_SEXO_FEMALE].groupby("codigo_municipio")
    female = female_grp.size().to_frame("female_count")

    # Age bins (exclude P_EDADR == 0).
    edadr = chunk["P_EDADR"].copy().astype(float)
    age_parts: list[pd.DataFrame] = []
    for bin_name, codes in _AGE_BINS.items():
        mask = edadr.isin({float(c) for c in codes})
        age_parts.append(chunk.loc[mask].groupby("codigo_municipio").size().to_frame(bin_name))

    # Merge all partial aggregations.
    parts: list[pd.DataFrame] = [
        grouped_total,
        grouped_indigena,
        grouped_afro,
        grouped_rural,
        anosr_sum,
        anosr_count,
        asist_yes,
        asist_denom,
        active,
        denom,
        female,
    ]
    parts.extend(age_parts)
    result = parts[0]
    for part in parts[1:]:
        result = result.join(part, how="outer")
    return result.fillna(0).astype(
        {
            "poblacion_total": int,
            "poblacion_indigena": int,
            "poblacion_afrocolombiana": int,
            "poblacion_rural_dispersa": int,
            "school_attendance_yes": int,
            "school_attendance_denom": int,
            "labor_force_active": int,
            "labor_force_denom": int,
            "female_count": int,
            "age_18_29": int,
            "age_30_54": int,
            "age_55_plus": int,
            "years_schooling_count": int,
        }
    )


def _aggregate_chunk_f8(chunk: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one F8 (viviendas) chunk to per-municipio internet counts.

    Args:
        chunk: A single chunk of F8 data with columns ``U_DPTO``,
            ``U_MPIO``, ``VF_INTERNET``.

    Returns:
        DataFrame indexed by ``codigo_municipio`` with ``internet_yes``
        and ``internet_total`` columns.

    """
    chunk = chunk.copy()
    chunk["codigo_municipio"] = _format_dane_code_col(chunk["U_DPTO"], chunk["U_MPIO"])

    internet = chunk["VF_INTERNET"].copy().astype(float)
    grouped_yes = (
        chunk.loc[internet == 1].groupby("codigo_municipio").size().to_frame("internet_yes")
    )
    grouped_total = chunk.groupby("codigo_municipio").size().to_frame("internet_total")

    result = grouped_total.join(grouped_yes, how="outer")
    return result.fillna(0).astype({"internet_total": int, "internet_yes": int})


def _aggregate_chunk_f9(chunk: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one F9 (hogares) chunk to per-municipio household stats.

    Args:
        chunk: A single chunk of F9 data with columns ``U_DPTO``,
            ``U_MPIO``, ``H_NRO_DORMIT``, ``HA_TOT_PER``.

    Returns:
        DataFrame indexed by ``codigo_municipio`` with ``rooms_sum``,
        ``rooms_count``, ``persons_per_hh_sum``, and
        ``persons_per_hh_count`` columns.

    """
    chunk = chunk.copy()
    chunk["codigo_municipio"] = _format_dane_code_col(chunk["U_DPTO"], chunk["U_MPIO"])
    chunk["H_NRO_DORMIT"] = pd.to_numeric(chunk["H_NRO_DORMIT"], errors="coerce")
    chunk["HA_TOT_PER"] = pd.to_numeric(chunk["HA_TOT_PER"], errors="coerce")

    # Rooms (exclude H_NRO_DORMIT == 99 = No Informa).
    dormit = chunk["H_NRO_DORMIT"].copy().astype(float)
    dormit_valid = dormit != _H_NRO_DORMIT_NO_INFORMA
    rooms_sum = (
        chunk.loc[dormit_valid, ["codigo_municipio", "H_NRO_DORMIT"]]
        .groupby("codigo_municipio")["H_NRO_DORMIT"]
        .sum()
        .to_frame("rooms_sum")
    )
    rooms_count = chunk.loc[dormit_valid].groupby("codigo_municipio").size().to_frame("rooms_count")

    # Persons per household (HA_TOT_PER).
    hh_per = chunk["HA_TOT_PER"].copy().astype(float)
    hh_per_valid = hh_per > 0
    hh_sum = (
        chunk.loc[hh_per_valid, ["codigo_municipio", "HA_TOT_PER"]]
        .groupby("codigo_municipio")["HA_TOT_PER"]
        .sum()
        .to_frame("persons_per_hh_sum")
    )
    hh_count = (
        chunk.loc[hh_per_valid].groupby("codigo_municipio").size().to_frame("persons_per_hh_count")
    )

    result = (
        rooms_sum.join(rooms_count, how="outer")
        .join(hh_sum, how="outer")
        .join(hh_count, how="outer")
    )
    return result.fillna(0).astype({"rooms_count": int, "persons_per_hh_count": int})


# ═══════════════════════════════════════════════════════════════════════
# Percentage computation
# ═══════════════════════════════════════════════════════════════════════


def _compute_percentages(df: pd.DataFrame) -> pd.DataFrame:
    """Add percentage and rate columns from raw counts.

    Adds the derived percentage columns and drops intermediate
    aggregation columns.

    Args:
        df: DataFrame with raw count columns (``poblacion_total``,
            ``poblacion_indigena``, etc.).

    Returns:
        The same DataFrame (modified in place, returned for convenience).

    """
    # Ethnic / area percentages (denom = poblacion_total).
    df["pct_indigenous"] = np.where(
        df["poblacion_total"] > 0,
        df["poblacion_indigena"] / df["poblacion_total"],
        np.nan,
    )
    df["pct_afro_colombian"] = np.where(
        df["poblacion_total"] > 0,
        df["poblacion_afrocolombiana"] / df["poblacion_total"],
        np.nan,
    )
    df["pct_rural_disperso"] = np.where(
        df["poblacion_total"] > 0,
        df["poblacion_rural_dispersa"] / df["poblacion_total"],
        np.nan,
    )

    # Years schooling mean.
    df["years_schooling_promedio"] = np.where(
        df["years_schooling_count"] > 0,
        df["years_schooling_sum"] / df["years_schooling_count"],
        np.nan,
    )

    # School attendance.
    df["pct_school_attendance"] = np.where(
        df["school_attendance_denom"] > 0,
        df["school_attendance_yes"] / df["school_attendance_denom"],
        np.nan,
    )

    # Internet access rate (per-vivienda).
    df["internet_access_rate"] = np.where(
        df["internet_total"] > 0,
        df["internet_yes"] / df["internet_total"],
        np.nan,
    )

    # Labour force participation.
    df["labor_force_participation_rate"] = np.where(
        df["labor_force_denom"] > 0,
        df["labor_force_active"] / df["labor_force_denom"],
        np.nan,
    )

    # Female percentage (denom = poblacion_total).
    df["pct_female"] = np.where(
        df["poblacion_total"] > 0,
        df["female_count"] / df["poblacion_total"],
        np.nan,
    )

    # Rooms per household.
    df["rooms_per_household"] = np.where(
        df["rooms_count"] > 0,
        df["rooms_sum"] / df["rooms_count"],
        np.nan,
    )

    # Persons per household.
    df["persons_per_household"] = np.where(
        df["persons_per_hh_count"] > 0,
        df["persons_per_hh_sum"] / df["persons_per_hh_count"],
        np.nan,
    )

    # Age distribution (denom = sum of all three age bins; these are
    # percentages of the 20+ population, not the total population, because
    # the census age codes do not cover 0-19 with sufficient granularity).
    df["_total_age"] = df["age_18_29"] + df["age_30_54"] + df["age_55_plus"]
    for col in ("age_18_29", "age_30_54", "age_55_plus"):
        out_col = f"pct_{col}"
        df[out_col] = np.where(
            df["_total_age"] > 0,
            df[col] / df["_total_age"],
            np.nan,
        )

    # Drop temp intermediates and raw count columns, leaving only the
    # final demographic profile columns.
    all_drops = ["_total_age", *_INTERMEDIATE_COLS]
    existing_drops = [c for c in all_drops if c in df.columns]
    return df.drop(columns=existing_drops)


_KIND_AGGREGATORS: dict[str, Callable[..., pd.DataFrame]] = {
    "F11": _aggregate_chunk_f11,
    "F8": _aggregate_chunk_f8,
    "F9": _aggregate_chunk_f9,
}


# ═══════════════════════════════════════════════════════════════════════
# Iterator
# ═══════════════════════════════════════════════════════════════════════


def iter_cnpv_zip(
    zip_path: Path,
) -> Iterator[pd.DataFrame]:
    """Stream per-municipio aggregated chunks from one departmental zip.

    Opens the outer zip, finds the inner ``*_CSV.zip``, reads all three
    CSVs (F8, F9, F11) as chunks, aggregates each chunk to per-municipio
    counts, and yields the result.  Each yielded DataFrame is a per-chunk
    aggregation already indexed by ``codigo_municipio``.

    Chunk size is controlled by the module-level ``_CHUNK_SIZE`` constant
    (default 200 000 rows).

    Args:
        zip_path: Path to a departmental zip (e.g. ``05_Antioquia.zip``).

    Yields:
        ``pd.DataFrame`` per chunk, indexed by 5-char ``codigo_municipio``,
        with raw count columns.

    """
    dept = _extract_dept_code(zip_path)
    with zipfile.ZipFile(zip_path) as outer_zf:
        # Find the inner _CSV.zip (e.g. 05Antioquia/05_Antioquia_CSV.zip).
        inner_zips = [n for n in outer_zf.namelist() if n.endswith("_CSV.zip")]
        if not inner_zips:
            msg = f"No inner _CSV.zip found in {zip_path.name}"
            raise FileNotFoundError(msg)
        inner_zip_name = inner_zips[0]

        inner_data = outer_zf.read(inner_zip_name)
        with zipfile.ZipFile(BytesIO(inner_data)) as inner_zf:
            for kind in ("F11", "F8", "F9"):
                aggregator = _KIND_AGGREGATORS[kind]
                for chunk in _read_kind_chunks(inner_zf, dept, kind):
                    yield aggregator(chunk)


# ═══════════════════════════════════════════════════════════════════════
# Data loading and validation
# ═══════════════════════════════════════════════════════════════════════


def load_cnpv_data(data_dir: Path | None = None) -> pd.DataFrame:
    """Load the previously-built CNPV 2018 feature CSV.

    Args:
        data_dir: Project data directory.  If ``None``, resolves via
            ``resolve_data_dir``.

    Returns:
        DataFrame with the 18 expected columns.

    Raises:
        FileNotFoundError: If ``cnpv_2018.csv`` is not found.

    """
    base = resolve_data_dir(data_dir)
    path = base / "fundamentals" / "cnpv_2018.csv"
    if not path.is_file():
        msg = f"CNPV 2018 features not found at {path} (run `make cnpv` first)"
        raise FileNotFoundError(msg)
    return pd.read_csv(path, dtype={"codigo_municipio": str})


def validate_cnpv(df: pd.DataFrame) -> list[str]:
    """Validate a CNPV 2018 feature DataFrame against acceptance criteria.

    Checks for:
    - All 18 expected columns present
    - Non-null ``codigo_municipio`` values
    - All 5-character ``codigo_municipio`` codes
    - Percentage columns in ``[0, 1]``

    Args:
        df: The CNPV 2018 feature DataFrame.

    Returns:
        List of warning messages (empty if all checks pass).

    """
    warnings: list[str] = []

    if "codigo_municipio" not in df.columns:
        warnings.append("Missing codigo_municipio column")
        return warnings

    missing = _EXPECTED_COLUMNS - set(df.columns)
    if missing:
        warnings.append(f"Missing columns: {sorted(missing)}")
        return warnings

    null_codes = int(df["codigo_municipio"].isna().sum())
    if null_codes > 0:
        warnings.append(f"{null_codes} null codigo_municipio value(s)")

    bad_length = int(df["codigo_municipio"].str.len().ne(5).sum())
    if bad_length > 0:
        warnings.append(f"{bad_length} codigo_municipio value(s) with length != 5")

    pct_cols = [c for c in _EXPECTED_COLUMNS if c.startswith("pct_") or c.endswith("_rate")]
    for col in pct_cols:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) == 0:
            warnings.append(f"{col}: all values are NaN")
        elif not vals.between(0, 1).all():
            warnings.append(f"{col}: values outside [0, 1]")

    return warnings


# ═══════════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════════


def build_cnpv_features(data_dir: Path | None = None) -> None:
    """Aggregate CNPV 2018 microdata to municipal demographics; save CSV.

    Iterates all 33 departmental zips in ``data/cnpv-2018/raw/``, reads
    F8 (vivienda), F9 (hogar), and F11 (persona) CSVs as 200 000-row
    chunks, aggregates each chunk to per-municipio raw counts, then merges
    the three data sources by ``codigo_municipio``, computes derived
    percentages, and writes the result.

    Args:
        data_dir: Root data directory.  If ``None``, resolves via
            ``resolve_data_dir``.

    Raises:
        FileNotFoundError: If the CNPV raw data directory is missing.
        OSError: If the output cannot be written.

    Examples:
        >>> build_cnpv_features()  # doctest: +SKIP

    """
    base = resolve_data_dir(data_dir)
    raw_dir = base / "cnpv-2018" / "raw"
    if not raw_dir.is_dir():
        msg = f"CNPV 2018 raw data directory not found: {raw_dir}"
        raise FileNotFoundError(msg)

    zips = sorted(raw_dir.glob("*.zip"))
    if not zips:
        logger.warning("No zip files found in %s — nothing to build", raw_dir)
        return

    per_dept: list[pd.DataFrame] = []
    for zip_path in zips:
        logger.info("Processing %s ...", zip_path.name)
        dept_parts: list[pd.DataFrame] = list(iter_cnpv_zip(zip_path))
        if dept_parts:
            dept_agg = pd.concat(dept_parts).groupby(level=0).sum()
            per_dept.append(dept_agg)

    if not per_dept:
        logger.warning("No data aggregated — skipping write")
        return

    final = pd.concat(per_dept).groupby(level=0).sum()
    _compute_percentages(final)

    # Post-aggregation validation: Colombia has 1 122 municipalities.
    if len(final) != EXPECTED_MUNICIPALITIES:
        logger.warning(
            "CNPV output has %d municipalities; expected %d "
            "(verify all departmental zips processed)",
            len(final),
            EXPECTED_MUNICIPALITIES,
        )

    fundamentals_dir = base / "fundamentals"
    fundamentals_dir.mkdir(parents=True, exist_ok=True)
    out_path = fundamentals_dir / "cnpv_2018.csv"

    final = final.reset_index()

    # Enforce deterministic column order: key first, then sorted metrics.
    output_columns = [
        "codigo_municipio",
        *sorted(c for c in _EXPECTED_COLUMNS if c != "codigo_municipio"),
    ]
    final = final[[c for c in output_columns if c in final.columns]]

    final.to_csv(out_path, index=False)
    logger.info(
        "CNPV 2018 features saved to %s (%d municipalities, %d columns)",
        out_path,
        len(final),
        len(final.columns),
    )
