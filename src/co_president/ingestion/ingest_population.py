"""SPEC-19: DANE population projections ingest (2018--2026).

Reads the DANE PPED (Proyecciones de Población 2018--2042 con Enfoque
Diferencial) Excel workbook, filters to ``ÁREA GEOGRÁFICA == "Total"``
and years 2018--2026, pivots to wide format, and writes a clean CSV at
``data/fundamentals/population_2018_2026.csv``.

The output schema is one row per municipality with columns

    codigo_municipio, pop_2018, pop_2019, ..., pop_2026

(10 columns total, 1 key + 9 yearly population estimates).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from co_president.paths import resolve_data_dir

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "build_population_features",
    "load_population_data",
    "validate_population",
]

logger = logging.getLogger(__name__)

_SHEET_NAME = "PobMunicipalxÁrea"
_YEAR_MIN = 2018
_YEAR_MAX = 2026
_OUTPUT_FILENAME = "population_2018_2026.csv"
_EXPECTED_2022_TOTAL = 50_000_000
_TOLERANCE = 0.05  # ±5 %
_EXPECTED_MUNICIPALITIES = 1_123
_NATIONAL_TOTAL_CHECK_MIN = 1_000
_CNP_CODE_LENGTH = 5
_CRITICAL_COLUMNS: frozenset[str] = frozenset({f"pop_{y}" for y in range(_YEAR_MIN, _YEAR_MAX + 1)})


def _read_population_xlsx(xlsx_path: Path) -> pd.DataFrame:
    """Read the "PobMunicipalxÃ¡rea" sheet from the DANE PPED Excel file.

    Args:
        xlsx_path: Path to the ``.xlsx`` file.

    Returns:
        Raw DataFrame with columns ``DP, DPNOM, MPIO, DPMP, AÃ±o,
        Ãrea geogrÃ¡fica, TOTAL``.

    Raises:
        FileNotFoundError: If the file does not exist.

    """
    if not xlsx_path.is_file():
        msg = f"DANE PPED file not found: {xlsx_path}"
        raise FileNotFoundError(msg)

    return pd.read_excel(  # type: ignore[reportUnknownMemberType, reportCallIssue, reportUnknownVariableType]
        xlsx_path,
        sheet_name=_SHEET_NAME,
        header=7,
        dtype={"MPIO": str},
    )


def _filter_total_rows(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows where Ãrea geogrÃ¡fica == "Total".

    Each municipality has three area rows per year (Cabecera Municipal,
    Centros Poblados y Rural Disperso, Total).  Only the ``Total`` rows
    carry the municipal population.

    Args:
        raw: Raw DataFrame from ``_read_population_xlsx``.

    Returns:
        Filtered DataFrame containing only "Total" rows.

    """
    return raw[raw["ÁREA GEOGRÁFICA"] == "Total"].copy()


def _pivot_population_wide(filtered: pd.DataFrame) -> pd.DataFrame:
    """Pivot population data from long to wide format (years 2018--2026).

    Steps:
    1. Filter to ``AÑO`` in [2018, 2026].
    2. Pivot on ``MPIO``, columns ``AÑO``, values ``TOTAL``.
    3. Rename columns ``MPIO -> codigo_municipio``, digit columns to
       ``pop_<year>``.
    4. Zero-pad ``codigo_municipio`` to 5 characters.
    5. Assert national 2022 total is within Â±5 % of 50â000â000.

    Args:
        filtered: DataFrame filtered to "Total" rows only.

    Returns:
        Wide-format DataFrame with one row per municipality and one
        ``pop_<year>`` column per year.

    Raises:
        AssertionError: If the national 2022 population total is outside
            the Â±5 % tolerance window around 50â000â000.

    """
    mask = (filtered["AÑO"] >= _YEAR_MIN) & (filtered["AÑO"] <= _YEAR_MAX)
    clipped = filtered[mask].copy()

    pivot = clipped.pivot_table(
        index="MPIO",
        columns="AÑO",
        values="TOTAL",
        aggfunc="first",
    )
    pivot.columns.name = None
    pivot = pivot.reset_index()
    pivot = pivot.rename(columns={"MPIO": "codigo_municipio"})

    year_cols = list(range(_YEAR_MIN, _YEAR_MAX + 1))
    present = [c for c in year_cols if c in pivot.columns]
    missing_years = sorted(set(year_cols) - set(present))
    if missing_years:
        msg = (
            f"Population data missing years: {missing_years}. "
            f"Expected {_YEAR_MIN}--{_YEAR_MAX}, found {present}"
        )
        raise ValueError(msg)
    pivot = pivot[["codigo_municipio", *present]]

    rename_map = {y: f"pop_{y}" for y in range(_YEAR_MIN, _YEAR_MAX + 1)}
    pivot = pivot.rename(columns=rename_map)

    pivot["codigo_municipio"] = pivot["codigo_municipio"].astype(str).str.strip().str.zfill(5)

    return pivot


def build_population_features(data_dir: Path | None = None) -> None:
    """Read DANE PPED projections, pivot to wide, and save as CSV.

    Orchestrator that:
    1. Resolves the data directory.
    2. Reads the PPED Excel workbook.
    3. Filters to "Total" area rows.
    4. Pivots years 2018--2026 to wide format.
    5. Validates the national 2022 total.
    6. Writes ``population_2018_2026.csv`` to ``fundamentals/``.

    Args:
        data_dir: Root data directory.  If ``None``, resolves via
            ``resolve_data_dir``.

    Returns:
        None.  The function writes a CSV file as a side effect.

    Raises:
        FileNotFoundError: If the PPED Excel file is missing.
        AssertionError: If the national 2022 population total is
            outside the tolerance window.
        ValueError: If expected years are missing from the source data.

    Examples:
        >>> build_population_features(data_dir=Path("data"))  # doctest: +SKIP

    """
    base = resolve_data_dir(data_dir)
    xlsx_path = base / "raw" / "PPED-AreaMun-2018-2042_VP.xlsx"

    raw = _read_population_xlsx(xlsx_path)
    logger.info("Read %d rows from PPED XLSX (%s)", len(raw), xlsx_path)

    filtered = _filter_total_rows(raw)
    logger.info("Filtered to %d 'Total' rows", len(filtered))

    wide = _pivot_population_wide(filtered)
    logger.info(
        "Pivoted to wide format: %d municipalities x %d columns",
        len(wide),
        len(wide.columns),
    )

    if len(wide) > _NATIONAL_TOTAL_CHECK_MIN and "pop_2022" in wide.columns:
        pop_2022 = wide["pop_2022"].sum()
        lower = _EXPECTED_2022_TOTAL * (1 - _TOLERANCE)
        upper = _EXPECTED_2022_TOTAL * (1 + _TOLERANCE)
        if not (lower <= pop_2022 <= upper):
            msg = (
                f"National 2022 population {int(pop_2022):,} is outside "
                f"tolerance window [{int(lower):,}, {int(upper):,}] "
                f"(expected {_EXPECTED_2022_TOTAL:,} ±{_TOLERANCE * 100:.0f} %)"
            )
            raise AssertionError(msg)

    fundamentals_dir = base / "fundamentals"
    fundamentals_dir.mkdir(parents=True, exist_ok=True)
    out_path = fundamentals_dir / _OUTPUT_FILENAME
    wide.to_csv(out_path, index=False)
    logger.info("Population features saved to %s (%d municipalities)", out_path, len(wide))


def load_population_data(data_dir: Path | None = None) -> pd.DataFrame:
    """Load the population features CSV from disk.

    Args:
        data_dir: Root data directory.  If ``None``, resolves via
            ``resolve_data_dir``.

    Returns:
        DataFrame with columns ``codigo_municipio, pop_2018, ...,
        pop_2026``.

    Raises:
        FileNotFoundError: If the population CSV file does not exist.

    Examples:
        >>> df = load_population_data(data_dir=Path("data"))  # doctest: +SKIP
        >>> "pop_2022" in df.columns  # doctest: +SKIP
        True

    """
    base = resolve_data_dir(data_dir)
    path = base / "fundamentals" / _OUTPUT_FILENAME
    if not path.is_file():
        msg = f"Population features file not found: {path}"
        raise FileNotFoundError(msg)
    return pd.read_csv(path, dtype={"codigo_municipio": str})


def validate_population(df: pd.DataFrame) -> list[str]:
    """Validate population features against acceptance criteria.

    Checks:
    - Required columns are present.
    - ``codigo_municipio`` values are 5-character strings.
    - At least ``_EXPECTED_MUNICIPALITIES`` rows exist.
    - No nulls in any population column.
    - Population values are non-negative.

    Args:
        df: Population features DataFrame to validate.

    Returns:
        List of warning strings.  An empty list means no issues found.

    Raises:
        ValueError: If ``df`` is not a DataFrame.

    Examples:
        >>> df = _make_valid_population(5)  # doctest: +SKIP
        >>> validate_population(df)  # doctest: +SKIP
        []

    """
    warnings: list[str] = []

    missing = _CRITICAL_COLUMNS - set(df.columns)
    if missing:
        warnings.append(f"Population: missing columns: {', '.join(sorted(missing))}")

    if "codigo_municipio" not in df.columns:
        warnings.append("Population: missing codigo_municipio column")
        return warnings

    bad_codes = df[df["codigo_municipio"].astype(str).str.len() != _CNP_CODE_LENGTH]
    if not bad_codes.empty:
        warnings.append(
            f"Population: {len(bad_codes)} codigo_municipio values are not 5 characters"
        )

    if len(df) < _EXPECTED_MUNICIPALITIES:
        warnings.append(
            f"Population: expected at least {_EXPECTED_MUNICIPALITIES} rows, got {len(df)}"
        )

    for col in _CRITICAL_COLUMNS:
        if col not in df.columns:
            continue
        if pd.api.types.is_float_dtype(df[col]):
            warnings.append(f"Population: {col} has float dtype (expected integer)")
        nulls = df[col].isna().sum()
        if nulls > 0:
            warnings.append(f"Population: {col} has {nulls} null value(s)")
        non_positive = (df[col] <= 0).sum()
        if non_positive > 0:
            warnings.append(f"Population: {col} has {non_positive} non-positive value(s)")

    return warnings
