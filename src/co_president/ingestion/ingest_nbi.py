"""SPEC-18: DANE NBI poverty indicator ingestion (primary poverty feature).

Reads the CNPV-2018-NBI.xlsx workbook from ``data/raw/DANE-NBI/``, extracts
municipal-level NBI (Necesidades Básicas Insatisfechas) rates for total,
urban (cabecera), and rural (centros poblados y rural disperso) domains,
normalises them to the [0, 1] interval, and writes the result to
``data/fundamentals/nbi_2018.csv``.

NBI is the primary poverty feature in the feature matrix.  It has zero
imputation — all ~1 122 municipalities have complete coverage from the
DANE census.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from co_president.config import EXPECTED_MUNICIPALITIES
from co_president.paths import resolve_data_dir

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "build_nbi_features",
    "load_nbi_data",
    "validate_nbi",
]

logger = logging.getLogger(__name__)

# Imported from co_president.config: EXPECTED_MUNICIPALITIES

# Column indices in the Municipios sheet (0-indexed after skiprows).
_COL_DEPT_CODE = 0
_COL_MPIO_CODE = 2
_COL_NBI_TOTAL = 4
_COL_NBI_URBAN = 11
_COL_NBI_RURAL = 18

_CRITICAL_COLUMNS: list[str] = ["nbi_rate", "nbi_urban", "nbi_rural"]


def build_nbi_features(data_dir: Path | None = None) -> None:
    """Read DANE NBI xlsx, normalise, and save to ``nbi_2018.csv``.

    Args:
        data_dir: Root data directory.  If ``None``, resolves via
            ``resolve_data_dir``.

    Raises:
        FileNotFoundError: If the NBI xlsx file is missing.

    Returns:
        ``None``.  Writes ``nbi_2018.csv`` to ``data_dir/fundamentals/``.

    """
    base = resolve_data_dir(data_dir)
    xlsx_path = base / "raw" / "DANE-NBI" / "CNPV-2018-NBI.xlsx"
    raw = _read_nbi_xlsx(xlsx_path)
    df = _clean_nbi_data(raw)
    fundamentals_dir = base / "fundamentals"
    fundamentals_dir.mkdir(parents=True, exist_ok=True)
    out_path = fundamentals_dir / "nbi_2018.csv"
    df.to_csv(out_path, index=False)
    logger.info(
        "NBI features saved to %s (%d municipalities)",
        out_path,
        len(df),
    )


def load_nbi_data(data_dir: Path | None = None) -> pd.DataFrame:
    """Load a previously-built NBI feature CSV.

    Args:
        data_dir: Project data directory.  If ``None``, resolves via
            ``resolve_data_dir``.

    Returns:
        DataFrame with columns ``codigo_municipio``, ``nbi_rate``,
        ``nbi_urban``, ``nbi_rural``.

    Raises:
        FileNotFoundError: If ``nbi_2018.csv`` is not found.

    """
    base = resolve_data_dir(data_dir)
    path = base / "fundamentals" / "nbi_2018.csv"
    if not path.is_file():
        msg = f"NBI features not found at {path} (run `make nbi` first)"
        raise FileNotFoundError(msg)
    return pd.read_csv(path, dtype={"codigo_municipio": str})


def validate_nbi(df: pd.DataFrame) -> list[str]:
    """Validate an NBI feature DataFrame against acceptance criteria.

    Checks for missing columns, nulls in critical columns,
    out-of-range values, and 5-character municipality codes.

    Args:
        df: The NBI feature DataFrame to validate.

    Returns:
        List of warning messages (empty if all checks pass).

    """
    warnings: list[str] = []

    required = {"codigo_municipio", "nbi_rate", "nbi_urban", "nbi_rural"}
    missing = required - set(df.columns)
    if missing:
        warnings.append(f"Missing columns: {sorted(missing)}")
        return warnings

    null_counts = df[_CRITICAL_COLUMNS].isna().sum()
    for col in _CRITICAL_COLUMNS:
        count = int(null_counts[col])
        if count > 0:
            warnings.append(
                f"{col}: {count} null value{'s' if count != 1 else ''} (out of {len(df)} rows)"
            )

    for col in ("nbi_rate", "nbi_urban", "nbi_rural"):
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        if not vals.between(0.0, 1.0).all():
            warnings.append(f"{col}: found values outside [0.0, 1.0] range")

    bad_length = int(df["codigo_municipio"].str.len().ne(5).sum())
    if bad_length > 0:
        warnings.append(f"{bad_length} codigo_municipio value(s) with length != 5")

    if len(df) < EXPECTED_MUNICIPALITIES:
        warnings.append(f"Expected {EXPECTED_MUNICIPALITIES} municipalities, got {len(df)}")

    return warnings


# ═══════════════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════════════


def _read_nbi_xlsx(xlsx_path: Path) -> pd.DataFrame:
    """Read the raw NBI Excel file, skipping header rows.

    The Municipios sheet has 10 leading header rows (title, blank,
    subtitle, blank, column group headers, column metric labels).
    Data starts at openpyxl row index 10 (0-indexed).

    Args:
        xlsx_path: Path to ``CNPV-2018-NBI.xlsx``.

    Returns:
        Raw DataFrame with column indices 0-24.

    Raises:
        FileNotFoundError: If the xlsx file does not exist.

    """
    if not xlsx_path.is_file():
        msg = f"NBI xlsx not found: {xlsx_path}"
        raise FileNotFoundError(msg)

    return pd.read_excel(  # type: ignore[reportUnknownMemberType, reportCallIssue, reportUnknownVariableType]
        xlsx_path,
        sheet_name="Municipios",
        header=None,
        skiprows=9,
        dtype=str,
    )


def _clean_nbi_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise the raw NBI DataFrame.

    Steps:
    1. Build 5-character zero-padded ``codigo_municipio``.
    2. Reject the ``TOTAL NACIONAL`` final row.
    3. Select and rename the three NBI percentage columns.
    4. Convert percentages (0-100) to rates (0-1).

    Args:
        raw: Raw DataFrame from ``_read_nbi_xlsx``.

    Returns:
        Cleaned DataFrame with columns ``codigo_municipio``,
        ``nbi_rate``, ``nbi_urban``, ``nbi_rural``.

    """
    dept_str = raw[_COL_DEPT_CODE].astype(str).str.strip().str.zfill(2)
    mpio_str = raw[_COL_MPIO_CODE].astype(str).str.strip().str.zfill(3)
    raw["codigo_municipio"] = dept_str + mpio_str

    raw = raw.dropna(subset=["codigo_municipio"])

    # Filter out the TOTAL NACIONAL row (codigo = 00000).
    raw = raw[raw["codigo_municipio"] != "00000"].copy()

    # Build the output DataFrame with rates in [0, 1].
    result = pd.DataFrame()
    result["codigo_municipio"] = raw["codigo_municipio"]
    result["nbi_rate"] = pd.to_numeric(raw[_COL_NBI_TOTAL], errors="coerce") / 100.0
    result["nbi_urban"] = pd.to_numeric(raw[_COL_NBI_URBAN], errors="coerce") / 100.0
    result["nbi_rural"] = pd.to_numeric(raw[_COL_NBI_RURAL], errors="coerce") / 100.0

    return result.reset_index(drop=True)
