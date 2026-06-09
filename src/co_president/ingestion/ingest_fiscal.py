"""SPEC-21a: TerriData fiscal autonomy variable ingestion.

Reads the TerriData ``Finanzas Públicas`` xlsx (inside ``.xlsx.zip``)
from ``data/raw/TerriData/`` and extracts four fiscal indicators at the
municipal level, computing multi-year averages over 2018--2024.
"""

from __future__ import annotations

import logging
from pathlib import Path
import tempfile
import zipfile

import pandas as pd

from co_president.paths import resolve_data_dir

__all__ = [
    "build_fiscal_features",
    "load_fiscal_data",
    "validate_fiscal",
]

logger = logging.getLogger(__name__)

_EXPECTED_MUNICIPALITIES = 1_122

# Target indicator names in Hoja01.
_IND_INGRESOS_TRIBUTARIOS = "Ingresos tributarios"
_IND_INGRESOS_NO_TRIBUTARIOS = "Ingresos no tributarios"
_IND_INGRESOS_CORRIENTES = "Ingresos corrientes"
_IND_GASTOS_TOTALES_PC = "Gastos totales per cápita"
_IND_TRANSFERENCIAS_PC = "Transferencias per cápita de los ingresos corrientes"
_IND_INGRESOS_TRIBUTARIOS_PC = "Ingresos tributarios per cápita"

_TARGET_INDICATORS: frozenset[str] = frozenset(
    {
        _IND_INGRESOS_TRIBUTARIOS,
        _IND_INGRESOS_NO_TRIBUTARIOS,
        _IND_INGRESOS_CORRIENTES,
        _IND_GASTOS_TOTALES_PC,
        _IND_TRANSFERENCIAS_PC,
        _IND_INGRESOS_TRIBUTARIOS_PC,
    }
)

_CRITICAL_COLUMNS: list[str] = [
    "pct_ingresos_propios",
    "gastos_totales_per_capita",
    "transferencias_per_capita",
    "ingresos_tributarios_per_capita",
]

_MIN_YEAR = 2018
_MAX_YEAR = 2024

# Column indices in TerriData Hoja01 (0-indexed).
_COL_ENTITY_CODE = 2
_COL_INDICATOR = 6
_COL_VALUE = 7
_COL_YEAR = 9

_OUTPUT_FILENAME = "fiscal.csv"


def _parse_colombian_number(val: object) -> float | None:
    """Convert a Colombian-formatted number string to float.

    Colombian convention uses ``.`` as the thousands separator and ``,``
    as the decimal separator (e.g. ``'4.977,34'`` → 4977.34).

    Args:
        val: Raw value from the spreadsheet (string, number, or ``None``).

    Returns:
        Float value, or ``None`` when the input is empty/unparseable.

    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if not s:
        return None
    s = s.replace(".", "")
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        logger.warning("Cannot parse Colombian number: %r", val)
        return None


def _safe_year(cell_value: object) -> int | None:
    """Extract an integer year from an openpyxl cell value, or ``None``."""
    if cell_value is None:
        return None
    if isinstance(cell_value, (int, float)):
        return int(cell_value)
    return None


def _safe_str(cell_value: object) -> str:
    """Extract a string from an openpyxl cell value, returning empty string for ``None``."""
    if cell_value is None:
        return ""
    return str(cell_value)


def _read_fiscal_zip(zip_path: Path) -> pd.DataFrame:
    """Read TerriData zip, extract ``Hoja01``, filter to target rows.

    Returns a DataFrame with columns ``codigo_municipio``, ``year``,
    ``indicator``, and ``value`` (parsed to ``float``).

    """
    if not zip_path.is_file():
        msg = f"TerriData zip not found: {zip_path}"
        raise FileNotFoundError(msg)

    try:
        import openpyxl  # noqa: PLC0415
    except ImportError:
        msg = "openpyxl is required to read TerriData xlsx"
        raise ImportError(msg) from None

    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(zip_path) as z:
        xlsx_name = next(
            n for n in z.namelist() if n.endswith(".xlsx") and not n.startswith("__MACOSX")
        )
        with tempfile.TemporaryDirectory() as tmp:
            z.extract(xlsx_name, tmp)
            wb = openpyxl.load_workbook(  # type: ignore[reportUnknownMemberType]
                Path(tmp) / xlsx_name,
                read_only=True,
                data_only=True,
            )
            ws = wb["Hoja01"]
            for cells in ws.iter_rows(min_row=2, values_only=True):
                indicator = cells[_COL_INDICATOR]
                year = _safe_year(cells[_COL_YEAR])
                if (
                    indicator in _TARGET_INDICATORS
                    and year is not None
                    and _MIN_YEAR <= year <= _MAX_YEAR
                ):
                    raw_code = _safe_str(cells[_COL_ENTITY_CODE])
                    parsed = _parse_colombian_number(cells[_COL_VALUE])
                    rows.append(
                        {
                            "codigo_municipio": raw_code,
                            "year": year,
                            "indicator": indicator,
                            "value": parsed,
                        }
                    )
            wb.close()

    return pd.DataFrame(rows)


def _compute_fiscal_features(records_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-format records and compute the four fiscal variables.

    For each municipality-year the ratio
    ``(tributarios + no_tributarios) / corrientes`` is computed to produce
    ``pct_ingresos_propios``.  The per-capita indicators are carried
    directly.  Finally all four variables are averaged across
    2018--2024.

    Department-level aggregate codes (``xxx000``) are dropped.

    """
    if records_df.empty:
        return pd.DataFrame(columns=["codigo_municipio", *_CRITICAL_COLUMNS])

    clean = records_df.dropna(subset=["value"]).copy()

    # Drop department-level aggregate codes (last 3 digits == "000").
    clean = clean[~clean["codigo_municipio"].str.endswith("000")].copy()

    # Ensure 5-character zero-padded codes.
    clean["codigo_municipio"] = clean["codigo_municipio"].str.zfill(5)

    # Pivot: one row per (municipio, year), one column per indicator.
    pivoted = clean.pivot_table(
        index=["codigo_municipio", "year"],
        columns="indicator",
        values="value",
        aggfunc="first",
    ).reset_index()

    required = {
        _IND_INGRESOS_TRIBUTARIOS,
        _IND_INGRESOS_NO_TRIBUTARIOS,
        _IND_INGRESOS_CORRIENTES,
        _IND_GASTOS_TOTALES_PC,
        _IND_TRANSFERENCIAS_PC,
        _IND_INGRESOS_TRIBUTARIOS_PC,
    }
    missing = required - set(pivoted.columns)
    if missing:
        logger.warning("Missing indicator columns after pivot: %s", sorted(missing))
        for col in missing:
            pivoted[col] = float("nan")

    # Per-year ratio: recursos propios / ingresos corrientes.
    trib = pd.to_numeric(pivoted[_IND_INGRESOS_TRIBUTARIOS], errors="coerce")
    no_trib = pd.to_numeric(pivoted[_IND_INGRESOS_NO_TRIBUTARIOS], errors="coerce")
    corrientes = pd.to_numeric(pivoted[_IND_INGRESOS_CORRIENTES], errors="coerce")

    propio_sum = trib.fillna(0.0) + no_trib.fillna(0.0)
    safe_denom = corrientes.replace(0, pd.NA)
    pivoted["pct_ingresos_propios"] = (propio_sum / safe_denom).clip(0.0, 1.0)

    # Per-capita indicators (direct pass-through).
    pivoted["gastos_totales_per_capita"] = pd.to_numeric(
        pivoted[_IND_GASTOS_TOTALES_PC], errors="coerce"
    )
    pivoted["transferencias_per_capita"] = pd.to_numeric(
        pivoted[_IND_TRANSFERENCIAS_PC], errors="coerce"
    )
    pivoted["ingresos_tributarios_per_capita"] = pd.to_numeric(
        pivoted[_IND_INGRESOS_TRIBUTARIOS_PC], errors="coerce"
    )

    # Average across years 2018-2024.
    return (
        pivoted.groupby("codigo_municipio", observed=True)
        .agg(
            {
                "pct_ingresos_propios": "mean",
                "gastos_totales_per_capita": "mean",
                "transferencias_per_capita": "mean",
                "ingresos_tributarios_per_capita": "mean",
            }
        )
        .reset_index()
    )


def build_fiscal_features(data_dir: Path | None = None) -> None:
    """Read TerriData zip and save fiscal features to ``fiscal.csv``.

    Args:
        data_dir: Root data directory.  If ``None``, resolves via
            ``resolve_data_dir``.

    Raises:
        FileNotFoundError: If the TerriData zip file is missing.
        ImportError: If ``openpyxl`` is not installed.

    Returns:
        ``None``.  Writes ``fiscal.csv`` to ``data_dir/fundamentals/``.

    """
    base = resolve_data_dir(data_dir)
    zip_path = base / "raw" / "TerriData" / "TerriData_Finanzas_Publicas.xlsx.zip"

    records = _read_fiscal_zip(zip_path)
    df = _compute_fiscal_features(records)

    warnings = validate_fiscal(df)
    for warning in warnings:
        logger.warning("Fiscal validation: %s", warning)

    fundamentals_dir = base / "fundamentals"
    fundamentals_dir.mkdir(parents=True, exist_ok=True)
    out_path = fundamentals_dir / _OUTPUT_FILENAME
    df.to_csv(out_path, index=False)
    logger.info(
        "Fiscal features saved to %s (%d municipalities)",
        out_path,
        len(df),
    )


def load_fiscal_data(data_dir: Path | None = None) -> pd.DataFrame:
    """Load a previously-built fiscal feature CSV.

    Args:
        data_dir: Project data directory.  If ``None``, resolves via
            ``resolve_data_dir``.

    Returns:
        DataFrame with columns ``codigo_municipio``, ``pct_ingresos_propios``,
        ``gastos_totales_per_capita``, ``transferencias_per_capita``,
        ``ingresos_tributarios_per_capita``.

    Raises:
        FileNotFoundError: If ``fiscal.csv`` is not found.

    """
    base = resolve_data_dir(data_dir)
    path = base / "fundamentals" / _OUTPUT_FILENAME
    if not path.is_file():
        msg = f"Fiscal features not found at {path} (run `build_fiscal_features` first)"
        raise FileNotFoundError(msg)
    return pd.read_csv(path, dtype={"codigo_municipio": str})


def validate_fiscal(df: pd.DataFrame) -> list[str]:
    """Validate a fiscal feature DataFrame against acceptance criteria.

    Checks for missing columns, nulls in critical columns, out-of-range
    values, and 5-character municipality codes.

    Args:
        df: The fiscal feature DataFrame to validate.

    Returns:
        List of warning messages (empty if all checks pass).

    """
    warnings: list[str] = []

    required = {"codigo_municipio", *_CRITICAL_COLUMNS}
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

    if "pct_ingresos_propios" in df.columns:
        vals = df["pct_ingresos_propios"].dropna()
        if len(vals) > 0 and not vals.between(0.0, 1.0).all():
            warnings.append("pct_ingresos_propios: found values outside [0.0, 1.0] range")

    bad_length = int(df["codigo_municipio"].str.len().ne(5).sum())
    if bad_length > 0:
        warnings.append(f"{bad_length} codigo_municipio value(s) with length != 5")

    if len(df) < _EXPECTED_MUNICIPALITIES - 20:
        warnings.append(f"Expected ~{_EXPECTED_MUNICIPALITIES} municipalities, got {len(df)}")

    return warnings
