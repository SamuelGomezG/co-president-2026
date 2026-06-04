"""SPEC-18: IPM poverty indicators ingest (secondary poverty feature, opt-in).

Reads the ECV 2018 and 2022 household microdata from ``data/raw/IPM-*/``,
aggregates the pre-computed IPM variable to the departamento-level using
household expansion factors (``fex_c``), propagates the departamento IPM
to all municipalities via LEFT-JOIN on departamento code, and computes
an R² guard against NBI to determine whether the IPM column is informative
enough to include in the feature matrix.

This is an **opt-in** component: it only runs when ``--component ipm``
is passed.  A missing IPM file on disk is tolerated by
``validate_component_health`` without error.

Note on data scope:
    The IPM-2018/2022 microdata in this repository is the **ECV (Encuesta
    Nacional de Calidad de Vida)** sample survey, not the CNPV 2018
    census.  DANE publishes IPM at the departamento level from this
    survey, not at the municipal level.  Every municipality in a
    departamento inherits that departamento's IPM score.  The
    ``imputed`` flag is ``True`` for all rows as a signal that the value
    is derived from a coarser geography.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from co_president.paths import resolve_data_dir

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "build_ipm_features",
    "load_ipm_data",
    "validate_ipm",
]

logger = logging.getLogger(__name__)

_EXPECTED_MUNICIPALITIES = 1_122
_R2_GUARD_THRESHOLD = 0.5

# Map year → (subdirectory, zip_basename, csv_name, separator).
_IPM_YEARS: dict[int, tuple[str, str, str | None, str]] = {
    2018: ("IPM-2018", "Hogares (departamental)", "Hogares (departamental) .csv", ";"),
    2022: ("IPM-2022", "hogares (Departamental) 2022", None, ","),
}

_CRITICAL_COLUMNS: list[str] = ["ipm_2018", "ipm_2022"]


def build_ipm_features(data_dir: Path | None = None) -> None:
    """Compute departamento-level IPM from ECV microdata; save to CSV.

    For each available ECV year (2018, 2022), reads the departamental
    Hogares microdata, computes the weighted-mean IPM per departamento
    using household expansion factors, then propagates the departamento
    score to all municipalities by LEFT-JOIN on the 2-digit departamento
    code.

    Before writing, computes: R² = corr(imputed_IPM, nbi_rate)².
    If R² < 0.5, the IPM column is excluded from the output CSV (only
    the header is written, with zero data rows).

    Args:
        data_dir: Root data directory.  If ``None``, resolves via
            ``resolve_data_dir``.

    Raises:
        FileNotFoundError: If the ECV microdata is missing for a year.

    """
    base = resolve_data_dir(data_dir)
    fundamentals_dir = base / "fundamentals"
    fundamentals_dir.mkdir(parents=True, exist_ok=True)

    divipola = _load_divipola(base)

    ipm_data: dict[str, pd.Series] = {"codigo_municipio": divipola["codigo_municipio"]}
    for year in (2018, 2022):
        try:
            hogares = _read_ecv_hogares(base, year)
            ipm_depto = _aggregate_ipm_to_departamento(hogares)
            ipm_muni = _propagate_departamento_to_municipio(ipm_depto, divipola)
            col_name = f"ipm_{year}"
            ipm_data[col_name] = ipm_muni["ipm"]
            ipm_data[f"{col_name}_imputed"] = pd.Series(True, index=divipola.index)  # noqa: FBT003
        except (FileNotFoundError, KeyError) as exc:
            logger.warning("IPM %d: data not available (%s) — emitting NaN", year, exc)
            col_name = f"ipm_{year}"
            ipm_data[col_name] = pd.Series(pd.NA, index=divipola.index)
            ipm_data[f"{col_name}_imputed"] = pd.Series(True, index=divipola.index)  # noqa: FBT003

    result = pd.DataFrame(ipm_data)

    # R² guard: compute R² between IPM-2018 and NBI.
    if result["ipm_2018"].notna().any():
        nbi_path = fundamentals_dir / "nbi_2018.csv"
        if nbi_path.is_file():
            nbi_df = pd.read_csv(nbi_path, dtype={"codigo_municipio": str})
            merged = result.merge(nbi_df[["codigo_municipio", "nbi_rate"]], on="codigo_municipio")
            valid = merged.dropna(subset=["ipm_2018", "nbi_rate"])
            if len(valid) > 3:  # noqa: PLR2004
                r2 = _pearson_r2(
                    np.asarray(valid["ipm_2018"]),
                    np.asarray(valid["nbi_rate"]),
                )
                if r2 < _R2_GUARD_THRESHOLD:
                    logger.warning(
                        "IPM R² guard: R² = %.3f < %.1f — dropping IPM-2018 column",
                        r2,
                        _R2_GUARD_THRESHOLD,
                    )
                    result = result.drop(columns=["ipm_2018", "ipm_2018_imputed"])
                else:
                    logger.info(
                        "IPM 2018 R² = %.3f (pass, threshold = %.1f)", r2, _R2_GUARD_THRESHOLD
                    )

    out_path = fundamentals_dir / "ipm_2018.csv"
    result.to_csv(out_path, index=False)
    logger.info(
        "IPM features saved to %s (%d municipalities, %d columns)",
        out_path,
        len(result),
        len(result.columns),
    )


def load_ipm_data(data_dir: Path | None = None) -> pd.DataFrame:
    """Load a previously-built IPM feature CSV.

    Args:
        data_dir: Project data directory.  If ``None``, resolves via
            ``resolve_data_dir``.

    Returns:
        DataFrame with columns ``codigo_municipio``, ``ipm_2018``,
        ``ipm_2018_imputed``, ``ipm_2022``, ``ipm_2022_imputed``.

    Raises:
        FileNotFoundError: If ``ipm_2018.csv`` is not found.

    """
    base = resolve_data_dir(data_dir)
    path = base / "fundamentals" / "ipm_2018.csv"
    if not path.is_file():
        msg = f"IPM features not found at {path} (run `make ipm` first)"
        raise FileNotFoundError(msg)
    return pd.read_csv(path, dtype={"codigo_municipio": str})


def validate_ipm(df: pd.DataFrame) -> list[str]:
    """Validate an IPM feature DataFrame against acceptance criteria.

    Checks for expected columns, null handling, and boolean imputation
    flags.

    Args:
        df: The IPM feature DataFrame to validate.

    Returns:
        List of warning messages (empty if all checks pass).

    """
    warnings: list[str] = []

    expected_bases = {
        "codigo_municipio",
        "ipm_2018",
        "ipm_2018_imputed",
        "ipm_2022",
        "ipm_2022_imputed",
    }
    has_columns = set(df.columns)
    if not expected_bases.issubset(has_columns) and expected_bases.intersection(has_columns) != {
        "codigo_municipio",
    }:
        warnings.append(f"Expected IPM columns, got: {sorted(has_columns)}")
        return warnings

    warnings.extend(
        f"{col}: expected boolean dtype"
        for col in ("ipm_2018_imputed", "ipm_2022_imputed")
        if col in df.columns
        and df[col].dropna().dtype not in (bool, np.dtype("bool"), np.dtype("int64"))
    )

    for ipm_col in ("ipm_2018", "ipm_2022"):
        if ipm_col in df.columns:
            vals = df[ipm_col].dropna()
            if len(vals) > 0 and not vals.between(0.0, 1.0).all():
                warnings.append(f"{ipm_col}: found values outside [0.0, 1.0] range")

    if "codigo_municipio" in df.columns:
        bad_length = int(df["codigo_municipio"].str.len().ne(5).sum())
        if bad_length > 0:
            warnings.append(f"{bad_length} codigo_municipio value(s) with length != 5")

    return warnings


# ═══════════════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════════════


def _read_ecv_hogares(base: Path, year: int) -> pd.DataFrame:
    """Read the ECV Hogares departamental microdata for *year*.

    The ECV Hogares file contains the pre-computed ``ipm`` variable
    (household-level IPM score) and the household expansion factor
    ``fex_c``.  The ``DEPARTAMENTO`` column is a 2-digit integer code.

    Args:
        base: Resolved project data directory.
        year: 2018 or 2022.

    Returns:
        DataFrame with columns ``DEPARTAMENTO``, ``fex_c``, ``ipm``.

    Raises:
        FileNotFoundError: If the zip or CSV is not found.

    """
    try:
        subdir, zip_basename, csv_name, sep = _IPM_YEARS[year]
    except KeyError:
        msg = f"Unsupported IPM year: {year}"
        raise ValueError(msg) from None

    if csv_name is None:
        csv_name = f"{zip_basename}.csv"

    zip_path = base / "raw" / subdir / f"{zip_basename}.zip"
    if not zip_path.is_file():
        msg = f"ECV zip not found for IPM {year}: {zip_path}"
        raise FileNotFoundError(msg)

    df = _read_ecv_csv_from_zip(zip_path, csv_name, sep=sep)

    df["DEPARTAMENTO"] = df["DEPARTAMENTO"].astype(str).str.strip().str.zfill(2)
    df["fex_c"] = pd.to_numeric(df["fex_c"], errors="coerce")
    df["ipm"] = pd.to_numeric(df["ipm"], errors="coerce")
    return df


def _aggregate_ipm_to_departamento(hogares: pd.DataFrame) -> pd.DataFrame:
    """Compute weighted-mean IPM per departamento using household expansion factors.

    Args:
        hogares: DataFrame from ``_read_ecv_hogares`` with columns
            ``DEPARTAMENTO``, ``fex_c``, ``ipm``.

    Returns:
        DataFrame with columns ``DEPARTAMENTO`` and ``ipm`` (weighted
        mean).  33 rows (one per departamento).

    """
    valid = hogares.dropna(subset=["fex_c", "ipm"])
    ipm_weighted = valid["ipm"] * valid["fex_c"]
    weighted_sum = ipm_weighted.groupby(valid["DEPARTAMENTO"], sort=False).sum()
    weight_sum = valid.groupby("DEPARTAMENTO", sort=False)["fex_c"].sum()
    result = (weighted_sum / weight_sum).reset_index()
    result.columns = ["DEPARTAMENTO", "ipm"]
    return result


def _propagate_departamento_to_municipio(
    ipm_depto: pd.DataFrame,
    divipola: pd.DataFrame,
) -> pd.DataFrame:
    """LEFT-JOIN departamento IPM onto municipality registry.

    The first 2 characters of ``codigo_municipio`` are the departamento
    code.

    Args:
        ipm_depto: DataFrame with ``DEPARTAMENTO`` and ``ipm`` columns.
        divipola: DIVIPOLA registry with ``codigo_municipio`` column.

    Returns:
        DataFrame with ``codigo_municipio`` and ``ipm`` columns, indexed
        to match the DIVIPOLA registry.

    """
    divipola = divipola.copy()
    divipola["DEPARTAMENTO"] = divipola["codigo_municipio"].str[:2]
    merged = divipola[["codigo_municipio", "DEPARTAMENTO"]].merge(
        ipm_depto,
        on="DEPARTAMENTO",
        how="left",
    )
    return merged[["codigo_municipio", "ipm"]]


def _load_divipola(base: Path) -> pd.DataFrame:
    """Load the DIVIPOLA master registry.

    Args:
        base: Resolved project data directory.

    Returns:
        DataFrame with at least ``codigo_municipio`` column.

    Raises:
        FileNotFoundError: If ``divipola_master.csv`` is not found.

    """
    divipola_path = base / "fundamentals" / "divipola_master.csv"
    if not divipola_path.is_file():
        msg = f"DIVIPOLA master not found at {divipola_path}"
        raise FileNotFoundError(msg)
    return pd.read_csv(divipola_path, dtype={"codigo_municipio": str})


def _pearson_r2(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson R² between two 1-D arrays.

    Args:
        x: First 1-D numeric array.
        y: Second 1-D numeric array.

    Returns:
        R² = Pearson correlation coefficient squared.

    """
    corr = np.corrcoef(x, y)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr * corr)


def _read_ecv_csv_from_zip(
    zip_path: Path,
    csv_name: str,
    sep: str = ";",
) -> pd.DataFrame:
    """Read the ECV Hogares CSV from inside a zip file.

    Args:
        zip_path: Path to the zip file.
        csv_name: Exact filename of the CSV inside the zip.
        sep: Column separator (``";"`` for 2018, ``","`` for 2022).

    Returns:
        DataFrame with columns ``DEPARTAMENTO``, ``fex_c``, ``ipm``.

    """
    import zipfile  # noqa: PLC0415

    with zipfile.ZipFile(zip_path) as zf, zf.open(csv_name) as raw:
        raw_bytes = raw.read()
    decoded = raw_bytes.decode("utf-8-sig")
    from io import StringIO  # noqa: PLC0415

    df = pd.read_csv(StringIO(decoded), sep=sep, decimal=",")
    return df[["DEPARTAMENTO", "fex_c", "ipm"]]