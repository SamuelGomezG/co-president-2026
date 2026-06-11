"""SPEC-24: ECP (Encuesta de Cultura Política) region-level panel ingestion.

Aggregates DANE-DIMPE Encuesta de Cultura Política microdata from
seven waves (2011--2023) into a region-level panel file at
``data/fundamentals/ecp_region_panel.csv``.

Latent constructs extracted:
* Trust-in-institutions index (Democracia module)
* Political participation index (Participacion module)
* Civic engagement index (Elecciones y partidos module;
  falls back to Capital social module if unavailable)

NOTE: The ECP survey is designed for **regional-level** inference only.
Official microdata does not include DANE department codes; the finest
geographic identifier is the ``REGION`` variable (1--5, representing
Colombia's five natural regions: Caribe, Oriental, Central, Pacifica,
and Bogotá).  The output is therefore a **region-level** panel, not a
departmental one.  Downstream models can cross-walk regions to
departments if needed.

The module follows the same 3-function ingestion pattern established
by other fundamental-data pipelines (SPEC-15, SPEC-17).
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
    "build_ecp_features",
    "load_ecp_data",
    "validate_ecp",
]

logger = logging.getLogger(__name__)

# ECP survey waves available.
_ECP_WAVES: list[int] = [2011, 2013, 2015, 2017, 2019, 2021, 2023]

# Known file-name fragments for each thematic module.
_MODULE_ZIP_FRAGMENTS: dict[str, list[str]] = {
    "trust": ["Democracia"],
    "participation": ["Participacion"],
    "engagement": ["Elecciones", "Capital"],
}

_WEIGHT_COLUMN = "FEX_P"
_REGION_COLUMN = "REGION"

# Magic-value constants (lint: PLR2004).
_WAVE_DELIMITER_CUTOFF = 2013
_MIN_COLUMNS_VALID_ZIP = 2
_MIN_COLUMNS_VALID_MODULE = 3
_EPSILON: float = 1e-12

# Output columns.
_OUTPUT_COLUMNS: list[str] = [
    "year",
    "region_code",
    "trust_index",
    "participation_index",
    "engagement_index",
    "weighted_n",
]

# Valid region codes in the REGION variable.
_VALID_REGIONS: list[int] = [1, 2, 3, 4, 5]


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════


def build_ecp_features(data_dir: Path | None = None) -> None:
    """Build the ECP region-level panel from all survey waves.

    Iterates over each wave directory under ``data/raw/DANE-DIMPE-ECP-{year}/``,
    extracts thematic module data, aggregates to the geographic region level
    using survey expansion factors, and writes a single panel CSV.

    Args:
        data_dir: Root data directory.  If ``None``, resolves via
            ``resolve_data_dir``.

    Raises:
        FileNotFoundError: If no ECP wave data is found.

    """
    base = resolve_data_dir(data_dir)
    wave_frames: list[pd.DataFrame] = []

    for year in _ECP_WAVES:
        wave_dir = base / "raw" / f"DANE-DIMPE-ECP-{year}"
        if not wave_dir.is_dir():
            logger.info("ECP wave %d not found at %s — skipping", year, wave_dir)
            continue

        df = _build_wave_panel(wave_dir, year)
        if df is not None:
            wave_frames.append(df)
            logger.info(
                "ECP %d: %d regions processed",
                year,
                df["region_code"].nunique(),
            )

    if not wave_frames:
        msg = f"No ECP wave data found under {base / 'raw' / 'DANE-DIMPE-ECP-*'}"
        raise FileNotFoundError(msg)

    result = pd.concat(wave_frames, ignore_index=True)

    fundamentals_dir = base / "fundamentals"
    fundamentals_dir.mkdir(parents=True, exist_ok=True)
    out_path = fundamentals_dir / "ecp_region_panel.csv"
    result.to_csv(out_path, index=False)
    logger.info(
        "ECP region panel saved to %s (%d rows, %d waves)",
        out_path,
        len(result),
        result["year"].nunique(),
    )


def load_ecp_data(data_dir: Path | None = None) -> pd.DataFrame:
    """Load a previously-built ECP region panel CSV.

    Args:
        data_dir: Project data directory.  If ``None``, resolves via
            ``resolve_data_dir``.

    Returns:
        DataFrame with columns ``year``, ``region_code``,
        ``trust_index``, ``participation_index``, ``engagement_index``,
        ``weighted_n``.

    Raises:
        FileNotFoundError: If ``ecp_region_panel.csv`` is not found.

    """
    base = resolve_data_dir(data_dir)
    path = base / "fundamentals" / "ecp_region_panel.csv"
    if not path.is_file():
        msg = (
            f"ECP panel not found at {path} "
            f"(run `python -m co_president ingest --component ecp` first)"
        )
        raise FileNotFoundError(msg)
    return pd.read_csv(
        path,
        dtype={"region_code": int, "year": int},
    )


def validate_ecp(df: pd.DataFrame) -> list[str]:
    """Validate an ECP region panel DataFrame.

    Checks:
    - Required columns present.
    - ``year`` is one of the known ECP waves.
    - ``region_code`` is a valid integer (1--5).
    - No nulls in critical columns.

    Args:
        df: The ECP panel DataFrame to validate.

    Returns:
        List of warning messages (empty if all checks pass).

    """
    warnings: list[str] = []

    required = set(_OUTPUT_COLUMNS)
    missing = required - set(df.columns)
    if missing:
        warnings.append(f"Missing columns: {sorted(missing)}")
        return warnings

    unknown_years = set(df["year"].unique()) - set(_ECP_WAVES)
    if unknown_years:
        warnings.append(f"Unexpected year(s): {sorted(unknown_years)}")

    region_vals = df["region_code"].dropna()
    bad_region = int((~region_vals.isin(_VALID_REGIONS)).sum())
    if bad_region > 0:
        warnings.append(f"{bad_region} region_code value(s) outside {_VALID_REGIONS}")

    index_cols = ["trust_index", "participation_index", "engagement_index"]
    for col in index_cols:
        nulls = int(df[col].isna().sum())
        if nulls > 0:
            warnings.append(f"{col}: {nulls} null value(s)")

    null_weight = int(df["weighted_n"].isna().sum())
    if null_weight > 0:
        warnings.append(f"weighted_n: {null_weight} null value(s)")

    return warnings


# ═══════════════════════════════════════════════════════════════════════
# Private helpers — panel construction
# ═══════════════════════════════════════════════════════════════════════


def _build_wave_panel(wave_dir: Path, year: int) -> pd.DataFrame | None:
    """Assemble a single ECP wave into region-level panel rows.

    Steps:
    1. Load the housing table for FEX_P and REGION identifiers.
    2. For each thematic module, compute per-respondent index.
    3. Aggregate to region via weighted mean (by ``FEX_P``).

    Args:
        wave_dir: Path to the wave directory.
        year: The survey wave year.

    Returns:
        DataFrame with one row per region, or ``None`` if no data.

    """
    housing = _load_housing_table(wave_dir, year)
    if housing is None or housing.empty:
        logger.debug("ECP %d: no housing table found", year)
        return None

    if _REGION_COLUMN not in housing.columns:
        logger.debug("ECP %d: no REGION column in housing table", year)
        return None

    before = len(housing)
    housing["region_code"] = pd.to_numeric(housing[_REGION_COLUMN], errors="coerce")
    housing = housing.dropna(subset=["region_code"])
    if len(housing) < before:
        logger.debug("ECP %d: dropped %d rows with non-numeric REGION", year, before - len(housing))

    if _WEIGHT_COLUMN not in housing.columns:
        logger.debug("ECP %d: no FEX_P in housing table", year)
        return None

    # Coerce FEX_P from string (dtype=str in _read_csv_from_zip) to numeric.
    # Without this, weighted_n sum() produces string concatenation and
    # _weighted_groupby raises TypeError in np.average.
    housing[_WEIGHT_COLUMN] = pd.to_numeric(housing[_WEIGHT_COLUMN], errors="coerce")
    housing = housing.dropna(subset=[_WEIGHT_COLUMN])

    module_indices: dict[str, pd.Series] = {}

    for construct, zip_fragments in _MODULE_ZIP_FRAGMENTS.items():
        series = _compute_module_index(wave_dir, year, construct, zip_fragments, housing)
        if series is not None:
            module_indices[construct] = series

    if not module_indices:
        return None

    result = pd.DataFrame({"region_code": housing["region_code"].unique().astype(int)})

    for construct, series in module_indices.items():
        merged_series = series.rename(f"{construct}_index").reset_index()
        merged_series["region_code"] = merged_series["region_code"].astype(int)
        result = result.merge(merged_series, on="region_code", how="left")

    for col in module_indices:
        _min_max_normalise(result, f"{col}_index")

    weighted_n = housing.groupby("region_code")[_WEIGHT_COLUMN].sum().reset_index()
    weighted_n["region_code"] = weighted_n["region_code"].astype(int)
    result = result.merge(
        weighted_n.rename(columns={_WEIGHT_COLUMN: "weighted_n"}),
        on="region_code",
        how="left",
    )

    result["year"] = year
    return result[_OUTPUT_COLUMNS].copy()


def _compute_module_index(
    wave_dir: Path,
    year: int,
    construct: str,
    zip_fragments: list[str],
    housing: pd.DataFrame,
) -> pd.Series | None:
    """Compute one region-level index for a thematic module.

    Loads zip, merges module questions with housing table, averages
    per-respondent responses, then aggregates to region via weighted
    mean by ``FEX_P``.

    Returns:
        Series ``region_code → weighted_mean``, or ``None``.

    """
    module_df = _load_module(wave_dir, zip_fragments, year)
    if module_df is None:
        logger.debug("ECP %d: module %s not found", year, construct)
        return None

    if "DIRECTORIO" not in module_df.columns:
        logger.debug("ECP %d: no DIRECTORIO in %s module", year, construct)
        return None

    merged = module_df.merge(
        housing[["DIRECTORIO", "region_code", _WEIGHT_COLUMN]],
        on="DIRECTORIO",
        how="inner",
    )
    if merged.empty:
        return None

    question_cols = [
        c for c in merged.columns if c not in {"DIRECTORIO", "region_code", _WEIGHT_COLUMN}
    ]
    if not question_cols:
        return None

    merged[question_cols] = merged[question_cols].apply(pd.to_numeric, errors="coerce")

    person_mean = merged[question_cols].mean(axis=1, skipna=True)
    merged["_index"] = person_mean

    valid = merged.dropna(subset=["_index", _WEIGHT_COLUMN])
    if valid.empty:
        return None

    return _weighted_groupby(valid, "region_code", "_index", _WEIGHT_COLUMN)


# ═══════════════════════════════════════════════════════════════════════
# Private helpers — file loading
# ═══════════════════════════════════════════════════════════════════════


def _find_zip(wave_dir: Path, fragment: str) -> Path | None:
    """Find a zip file in *wave_dir* whose name contains *fragment*."""
    for p in sorted(wave_dir.iterdir()):
        if p.suffix.lower() == ".zip" and fragment.lower() in p.stem.lower():
            return p
    return None


def _read_csv_from_zip(zip_path: Path, year: int) -> pd.DataFrame | None:
    """Read the first CSV/TSV from *zip_path* with format detection.

    Args:
        zip_path: Path to the zip archive.
        year: Wave year (pre-2013 uses tab delimiter, otherwise comma).

    Returns:
        DataFrame, or ``None`` on failure.

    """
    import zipfile  # noqa: PLC0415

    try:
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if not (name.lower().endswith(".csv") or name.lower().endswith(".txt")):
                    continue
                with zf.open(name) as fh:
                    if year < _WAVE_DELIMITER_CUTOFF:
                        df = pd.read_csv(fh, sep="\t", encoding="latin-1", dtype=str)
                    else:
                        df = pd.read_csv(fh, sep=",", encoding="utf-8-sig", dtype=str)
                    if len(df.columns) >= _MIN_COLUMNS_VALID_ZIP:
                        return df
    except Exception:  # noqa: BLE001
        logger.debug("Failed to read %s", zip_path, exc_info=True)
    return None


def _load_housing_table(wave_dir: Path, year: int) -> pd.DataFrame | None:
    """Load the housing / household table containing FEX_P and REGION.

    In 2011 the household table is ``Tabla hogares.zip``.
    In 2013--2023 it is ``Tabla de viviendas.zip`` or ``Viviendas.zip``.

    """
    for frag in ["Tabla", "Viviendas", "viviendas", "hogares"]:
        zip_path = _find_zip(wave_dir, frag)
        if zip_path is not None:
            df = _read_csv_from_zip(zip_path, year)
            if df is not None and _WEIGHT_COLUMN in df.columns:
                return df
    return None


def _load_module(
    wave_dir: Path,
    zip_fragments: list[str],
    year: int,
) -> pd.DataFrame | None:
    """Load the first available module zip matching *zip_fragments*."""
    for frag in zip_fragments:
        zip_path = _find_zip(wave_dir, frag)
        if zip_path is not None:
            df = _read_csv_from_zip(zip_path, year)
            if df is not None and len(df.columns) >= _MIN_COLUMNS_VALID_MODULE:
                return df
    return None


# ═══════════════════════════════════════════════════════════════════════
# Private helpers — computation
# ═══════════════════════════════════════════════════════════════════════


def _weighted_groupby(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    weight_col: str,
) -> pd.Series:
    """Compute weighted mean of *value_col* grouped by *group_col*.

    Returns:
        Series indexed by *group_col* values.

    """
    return df.groupby(group_col).apply(
        lambda g: np.average(g[value_col], weights=g[weight_col]),
    )


def _min_max_normalise(df: pd.DataFrame, col: str) -> None:
    """Min-max normalise *col* in-place to ``[0, 1]``."""
    lo = df[col].min()
    hi = df[col].max()
    if pd.isna(lo) or pd.isna(hi) or abs(hi - lo) < _EPSILON:
        df[col] = float("nan")
        return
    df[col] = (df[col] - lo) / (hi - lo)
