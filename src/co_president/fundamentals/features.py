"""SPEC-21: MunicipalFeatures dataclass and load_features() entry point.

Defines a typed, frozen data contract for the municipal feature matrix and
provides the ``load_features()`` function that loads the persisted matrix
from ``data/processed/``, normalises column names, attaches derived columns
(``historical``, ``historical_turnout_m``), and validates the schema.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import logging
from typing import TYPE_CHECKING, Literal

import pandas as pd

from co_president.config import (
    HISTORICAL_CANDIDATE_IDEOLOGY,
    HISTORICAL_ROUND2_IDEOLOGY,
)
from co_president.paths import resolve_data_dir

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "HistoricalRecord",
    "MunicipalFeatures",
    "load_features",
]

logger = logging.getLogger(__name__)

_EXPECTED_MUNICIPALITIES = 1_122
_NULL_RATE_THRESHOLD = 0.20
_CNP_CODE_LENGTH = 5
_ROUND_TWO = 2

# Columns that overlap between the socioeconomic stub (merged first = _x)
# and the CNPV 2018 census (merged second = _y).  We prefer CNPV.
_OVERLAPPING_COLUMNS: frozenset[str] = frozenset(
    {
        "pct_afro_colombian",
        "pct_indigenous",
        "pct_rural_disperso",
        "internet_access_rate",
    }
)

_BOOL_MAP: dict[str, bool] = {
    "true": True,
    "1": True,
    "1.0": True,
    "false": False,
    "0": False,
    "0.0": False,
}


@dataclass(frozen=True)
class HistoricalRecord:
    """A single per-election HistoricalRecord for a municipality.

    Captures the left/right candidate vote shares and abstention rate for
    one election round (e.g. 2022 round 1) in one municipality.

    Attributes:
        year: Election year (e.g. 2002, 2006, ..., 2022).
        round: Election round (1 or 2).
        left_candidate: Canonical key of the left-wing candidate.
        right_candidate: Canonical key of the right-wing candidate.
        left_share: Vote share of the left candidate (0-1).
        right_share: Vote share of the right candidate (0-1).
        abstention_rate: Abstention rate for this municipality-year-round (0-1).

    """

    year: int
    round: int
    left_candidate: str
    right_candidate: str
    left_share: float
    right_share: float
    abstention_rate: float


@dataclass(frozen=True)
class MunicipalFeatures:
    """Typed, immutable data contract for one municipality's feature row.

    Each field corresponds to a column in the validated feature matrix
    returned by ``load_features()``.  The ``historical`` field holds a
    tuple of ``HistoricalRecord`` objects (immutable by construction).

    Attributes:
        codigo_municipio: 5-digit DANE municipality code.
        nombre_municipio: Human-readable municipality name.
        departamento: Department name.
        region: Geographic region (Andina, Caribe, Pacifica, Orinoquia,
            Amazonia) or ``None`` if unavailable.

        -- CNPV 2018 demographic features (always present) --
        poblacion_total: Total population from CNPV 2018.
        poblacion_afrocolombiana: Afro-Colombian population count.
        poblacion_indigena: Indigenous population count.
        poblacion_rural_dispersa: Population in dispersed rural areas.
        pct_afro_colombian: Proportion of afro-Colombian population (0-1).
        pct_indigenous: Proportion of indigenous population (0-1).
        pct_rural_disperso: Proportion in dispersed rural areas (0-1).
        years_schooling_promedio: Mean years of schooling.
        pct_school_attendance: School attendance rate (0-1).
        internet_access_rate: Internet access rate (0-1).
        labor_force_participation_rate: Labour force participation rate (0-1).
        pct_female: Proportion female (0-1).
        rooms_per_household: Mean rooms per household.
        persons_per_household: Mean persons per household.
        pct_age_18_29: Proportion aged 18-29 (0-1).
        pct_age_30_54: Proportion aged 30-54 (0-1).
        pct_age_55_plus: Proportion aged 55+ (0-1).

        -- Poverty (NBI primary, IPM secondary) --
        nbi_rate: Necesidades Basicas Insatisfechas rate (0-1).
        nbi_urban: NBI in urban areas (0-1).
        nbi_rural: NBI in rural areas (0-1).
        ipm_2018: Multidimensional Poverty Index (2018, may be imputed).
        ipm_2018_imputed: Whether the 2018 IPM value was imputed.
        ipm_2022: Multidimensional Poverty Index (2022, may be imputed).
        ipm_2022_imputed: Whether the 2022 IPM value was imputed.

        -- Population projections --
        pop_2018: Projected population for 2018.
        pop_2019: Projected population for 2019.
        pop_2020: Projected population for 2020.
        pop_2021: Projected population for 2021.
        pop_2022: Projected population for 2022.
        pop_2023: Projected population for 2023.
        pop_2024: Projected population for 2024.
        pop_2025: Projected population for 2025.
        pop_2026: Projected population for 2026.

        -- Risk --
        risk_level: MOE electoral risk classification.
        is_pdet: Whether the municipality is a PDET priority area.
        armed_group_presence: Whether armed groups are documented present.
        coca_hectares: Hectares of coca cultivation (UNODC estimate).
        high_risk_flag: ``True`` when ``risk_level`` is ``"extreme"`` or ``"high"``.

        -- Historical --
        historical: Tuple of per-election HistoricalRecord objects.
        historical_turnout_m: Mean municipal turnout across available years (0-1).

    """

    # Municipality identifiers
    codigo_municipio: str
    nombre_municipio: str
    departamento: str
    region: str | None

    # CNPV 2018 demographic features
    poblacion_total: int
    poblacion_afrocolombiana: int
    poblacion_indigena: int
    poblacion_rural_dispersa: int
    pct_afro_colombian: float
    pct_indigenous: float
    pct_rural_disperso: float
    years_schooling_promedio: float
    pct_school_attendance: float
    internet_access_rate: float
    labor_force_participation_rate: float
    pct_female: float
    rooms_per_household: float
    persons_per_household: float
    pct_age_18_29: float
    pct_age_30_54: float
    pct_age_55_plus: float

    # Poverty
    nbi_rate: float
    nbi_urban: float
    nbi_rural: float
    ipm_2018: float | None
    ipm_2018_imputed: bool
    ipm_2022: float | None
    ipm_2022_imputed: bool

    # Population projections
    pop_2018: int
    pop_2019: int
    pop_2020: int
    pop_2021: int
    pop_2022: int
    pop_2023: int
    pop_2024: int
    pop_2025: int
    pop_2026: int

    # Risk
    risk_level: Literal["extreme", "high", "medium", "low"] | None
    is_pdet: bool
    armed_group_presence: bool
    coca_hectares: float
    high_risk_flag: bool

    # Historical
    historical: tuple[HistoricalRecord, ...]
    historical_turnout_m: float


# ═══════════════════════════════════════════════════════════════════════
# Ideology lookup helpers
# ═══════════════════════════════════════════════════════════════════════


def _get_ideology(year: int, round_num: int) -> dict[str, str]:
    """Return the left/right candidate mapping for a given year and round.

    Args:
        year: Election year.
        round_num: Election round (1 or 2).

    Returns:
        Dict with ``"left"`` and ``"right"`` keys, or empty dict if
        the year is not in the registry.

    """
    if round_num == _ROUND_TWO and year in HISTORICAL_ROUND2_IDEOLOGY:
        return HISTORICAL_ROUND2_IDEOLOGY[year]
    return HISTORICAL_CANDIDATE_IDEOLOGY.get(year, {})


# ═══════════════════════════════════════════════════════════════════════
# Matrix loading
# ═══════════════════════════════════════════════════════════════════════


def _load_matrix_csv(data_dir: Path) -> pd.DataFrame:
    """Load the feature matrix CSV/Parquet from ``data/processed/``.

    Args:
        data_dir: Root data directory.

    Returns:
        Raw DataFrame as read from disk.

    Raises:
        FileNotFoundError: If neither CSV nor Parquet file exists.

    """
    processed_dir = data_dir / "processed"
    parquet_path = processed_dir / "municipal_feature_matrix.parquet"
    csv_path = processed_dir / "municipal_feature_matrix.csv"

    if parquet_path.is_file():
        return pd.read_parquet(parquet_path)  # type: ignore[reportUnknownMemberType]
    if csv_path.is_file():
        return pd.read_csv(csv_path, dtype={"codigo_municipio": str})
    msg = (
        f"municipal_feature_matrix not found at {parquet_path} or {csv_path}. "
        f"Run ``build_feature_matrix()`` first."
    )
    raise FileNotFoundError(msg)


def _load_historical_results(data_dir: Path) -> pd.DataFrame:
    """Load the historical election results CSV.

    Args:
        data_dir: Root data directory.

    Returns:
        DataFrame with historical results.

    Raises:
        FileNotFoundError: If the file does not exist.

    """
    path = data_dir / "fundamentals" / "historical_results.csv"
    if not path.is_file():
        msg = (
            f"historical_results.csv not found at {path}. "
            f"Run the historical ingestion pipeline first."
        )
        raise FileNotFoundError(msg)
    return pd.read_csv(
        path,
        dtype={
            "codigo_municipio": str,
            "year": int,
            "round": int,
        },
    )


# ═══════════════════════════════════════════════════════════════════════
# Column normalisation
# ═══════════════════════════════════════════════════════════════════════


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names to match the ``MunicipalFeatures`` schema.

    - Renames ``REGION`` to ``region``.
    - Resolves CNPV-over-socioeconomic column suffix overlap (prefers ``_y``).
    - Drops internal DIVIPOLA columns.
    - Drops the ``_x``/``_y`` originals after resolution.
    - Drops legacy columns from pre-SPEC-19 matrix builds.

    Args:
        df: Raw feature matrix as loaded from disk.

    Returns:
        DataFrame with normalised column names.

    """
    result = df.copy()

    if "REGION" in result.columns:
        result = result.rename(columns={"REGION": "region"})

    if "CÓDIGO DANE DEL DEPARTAMENTO" in result.columns:
        result = result.drop(columns=["CÓDIGO DANE DEL DEPARTAMENTO"])

    result = _reconcile_nombre_municipio(result)
    result = _reconcile_overlapping_columns(result)
    return _drop_legacy_columns(result)


def _reconcile_nombre_municipio(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve ``nombre_municipio`` merge suffix, preferring DIVIPOLA."""
    if "nombre_municipio_x" not in df.columns:
        return df
    df["nombre_municipio"] = df["nombre_municipio_x"].fillna(df.get("nombre_municipio_y"))
    return df.drop(columns=["nombre_municipio_x", "nombre_municipio_y"])


def _reconcile_overlapping_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve overlapping CNPV/socio columns, preferring CNPV (_y)."""
    result = df.copy()
    for base in _OVERLAPPING_COLUMNS:
        y_col = f"{base}_y"
        x_col = f"{base}_x"
        if y_col in result.columns and x_col in result.columns:
            result[base] = result[y_col].fillna(result[x_col])
            result = result.drop(columns=[x_col, y_col])
        elif y_col in result.columns:
            result[base] = result[y_col]
            result = result.drop(columns=[y_col])
        elif x_col in result.columns:
            result[base] = result[x_col]
            result = result.drop(columns=[x_col])
    return result


def _drop_legacy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop legacy population columns that predate SPEC-19."""
    drops = [c for c in ("proyeccion_2022", "population_2022") if c in df.columns]
    return df.drop(columns=drops) if drops else df


# ═══════════════════════════════════════════════════════════════════════
# Historical record & turnout computation
# ═══════════════════════════════════════════════════════════════════════


def _filter_valid_historical(historical: pd.DataFrame) -> pd.DataFrame:
    """Filter historical results to valid municipality codes.

    Excludes sentinel / placeholder codes (``000NA``) and invalid-length
    codes.

    """
    return historical[
        historical["codigo_municipio"].notna()
        & (historical["codigo_municipio"] != "000NA")
        & (historical["codigo_municipio"].str.len() == _CNP_CODE_LENGTH)
    ].copy()


def _compute_turnout(df: pd.DataFrame, historical: pd.DataFrame) -> pd.DataFrame:
    """Compute ``historical_turnout_m`` per municipality and merge into *df*.

    Args:
        df: Feature matrix DataFrame (must have ``codigo_municipio``).
        historical: Historical results DataFrame.

    Returns:
        The feature matrix with an added ``historical_turnout_m`` column.

    """
    valid = _filter_valid_historical(historical)

    if valid.empty:
        logger.warning("No valid historical results -- setting historical_turnout_m to NaN")
        df["historical_turnout_m"] = float("nan")
        return df

    safe_registered = valid["registered_voters"].replace(0, pd.NA)
    valid["turnout"] = (
        (valid["total_votes"] / safe_registered).clip(0.0, 1.0)
        if safe_registered.notna().any()
        else float("nan")
    )

    turnout_per_yr = (
        valid.groupby(["codigo_municipio", "year", "round"])["turnout"].first().reset_index()
    )
    turnout_mean = turnout_per_yr.groupby("codigo_municipio")["turnout"].mean()

    return df.merge(
        turnout_mean.rename("historical_turnout_m").reset_index(),
        on="codigo_municipio",
        how="left",
    )


def _build_historical_column(df: pd.DataFrame, historical: pd.DataFrame) -> pd.DataFrame:
    """Build the ``historical`` Series of ``tuple[HistoricalRecord, ...]``.

    Args:
        df: Feature matrix (must have ``codigo_municipio`` column).
        historical: Historical results DataFrame.

    Returns:
        The feature matrix with an added ``historical`` column.

    """
    valid = _filter_valid_historical(historical)

    records_map: dict[str, list[HistoricalRecord]] = {code: [] for code in df["codigo_municipio"]}

    for (municipio_raw, year_raw, round_raw), group in valid.groupby(
        ["codigo_municipio", "year", "round"]
    ):
        municipio = str(municipio_raw)  # type: ignore[arg-type]
        year = int(year_raw)  # type: ignore[arg-type]
        round_num = int(round_raw)  # type: ignore[arg-type]
        if municipio not in records_map:
            continue
        ideology = _get_ideology(year, round_num)
        left_cand = ideology.get("left")
        right_cand = ideology.get("right")
        if not left_cand or not right_cand:
            continue

        cand_shares = dict(zip(group["candidate"], group["vote_share"], strict=False))
        left_share = cand_shares.get(left_cand)
        right_share = cand_shares.get(right_cand)
        if left_share is None or right_share is None:
            continue

        first_row = group.iloc[0]
        safe_reg = first_row["registered_voters"]
        safe_total = first_row["total_votes"]
        if safe_reg and safe_reg > 0:
            abstention = max(0.0, 1.0 - (safe_total / safe_reg))
        else:
            abstention = float("nan")

        records_map[municipio].append(
            HistoricalRecord(
                year=int(year),
                round=int(round_num),
                left_candidate=left_cand,
                right_candidate=right_cand,
                left_share=float(left_share),
                right_share=float(right_share),
                abstention_rate=float(abstention),
            )
        )

    df["historical"] = pd.Series(
        [tuple(records_map[code]) for code in df["codigo_municipio"]],
        index=df.index,
    )
    return df


# ═══════════════════════════════════════════════════════════════════════
# Type coercion
# ═══════════════════════════════════════════════════════════════════════


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns to their ``MunicipalFeatures``-specified types.

    Handles the transition from CSV/Parquet string representations to
    Python-native types expected by the schema (bool, int).

    Args:
        df: Feature matrix with normalised column names.

    Returns:
        DataFrame with coerced column dtypes.

    """
    result = df.copy()

    _coerce_bool_columns(result)
    _coerce_population_columns(result)

    if "codigo_municipio" in result.columns:
        result["codigo_municipio"] = result["codigo_municipio"].astype(str).str.strip()

    return result


def _coerce_bool_columns(df: pd.DataFrame) -> None:
    """Coerce known boolean columns from string/int to bool."""
    bool_cols = [
        "ipm_2018_imputed",
        "ipm_2022_imputed",
        "is_pdet",
        "armed_group_presence",
        "high_risk_flag",
    ]
    for col in bool_cols:
        if col in df.columns and not pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip().str.lower().map(_BOOL_MAP).astype(bool)


def _coerce_population_columns(df: pd.DataFrame) -> None:
    """Coerce population projection columns to nullable integer."""
    for year in range(2018, 2027):
        col = f"pop_{year}"
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")


# ═══════════════════════════════════════════════════════════════════════
# Schema validation
# ═══════════════════════════════════════════════════════════════════════


def _validate_schema(df: pd.DataFrame) -> None:
    """Validate the feature matrix against the ``MunicipalFeatures`` schema.

    Checks:
    - All required scalar columns are present.
    - ``codigo_municipio`` is unique and 5-character.
    - ``nbi_rate`` is in [0, 1].
    - ``historical_turnout_m`` is in [0, 1] and non-null.
    - ``pop_*`` columns are present and non-null.
    - Null rates are within acceptable thresholds.

    Args:
        df: The feature matrix DataFrame to validate.

    Raises:
        ValueError: If any validation check fails.

    """
    mf_fields = {f.name for f in fields(MunicipalFeatures)}
    scalar_fields = mf_fields - {"historical"}

    missing = scalar_fields - set(df.columns)
    if missing:
        msg = f"Feature matrix missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    if "codigo_municipio" in df.columns:
        _validate_codigo_municipio(df)

    if "nbi_rate" in df.columns:
        _validate_nbi_rate(df)

    if "historical_turnout_m" in df.columns:
        _validate_turnout(df)

    pop_cols = [f"pop_{y}" for y in range(2018, 2027)]
    missing_pop = [c for c in pop_cols if c not in df.columns]
    if missing_pop:
        msg = f"Missing population columns: {missing_pop}"
        raise ValueError(msg)

    _validate_null_rate(df, scalar_fields)


def _validate_codigo_municipio(df: pd.DataFrame) -> None:
    """Validate ``codigo_municipio`` uniqueness and length."""
    if not df["codigo_municipio"].is_unique:
        msg = "codigo_municipio is not unique"
        raise ValueError(msg)
    bad_length = int(df["codigo_municipio"].astype(str).str.len().ne(_CNP_CODE_LENGTH).sum())
    if bad_length > 0:
        msg = f"{bad_length} codigo_municipio value(s) with length != {_CNP_CODE_LENGTH}"
        raise ValueError(msg)


def _validate_nbi_rate(df: pd.DataFrame) -> None:
    """Validate ``nbi_rate`` is in [0, 1]."""
    vals = df["nbi_rate"].dropna()
    if len(vals) > 0 and not vals.between(0.0, 1.0).all():
        msg = "nbi_rate has values outside [0, 1]"
        raise ValueError(msg)


def _validate_turnout(df: pd.DataFrame) -> None:
    """Validate ``historical_turnout_m`` range and nulls."""
    vals = df["historical_turnout_m"].dropna()
    if len(vals) > 0 and not vals.between(0.0, 1.0).all():
        msg = "historical_turnout_m has values outside [0, 1]"
        raise ValueError(msg)
    null_count = int(df["historical_turnout_m"].isna().sum())
    if null_count > 0:
        msg = f"historical_turnout_m has {null_count} null value(s)"
        raise ValueError(msg)


def _validate_null_rate(df: pd.DataFrame, scalar_fields: set[str]) -> None:
    """Validate that null rate is within threshold."""
    total_cells = len(df) * len(scalar_fields)
    null_count = int(df[list(scalar_fields)].isna().sum().sum())
    if total_cells > 0 and null_count / total_cells > _NULL_RATE_THRESHOLD:
        msg = (
            f"Feature matrix null rate {null_count / total_cells:.1%}"
            f" exceeds threshold {_NULL_RATE_THRESHOLD:.0%}"
        )
        raise ValueError(msg)


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════


def load_features(data_dir: Path | None = None) -> pd.DataFrame:
    """Load and validate the municipal feature matrix.

    Loads the persisted feature matrix from ``data/processed/``, normalises
    column names (resolves CNPV/socioeconomic overlap), attaches the
    ``historical`` column of ``HistoricalRecord`` tuples and the
    ``historical_turnout_m`` turnout baseline, and validates every column
    against the ``MunicipalFeatures`` schema.

    Args:
        data_dir: Root data directory.  If ``None``, resolves via
            ``resolve_data_dir``.

    Returns:
        DataFrame with one row per municipality and columns matching the
        scalar fields of ``MunicipalFeatures``.

    Raises:
        FileNotFoundError: If the feature matrix or historical results CSV
            is missing.
        ValueError: If schema validation fails.

    Examples:
        >>> mf = load_features()
        >>> len(mf)
        1122
        >>> mf["nbi_rate"].between(0, 1).all()
        True

    """
    base = resolve_data_dir(data_dir)

    df = _load_matrix_csv(base)
    logger.info("Loaded feature matrix: %d rows x %d columns", len(df), len(df.columns))

    df = _normalise_columns(df)
    logger.debug("Normalised columns: %d remaining", len(df.columns))

    df = _coerce_types(df)
    logger.debug("Coerced column types")

    historical = _load_historical_results(base)
    logger.info("Loaded historical results: %d rows", len(historical))

    df = _compute_turnout(df, historical)
    logger.debug("Computed historical_turnout_m")

    df = _build_historical_column(df, historical)
    logger.debug("Built historical records column")

    _validate_schema(df)
    logger.info("Feature matrix validated against MunicipalFeatures schema")

    return df
