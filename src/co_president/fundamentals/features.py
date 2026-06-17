"""SPEC-21: MunicipalFeatures dataclass and load_features() entry point.

Defines a typed, frozen data contract for the municipal feature matrix and
provides the ``load_features()`` function that loads the persisted matrix
from ``data/processed/``, normalises column names, attaches derived columns
(``historical``, ``historical_turnout_m``), and validates the schema.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import logging
import math
from typing import TYPE_CHECKING, Literal, cast

import pandas as pd

from co_president.config import get_ideology
from co_president.paths import resolve_data_dir

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "HistoricalRecord",
    "MunicipalFeatures",
    "clr",
    "load_features",
    "logit",
]

logger = logging.getLogger(__name__)

_NULL_RATE_THRESHOLD = 0.20
_CNP_CODE_LENGTH = 5
_BOGOTA_LOCALIDAD_CODE_LENGTH = 7
_BOGOTA_CODE = "11001"
_EPSILON = 1e-10

# Columns that overlap between the socioeconomic stub (merged first = _x)
# and the real component files merged second (_y).  We prefer the real
# component data over the 3-row stub.
_OVERLAPPING_COLUMNS: frozenset[str] = frozenset(
    {
        # CNPV-over-socioeconomic (CNPV has full 1,122-row coverage)
        "pct_afro_colombian",
        "pct_indigenous",
        "pct_rural_disperso",
        "internet_access_rate",
        "poblacion_total",
        "poblacion_afrocolombiana",
        "poblacion_indigena",
        "poblacion_rural_dispersa",
        # NBI-over-socioeconomic (socioeconomic stub has stale nbi_rate)
        "nbi_rate",
        # Bogotá localidad name (DIVIPOLA -> comuna_nombre_x, others -> _y)
        "comuna_nombre",
    }
)

_BOOL_MAP: dict[str, bool] = {
    "true": True,
    "1": True,
    "1.0": True,
    "yes": True,
    "y": True,
    "t": True,
    "si": True,
    "false": False,
    "0": False,
    "0.0": False,
    "no": False,
    "n": False,
    "f": False,
}


@dataclass(frozen=True)
class HistoricalRecord:
    """A single per-election HistoricalRecord for a municipality.

    Captures the left/right candidate vote shares and abstention rate for
    one election round (e.g. 2022 round 1) in one municipality.

    ``left_share`` and ``right_share`` form a 2-part composition (they
    represent the vote shares of the two dominant ideological blocs).  Use
    ``clr_shares()`` to obtain their centred log-ratio transform before
    feeding them into a regression (see SPEC-21c).

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

    def clr_shares(self) -> tuple[float, float]:
        """CLR of the ``(left_share, right_share)`` binary composition.

        Normalises the two shares to sum to 1 and applies the centred
        log-ratio transform.  For the binary (D=2) case the result
        satisfies:

        - ``clr_left + clr_right == 0``
        - ``clr_left = 0.5 * logit(normalised_left_share)``

        Returns:
            A 2-tuple ``(clr_left, clr_right)``.

        Raises:
            ValueError: If either share is negative, non-finite, or both
                are zero (propagated from :func:`clr`).

        Examples:
            >>> hr = HistoricalRecord(2022, 2, "petro", "hernandez",
            ...                       0.55, 0.45, 0.30)
            >>> left, right = hr.clr_shares()
            >>> abs(left + right) < 1e-12
            True

        """
        return cast("tuple[float, float]", clr((self.left_share, self.right_share)))


@dataclass(frozen=True)
class MunicipalFeatures:
    """Typed, immutable data contract for one municipality's feature row.

    Each field corresponds to a column in the validated feature matrix
    returned by ``load_features()``.  The ``historical`` field holds a
    tuple of ``HistoricalRecord`` objects (immutable by construction).

    Attributes:
        codigo_municipio: 5-digit DANE municipality code (or 7-digit for
            Bogotá D.C. localidades, e.g. ``"1100101"`` for Usaquén).
        nombre_municipio: Human-readable municipality name.
        comuna_nombre: For Bogotá D.C. localidades, the localidad name
            (e.g. ``"Usaquén"``, ``"Chapinero"``).  ``None`` for all other
            municipalities.
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

        -- Fiscal autonomy (TerriData, multi-year avg 2018--2024) --
        pct_ingresos_propios: proportion of current income from own
            resources (tax + non-tax) over total current income (0-1).
        gastos_totales_per_capita: Total expenditure per capita (COP).
        transferencias_per_capita: Transfers per capita (COP).
        ingresos_tributarios_per_capita: Tax revenue per capita (COP).

        -- Risk --
        risk_level: MOE electoral risk classification.
        is_pdet: Whether the municipality is a PDET priority area.
        armed_group_presence: Whether armed groups are documented present.
        coca_hectares: Hectares of coca cultivation (UNODC estimate).
        high_risk_flag: ``True`` when ``risk_level`` is ``"extreme"`` or ``"high"``.

        -- Historical --
        historical: Tuple of per-election HistoricalRecord objects.
        historical_turnout_m: Mean municipal turnout across available years (0-1).

    .. rubric:: Compositional feature groups

    The following groups contain compositional data (parts of a whole) and
    should be log-ratio transformed before entering a linear regression:

    * **Historical vote shares** (``HistoricalRecord.left_share``,
      ``.right_share``) — 2-part simplex for the left/right pair.
      Use ``HistoricalRecord.clr_shares()`` → CLR reduces to
      ``0.5 * logit(normalised_left_share)`` for D=2.
    * **Multi-candidate first-round shares** (future) — apply full K-part
      CLR via ``clr()`` when the full candidate vector is available.
    * **Poverty composition** (``nbi_rate``, ``1 - nbi_rate``) — binary
      poverty / non-poverty simplex.  Use ``clr_poverty()``.
    * **NBI area composition** (``nbi_urban``, ``nbi_rural``,
      ``1 - nbi_urban - nbi_rural``) — 3-part simplex when all three NBI
      values are present.  Available via ``clr_nbi_areas()``.

    """

    # Municipality identifiers
    codigo_municipio: str
    nombre_municipio: str
    comuna_nombre: str | None
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

    # Fiscal autonomy
    pct_ingresos_propios: float
    gastos_totales_per_capita: float
    transferencias_per_capita: float
    ingresos_tributarios_per_capita: float

    # Risk
    risk_level: Literal["extreme", "high", "medium", "low"] | None
    is_pdet: bool
    armed_group_presence: bool
    coca_hectares: float
    high_risk_flag: bool

    # Historical
    historical: tuple[HistoricalRecord, ...]
    historical_turnout_m: float

    # ── Compositional data helpers (SPEC-21c) ────────────────────────

    def clr_poverty(self) -> tuple[float, float]:
        """CLR of the ``(nbi_rate, 1 - nbi_rate)`` poverty composition.

        ``nbi_rate`` represents the proportion of households with unmet
        basic needs, making the pair ``(nbi_rate, 1 - nbi_rate)`` a
        binary composition (poverty / non-poverty simplex).

        Returns:
            A 2-tuple ``(clr_nbi, clr_non_nbi)``.

        Raises:
            ValueError: If ``nbi_rate`` is negative, non-finite, or
                exactly 1.0 (propagated from :func:`clr`).

        Examples:
            >>> mf = MunicipalFeatures(
            ...     codigo_municipio="05001", nombre_municipio="Test",
            ...     comuna_nombre=None, departamento="Antioquia", region=None,
            ...     poblacion_total=100, poblacion_afrocolombiana=10,
            ...     poblacion_indigena=5, poblacion_rural_dispersa=2,
            ...     pct_afro_colombian=0.10, pct_indigenous=0.05,
            ...     pct_rural_disperso=0.02, years_schooling_promedio=10.0,
            ...     pct_school_attendance=0.85, internet_access_rate=0.70,
            ...     labor_force_participation_rate=0.65, pct_female=0.51,
            ...     rooms_per_household=3.0, persons_per_household=3.5,
            ...     pct_age_18_29=0.30, pct_age_30_54=0.40,
            ...     pct_age_55_plus=0.30, nbi_rate=0.15, nbi_urban=0.08,
            ...     nbi_rural=0.22, ipm_2018=0.12, ipm_2018_imputed=True,
            ...     ipm_2022=0.14, ipm_2022_imputed=True,
            ...     pop_2018=100_000, pop_2019=101_000, pop_2020=102_000,
            ...     pop_2021=103_000, pop_2022=104_000, pop_2023=105_000,
            ...     pop_2024=106_000, pop_2025=107_000, pop_2026=108_000,
            ...     pct_ingresos_propios=0.60, gastos_totales_per_capita=2.0,
            ...     transferencias_per_capita=0.9,
            ...     ingresos_tributarios_per_capita=1.0, risk_level="low",
            ...     is_pdet=False, armed_group_presence=False,
            ...     coca_hectares=0.0, high_risk_flag=False, historical=(),
            ...     historical_turnout_m=0.70,
            ... )
            >>> clr_nbi, clr_non_nbi = mf.clr_poverty()
            >>> abs(clr_nbi + clr_non_nbi) < 1e-12
            True

        """
        return cast("tuple[float, float]", clr((self.nbi_rate, 1.0 - self.nbi_rate)))

    def clr_nbi_areas(self) -> tuple[float, float, float]:
        """CLR of the ``(nbi_urban, nbi_rural, residual)`` 3-part composition.

        When all three NBI values are present, ``(nbi_urban, nbi_rural,
        1 - nbi_urban - nbi_rural)`` forms a 3-part simplex (urban
        poverty / rural poverty / no-poverty areas).

        Returns:
            A 3-tuple ``(clr_urban, clr_rural, clr_non_nbi)``.

        Raises:
            ValueError: If any NBI value is negative, non-finite, or if
                ``nbi_urban + nbi_rural > 1`` producing a negative residual
                (propagated from :func:`clr`).

        Examples:
            >>> mf = MunicipalFeatures(
            ...     codigo_municipio="05001", nombre_municipio="Test",
            ...     comuna_nombre=None, departamento="Antioquia", region=None,
            ...     poblacion_total=100, poblacion_afrocolombiana=10,
            ...     poblacion_indigena=5, poblacion_rural_dispersa=2,
            ...     pct_afro_colombian=0.10, pct_indigenous=0.05,
            ...     pct_rural_disperso=0.02, years_schooling_promedio=10.0,
            ...     pct_school_attendance=0.85, internet_access_rate=0.70,
            ...     labor_force_participation_rate=0.65, pct_female=0.51,
            ...     rooms_per_household=3.0, persons_per_household=3.5,
            ...     pct_age_18_29=0.30, pct_age_30_54=0.40,
            ...     pct_age_55_plus=0.30, nbi_rate=0.15, nbi_urban=0.08,
            ...     nbi_rural=0.22, ipm_2018=0.12, ipm_2018_imputed=True,
            ...     ipm_2022=0.14, ipm_2022_imputed=True,
            ...     pop_2018=100_000, pop_2019=101_000, pop_2020=102_000,
            ...     pop_2021=103_000, pop_2022=104_000, pop_2023=105_000,
            ...     pop_2024=106_000, pop_2025=107_000, pop_2026=108_000,
            ...     pct_ingresos_propios=0.60, gastos_totales_per_capita=2.0,
            ...     transferencias_per_capita=0.9,
            ...     ingresos_tributarios_per_capita=1.0, risk_level="low",
            ...     is_pdet=False, armed_group_presence=False,
            ...     coca_hectares=0.0, high_risk_flag=False, historical=(),
            ...     historical_turnout_m=0.70,
            ... )
            >>> clr_u, clr_r, clr_n = mf.clr_nbi_areas()
            >>> abs(clr_u + clr_r + clr_n) < 1e-12
            True

        """
        residual = 1.0 - self.nbi_urban - self.nbi_rural
        return cast("tuple[float, float, float]", clr((self.nbi_urban, self.nbi_rural, residual)))


# ═══════════════════════════════════════════════════════════════════════
# Compositional data helpers (SPEC-21c)
# ═══════════════════════════════════════════════════════════════════════


def logit(p: float) -> float:
    """Logit (log-odds) transform, clamped away from 0/1 to avoid infinities.

    Args:
        p: Probability or proportion in ``[0, 1]``.

    Returns:
        ``log(p / (1 - p))``, with *p* clamped to
        ``[_EPSILON, 1 - _EPSILON]``.

    Raises:
        This function never raises.  Inputs at 0.0 and 1.0 are clamped to
        the safe interval to keep the result finite.

    Examples:
        >>> logit(0.5)
        0.0
        >>> logit(0.75)
        1.0986122886681098
        >>> math.isfinite(logit(0.0))
        True
        >>> math.isfinite(logit(1.0))
        True

    """
    p = max(min(p, 1.0 - _EPSILON), _EPSILON)
    return math.log(p / (1.0 - p))


def clr(shares: tuple[float, ...]) -> tuple[float, ...]:
    """Centred log-ratio transform for a composition (Aitchison, 1982).

    Computes ``clr(x_i) = log(x_i / g(x))`` where ``g(x)`` is the geometric
    mean of the composition.  The elements of the result always sum to zero.

    Args:
        shares: Compositional values.  Must all be non-negative and sum to a
            positive number.

    Returns:
        CLR-transformed shares (same length as input).

    Raises:
        ValueError: If the input is empty, contains non-finite or negative
            values, or has zero total.

    Example:
        >>> clr((0.3, 0.7))
        ...  # doctest: +SKIP
        (-0.423648..., 0.423648...)

    """
    if not shares:
        msg = "Composition must be non-empty"
        raise ValueError(msg)
    for i, s in enumerate(shares):
        if not math.isfinite(s):
            msg = f"Composition part at index {i} is not finite: {s}"
            raise ValueError(msg)
        if s < 0:
            msg = f"Composition part at index {i} is negative: {s}"
            raise ValueError(msg)
    total = sum(shares)
    if total <= 0.0:
        msg = f"Composition must sum to a positive value, got sum={total}"
        raise ValueError(msg)
    normalized = tuple(max(min(s / total, 1.0 - _EPSILON), _EPSILON) for s in shares)
    log_geom = sum(math.log(v) for v in normalized) / len(normalized)
    return tuple(math.log(v) - log_geom for v in normalized)


# ═══════════════════════════════════════════════════════════════════════
# Ideology lookup helpers
# ═══════════════════════════════════════════════════════════════════════


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
    codes.  Zero-pads codes to 5 digits so that 4-digit forms from
    departments 05 (Antioquia) and 08 (Atlántico) are not discarded.

    """
    historical = historical[historical["codigo_municipio"].notna()].copy()
    historical["codigo_municipio"] = (
        historical["codigo_municipio"].astype(str).str.strip().str.zfill(_CNP_CODE_LENGTH)
    )
    return historical[
        (historical["codigo_municipio"] != "000NA")
        & (historical["codigo_municipio"].str.len() == _CNP_CODE_LENGTH)
    ].copy()


def _compute_turnout(
    df: pd.DataFrame, historical: pd.DataFrame, data_dir: Path | None = None
) -> pd.DataFrame:
    """Compute ``historical_turnout_m`` per municipality and merge into *df*.

    Attempts to compute turnout from historical results (``registered_voters``).
    If ``registered_voters`` is unavailable (all NaN), falls back to MOE Cámara
    2022 turnout data from ``fundamentals/historical_turnout_moe.csv``.
    Municipalities still missing after MOE are filled with the mean of
    available MOE values.  If no data is available at all, the column is left
    as NaN for downstream imputation.

    Args:
        df: Feature matrix DataFrame (must have ``codigo_municipio``).
        historical: Historical results DataFrame.
        data_dir: Root data directory for fallback turnout file resolution.

    Returns:
        The feature matrix with an added ``historical_turnout_m`` column.

    """
    valid = _filter_valid_historical(historical)

    if valid.empty:
        logger.warning("No valid historical results -- setting historical_turnout_m to NaN")
        df["historical_turnout_m"] = float("nan")
        return df

    safe_registered = valid["registered_voters"].replace(0, pd.NA)

    if safe_registered.notna().any():
        valid["turnout"] = (valid["total_votes"] / safe_registered).clip(0.0, 1.0)
        # Collapse rounds into a single per-year turnout to avoid
        # double-weighting election years with runoffs (e.g., 2022).
        turnout_per_yr = valid.groupby(["codigo_municipio", "year"])["turnout"].mean().reset_index()
        turnout_mean = turnout_per_yr.groupby("codigo_municipio")["turnout"].mean()
        result = df.merge(
            turnout_mean.rename("historical_turnout_m").reset_index(),
            on="codigo_municipio",
            how="left",
        )
        # Fill Bogotá localidad NaN (7-digit codes) with Bogotá-wide turnout.
        bogota_val = turnout_mean.get(_BOGOTA_CODE)
        if bogota_val is not None and pd.notna(bogota_val):
            localidad_mask = (
                result["codigo_municipio"].astype(str).str.len() == _BOGOTA_LOCALIDAD_CODE_LENGTH
            )
            fill_mask = localidad_mask & result["historical_turnout_m"].isna()
            result.loc[fill_mask, "historical_turnout_m"] = float(bogota_val)
        if result["historical_turnout_m"].isna().any():
            missing_count = int(result["historical_turnout_m"].isna().sum())
            fill_val = turnout_mean.mean()
            logger.warning(
                "%d municipalities without historical data -- filling turnout with %.2f",
                missing_count,
                fill_val,
            )
            result["historical_turnout_m"] = result["historical_turnout_m"].fillna(fill_val)
        return result

    logger.warning("registered_voters is all NaN -- falling back to MOE 2022 turnout data")
    _moe_turnout = _load_fallback_turnout(df, data_dir=data_dir)
    _moe_turnout = _moe_turnout.rename(columns={"turnout": "historical_turnout_m"})
    result = df.merge(
        _moe_turnout[["codigo_municipio", "historical_turnout_m"]],
        on="codigo_municipio",
        how="left",
    )
    if result["historical_turnout_m"].isna().any():
        missing_count = int(result["historical_turnout_m"].isna().sum())
        known_mean = result["historical_turnout_m"].mean()
        fill_val = known_mean if pd.notna(known_mean) else float("nan")
        logger.warning(
            "%d municipalities missing from fallback turnout -- filling with %.2f",
            missing_count,
            fill_val,
        )
        result["historical_turnout_m"] = result["historical_turnout_m"].fillna(fill_val)
    if result["historical_turnout_m"].isna().all():
        msg = "historical_turnout_m is entirely NaN after fallback fill"
        raise ValueError(msg)
    return result


def _load_fallback_turnout(df: pd.DataFrame, data_dir: Path | None = None) -> pd.DataFrame:
    """Load MOE-based fallback turnout and align to feature matrix municipalities.

    Populates Bogotá localidad codes (7-digit) with Bogotá-wide turnout
    when they are missing from the cached file.
    """
    base = resolve_data_dir(data_dir)
    turnout_path = base / "fundamentals" / "historical_turnout_moe.csv"
    if turnout_path.is_file():
        loaded = pd.read_csv(turnout_path, dtype={"codigo_municipio": str})
        result = loaded[loaded["codigo_municipio"].isin(df["codigo_municipio"])].copy()
        # Fill Bogotá localidad codes missing from the cached file
        bogota_row = loaded[loaded["codigo_municipio"] == _BOGOTA_CODE]
        if not bogota_row.empty and pd.notna(bogota_row.iloc[0]["turnout"]):
            bogo_val = float(bogota_row.iloc[0]["turnout"])
            localidad_codes = df[
                df["codigo_municipio"].astype(str).str.len() == _BOGOTA_LOCALIDAD_CODE_LENGTH
            ]["codigo_municipio"]
            missing_loc = localidad_codes[~localidad_codes.isin(result["codigo_municipio"])]
            if not missing_loc.empty:
                loc_rows = pd.DataFrame(
                    {
                        "codigo_municipio": missing_loc.tolist(),
                        "turnout": bogo_val,
                    }
                )
                result = pd.concat([result, loc_rows], ignore_index=True)
        return result
    logger.warning("Fallback turnout file not found -- returning NaN turnout")
    return pd.DataFrame({"codigo_municipio": df["codigo_municipio"], "turnout": float("nan")})


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
        ideology = get_ideology(year, round_num)
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
        if safe_reg and safe_reg > 0 and pd.notna(safe_total):
            ratio = safe_total / safe_reg
            abstention = max(0.0, 1.0 - ratio) if pd.notna(ratio) else float("nan")
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
    """Coerce known boolean columns from string/int to bool.

    Raises:
        ValueError: If any value in a boolean column cannot be mapped
            via ``_BOOL_MAP``.

    """
    bool_cols = [
        "ipm_2018_imputed",
        "ipm_2022_imputed",
        "is_pdet",
        "armed_group_presence",
        "high_risk_flag",
    ]
    for col in bool_cols:
        if col in df.columns and not pd.api.types.is_bool_dtype(df[col]):
            mapped = df[col].astype(str).str.strip().str.lower().map(_BOOL_MAP)
            if mapped.isna().any():
                bad_mask = mapped.isna()
                bad_entries = list(
                    zip(
                        df.index[bad_mask].tolist(),
                        df.loc[bad_mask, col].tolist(),
                        strict=True,
                    )
                )
                msg = (
                    f"Unrecognized boolean value(s) in column '{col}': "
                    f"{bad_entries}. "
                    f"Accepted values: {sorted(_BOOL_MAP)}"
                )
                raise ValueError(msg)
            df[col] = mapped.astype(bool)


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
    - ``historical_turnout_m`` is in [0, 1].
    - ``pop_*`` columns are present and non-null.
    - Null rates are within acceptable thresholds.

    Args:
        df: The feature matrix DataFrame to validate.

    Raises:
        ValueError: If any validation check fails.

    """
    mf_fields = {f.name for f in fields(MunicipalFeatures)}
    scalar_fields = mf_fields - {"historical"}

    # IPM is an optional component (dropped by R² guard when R² < 0.5).
    _optional_ipm_fields = {"ipm_2018", "ipm_2018_imputed", "ipm_2022", "ipm_2022_imputed"}

    missing = scalar_fields - set(df.columns) - _optional_ipm_fields
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

    null_pop = {c: int(df[c].isna().sum()) for c in pop_cols if df[c].isna().any()}
    if null_pop:
        msg = f"Population columns contain nulls: {null_pop}"
        raise ValueError(msg)

    _validate_null_rate(df, scalar_fields - _optional_ipm_fields)


def _validate_codigo_municipio(df: pd.DataFrame) -> None:
    """Validate ``codigo_municipio`` uniqueness and length.

    Accepts both 5-digit DANE codes and 7-digit Bogotá localidad codes
    (e.g. ``"1100101"`` for Usaquén).

    """
    if not df["codigo_municipio"].is_unique:
        msg = "codigo_municipio is not unique"
        raise ValueError(msg)
    bad_length = int(
        df["codigo_municipio"]
        .astype(str)
        .str.len()
        .isin({_CNP_CODE_LENGTH, _BOGOTA_LOCALIDAD_CODE_LENGTH})
        .value_counts()
        .get(False, 0)
    )
    if bad_length > 0:
        msg = (
            f"{bad_length} codigo_municipio value(s) with length != "
            f"{_CNP_CODE_LENGTH} or {_BOGOTA_LOCALIDAD_CODE_LENGTH}"
        )
        raise ValueError(msg)


def _validate_nbi_rate(df: pd.DataFrame) -> None:
    """Validate ``nbi_rate`` is in [0, 1]."""
    vals = df["nbi_rate"].dropna()
    if len(vals) > 0 and not vals.between(0.0, 1.0).all():
        msg = "nbi_rate has values outside [0, 1]"
        raise ValueError(msg)


def _validate_turnout(df: pd.DataFrame) -> None:
    """Validate ``historical_turnout_m`` range."""
    vals = df["historical_turnout_m"].dropna()
    if len(vals) > 0 and not vals.between(0.0, 1.0).all():
        msg = "historical_turnout_m has values outside [0, 1]"
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
        1142
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

    # Drop municipalities with no population data (e.g. 27086 Belén de Bajirá
    # which exists in DIVIPOLA but has no DANE PPED projections).
    pop_cols = [f"pop_{y}" for y in range(2018, 2027)]
    available_pop = [c for c in pop_cols if c in df.columns]
    if available_pop:
        before = len(df)
        df = df.dropna(subset=available_pop, how="all")
        dropped = before - len(df)
        if dropped > 0:
            logger.warning(
                "Dropped %d municipality/municipalities with no population data", dropped
            )

    historical = _load_historical_results(base)
    logger.info("Loaded historical results: %d rows", len(historical))

    df = _compute_turnout(df, historical, data_dir=data_dir)
    logger.debug("Computed historical_turnout_m")

    df = _build_historical_column(df, historical)
    logger.debug("Built historical records column")

    _validate_schema(df)
    logger.info("Feature matrix validated against MunicipalFeatures schema")

    return df
