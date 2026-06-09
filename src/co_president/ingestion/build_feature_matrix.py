"""SPEC-14: Municipal feature matrix integration (extended by SPEC-18, SPEC-19).

Loads all eight fundamental components (DIVIPOLA, historical results,
NBI poverty indicators, IPM poverty indicators, socioeconomic indicators,
risk factors, CNPV 2018 census data, and DANE population projections)
produced by SPEC-12, SPEC-13, SPEC-17, SPEC-18, and SPEC-19, joins them
on ``codigo_municipio``, and produces a single validated municipal
feature matrix ready for modeling.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from co_president.ingestion.ingest_bogota import (
    disaggregate_bogota_election_results,
    disaggregate_bogota_population,
    get_bogota_localidad_rows,
)
from co_president.paths import resolve_data_dir

__all__ = [
    "build_feature_matrix",
    "generate_data_dictionary",
    "load_all_components",
    "pivot_historical_wide",
    "save_feature_matrix",
    "validate_component_health",
]

logger = logging.getLogger(__name__)

_BOGOTA_CODE = "11001"

_COMPONENT_FILES: dict[str, str] = {
    "divipola": "divipola_master.csv",
    "historical": "historical_results.csv",
    "nbi": "nbi_2018.csv",
    "ipm": "ipm_2018.csv",
    "socioeconomic": "socioeconomic.csv",
    "risk": "risk_factors.csv",
    "cnpv": "cnpv_2018.csv",
    "population": "population_2018_2026.csv",
    "fiscal": "fiscal.csv",
}

_OPTIONAL_COMPONENTS: frozenset[str] = frozenset({"ipm"})

# NOTE(SPEC-17): ``cnpv_2018.csv`` shares column names with the
# ``socioeconomic.csv`` stub (pct_afro_colombian, pct_indigenous, etc.).
# ``build_feature_matrix`` appends ``_x``/``_y`` suffixes during merge.
# SPEC-21 (Fundamentals API) will reconcile; this file defers that choice.


def load_all_components(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all fundamental components from disk.

    Each component CSV is loaded from ``data_dir / "fundamentals" / filename``.
    Missing or unparseable files produce an empty DataFrame (with a logged
    warning) so that ``validate_component_health`` can report the issue
    rather than raising.

    Args:
        data_dir: Root data directory containing the ``fundamentals/`` subdirectory.

    Returns:
        Dict mapping component names to DataFrames.  Keys match
        ``_COMPONENT_FILES``.

    Examples:
        >>> components = load_all_components(Path("data"))
        >>> isinstance(components, dict)
        True
        >>> "divipola" in components
        True

    """
    fundamentals_dir = data_dir / "fundamentals"
    components: dict[str, pd.DataFrame] = {}
    for name, filename in _COMPONENT_FILES.items():
        path = fundamentals_dir / filename
        if not path.is_file():
            logger.warning("Component file not found: %s", path)
            components[name] = pd.DataFrame()
            continue
        try:
            df = pd.read_csv(path, dtype={"codigo_municipio": str})
            if "codigo_municipio" in df.columns:
                df["codigo_municipio"] = df["codigo_municipio"].astype(str)
            components[name] = df
            logger.info(
                "Loaded %s: %d rows, %d columns",
                name,
                len(df),
                len(df.columns),
            )
        except (OSError, pd.errors.ParserError, ValueError) as exc:
            logger.warning("Failed to load component %s from %s: %s", name, path, exc)
            components[name] = pd.DataFrame()
    return components


_STUB_THRESHOLDS: dict[str, int] = {
    "socioeconomic": 5,
    "risk": 15,
}


def validate_component_health(
    components: dict[str, pd.DataFrame],
    *,
    strict: bool = False,
    thresholds: dict[str, int] | None = None,
) -> list[str]:
    """Check each component for basic data quality.

    Inspects every DataFrame for:
    - Presence of ``codigo_municipio`` column
    - Null values in ``codigo_municipio``
    - Duplicate ``codigo_municipio`` values
    - Stub detection for ``socioeconomic`` (<= 5 rows) and ``risk`` (<= 15 rows)
    - Placeholder detection for ``historical`` (contains ``codigo_municipio == "000NA"``)

    Optional components (``ipm``) do not produce empty-DataFrame warnings;
    they are silently tolerated when the file is missing.

    When *strict* is ``True``, stub and placeholder conditions raise
    ``ValueError`` instead of returning warning strings.

    Args:
        components: Dict of component DataFrames (as returned by
            ``load_all_components``).
        strict: If ``True``, raise ``ValueError`` on stub/placeholder
            components instead of appending warnings.
        thresholds: Optional per-component stub thresholds.  If ``None``,
            defaults from ``_STUB_THRESHOLDS`` are used.

    Returns:
        List of warning messages.  An empty list signals a clean bill of
        health.

    Raises:
        ValueError: When *strict* is ``True`` and a stub or placeholder
            is detected.

    Examples:
        >>> components = {"divipola": pd.DataFrame({"codigo_municipio": ["05001"]})}
        >>> validate_component_health(components)
        []

    """
    if thresholds is None:
        thresholds = dict(_STUB_THRESHOLDS)
    warnings: list[str] = []

    for name, df in components.items():
        _check_component_basics(name, df, warnings)
        _check_stub(name, df, warnings, strict=strict, threshold=thresholds.get(name))
        _check_historical_placeholder(name, df, warnings, strict=strict)

    return warnings


def _check_component_basics(
    name: str,
    df: pd.DataFrame,
    warnings: list[str],
) -> None:
    """Check for empty DataFrame, missing column, nulls, and duplicates."""
    if df.empty:
        if name not in _OPTIONAL_COMPONENTS:
            warnings.append(f"{name}: empty DataFrame (no data loaded)")
        return

    if "codigo_municipio" not in df.columns:
        warnings.append(f"{name}: missing codigo_municipio column")
        return

    null_count = int(df["codigo_municipio"].isna().sum())
    if null_count > 0:
        msg = f"{name}: {null_count} null codigo_municipio value"
        if null_count != 1:
            msg += "s"
        warnings.append(msg)

    # Historical is long-format — duplicates expected.
    if name != "historical":
        dup_count = int(df["codigo_municipio"].duplicated().sum())
        if dup_count > 0:
            msg = f"{name}: {dup_count} duplicate codigo_municipio value"
            if dup_count != 1:
                msg += "s"
            warnings.append(msg)


def _check_stub(
    name: str,
    df: pd.DataFrame,
    warnings: list[str],
    *,
    strict: bool,
    threshold: int | None,
) -> None:
    """Check if the component is a stub (too few rows)."""
    if df.empty:
        return
    if threshold is None or len(df) > threshold:
        return
    msg = (
        f"{name}: stub detected ({len(df)} rows, expected > {threshold}). "
        "Verify ingestion pipeline completed successfully."
    )
    if strict:
        raise ValueError(msg)
    warnings.append(msg)


def _check_historical_placeholder(
    name: str,
    df: pd.DataFrame,
    warnings: list[str],
    *,
    strict: bool,
) -> None:
    """Check historical data for ``000NA`` sentinel municipality codes."""
    if name != "historical" or df.empty or "codigo_municipio" not in df.columns:
        return
    placeholder_count = int(df["codigo_municipio"].eq("000NA").sum())
    if placeholder_count == 0:
        return
    msg = (
        f"historical: contains {placeholder_count} placeholder row(s) "
        "with codigo_municipio='000NA'. Verify CEDAE ingestion."
    )
    if strict:
        raise ValueError(msg)
    warnings.append(msg)


def pivot_historical_wide(historical: pd.DataFrame) -> pd.DataFrame:
    """Pivot historical election results to wide format.

    Transforms from long format (one row per municipality-year-round-candidate)
    to wide format (one row per municipality, one column per
    candidate-year-round combination).

    Args:
        historical: DataFrame with columns ``codigo_municipio``, ``year``,
            ``round``, ``candidate``, and ``vote_share`` (among others).

    Returns:
        DataFrame with one row per municipality and columns named
        ``vote_share_{year}_r{round}_{candidate}``.

    Raises:
        ValueError: If required columns are missing.

    Examples:
        >>> df = pd.DataFrame({"codigo_municipio": ["05001"], "year": [2022],
        ...                    "round": [1], "candidate": ["A"], "vote_share": [0.5]})
        >>> wide = pivot_historical_wide(df)
        >>> "vote_share_2022_r1_A" in wide.columns
        True

    """
    required = {"codigo_municipio", "year", "round", "candidate", "vote_share"}
    missing = required - set(historical.columns)
    if missing:
        msg = f"Historical DataFrame missing required columns: {sorted(missing)}"
        raise ValueError(msg)

    if historical.empty:
        return pd.DataFrame(columns=["codigo_municipio"])

    pivoted = historical.pivot_table(
        index="codigo_municipio",
        columns=["year", "round", "candidate"],
        values="vote_share",
        aggfunc="first",
    )

    pivoted.columns = [
        f"vote_share_{year}_r{round_}_{cand}" for year, round_, cand in pivoted.columns
    ]

    return pivoted.reset_index()


def _merge_components(
    matrix: pd.DataFrame,
    components: list[tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Sequentially left-merge component DataFrames onto the matrix.

    Each component is merged on ``codigo_municipio`` with ``validate="m:1"``
    only if it is non-empty.  Empty components are silently skipped.

    Args:
        matrix: The anchor DataFrame (typically DIVIPOLA).
        components: Ordered list of ``(name, dataframe)`` tuples.

    Returns:
        The matrix with all non-empty components merged in order.

    """
    for name, df in components:
        if df.empty:
            logger.debug("Skipping empty component: %s", name)
            continue
        matrix = matrix.merge(df, on="codigo_municipio", how="left", validate="m:1")
    return matrix


def _replace_bogota_with_localidades(  # noqa: C901
    components: dict[str, pd.DataFrame],
    data_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Replace Bogotá D.C. municipality row with 21 localidad rows.

    Modifies the DIVIPOLA, historical, and population components:
    - DIVIPOLA: Bogotá row replaced by 21 localidad rows (7-digit codes)
    - Historical: Bogotá rows replaced by per-localidad election results
    - Population: Bogotá row replaced by proportionally split rows

    All other components (NBI, IPM, risk, CNPV, socioeconomic) keep
    Bogotá's row — localidades inherit Bogotá-wide rates via subsequent
    merge.

    If the Bogotá MMV/reg-participacion data files are not available
    (e.g. in test environments), the replacement is skipped entirely and
    the components are returned unchanged.

    Args:
        components: Dict of component DataFrames.
        data_dir: Root data directory (for reading MMV/reg data).

    Returns:
        Updated components dict.

    """
    # Probe for Bogotá data files before making any changes.
    mmv_path_1v = data_dir / "2022-presidential-results" / "MMV_NACIONAL_PRESIDENTE_2022_1v.csv.gz"
    reg_path_1v = data_dir / "2022-presidential-results" / "reg_participacion_vuelta1.csv"
    if not mmv_path_1v.is_file() or not reg_path_1v.is_file():
        logger.info(
            "Bogotá MMV/reg data not found (%s, %s) — skipping localidad disaggregation",
            mmv_path_1v,
            reg_path_1v,
        )
        return components

    result = dict(components)

    # ── DIVIPOLA ──────────────────────────────────────────────────
    divipola = result["divipola"]
    if not divipola.empty and "codigo_municipio" in divipola.columns:
        non_bogota = divipola[divipola["codigo_municipio"] != _BOGOTA_CODE]
        localidad_rows = get_bogota_localidad_rows()
        for col in divipola.columns:
            if col not in localidad_rows.columns:
                localidad_rows[col] = None
        result["divipola"] = pd.concat([non_bogota, localidad_rows], ignore_index=True)
        logger.info(
            "DIVIPOLA: replaced Bogotá with %d localidad rows (total %d)",
            len(localidad_rows),
            len(result["divipola"]),
        )

    # ── Historical results ────────────────────────────────────────
    historical = result["historical"]
    if not historical.empty and "codigo_municipio" in historical.columns:
        non_bogota_hist = historical[historical["codigo_municipio"] != _BOGOTA_CODE]
        try:
            localidad_hist = disaggregate_bogota_election_results(data_dir)
        except FileNotFoundError:
            logger.warning("Bogotá MMV/reg data not found — skipping historical disaggregation")
            localidad_hist = pd.DataFrame()
        if not localidad_hist.empty:
            result["historical"] = pd.concat([non_bogota_hist, localidad_hist], ignore_index=True)
            logger.info(
                "Historical: replaced Bogotá rows with %d localidad rows (total %d)",
                len(localidad_hist),
                len(result["historical"]),
            )

    # ── Population projections ────────────────────────────────────
    population = result["population"]
    if not population.empty and "codigo_municipio" in population.columns:
        non_bogota_pop = population[population["codigo_municipio"] != _BOGOTA_CODE]
        try:
            localidad_pop = disaggregate_bogota_population(data_dir)
        except FileNotFoundError:
            logger.warning("Bogotá MMV/reg data not found — skipping population disaggregation")
            localidad_pop = pd.DataFrame()
        if not localidad_pop.empty:
            result["population"] = pd.concat([non_bogota_pop, localidad_pop], ignore_index=True)
            logger.info(
                "Population: replaced Bogotá row with %d localidad rows (total %d)",
                len(localidad_pop),
                len(result["population"]),
            )

    return result


def build_feature_matrix(data_dir: Path | None = None) -> pd.DataFrame:
    """Build the final municipal feature matrix by joining all components.

    Loads all seven fundamental components from disk, pivots historical
    results to wide format, and sequentially LEFT-joins them anchored on
    the DIVIPOLA master registry.

    Args:
        data_dir: Root data directory.  If ``None``, resolves via
            ``resolve_data_dir``.

    Returns:
        DataFrame with exactly one row per municipality and all feature
        columns from every component.

    Raises:
        FileNotFoundError: If the data directory cannot be resolved.

    Examples:
        >>> matrix = build_feature_matrix(Path("data"))
        >>> matrix.shape[1] > 0
        True

    """
    if data_dir is None:
        data_dir = resolve_data_dir(None)

    components = load_all_components(data_dir)
    warnings = validate_component_health(components)
    for warning in warnings:
        logger.warning("Component health: %s", warning)

    components = _replace_bogota_with_localidades(components, data_dir)

    divipola = components["divipola"]
    historical = components["historical"]
    nbi = components["nbi"]
    ipm = components["ipm"]
    socioeconomic = components["socioeconomic"]
    risk = components["risk"]
    cnpv = components["cnpv"]
    population = components["population"]
    fiscal = components["fiscal"]

    if divipola.empty:
        logger.error("DIVIPOLA component is empty — cannot anchor feature matrix")
        return pd.DataFrame()

    historical_wide = pivot_historical_wide(historical) if not historical.empty else pd.DataFrame()

    matrix = divipola.copy()
    matrix = _merge_components(
        matrix,
        [
            ("historical_wide", historical_wide),
            ("nbi", nbi),
            ("ipm", ipm),
            ("socioeconomic", socioeconomic),
            ("risk", risk),
            ("cnpv", cnpv),
            ("population", population),
            ("fiscal", fiscal),
        ],
    )

    _log_matrix_stats(matrix)
    return matrix


def generate_data_dictionary(matrix: pd.DataFrame, output_path: str) -> None:
    """Write a human-readable data dictionary as a Markdown file.

    For every column in *matrix* the dictionary records the data type,
    count of null values, and a sample non-null value.

    Args:
        matrix: The municipal feature matrix.
        output_path: File path for the generated Markdown file.

    Returns:
        ``None`` — writes output to disk as a side effect.

    Examples:
        >>> matrix = pd.DataFrame({"x": [1, 2, 3]})
        >>> generate_data_dictionary(matrix, "/tmp/dictionary.md")

    """
    lines: list[str] = [
        "# Municipal Feature Matrix — Data Dictionary",
        "",
        f"Generated: {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}",
        f"Rows: {len(matrix)}, Columns: {len(matrix.columns)}",
        "",
    ]

    for col in matrix.columns:
        dtype = matrix[col].dtype
        nulls = int(matrix[col].isna().sum())
        non_null = matrix[col].dropna()
        sample = str(non_null.iloc[0]) if len(non_null) > 0 else "ALL NULL"
        lines.extend(
            [
                f"## {col}",
                "",
                f"- **Type**: {dtype}",
                f"- **Nulls**: {nulls} / {len(matrix)}",
                f"- **Sample**: {sample}",
                "",
            ]
        )

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def save_feature_matrix(matrix: pd.DataFrame, data_dir: Path | None = None) -> None:
    """Save the feature matrix in multiple formats with supporting artifacts.

    Writes:
    - ``municipal_feature_matrix.csv`` (human-readable CSV)
    - ``municipal_feature_matrix.parquet`` (efficient loading)
    - ``feature_dictionary.md`` (column-by-column data dictionary)
    - ``build_summary.json`` (build statistics)

    Args:
        matrix: The municipal feature matrix to save.
        data_dir: Root data directory.  If ``None``, resolves via
            ``resolve_data_dir``.

    Returns:
        ``None`` — writes CSV, Parquet, data dictionary, and summary
        JSON as side effects.

    Raises:
        ValueError: If ``matrix`` is empty.

    Examples:
        >>> matrix = pd.DataFrame({"x": [1, 2]})
        >>> save_feature_matrix(matrix, Path("/tmp"))

    """
    if data_dir is None:
        data_dir = resolve_data_dir(None)

    if matrix.empty:
        msg = "Cannot save an empty feature matrix"
        raise ValueError(msg)

    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    csv_path = processed_dir / "municipal_feature_matrix.csv"
    matrix.to_csv(csv_path, index=False)
    logger.info("Saved CSV: %s (%d rows, %d cols)", csv_path, len(matrix), len(matrix.columns))

    parquet_path = processed_dir / "municipal_feature_matrix.parquet"
    matrix.to_parquet(parquet_path, index=False)
    logger.info("Saved Parquet: %s", parquet_path)

    dict_path = processed_dir / "feature_dictionary.md"
    generate_data_dictionary(matrix, str(dict_path))
    logger.info("Saved data dictionary: %s", dict_path)

    total_cells = len(matrix) * len(matrix.columns)
    null_count = int(matrix.isna().sum().sum())
    null_rate = null_count / total_cells if total_cells > 0 else 0.0
    memory_bytes = matrix.memory_usage(deep=True).sum()
    memory_mb = memory_bytes / 1024 / 1024

    summary: dict[str, object] = {
        "municipalities": len(matrix),
        "columns": len(matrix.columns),
        "null_count": null_count,
        "null_rate": round(null_rate, 4),
        "memory_bytes": int(memory_bytes),
        "memory_mb": round(memory_mb, 2),
        "column_names": _summarize_columns(matrix),
    }
    summary_path = processed_dir / "build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Saved build summary: %s", summary_path)


# ═══════════════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════════════


def _summarize_columns(matrix: pd.DataFrame) -> list[dict[str, str | int | float]]:
    """Build a compact column summary for the build summary JSON."""
    return [
        {
            "name": col,
            "dtype": str(matrix[col].dtype),
            "nulls": int(matrix[col].isna().sum()),
        }
        for col in matrix.columns
    ]


def _log_matrix_stats(matrix: pd.DataFrame) -> None:
    """Log summary statistics about the built feature matrix."""
    null_total = int(matrix.isna().sum().sum())
    total_cells = len(matrix) * len(matrix.columns)
    logger.info(
        "Feature matrix built: %d municipalities x %d columns",
        len(matrix),
        len(matrix.columns),
    )
    if null_total > 0:
        logger.warning(
            "Feature matrix contains %d null values (%.1f%% of %d cells)",
            null_total,
            null_total / total_cells * 100,
            total_cells,
        )
