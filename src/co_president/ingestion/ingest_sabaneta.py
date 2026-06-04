"""SPEC-16: Sabaneta Cámara de Representantes historical results loader.

Parses the Sabaneta municipal CSV (81-line, UTF-8) containing Cámara de
Representantes vote totals for 2002--2022 at the (period, party) level.
Returns a long-form DataFrame suitable for ingestion into the feature
matrix pipeline.

The file serves as a *gold-standard fixture* for the party-name-to-vote
schema that the 2026 election results will also use.
"""

from __future__ import annotations

import logging
from pathlib import Path  # noqa: TC003 — needed at runtime by typing.get_type_hints in tests

import pandas as pd

from co_president.paths import resolve_data_dir

__all__ = [
    "build_sabaneta_camara_matrix",
    "load_sabaneta_camara",
    "validate_sabaneta",
]

logger = logging.getLogger(__name__)

_EXPECTED_PERIODS: frozenset[int] = frozenset({2002, 2006, 2010, 2015, 2019})
_EXPECTED_COLUMNS: frozenset[str] = frozenset(
    {"municipio", "periodo", "partido", "total_votes"},
)


def validate_sabaneta(df: pd.DataFrame) -> list[str]:
    """Validate a Sabaneta Camara DataFrame against acceptance criteria.

    Checks for expected columns, expected period count, and non-negative
    vote totals.

    Args:
        df: The Sabaneta DataFrame to validate.

    Returns:
        List of warning messages (empty if all checks pass).

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "municipio": ["Sabaneta", "Sabaneta"],
        ...     "periodo": [2002, 2006],
        ...     "partido": ["Liberal", "Conservador"],
        ...     "total_votes": [1525, 837],
        ... })
        >>> validate_sabaneta(df)
        []

    """
    warnings: list[str] = []
    missing = _EXPECTED_COLUMNS - set(df.columns)
    if missing:
        warnings.append(f"Missing columns: {sorted(missing)}")
        return warnings
    actual_periods = df["periodo"].dropna().astype(int)
    actual_set = frozenset(actual_periods)
    if actual_set != _EXPECTED_PERIODS:
        warnings.append(
            f"Expected periods {_EXPECTED_PERIODS}, got {actual_set}",
        )
    if (df["total_votes"] < 0).any():
        warnings.append("Found negative vote totals")
    return warnings


def load_sabaneta_camara(data_dir: Path | None = None) -> pd.DataFrame:
    """Load the Sabaneta Cámara de Representantes 2002--2022 file.

    Args:
        data_dir: Project data directory.  If ``None``, resolves via
            ``resolve_data_dir``.

    Returns:
        Long-form DataFrame with columns ``municipio``, ``periodo``,
        ``partido``, ``total_votes`` — one row per (period, party).

    Raises:
        FileNotFoundError: If the Sabaneta CSV is not found in the
            ``sabaneta/`` subdirectory of *data_dir*.

    """
    base = resolve_data_dir(data_dir)
    sabaneta_dir = base / "sabaneta"
    return _parse_sabaneta_file(sabaneta_dir)


def _parse_sabaneta_file(sabaneta_dir: Path) -> pd.DataFrame:
    """Parse the Sabaneta CSV from *sabaneta_dir* into a long-form DataFrame.

    Args:
        sabaneta_dir: Directory containing the Sabaneta CSV file.

    Returns:
        DataFrame with columns ``municipio``, ``periodo``, ``partido``,
        ``total_votes``.

    Raises:
        FileNotFoundError: If no CSV is found in *sabaneta_dir*.
        ValueError: If the CSV is empty, contains non-numeric Total values,
            or has malformed Periodo strings.

    """
    csvs = sorted(sabaneta_dir.glob("*.csv"))
    if not csvs:
        msg = f"No CSV file found in {sabaneta_dir}"
        raise FileNotFoundError(msg)

    raw = pd.read_csv(
        csvs[0],
        encoding="utf-8",
        dtype={"Partido Político": str, "Periodo": str, "Total": str},
    )
    if raw.empty:
        msg = f"Sabaneta CSV is empty (no data rows): {csvs[0]}"
        raise ValueError(msg)

    # Strip comma thousands separators and cast to int.
    _total_cleaned = raw["Total"].str.replace(",", "", regex=False).str.strip()
    _total_numeric = pd.to_numeric(_total_cleaned, errors="coerce")
    _bad_total = _total_numeric.isna()
    if _bad_total.any():
        _offending = raw.loc[_bad_total, "Total"].to_dict()
        msg = (
            f"Non-numeric Total values at row indices "
            f"{list(_offending.keys())}: {list(_offending.values())}"
        )
        raise ValueError(msg)
    raw["Total"] = _total_numeric.astype(int)

    # Extract the start year from the "YYYY/YYYY" period string.
    _periodo_extracted = raw["Periodo"].str.extract(r"(\d{4})", expand=False)
    _periodo_numeric = pd.to_numeric(_periodo_extracted, errors="coerce")
    _bad_periodo = _periodo_numeric.isna()
    if _bad_periodo.any():
        _offending = raw.loc[_bad_periodo, "Periodo"].to_dict()
        msg = (
            f"Failed to extract year from Periodo values at row indices "
            f"{list(_offending.keys())}: {list(_offending.values())}"
        )
        raise ValueError(msg)
    raw["periodo"] = _periodo_numeric.astype(int)
    # Rename and select columns.
    result = raw.rename(
        columns={
            "Partido Político": "partido",
            "Total": "total_votes",
        },
    )
    result["municipio"] = "Sabaneta"
    return result[["municipio", "periodo", "partido", "total_votes"]].copy()


def build_sabaneta_camara_matrix(data_dir: Path | None = None) -> None:
    """Load, parse, and save the Sabaneta Camara de Representantes data.

    Args:
        data_dir: Target data directory.  If ``None``, resolves via
            ``resolve_data_dir``.

    Returns:
        ``None``.  Results are written to
        ``data_dir/fundamentals/sabaneta_camara.csv``.

    Raises:
        FileNotFoundError: If the Sabaneta CSV source file is not found.
        ValueError: If the CSV is empty or contains unparseable values.
        OSError: If the ``fundamentals/`` output directory cannot be
            created or the CSV cannot be written.

    Examples:
        >>> build_sabaneta_camara_matrix()  # writes to default data dir

    """
    df = load_sabaneta_camara(data_dir)
    target_dir = resolve_data_dir(data_dir) / "fundamentals"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "sabaneta_camara.csv"
    df.to_csv(target_path, index=False)
    logger.info(
        "Sabaneta Camara data saved to %s (%d records)",
        target_path,
        len(df),
    )
