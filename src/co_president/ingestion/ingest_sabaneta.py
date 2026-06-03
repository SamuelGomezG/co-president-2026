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
    "load_sabaneta_camara",
]

logger = logging.getLogger(__name__)


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
    # Strip comma thousands separators and cast to int.
    raw["Total"] = (
        raw["Total"]
        .str.replace(",", "", regex=False)
        .str.strip()
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(int)
    )
    # Extract the start year from the "YYYY/YYYY" period string.
    raw["periodo"] = (
        raw["Periodo"]
        .str.extract(r"(\d{4})", expand=False)
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(int)
    )
    # Rename and select columns.
    result = raw.rename(
        columns={
            "Partido Político": "partido",
            "Total": "total_votes",
        },
    )
    result["municipio"] = "Sabaneta"
    return result[["municipio", "periodo", "partido", "total_votes"]].copy()
