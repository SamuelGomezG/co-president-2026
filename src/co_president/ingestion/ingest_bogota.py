"""SPEC-21b: Bogotá D.C. localidad disaggregation.

Disaggregates Bogotá D.C. (DANE code ``11001``) into its 20 localidades
plus a SIN COMUNA catch-all row.  Combines three data sources:

1. **Electoral results**: joins ``MMV`` ⋈ ``reg_participacion`` on
   normalised polling-station name to assign per-candidate votes to each
   localidad, then aggregates.
2. **CNPV 2018 demographics**: localidad-level microdata is unavailable
   (``UA1_LOCALIDAD`` only exists in F12, which is absent from our zip
   files).  All Bogotá localidades inherit Bogotá-wide demographic rates.
3. **Population projections**: Bogotá's municipal DANE PPED projection
   is split across localidades proportionally to each localidad's share
   of Bogotá's registered-voter census (``Total censo`` from
   ``reg_participacion``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from co_president.config import BOGOTA_LOCALIDADES
from co_president.paths import resolve_data_dir

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "disaggregate_bogota_election_results",
    "disaggregate_bogota_population",
    "get_bogota_localidad_rows",
]

logger = logging.getLogger(__name__)

_BOGOTA_DANE_MUN = "11001"
_BOGOTA_DEP_CODE = 11
_BOGOTA_MUN_CODE = 1

_BOGOTA_CODIGO_MUNICIPIO = 11001
_PUESTO_CODE_LENGTH = 11
_MMV_RECONSTRUCT_LENGTH = 9
_LOCALIDAD_START = 5
_LOCALIDAD_END = 7

_MMV_FILE_1V = "MMV_NACIONAL_PRESIDENTE_2022_1v.csv.gz"
_MMV_FILE_2V = "MMV_NACIONAL_PRESIDENTE_2022_2v.csv.gz"
_REG_PART_1V = "reg_participacion_vuelta1.csv"
_REG_PART_2V = "reg_participacion_vuelta2.csv"

_LOCALIDAD_CODE_LENGTH = 7


# ═══════════════════════════════════════════════════════════════════════
# Puesto name normalisation
# ═══════════════════════════════════════════════════════════════════════


def _normalize_puesto(name: object) -> str:
    """Normalise a polling-station name for cross-source matching.

    Strips whitespace, lowercases, removes accents, and collapses runs of
    non-alphanumeric characters to a single space.

    Args:
        name: Raw polling-station name.

    Returns:
        Normalised string usable as a join key.

    """
    import unicodedata  # noqa: PLC0415

    raw = str(name).strip().lower()
    raw = unicodedata.normalize("NFKD", raw).encode("ascii", errors="ignore").decode("ascii")
    return " ".join(raw.split()).strip()


# ═══════════════════════════════════════════════════════════════════════
# Localidad code helpers
# ═══════════════════════════════════════════════════════════════════════


def _extract_localidad_code(codigo_puesto: object) -> str | None:
    """Extract the 2-digit localidad code from a Bogotá 11-digit puesto code.

    Args:
        codigo_puesto: The 11-digit polling-station code.

    Returns:
        2-digit localidad code string, or ``None`` if the code cannot
        be parsed.

    """
    raw = str(codigo_puesto).strip()
    if len(raw) != _PUESTO_CODE_LENGTH or not raw.startswith(_BOGOTA_DANE_MUN):
        return None
    return raw[_LOCALIDAD_START:_LOCALIDAD_END]


def _build_7digit_code(localidad_code: str | None) -> str:
    """Build a 7-digit DANE code for a Bogotá localidad.

    Args:
        localidad_code: 2-digit localidad code (e.g. ``"01"`` for Usaquén)
            or ``"99"`` for the SIN COMUNA catch-all.

    Returns:
        7-digit code (e.g. ``"1100101"``).

    """
    return _BOGOTA_DANE_MUN + (localidad_code or "99")


def _localidad_name(localidad_code: str | None) -> str | None:
    """Return the human-readable localidad name for a given code.

    Args:
        localidad_code: 2-digit localidad code.

    Returns:
        Localidad name, or ``None`` if the code is unknown.

    """
    return BOGOTA_LOCALIDADES.get(localidad_code or "99")


def _classify_localidad(localidad_code: str | None) -> str:
    """Map a 2-digit localidad code to its final 2-digit bucket.

    Codes 01-20 map to themselves.  Code 00 (SIN COMUNA) and codes 21-32
    (non-admin zones) map to the catch-all code ``"99"``.

    Args:
        localidad_code: Raw 2-digit code from the polling-station data.

    Returns:
        2-digit bucket code: ``"01"`` through ``"20"`` for known
        localidades, ``"99"`` for the catch-all.

    """
    if localidad_code and localidad_code in BOGOTA_LOCALIDADES:
        return localidad_code
    return "99"


# ═══════════════════════════════════════════════════════════════════════
# Data loaders
# ═══════════════════════════════════════════════════════════════════════


def _load_bogota_mmv(data_dir: Path, round_num: int) -> pd.DataFrame:
    """Load MMV data for Bogotá D.C. for a given round.

    Args:
        data_dir: Root data directory.
        round_num: 1 or 2.

    Returns:
        MMV DataFrame for Bogotá (DEP=11, MUN=1).

    """
    fname = _MMV_FILE_1V if round_num == 1 else _MMV_FILE_2V
    path = data_dir / "2022-presidential-results" / fname
    if not path.is_file():
        msg = f"MMV file not found: {path}"
        raise FileNotFoundError(msg)
    df = pd.read_csv(path, sep=";", encoding="latin-1", low_memory=False)  # type: ignore[reportUnknownMemberType]
    bog = df[(df["DEP"] == _BOGOTA_DEP_CODE) & (df["MUN"] == _BOGOTA_MUN_CODE)].copy()
    bog["round"] = round_num
    bog["year"] = 2022
    return bog


def _load_bogota_reg_participacion(data_dir: Path, round_num: int) -> pd.DataFrame:
    """Load reg_participacion data for Bogotá for a given round.

    Args:
        data_dir: Root data directory.
        round_num: 1 or 2.

    Returns:
        Reg_participacion DataFrame for Bogotá.

    """
    fname = _REG_PART_1V if round_num == 1 else _REG_PART_2V
    path = data_dir / "2022-presidential-results" / fname
    if not path.is_file():
        msg = f"Reg_participacion file not found: {path}"
        raise FileNotFoundError(msg)
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False, on_bad_lines="warn")  # type: ignore[reportUnknownMemberType]
    bog = df[df["Código Municipio"] == _BOGOTA_CODIGO_MUNICIPIO].copy()
    bog["round"] = round_num
    return bog


# ═══════════════════════════════════════════════════════════════════════
# MMV ↔ reg_participacion join
# ═══════════════════════════════════════════════════════════════════════


def _prepare_mmv_join(mmv: pd.DataFrame) -> pd.DataFrame:
    """Normalise MMV data for a puesto-name join.

    Builds the join key as the ``(year, round, puesto_name_norm)`` tuple,
    and pre-aggregates votes per candidate per puesto (summing across
    mesas within each puesto).

    Args:
        mmv: Raw MMV DataFrame for Bogotá.

    Returns:
        DataFrame with columns ``year``, ``round``, ``_join_key``,
        ``CAN``, ``CANNOMBRE``, ``VOTOS``, aggregated to the
        (puesto, candidate) level.

    """
    mmv["_puesto_norm"] = mmv["PUESNOMBRE"].apply(_normalize_puesto)
    mmv["_join_key"] = (
        mmv["year"].astype(str) + "_" + mmv["round"].astype(str) + "_" + mmv["_puesto_norm"]
    )

    agg = mmv.groupby(["_join_key", "CAN", "CANNOMBRE"], as_index=False)[["VOTOS"]].sum()
    agg["year"] = 2022
    agg["round"] = mmv["round"].iloc[0]
    return agg


def _prepare_reg_join(reg: pd.DataFrame) -> pd.DataFrame:
    """Normalise reg_participacion data for a puesto-name join.

    Extracts the localidad code from ``Código Puesto``, builds the
    same join key as MMV, and collapses to one row per puesto.

    Args:
        reg: Raw reg_participacion DataFrame for Bogotá.

    Returns:
        DataFrame with one row per puesto, columns ``_join_key``,
        ``localidad_code_raw``, ``comuna_name``, ``Total censo``.

    """
    reg["_puesto_norm"] = reg["Puesto"].apply(_normalize_puesto)
    reg["_join_key"] = "2022" + "_" + reg["round"].astype(str) + "_" + reg["_puesto_norm"]

    reg["localidad_code_raw"] = reg["Código Puesto"].apply(_extract_localidad_code)

    # type: ignore[returnValue] -- groupby-agg returns DataFrame; pyright stubs disagree
    return (
        reg.groupby("_join_key")
        .agg(
            {
                "localidad_code_raw": "first",
                "Comuna": "first",
                "Voto total": "sum",
                "Total censo": "sum",
            }
        )
        .reset_index()
    )  # type: ignore[returnValue]


def _join_round(
    mmv: pd.DataFrame,
    reg: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join MMV data to reg_participacion for one election round.

    Args:
        mmv: MMV DataFrame for Bogotá (one round).
        reg: Reg_participacion DataFrame for Bogotá (one round).

    Returns:
        Tuple of ``(joined, reg_with_localidad)``:
        - ``joined``: DataFrame with per-candidate votes assigned to each
          localidad (one row per candidate per puesto).
        - ``reg_with_localidad``: reg data with ``localidad_bucket`` and
          ``codigo_municipio`` columns, aggregated to one row per puesto.

    """
    mmv_join = _prepare_mmv_join(mmv)
    reg_join = _prepare_reg_join(reg)

    joined = mmv_join.merge(
        reg_join[["_join_key", "localidad_code_raw", "Comuna"]],
        on="_join_key",
        how="left",
    )

    unmatched = joined["localidad_code_raw"].isna().sum()
    if unmatched > 0:
        logger.warning(
            "%d MMV rows (%d mesas) could not be joined to reg_participacion "
            "- assigning to SIN COMUNA catch-all",
            int(joined[joined["localidad_code_raw"].isna()]["_join_key"].nunique()),
            int(unmatched),
        )

    joined["localidad_code_raw"] = joined["localidad_code_raw"].fillna("00")
    joined["localidad_bucket"] = joined["localidad_code_raw"].apply(_classify_localidad)
    joined["codigo_municipio"] = joined["localidad_bucket"].apply(_build_7digit_code)
    joined["comuna_nombre"] = joined["localidad_bucket"].apply(_localidad_name)

    reg_join["localidad_code_raw"] = reg_join["localidad_code_raw"].fillna("00")
    reg_join["localidad_bucket"] = reg_join["localidad_code_raw"].apply(_classify_localidad)
    reg_join["codigo_municipio"] = reg_join["localidad_bucket"].apply(_build_7digit_code)
    reg_join["comuna_nombre"] = reg_join["localidad_bucket"].apply(_localidad_name)

    return joined, reg_join


# ═══════════════════════════════════════════════════════════════════════
# Vote aggregation by localidad
# ═══════════════════════════════════════════════════════════════════════


def _aggregate_votes(joined: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-candidate votes to the localidad level.

    Args:
        joined: Joined MMV+reg data with ``codigo_municipio``,
            ``CANNOMBRE``, ``VOTOS`` columns.

    Returns:
        DataFrame with columns ``codigo_municipio``, ``comuna_nombre``,
        ``year``, ``round``, ``candidate``, ``votes``.

    """
    result = joined.groupby(
        ["codigo_municipio", "comuna_nombre", "year", "round", "CANNOMBRE"],
        as_index=False,
    )[["VOTOS"]].sum()
    result.columns = ["codigo_municipio", "comuna_nombre", "year", "round", "candidate", "votes"]
    return result


def _aggregate_totals(reg: pd.DataFrame) -> pd.DataFrame:
    """Compute total votes and registered voters per localidad from reg data.

    Args:
        reg: Raw reg_participacion DataFrame for Bogotá (with localidad
            columns already added).

    Returns:
        DataFrame with columns ``codigo_municipio``, ``total_votes``,
        ``registered_voters``.

    """
    return reg.groupby("codigo_municipio", as_index=False).agg(
        total_votes=("Voto total", "sum"),
        registered_voters=("Total censo", "sum"),
    )


def _enrich_with_totals(
    votes: pd.DataFrame,
    totals: pd.DataFrame,
) -> pd.DataFrame:
    """Merge per-candidate votes with localidad-level totals.

    Computes ``vote_share = votes / total_votes`` and
    ``abstention_rate = 1 - (total_votes / registered_voters)``.

    Args:
        votes: Per-candidate votes from ``_aggregate_votes``.
        totals: Per-localidad totals from ``_aggregate_totals``.

    Returns:
        DataFrame with full historical results schema.

    """
    result = votes.merge(totals, on="codigo_municipio", how="left")

    safe_total = result["total_votes"].replace(0, pd.NA)
    safe_registered = result["registered_voters"].replace(0, pd.NA)
    result["vote_share"] = (result["votes"] / safe_total).fillna(0.0).clip(0.0, 1.0)
    raw_abstention = 1 - (result["total_votes"] / safe_registered)
    result["abstention_rate"] = raw_abstention.clip(0.0, 1.0)

    return result


# ═══════════════════════════════════════════════════════════════════════
# Public API — election results
# ═══════════════════════════════════════════════════════════════════════


def disaggregate_bogota_election_results(
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Compute per-localidad election results for all 2022 rounds.

    Loads MMV and reg_participacion for both rounds, joins on
    normalised puesto name, and aggregates per-candidate votes to
    the localidad level.  The output follows the same schema as
    ``historical_results.csv`` so it can be appended directly.

    Args:
        data_dir: Root data directory.  If ``None``, resolves via
            ``resolve_data_dir``.

    Returns:
        DataFrame with columns matching the historical results schema:
        ``codigo_municipio``, ``comuna_nombre``, ``year``, ``round``,
        ``candidate``, ``votes``, ``total_votes``, ``registered_voters``,
        ``vote_share``, ``abstention_rate``.

    Raises:
        FileNotFoundError: If any required file is missing.

    """
    base = resolve_data_dir(data_dir)

    all_parts: list[pd.DataFrame] = []

    for round_num in (1, 2):
        mmv = _load_bogota_mmv(base, round_num)
        reg = _load_bogota_reg_participacion(base, round_num)

        joined, reg_enriched = _join_round(mmv, reg)
        votes = _aggregate_votes(joined)
        totals = _aggregate_totals(reg_enriched)
        enriched = _enrich_with_totals(votes, totals)
        all_parts.append(enriched)

    result = pd.concat(all_parts, ignore_index=True)
    logger.info(
        "Bogotá localidad results: %d rows, %d localidades",
        len(result),
        result["codigo_municipio"].nunique(),
    )
    return result


# ═══════════════════════════════════════════════════════════════════════
# Population projections — proportional split
# ═══════════════════════════════════════════════════════════════════════


def _load_bogota_registered_voters(data_dir: Path) -> pd.DataFrame:
    """Load Bogotá voter census totals per localidad.

    Uses the reg_participacion round-1 data to get the total census
    (``Total censo``) for each polling station, then sums by localidad.

    Args:
        data_dir: Root data directory.

    Returns:
        DataFrame with columns ``localidad_bucket``, ``total_censo``.

    """
    reg = _load_bogota_reg_participacion(data_dir, round_num=1)
    reg["localidad_code_raw"] = reg["Código Puesto"].apply(_extract_localidad_code)
    reg["localidad_code_raw"] = reg["localidad_code_raw"].fillna("00")
    reg["localidad_bucket"] = reg["localidad_code_raw"].apply(_classify_localidad)
    reg["codigo_municipio"] = reg["localidad_bucket"].apply(_build_7digit_code)

    counts = reg.groupby(["localidad_bucket", "codigo_municipio"], as_index=False)[
        ["Total censo"]
    ].sum()
    by_localidad = counts.rename(columns={"Total censo": "total_censo"})
    by_localidad["weight"] = by_localidad["total_censo"] / by_localidad["total_censo"].sum()
    return by_localidad


def _load_bogota_municipal_population(data_dir: Path) -> pd.Series:
    """Load Bogotá's single-municipality population projection row.

    Args:
        data_dir: Root data directory.

    Returns:
        Series indexed by year (``pop_2018`` through ``pop_2026``).

    Raises:
        FileNotFoundError: If population file is missing.

    """
    pop_path = data_dir / "fundamentals" / "population_2018_2026.csv"
    if not pop_path.is_file():
        msg = f"Population file not found: {pop_path}"
        raise FileNotFoundError(msg)
    pop_df = pd.read_csv(pop_path, dtype={"codigo_municipio": str})
    bog_row = pop_df[pop_df["codigo_municipio"] == _BOGOTA_DANE_MUN]
    if bog_row.empty:
        logger.warning("Bogotá row not found in population projections")
        return pd.Series(dtype=float)
    pop_cols = [f"pop_{y}" for y in range(2018, 2027)]
    return bog_row.iloc[0][pop_cols]


def disaggregate_bogota_population(data_dir: Path | None = None) -> pd.DataFrame:
    """Split Bogotá's municipal population projection by localidad.

    Uses each localidad's share of Bogotá's total registered voters
    (from ``reg_participacion``) as the allocation weight for every
    projection year.

    Args:
        data_dir: Root data directory.  If ``None``, resolves via
            ``resolve_data_dir``.

    Returns:
        DataFrame with columns ``codigo_municipio``, ``comuna_nombre``,
        and ``pop_2018`` through ``pop_2026``, one row per localidad.

    """
    base = resolve_data_dir(data_dir)

    weights = _load_bogota_registered_voters(base)
    bog_pop = _load_bogota_municipal_population(base)

    if bog_pop.empty:
        logger.warning("No Bogotá population data — returning empty DataFrame")
        return pd.DataFrame()

    pop_cols = [f"pop_{y}" for y in range(2018, 2027)]

    rows: list[dict[str, object]] = []
    for _, row in weights.iterrows():
        entry: dict[str, object] = {
            "codigo_municipio": row["codigo_municipio"],
            "comuna_nombre": _localidad_name(row["localidad_bucket"]),
        }
        weight = float(row["weight"])
        for col in pop_cols:
            entry[col] = round(float(bog_pop[col]) * weight)
        rows.append(entry)

    result = pd.DataFrame(rows)
    logger.info(
        "Bogotá localidad population: %d rows, sum pop_2022 = %s",
        len(result),
        f"{result['pop_2022'].sum():,}" if "pop_2022" in result.columns else "N/A",
    )
    return result


# ═══════════════════════════════════════════════════════════════════════
# DIVIPOLA-like localidad rows
# ═══════════════════════════════════════════════════════════════════════


def get_bogota_localidad_rows() -> pd.DataFrame:
    """Generate DIVIPOLA-style rows for Bogotá localidades.

    Each localidad gets a 7-digit ``codigo_municipio``, a
    ``nombre_municipio`` of ``"Bogotá D.C."``, ``comuna_nombre`` with
    the localidad name, ``departamento`` of ``"Cundinamarca"``, and
    ``region`` of ``"Andina"``.

    Returns:
        DataFrame with one row per localidad.

    """
    rows: list[dict[str, str | None]] = []
    for code, name in BOGOTA_LOCALIDADES.items():
        rows.append(
            {
                "codigo_municipio": _build_7digit_code(code),
                "nombre_municipio": "Bogotá D.C.",
                "comuna_nombre": name,
                "departamento": "Cundinamarca",
                "region": "Andina",
            }
        )
    return pd.DataFrame(rows)
