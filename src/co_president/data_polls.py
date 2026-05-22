"""SPEC-04: Poll data loading & cleaning.

Loads, cleans, normalizes, and filters poll data into a typed,
analysis-ready ``CleanPolls`` container.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
import logging
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal
import unicodedata

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import date
    from pathlib import Path

import pandas as pd

from co_president.config import (
    CONSULTATION_DATE,
    CONSULTATION_KEY_MAP,
    ELECTION_DATE_ROUND1,
    get_active_candidates,
)
from co_president.paths import resolve_data_dir

__all__ = [
    "CandidateShares",
    "CleanPolls",
    "ConsultationPoll",
    "PollRow",
    "UnclassifiedPollRow",
    "deduplicate_polls",
    "fix_invamer_date",
    "infer_round_number",
    "load_and_clean_all",
    "load_raw_consultas",
    "load_raw_polls",
    "map_consultation_name_to_key",
    "normalize_undecided",
    "parse_consultations",
    "retain_active_candidates",
]

logger = logging.getLogger(__name__)


_ROUND_TWO = 2  # second (runoff) round identifier

_MIN_ROUND1_POLLSTERS = 5
_MIN_ROUND2_POLLSTERS = 2
_NORMALIZATION_TOLERANCE_PCT = 1.0
_RENORMALIZE_THRESHOLD = 0.01
_POST_RENORMALIZE_TOLERANCE_PCT = 0.1

_SHARE_COLS_EXCLUDED = frozenset(
    (
        "n",
        "encuestadora",
        "fecha",
        "muestra",
        "tasa_respuesta",
        "margen_error",
        "fuente",
        "link",
        "muestreo",
        "hipotesis",
        "tipo",
        "muestra_int_voto",
        "municipios",
        "ns_nr",
        "round_number",
    )
)
"""Frozen set of metadata column names excluded from undecided redistribution.

``round_number`` is not in the raw CSV; it is added by ``infer_round_number``
and excluded from share normalization."""


def _normalize_consultation_name(name: str) -> str:
    """Normalize a candidate name for accent-insensitive lookup.

    Strips whitespace, applies NFKD Unicode normalization, removes
    combining marks (accents), and lowercases.
    """
    stripped = name.strip()
    normalized = unicodedata.normalize("NFKD", stripped)
    return "".join(c for c in normalized if not unicodedata.combining(c)).lower()


@cache
def _build_normalized_consultation_map() -> dict[str, str]:
    """Build a cached, normalized-key version of CONSULTATION_KEY_MAP."""
    result: dict[str, str] = {}
    for raw_name, key in CONSULTATION_KEY_MAP.items():
        result[_normalize_consultation_name(raw_name)] = key
    return result


@dataclass(frozen=True)
class CandidateShares:
    """Vote share breakdown for a single poll.

    Attributes:
        candidates: Mapping of candidate key to normalized vote share (``%``).
        ns_nr: Undecided / no response share (``%``, always 0 after normalization).
        blanco: Blank vote share (``%``).
        otros: Other candidates' vote share (``%``).

    """

    candidates: Mapping[str, float]
    ns_nr: float
    blanco: float
    otros: float

    def __post_init__(self) -> None:
        """Defensive-copy the candidates mapping to prevent external mutation."""
        object.__setattr__(self, "candidates", MappingProxyType(dict(self.candidates)))

    def total(self) -> float:
        """Sum all shares including ``ns_nr``, ``blanco``, and ``otros``.

        Returns:
            Sum of shares in percent, nominally 100 after normalization.

        """
        return sum(self.candidates.values()) + self.ns_nr + self.blanco + self.otros


@dataclass(frozen=True)
class PollRow:
    """One row of cleaned, classified poll data.

    Attributes:
        date: Poll date.
        pollster: Pollster name.
        sample_size: Total sample size.
        sample_voting: Effective voting sample (``muestra_int_voto``), if available.
        margin_of_error: Margin of error in percentage points, if available.
        survey_method: Survey methodology (e.g. ``"presencial"``, ``"telefonica"``).
        round_number: Classified election round (1 or 2).
        shares: Vote share breakdown.

    """

    date: date
    pollster: str
    sample_size: int | None
    sample_voting: int | None
    margin_of_error: float | None
    survey_method: str
    round_number: Literal[1, 2]
    shares: CandidateShares


@dataclass(frozen=True)
class UnclassifiedPollRow:
    """One row of cleaned poll data that could not be classified to a round.

    Identical to ``PollRow`` but omits ``round_number``.
    """

    date: date
    pollster: str
    sample_size: int | None
    sample_voting: int | None
    margin_of_error: float | None
    survey_method: str
    shares: CandidateShares


@dataclass(frozen=True)
class ConsultationPoll:
    """One row of inter-party consultation poll data.

    Attributes:
        date: Poll date.
        pollster: Pollster name.
        coalition: Coalition name (e.g. ``"Pacto Historico"``).
        candidate: Raw candidate name from the CSV.
        candidate_key: Canonical candidate key.
        share: Vote intention within the coalition (``%``).
        sample_size: Poll sample size.
        margin_of_error: Margin of error in percentage points, if available.

    """

    date: date
    pollster: str
    coalition: str
    candidate: str
    candidate_key: str
    share: float
    sample_size: int | None
    margin_of_error: float | None


@dataclass(frozen=True)
class CleanPolls:
    """Cleaned poll dataset split by round.

    Attributes:
        round1: Post-consultation round 1 polls (``>=5`` unique pollsters).
        round2: Post-round-1 runoff polls (``>=2`` unique pollsters).
        consultation: Pre-consultation coalition poll data (typed objects).
        all_polls: All cleaned rows (pre-round split) for exploration.

    """

    round1: pd.DataFrame
    round2: pd.DataFrame
    consultation: list[ConsultationPoll]
    all_polls: pd.DataFrame

    def __post_init__(self) -> None:
        """Defensive-copy DataFrames and validate minimum pollster diversity.

        Stores copies of all DataFrames to prevent external in-place mutation.
        Raises ``ValueError`` if round pollster diversity is too low.
        """
        for field in ("round1", "round2", "all_polls"):
            object.__setattr__(self, field, getattr(self, field).copy())
        object.__setattr__(self, "consultation", list(self.consultation))

        round1_pollsters = self.round1["encuestadora"].nunique()
        round2_pollsters = self.round2["encuestadora"].nunique()

        if round1_pollsters < _MIN_ROUND1_POLLSTERS:
            msg = f"Round 1 needs >= {_MIN_ROUND1_POLLSTERS} pollsters, got {round1_pollsters}"
            raise ValueError(msg)
        if round2_pollsters < _MIN_ROUND2_POLLSTERS:
            msg = f"Round 2 needs >= {_MIN_ROUND2_POLLSTERS} pollsters, got {round2_pollsters}"
            raise ValueError(msg)


def map_consultation_name_to_key(name: str) -> str:
    """Map a consultation candidate display name to its canonical key.

    Uses accent-insensitive matching via NFKD Unicode normalization.
    Raises ``ValueError`` if the name is not found in ``CONSULTATION_KEY_MAP``.

    Args:
        name: Raw candidate name from the consultation CSV.

    Returns:
        Canonical candidate key (e.g. ``"gustavo_petro"``).

    Raises:
        ValueError: If ``name`` has no mapping.

    """
    lookup = _build_normalized_consultation_map()
    normalized = _normalize_consultation_name(name)
    key = lookup.get(normalized)
    if key is None:
        msg = f"Unrecognized consultation candidate name: {name!r}"
        raise ValueError(msg)
    return key


def load_raw_polls(data_dir: Path | None = None) -> pd.DataFrame:
    """Load raw poll data from ``encuestas_2022.csv``.

    Parses ``fecha`` to datetime, coerces numeric columns, and preserves
    all columns from the source CSV.

    Args:
        data_dir: Path to the project data directory. Auto-resolved if ``None``.

    Returns:
        DataFrame with parsed types; ``fecha`` is ``datetime64[ns]``.

    Raises:
        ValueError: If any dates fail to parse.

    """
    resolved = resolve_data_dir(data_dir)
    path = resolved / "2022-polls" / "encuestas_2022.csv"
    df = pd.read_csv(path, encoding="utf-8", low_memory=False)

    # Validate required columns exist
    required_poll_cols = {
        "fecha",
        "encuestadora",
        "muestra",
        "federico_gutierrez",
        "gustavo_petro",
        "rodolfo_hernandez",
        "ns_nr",
    }
    missing = required_poll_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {', '.join(sorted(missing))}"
        raise ValueError(msg)

    # Parse fecha
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    invalid_dates = df["fecha"].isna()
    if invalid_dates.any():
        bad_rows = invalid_dates[invalid_dates].index.tolist()
        msg = f"Failed to parse fecha in rows: {bad_rows}"
        raise ValueError(msg)

    # Coerce numeric columns
    df["muestra"] = pd.to_numeric(df["muestra"], errors="coerce").astype("Int64")
    if "muestra_int_voto" in df.columns:
        df["muestra_int_voto"] = pd.to_numeric(df["muestra_int_voto"], errors="coerce").astype(
            "Int64"
        )
    if "margen_error" in df.columns:
        df["margen_error"] = pd.to_numeric(
            df["margen_error"].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )
    if "ns_nr" in df.columns:
        df["ns_nr"] = pd.to_numeric(df["ns_nr"], errors="coerce")

    return df


def load_raw_consultas(data_dir: Path | None = None) -> pd.DataFrame:
    """Load raw consultation poll data from ``consultas.csv``.

    Args:
        data_dir: Path to the project data directory. Auto-resolved if ``None``.

    Returns:
        DataFrame with parsed types.

    Raises:
        ValueError: If any dates fail to parse.

    """
    resolved = resolve_data_dir(data_dir)
    path = resolved / "2022-polls" / "consultas.csv"
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)

    # Validate required columns exist
    required_consultas_cols = {
        "fecha",
        "encuestadora",
        "consulta",
        "candidato",
        "int_voto",
        "muestra",
    }
    missing = required_consultas_cols - set(df.columns)
    if missing:
        msg = f"Missing required columns: {', '.join(sorted(missing))}"
        raise ValueError(msg)

    # Parse fecha (M/D/YYYY format)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    invalid_dates = df["fecha"].isna()
    if invalid_dates.any():
        bad_rows = invalid_dates[invalid_dates].index.tolist()
        msg = f"Failed to parse fecha in consultas rows: {bad_rows}"
        raise ValueError(msg)

    # Coerce numeric columns
    df["int_voto"] = pd.to_numeric(
        df["int_voto"].astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    if df["int_voto"].isna().any():
        bad_rows = df.index[df["int_voto"].isna()].tolist()
        msg = f"Failed to parse int_voto in rows: {bad_rows}"
        raise ValueError(msg)
    df["muestra"] = pd.to_numeric(df["muestra"], errors="coerce").astype("Int64")
    if "margen_error" in df.columns:
        df["margen_error"] = pd.to_numeric(
            df["margen_error"].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )

    return df


def parse_consultations(df: pd.DataFrame) -> list[ConsultationPoll]:
    """Convert raw consultation DataFrame into typed ``ConsultationPoll`` objects.

    Skips rows with unrecognized candidate names (logged as warnings).
    If **all** rows are unrecognized, raises ``ValueError``.

    Args:
        df: DataFrame from ``load_raw_consultas``.

    Returns:
        List of ``ConsultationPoll`` objects.

    Raises:
        ValueError: If no rows could be mapped (all candidate names unrecognized).

    """
    results: list[ConsultationPoll] = []
    skipped = 0
    for _, row in df.iterrows():
        try:
            candidate_key = map_consultation_name_to_key(str(row["candidato"]))
        except ValueError:
            logger.warning(
                "Skipping consultation row: unrecognized candidate %r",
                row.get("candidato"),
            )
            skipped += 1
            continue
        cp = ConsultationPoll(
            date=row["fecha"].date(),
            pollster=str(row["encuestadora"]).strip(),
            coalition=str(row["consulta"]).strip(),
            candidate=str(row["candidato"]).strip(),
            candidate_key=candidate_key,
            share=float(row["int_voto"]),
            sample_size=int(row["muestra"]) if pd.notna(row["muestra"]) else None,
            margin_of_error=(float(row["margen_error"]) if pd.notna(row["margen_error"]) else None),
        )
        results.append(cp)

    if not results:
        msg = (
            f"parse_consultations: all {len(df)} rows had unrecognized candidate "
            f"names ({skipped} skipped). Consider expanding CONSULTATION_KEY_MAP."
        )
        raise ValueError(msg)

    if skipped:
        logger.info(
            "parse_consultations: parsed %d rows, skipped %d unrecognized",
            len(results),
            skipped,
        )

    return results


# ═══════════════════════════════════════════════════════════════════
# SPEC-04: Poll Data Cleaning
# ═══════════════════════════════════════════════════════════════════


def fix_invamer_date(df: pd.DataFrame) -> pd.DataFrame:
    """Correct the known Invamer data-entry error.

    The Invamer poll dated ``2022-04-19`` should be ``2022-05-19``.
    Operates on and returns a **copy** of the input.

    Args:
        df: Raw poll DataFrame (must have ``encuestadora`` and ``fecha`` columns).

    Returns:
        Copy of ``df`` with the corrected date.

    """
    result = df.copy()
    mask = (result["encuestadora"].str.strip() == "Invamer") & (
        result["fecha"] == pd.Timestamp("2022-04-19")
    )
    result.loc[mask, "fecha"] = pd.Timestamp("2022-05-19")
    n_corrected = int(mask.sum())
    if n_corrected:
        logger.info(
            "fix_invamer_date: corrected %d row(s) from 2022-04-19 to 2022-05-19",
            n_corrected,
        )
    return result


def _detect_forced_choice(df: pd.DataFrame) -> pd.Series:
    """Detect forced-choice R2 polls.

    Flags rows where blanco is NA, ns_nr is absent or zero, and
    petro+hernandez sum to 100% ± 1pp.

    Handles detection both before and after ``normalize_undecided``
    (which fills NaN ns_nr values with 0.0).

    Args:
        df: Poll DataFrame with candidate share columns.

    Returns:
        Boolean Series indexed like ``df``, True for forced-choice polls.

    """
    petro = df["gustavo_petro"].fillna(0)
    hernandez = df["rodolfo_hernandez"].fillna(0)
    both_present = df["gustavo_petro"].notna() & df["rodolfo_hernandez"].notna()

    blanco_na = df["blanco"].isna()
    ns_nr_zero = df["ns_nr"].fillna(0) == 0
    sum_check = (petro + hernandez - 100).abs() <= 1

    return blanco_na & ns_nr_zero & sum_check & both_present


def _fix_yanhaas_20220611(df: pd.DataFrame) -> pd.DataFrame:
    """Correct YanHaas 103% sum anomaly on 2022-06-11.

    Redistributes ns_nr (10.0pp) proportionally to Petro, Hernandez, Blanco.
    If total > 100% ± 0.01%, renormalizes rows.
    """
    result = df.copy()
    mask = (result["encuestadora"].str.strip() == "YanHaas") & (
        result["fecha"] == pd.Timestamp("2022-06-11")
    )
    if not mask.any():
        return result

    share_cols = ["gustavo_petro", "rodolfo_hernandez", "blanco"]
    for idx in result.index[mask]:
        ns_nr = result.loc[idx, "ns_nr"]
        if not isinstance(ns_nr, (int, float)) or pd.isna(ns_nr):
            continue

        # Proportional redistribution
        scale = 100.0 / (100.0 - ns_nr)
        for col in share_cols:
            raw = result.loc[idx, col]
            if isinstance(raw, (int, float)) and not pd.isna(raw):
                result.loc[idx, col] = raw * scale

        result.loc[idx, "ns_nr"] = 0.0
        logger.info(
            "_fix_yanhaas_20220611: redistributed %.1fpp ns_nr for row %s",
            ns_nr,
            idx,
        )

    # Renormalize if total > 100% ± 0.01%
    _renormalize_rows(result, [*share_cols, "otros"], set(result.index[mask]))

    return result


_NS_NR_HUNDRED = 100.0


def _normalize_share_rows(
    df: pd.DataFrame,
    share_cols: list[str],
) -> tuple[set[int], set[int]]:
    """Apply undecided redistribution to rows with ``ns_nr > 0``.

    **Warning:** Mutates ``df`` in-place. Caller must pass a copy.

    For rows where ``ns_nr == 100``, no normalization occurs and the row is
    recorded as skipped.

    Args:
        df: Poll DataFrame with ``ns_nr`` and share columns.
        share_cols: Columns to redistribute (candidate shares plus ``blanco``/``otros``).

    Returns:
        Tuple of ``(normalized_indices, skipped_100_indices)`` describing which
        rows were normalized and which were skipped due to ``ns_nr == 100``.

    """
    normalized: set[int] = set()
    skipped_100: set[int] = set()
    for idx in df.index:
        ns_nr = df.loc[idx, "ns_nr"]  # pyright: ignore[reportUnknownVariableType]
        if not isinstance(ns_nr, (int, float)):
            ns_nr = 0.0

        if ns_nr == _NS_NR_HUNDRED:
            logger.warning("Row %s: ns_nr = 100, all shares unchanged", str(idx))
            skipped_100.add(int(idx))
            continue
        if ns_nr > 0:
            scale = 100.0 / (100.0 - ns_nr)
            for col in share_cols:
                raw = df.loc[idx, col]  # pyright: ignore[reportUnknownVariableType]
                if isinstance(raw, (int, float)) and not pd.isna(raw):
                    df.loc[idx, col] = raw * scale
            normalized.add(int(idx))
        df.loc[idx, "ns_nr"] = 0.0  # pyright: ignore[reportUnknownArgumentType]
    return normalized, skipped_100


def _renormalize_rows(
    df: pd.DataFrame,
    share_cols: list[str],
    indices: set[int],
) -> None:
    """Renormalise rows so share columns sum to 100, handling rounding drift.

    **Warning:** Mutates ``df`` in-place. Caller must pass a copy.

    Rows with a non-positive share sum are skipped.

    Args:
        df: Poll DataFrame with share columns.
        share_cols: Columns to renormalize.
        indices: Row indices to renormalize.

    Returns:
        None.

    """
    for idx in indices:
        vals = [df.loc[idx, col] for col in share_cols]  # pyright: ignore[reportUnknownVariableType]
        row_sum = sum(v for v in vals if isinstance(v, (int, float)) and not pd.isna(v))
        if row_sum <= 0:
            continue
        if abs(row_sum - 100.0) > _RENORMALIZE_THRESHOLD:
            fix_scale = 100.0 / row_sum
            for col in share_cols:
                val = df.loc[idx, col]  # pyright: ignore[reportUnknownVariableType]
                if isinstance(val, (int, float)) and not pd.isna(val):
                    df.loc[idx, col] = val * fix_scale


def _validate_normalized_rows(
    df: pd.DataFrame,
    share_cols: list[str],
    indices: set[int],
    tolerance_pct: float = _NORMALIZATION_TOLERANCE_PCT,
) -> None:
    """Assert that every normalised row sums to 100 +/- tolerance."""
    for idx in indices:
        vals = [df.loc[idx, col] for col in share_cols]  # pyright: ignore[reportUnknownVariableType]
        row_sum = sum(v for v in vals if isinstance(v, (int, float)) and not pd.isna(v))
        if row_sum <= 0:
            continue
        if abs(row_sum - 100.0) > tolerance_pct:
            msg = (
                f"Row {idx}: normalized share sum = {row_sum:.2f}%, expected 100.0 +/- "
                f"{tolerance_pct:.1f}"
            )
            raise ValueError(msg)


def normalize_undecided(df: pd.DataFrame) -> pd.DataFrame:
    """Redistribute undecided voters (``ns_nr``) proportionally to all share columns.

    For each row where ``ns_nr > 0``:
        ``normalized[i] = raw[i] / (100 - ns_nr) * 100``

    Applies to all candidate columns, ``blanco``, and ``otros``.
    Sets ``ns_nr = 0.0`` after redistribution.
    Rows with ``ns_nr == 100`` are skipped with a warning.
    ``ns_nr = NA`` is treated as ``0`` (no redistribution).

    Post-condition: each row's normalized shares sum to ``100.0 +/- 1.0``. If a
    row violates this, a ``ValueError`` is raised.

    Args:
        df: Raw poll DataFrame with candidate share columns and ``ns_nr``.

    Returns:
        Copy of ``df`` with normalized shares and ``ns_nr = 0.0``.

    Raises:
        ValueError: If a row's normalized share sum deviates from 100 by >1pp.

    """
    result = df.copy()
    share_cols = [c for c in result.columns if c not in _SHARE_COLS_EXCLUDED]
    normalized_indices, skipped_100 = _normalize_share_rows(result, share_cols)
    all_check = set(result.index) - skipped_100
    _renormalize_rows(result, share_cols, all_check)
    _validate_normalized_rows(result, share_cols, normalized_indices)
    return result


def retain_active_candidates(
    df: pd.DataFrame,
    candidates: list[str],
) -> pd.DataFrame:
    """Drop candidate columns that are all-NA or not in the active list.

    Preserves metadata columns (``_SHARE_COLS_EXCLUDED``), ``blanco``,
    and ``otros`` unconditionally. Drops columns in ``candidates`` that
    are all-NA, and drops share columns not in ``candidates``.

    Args:
        df: Poll DataFrame.
        candidates: List of candidate keys to retain (e.g. from
            ``get_active_candidates``).

    Returns:
        DataFrame with non-active and all-NA candidate columns removed.

    """
    always_preserve = _SHARE_COLS_EXCLUDED | {"blanco", "otros"}
    candidate_set = set(candidates)
    result = df.copy()
    for col in list(result.columns):
        if col in always_preserve:
            continue
        if col not in candidate_set:
            result = result.drop(columns=[col])
            continue
        if result[col].isna().all():
            result = result.drop(columns=[col])
    return result


_ROUND1_REQUIRED = ["gustavo_petro", "federico_gutierrez", "rodolfo_hernandez"]
_ROUND1_OPTIONAL = ["sergio_fajardo", "ingrid_betancourt"]
_ROUND2_REQUIRED = ["gustavo_petro", "rodolfo_hernandez"]
_ROUND2_ABSENT = ["federico_gutierrez", "sergio_fajardo", "ingrid_betancourt"]


def _is_round1_candidate(row: pd.Series, columns: pd.Index) -> bool:
    """Check if a poll matches the round 1 candidate pattern."""
    for col in _ROUND1_REQUIRED:
        if col not in columns or pd.isna(row[col]):  # pyright: ignore[reportUnknownMemberType]
            return False
    optional = [
        col
        for col in _ROUND1_OPTIONAL
        if col in columns and pd.notna(row[col])  # pyright: ignore[reportUnknownMemberType]
    ]
    if not optional:
        return False
    fecha = row["fecha"]  # pyright: ignore[reportUnknownVariableType]
    return fecha >= pd.Timestamp(CONSULTATION_DATE)


def _is_round2_candidate(row: pd.Series, columns: pd.Index) -> bool:
    """Check if a poll matches the round 2 candidate pattern."""
    for col in _ROUND2_REQUIRED:
        if col not in columns or pd.isna(row[col]):  # pyright: ignore[reportUnknownMemberType]
            return False
    for col in _ROUND2_ABSENT:
        if col in columns and pd.notna(row[col]):  # pyright: ignore[reportUnknownMemberType]
            return False
    fecha = row["fecha"]  # pyright: ignore[reportUnknownVariableType]
    return fecha >= pd.Timestamp(ELECTION_DATE_ROUND1)


def infer_round_number(df: pd.DataFrame) -> pd.DataFrame:
    """Assign ``round_number`` (1, 2, or NaN) to each poll based on candidate pattern.

    Round 1 (relaxed criteria):
        - ``gustavo_petro``, ``federico_gutierrez``, ``rodolfo_hernandez`` non-NA
        - At least one of ``sergio_fajardo`` or ``ingrid_betancourt`` non-NA
        - ``fecha >= CONSULTATION_DATE``

    Round 2:
        - ``gustavo_petro``, ``rodolfo_hernandez`` non-NA
        - ``federico_gutierrez``, ``sergio_fajardo``, ``ingrid_betancourt`` all NA
        - ``fecha >= ELECTION_DATE_ROUND1``

    All other polls: ``round_number = pd.NA`` (nullable Int64).

    Args:
        df: Poll DataFrame with candidate columns and ``fecha``.

    Returns:
        Copy of ``df`` with a new ``round_number`` column (Int64, nullable).

    """
    result = df.copy()
    round_numbers: list[int | None] = []
    for _, row in result.iterrows():
        if _is_round1_candidate(row, result.columns):
            round_numbers.append(1)
        elif _is_round2_candidate(row, result.columns):
            round_numbers.append(2)
        else:
            round_numbers.append(None)
    result["round_number"] = pd.array(round_numbers, dtype="Int64")
    return result


def deduplicate_polls(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate pollster+date entries, keeping the largest sample.

    For each group of rows with the same ``encuestadora`` and ``fecha``:
    1. Fill any NA ``muestra_int_voto`` with ``muestra``.
    2. Keep the row with the largest filled ``muestra_int_voto``.
    3. If the filled sample is still NA or there's a tie, keep the first row
       (with a warning when both samples were NA).

    Args:
        df: Poll DataFrame.

    Returns:
        DataFrame with duplicates removed.

    """
    if df.empty:
        return df.copy()

    n_before = len(df)

    result = df.copy()
    # Create a temporary effective sample column
    if "muestra_int_voto" not in result.columns:
        result["muestra_int_voto"] = result["muestra"]
    temp_sample = result["muestra_int_voto"].fillna(result["muestra"])
    result["_eff_sample"] = temp_sample

    # Determine sort order: groups first, then _eff_sample descending
    result = result.sort_values("_eff_sample", ascending=False, na_position="last")

    # Warn about groups where both samples were NA
    na_mask = result["_eff_sample"].isna()
    if na_mask.any():
        logger.warning(
            "deduplicate_polls: %d rows have both muestra_int_voto and muestra = NA, "
            "keeping first row per group",
            na_mask.sum(),
        )

    # Keep first row per pollster+date group
    deduped = result.groupby(["encuestadora", "fecha"], sort=False).head(1).reset_index(drop=True)

    # Drop the temporary column
    deduped = deduped.drop(columns=["_eff_sample"])
    n_after = len(deduped)
    if n_before != n_after:
        logger.info(
            "deduplicate_polls: %d -> %d rows (%d duplicates removed)",
            n_before,
            n_after,
            n_before - n_after,
        )
    return deduped


# ═══════════════════════════════════════════════════════════════════
# SPEC-04: Orchestration
# ═══════════════════════════════════════════════════════════════════


def load_and_clean_all(data_dir: Path | None = None) -> CleanPolls:
    """Load, clean, normalize, and classify all poll data.

    Pipeline:
        1. ``load_raw_polls`` → parse dates, coerce types
        2. ``fix_invamer_date`` → correct April 19 → May 19
        3. ``normalize_undecided`` → redistribute, validate 100±1%
           └─ ``all_polls`` snapshot taken here (before column filtering)
        4. ``retain_active_candidates`` (round 1 superset)
        5. ``infer_round_number`` → classify polls
        6. Split into round1 / round2 / unclassified
        7. Per-round: ``retain_active_candidates`` (round-specific keys)
        8. Per-round: ``deduplicate_polls``
        9. ``load_raw_consultas`` + ``parse_consultations``
        10. Build ``CleanPolls``

    Args:
        data_dir: Path to the project data directory. Auto-resolved if ``None``.

    Returns:
        A validated ``CleanPolls`` container.

    """
    # Step 1: Load raw polls
    polls = load_raw_polls(data_dir=data_dir)

    # Step 2: Fix Invamer date
    polls = fix_invamer_date(polls)
    polls = _fix_yanhaas_20220611(polls)

    # Step 3: Normalize undecided
    polls = normalize_undecided(polls)

    # ── all_polls snapshot ──
    all_polls = polls.copy()

    # Step 4: Retain active candidates (round 1 superset)
    r1_keys = [c.key for c in get_active_candidates(1)]
    polls = retain_active_candidates(polls, r1_keys)

    # Step 5: Infer round number
    polls = infer_round_number(polls)
    polls["forced_choice"] = _detect_forced_choice(polls)
    all_polls["round_number"] = polls["round_number"]
    all_polls["forced_choice"] = polls["forced_choice"]

    # Step 6: Split by round
    mask_r1 = polls["round_number"] == 1
    mask_r2 = (polls["round_number"] == _ROUND_TWO) & (~polls["forced_choice"])
    round1_df = polls[mask_r1].copy()
    round2_df = polls[mask_r2].copy()

    # Step 7: Per-round retain_active_candidates
    r2_keys = [c.key for c in get_active_candidates(2)]
    if not round1_df.empty:
        round1_df = retain_active_candidates(round1_df, r1_keys)
    if not round2_df.empty:
        round2_df = retain_active_candidates(round2_df, r2_keys)

    # Step 8: Per-round deduplicate
    if not round1_df.empty:
        round1_df = deduplicate_polls(round1_df)
    if not round2_df.empty:
        round2_df = deduplicate_polls(round2_df)

    # Step 8b: Renormalise round DFs to 100% (retain_active_candidates may have
    # dropped pre-consultation share columns that were part of the renormalisation).
    for df_round in [round1_df, round2_df]:
        if df_round.empty:
            continue
        round_share_cols = [c for c in df_round.columns if c not in _SHARE_COLS_EXCLUDED]
        _renormalize_rows(df_round, round_share_cols, set(df_round.index))
        _validate_normalized_rows(
            df_round,
            round_share_cols,
            set(df_round.index),
            tolerance_pct=_POST_RENORMALIZE_TOLERANCE_PCT,
        )

    # Step 9: Consultation data
    consultas_df = load_raw_consultas(data_dir=data_dir)
    consultation_list = parse_consultations(consultas_df)

    # Step 10: Build CleanPolls
    return CleanPolls(
        round1=round1_df,
        round2=round2_df,
        consultation=consultation_list,
        all_polls=all_polls,
    )
