"""SPEC-03+04: Election results consolidation + poll data loading & cleaning.

SPEC-03 ingests Registraduría (MMV) and MOE result files, aggregates to
national vote totals per candidate, cross-validates them, and produces a
single canonical ``RoundResult`` per round.

SPEC-04 loads, cleans, normalizes, and filters poll data into a typed,
analysis-ready ``CleanPolls`` container.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal
import unicodedata

if TYPE_CHECKING:
    from datetime import date

import pandas as pd

from co_president.config import (
    COALITION_TO_CANDIDATE,
    CONSULTATION_DATE,
    CONSULTATION_KEY_MAP,
    ELECTION_DATE_ROUND1,
    ELECTION_DATE_ROUND2,
    get_active_candidates,
)

logger = logging.getLogger(__name__)


_TOLERANCE_VALID_VOTES_PCT = 0.10  # % difference in total valid votes → WARNING
_TOLERANCE_CANDIDATE_SHARE_PCT = 0.50  # % point difference in vote share → WARNING
_MIN_CANDIDATES_FOR_TOP_TWO = 2  # minimum candidates needed for top_two()
_ROUND_TWO = 2  # second (runoff) round identifier


# ═══════════════════════════════════════════════════════════════════
# Public dataclasses
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class CandidateResult:
    """A single candidate's result in an election round.

    Attributes:
        candidate_key: Canonical candidate key (e.g. ``"gustavo_petro"``).
        votes: Total votes received.
        vote_share: Share of total valid votes (candidates + blank).

    """

    candidate_key: str
    votes: int
    vote_share: float


@dataclass(frozen=True)
class RoundResult:
    """Complete election results for one round.

    Attributes:
        round_number: 1 for first round, 2 for runoff.
        date: Election date.
        total_valid_votes: Sum of candidate votes (excludes blank, null, unmarked).
        total_votes_incl_blank: Sum of candidate + blank votes (vote share
            denominator, also used for turnout).
        registered_voters: Total registered voters from participation data.
        polling_stations: Unique polling stations from participation data.
        candidates: Tuple of ``CandidateResult`` for each active candidate.
        blank_votes: Blank vote count.
        null_votes: Null vote count.
        unmarked_votes: Unmarked vote count.

    """

    round_number: Literal[1, 2]
    date: date
    total_valid_votes: int
    total_votes_incl_blank: int
    registered_voters: int
    polling_stations: int
    candidates: tuple[CandidateResult, ...]
    blank_votes: int
    null_votes: int
    unmarked_votes: int

    def get_share(self, candidate_key: str) -> float:
        """Return vote share for a candidate.

        Args:
            candidate_key: Canonical candidate key.

        Returns:
            Vote share as a float in [0, 1].

        Raises:
            KeyError: If the candidate is not found in this round.

        """
        for c in self.candidates:
            if c.candidate_key == candidate_key:
                return c.vote_share
        msg = f"Candidate {candidate_key!r} not found in round {self.round_number}"
        raise KeyError(msg)

    def get_candidates_above(self, threshold_pct: float) -> list[CandidateResult]:
        """Return candidates whose vote share exceeds a percentage threshold.

        Args:
            threshold_pct: Threshold in percentage points (e.g. 20.0 means 20%).

        Returns:
            List of ``CandidateResult`` meeting the threshold, in descending
            vote order. Empty list if none meet it.

        """
        return [c for c in self.candidates if c.vote_share * 100 > threshold_pct]

    def top_two(self) -> tuple[CandidateResult, CandidateResult]:
        """Return the two highest-vote candidates, sorted descending.

        Returns:
            Tuple of ``(first_place, second_place)``.

        Raises:
            ValueError: If fewer than 2 candidates exist.

        """
        if len(self.candidates) < _MIN_CANDIDATES_FOR_TOP_TWO:
            msg = f"Round {self.round_number} has fewer than 2 candidates"
            raise ValueError(msg)
        sorted_candidates = sorted(self.candidates, key=lambda c: c.votes, reverse=True)
        return sorted_candidates[0], sorted_candidates[1]

    def turnout(self) -> float:
        """Compute turnout rate.

        Returns:
            ``total_votes_incl_blank / registered_voters``.

        Raises:
            ZeroDivisionError: If ``registered_voters`` is 0.

        """
        return self.total_votes_incl_blank / self.registered_voters


# ═══════════════════════════════════════════════════════════════════
# SPEC-04: Poll Data Constants & Dataclasses
# ═══════════════════════════════════════════════════════════════════


_MIN_ROUND1_POLLSTERS = 5
_MIN_ROUND2_POLLSTERS = 2
_NORMALIZATION_TOLERANCE_PCT = 1.0
_RENORMALIZE_THRESHOLD = 0.01

_SHARE_COLS_EXCLUDED = {
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
}
"""Set of metadata column names excluded from undecided redistribution."""


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

    candidates: dict[str, float]
    ns_nr: float
    blanco: float
    otros: float

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
    sample_size: int
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
    sample_size: int
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
    sample_size: int
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


def _normalize_coalition_name(name: str) -> str:
    """Normalize a coalition name for consistent lookup.

    Strips leading/trailing whitespace, applies NFKD Unicode normalization,
    and removes combining marks (accents). This handles the mismatch between
    ``COALITION_TO_CANDIDATE`` keys (unaccented) and MMV data that may use
    accented characters (e.g. ``"COALICIÓN PACTO HISTÓRICO"``).

    Args:
        name: Raw coalition name from the data file.

    Returns:
        Normalized, unaccented uppercase string.

    """
    stripped = name.strip().upper()
    normalized = unicodedata.normalize("NFKD", stripped)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def _resolve_data_dir(data_dir: Path | None) -> Path:
    """Locate the project data directory.

    When ``data_dir`` is provided, returns it directly. Otherwise resolves
    relative to the installed package, falling back to a file-relative path.
    """
    if data_dir is not None:
        return data_dir
    try:
        import co_president  # noqa: PLC0415

        candidate = Path(co_president.__file__).resolve().parent.parent / "data"
        if candidate.is_dir():
            return candidate
    except (ImportError, AttributeError):
        pass
    fallback = Path(__file__).resolve().parent.parent.parent / "data"
    if not fallback.is_dir():
        msg = f"data/ directory not found at {fallback}"
        raise FileNotFoundError(msg)
    return fallback


def _resolve_mmv_path(results_dir: Path, stem: str) -> Path:
    """Resolve the MMV file path, preferring decompressed .csv over .csv.gz."""
    csv_path = results_dir / f"{stem}.csv"
    if csv_path.exists():
        return csv_path
    gz_path = results_dir / f"{stem}.csv.gz"
    if gz_path.exists():
        return gz_path
    msg = f"Neither {csv_path} nor {gz_path} found"
    raise FileNotFoundError(msg)


def _read_mmv(path: Path) -> pd.DataFrame:
    """Read a Registraduría MMV file.

    Handles both ``.csv`` and ``.csv.gz``, latin-1 encoding, semicolon delimiter.
    """
    return pd.read_csv(
        path,
        encoding="latin-1",
        sep=";",
        low_memory=False,
        dtype=None,
    )


def _read_moe(path: Path) -> pd.DataFrame:
    """Read an MOE results file (UTF-8, comma-delimited)."""
    return pd.read_csv(path, encoding="utf-8")


def _read_participation(path: Path) -> pd.DataFrame:
    """Read a participation file (UTF-8 with BOM, comma-delimited).

    Some rows have extra fields due to commas in school names; those rows are
    skipped with a warning since they are edge cases (~1 per 12,500 rows).
    """
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        sep=",",
        on_bad_lines="warn",
    )


def _aggregate_and_map(
    df: pd.DataFrame,
    name_col: str,
    votes_col: str,
) -> pd.DataFrame:
    """Aggregate votes by normalized coalition name and map to candidate keys.

    Args:
        df: Raw DataFrame with coalition name and vote columns.
        name_col: Column name holding coalition/candidate names.
        votes_col: Column name holding vote counts.

    Returns:
        DataFrame with ``candidate_key`` as index and a ``votes`` column.

    """
    names = df[name_col].astype(str).str.strip()
    grouped = df.groupby(names)[votes_col].sum()
    result: dict[str, int] = {}
    for raw_name, votes in grouped.items():
        normalized = _normalize_coalition_name(str(raw_name))
        key = COALITION_TO_CANDIDATE.get(normalized)
        if key is None:
            logger.warning(
                "Unmapped coalition name: %r -> %r — accumulating into 'rest'", raw_name, normalized
            )
            result["rest"] = result.get("rest", 0) + int(votes)
            continue
        result[key] = result.get(key, 0) + int(votes)
    result_df = pd.DataFrame(list(result.items()), columns=["candidate_key", "votes"])
    return result_df.set_index("candidate_key")


def _build_round_result(
    aggregated: pd.DataFrame,
    round_number: Literal[1, 2],
    registered_voters: int,
    polling_stations: int,
) -> RoundResult:
    """Build a ``RoundResult`` from aggregated candidate vote data.

    Args:
        aggregated: DataFrame with ``candidate_key`` index and ``votes`` column.
        round_number: 1 or 2.
        registered_voters: Total registered voters.
        polling_stations: Unique polling stations.

    Returns:
        A fully populated ``RoundResult``.

    Raises:
        ValueError: If ``aggregated`` contains unexpected keys.

    """
    election_date = ELECTION_DATE_ROUND1 if round_number == 1 else ELECTION_DATE_ROUND2

    # Separate excluded vote types (convert numpy scalars to Python int)
    null_votes: int = (
        int(aggregated.loc["nulos", "votes"])  # pyright: ignore[reportArgumentType]
        if "nulos" in aggregated.index
        else 0
    )
    unmarked_votes: int = (
        int(aggregated.loc["no_marcados", "votes"])  # pyright: ignore[reportArgumentType]
        if "no_marcados" in aggregated.index
        else 0
    )

    # Remove excluded types from the working set
    candidate_df = aggregated.drop(index=["nulos", "no_marcados"], errors="ignore")

    # Track blank votes before any round-specific merging
    blank_votes: int = (
        int(candidate_df.loc["blanco", "votes"])  # pyright: ignore[reportArgumentType]
        if "blanco" in candidate_df.index
        else 0
    )

    # For round 2: merge blanco into rest (round 2 has no separate blanco candidate)
    if round_number == _ROUND_TWO and "blanco" in candidate_df.index:
        blanco_count = int(candidate_df.loc["blanco", "votes"])  # pyright: ignore[reportArgumentType]
        candidate_df = candidate_df.drop(index="blanco")
        existing = int(candidate_df.loc["rest", "votes"]) if "rest" in candidate_df.index else 0  # pyright: ignore[reportArgumentType]
        candidate_df.loc["rest", "votes"] = existing + blanco_count

    # Sums: total_valid keeps candidate votes only; total_votes_incl_blank adds
    # blank votes back (= candidates + blank, matching official Colombian
    # denominator for vote-share calculation).
    candidate_total = int(candidate_df["votes"].sum())  # pyright: ignore[reportArgumentType]
    total_valid_votes = candidate_total - blank_votes
    total_votes_incl_blank = candidate_total  # candidate_votes + blank_votes
    if total_valid_votes == 0:
        msg = f"Round {round_number} has zero total valid votes after aggregation"
        raise ValueError(msg)

    # Compute vote shares, sorted by votes descending.
    # Denominator is total_votes_incl_blank (= candidates + blank), which
    # matches the official Colombian percentage calculation.
    sorted_pairs = sorted(
        ((idx, row) for idx, row in candidate_df.iterrows()),
        key=lambda pair: int(pair[1]["votes"]),  # pyright: ignore[reportArgumentType]
        reverse=True,
    )
    candidates = tuple(
        CandidateResult(
            candidate_key=str(idx),
            votes=int(row["votes"]),  # pyright: ignore[reportArgumentType]
            vote_share=int(row["votes"]) / total_votes_incl_blank,  # pyright: ignore[reportArgumentType]
        )
        for idx, row in sorted_pairs
    )

    return RoundResult(
        round_number=round_number,
        date=election_date,
        total_valid_votes=total_valid_votes,
        total_votes_incl_blank=total_votes_incl_blank,
        registered_voters=registered_voters,
        polling_stations=polling_stations,
        candidates=candidates,
        blank_votes=blank_votes,
        null_votes=null_votes,
        unmarked_votes=unmarked_votes,
    )


# ═══════════════════════════════════════════════════════════════════
# Public loader functions
# ═══════════════════════════════════════════════════════════════════


def load_registraduria_round1(data_dir: Path | None = None) -> pd.DataFrame:
    """Load and aggregate Registraduría first-round results.

    Args:
        data_dir: Path to the project data directory. Auto-resolved if ``None``.

    Returns:
        DataFrame indexed by candidate key with a ``votes`` column.

    """
    resolved = _resolve_data_dir(data_dir)
    results_dir = resolved / "2022-presidential-results"
    path = _resolve_mmv_path(results_dir, "MMV_NACIONAL_PRESIDENTE_2022_1v")
    df = _read_mmv(path)
    return _aggregate_and_map(df, "PARNOMBRE", "VOTOS")


def load_registraduria_round2(data_dir: Path | None = None) -> pd.DataFrame:
    """Load and aggregate Registraduría second-round (runoff) results.

    Args:
        data_dir: Path to the project data directory. Auto-resolved if ``None``.

    Returns:
        DataFrame indexed by candidate key with a ``votes`` column.

    """
    resolved = _resolve_data_dir(data_dir)
    results_dir = resolved / "2022-presidential-results"
    path = _resolve_mmv_path(results_dir, "MMV_NACIONAL_PRESIDENTE_2022_2v")
    df = _read_mmv(path)
    return _aggregate_and_map(df, "PARNOMBRE", "VOTOS")


def load_moe_round1(data_dir: Path | None = None) -> pd.DataFrame:
    """Load and aggregate MOE first-round results.

    Args:
        data_dir: Path to the project data directory. Auto-resolved if ``None``.

    Returns:
        DataFrame indexed by candidate key with a ``votes`` column.

    """
    resolved = _resolve_data_dir(data_dir)
    path = resolved / "2022-presidential-results" / "moe_vuelta1.csv"
    df = _read_moe(path)
    return _aggregate_and_map(df, "nomparti", "votos")


def load_moe_round2(data_dir: Path | None = None) -> pd.DataFrame:
    """Load and aggregate MOE second-round (runoff) results.

    Args:
        data_dir: Path to the project data directory. Auto-resolved if ``None``.

    Returns:
        DataFrame indexed by candidate key with a ``votes`` column.

    """
    resolved = _resolve_data_dir(data_dir)
    path = resolved / "2022-presidential-results" / "moe_vuelta2.csv"
    df = _read_moe(path)
    return _aggregate_and_map(df, "nomparti", "votos")


def load_participation_round1(data_dir: Path | None = None) -> pd.DataFrame:
    """Load first-round participation data.

    Args:
        data_dir: Path to the project data directory. Auto-resolved if ``None``.

    Returns:
        DataFrame with columns including ``Total censo`` and ``Código Puesto``.

    """
    resolved = _resolve_data_dir(data_dir)
    path = resolved / "2022-presidential-results" / "reg_participacion_vuelta1.csv"
    return _read_participation(path)


def load_participation_round2(data_dir: Path | None = None) -> pd.DataFrame:
    """Load second-round participation data.

    Args:
        data_dir: Path to the project data directory. Auto-resolved if ``None``.

    Returns:
        DataFrame with columns including ``Total censo`` and ``Código Puesto``.

    """
    resolved = _resolve_data_dir(data_dir)
    path = resolved / "2022-presidential-results" / "reg_participacion_vuelta2.csv"
    return _read_participation(path)


# ═══════════════════════════════════════════════════════════════════
# Cross-validation & consolidation
# ═══════════════════════════════════════════════════════════════════


def cross_validate(
    registraduria: RoundResult,
    moe: RoundResult,
) -> list[str]:
    """Compare two ``RoundResult`` sources for the same election round.

    Args:
        registraduria: Result from the Registraduría (canonical source).
        moe: Result from MOE (validation source).

    Returns:
        List of warning message strings. Empty list = perfect match.

    """
    warnings: list[str] = []

    # Compare total valid votes
    if registraduria.total_valid_votes == 0:
        warnings.append("Registraduría total valid votes is 0; cannot compute percentage")
        return warnings
    diff_pct = (
        abs(registraduria.total_valid_votes - moe.total_valid_votes)
        / registraduria.total_valid_votes
        * 100
    )
    if diff_pct > _TOLERANCE_VALID_VOTES_PCT:
        warnings.append(
            f"Total valid votes differ by {diff_pct:.4f}% "
            f"(Reg: {registraduria.total_valid_votes}, "
            f"MOE: {moe.total_valid_votes})"
        )

    # Compare each candidate's vote share
    reg_shares = {c.candidate_key: c.vote_share for c in registraduria.candidates}
    moe_shares = {c.candidate_key: c.vote_share for c in moe.candidates}
    all_keys = set(reg_shares) | set(moe_shares)
    for key in sorted(all_keys):
        reg_val = reg_shares.get(key, 0.0)
        moe_val = moe_shares.get(key, 0.0)
        diff_pp = abs(reg_val - moe_val) * 100  # difference in percentage points
        if diff_pp > _TOLERANCE_CANDIDATE_SHARE_PCT:
            warnings.append(
                f"Candidate {key!r} share differs by {diff_pp:.4f}pp "
                f"(Reg: {reg_val:.4f}, MOE: {moe_val:.4f})"
            )

    return warnings


def consolidate_round(
    registraduria: RoundResult,
    moe: RoundResult,
    round_number: Literal[1, 2],
) -> RoundResult:
    """Return the canonical ``RoundResult`` for a round.

    The Registraduría result is treated as canonical. If MOE discrepancies
    exceed tolerances, a ``ValueError`` is raised with the validation report.

    Args:
        registraduria: Registraduría result (used as canonical).
        moe: MOE result (used for validation only).
        round_number: 1 or 2 (for error messages).

    Returns:
        The ``registraduria`` ``RoundResult`` unchanged.

    Raises:
        ValueError: If discrepancies exceed defined tolerances.

    """
    warnings = cross_validate(registraduria, moe)
    if warnings:
        report = "\n".join(warnings)
        msg = (
            f"Consolidation failed for round {round_number}: "
            f"MOE comparison exceeded tolerances.\n{report}"
        )
        raise ValueError(msg)
    return registraduria


# ═══════════════════════════════════════════════════════════════════
# Single entry point
# ═══════════════════════════════════════════════════════════════════


def load_actual_results(
    data_dir: Path | None = None,
) -> tuple[RoundResult, RoundResult]:
    """Load and consolidate both rounds' election results.

    Loads Registraduría and MOE data for both rounds, cross-validates,
    and returns the canonical (Registraduría) results.

    Args:
        data_dir: Path to the project data directory. Auto-resolved if ``None``.

    Returns:
        ``(round1, round2)`` as ``RoundResult`` objects.

    """
    resolved = _resolve_data_dir(data_dir)

    # Load Registraduría data
    reg1_df = load_registraduria_round1(resolved)
    reg2_df = load_registraduria_round2(resolved)

    # Load MOE data
    moe1_df = load_moe_round1(resolved)
    moe2_df = load_moe_round2(resolved)

    # Load participation data (used for census totals for both rounds)
    part1 = load_participation_round1(resolved)
    registered_voters = int(part1["Total censo"].sum())
    polling_stations = int(part1["Código Puesto"].nunique())

    # Build RoundResults for Registraduría
    reg_r1 = _build_round_result(reg1_df, 1, registered_voters, polling_stations)
    reg_r2 = _build_round_result(reg2_df, 2, registered_voters, polling_stations)

    # Build RoundResults for MOE (using same census totals)
    moe_r1 = _build_round_result(moe1_df, 1, registered_voters, polling_stations)
    moe_r2 = _build_round_result(moe2_df, 2, registered_voters, polling_stations)

    # Cross-validate and consolidate
    result1 = consolidate_round(reg_r1, moe_r1, 1)
    result2 = consolidate_round(reg_r2, moe_r2, 2)

    return result1, result2


# ═══════════════════════════════════════════════════════════════════
# SPEC-04: Poll Data Loading
# ═══════════════════════════════════════════════════════════════════


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
    resolved = _resolve_data_dir(data_dir)
    path = resolved / "2022-polls" / "encuestas_2022.csv"
    df = pd.read_csv(path, encoding="utf-8", low_memory=False)

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
    resolved = _resolve_data_dir(data_dir)
    path = resolved / "2022-polls" / "consultas.csv"
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)

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
            sample_size=int(row["muestra"]) if pd.notna(row["muestra"]) else 0,
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
    return result


_NS_NR_HUNDRED = 100.0


def _normalize_share_rows(
    df: pd.DataFrame,
    share_cols: list[str],
) -> tuple[set[int], set[int]]:
    """Apply undecided redistribution to rows with ``ns_nr > 0``.

    Returns ``(normalized_indices, skipped_100_indices)``.
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
    """Renormalise rows so share columns sum to 100, handling rounding drift."""
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
) -> None:
    """Assert that every normalised row sums to 100 +/- tolerance."""
    for idx in indices:
        vals = [df.loc[idx, col] for col in share_cols]  # pyright: ignore[reportUnknownVariableType]
        row_sum = sum(v for v in vals if isinstance(v, (int, float)) and not pd.isna(v))
        if row_sum <= 0:
            continue
        if abs(row_sum - 100.0) > _NORMALIZATION_TOLERANCE_PCT:
            msg = f"Row {idx}: normalized share sum = {row_sum:.2f}%, expected 100.0 +/- 1.0"
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

    All other polls: ``round_number = NaN``.

    Args:
        df: Poll DataFrame with candidate columns and ``fecha``.

    Returns:
        Copy of ``df`` with a new ``round_number`` column (Int64, nullable).

    """
    result = df.copy()
    round_numbers: list[int | float] = []
    for _, row in result.iterrows():
        if _is_round1_candidate(row, result.columns):
            round_numbers.append(1)
        elif _is_round2_candidate(row, result.columns):
            round_numbers.append(2)
        else:
            round_numbers.append(float("nan"))
    result["round_number"] = round_numbers
    result["round_number"] = result["round_number"].astype("Int64")
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
        return df

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
    return deduped.drop(columns=["_eff_sample"])


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

    # Step 3: Normalize undecided
    polls = normalize_undecided(polls)

    # ── all_polls snapshot ──
    all_polls = polls.copy()

    # Step 4: Retain active candidates (round 1 superset)
    r1_keys = [c.key for c in get_active_candidates(1)]
    polls = retain_active_candidates(polls, r1_keys)

    # Step 5: Infer round number
    polls = infer_round_number(polls)
    all_polls["round_number"] = polls["round_number"]

    # Step 6: Split by round
    mask_r1 = polls["round_number"] == 1
    mask_r2 = polls["round_number"] == _ROUND_TWO
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
        round_share_cols = [c for c in df_round.columns if c not in _SHARE_COLS_EXCLUDED]
        _renormalize_rows(df_round, round_share_cols, set(df_round.index))

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
