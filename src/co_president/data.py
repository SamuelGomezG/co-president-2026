"""SPEC-03: Election Results Data Consolidation.

Ingests Registraduría (MMV) and MOE result files, aggregates to national
vote totals per candidate, cross-validates them, and produces a single
canonical ``RoundResult`` per round.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal
import unicodedata

if TYPE_CHECKING:
    from datetime import date

import pandas as pd

from co_president.config import (
    COALITION_TO_CANDIDATE,
    ELECTION_DATE_ROUND1,
    ELECTION_DATE_ROUND2,
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
# Internal helpers
# ═══════════════════════════════════════════════════════════════════


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
