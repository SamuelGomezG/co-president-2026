"""SPEC-03, SPEC-23: Election results consolidation.

Ingests Registraduría (MMV) and MOE result files, aggregates to
national vote totals per candidate, cross-validates them, and produces a
single canonical ``RoundResult`` per round. SPEC-23 adds MOE PDF
cross-validation against the canonical results.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Literal
import unicodedata

if TYPE_CHECKING:
    from datetime import date
    from pathlib import Path

import numpy as np
import pandas as pd

from co_president.config import (
    COALITION_TO_CANDIDATE,
    ELECTION_DATE_ROUND1,
    ELECTION_DATE_ROUND2,
)
from co_president.paths import resolve_data_dir

__all__ = [
    "PDF_RESULTS_2022",
    "CandidateResult",
    "RoundResult",
    "consolidate_round",
    "cross_validate",
    "cross_validate_against_moe_pdf",
    "load_canonical_results",
    "load_moe_round1",
    "load_moe_round2",
    "load_participation_round1",
    "load_participation_round2",
    "load_registraduria_round1",
    "load_registraduria_round2",
]

logger = logging.getLogger(__name__)


_TOLERANCE_VALID_VOTES_PCT_DIFF = 0.10  # % difference in total valid votes → WARNING
_TOLERANCE_CANDIDATE_SHARE_PP = 0.50  # percentage point difference in vote share → WARNING
_MIN_CANDIDATES_FOR_TOP_TWO = 2  # minimum candidates needed for top_two()
_ROUND_TWO = 2  # second (runoff) round identifier
_MIN_PDF_ROW_LENGTH = 2  # minimum cells in a PDF table row to attempt parsing


def _get_series_int(series: pd.Series, key: str) -> int:
    """Return a series value as int, defaulting to 0 when missing/invalid."""
    if key not in series.index:
        return 0
    value = series[key]
    if pd.isna(value) or not isinstance(value, (int, float, np.integer, np.floating)):
        return 0
    if isinstance(value, (np.integer, np.floating)):
        return int(value.item())
    return int(value)


def _get_votes_int(df: pd.DataFrame, key: str) -> int:
    """Return vote count from DataFrame index or raise if missing."""
    if key not in df.index:
        msg = f"Expected key {key!r} missing from votes"
        raise KeyError(msg)
    value = df.loc[key, "votes"]
    if pd.isna(value) or not isinstance(value, (int, float, np.integer, np.floating)):
        msg = f"Votes for {key!r} are NA or non-numeric"
        raise ValueError(msg)
    if isinstance(value, (np.integer, np.floating)):
        return int(value.item())
    return int(value)


def _get_votes_int_or_zero(df: pd.DataFrame, key: str) -> int:
    """Return vote count as int or 0 when missing/invalid."""
    if key not in df.index:
        return 0
    value = df.loc[key, "votes"]
    if pd.isna(value) or not isinstance(value, (int, float, np.integer, np.floating)):
        return 0
    if isinstance(value, (np.integer, np.floating)):
        return int(value.item())
    return int(value)


def _get_column_sum_int(df: pd.DataFrame, col: str) -> int:
    """Return the sum of ``col`` as int, defaulting to 0 when empty."""
    if col not in df.columns:
        return 0
    total = df[col].sum()
    if pd.isna(total) or not isinstance(total, (int, float, np.integer, np.floating)):
        return 0
    if isinstance(total, (np.integer, np.floating)):
        return int(total.item())
    return int(total)


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
# Normalization & path resolution
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

    Raises:
        ValueError: If the file is missing required columns.

    """
    df = pd.read_csv(
        path,
        encoding="latin-1",
        sep=";",
        engine="pyarrow",
    )
    required_cols = {"CANNOMBRE", "PARNOMBRE", "VOTOS"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"MMV file {path} missing columns: {', '.join(sorted(missing))}"
        raise ValueError(msg)
    return df


def _read_moe(path: Path) -> pd.DataFrame:
    """Read an MOE results file (UTF-8, comma-delimited).

    Raises:
        ValueError: If the file is missing required columns.

    """
    df = pd.read_csv(path, encoding="utf-8", engine="pyarrow")
    required_cols = {"nomparti", "votos"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"MOE file {path} missing columns: {', '.join(sorted(missing))}"
        raise ValueError(msg)
    return df


def _read_participation(path: Path) -> pd.DataFrame:
    """Read a participation file (UTF-8 with BOM, comma-delimited).

    Some rows have extra fields due to commas in school names; those rows are
    skipped with a warning since they are edge cases (~1 per 12,500 rows).
    """
    df: pd.DataFrame
    try:
        df = pd.read_csv(
            path,
            encoding="utf-8-sig",
            sep=",",
            on_bad_lines="error",
        )
    except pd.errors.ParserError:
        logger.warning("Participation file %s has malformed rows; skipping bad lines", path)
        df = pd.read_csv(
            path,
            encoding="utf-8-sig",
            sep=",",
            on_bad_lines="skip",
        )
    required_cols = {"Total censo", "Código Puesto"}
    missing = required_cols - set(df.columns)
    if missing:
        msg = f"Participation file {path} missing columns: {', '.join(sorted(missing))}"
        raise ValueError(msg)
    return df


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

    Raises:
        ValueError: If any coalition name cannot be mapped via
            ``COALITION_TO_CANDIDATE``.

    """
    names = df[name_col].astype(str).str.strip()
    grouped = df.groupby(names)[votes_col].sum()
    unmapped: set[str] = set()
    result: dict[str, int] = {}
    for raw_name, votes in grouped.items():
        normalized = _normalize_coalition_name(str(raw_name))
        key = COALITION_TO_CANDIDATE.get(normalized)
        if key is None:
            unmapped.add(f"{raw_name!r} -> {normalized!r}")
            continue
        result[key] = result.get(key, 0) + int(votes)
    if unmapped:
        names_list = ", ".join(sorted(unmapped))
        msg = f"Unmapped coalition names (must be in COALITION_TO_CANDIDATE): {names_list}"
        raise ValueError(msg)
    result_df = pd.DataFrame(list(result.items()), columns=["candidate_key", "votes"])
    return result_df.set_index("candidate_key")


def _extract_excluded_votes(aggregated: pd.DataFrame) -> tuple[int, int, int]:
    """Extract null, unmarked, and blank votes from aggregated results.

    Args:
        aggregated: DataFrame with candidate_key index and votes column.

    Returns:
        Tuple of ``(null_votes, unmarked_votes, blank_votes)``. Missing keys
        default to 0.

    """
    null_votes = _get_votes_int_or_zero(aggregated, "nulos")
    unmarked_votes = _get_votes_int_or_zero(aggregated, "no_marcados")
    blank_votes = _get_votes_int_or_zero(aggregated, "blanco")
    return null_votes, unmarked_votes, blank_votes


def _merge_blanco_into_rest(
    candidate_df: pd.DataFrame,
    round_number: Literal[1, 2],
) -> pd.DataFrame:
    """Merge blanco votes into rest for round 2.

    In round 2 there is no separate blanco candidate, so blanco votes are
    merged into the ``"rest"`` category.

    Args:
        candidate_df: DataFrame with candidate_key index and votes column.
        round_number: 1 or 2.

    Returns:
        Modified DataFrame with blanco merged into rest if applicable.

    """
    if round_number != _ROUND_TWO or "blanco" not in candidate_df.index:
        return candidate_df.copy()
    blanco_count = _get_votes_int(candidate_df, "blanco")
    candidate_df = candidate_df.drop(index="blanco")
    existing = _get_votes_int_or_zero(candidate_df, "rest")
    candidate_df.loc["rest", "votes"] = existing + blanco_count
    return candidate_df


def _compute_candidate_results(
    candidate_df: pd.DataFrame,
    total_votes_incl_blank: int,
) -> tuple[CandidateResult, ...]:
    """Compute sorted CandidateResult objects from candidate vote data.

    Args:
        candidate_df: DataFrame with candidate_key index and votes column
            (candidate-only votes after excluded types are removed).
        total_votes_incl_blank: Total votes including blank (denominator for
            vote share).

    Returns:
        Tuple of ``CandidateResult`` sorted by votes descending.

    Raises:
        ZeroDivisionError: If ``total_votes_incl_blank`` is 0.

    """
    sorted_pairs = sorted(
        ((idx, row) for idx, row in candidate_df.iterrows()),
        key=lambda pair: _get_series_int(pair[1], "votes"),
        reverse=True,
    )
    return tuple(
        CandidateResult(
            candidate_key=str(idx),
            votes=_get_series_int(row, "votes"),
            vote_share=_get_series_int(row, "votes") / total_votes_incl_blank,
        )
        for idx, row in sorted_pairs
    )


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

    null_votes, unmarked_votes, blank_votes = _extract_excluded_votes(aggregated)

    candidate_df = aggregated.drop(index=["nulos", "no_marcados"], errors="ignore")
    candidate_df = _merge_blanco_into_rest(candidate_df, round_number)

    candidate_total = _get_column_sum_int(candidate_df, "votes")
    total_valid_votes = candidate_total - blank_votes
    total_votes_incl_blank = candidate_total
    if total_valid_votes == 0:
        msg = f"Round {round_number} has zero total valid votes after aggregation"
        raise ValueError(msg)

    candidates = _compute_candidate_results(candidate_df, total_votes_incl_blank)

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
    resolved = resolve_data_dir(data_dir)
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
    resolved = resolve_data_dir(data_dir)
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
    resolved = resolve_data_dir(data_dir)
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
    resolved = resolve_data_dir(data_dir)
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
    resolved = resolve_data_dir(data_dir)
    path = resolved / "2022-presidential-results" / "reg_participacion_vuelta1.csv"
    return _read_participation(path)


def load_participation_round2(data_dir: Path | None = None) -> pd.DataFrame:
    """Load second-round participation data.

    Args:
        data_dir: Path to the project data directory. Auto-resolved if ``None``.

    Returns:
        DataFrame with columns including ``Total censo`` and ``Código Puesto``.

    """
    resolved = resolve_data_dir(data_dir)
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
    if diff_pct > _TOLERANCE_VALID_VOTES_PCT_DIFF:
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
        if diff_pp > _TOLERANCE_CANDIDATE_SHARE_PP:
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


def load_canonical_results(
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
    resolved = resolve_data_dir(data_dir)

    # Load Registraduría data
    reg1_df = load_registraduria_round1(resolved)
    reg2_df = load_registraduria_round2(resolved)

    # Load MOE data
    moe1_df = load_moe_round1(resolved)
    moe2_df = load_moe_round2(resolved)

    # Load participation data
    part1 = load_participation_round1(resolved)
    part2 = load_participation_round2(resolved)
    registered_voters_r1 = int(part1["Total censo"].sum())
    polling_stations_r1 = int(part1["Código Puesto"].nunique())
    registered_voters_r2 = int(part2["Total censo"].sum())
    polling_stations_r2 = int(part2["Código Puesto"].nunique())

    # Build RoundResults for Registraduría
    reg_r1 = _build_round_result(reg1_df, 1, registered_voters_r1, polling_stations_r1)
    reg_r2 = _build_round_result(reg2_df, 2, registered_voters_r2, polling_stations_r2)

    # Build RoundResults for MOE
    moe_r1 = _build_round_result(moe1_df, 1, registered_voters_r1, polling_stations_r1)
    moe_r2 = _build_round_result(moe2_df, 2, registered_voters_r2, polling_stations_r2)

    # Cross-validate and consolidate
    result1 = consolidate_round(reg_r1, moe_r1, 1)
    result2 = consolidate_round(reg_r2, moe_r2, 2)

    return result1, result2


# ═══════════════════════════════════════════════════════════════════
# MOE PDF cross-validation (SPEC-23)
# ═══════════════════════════════════════════════════════════════════


PDF_RESULTS_2022 = "2022.11.09-LIBRO-RESULTADOS-ELECTORALES-PRESIDENCIALES-2022.pdf"

_SKIP_HEADER_WORDS = {"partido", "movimiento", "votos", "porcentaje", "%", "total", "subtotal"}


def _clean_vote_count(vote_str: str) -> int:
    """Convert a vote count string to an integer.

    Handles Colombian format (dots as thousands separators, e.g.
    ``8.542.020``) and international format (commas, e.g. ``8,542,020``).

    Args:
        vote_str: Raw vote count string from the PDF table.

    Returns:
        Integer vote count. Returns 0 if parsing fails.

    """
    cleaned = vote_str.strip().replace(".", "").replace(",", "")
    try:
        return int(cleaned)
    except ValueError:
        return 0


def _try_extract_pdfplumber(
    pdf_path: Path,
    page_range: tuple[int, int] | None = None,
) -> list[list[list[str]]] | None:
    """Extract tables from a PDF using pdfplumber.

    Args:
        pdf_path: Path to the PDF file.
        page_range: Optional ``(first_page, last_page_inclusive)`` 1-based
            range. When provided, only tables on pages within this range
            are extracted.

    Returns:
        List of tables (each table is a list of rows, each row is a list
        of cell strings). Returns ``None`` if pdfplumber is unavailable
        or extraction fails.

    """
    try:
        import pdfplumber  # noqa: PLC0415
    except ImportError:
        return None
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = pdf.pages
            if page_range is not None:
                first, last = page_range
                start_idx = max(0, first - 1)
                end_idx = min(len(pages), last)
                pages = pages[start_idx:end_idx]
            tables: list[list[list[str]]] = []
            for page in pages:
                page_tables = page.extract_tables()
                if not page_tables:
                    continue
                for raw_table in page_tables:
                    cleaned = [
                        [str(cell) if cell is not None else "" for cell in row] for row in raw_table
                    ]
                    tables.append(cleaned)
            return tables or None
    except (OSError, ValueError):
        logger.exception("pdfplumber extraction failed for %s", pdf_path)
        return None


def _try_extract_pymupdf(
    pdf_path: Path,
    page_range: tuple[int, int] | None = None,
) -> list[list[list[str]]] | None:
    """Extract tables from a PDF using PyMuPDF.

    Note: ``find_tables()`` works on text-based PDFs with embedded table
    structures. It does not perform OCR and will not extract tables from
    image-only (scanned) pages.

    Args:
        pdf_path: Path to the PDF file.
        page_range: Optional ``(first_page, last_page_inclusive)`` 1-based
            range. When provided, only tables on pages within this range
            are extracted.

    Returns:
        List of tables (same structure as pdfplumber). Returns ``None``
        if pymupdf is unavailable or extraction fails.

    """
    try:
        import pymupdf  # noqa: PLC0415
    except ImportError:
        return None
    try:
        doc = pymupdf.open(pdf_path)
    except (OSError, ValueError):
        logger.exception("pymupdf open failed for %s", pdf_path)
        return None
    tables: list[list[list[str]]] = []
    try:
        pages = list(doc)  # type: ignore[reportUnknownVariableType]
        if page_range is not None:
            first, last = page_range
            start_idx = max(0, first - 1)
            end_idx = min(len(pages), last)  # type: ignore[reportUnknownArgumentType]
            pages = pages[start_idx:end_idx]  # type: ignore[reportUnknownVariableType]
        for page in pages:  # type: ignore[reportUnknownVariableType]
            page_tables = page.find_tables()  # type: ignore[reportUnknownMemberType]
            for table in page_tables or []:  # type: ignore[reportUnknownVariableType]
                data = table.extract()  # type: ignore[reportUnknownMemberType]
                cleaned = [
                    [str(cell) if cell is not None else "" for cell in row]  # type: ignore[reportUnknownArgumentType]
                    for row in data  # type: ignore[reportUnknownVariableType]
                ]
                tables.append(cleaned)
    except (IndexError, ValueError, OSError, RuntimeError):
        logger.exception("pymupdf page extraction failed for %s", pdf_path)
        return None
    finally:
        doc.close()
    return tables or None


def _extract_tables_with_fallback(
    pdf_path: Path,
    page_range: tuple[int, int] | None = None,
) -> list[list[list[str]]]:
    """Extract tables from a PDF with pdfplumber to pymupdf fallback.

    Tries pdfplumber first. If it returns no tables or raises an error,
    falls back to pymupdf as a secondary table-detection engine.

    Args:
        pdf_path: Path to the PDF file.
        page_range: Optional ``(first_page, last_page_inclusive)`` 1-based
            range. When provided, only tables on pages within this range
            are extracted.

    Returns:
        List of extracted tables. Empty list if both engines fail.

    """
    tables = _try_extract_pdfplumber(pdf_path, page_range)
    if tables:
        return tables
    tables = _try_extract_pymupdf(pdf_path, page_range)
    if tables:
        return tables
    return []


def _is_header_or_summary(row: list[str]) -> bool:
    """Check if a table row is a header or summary row.

    A row is considered a header/summary row if any cell (lowercased
    and stripped) exactly matches a known header word. This avoids
    false positives on coalition names like ``"VOTOS EN BLANCO"``.

    Args:
        row: A single row from an extracted PDF table.

    Returns:
        ``True`` if the row should be skipped.

    """
    for cell in row:
        cell_lower = cell.strip().lower()
        if cell_lower in _SKIP_HEADER_WORDS:
            return True
    return False


def _parse_pdf_tables(tables: list[list[list[str]]]) -> pd.DataFrame:
    """Convert extracted PDF tables into a name/votes DataFrame.

    Scans all tables for rows containing a coalition/candidate name
    and a vote count. Header rows, summary rows, and unparseable rows
    are skipped.

    Args:
        tables: Raw tables from ``_extract_tables_with_fallback``.

    Returns:
        DataFrame with columns ``name`` (coalition/candidate name) and
        ``votes`` (integer vote count). The caller is responsible for
        mapping names to canonical keys via ``_aggregate_and_map``.

    Raises:
        ValueError: If no parseable rows are found in the extracted tables.

    """
    rows: list[dict[str, str | int]] = []
    for table in tables:
        for row in table:
            if len(row) < _MIN_PDF_ROW_LENGTH:
                continue
            name = str(row[0]).strip()
            vote_str = str(row[1]).strip()
            if not name or not vote_str:
                continue
            if _is_header_or_summary(row):
                continue
            votes = _clean_vote_count(vote_str)
            if votes <= 0:
                continue
            rows.append({"name": name, "votes": votes})
    if not rows:
        msg = (
            "No parseable rows found in extracted PDF tables. "
            "Check that the PDF contains tables with coalition names "
            "and vote counts."
        )
        raise ValueError(msg)
    return pd.DataFrame(rows)


def _build_from_pdf_tables(
    tables: list[list[list[str]]],
    round_number: Literal[1, 2],
    registered_voters: int,
    polling_stations: int,
) -> RoundResult:
    """Build a ``RoundResult`` from extracted PDF tables.

    Parses the tables into a DataFrame, maps coalition names to
    candidate keys via ``COALITION_TO_CANDIDATE``, and builds a
    complete ``RoundResult`` using the standard pipeline.

    Args:
        tables: Raw tables from ``_extract_tables_with_fallback``.
        round_number: 1 or 2.
        registered_voters: Total registered voters for this round.
        polling_stations: Unique polling stations for this round.

    Returns:
        A fully populated ``RoundResult`` derived from the PDF data.

    """
    parsed = _parse_pdf_tables(tables)
    aggregated = _aggregate_and_map(parsed, "name", "votes")
    return _build_round_result(aggregated, round_number, registered_voters, polling_stations)


def cross_validate_against_moe_pdf(
    registraduria: RoundResult,
    moe_csv: RoundResult,
    pdf_path: Path,
) -> list[str]:
    """Parse the MOE PDF and perform 3-way cross-validation.

    Extracts tables from the MOE PDF (pages 6--14 which contain the
    presidential results), reconstructs a ``RoundResult`` from the
    extracted data, and cross-validates it against both the Registraduría
    and MOE CSV results using the same tolerances as ``cross_validate``.

    Args:
        registraduria: Canonical result from the Registraduría.
        moe_csv: Result from the MOE CSV files.
        pdf_path: Path to the MOE PDF file.

    Returns:
        List of warning messages. Empty list = all three sources agree
        within tolerances.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If the PDF contains no parseable result tables.

    Examples:
        >>> from pathlib import Path
        >>> from co_president.data_results import (
        ...     cross_validate_against_moe_pdf,
        ...     load_canonical_results,
        ... )
        >>> r1, _r2 = load_canonical_results()
        >>> pdf_path = Path("data/2022-presidential-results/2022.11.09-...pdf")
        >>> warnings = cross_validate_against_moe_pdf(r1, r1, pdf_path)
        >>> if warnings:
        ...     for w in warnings:
        ...         print(w)

    """
    if not pdf_path.exists():
        msg = f"MOE PDF not found: {pdf_path}"
        raise FileNotFoundError(msg)

    pdf_tables = _extract_tables_with_fallback(pdf_path, page_range=(6, 14))

    round_number = registraduria.round_number
    pdf_result = _build_from_pdf_tables(
        pdf_tables,
        round_number,
        registraduria.registered_voters,
        registraduria.polling_stations,
    )

    warnings: list[str] = []
    warnings.extend(f"Reg vs PDF: {w}" for w in cross_validate(registraduria, pdf_result))
    warnings.extend(f"MOE CSV vs PDF: {w}" for w in cross_validate(moe_csv, pdf_result))
    return warnings
