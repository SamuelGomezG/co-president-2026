"""SPEC-12.2: Historical election results ingestion.

Downloads municipal-level results for all presidential elections from
2002 to 2022 from the CEDAE database (``cedae.datasketch.co``) and
computes lagged features including vote share deltas and abstention
rates.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import unicodedata

import pandas as pd
import requests

if TYPE_CHECKING:
    from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential

from co_president.ingestion._download_cedae import fetch_local_cedae_results
from co_president.paths import resolve_data_dir

__all__ = [
    "build_historical_matrix",
    "compute_derived_features",
    "compute_lagged_features",
    "fetch_cedae_results",
    "fetch_datos_gov_results",
    "map_historical_candidate",
    "validate_vote_shares",
]

logger = logging.getLogger(__name__)

_ELECTION_YEARS = [2002, 2006, 2010, 2014, 2018, 2022]
_CEDAE_BASE_URL = "https://cedae.datasketch.co/api/results"
_fallback_results_cache: pd.DataFrame | None = None

_HISTORICAL_CANDIDATE_MAP: dict[str, str] = {
    "gustavo_petro": "gustavo_petro",
    "rodolfo_hernandez": "rodolfo_hernandez",
    "federico_gutierrez": "federico_gutierrez",
    "sergio_fajardo": "sergio_fajardo",
    "ingrid_betancourt": "ingrid_betancourt",
    "ivan_duque": "ivan_duque",
    "oscar_ivan_zuluaga": "oscar_ivan_zuluaga",
    "luis_eduardo_garzon": "luis_eduardo_garzon",
    "juan_manuel_santos": "juan_manuel_santos",
    "alvaro_uribe": "alvaro_uribe",
    "carlos_gaviria": "carlos_gaviria",
    "clara_lopez": "clara_lopez",
    "horacio_serpa": "horacio_serpa",
    "antanas_mockus": "antanas_mockus",
    "noemi_sanin": "noemi_sanin",
    "enrique_penalosa": "enrique_penalosa",
}

_CANDIDATE_IDEOLOGY: dict[int, dict[str, str]] = {
    2002: {"left": "horacio_serpa", "right": "alvaro_uribe"},
    2006: {"left": "carlos_gaviria", "right": "alvaro_uribe"},
    2010: {"left": "gustavo_petro", "right": "juan_manuel_santos"},
    2014: {"left": "clara_lopez", "right": "oscar_ivan_zuluaga"},
    2018: {"left": "gustavo_petro", "right": "ivan_duque"},
    2022: {"left": "gustavo_petro", "right": "federico_gutierrez"},
}

# Round-2 overrides for elections that had different right-wing candidates in runoff
_ROUND2_IDEOLOGY: dict[int, dict[str, str]] = {
    2022: {"left": "gustavo_petro", "right": "rodolfo_hernandez"},
}
_ROUND2: int = 2  # Round number for the runoff election


def _get_ideology(year: int, round_num: int) -> dict[str, str]:
    """Return the left/right ideology mapping for a given year and round.

    The main ``_CANDIDATE_IDEOLOGY`` covers round-1 configurations.
    ``_ROUND2_IDEOLOGY`` provides round-specific overrides (e.g., 2022 R2
    used Rodolfo Hernandez as the right candidate instead of Gutierrez).
    """
    if round_num == _ROUND2 and year in _ROUND2_IDEOLOGY:
        return _ROUND2_IDEOLOGY[year]
    return _CANDIDATE_IDEOLOGY.get(year, {})


_historical_candidate_lookup: dict[str, str] | None = None


def map_historical_candidate(name: str) -> str:
    """Map a historical candidate name to a canonical key.

    Normalises the input name (Unicode NFKD decomposition, accent stripping,
    lowercasing, hyphens/spaces to underscores) and looks it up in the
    candidate registry. Raises ``ValueError`` for unrecognised names
    (no silent data loss).

    Args:
        name: Candidate name as returned by the CEDAE API.

    Returns:
        Canonical candidate key (e.g. ``"gustavo_petro"``).

    Raises:
        ValueError: If the name cannot be mapped.

    """
    global _historical_candidate_lookup  # noqa: PLW0603
    if _historical_candidate_lookup is None:
        _historical_candidate_lookup = {
            k.lower().replace("-", "_").replace(" ", "_"): v
            for k, v in _HISTORICAL_CANDIDATE_MAP.items()
        }
    # Strip first so leading/trailing spaces don't become underscores
    clean = (
        unicodedata.normalize("NFKD", name)
        .encode("ascii", "ignore")
        .decode("ascii")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    if clean in _historical_candidate_lookup:
        return _historical_candidate_lookup[clean]
    msg = f"Unrecognised historical candidate: {name!r}"
    raise ValueError(msg)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_cedae_results(year: int, round_num: int) -> pd.DataFrame:
    """Fetch election results for a specific year and round from CEDAE.

    Args:
        year: Election year (e.g., 2022).
        round_num: Election round (1 or 2).

    Returns:
        DataFrame with columns ``codigo_municipio``, ``year``, ``round``,
        ``candidate``, ``votes``, ``total_votes``, ``registered_voters``.

    Raises:
        requests.RequestException: If the API is unreachable.

    """
    url = f"{_CEDAE_BASE_URL}/{year}/{round_num}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame.from_records(data)
    df["year"] = year
    df["round"] = round_num
    return df


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
def fetch_datos_gov_results() -> pd.DataFrame:
    """Fallback: fetch from ``datos.gov.co`` Socrata API.

    Returns:
        DataFrame with election result data.

    """
    from sodapy import Socrata  # type: ignore[reportMissingTypeStubs]  # noqa: PLC0415

    with Socrata("www.datos.gov.co", None) as client:
        results = client.get("jhy3-m55z", limit=100000)  # type: ignore[reportUnknownMemberType]
        return pd.DataFrame.from_records(results)


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived election features (vote share and abstention rate).

    For each row:
    - ``vote_share = votes / total_votes``  (guarded against zero total_votes)
    - ``abstention_rate = 1 - (total_votes / registered_voters)``
      (guarded against zero registered_voters)

    Args:
        df: DataFrame with columns ``codigo_municipio``, ``year``, ``round``,
            ``candidate``, ``votes``, ``total_votes``, ``registered_voters``.

    Returns:
        DataFrame with added ``vote_share`` and ``abstention_rate`` columns.

    """
    result = df.copy()
    safe_total = result["total_votes"].replace(0, pd.NA)
    safe_registered = result["registered_voters"].replace(0, pd.NA)
    result["vote_share"] = (result["votes"] / safe_total).fillna(0.0).clip(0.0, 1.0)
    result["abstention_rate"] = (
        (1 - (result["total_votes"] / safe_registered)).fillna(1.0).clip(0.0, 1.0)
    )
    return result


def compute_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-municipality per-year historical features.

    For each (municipality, year) pair identifies the left and right
    candidates from ``_CANDIDATE_IDEOLOGY`` and returns their vote
    shares together with the abstention rate and election-to-election
    deltas.

    Args:
        df: DataFrame with columns ``codigo_municipio``, ``year``,
            ``round``, ``candidate``, ``vote_share``, ``abstention_rate``
            (typically the output of ``compute_derived_features``).

    Returns:
        DataFrame with one row per (municipio, year, round) and columns:
        ``codigo_municipio``, ``year``, ``round``, ``left_share``,
        ``right_share``, ``abstention_rate``, ``delta_left``,
        ``delta_right``, ``delta_abstention``.

    """
    mapped = df.copy()
    # Attempt to canonicalize candidate names; skip unrecognized (minor/third-party)
    # candidates — only the left/right picks per group need to match.
    _canonicalized: list[str | None] = []
    for raw in mapped["candidate"]:
        try:
            _canonicalized.append(map_historical_candidate(str(raw)))
        except ValueError:
            _canonicalized.append(None)
    mapped["candidate"] = _canonicalized

    features_rows: list[dict[str, object]] = []
    for (municipio, year, round_num), group in mapped.groupby(
        ["codigo_municipio", "year", "round"]
    ):
        # Filter out unrecognized candidates before lookup
        recognized = group[group["candidate"].notna()]
        if recognized.empty:
            continue
        # year/round_num are Hashable from groupby; cast to int for lookup
        year_int: int = int(year)  # type: ignore[arg-type]
        round_int: int = int(round_num)  # type: ignore[arg-type]
        ideology = _get_ideology(year_int, round_int)
        left_cand: str | None = ideology.get("left")
        right_cand: str | None = ideology.get("right")

        cand_shares: dict[str | None, float] = dict(
            zip(recognized["candidate"], recognized["vote_share"], strict=False)
        )
        left_share = cand_shares.get(left_cand) if left_cand else None
        right_share = cand_shares.get(right_cand) if right_cand else None
        abs_rates = group["abstention_rate"]
        abs_rate = float(abs_rates.iloc[0])
        if abs_rates.max() - abs_rates.min() > 0:
            logger.warning(
                "Inconsistent abstention rates for %s/%s/%s; using first: %s",
                municipio,
                year,
                round_num,
                list(abs_rates.unique()),
            )

        features_rows.append(
            {
                "codigo_municipio": str(municipio),
                "year": year,
                "round": round_num,
                "left_share": left_share if left_share is not None else float("nan"),
                "right_share": right_share if right_share is not None else float("nan"),
                "abstention_rate": abs_rate,
            }
        )

    features = pd.DataFrame(features_rows)
    if features.empty:
        return features
    features = features.sort_values(["codigo_municipio", "year", "round"]).reset_index(drop=True)

    # Deltas are computed per municipality per round to avoid mixing R1/R2 comparisons
    features["delta_left"] = features.groupby(["codigo_municipio", "round"])["left_share"].diff()
    features["delta_right"] = features.groupby(["codigo_municipio", "round"])["right_share"].diff()
    features["delta_abstention"] = features.groupby(["codigo_municipio", "round"])[
        "abstention_rate"
    ].diff()
    return features


_VOTE_SHARE_TOLERANCE: float = 0.02  # Maximum acceptable deviation from 100% sum


def validate_vote_shares(df: pd.DataFrame) -> list[str]:
    """Check that vote shares sum to ~100% per municipality-year-round.

    Args:
        df: DataFrame with columns ``codigo_municipio``, ``year``,
            ``round``, ``vote_share``.

    Returns:
        List of warning messages (empty if all checks pass).

    """
    warnings: list[str] = []
    grouped = df.groupby(["codigo_municipio", "year", "round"])
    for (municipio, year, round_num), group in grouped:
        total = float(group["vote_share"].sum())
        if abs(total - 1.0) > _VOTE_SHARE_TOLERANCE:
            warnings.append(f"{municipio} {year} round {round_num}: vote shares sum to {total:.3f}")
    return warnings


def build_historical_matrix(data_dir: Path | None = None) -> None:
    """Fetch all election years, compute features, and save the result.

    Args:
        data_dir: Target data directory. If ``None``, resolves via
            ``resolve_data_dir``.

    """
    if data_dir is None:
        data_dir = resolve_data_dir(None)

    combined = _fetch_all_years(data_dir)
    if combined.empty:
        logger.error(
            "No historical election data fetched for any year in %s. "
            "Check network connectivity and CEDAE/datos.gov API availability.",
            _ELECTION_YEARS,
        )
        return
    features = compute_derived_features(combined)

    # Canonicalize candidate names before writing to disk and before
    # downstream consumers (validate_vote_shares, compute_lagged_features).
    # Unrecognised names are left as-is (logged) rather than raising.
    def _safe_canonicalize(name: str) -> str:
        try:
            return map_historical_candidate(name)
        except ValueError:
            logger.warning("Unrecognised historical candidate name preserved as-is: %r", name)
            return name

    features["candidate"] = features["candidate"].apply(_safe_canonicalize)

    # Validate vote shares
    validation_warnings = validate_vote_shares(features)
    for warning in validation_warnings:
        logger.warning("Vote share validation: %s", warning)

    # Compute lagged features
    lagged = compute_lagged_features(features)

    target_dir = data_dir / "fundamentals"
    target_dir.mkdir(parents=True, exist_ok=True)

    target_path = target_dir / "historical_results.csv"
    features.to_csv(target_path, index=False)
    logger.info(
        "Historical results saved to %s (%d rows, %d years)",
        target_path,
        len(features),
        features["year"].nunique(),
    )

    if not lagged.empty:
        lagged_path = target_dir / "historical_lagged_features.csv"
        lagged.to_csv(lagged_path, index=False)
        logger.info(
            "Lagged features saved to %s (%d rows)",
            lagged_path,
            len(lagged),
        )


# ═══════════════════════════════════════════════════════════════════
# Private helpers
# ═══════════════════════════════════════════════════════════════════


def _get_fallback_results() -> pd.DataFrame:
    """Fetch and cache the full datos.gov.co Socrata dataset (lazy, one-time).

    Caching avoids re-downloading the ~100k-row dataset on every fallback
    call across the 12 (year, round) iterations.

    Returns:
        DataFrame with election result data from datos.gov.co.

    """
    global _fallback_results_cache  # noqa: PLW0603
    if _fallback_results_cache is None:
        _fallback_results_cache = fetch_datos_gov_results()
    return _fallback_results_cache.copy()


def _fetch_all_years(data_dir: Path | None = None) -> pd.DataFrame:
    """Fetch data for every combination of election year and round.

    Priority order:
    1. Local CEDAE ``.dta.csv.gz`` files (fastest, no network)
    2. Remote CEDAE REST API
    3. Socrata ``datos.gov.co`` fallback.

    Args:
        data_dir: Root data directory.  If ``None``, resolves via
            ``resolve_data_dir``.

    Returns:
        DataFrame with all election years and rounds combined.

    """
    if data_dir is None:
        data_dir = resolve_data_dir(None)
    all_frames: list[pd.DataFrame] = []

    for year in _ELECTION_YEARS:
        for round_num in (1, 2):
            # Level 1: Local CEDAE files
            local_frame = fetch_local_cedae_results(year, round_num, data_dir=data_dir)
            if local_frame is not None and not local_frame.empty:
                all_frames.append(local_frame)
                logger.info(
                    "Loaded %s round %d from local CEDAE (%d rows)",
                    year,
                    round_num,
                    len(local_frame),
                )
                continue

            # Level 2: Remote CEDAE API
            try:
                frame = fetch_cedae_results(year, round_num)
                all_frames.append(frame)
                logger.info("Fetched %s round %d (%d rows)", year, round_num, len(frame))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to fetch %s round %d (%s); trying fallback",
                    year,
                    round_num,
                    exc,
                )
                try:
                    frame = _get_fallback_results()
                    # Defensively check required columns before filtering
                    if "year" not in frame.columns or "round" not in frame.columns:
                        logger.warning(
                            "Fallback data missing required columns; "
                            "year present=%s, round present=%s",
                            "year" in frame.columns,
                            "round" in frame.columns,
                        )
                        continue
                    # Coerce to numeric to handle string-typed API responses
                    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
                    frame["round"] = pd.to_numeric(frame["round"], errors="coerce")
                    frame = frame[(frame["year"] == year) & (frame["round"] == round_num)]
                    if not frame.empty:
                        all_frames.append(frame)
                    else:
                        logger.warning(
                            "Fallback returned no rows for %s round %d after filtering",
                            year,
                            round_num,
                        )
                except Exception:
                    logger.exception(
                        "Fallback also failed for %s round %d",
                        year,
                        round_num,
                    )

    if not all_frames:
        logger.warning("No historical data fetched for any year/round pair")
        return pd.DataFrame()
    return pd.concat(all_frames, ignore_index=True)
