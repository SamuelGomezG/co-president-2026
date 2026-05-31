"""SPEC-12.2: Historical election results ingestion.

Downloads municipal-level results for all presidential elections from
2002 to 2022 from the CEDAE database (``cedae.datasketch.co``) and
computes lagged features including vote share deltas and abstention
rates.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
import requests

if TYPE_CHECKING:
    from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential

from co_president.paths import resolve_data_dir

__all__ = [
    "build_historical_matrix",
    "compute_derived_features",
    "fetch_cedae_results",
    "fetch_datos_gov_results",
]

logger = logging.getLogger(__name__)

_ELECTION_YEARS = [2002, 2006, 2010, 2014, 2018, 2022]
_CEDAE_BASE_URL = "https://cedae.datasketch.co/api/results"
_fallback_results_cache: pd.DataFrame | None = None


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


def build_historical_matrix(data_dir: Path | None = None) -> None:
    """Fetch all election years, compute features, and save the result.

    Args:
        data_dir: Target data directory. If ``None``, resolves via
            ``resolve_data_dir``.

    """
    if data_dir is None:
        data_dir = resolve_data_dir(None)

    combined = _fetch_all_years()
    if combined.empty:
        logger.error(
            "No historical election data fetched for any year in %s. "
            "Check network connectivity and CEDAE/datos.gov API availability.",
            _ELECTION_YEARS,
        )
        return
    features = compute_derived_features(combined)

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


def _fetch_all_years() -> pd.DataFrame:
    """Fetch data for every combination of election year and round."""
    all_frames: list[pd.DataFrame] = []

    for year in _ELECTION_YEARS:
        for round_num in (1, 2):
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
