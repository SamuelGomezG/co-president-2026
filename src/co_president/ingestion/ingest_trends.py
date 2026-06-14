"""SPEC-38: Google Trends ingestion for Colombian election forecasting.

Provides ``fetch_trends`` to retrieve search interest from Google Trends
(via ``pytrends``) with tenacity retry and combined Google+YouTube mode,
and ``compute_prop_fav`` to transform normalized interest into
favorable-propensity proportions.

Based on SciELO 2023 (Perez-Rave et al.): T-1 day-before window achieves
1.86pp MAE on 2022 R2 vs 6.56pp on election-day distortion.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
import logging
from typing import TypedDict

import pandas as pd
from pytrends.request import TrendReq  # type: ignore[import-untyped]
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

_PYTRENDS_RETRIES = 3


def _resolve_window(window: str) -> str:
    """Convert window alias to pytrends timeframe string.

    Returns explicit date ranges ending at yesterday (avoids
    election-day bot-traffic distortion per SciELO 2023).

    Args:
        window: One of ``"T-1"`` (yesterday), ``"T-7"`` (week ending
            yesterday), ``"full_cycle"`` (full campaign window).

    Returns:
        pytrends-compatible timeframe string.

    Raises:
        ValueError: If window is not recognised.

    """
    if window in ("T-1", "T-7"):
        today = datetime.now(tz=UTC).date()
        yesterday = (today - timedelta(days=1)).isoformat()
        if window == "T-1":
            return f"{yesterday} {yesterday}"
        return f"{(today - timedelta(days=7)).isoformat()} {yesterday}"
    if window == "full_cycle":
        return "2021-09-01 2022-06-18"
    msg = f"Unknown window: {window!r}"
    raise ValueError(msg)


@retry(
    stop=stop_after_attempt(_PYTRENDS_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=30),
)
def _fetch_pytrends(
    kw_list: list[str],
    timeframe: str,
    geo: str = "CO",
    gprop: str = "",
) -> pd.DataFrame:
    """Execute a single pytrends interest_over_time call.

    Args:
        kw_list: Keywords to query.
        timeframe: pytrends-compatible timeframe string.
        geo: Geographic region (default ``"CO"``).
        gprop: Google property (``""`` for web, ``"youtube"`` for
            YouTube search, etc.).

    Returns:
        DataFrame indexed by date with keyword columns, or empty
        DataFrame if no data returned.

    """
    req = TrendReq(hl="es-419")
    req.build_payload(kw_list=kw_list, timeframe=timeframe, geo=geo, gprop=gprop)  # type: ignore[reportUnknownMemberType]
    result = req.interest_over_time()
    if result.empty:
        logger.warning("pytrends returned empty DataFrame for %s", kw_list)
        return pd.DataFrame()
    return result.drop(columns=["isPartial"], errors="ignore")


def fetch_trends(
    keywords: str | list[str],
    start_date: str | None = None,
    end_date: str | None = None,
    *,
    window: str = "T-1",
    geo: str = "CO",
) -> pd.DataFrame:
    """Fetch Google Trends search interest for election keywords.

    Two modes:

    - **Date range**: provide ``start_date`` and ``end_date`` in
      ``YYYY-MM-DD`` format.
    - **Window alias**: omit dates, provide ``window`` (default
      ``"T-1"``).

    Averages Google web search and YouTube search interest with equal
    weight (combined mode).

    Args:
        keywords: Single keyword string or list of keywords.
        start_date: Start date in ``YYYY-MM-DD`` format.
        end_date: End date in ``YYYY-MM-DD`` format.
        window: Window alias (``"T-1"``, ``"T-7"``, or
            ``"full_cycle"``). Ignored when dates are provided.
        geo: Geographic region (default ``"CO"``).

    Returns:
        DataFrame indexed by date with columns for each keyword.

    Warnings:
        Logs a WARN if the user requests an election-day window (T=0),
        since that introduces bot-traffic distortion.

    """
    if (start_date is None) != (end_date is None):
        msg = "Both start_date and end_date must be provided together, or neither."
        raise ValueError(msg)

    if start_date and end_date and start_date == end_date:
        logger.warning(
            "Single-day query (%s) requested - election-day Google Trends "
            "data may include bot-traffic distortion. "
            "Prefer T-1 (day before) per SciELO 2023.",
            start_date,
        )

    timeframe = f"{start_date} {end_date}" if start_date and end_date else _resolve_window(window)

    kw_list = [keywords] if isinstance(keywords, str) else keywords

    web_df = _fetch_pytrends(kw_list, timeframe, geo=geo, gprop="")
    yt_df = _fetch_pytrends(kw_list, timeframe, geo=geo, gprop="youtube")

    if web_df.empty and yt_df.empty:
        logger.warning("Both web and YouTube trends returned empty for %s", kw_list)
        return pd.DataFrame()
    if web_df.empty:
        return yt_df
    if yt_df.empty:
        return web_df

    return (web_df.astype(float) + yt_df.astype(float)) / 2.0


class PropFavRow(TypedDict):
    """Row type for ``compute_prop_fav`` output.

    Each row records favorable propensity for one candidate on one date.

    Attributes:
        as_of_date: Date of the observation.
        candidate: Internal candidate key (e.g. ``gustavo_petro``).
        prop_fav: Favorable propensity proportion ∈ [0, 1].

    """

    as_of_date: date
    candidate: str
    prop_fav: float


def compute_prop_fav(
    trends_df: pd.DataFrame,
    query_map: dict[str, str],
) -> pd.DataFrame:
    """Compute favorable propensity proportion from trends interest.

    ``prop_fav(A) = interest_A / sum(interest_all)`` per day.
    Rows where total interest is zero are dropped.

    Uses ``query_map`` (candidate_key -> query_string) to map trend
    column names back to internal candidate keys.

    Args:
        trends_df: DataFrame indexed by date with query-string columns.
        query_map: Mapping of candidate key to search query string.

    Returns:
        Long-format DataFrame with columns ``as_of_date``, ``candidate``
        (candidate key), ``prop_fav``.

    """
    query_to_candidate = {v: k for k, v in query_map.items()}
    available = [q for q in query_to_candidate if q in trends_df.columns]
    if not available:
        return pd.DataFrame(columns=["as_of_date", "candidate", "prop_fav"])

    subset = trends_df[available].copy()
    total = subset.sum(axis=1)
    has_interest = total > 0
    subset = subset.loc[has_interest]
    totals = total[has_interest]
    proportions = subset.div(totals, axis=0)

    rows: list[PropFavRow] = []
    for date_idx in proportions.index:
        day = date_idx.date() if hasattr(date_idx, "date") else date_idx
        rows.extend(
            {  # type: ignore[reportArgumentType]
                "as_of_date": day,
                "candidate": query_to_candidate[q],
                "prop_fav": proportions.loc[date_idx, q],
            }
            for q in available
        )

    return pd.DataFrame(rows)


__all__ = [
    "PropFavRow",
    "compute_prop_fav",
    "fetch_trends",
]
