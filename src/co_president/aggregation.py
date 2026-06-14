"""SPEC-05: Poll Aggregation (Baseline).

Produces simple weighted polling averages as a baseline for comparison
with the Bayesian model. This is NOT the final forecast — it provides
a benchmark to validate that the Bayesian model adds value over simple
heuristics.
"""

from __future__ import annotations

from datetime import date, timedelta
import logging
import statistics

import numpy as np
import pandas as pd

from co_president.config import (
    CONSULTATION_DATE,
    pollster_weight_formula,
)
from co_president.ingestion.ingest_trends import compute_prop_fav
from co_president.ingestion.trends_keywords import CANDIDATE_QUERY_MAP_2026

__all__ = [
    "aggregation_snapshot",
    "combined_weight",
    "evolution_series",
    "pollster_weight_map",
    "sample_size_weight",
    "time_weight",
    "weighted_average",
]

logger = logging.getLogger(__name__)

_DEFAULT_HALF_LIFE = 30.0


def time_weight(
    dates: pd.Series,
    election_date: date,
    half_life: float = _DEFAULT_HALF_LIFE,
) -> pd.Series:
    """Compute exponential decay time weights relative to an election date.

    ``w_time = 0.5 ** (abs((election_date - poll_date).days) / half_life)``.

    A poll on election day has weight 1.0. A poll 30 days before election
    day (with default half-life) has weight 0.5.

    Args:
        dates: Series of poll dates (``datetime64`` or ``date``).
        election_date: Reference date for weight computation.
        half_life: Days for weight to halve.

    Returns:
        Series of time weights in ``[0, 1]``.

    Examples:
        >>> import pandas as pd
        >>> from datetime import date
        >>> tw = time_weight(pd.Series([date(2022, 5, 29), date(2022, 4, 29)]), date(2022, 5, 29))
        >>> round(float(tw.iloc[0]), 2)
        1.0
        >>> round(float(tw.iloc[1]), 2)
        0.5

    """
    days_diff = (pd.Timestamp(election_date) - pd.to_datetime(dates)).dt.days

    if (days_diff < 0).any():
        num_future = (days_diff < 0).sum()
        logger.warning(
            "time_weight: %d poll(s) have dates after election date %s. "
            "Setting their weights to 0.",
            num_future,
            election_date,
        )

    return (0.5 ** (days_diff / half_life)).where(days_diff >= 0, 0.0)


def sample_size_weight(sample_sizes: pd.Series) -> pd.Series:
    """Compute diminishing-returns sample size weights.

    ``w_sample = log(max(sample_size, 0) + 1)``.

    Args:
        sample_sizes: Series of sample sizes (may contain ``NaN`` or negative).

    Returns:
        Series of positive sample-size weights.

    Examples:
        >>> import pandas as pd
        >>> sw = sample_size_weight(pd.Series([100, 1000]))
        >>> round(float(sw.iloc[0]), 2)
        4.62
        >>> round(float(sw.iloc[1]), 2)
        6.91

    """
    safe = sample_sizes.fillna(0).clip(lower=0)
    arr = safe.to_numpy(dtype=float)
    return pd.Series(np.log(arr + 1.0), index=safe.index)


def pollster_weight_map(
    pollsters: pd.Series,
    ratings: dict[str, float],
) -> pd.Series:
    """Map pollster names to rating-based weights.

    Known pollsters are mapped via ``pollster_weight_formula``.
    Unrated pollsters receive the median weight of all rated pollsters.

    Args:
        pollsters: Series of pollster names.
        ratings: Mapping of pollster name to rating (0-10 scale).

    Returns:
        Series of pollster weights in ``[0.8, 1.0]``.

    Raises:
        ValueError: If ``ratings`` is empty.

    Examples:
        >>> import pandas as pd
        >>> s = pd.Series(["Invamer", "Mosqueteros"])
        >>> r = {"Invamer": 10.0, "Mosqueteros": 1.0}
        >>> pm = pollster_weight_map(s, r)
        >>> float(pm.iloc[0])
        1.0
        >>> float(pm.iloc[1])
        0.82

    """
    if not ratings:
        msg = "ratings dict cannot be empty"
        raise ValueError(msg)

    weight_map = {k: pollster_weight_formula(v) for k, v in ratings.items()}
    median_rating = statistics.median(ratings.values())
    default_weight = pollster_weight_formula(median_rating)

    return pollsters.map(lambda x: weight_map.get(x, default_weight))


def combined_weight(
    df: pd.DataFrame,
    election_date: date,
    ratings: dict[str, float],
    half_life: float = _DEFAULT_HALF_LIFE,
) -> pd.Series:
    """Compute globally normalized combined weights for each poll.

    ``w = w_time * w_sample * w_pollster``, then normalized so ``sum(w) = 1``.

    All candidates in the same poll share the same weight. Sample sizes use
    ``muestra_int_voto``, falling back to ``muestra`` when unavailable.

    Args:
        df: DataFrame with columns ``fecha``, ``encuestadora``,
            ``muestra_int_voto`` (or ``muestra``).
        election_date: Reference date for time decay.
        ratings: Pollster ratings mapping.
        half_life: Half-life for time decay.

    Returns:
        Series of normalized weights summing to 1.0.

    Examples:
        >>> import pandas as pd
        >>> from datetime import date
        >>> df = pd.DataFrame(
        ...     {"fecha": pd.to_datetime([date(2022, 5, 29), date(2022, 4, 29)]),
        ...      "encuestadora": ["Invamer", "Invamer"],
        ...      "muestra_int_voto": pd.array([1000, 1000], dtype="Int64")})
        >>> cw = combined_weight(df, date(2022, 5, 29), {"Invamer": 10.0})
        >>> cw.sum()
        np.float64(1.0)

    """
    w_time = time_weight(df["fecha"], election_date, half_life)

    # Prefer muestra_int_voto (voting-intent sample) over muestra (total sample),
    # since voting-intent sample better reflects the relevant population.
    # Fall back to 0 if neither column exists to avoid errors.
    if "muestra_int_voto" in df.columns:
        sample_sizes = df["muestra_int_voto"].fillna(
            df["muestra"] if "muestra" in df.columns else 0
        )
    else:
        sample_sizes = df.get("muestra", pd.Series(0, index=df.index))
    w_sample = sample_size_weight(sample_sizes)

    w_pollster = pollster_weight_map(df["encuestadora"], ratings)

    raw = w_time * w_sample * w_pollster
    total = raw.sum()
    if total > 0:
        return raw / total
    return pd.Series(0.0, index=df.index)


def weighted_average(
    df: pd.DataFrame,
    candidates: list[str],
    election_date: date,
    ratings: dict[str, float],
) -> dict[str, float]:
    """Compute weighted mean vote share for each candidate.

    Uses ``combined_weight`` to determine per-poll weights.

    Args:
        df: DataFrame with poll data and candidate columns.
        candidates: List of candidate keys to compute averages for.
        election_date: Reference date for time decay.
        ratings: Pollster ratings mapping.

    Returns:
        Mapping of candidate key to weighted mean vote share (``%``).

    Examples:
        >>> import pandas as pd
        >>> from datetime import date
        >>> df = pd.DataFrame(
        ...     {"fecha": pd.to_datetime([date(2022, 5, 29), date(2022, 4, 29)]),
        ...      "encuestadora": ["Invamer", "Invamer"],
        ...      "muestra_int_voto": pd.array([1000, 1000], dtype="Int64"),
        ...      "candidate_a": [50.0, 30.0]})
        >>> wa = weighted_average(df, ["candidate_a"], date(2022, 5, 29), {"Invamer": 10.0})
        >>> round(wa["candidate_a"], 2)
        43.33

    """
    weights = combined_weight(df, election_date, ratings)
    result: dict[str, float] = {}
    for candidate in candidates:
        if candidate in df.columns:
            values = df[candidate].astype(float)
            mask = ~values.isna()
            if mask.any():
                filtered_vals = values[mask]
                filtered_wts = weights[mask]
                result[candidate] = float(np.average(filtered_vals, weights=filtered_wts))
            else:
                result[candidate] = float("nan")
        else:
            result[candidate] = float("nan")
    return result


def aggregation_snapshot(
    df: pd.DataFrame,
    candidates: list[str],
    as_of_date: date,
    ratings: dict[str, float],
) -> dict[str, float]:
    """Compute weighted average as of a specific date.

    Only polls with ``fecha <= as_of_date`` are included. Time decay is
    relative to ``as_of_date`` itself, simulating a forecast from that date.

    Args:
        df: DataFrame with poll data.
        candidates: List of candidate keys.
        as_of_date: Cut-off date for poll inclusion; also the time anchor.
        ratings: Pollster ratings mapping.

    Returns:
        Mapping of candidate key to weighted mean vote share (``%``), or
        ``nan`` if no polls are available before ``as_of_date``.

    Examples:
        >>> import pandas as pd
        >>> from datetime import date
        >>> df = pd.DataFrame(
        ...     {"fecha": pd.to_datetime([date(2022, 5, 29)]),
        ...      "encuestadora": ["Invamer"],
        ...      "muestra_int_voto": pd.array([1000], dtype="Int64"),
        ...      "gustavo_petro": [42.0]})
        >>> snap = aggregation_snapshot(df, ["gustavo_petro"], date(2022, 6, 1), {"Invamer": 10.0})
        >>> round(snap["gustavo_petro"], 2)
        42.0

    """
    safe_dates = pd.to_datetime(df["fecha"], errors="coerce")
    mask = safe_dates.notna() & (safe_dates <= pd.Timestamp(as_of_date))
    filtered = df[mask].copy()
    if len(filtered) == 0:
        return {c: float("nan") for c in candidates}
    return weighted_average(filtered, candidates, as_of_date, ratings)


def evolution_series(  # noqa: PLR0913
    df: pd.DataFrame,
    candidates: list[str],
    election_date: date,
    ratings: dict[str, float],
    n_snapshots: int = 15,
    trends_df: pd.DataFrame | None = None,
    trends_query_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Compute weighted averages across a series of snapshot dates.

    Produces ``n_snapshots`` evenly spaced dates from ``CONSULTATION_DATE``
    to ``election_date - 2 days``, runs ``aggregation_snapshot`` at each,
    and returns a long-format DataFrame.

    When ``trends_df`` is provided, merges Google Trends favorable
    propensity as an additional ``trends`` column (6th source per P27).

    Args:
        df: DataFrame with poll data.
        candidates: List of candidate keys.
        election_date: Final election date.
        ratings: Pollster ratings mapping.
        n_snapshots: Number of evenly spaced snapshot dates.
        trends_df: Optional DataFrame indexed by date with query-string
            columns (output of ``fetch_trends``).
        trends_query_map: Optional mapping of candidate key to search query
            string.  Defaults to ``CANDIDATE_QUERY_MAP_2026`` when ``None``.

    Returns:
        DataFrame with columns ``as_of_date``, ``candidate``,
        ``weighted_average``, and optionally ``trends``.

    Examples:
        >>> import pandas as pd
        >>> from datetime import date
        >>> df = pd.DataFrame(
        ...     {"fecha": pd.to_datetime([date(2022, 5, 29)]),
        ...      "encuestadora": ["Invamer"],
        ...      "muestra_int_voto": pd.array([1000], dtype="Int64"),
        ...      "a": [50.0]})
        >>> ev = evolution_series(df, ["a"], date(2022, 5, 29), {"Invamer": 10.0}, n_snapshots=3)
        >>> list(ev.columns)
        ['as_of_date', 'candidate', 'weighted_average']
        >>> len(ev)
        3

    """
    last_snapshot = election_date - timedelta(days=2)
    snapshot_dates = pd.date_range(
        start=CONSULTATION_DATE,
        end=last_snapshot,
        periods=n_snapshots,
    )
    rows: list[dict[str, object]] = []
    for d in snapshot_dates:
        snap_date = d.date()
        snap = aggregation_snapshot(df, candidates, snap_date, ratings)
        rows.extend(
            {
                "as_of_date": snap_date,
                "candidate": c,
                "weighted_average": snap.get(c, float("nan")),
            }
            for c in candidates
        )
    result = pd.DataFrame(rows)

    if trends_df is not None and not trends_df.empty:
        query_map = trends_query_map if trends_query_map is not None else CANDIDATE_QUERY_MAP_2026
        active_map = {k: v for k, v in query_map.items() if k in candidates}
        if active_map:
            prop_fav_long = compute_prop_fav(trends_df, active_map)
            if not prop_fav_long.empty:
                result["as_of_date"] = pd.to_datetime(result["as_of_date"])
                prop_fav_long["as_of_date"] = pd.to_datetime(prop_fav_long["as_of_date"])
                result = result.sort_values(["as_of_date"], ignore_index=True)
                prop_fav_long = prop_fav_long.sort_values(["as_of_date"], ignore_index=True)
                # Merge each candidate independently to avoid global
                # monotonic-sort requirement when by groups interleave.
                merged_parts: list[pd.DataFrame] = []
                for cand in candidates:
                    left_part = result[result["candidate"] == cand].reset_index(drop=True)
                    right_part = prop_fav_long[prop_fav_long["candidate"] == cand].reset_index(
                        drop=True
                    )
                    if not right_part.empty:
                        right_part = right_part.drop(columns=["candidate"])
                        merged = pd.merge_asof(
                            left_part,
                            right_part,
                            on="as_of_date",
                            direction="backward",
                        )
                    else:
                        left_part["prop_fav"] = float("nan")
                        merged = left_part
                    merged_parts.append(merged)
                result = pd.concat(merged_parts, ignore_index=True)
                result = result.rename(columns={"prop_fav": "trends"})
            else:
                result["trends"] = float("nan")
        else:
            result["trends"] = float("nan")

    return result
