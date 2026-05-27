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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

import numpy as np
import pandas as pd

from co_president.config import (
    CONSULTATION_DATE,
    pollster_weight_formula,
)

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

    """
    days_diff = (pd.Timestamp(election_date) - pd.to_datetime(dates)).dt.days.abs()
    return 0.5 ** (days_diff / half_life)


def sample_size_weight(sample_sizes: pd.Series) -> pd.Series:
    """Compute diminishing-returns sample size weights.

    ``w_sample = log(max(sample_size, 0) + 1)``.

    Args:
        sample_sizes: Series of sample sizes (may contain ``NaN`` or negative).

    Returns:
        Series of positive sample-size weights.

    """
    safe = sample_sizes.fillna(0).clip(lower=0)
    arr = safe.to_numpy(dtype=float, na_value=0.0)
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

    """
    w_time = time_weight(df["fecha"], election_date, half_life)

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

    """
    weights = combined_weight(df, election_date, ratings)
    total_weight = weights.sum()
    result: dict[str, float] = {}
    for candidate in candidates:
        if candidate in df.columns:
            values = df[candidate].astype(float)
            if total_weight > 0 and not values.isna().all():
                result[candidate] = float(np.average(values, weights=weights))
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

    """
    mask = df["fecha"] <= pd.Timestamp(as_of_date)
    filtered = df[mask].copy()
    if len(filtered) == 0:
        return {c: float("nan") for c in candidates}
    return weighted_average(filtered, candidates, as_of_date, ratings)


def evolution_series(
    df: pd.DataFrame,
    candidates: list[str],
    election_date: date,
    ratings: dict[str, float],
    n_snapshots: int = 15,
) -> pd.DataFrame:
    """Compute weighted averages across a series of snapshot dates.

    Produces ``n_snapshots`` evenly spaced dates from ``CONSULTATION_DATE``
    to ``election_date - 2 days``, runs ``aggregation_snapshot`` at each,
    and returns a long-format DataFrame.

    Args:
        df: DataFrame with poll data.
        candidates: List of candidate keys.
        election_date: Final election date.
        ratings: Pollster ratings mapping.
        n_snapshots: Number of evenly spaced snapshot dates.

    Returns:
        DataFrame with columns ``as_of_date``, ``candidate``,
        ``weighted_average``.

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
    return pd.DataFrame(rows)
