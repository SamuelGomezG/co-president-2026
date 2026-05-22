"""SPEC-04+09: Data quality diagnostics for the poll pipeline.

Functions here are diagnostic/validation only. They do not modify the data
pipeline, the configuration, or the model. They surface insights about data
quality for human review.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import pandas as pd

from co_president.config import (
    CONSULTATION_DATE,
    ELECTION_DATE_ROUND1,
    POLLSTER_RATINGS,
    ModelConfig,
)

if TYPE_CHECKING:
    from co_president.data_results import RoundResult

logger = logging.getLogger(__name__)

_MAE_CANDIDATES: tuple[str, ...] = (
    "gustavo_petro",
    "federico_gutierrez",
    "rodolfo_hernandez",
    "sergio_fajardo",
    "ingrid_betancourt",
)
"""Five named individual candidates used for MAE computation across diagnostics."""

_FLAG_DEVIATION_THRESHOLD: float = 2.0
"""Threshold for flagging pollster rating deviation."""

_FLAG_METHODOLOGY_THRESHOLD_PP: float = 3.0
"""Threshold (percentage points) for flagging methodology differences."""

_MAX_MAE_SCALE_PP: float = 50.0
"""Absolute MAE scale for empirical scores (percentage points)."""

_MIN_POLLS_FOR_REGRESSION: int = 5
"""Minimum number of polls required for time-decay regression."""

_MIN_METHODS_FOR_DIFF: int = 2
"""Minimum methodology groups required for diff comparisons."""

_HALF_LIFE_WARN_THRESHOLD_DAYS: float = 15.0
"""If optimal half-life differs from current by more than this, log a warning."""

_LATE_STAGE_START: pd.Timestamp = pd.Timestamp("2022-05-01")
"""Start date for late-stage methodology comparisons (May 1st, 2022)."""


class TimeDecayResult(TypedDict):
    """Return schema for validate_time_decay."""

    optimal_half_life_days: float
    current_half_life_days: float
    r_squared: float
    recommendation: str


class PollsterRatingRow(TypedDict):
    """Row schema for pollster rating comparisons."""

    pollster: str
    la_silla_rating: float
    empirical_mae: float
    empirical_score: float
    deviation: float


class MethodologyCandidateStats(TypedDict):
    """Per-candidate methodology diagnostics based on MAE."""

    methodology_mae: dict[str, float]
    max_diff_pp: float
    flagged: bool


class MethodologyEffectResult(TypedDict):
    """Return schema for quantify_methodology_effect."""

    per_candidate: dict[str, MethodologyCandidateStats]
    methodology_counts: dict[str, int]
    flag_threshold_pp: float
    any_flagged: bool


def _compute_mae(
    row: pd.Series[float],
    results: RoundResult,
) -> float:
    """Compute MAE for a single poll row against actual RoundResult.

    Args:
        row: A single row from a polls DataFrame containing candidate share columns.
        results: RoundResult for the corresponding election round.

    Returns:
        Mean absolute error across the five named candidates (percentage points).

    """
    errors: list[float] = []
    for key in _MAE_CANDIDATES:
        predicted = row.get(key)
        if predicted is None or pd.isna(predicted):
            continue
        actual_pct = results.get_share(key) * 100
        errors.append(abs(float(predicted) - actual_pct))
    if not errors:
        return 0.0
    return float(np.mean(errors))


def validate_pollster_ratings(
    polls: pd.DataFrame,
    results: RoundResult,
) -> pd.DataFrame:
    """Compare empirical pollster accuracy against La Silla Vacía ratings.

    Takes the final pre-election poll per pollster (date <= election day - 1),
    computes the MAE against the actual RoundResult across active candidates,
    normalises to a 0-10 scale, and returns a comparison DataFrame.

    Args:
        polls: DataFrame of cleaned R1 polls with ``encuestadora``, ``fecha``,
            and candidate share columns.
        results: RoundResult for round 1.

    Returns:
        DataFrame with columns:
            pollster
            la_silla_rating
            empirical_mae
            empirical_score
            deviation

    """
    # Exclude post-election tracking polls with different assumptions.
    cutoff = pd.Timestamp(ELECTION_DATE_ROUND1) - pd.Timedelta(days=1)
    pre_election = polls[polls["fecha"] <= cutoff].copy()

    last_polls = pre_election.sort_values("fecha").groupby("encuestadora", sort=False).last()

    # Compute MAE per pollster (iterate rows to satisfy pyright)
    mae_values: list[float] = []
    pollster_names: list[str] = []
    for pollster_name, row in last_polls.iterrows():
        mae_values.append(_compute_mae(row, results))
        pollster_names.append(str(pollster_name))
    mae_series = pd.Series(mae_values, index=pollster_names)

    # Normalise to a fixed 0-10 scale for cross-sample comparability.
    if _MAX_MAE_SCALE_PP > 0:
        empirical_scores = (10.0 * (1.0 - mae_series / _MAX_MAE_SCALE_PP)).clip(0, 10)
    else:
        empirical_scores = pd.Series(10.0, index=mae_series.index)

    # Build result DataFrame
    median_rating = float(np.median(list(POLLSTER_RATINGS.values())))
    rows: list[PollsterRatingRow] = []
    mae_array = mae_series.to_numpy(dtype=float)
    empirical_array = empirical_scores.to_numpy(dtype=float)
    for idx, pollster in enumerate(mae_series.index):
        pollster_str = str(pollster)
        rating = POLLSTER_RATINGS.get(pollster_str, median_rating)
        mae_val = float(mae_array[idx])
        empirical_score_val = float(empirical_array[idx])
        rows.append(
            PollsterRatingRow(
                pollster=pollster_str,
                la_silla_rating=rating,
                empirical_mae=mae_val,
                empirical_score=empirical_score_val,
                deviation=empirical_score_val - rating,
            ),
        )

    result = pd.DataFrame(rows)

    # Flag large deviations
    large_dev = result[result["deviation"].abs() > _FLAG_DEVIATION_THRESHOLD]
    for _, row in large_dev.iterrows():
        logger.warning(
            "Pollster %s: deviation=%.2f (empirical=%.2f, rating=%.2f)",
            row["pollster"],
            row["deviation"],
            row["empirical_score"],
            row["la_silla_rating"],
        )

    return result


def validate_time_decay(
    polls: pd.DataFrame,
    results: RoundResult,
) -> TimeDecayResult:
    """Validate the optimal time decay half-life against 2022 poll accuracy.

    For each poll, computes the MAE against the actual RoundResult. Fits a
    log-linear regression: log(MAE) ~ days_before_election. Uses the
    regression coefficient to estimate the optimal half-life.

    Args:
        polls: DataFrame of cleaned R1 polls.
        results: RoundResult for round 1.

    Returns:
        dict with keys:
            optimal_half_life_days
            current_half_life_days
            r_squared
            recommendation

    """
    election_date_ts = pd.Timestamp(ELECTION_DATE_ROUND1)
    current_half_life_days = ModelConfig().time_decay_half_life_days

    # Compute per-poll MAE and days before election
    mae_list: list[float] = []
    days_list: list[float] = []
    for _, row in polls.iterrows():
        mae = _compute_mae(row, results)
        days = float((election_date_ts - row["fecha"]).days)
        if days < 0:
            continue
        mae_list.append(mae)
        days_list.append(days)

    n = len(mae_list)
    if n < _MIN_POLLS_FOR_REGRESSION:
        return {
            "optimal_half_life_days": -1.0,
            "current_half_life_days": current_half_life_days,
            "r_squared": 0.0,
            "recommendation": "insufficient_data",
        }

    # Log-linear fit: log(MAE) ~ days
    x = np.array(days_list, dtype=float)
    eps = 1e-12
    y = np.log(np.array(mae_list) + eps)

    coeffs = np.polyfit(x, y, 1)
    beta = coeffs[0]

    if beta <= 0:
        return {
            "optimal_half_life_days": -1.0,
            "current_half_life_days": current_half_life_days,
            "r_squared": 0.0,
            "recommendation": "relationship_not_detected",
        }

    optimal_half_life = float(np.log(2.0) / beta)

    # R-squared
    y_pred = np.polyval(coeffs, x)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    diff = abs(optimal_half_life - current_half_life_days)
    if diff > _HALF_LIFE_WARN_THRESHOLD_DAYS:
        recommendation = "update_ModelConfig"
        logger.warning(
            "Optimal half-life (%.1f days) differs from current (%.1f) by >15 days",
            optimal_half_life,
            current_half_life_days,
        )
    else:
        recommendation = "no_change_needed"

    return {
        "optimal_half_life_days": optimal_half_life,
        "current_half_life_days": current_half_life_days,
        "r_squared": r_squared,
        "recommendation": recommendation,
    }


def quantify_methodology_effect(
    polls: pd.DataFrame,
    results: RoundResult,
) -> MethodologyEffectResult:
    """Quantify the effect of survey methodology on poll accuracy.

    Groups post-consultation R1 polls by ``tipo``, computes mean candidate
    MAE per methodology against actual results, and flags large differences
    across methodologies.

    Args:
        polls: DataFrame of cleaned R1 polls with a ``tipo`` column.
        results: RoundResult for round 1.

    Returns:
        dict with keys:
            per_candidate: per-candidate MAE by methodology and max diff
            methodology_counts: count of polls per methodology
            flag_threshold_pp: threshold for flagging (default 3.0)
            any_flagged: whether any candidate exceeded the threshold

    """
    consultation_ts = pd.Timestamp(CONSULTATION_DATE)
    post_consulta = polls[polls["fecha"] >= consultation_ts].copy()

    # Methodology counts across all post-consultation polls
    methodology_counts: dict[str, int] = {}
    known_tipo = post_consulta.dropna(subset=["tipo"])
    for tipo_val in known_tipo["tipo"].unique():
        count = int((known_tipo["tipo"] == tipo_val).sum())
        methodology_counts[str(tipo_val)] = count

    # Late-stage comparison: May 1-28 only
    late_start = _LATE_STAGE_START
    late_end = pd.Timestamp(ELECTION_DATE_ROUND1) - pd.Timedelta(days=1)
    late_mask = (known_tipo["fecha"] >= late_start) & (known_tipo["fecha"] <= late_end)
    late_polls = known_tipo[late_mask]

    methodologies = sorted(methodology_counts)
    per_candidate: dict[str, MethodologyCandidateStats] = {}
    any_flagged = False
    for key in _MAE_CANDIDATES:
        actual_pct = results.get_share(key) * 100
        methodology_mae: dict[str, float] = {}
        for method in methodologies:
            subset = late_polls[late_polls["tipo"] == method]
            if subset.empty:
                methodology_mae[method] = float("nan")
                continue
            abs_errors = (subset[key] - actual_pct).abs()
            methodology_mae[method] = float(abs_errors.mean())
        valid_values = [val for val in methodology_mae.values() if not np.isnan(val)]
        if len(valid_values) < _MIN_METHODS_FOR_DIFF:
            max_diff = float("nan")
            flagged = False
        else:
            max_diff = float(np.max(valid_values) - np.min(valid_values))
            flagged = max_diff > _FLAG_METHODOLOGY_THRESHOLD_PP
        if flagged:
            any_flagged = True
        per_candidate[key] = {
            "methodology_mae": methodology_mae,
            "max_diff_pp": max_diff,
            "flagged": flagged,
        }

    return {
        "per_candidate": per_candidate,
        "methodology_counts": methodology_counts,
        "flag_threshold_pp": _FLAG_METHODOLOGY_THRESHOLD_PP,
        "any_flagged": any_flagged,
    }
