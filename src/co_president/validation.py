"""SPEC-09: Validation & Backtesting.

Systematically compare model predictions against actual 2022 results using
multiple metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from co_president.config import ModelConfig
    from co_president.data_polls import CleanPolls
    from co_president.data_results import RoundResult
    from co_president.model_runoff_simple import RunoffForecast

from co_president.model_round1 import Round1Forecast

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandidateValidation:
    """Validation result for a single candidate.

    Attributes:
        candidate_key: Candidate identifier key.
        actual_share: Actual vote share from official results.
        predicted_mean: Posterior mean vote share.
        predicted_median: Posterior median vote share.
        error: ``predicted_mean - actual_share``.
        abs_error: ``|error|``.
        within_95ci: Whether actual_share falls inside 95% credible interval.
        within_50ci: Whether actual_share falls inside 50% credible interval.

    """

    candidate_key: str
    actual_share: float
    predicted_mean: float
    predicted_median: float
    error: float
    abs_error: float
    within_95ci: bool
    within_50ci: bool


@dataclass(frozen=True)
class RoundValidation:
    """Aggregate validation results for one election round.

    Attributes:
        round_number: 1 for first round, 2 for runoff.
        candidates: Per-candidate validation entries.
        mae: Mean Absolute Error across candidates.
        rmse: Root Mean Square Error.
        calibration_95: Fraction of candidates whose actual is within 95% CI.
        calibration_50: Fraction of candidates whose actual is within 50% CI.

    """

    round_number: Literal[1, 2]
    candidates: list[CandidateValidation]
    mae: float
    rmse: float
    calibration_95: float
    calibration_50: float


def validate_round1(
    forecast: Round1Forecast,
    results: RoundResult,
) -> RoundValidation:
    """Validate first-round forecast against actual results.

    Args:
        forecast: Round 1 forecast from the Bayesian model.
        results: Actual election results (canonical RoundResult).

    Returns:
        RoundValidation with per-candidate metrics and aggregate scores.

    """
    candidates: list[CandidateValidation] = []
    for fc in forecast.candidates:
        actual_share = results.get_share(fc.candidate_key)
        error = fc.mean_share - actual_share
        abs_error = abs(error)
        within_95ci = fc.ci_95[0] <= actual_share <= fc.ci_95[1]
        within_50ci = fc.ci_50[0] <= actual_share <= fc.ci_50[1]
        candidates.append(
            CandidateValidation(
                candidate_key=fc.candidate_key,
                actual_share=actual_share,
                predicted_mean=fc.mean_share,
                predicted_median=fc.median_share,
                error=error,
                abs_error=abs_error,
                within_95ci=within_95ci,
                within_50ci=within_50ci,
            )
        )
    errors = np.array([cv.error for cv in candidates])
    mae = float(np.abs(errors).mean())
    rmse = float(np.sqrt(np.mean(errors**2)))
    n_candidates = len(candidates)
    cal_95 = (
        sum(1 for cv in candidates if cv.within_95ci) / n_candidates if n_candidates > 0 else 0.0
    )
    cal_50 = (
        sum(1 for cv in candidates if cv.within_50ci) / n_candidates if n_candidates > 0 else 0.0
    )
    return RoundValidation(
        round_number=1,
        candidates=candidates,
        mae=mae,
        rmse=rmse,
        calibration_95=cal_95,
        calibration_50=cal_50,
    )


def validate_runoff(
    forecast: RunoffForecast,
    results: RoundResult,
) -> RoundValidation:
    """Validate runoff forecast against actual results.

    Args:
        forecast: Runoff forecast from the Bayesian model.
        results: Actual election results (canonical RoundResult).

    Returns:
        RoundValidation with per-candidate metrics.

    """
    candidate_keys = [forecast.candidate_a_key, forecast.candidate_b_key]
    mean_shares = [forecast.mean_share_a, forecast.mean_share_b]
    ci_95_intervals = [forecast.ci_95_a, forecast.ci_95_b]
    candidates: list[CandidateValidation] = []
    for key, mean_share, ci_95 in zip(candidate_keys, mean_shares, ci_95_intervals, strict=True):
        actual_share = results.get_share(key)
        within_95ci = ci_95[0] <= actual_share <= ci_95[1]
        error = mean_share - actual_share
        candidates.append(
            CandidateValidation(
                candidate_key=key,
                actual_share=actual_share,
                predicted_mean=mean_share,
                predicted_median=mean_share,
                error=error,
                abs_error=abs(error),
                within_95ci=within_95ci,
                within_50ci=False,
            )
        )

    errors = np.array([cv.error for cv in candidates])
    mae = float(np.abs(errors).mean())
    rmse = float(np.sqrt(np.mean(errors**2)))
    n_candidates = len(candidates)
    cal_95 = (
        sum(1 for cv in candidates if cv.within_95ci) / n_candidates if n_candidates > 0 else 0.0
    )
    cal_50 = (
        sum(1 for cv in candidates if cv.within_50ci) / n_candidates if n_candidates > 0 else 0.0
    )
    return RoundValidation(
        round_number=2,
        candidates=candidates,
        mae=mae,
        rmse=rmse,
        calibration_95=cal_95,
        calibration_50=cal_50,
    )


def brier_score_round1(
    forecast: Round1Forecast,
    results: RoundResult,
) -> float:
    """Compute Brier score for ``prob_top_two`` predictions.

    Args:
        forecast: Round 1 forecast with ``prob_top_two`` per candidate.
        results: Actual election results (used to determine which candidates
            actually made the runoff).

    Returns:
        Brier score (0 = perfect, 1 = worst).

    """
    top_two = {c.candidate_key for c in results.top_two()}
    n = len(forecast.candidates)
    if n == 0:
        return 0.0
    total = sum(
        (fc.prob_top_two - (1.0 if fc.candidate_key in top_two else 0.0)) ** 2
        for fc in forecast.candidates
    )
    return total / n


def rolling_forecast(
    polls: CleanPolls,
    results: tuple[RoundResult, RoundResult],
    config: ModelConfig,
    n_snapshots: int = 10,
) -> list[tuple[date, Round1Forecast]]:
    """Re-fit the Round 1 model using polls available up to each cutoff date.

    .. note::
        This function requires MCMC sampling and is intentionally a no-op
        placeholder that returns an empty list. Full implementation requires
        SPEC-10 integration.

    Args:
        polls: CleanPolls container.
        results: Canonical election results.
        config: ModelConfig with hyperparameters.
        n_snapshots: Number of evenly spaced cutoff dates.

    Returns:
        List of (cutoff_date, forecast) tuples. Empty if not enough data.

    """
    _ = polls, results, config, n_snapshots
    return []


def compute_rolling_errors(
    rolling: list[tuple[date, Round1Forecast]],
    results: RoundResult,
) -> pd.DataFrame:
    """Compute MAE and RMSE for each snapshot in a rolling forecast series.

    Args:
        rolling: List of (cutoff_date, forecast) tuples.
        results: Actual election results.

    Returns:
        DataFrame with columns ``as_of_date``, ``mae``, ``rmse``.

    """
    rows: list[dict[str, date | float]] = []
    for cutoff_date, forecast in rolling:
        validation = validate_round1(forecast, results)
        rows.append(
            {
                "as_of_date": cutoff_date,
                "mae": validation.mae,
                "rmse": validation.rmse,
            }
        )
    return pd.DataFrame(rows)


def save_rolling_snapshot(
    snapshot: tuple[date, Round1Forecast],
    output_dir: str,
) -> None:
    """Save a single rolling forecast snapshot to disk as JSON.

    Args:
        snapshot: (cutoff_date, forecast) tuple.
        output_dir: Directory path for saving snapshots.

    """
    cutoff_date, forecast = snapshot
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / f"snapshot_{cutoff_date.isoformat()}.json"
    with filepath.open("w") as f:
        f.write(forecast.to_json())


def load_rolling_snapshots(
    output_dir: str,
) -> list[tuple[date, Round1Forecast]]:
    """Load all saved rolling forecast snapshots from disk.

    Args:
        output_dir: Directory containing snapshot JSON files.

    Returns:
        List of (cutoff_date, forecast) tuples, sorted by date.

    """
    path = Path(output_dir)
    if not path.is_dir():
        return []
    snapshots: list[tuple[date, Round1Forecast]] = []
    for filepath in sorted(path.glob("snapshot_*.json")):
        with filepath.open() as f:
            forecast = Round1Forecast.from_json(f.read())
        # Extract date from filename: "snapshot_2022-05-15.json"
        stem = filepath.stem  # "snapshot_2022-05-15"
        date_str = stem.replace("snapshot_", "")
        cutoff_date = date.fromisoformat(date_str)  # type: ignore[misc]
        snapshots.append((cutoff_date, forecast))
    return snapshots
