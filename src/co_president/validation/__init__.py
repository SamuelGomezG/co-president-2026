"""SPEC-09: Validation & Backtesting.

Systematically compare model predictions against actual 2022 results using
multiple metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import logging
import math
import numbers
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from co_president.config import ModelConfig
    from co_president.data import CleanPolls, RoundResult
    from co_president.model_round1 import Round1Forecast
    from co_president.model_runoff_simple import RunoffForecast

from co_president.config import (
    CONSULTATION_DATE,
    ELECTION_DATE_ROUND1,
    FIRST_ROUND_CANDIDATES,
)

logger = logging.getLogger(__name__)

# Threshold for filtering polls with high undecided/no-response rates.
_NS_NR_FILTER = 10.0


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
        ci_95_lower: Lower bound of 95% credible interval.
        ci_95_upper: Upper bound of 95% credible interval.
        within_95ci: Whether actual_share falls inside 95% credible interval.
        within_50ci: Whether actual_share falls inside 50% credible interval.

    """

    candidate_key: str
    actual_share: float
    predicted_mean: float
    predicted_median: float
    error: float
    abs_error: float
    ci_95_lower: float
    ci_95_upper: float
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
                ci_95_lower=fc.ci_95[0],
                ci_95_upper=fc.ci_95[1],
                within_95ci=within_95ci,
                within_50ci=within_50ci,
            )
        )
    errors = np.array([cv.error for cv in candidates])
    mae = float(np.abs(errors).mean())
    rmse = float(np.sqrt(np.mean(errors**2)))
    cal_95, cal_50 = _compute_calibration(candidates)
    return RoundValidation(
        round_number=1,
        candidates=candidates,
        mae=mae,
        rmse=rmse,
        calibration_95=cal_95,
        calibration_50=cal_50,
    )


def _compute_calibration(
    candidates: list[CandidateValidation],
) -> tuple[float, float]:
    """Compute calibration fractions (within 95% CI, within 50% CI).

    Args:
        candidates: List of per-candidate validation results.

    Returns:
        Tuple of ``(calibration_95, calibration_50)`` in [0, 1].

    """
    n = len(candidates)
    if n == 0:
        return (0.0, 0.0)
    cal_95 = sum(1 for cv in candidates if cv.within_95ci) / n
    cal_50 = sum(1 for cv in candidates if cv.within_50ci) / n
    return (cal_95, cal_50)


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
    median_shares = [forecast.median_share_a, forecast.median_share_b]
    ci_95_intervals = [forecast.ci_95_a, forecast.ci_95_b]
    ci_50_intervals = [forecast.ci_50_a, forecast.ci_50_b]
    candidates: list[CandidateValidation] = []
    for key, mean_share, median_share, ci_95, ci_50 in zip(
        candidate_keys, mean_shares, median_shares, ci_95_intervals, ci_50_intervals, strict=True
    ):
        actual_share = results.get_share(key)
        within_95ci = ci_95[0] <= actual_share <= ci_95[1]
        within_50ci = ci_50[0] <= actual_share <= ci_50[1]
        error = mean_share - actual_share
        candidates.append(
            CandidateValidation(
                candidate_key=key,
                actual_share=actual_share,
                predicted_mean=mean_share,
                predicted_median=median_share,
                error=error,
                abs_error=abs(error),
                ci_95_lower=ci_95[0],
                ci_95_upper=ci_95[1],
                within_95ci=within_95ci,
                within_50ci=within_50ci,
            )
        )

    errors = np.array([cv.error for cv in candidates])
    mae = float(np.abs(errors).mean())
    rmse = float(np.sqrt(np.mean(errors**2)))
    cal_95, cal_50 = _compute_calibration(candidates)
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
    results: tuple[RoundResult, RoundResult],  # noqa: ARG001
    config: ModelConfig,
    n_snapshots: int = 10,
    min_polls: int = 3,
) -> list[tuple[date, Round1Forecast]]:
    """Re-fit the Round 1 model using polls available up to each cutoff date.

    Generates ``n_snapshots`` evenly spaced cutoff dates from 30 days after
    the consultation date to 2 days before round 1. For each cutoff, polls
    with ``fecha <= cutoff`` are used to build, sample, and forecast the
    Round 1 Bayesian model.

    Snapshots with fewer than ``min_polls`` usable polls (non-NaN candidate
    shares and sample sizes) after filtering are skipped.  MCMC failures are
    logged and skipped individually — one failing snapshot does not abort the
    whole series.

    Args:
        polls: CleanPolls container.
        results: Canonical election results (unused in forecast-only mode).
        config: ModelConfig with hyperparameters.
        n_snapshots: Number of evenly spaced cutoff dates.
        min_polls: Minimum number of polls (with non-NaN shares/size) required
            to fit a snapshot.

    Returns:
        List of ``(cutoff_date, forecast)`` tuples. Empty if not enough data.

    Examples:
        Minimal usage (requires pre-loaded poll data)::

            >>> from co_president.config import ModelConfig
            >>> from co_president.data import load_and_clean_all
            >>> polls = load_and_clean_all()  # doctest: +SKIP
            >>> config = ModelConfig(mcmc_draws=500, mcmc_tune=500)
            >>> snapshots = rolling_forecast(polls, ..., config)  # doctest: +SKIP

    """
    # Start 30 days post-consultation so consultation results have settled and
    # enough polls have accumulated for a stable baseline.  End 2 days before
    # the election because most pollsters stop fielding work close to election
    # day (logistical cutoff), and final pre-election volatility is excluded.
    start_date = CONSULTATION_DATE + timedelta(days=30)
    end_date = ELECTION_DATE_ROUND1 - timedelta(days=2)
    cutoff_dates = pd.date_range(
        start=start_date, end=end_date, periods=n_snapshots
    ).to_pydatetime()

    candidate_keys = sorted(set(FIRST_ROUND_CANDIDATES) & set(polls.round1.columns))
    if not candidate_keys:
        logger.warning("rolling_forecast: no candidate columns found in round1 polls")
        return []

    snapshots: list[tuple[date, Round1Forecast]] = []
    for cutoff in cutoff_dates:  # to_pydatetime() -> datetime.datetime
        cutoff_date = cutoff.date()
        snapshot_df = polls.round1.loc[polls.round1["fecha"] <= pd.Timestamp(cutoff_date)].copy()

        # Drop rows with NaN in any candidate-share or sample-size column.
        # This mirrors build_round1_model's internal nan_mask so that
        # min_polls reflects the number of truly usable polls.
        nan_mask = snapshot_df[[*candidate_keys, "muestra"]].isna().any(axis=1)
        if nan_mask.any():
            snapshot_df = snapshot_df.loc[~nan_mask].copy()

        if len(snapshot_df) < min_polls:
            logger.info(
                "rolling_forecast: skipping %s — only %d usable polls",
                cutoff_date,
                len(snapshot_df),
            )
            continue

        try:
            from co_president.model_round1 import (  # noqa: PLC0415
                build_round1_model,
                forecast_round1,
                sample_round1,
            )

            model = build_round1_model(snapshot_df, results=None, config=config)
            idata = sample_round1(model, config)
            forecast = forecast_round1(idata, candidate_keys)
            snapshots.append((cutoff_date, forecast))
        except (ValueError, RuntimeError, TypeError, AttributeError):
            logger.exception(
                "rolling_forecast: MCMC failed for snapshot %s, skipping",
                cutoff_date,
                extra={"cutoff_date": str(cutoff_date)},
            )

    return snapshots


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


@dataclass(frozen=True)
class SensitivityFlag:
    """Result of a single candidate's ns_nr sensitivity check.

    Attributes:
        candidate_key: Candidate identifier key.
        baseline_mean: Posterior mean vote share from the full data.
        sensitive_mean: Posterior mean vote share after excluding high-ns_nr polls.
        shift: ``sensitive_mean - baseline_mean`` (positive means higher with
            high-ns_nr polls removed).
        exceeds_threshold: Whether ``abs(shift) > threshold``.

    """

    candidate_key: str
    baseline_mean: float
    sensitive_mean: float
    shift: float
    exceeds_threshold: bool


@dataclass(frozen=True)
class SensitivityResult:
    """Aggregate ns_nr sensitivity analysis results.

    Attributes:
        flags: Per-candidate sensitivity flags.
        total_candidates_flagged: Number of candidates whose shift exceeds
            the threshold.

    """

    flags: list[SensitivityFlag]
    total_candidates_flagged: int

    @property
    def has_failures(self) -> bool:
        """True when at least one candidate exceeds the shift threshold."""
        return self.total_candidates_flagged > 0


def sensitivity_ns_nr(
    baseline_forecast: Round1Forecast,
    polls: CleanPolls,
    results: RoundResult | None,
    config: ModelConfig,
    threshold: float = 0.02,
) -> SensitivityResult:
    """Compare baseline vs forecast excluding polls with high ns_nr.

    Filters out first-round polls where ``ns_nr > 10%``, re-fits the model,
    and flags any candidate whose posterior mean shifts by more than
    *threshold*.

    Args:
        baseline_forecast: Forecast from the full poll dataset.
        polls: CleanPolls containing all poll data.
        results: Optional election results for the model (pass ``None`` for
            forecast-only mode).
        config: ModelConfig with hyperparameters.
        threshold: Maximum allowed absolute shift in vote share (decimal,
            default 0.02 = 2 percentage points, must be non-negative).

    Returns:
        SensitivityResult with per-candidate flags.

    Raises:
        ValueError: If *threshold* is negative, no polls remain after the
            ns_nr filter, or the filtered data is insufficient for model
            building.

    Examples:
        Forecast-only mode (model uses polls but not election results)::

            >>> from co_president.config import ModelConfig
            >>> from co_president.data import load_and_clean_all, load_canonical_results
            >>> from co_president.model_round1 import Round1Forecast, forecast_round1
            >>> baseline_forecast = Round1Forecast(...)  # doctest: +SKIP
            >>> polls = load_and_clean_all()
            >>> result = sensitivity_ns_nr(
            ...     baseline_forecast,
            ...     polls,
            ...     results=None,
            ...     config=ModelConfig(mcmc_draws=10, mcmc_tune=5),
            ...     threshold=0.02,
            ... )
            >>> result.has_failures
            False

        With election results for comparison::

            >>> results_r1, _ = load_canonical_results()
            >>> result = sensitivity_ns_nr(
            ...     baseline_forecast, polls,
            ...     results=results_r1,
            ...     config=ModelConfig(),
            ... )

    """
    if threshold < 0:
        msg = f"threshold must be non-negative, got {threshold}"
        raise ValueError(msg)

    # Filter out polls with ns_nr > 10%
    sensitive_polls = polls.round1.copy()
    if "ns_nr" in sensitive_polls.columns:
        ns_nr_mask = sensitive_polls["ns_nr"] <= _NS_NR_FILTER
        sensitive_polls = sensitive_polls.loc[ns_nr_mask].copy()
        dropped = int((~ns_nr_mask).sum())
        if dropped > 0:
            logger.info(
                "sensitivity_ns_nr: dropped %d poll(s) where ns_nr > %s%%",
                dropped,
                _NS_NR_FILTER,
            )

    if len(sensitive_polls) == 0:
        msg = (
            "sensitivity_ns_nr: no polls remain after filtering "
            f"out rows where ns_nr > {_NS_NR_FILTER}%"
        )
        raise ValueError(msg)

    # Extract candidate keys from the baseline forecast
    candidate_keys = [c.candidate_key for c in baseline_forecast.candidates]

    # Build and sample the sensitive model
    from co_president.model_round1 import (  # noqa: PLC0415
        build_round1_model,
        forecast_round1,
        sample_round1,
    )

    sensitive_model = build_round1_model(sensitive_polls, results, config)
    sensitive_idata = sample_round1(sensitive_model, config)
    sensitive_forecast = forecast_round1(sensitive_idata, candidate_keys)

    # Compare means
    baseline_map = {c.candidate_key: c.mean_share for c in baseline_forecast.candidates}
    flags: list[SensitivityFlag] = []
    for fc in sensitive_forecast.candidates:
        if fc.candidate_key not in baseline_map:
            logger.warning(
                "sensitivity_ns_nr: candidate %s not found in baseline forecast, skipping",
                fc.candidate_key,
            )
            continue
        baseline_mean = baseline_map[fc.candidate_key]
        shift = fc.mean_share - baseline_mean
        flags.append(
            SensitivityFlag(
                candidate_key=fc.candidate_key,
                baseline_mean=baseline_mean,
                sensitive_mean=fc.mean_share,
                shift=shift,
                exceeds_threshold=abs(shift) > threshold,
            )
        )

    total_flagged = sum(1 for f in flags if f.exceeds_threshold)
    return SensitivityResult(flags=flags, total_candidates_flagged=total_flagged)


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
            from co_president.model_round1 import Round1Forecast  # noqa: PLC0415

            forecast = Round1Forecast.from_json(f.read())
        # Extract date from filename: "snapshot_2022-05-15.json"
        stem = filepath.stem  # "snapshot_2022-05-15"
        date_str = stem.replace("snapshot_", "")
        cutoff_date = date.fromisoformat(date_str)  # type: ignore[misc]
        snapshots.append((cutoff_date, forecast))
    return snapshots


# ═══════════════════════════════════════════════════════════════════
# SPEC-09: Cross-Pollster Consistency
# ═══════════════════════════════════════════════════════════════════


_SHARE_COLS_EXCLUDED = frozenset(
    (
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
    )
)
"""Frozen set of metadata column names excluded from candidate-share detection."""

_COMPARISON_THRESHOLD_MULTIPLIER = 2.0
_MIN_POLLSTERS_FOR_COMPARISON = 2

_RESULT_COLS = [
    "date_group",
    "candidate",
    "pollster_a",
    "pollster_b",
    "max_diff",
    "moe_combined",
]


def _empty_validation_result() -> pd.DataFrame:
    """Return an empty DataFrame with the validation result schema."""
    return pd.DataFrame(columns=_RESULT_COLS)


def _compare_pollster_pair(  # noqa: PLR0913
    date_group: object,
    pollster_a: str,
    moe_a: float,
    pollster_b: str,
    moe_b: float,
    values: pd.DataFrame,
    i: int,
    j: int,
) -> list[dict[str, object]]:
    """Compare two pollsters' candidate shares and flag entries exceeding 2x combined MoE."""
    moe_combined = math.sqrt(moe_a**2 + moe_b**2)
    threshold = _COMPARISON_THRESHOLD_MULTIPLIER * moe_combined
    diffs = (values.loc[i] - values.loc[j]).abs()
    flagged: list[dict[str, object]] = []
    for candidate, raw in diffs.to_dict().items():
        if raw > threshold:
            flagged.append(
                {
                    "date_group": date_group,
                    "candidate": candidate,
                    "pollster_a": pollster_a,
                    "pollster_b": pollster_b,
                    "max_diff": float(raw),
                    "moe_combined": moe_combined,
                }
            )
    return flagged


def validate_cross_pollster_consistency(polls: pd.DataFrame) -> pd.DataFrame:  # noqa: C901, PLR0912
    """Check cross-pollster consistency within same-day windows.

    This diagnostic scans polls grouped by same-day buckets (using ``fecha.dt.floor("D")``),
    compares candidate shares between pollster pairs, and flags candidates
    whose pairwise differences exceed twice the combined margin of error.
    The input DataFrame is not modified.

    Candidate share columns are auto-detected by excluding known non-share
    columns (metadata, ``blanco``, ``otros``, ``date_group``,
    ``forced_choice``). Any row with NaN in any candidate-share column is
    skipped from all pairwise comparisons. Comparisons where both rows
    belong to the same pollster are also skipped.

    Args:
        polls: Poll DataFrame with ``fecha`` and ``encuestadora`` columns
            plus candidate share columns (auto-detected).

    Returns:
        DataFrame with columns: ``date_group``, ``candidate``, ``pollster_a``,
        ``pollster_b``, ``max_diff``, ``moe_combined``.

    Raises:
        ValueError: If required columns are missing.

    Examples:
        >>> import pandas as pd
        >>> polls = pd.DataFrame({
        ...     "encuestadora": ["A", "B"],
        ...     "fecha": pd.to_datetime(["2022-05-15", "2022-05-15"]),
        ...     "gustavo_petro": [40.0, 50.0],
        ...     "rodolfo_hernandez": [28.0, 18.0],
        ...     "margen_error": [2.0, 2.0],
        ... })
        >>> result = validate_cross_pollster_consistency(polls)
        >>> result.columns.to_list()
        ['date_group', 'candidate', 'pollster_a', 'pollster_b', 'max_diff', 'moe_combined']
        >>> result["candidate"].iloc[0]
        'gustavo_petro'

        NaN candidate shares are skipped — no comparisons produced:
        >>> polls_with_nan = pd.DataFrame({
        ...     "encuestadora": ["A", "B"],
        ...     "fecha": pd.to_datetime(["2022-05-15", "2022-05-15"]),
        ...     "gustavo_petro": [40.0, None],
        ...     "margen_error": [2.0, 2.0],
        ... })
        >>> result = validate_cross_pollster_consistency(polls_with_nan)
        >>> result.empty
        True

    """
    required_cols = {"fecha", "encuestadora"}
    missing = required_cols - set(polls.columns)
    if missing:
        msg = f"validate_cross_pollster_consistency: missing columns {sorted(missing)}"
        raise ValueError(msg)

    if len(polls) == 0:
        return _empty_validation_result()

    df = polls.copy()
    df = df.assign(date_group=df["fecha"].dt.floor("D"))  # type: ignore[reportUnknownArgumentType]  # pyright cannot infer pandas .dt.floor return type; validated upstream

    excluded = _SHARE_COLS_EXCLUDED | {"blanco", "otros", "date_group", "forced_choice"}
    candidate_cols = [col for col in df.columns if col not in excluded]
    if not candidate_cols:
        return _empty_validation_result()

    median_moe = 0.0
    if "margen_error" in df.columns:
        median_value = df["margen_error"].median()
        if not pd.isna(median_value):
            median_moe = float(median_value)

    def _resolve_moe(value: object) -> float:
        """Return a margin-of-error value or fall back to median_moe for non-numeric/NaN inputs."""
        if not isinstance(value, numbers.Real):
            return median_moe
        v = float(value)
        if math.isnan(v):
            return median_moe
        return v

    results: list[dict[str, object]] = []
    for date_group, g in df.groupby("date_group", sort=False):
        if g["encuestadora"].nunique() < _MIN_POLLSTERS_FOR_COMPARISON:
            continue

        grp = g.reset_index(drop=True)
        values = grp[candidate_cols].astype(float)
        start_len = len(results)
        for i in range(len(grp) - 1):
            pollster_a = str(grp.loc[i, "encuestadora"]).strip()
            moe_a = (
                _resolve_moe(grp.loc[i, "margen_error"]) if "margen_error" in grp else median_moe
            )
            for j in range(i + 1, len(grp)):
                pollster_b = str(grp.loc[j, "encuestadora"]).strip()
                if pollster_a == pollster_b:
                    continue
                if values.iloc[i].isna().any():
                    continue
                if values.iloc[j].isna().any():
                    continue
                moe_b = (
                    _resolve_moe(grp.loc[j, "margen_error"])
                    if "margen_error" in grp
                    else median_moe
                )
                results.extend(
                    _compare_pollster_pair(
                        date_group, pollster_a, moe_a, pollster_b, moe_b, values, i, j
                    )
                )

        flagged_count = len(results) - start_len
        if flagged_count:
            logger.warning(
                "validate_cross_pollster_consistency: flagged %d comparison(s) for %s",
                flagged_count,
                date_group,
            )

    return pd.DataFrame(results, columns=_RESULT_COLS)
