"""SPEC-26: Out-of-sample validation framework for the municipal hierarchical model.

Provides leave-one-year-out cross-validation, 2018-holdout, and
leave-2022-out sparse-poll deployment test.  The R-squared gate (0.3 threshold)
prevents overfitting to 2022 data by failing the build with a
"missing data" report listing insufficient components.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import arviz as az  # type: ignore[import-untyped]
import numpy as np
import pandas as pd

from co_president.config import FIRST_ROUND_CANDIDATES, POLLSTER_RATINGS
from co_president.model_municipal import (
    build_municipal_model,
    sample_municipal_model,
)

if TYPE_CHECKING:
    from co_president.config import ModelConfig
    from co_president.data import RoundResult

logger = logging.getLogger(__name__)

__all__ = [
    "compare_modes",
    "leave_2022_out",
    "leave_one_year_out",
    "sample_all_low_polls",
    "sample_bootstrap_polls",
    "sample_stratified_polls",
    "year_2018_holdout",
]

# Rating threshold below which a pollster is considered "low" quality.
# Based on POLLSTER_RATINGS distribution: Medilab (3.8), YanHaas (3.6),
# CifrasYConceptos (3.6), Datexco (3.23), Mosqueteros (1.0).
_LOW_RATING_CUTOFF = 4.0

# Gating R-squared threshold: build fails below this value.
_R2_GATE = 0.3

# Hard cap on the number of polls in the leave-2022-out deployment test.
_MAX_POLLS_2022 = 10

# Number of bootstrap runs for sensitivity checks.
_BOOTSTRAP_RUNS = 20

# MAE threshold for top-3 candidates in leave-2022-out (decimal, 0.05 = 5 pp).
_MAE_PP_THRESHOLD = 0.05

# Diff threshold for compare_modes (percentage points, decimal).
_MODE_DIFF_THRESHOLD = 0.01


# ═══════════════════════════════════════════════════════════════════════
# Poll sampling strategies
# ═══════════════════════════════════════════════════════════════════════


def _categorize_pollster(
    rating: float,
    high_cutoff: float = 8.0,
    low_cutoff: float = _LOW_RATING_CUTOFF,
) -> str:
    """Categorize a pollster rating as ``"high"``, ``"mid"``, or ``"low"``.

    Args:
        rating: Pollster rating (0-10 scale).
        high_cutoff: Minimum rating for "high" category.
        low_cutoff: Maximum rating for "low" category.

    Returns:
        Category string.

    """
    if rating >= high_cutoff:
        return "high"
    if rating <= low_cutoff:
        return "low"
    return "mid"


def sample_stratified_polls(
    polls: pd.DataFrame,
    n_total: int = _MAX_POLLS_2022,
    pollster_ratings: dict[str, float] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Sample polls using a stratified strategy.

    Selects 4 high-rated, 3 mid-rated, 3 low-rated pollsters by
    ``POLLSTER_RATINGS``.  Deterministic and reproducible.

    Args:
        polls: DataFrame with an ``encuestadora`` column.
        n_total: Total number of polls to sample (default 10).
        pollster_ratings: Mapping of pollster name to rating.  If
            ``None``, defaults to :data:`co_president.config.POLLSTER_RATINGS`.
        seed: Random seed for reproducibility.

    Returns:
        Sampled DataFrame with at most *n_total* rows.

    Raises:
        ValueError: If *pollster_ratings* is empty or *polls* has no
            ``encuestadora`` column.

    """
    _ratings: dict[str, float] = (
        dict(POLLSTER_RATINGS) if pollster_ratings is None else pollster_ratings
    )

    if not _ratings:
        msg = "pollster_ratings cannot be empty"
        raise ValueError(msg)

    if "encuestadora" not in polls.columns:
        msg = "polls must have an 'encuestadora' column"
        raise ValueError(msg)

    _rng = np.random.default_rng(seed)
    categorized: dict[str, list[str]] = {"high": [], "mid": [], "low": []}
    for pollster, rating in _ratings.items():
        cat = _categorize_pollster(rating)
        categorized[cat].append(pollster)

    tier_counts = {"high": 4, "mid": 3, "low": 3}

    selected_pollsters: list[str] = []
    for tier in ("high", "mid", "low"):
        available = categorized.get(tier, [])
        _rng.shuffle(available)
        selected_pollsters.extend(available[: tier_counts[tier]])

    sampled = polls.loc[polls["encuestadora"].isin(selected_pollsters), :].copy()
    if len(sampled) > n_total:
        sampled = sampled.sample(n=n_total, random_state=seed)
    return sampled


def sample_all_low_polls(
    polls: pd.DataFrame,
    n_total: int = _MAX_POLLS_2022,
    pollster_ratings: dict[str, float] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Sample polls using only low-rated pollsters.

    Selects pollsters with ``POLLSTER_RATINGS`` values at or below
    ``_LOW_RATING_CUTOFF`` (default 4.0).  Stresses the model with the
    worst-quality trackers.

    Args:
        polls: DataFrame with an ``encuestadora`` column.
        n_total: Maximum number of polls to sample (default 10).
        pollster_ratings: Mapping of pollster name to rating.
        seed: Random seed for reproducibility.

    Returns:
        Sampled DataFrame with at most *n_total* rows.

    Raises:
        ValueError: If no low-rated pollsters are found.

    """
    _ratings: dict[str, float] = (
        dict(POLLSTER_RATINGS) if pollster_ratings is None else pollster_ratings
    )

    low_pollsters = [name for name, rating in _ratings.items() if rating <= _LOW_RATING_CUTOFF]
    if not low_pollsters:
        msg = f"no pollsters found with rating <= {_LOW_RATING_CUTOFF}; cannot sample all_low polls"
        raise ValueError(msg)

    sampled = polls.loc[polls["encuestadora"].isin(low_pollsters), :].copy()
    if len(sampled) > n_total:
        sampled = sampled.sample(n=n_total, random_state=seed)
    return sampled


def sample_bootstrap_polls(
    polls: pd.DataFrame,
    n_total: int = _MAX_POLLS_2022,
    n_runs: int = _BOOTSTRAP_RUNS,
    seed: int | None = None,
) -> list[pd.DataFrame]:
    """Generate bootstrap random subsets of polls for sensitivity checks.

    Args:
        polls: Poll DataFrame.
        n_total: Number of polls per bootstrap sample.
        n_runs: Number of bootstrap iterations.
        seed: Random seed for reproducibility.

    Returns:
        List of *n_runs* sampled DataFrames.

    """
    _rng = np.random.default_rng(seed)
    samples: list[pd.DataFrame] = []
    for _ in range(n_runs):
        sampled = polls.sample(n=min(n_total, len(polls)), random_state=_rng)
        samples.append(sampled)
    return samples


# ═══════════════════════════════════════════════════════════════════════
# Metric helpers
# ═══════════════════════════════════════════════════════════════════════


def _compute_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Compute the coefficient of determination (R-squared).

    Args:
        actual: Array of observed values.
        predicted: Array of predicted values.

    Returns:
        R-squared score in (-inf, 1]; 1 is perfect.

    """
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def _build_missing_data_report(
    year: int,
    r2: float,
    errors: list[dict[str, float | str]],
    extra_detail: str = "",
) -> str:
    """Build a formatted missing-data report when R-squared is below the gate.

    Args:
        year: The held-out year.
        r2: Computed R-squared.
        errors: List of dicts with keys ``candidate``, ``predicted_mean``,
            ``actual_share``, ``abs_error``.
        extra_detail: Optional additional diagnostic text.

    Returns:
        Formatted report string suitable for a ``ValueError``.

    """
    header = (
        f"OOS validation FAILED for year {year}: R² = {r2:.4f} < {_R2_GATE}\n"
        "Missing-data report: components with insufficient predictive signal.\n"
    )
    col_header = "\nCandidate  Predicted  Actual  |Error|\n" + "-" * 50
    rows = "\n".join(
        f"{err['candidate']:<20} {err['predicted_mean']:>8.4f}  "
        f"{err['actual_share']:>6.4f}  {err['abs_error']:>7.4f}"
        for err in errors
    )
    extra = f"\n{extra_detail}" if extra_detail else ""
    footer = "\n\nAction required: add more training data or improve fundamentals."

    return header + col_header + "\n" + rows + extra + footer


# ═══════════════════════════════════════════════════════════════════════
# Leave-one-year-out cross-validation
# ═══════════════════════════════════════════════════════════════════════


def leave_one_year_out(
    features: pd.DataFrame,
    all_years_polls: dict[int, pd.DataFrame],
    all_years_results: dict[int, RoundResult],
    config: ModelConfig,
) -> pd.DataFrame:
    """Run leave-one-year-out cross-validation.

    For each held-out year *y*:
        Train on years != *y* (using CEDAE historical results for those
        years as anchors).
        Predict *y* using polls up to (*y* - 1 month) and fundamentals.
        Compare to actual *y* results.

    Args:
        features: Municipal feature matrix (from
            :func:`co_president.fundamentals.features.load_features`).
        all_years_polls: Mapping of ``{year: DataFrame}`` with polls for
            each election year.
        all_years_results: Mapping of ``{year: RoundResult}`` with actual
            election results for each year.
        config: Model hyperparameters.

    Returns:
        DataFrame with columns: ``held_out_year``, ``candidate``,
        ``predicted_mean``, ``actual_share``, ``abs_error``, ``r2``.

    Raises:
        ValueError: If the cross-validated R-squared is below 0.3.

    """
    rows: list[dict[str, object]] = []
    for year, result in all_years_results.items():
        train_polls_list = [df for y, df in all_years_polls.items() if y != year]
        if not train_polls_list:
            logger.warning("leave_one_year_out: no training polls for year %d, skipping", year)
            continue
        train_polls = pd.concat(train_polls_list, ignore_index=True)

        candidate_keys = sorted(set(FIRST_ROUND_CANDIDATES) & set(train_polls.columns))
        if not candidate_keys:
            logger.warning("leave_one_year_out: no candidate overlap for year %d, skipping", year)
            continue

        model = build_municipal_model(
            features,
            train_polls,
            None,
            config,
            target_year=year,
        )
        idata = sample_municipal_model(model, config)

        p_natl = idata.posterior["p_natl"].to_numpy()
        means = p_natl.mean(axis=(0, 1))

        errors: list[dict[str, float | str]] = []
        for i, key in enumerate(candidate_keys):
            actual_share = result.get_share(key)
            pred_mean = float(means[i])
            abs_err = abs(pred_mean - actual_share)
            errors.append(
                {
                    "candidate": key,
                    "predicted_mean": pred_mean,
                    "actual_share": actual_share,
                    "abs_error": abs_err,
                }
            )
            rows.append(
                {
                    "held_out_year": year,
                    "candidate": key,
                    "predicted_mean": pred_mean,
                    "actual_share": actual_share,
                    "abs_error": abs_err,
                    "r2": 0.0,
                }
            )

        r2 = _compute_r2(
            np.array([e["actual_share"] for e in errors]),
            np.array([e["predicted_mean"] for e in errors]),
        )
        for row_dict in rows:
            if row_dict["held_out_year"] == year:
                row_dict["r2"] = r2

        if r2 < _R2_GATE:
            report = _build_missing_data_report(year, r2, errors)
            raise ValueError(report)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# Year 2018 holdout test
# ═══════════════════════════════════════════════════════════════════════


def year_2018_holdout(
    features: pd.DataFrame,
    polls_to_2014: pd.DataFrame,
    results_2018: RoundResult,
    config: ModelConfig,
) -> pd.DataFrame:
    """Train on 2002-2014 data and predict 2018.

    Tests whether beta coefficients transfer across coalition alignments
    (Petro/Duque vs. Petro/Hernandez).  This is the **strongest** OOS
    test because the 2018 election was the first Petro-vs-Duque race
    with fundamentally different coalition structures.

    Args:
        features: Municipal feature matrix with ``historical`` column
            containing a pre-2018 record (typically 2014 round 1).
        polls_to_2014: Poll DataFrame with data up to 2014.
        results_2018: Actual 2018 round 1 results.
        config: Model hyperparameters.

    Returns:
        DataFrame with columns: ``candidate``, ``predicted_mean``,
        ``actual_share``, ``abs_error``.

    Raises:
        ValueError: If R-squared < 0.3 (gating test).

    """
    candidate_keys = sorted(set(FIRST_ROUND_CANDIDATES) & set(polls_to_2014.columns))
    if not candidate_keys:
        msg = (
            "year_2018_holdout: no candidate columns overlap between "
            "FIRST_ROUND_CANDIDATES and polls_to_2014"
        )
        raise ValueError(msg)

    model = build_municipal_model(
        features,
        polls_to_2014,
        None,
        config,
        target_year=2014,
    )
    idata = sample_municipal_model(model, config)

    p_natl = idata.posterior["p_natl"].to_numpy()
    means = p_natl.mean(axis=(0, 1))

    errors: list[dict[str, float | str]] = []
    for i, key in enumerate(candidate_keys):
        actual_share = results_2018.get_share(key)
        pred_mean = float(means[i])
        errors.append(
            {
                "candidate": key,
                "predicted_mean": pred_mean,
                "actual_share": actual_share,
                "abs_error": abs(pred_mean - actual_share),
            }
        )

    r2 = _compute_r2(
        np.array([e["actual_share"] for e in errors]),
        np.array([e["predicted_mean"] for e in errors]),
    )

    _log_generalization_report(2018, r2, errors)

    if r2 < _R2_GATE:
        report = _build_missing_data_report(2018, r2, errors)
        raise ValueError(report)

    return pd.DataFrame(errors)


# ═══════════════════════════════════════════════════════════════════════
# Leave 2022 out (sparse-poll deployment test)
# ═══════════════════════════════════════════════════════════════════════


def leave_2022_out(  # noqa: C901, PLR0913
    features: pd.DataFrame,
    polls_2022: pd.DataFrame,
    results_2022: RoundResult,
    config: ModelConfig,
    sampling_strategy: Literal["stratified", "all_low", "bootstrap"] = "stratified",
    pollster_ratings: dict[str, float] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Predict 2022 using fundamentals + at most 10 sampled polls.

    Enforces a strict <=10-poll cap (raises ``ValueError`` if
    ``len(polls_2022) > 10``).  Simulates the sparse-poll scenario
    expected in 2026.

    Args:
        features: Municipal feature matrix.
        polls_2022: Poll DataFrame from 2022 (will be sampled down to
            at most 10 rows).
        results_2022: Actual 2022 round 1 results.
        config: Model hyperparameters.
        sampling_strategy: One of ``"stratified"`` (default),
            ``"all_low"``, or ``"bootstrap"``.  Note: ``"bootstrap"``
            is a pass-through that passes ``polls_2022`` through
            unchanged.  Bootstrap resampling is not handled inline
            because :func:`sample_bootstrap_polls` returns a list of
            DataFrames (one per run).  For actual resampling, call
            :func:`sample_bootstrap_polls` externally and loop over
            the returned samples.
        pollster_ratings: Optional ratings dict; defaults to
            :data:`co_president.config.POLLSTER_RATINGS`.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: ``candidate``, ``predicted_mean``,
        ``actual_share``, ``abs_error``.

    Raises:
        ValueError: If ``len(polls_2022) > 10`` or R-squared < 0.3.

    """
    if len(polls_2022) > _MAX_POLLS_2022:
        msg = (
            f"leave_2022_out: received {len(polls_2022)} polls, "
            f"but the hard cap is {_MAX_POLLS_2022}. "
            "Use a sampling strategy to reduce the poll count."
        )
        raise ValueError(msg)

    if sampling_strategy == "bootstrap":
        # Bootstrap runs are not handled inline because
        # sample_bootstrap_polls() returns a list of DataFrames
        # (one per run).  To use bootstrap, call
        # sample_bootstrap_polls() externally and run
        # leave_2022_out per sample.  Currently this branch
        # passes polls_2022 through unchanged as a convenience
        # for users who want to bypass sampling entirely.
        _polls = polls_2022
    elif sampling_strategy == "all_low":
        _polls = sample_all_low_polls(polls_2022, pollster_ratings=pollster_ratings, seed=seed)
    else:
        _polls = sample_stratified_polls(polls_2022, pollster_ratings=pollster_ratings, seed=seed)

    if len(_polls) == 0:
        msg = f"leave_2022_out: no polls remain after '{sampling_strategy}' sampling"
        raise ValueError(msg)

    candidate_keys = sorted(set(FIRST_ROUND_CANDIDATES) & set(_polls.columns))
    if not candidate_keys:
        msg = (
            "leave_2022_out: no candidate columns overlap between FIRST_ROUND_CANDIDATES and polls"
        )
        raise ValueError(msg)

    model = build_municipal_model(features, _polls, None, config, target_year=2022)
    idata = sample_municipal_model(model, config)

    p_natl = idata.posterior["p_natl"].to_numpy()
    means = p_natl.mean(axis=(0, 1))

    errors: list[dict[str, float | str]] = []
    for i, key in enumerate(candidate_keys):
        actual_share = results_2022.get_share(key)
        pred_mean = float(means[i])
        errors.append(
            {
                "candidate": key,
                "predicted_mean": pred_mean,
                "actual_share": actual_share,
                "abs_error": abs(pred_mean - actual_share),
            }
        )

    r2 = _compute_r2(
        np.array([e["actual_share"] for e in errors]),
        np.array([e["predicted_mean"] for e in errors]),
    )

    abs_errors = np.array([float(e["abs_error"]) for e in errors])
    mae = float(abs_errors.mean())

    # ── Gating: top-3 candidates by actual vote share must have MAE < 5 pp ──
    sorted_by_share = sorted(
        errors,
        key=lambda x: float(x["actual_share"]),
        reverse=True,
    )
    top3_by_share = sorted_by_share[:3]
    for e in top3_by_share:
        if float(e["abs_error"]) > _MAE_PP_THRESHOLD:
            msg = (
                f"leave_2022_out: candidate {e['candidate']} "
                f"(actual {float(e['actual_share']):.4f}) "
                f"abs_error={float(e['abs_error']):.4f} exceeds "
                f"{_MAE_PP_THRESHOLD * 100:.0f} pp threshold"
            )
            raise ValueError(msg)

    # ── Gating: 94% HDI for Gustavo Petro must contain 40.34 % ──────────────
    petro_idx = candidate_keys.index("gustavo_petro")
    petro_draws = idata.posterior["p_natl"].to_numpy()[:, :, petro_idx].flatten()
    hdi_94 = az.hdi(petro_draws, hdi_prob=0.94)  # type: ignore[reportUnknownMemberType]
    petro_actual = 0.4034
    if not (hdi_94[0] <= petro_actual <= hdi_94[1]):
        msg = (
            f"leave_2022_out: 94% HDI for Gustavo Petro [{hdi_94[0]:.4f}, {hdi_94[1]:.4f}] "
            f"does not contain actual share {petro_actual}"
        )
        raise ValueError(msg)

    logger.info(
        "leave_2022_out: R²=%.4f, MAE=%.4f (strategy=%s, n_polls=%d)",
        r2,
        mae,
        sampling_strategy,
        len(_polls),
    )

    if r2 < _R2_GATE:
        report = _build_missing_data_report(2022, r2, errors)
        raise ValueError(report)

    return pd.DataFrame(errors)


# ═══════════════════════════════════════════════════════════════════════
# Mode comparison: "off" vs "joint" posteriors
# ═══════════════════════════════════════════════════════════════════════


def compare_modes(
    features: pd.DataFrame,
    polls: pd.DataFrame,
    results: RoundResult | None,
    config_off: ModelConfig,
    config_joint: ModelConfig,
) -> dict[str, float]:
    """Compare ``"off"`` vs ``"joint"`` mode posteriors.

    Asserts bounded difference between the two modes to detect
    overfitting.  If the national-level difference exceeds 1 percentage
    point for any candidate, a warning is logged.

    Args:
        features: Municipal feature matrix.
        polls: Poll DataFrame.
        results: Optional RoundResult (passed to ``build_municipal_model``).
        config_off: ModelConfig with ``fundamentals_mode="off"`` (or
            equivalent minimal configuration).
        config_joint: ModelConfig with ``fundamentals_mode="joint"``
            (full hierarchical model).

    Returns:
        Dict mapping candidate keys to ``abs_diff`` (absolute posterior
        mean difference between modes).

    """
    candidate_keys = sorted(set(FIRST_ROUND_CANDIDATES) & set(polls.columns))

    joint_model = build_municipal_model(features, polls, results, config_joint, target_year=2022)
    joint_idata = sample_municipal_model(joint_model, config_joint)

    off_model = build_municipal_model(features, polls, results, config_off, target_year=2022)
    off_idata = sample_municipal_model(off_model, config_off)

    joint_mean = joint_idata.posterior["p_natl"].to_numpy().mean(axis=(0, 1))
    off_mean = off_idata.posterior["p_natl"].to_numpy().mean(axis=(0, 1))

    diffs: dict[str, float] = {}
    for i, key in enumerate(candidate_keys):
        diff = abs(float(joint_mean[i]) - float(off_mean[i]))
        diffs[key] = diff
        if diff > _MODE_DIFF_THRESHOLD:
            logger.warning(
                "compare_modes: candidate %s diff=%.4f exceeds 1 pp threshold",
                key,
                diff,
            )

    return diffs


# ═══════════════════════════════════════════════════════════════════════
# Report logging
# ═══════════════════════════════════════════════════════════════════════


def _log_generalization_report(
    year: int,
    r2: float,
    errors: list[dict[str, float | str]],
) -> None:
    """Log the generalization test result to ``results/generalization_report.md``.

    Args:
        year: The held-out year.
        r2: R-squared value.
        errors: List of error dicts per candidate.

    """
    report_path = Path("results") / "generalization_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        f"# Generalization Report - {year} Holdout\n"
        f"\n"
        f"**R²**: {r2:.4f}\n"
        f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"\n"
        f"| Candidate | Predicted | Actual | |Error| |\n"
        f"|-----------|-----------|--------|--------|\n"
    )
    rows = "\n".join(
        f"| {err['candidate']} | {err['predicted_mean']:.4f} | "
        f"{err['actual_share']:.4f} | {err['abs_error']:.4f} |"
        for err in errors
    )

    with report_path.open("w") as f:
        f.write(header + rows + "\n")
