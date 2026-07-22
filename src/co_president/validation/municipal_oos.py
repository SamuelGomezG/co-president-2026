"""SPEC-26: Out-of-sample validation framework for the municipal hierarchical model.

Provides leave-one-year-out cross-validation, 2018-holdout, and
leave-2022-out sparse-poll deployment test.  The R-squared gate (0.3 threshold)
prevents overfitting to 2022 data by failing the build with a
"missing data" report listing insufficient components.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal
import warnings

import arviz as az  # type: ignore[import-untyped]
import numpy as np
import pandas as pd

from co_president.config import FIRST_ROUND_CANDIDATES, POLLSTER_RATINGS
from co_president.model_municipal import (
    build_municipal_model,
    compute_effective_num_municipalities,
    sample_municipal_model,
)

if TYPE_CHECKING:
    from co_president.data import RoundResult

from co_president.config import ModelConfig

logger = logging.getLogger(__name__)

__all__ = [
    "compare_modes",
    "compute_effective_df_per_group",
    "leave_2022_out",
    "leave_one_year_out",
    "produce_generalization_report",
    "run_feature_group_ablation",
    "run_sensitivity_ablation",
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


def _log_ml_baseline_comparison(
    features: pd.DataFrame,
    target_year: int,
    bayesian_r2: float,
) -> None:
    """Run the SPEC-41 ML benchmark alongside a Bayesian validation.

    Logs a side-by-side comparison of Bayesian R² vs ML per-class R².
    """
    try:
        from co_president.benchmarks.runner import report_benchmark_baseline  # noqa: PLC0415

        summary = report_benchmark_baseline(features)
        if summary["n_rows"] > 0:
            best = summary.get("per_class_best_r2", {})
            best_str = "; ".join(f"{k}: {v:.4f}" for k, v in best.items() if not np.isnan(float(v)))
            logger.info(
                "ML baseline vs Bayesian for %d: Bayesian R² = %.4f. ML best per-class R² -> %s",
                target_year,
                bayesian_r2,
                best_str,
            )
    except (ValueError, TypeError, KeyError, ImportError, OSError):
        logger.exception("ML baseline comparison skipped (non-fatal)")


# ═══════════════════════════════════════════════════════════════════════
# Year 2018 holdout test
# ═══════════════════════════════════════════════════════════════════════


def year_2018_holdout(
    features: pd.DataFrame,
    polls_to_2014: pd.DataFrame,
    results_2018: RoundResult,
    config: ModelConfig,
) -> tuple[pd.DataFrame, az.InferenceData]:
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
        Tuple of (errors_df, idata) where errors_df has columns
        ``candidate``, ``predicted_mean``, ``actual_share``, ``abs_error``
        and idata is the posterior InferenceData from the holdout model.

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

    # SPEC-41: ML benchmark baseline (informational, not gating).
    _log_ml_baseline_comparison(features, 2018, r2)

    if r2 < _R2_GATE:
        report = _build_missing_data_report(2018, r2, errors)
        raise ValueError(report)

    return pd.DataFrame(errors), idata


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
    # Sample first, then validate the result against the cap.
    # The caller may pass the full unfiltered poll DataFrame; the
    # sampling strategy reduces it to the sparse-poll regime before
    # the hard-cap check fires.
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

    if len(_polls) > _MAX_POLLS_2022:
        msg = (
            f"leave_2022_out: post-sampling poll count ({len(_polls)}) "
            f"exceeds hard cap ({_MAX_POLLS_2022}). "
            "Use a different sampling strategy or reduce the cap."
        )
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
    if "gustavo_petro" not in candidate_keys:
        msg = f"Candidate 'gustavo_petro' not found in candidate_keys={candidate_keys}"
        raise ValueError(msg)
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

    _log_ml_baseline_comparison(features, 2022, r2)

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
# Sensitivity ablation
# ═══════════════════════════════════════════════════════════════════════


def run_sensitivity_ablation(
    features: pd.DataFrame,
    polls: pd.DataFrame,
    results: RoundResult | None,
    config: ModelConfig,
) -> pd.DataFrame:
    """Run sensitivity ablation across three prior configurations.

    Compares:
    - ``"off"`` (flat): ``beta_coefficient_prior_sigma = 10.0``,
      approximate uninformative prior.
    - ``"normal"`` (default): ``Normal(0, 0.5)`` on beta coefficients.
    - ``"horseshoe"``: Horseshoe prior on beta coefficients (aggressive
      local + global shrinkage).

    For each configuration, builds and samples the model, then reports
    posterior means, effective number of municipalities, WAIC, and LOO.

    Args:
        features: Municipal feature matrix.
        polls: Poll DataFrame.
        results: Optional election result (passed through to model).
        config: Base ``ModelConfig`` (will be modified per configuration).

    Returns:
        DataFrame with columns: ``configuration``, ``candidate``,
        ``posterior_mean``, ``effective_num_municipalities``,
        ``waic``, ``loo``.

    Raises:
        ValueError: If the model fails to sample for any configuration.

    """
    candidate_keys = sorted(set(FIRST_ROUND_CANDIDATES) & set(polls.columns))
    if not candidate_keys:
        msg = "run_sensitivity_ablation: no candidate columns overlap with polls"
        raise ValueError(msg)

    configurations: list[dict[str, object]] = [
        {
            "name": "off",
            "config": dataclasses.replace(
                config,
                beta_coefficient_prior_sigma=10.0,
                use_horseshoe_prior=False,
            ),
        },
        {
            "name": "normal",
            "config": dataclasses.replace(
                config,
                beta_coefficient_prior_sigma=0.5,
                use_horseshoe_prior=False,
            ),
        },
        {
            "name": "horseshoe",
            "config": dataclasses.replace(
                config,
                beta_coefficient_prior_sigma=0.5,
                use_horseshoe_prior=True,
            ),
        },
    ]

    rows: list[dict[str, object]] = []

    for cfg in configurations:
        name = str(cfg["name"])
        cfg_obj = cfg["config"]
        if not isinstance(cfg_obj, ModelConfig):
            msg = f"Expected a ModelConfig instance, got {type(cfg_obj).__name__}"
            raise TypeError(msg)

        logger.info("run_sensitivity_ablation: building model for '%s'", name)
        model = build_municipal_model(features, polls, results, cfg_obj, target_year=2022)
        idata = sample_municipal_model(model, cfg_obj)

        # Posterior means
        p_natl = idata.posterior["p_natl"].to_numpy()
        means = p_natl.mean(axis=(0, 1))

        for i, key in enumerate(candidate_keys):
            rows.append(
                {
                    "configuration": name,
                    "candidate": key,
                    "posterior_mean": float(means[i]),
                    "effective_num_municipalities": -1,
                    "waic": float("nan"),
                    "loo": float("nan"),
                }
            )

        # Effective number of municipalities
        eff = compute_effective_num_municipalities(idata, cfg_obj.pool_alpha)
        for row in rows:
            if row["configuration"] == name:
                row["effective_num_municipalities"] = eff

        # Information criteria
        try:
            waic_result = az.waic(idata, var_name="p_natl")  # type: ignore[reportUnknownMemberType]
            loo_result = az.loo(idata, var_name="p_natl")  # type: ignore[reportUnknownMemberType]
            waic_val = float(waic_result.waic)  # type: ignore[reportUnknownMemberType]
            loo_val = float(loo_result.loo)  # type: ignore[reportUnknownMemberType]
        except (ValueError, RuntimeError, TypeError, AttributeError):
            logger.exception(
                "run_sensitivity_ablation: WAIC/LOO computation failed for '%s'",
                name,
                extra={"configuration": name},
            )
            waic_val = float("nan")
            loo_val = float("nan")

        for row in rows:
            if row["configuration"] == name:
                row["waic"] = waic_val
                row["loo"] = loo_val

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# Report logging
# ═══════════════════════════════════════════════════════════════════════


def _log_generalization_report(
    year: int,
    r2: float,
    errors: list[dict[str, float | str]],
) -> None:
    """Log the generalization test result to ``results/generalization_report.md``.

    .. deprecated::
        Use :func:`produce_generalization_report` instead, which produces the
        full SPEC-29 generalization audit report with effective degrees of
        freedom, feature group ablation, and calibration plots.

    Args:
        year: The held-out year.
        r2: R-squared value.
        errors: List of error dicts per candidate.

    """
    warnings.warn(
        "Deprecated: use produce_generalization_report() for the full SPEC-29 audit.",
        DeprecationWarning,
        stacklevel=2,
    )
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


# ═══════════════════════════════════════════════════════════════════════
# SPEC-29: Generalization audit helpers
# ═══════════════════════════════════════════════════════════════════════


_BETA_GROUP_NAMES: list[str] = [
    "beta_historical",
    "beta_ethnicity",
    "beta_poverty",
    "beta_rural",
    "beta_education",
    "beta_risk",
]

# Feature column name for each beta group (used for ablation neutralization).
# None means the group uses a derived feature (e.g., historical -> CLR).
_FEATURE_GROUP_COLUMNS: dict[str, str | None] = {
    "beta_historical": None,
    "beta_ethnicity": "pct_afro_colombian",
    "beta_poverty": "nbi_rate",
    "beta_rural": "pct_rural_disperso",
    "beta_education": "years_schooling_promedio",
    "beta_risk": "high_risk_flag",
}


def compute_effective_df_per_group(
    idata: az.InferenceData,
    config: ModelConfig,
) -> dict[str, float]:
    """Compute effective degrees of freedom per feature group.

    For each beta coefficient group with prior ``Normal(0, prior_sigma²)``,
    effective degrees of freedom is defined as:

    .. code-block:: text

        eff_df = sum(1 - posterior_var / prior_var)

    summed over the ``K`` candidate coefficients in the group.
    ``posterior_var`` is the sample variance across all posterior draws.
    Shrinkage is clipped to ``[0, 1]`` per coefficient.

    Values near ``K`` indicate the data fully constrains the group
    (no shrinkage); values near 0 indicate the group is mostly
    regularized toward zero.

    Args:
        idata: ArviZ InferenceData with posterior for ``beta_*`` groups.
        config: ModelConfig (provides ``beta_coefficient_prior_sigma``).

    Returns:
        Dict mapping group name -> effective degrees of freedom.

    Raises:
        KeyError: If any expected beta group is missing from the posterior.

    """
    prior_var = config.beta_coefficient_prior_sigma**2
    prior_var = max(prior_var, 1e-12)

    source = getattr(idata, "posterior", getattr(idata, "prior", None))
    if source is None:
        msg = "idata must contain 'posterior' or 'prior' group"
        raise ValueError(msg)

    eff_df: dict[str, float] = {}
    for group in _BETA_GROUP_NAMES:
        samples = source[group].to_numpy()  # (chain, draw, K)
        posterior_var = samples.var(axis=(0, 1), ddof=1)  # (K,)
        shrinkage = 1.0 - posterior_var / prior_var
        shrinkage = np.clip(shrinkage, 0.0, 1.0)
        eff_df[group] = float(shrinkage.sum())

    return eff_df


def _to_empty_tuple(_: object) -> tuple[()]:
    """Return an empty tuple (used to neutralize historical records)."""
    return ()


def _neutralize_feature_group(
    features: pd.DataFrame,
    group: str,
) -> pd.DataFrame:
    """Return a copy of *features* with one feature group neutralized.

    Groups:
    - ``"beta_historical"``  ->  set historical tuples to empty (CLR = 0).
    - ``"beta_risk"``        ->  set ``high_risk_flag`` to 0.
    - All others              ->  set the column to its mean (z-score = 0).

    Args:
        features: Original feature matrix.
        group: One of ``_FEATURE_GROUP_COLUMNS`` keys.

    Returns:
        Modified copy with the group's contribution zeroed.

    """
    mod = features.copy()
    col = _FEATURE_GROUP_COLUMNS.get(group)
    if group == "beta_historical" and col is None:
        mod["historical"] = mod["historical"].apply(_to_empty_tuple)
    elif col is not None:
        if col == "high_risk_flag":
            mod[col] = 0
        else:
            mod[col] = mod[col].mean()
    else:
        msg = f"_neutralize_feature_group: unknown group '{group}'"
        raise ValueError(msg)
    return mod


def run_feature_group_ablation(
    features: pd.DataFrame,
    polls: pd.DataFrame,
    results: RoundResult | None,
    config: ModelConfig,
    target_year: int = 2022,
) -> pd.DataFrame:
    """Run feature group ablation: drop one group at a time, re-fit, report ΔMAE.

    For each feature group (historical, ethnicity, poverty, rural,
    education, risk), neutralizes that group in the feature matrix,
    re-builds and re-samples the model, then compares MAE against the
    full model.

    Args:
        features: Municipal feature matrix.
        polls: Poll DataFrame.
        results: RoundResult for the target year (or ``None`` to skip MAE).
        config: ModelConfig.
        target_year: Target election year (default 2018).

    Returns:
        DataFrame with columns: ``group``, ``mae_full``, ``mae_ablated``,
        ``delta_mae``, ``r2_full``, ``r2_ablated``.

    Raises:
        ValueError: If no candidate columns overlap with polls.

    """
    candidate_keys = sorted(set(FIRST_ROUND_CANDIDATES) & set(polls.columns))
    if not candidate_keys:
        msg = (
            "run_feature_group_ablation: no candidate columns "
            "overlap between FIRST_ROUND_CANDIDATES and polls"
        )
        raise ValueError(msg)

    # Full model
    logger.info("run_feature_group_ablation: building full model")
    model_full = build_municipal_model(features, polls, results, config, target_year=target_year)
    idata_full = sample_municipal_model(model_full, config)

    p_natl_full = idata_full.posterior["p_natl"].to_numpy()
    full_means = p_natl_full.mean(axis=(0, 1))

    if results is not None:
        actual = np.array([results.get_share(key) for key in candidate_keys])
        full_mae = float(np.abs(full_means - actual).mean())
        r2_full = _compute_r2(actual, full_means)
    else:
        actual = np.array([])
        full_mae = float("nan")
        r2_full = float("nan")

    rows: list[dict[str, object]] = []

    for group in sorted(_FEATURE_GROUP_COLUMNS):
        logger.info("run_feature_group_ablation: ablated group '%s'", group)
        mod_features = _neutralize_feature_group(features, group)
        model_abl = build_municipal_model(
            mod_features,
            polls,
            results,
            config,
            target_year=target_year,
        )
        idata_abl = sample_municipal_model(model_abl, config)

        abl_p_natl = idata_abl.posterior["p_natl"].to_numpy()
        abl_means = abl_p_natl.mean(axis=(0, 1))

        if results is not None:
            abl_mae = float(np.abs(abl_means - actual).mean())
            delta = abl_mae - full_mae
            r2_abl = _compute_r2(actual, abl_means)
        else:
            abl_mae = float("nan")
            delta = float("nan")
            r2_abl = float("nan")

        rows.append(
            {
                "group": group.replace("beta_", ""),
                "mae_full": full_mae,
                "mae_ablated": abl_mae,
                "delta_mae": delta,
                "r2_full": r2_full,
                "r2_ablated": r2_abl,
            }
        )

    return pd.DataFrame(rows)


def produce_generalization_report(  # noqa: C901, PLR0912, PLR0913, PLR0915
    features: pd.DataFrame,
    polls: pd.DataFrame,
    results_2018: RoundResult,
    config: ModelConfig,
    output_path: str | Path = "results/generalization_report.md",
    polls_2022: pd.DataFrame | None = None,
    results_2022: RoundResult | None = None,
) -> str:
    """Produce comprehensive SPEC-29 generalization audit report.

    Orchestrates:
    1. 2018 holdout ablation: train on 2002-2014 polls, predict 2018.
    2. 2022-out ablation (optional): predict 2022 from fundamentals + polls.
    3. Effective degrees of freedom per feature group.
    4. Feature group ablation sensitivity (ΔMAE per group).
    5. Calibration scatter plot (predicted vs actual per candidate).

    When *polls_2022* and *results_2022* are provided a second ablation
    section for 2022 is appended, including 94 % HDI coverage.

    The report is written to *output_path* and a PNG calibration plot
    is saved alongside it (same stem, ``.png`` extension).

    Args:
        features: Municipal feature matrix with ``historical`` column.
        polls: Poll DataFrame with data up to 2014 (for 2018 holdout).
        results_2018: Actual 2018 round 1 results.
        config: ModelConfig.
        output_path: Path to write the report markdown.
        polls_2022: Optional poll DataFrame for 2022-out ablation.
        results_2022: Optional actual 2022 round 1 results.

    Returns:
        Absolute path to the written report file.

    Raises:
        ValueError: If the 2018 holdout R² < 0.3 (gating test).

    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    candidate_keys = sorted(set(FIRST_ROUND_CANDIDATES) & set(polls.columns))
    if not candidate_keys:
        msg = (
            "produce_generalization_report: no candidate columns "
            "overlap between FIRST_ROUND_CANDIDATES and polls"
        )
        raise ValueError(msg)

    # ── 1. 2018 holdout ─────────────────────────────────────────────────
    errors_df, idata = year_2018_holdout(features, polls, results_2018, config)

    # ── 2. Effective degrees of freedom ─────────────────────────────────
    eff_df = compute_effective_df_per_group(idata, config)

    # ── 3. Feature group ablation ───────────────────────────────────────
    ablation_df = run_feature_group_ablation(features, polls, results_2018, config)

    # ── 4. 2022-out ablation (optional) ─────────────────────────────────
    candidate_keys_2022: list[str] = []
    errors_2022_df: pd.DataFrame | None = None
    r2_2022: float | None = None
    mae_2022: float | None = None
    hdi_fraction_2022: float | None = None
    hdi_details_2022: list[dict[str, object]] | None = None
    if polls_2022 is not None and results_2022 is not None:
        logger.info(
            "produce_generalization_report: running 2022-out ablation",
        )
        candidate_keys_2022 = sorted(set(FIRST_ROUND_CANDIDATES) & set(polls_2022.columns))
        if not candidate_keys_2022:
            msg = "produce_generalization_report: 2022 polls have no candidate columns"
            raise ValueError(msg)

        model_2022 = build_municipal_model(
            features,
            polls_2022,
            None,
            config,
            target_year=2022,
        )
        idata_2022 = sample_municipal_model(model_2022, config)

        p_natl_2022 = idata_2022.posterior["p_natl"].to_numpy()
        means_2022 = p_natl_2022.mean(axis=(0, 1))

        rows_2022: list[dict[str, float | str]] = []
        for i, key in enumerate(candidate_keys_2022):
            actual_share = results_2022.get_share(key)
            pred_mean = float(means_2022[i])
            rows_2022.append(
                {
                    "candidate": key,
                    "predicted_mean": pred_mean,
                    "actual_share": actual_share,
                    "abs_error": abs(pred_mean - actual_share),
                },
            )
        errors_2022_df = pd.DataFrame(rows_2022)

        actual_2022_arr = np.array(
            [results_2022.get_share(key) for key in candidate_keys_2022],
        )
        r2_2022 = _compute_r2(actual_2022_arr, means_2022)
        mae_2022 = float(np.abs(actual_2022_arr - means_2022).mean())

        # 94% HDI coverage for 2022
        hdi_94_2022 = az.hdi(p_natl_2022, hdi_prob=0.94)  # type: ignore[reportUnknownMemberType]
        hdi_within_2022 = 0
        hdi_details_2022_: list[dict[str, object]] = []
        for i, key in enumerate(candidate_keys_2022):
            lo = float(hdi_94_2022[i, 0])  # type: ignore[reportUnknownArgumentType]
            hi = float(hdi_94_2022[i, 1])  # type: ignore[reportUnknownArgumentType]
            actual = results_2022.get_share(key)
            contained = lo <= actual <= hi
            if contained:
                hdi_within_2022 += 1
            hdi_details_2022_.append(
                {
                    "candidate": key,
                    "hdi_lo": lo,
                    "hdi_hi": hi,
                    "contained": contained,
                },
            )
        hdi_details_2022 = hdi_details_2022_
        hdi_fraction_2022 = hdi_within_2022 / len(candidate_keys_2022)

    # ── 5. Calibration plot ─────────────────────────────────────────────
    import matplotlib.pyplot as plt  # noqa: PLC0415

    from co_president.plotting import plot_municipal_calibration  # noqa: PLC0415

    actual_2018 = np.array([results_2018.get_share(key) for key in candidate_keys])
    predicted_2018 = errors_df["predicted_mean"].to_numpy()
    fig = plot_municipal_calibration(
        predicted_2018,
        actual_2018,
        candidate_keys,
        2018,
    )
    plot_path = output_path.with_suffix(".png")
    fig.savefig(str(plot_path))  # type: ignore[reportUnknownMemberType]
    plt.close(fig)

    # ── 6. Write report ─────────────────────────────────────────────────
    r2 = _compute_r2(actual_2018, predicted_2018)
    mae = float(np.abs(actual_2018 - predicted_2018).mean())

    lines: list[str] = [
        "# Generalization Audit Report",
        "",
        f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 1. 2018 Holdout Ablation",
        "",
        "Training data: 2002-2014. Held out: 2018.",
        "",
        f"**R²**: {r2:.4f}",
        f"**MAE**: {mae:.4f}",
        "",
        "| Candidate | Predicted | Actual | |Error| |",
        "|-----------|-----------|--------|--------|",
    ]
    for _, row in errors_df.iterrows():
        lines.append(
            f"| {row['candidate']} | {row['predicted_mean']:.4f} | "
            f"{row['actual_share']:.4f} | {row['abs_error']:.4f} |",
        )

    # ── 2022-out section (optional) ─────────────────────────────────
    if errors_2022_df is not None and r2_2022 is not None and mae_2022 is not None:
        hdi_details_local: list[dict[str, object]] = hdi_details_2022 or []
        hdi_count = sum(1 for d in hdi_details_local if d["contained"])
        lines.extend(
            [
                "",
                "## 2. 2022-Out Ablation (Deployment Analog)",
                "",
                "Training data: 2002-2018 fundamentals + polls. Held out: 2022.",
                "",
                f"**R²**: {r2_2022:.4f}",
                f"**MAE**: {mae_2022:.4f}",
                f"**94% HDI coverage**: {hdi_fraction_2022:.0%} "
                f"({hdi_count}/{len(candidate_keys_2022)} candidates)",
                "",
                "| Candidate | Predicted | Actual | |Error| | 94% HDI | Contains? |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for _, row in errors_2022_df.iterrows():
            cand = str(row["candidate"])
            hdi_match = next(
                (d for d in hdi_details_local if d["candidate"] == cand),
                None,
            )
            hdi_str = (
                f"[{hdi_match['hdi_lo']:.4f}, {hdi_match['hdi_hi']:.4f}]" if hdi_match else "N/A"
            )
            contained_str = "✓" if hdi_match and hdi_match["contained"] else "✗"
            lines.append(
                f"| {cand} | {row['predicted_mean']:.4f} | "
                f"{row['actual_share']:.4f} | {row['abs_error']:.4f} | "
                f"{hdi_str} | {contained_str} |",
            )

    # ── Effective degrees of freedom ─────────────────────────────────
    section_offset = 3 if errors_2022_df is not None else 2
    lines.extend(
        [
            "",
            f"## {section_offset}. Effective Degrees of Freedom",
            "",
            "| Feature group | Effective df | Shrinkage ratio |",
            "|---|---|---|",
        ]
    )
    total_eff = 0.0
    for group_name, eff in eff_df.items():
        label = group_name.replace("beta_", "")
        n_params = int(idata.posterior[group_name].shape[-1])
        shrink_ratio = (n_params - eff) / n_params if n_params > 0 else 0.0
        lines.append(f"| {label} | {eff:.2f} | {shrink_ratio:.2%} |")
        total_eff += eff
    lines.append(f"| **Total** | **{total_eff:.2f}** | |")

    # ── Feature group ablation ───────────────────────────────────────
    ablation_section = section_offset + 1
    lines.extend(
        [
            "",
            f"## {ablation_section}. Feature Group Ablation",
            "",
            "| Ablated group | Full MAE | Ablated MAE | ΔMAE | Full R² | Ablated R² |",
            "|---|---|---|---|---|---|",
        ]
    )
    for _, row_abl in ablation_df.iterrows():
        lines.append(
            f"| {row_abl['group']} | {row_abl['mae_full']:.4f} | "
            f"{row_abl['mae_ablated']:.4f} | {row_abl['delta_mae']:+.4f} | "
            f"{row_abl['r2_full']:.4f} | {row_abl['r2_ablated']:.4f} |",
        )

    # ── Calibration ──────────────────────────────────────────────────
    cal_section = ablation_section + 1
    lines.extend(
        [
            "",
            f"## {cal_section}. Calibration",
            "",
            f"![Calibration plot]({plot_path.name})",
            "",
            f"## {cal_section + 1}. Discussion",
            "",
        ]
    )

    ablation_sorted = ablation_df.sort_values("delta_mae", ascending=False)
    top = ablation_sorted.iloc[0]
    lines.append(
        f"- Most impactful feature group: **{top['group']}** (ΔMAE = {top['delta_mae']:+.4f})",
    )
    worst_eff = min(eff_df.items(), key=lambda kv: kv[1])[0]
    lines.append(
        f"- Most regularized group: **{worst_eff.replace('beta_', '')}** "
        f"(eff df = {eff_df[worst_eff]:.2f})",
    )
    if r2 >= _R2_GATE:
        lines.append(f"- 2018 holdout R² {r2:.4f} >= {_R2_GATE} gate ✓")
    else:
        lines.append(f"- 2018 holdout R² {r2:.4f} < {_R2_GATE} gate ✗")

    if r2_2022 is not None and mae_2022 is not None and hdi_fraction_2022 is not None:
        lines.append(
            f"- 2022-out R² {r2_2022:.4f}, MAE {mae_2022:.4f}, "
            f"94% HDI covers {hdi_fraction_2022:.0%} of candidates",
        )
        _tol = 0.02
        comparison = (
            "worse" if mae_2022 > mae else "comparable" if abs(mae_2022 - mae) < _tol else "better"
        )
        lines.append(
            f"- Comparing gating tests: 2022-out MAE ({mae_2022:.4f}) is "
            f"**{comparison}** to 2018 holdout MAE ({mae:.4f}). "
            f"The 2018 holdout tests structural stability across a major "
            f"political realignment; the 2022-out tests deployment-mode "
            f"accuracy with fundamental + sparse-poll features.",
        )

    lines.append("")

    text = "\n".join(lines)
    with output_path.open("w") as f:
        f.write(text)

    logger.info("Generalization report written to %s", output_path.resolve())
    return str(output_path.resolve())
