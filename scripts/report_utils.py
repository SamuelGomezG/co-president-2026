"""Shared report utilities for backtesting accuracy reports.

Extracted from ``run_100k_report.py`` for reuse across multiple backtest
scripts (2022, 2026, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import logging
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import arviz as az
import numpy as np
import pandas as pd

from co_president.config import Candidate, ModelConfig, get_active_candidates

if TYPE_CHECKING:
    import xarray as xr

    from co_president.data import CleanPolls, RoundResult
    from co_president.model_runoff_matrix import RunoffMatrix

logger = logging.getLogger("report_utils")

_RHAT_THRESHOLD = 1.10
_MAE_TIGHT = 3.0
_MAE_LOOSE = 5.0


@dataclass
class ReportInputs:
    """Group all inputs for report generation to reduce function arity."""

    r1_metrics: dict
    r2_metrics: dict
    r1_rhat: float
    r1_converged: bool
    r2_rhat: float
    r2_converged: bool
    elapsed_r1_s: float
    elapsed_r2_s: float
    n_polls_r1: int
    n_polls_r2: int
    election_year: int = 2022
    candidate_registry: dict[str, Candidate] | None = None

    def __post_init__(self) -> None:
        """Derive the candidate registry from the election year if not provided."""
        if self.candidate_registry is None:
            self.candidate_registry = {
                c.key: c for c in get_active_candidates(1, year=self.election_year)
            }
        if self.candidate_registry:
            runoff_keys = {c.key for c in get_active_candidates(2, year=self.election_year)}
            for key in runoff_keys:
                if key not in self.candidate_registry:
                    self.candidate_registry[key] = Candidate(
                        key=key,
                        display_name=key,
                        coalition=None,
                        first_round=False,
                        runoff=True,
                    )


def format_pct(value: float) -> str:
    """Format a proportion as a percentage string."""
    if not np.isfinite(value):
        return "N/A"
    return f"{value * 100:.2f}%"


def format_pp(error: float) -> str:
    """Format an error with sign and pp suffix."""
    if not np.isfinite(error):
        return "N/A"
    return f"{error * 100:+.2f}pp"


def _hdi_95(samples: np.ndarray) -> tuple[float, float]:
    """Compute 95% highest-density interval for a 1-D array of samples.

    Args:
        samples: 1-D numpy array of posterior draws.

    Returns:
        Tuple of ``(lower, upper)`` bounds.

    """
    return az.hdi(samples, prob=0.95)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType, reportReturnType]


def compute_r1_accuracy(
    idata_r1: xr.DataTree,
    results_r1: RoundResult,
    poll_columns: set[str] | None = None,
    candidate_registry: dict[str, Candidate] | None = None,
) -> dict:
    """Compute round 1 accuracy metrics.

    Args:
        idata_r1: Round 1 posterior ``DataTree``.
        results_r1: Canonical round 1 ``RoundResult``.
        poll_columns: Set of poll column names used to determine model
            candidate ordering.  When ``None``, falls back to
            ``results_r1.candidates`` (may misalign if model excluded
            some candidates).
        candidate_registry: Mapping of candidate key to ``Candidate`` metadata.
            When ``None``, falls back to ``get_active_candidates(1)``.

    Returns:
        Dict with per-candidate metrics and aggregate scores.

    """
    from co_president.model_utils import extract_election_day_shares  # noqa: PLC0415

    if candidate_registry is None:
        candidate_registry = {c.key: c for c in get_active_candidates(1)}
    candidate_base = set(candidate_registry.keys())
    if poll_columns is not None:
        candidate_keys = sorted(candidate_base & poll_columns)
    else:
        candidate_keys = sorted(
            candidate_base & {c.candidate_key for c in results_r1.candidates},
        )
    shares = extract_election_day_shares(idata_r1, candidate_keys)

    candidates_metrics = []
    all_errors = []

    for key in candidate_keys:
        posterior = shares[key]
        try:
            actual = results_r1.get_share(key)
        except KeyError:
            actual = 0.0
        mean_pred = float(posterior.mean())
        median_pred = float(np.median(posterior))
        error = mean_pred - actual
        abs_error = abs(error)
        all_errors.append(abs_error)
        ci = _hdi_95(posterior)
        in_ci = bool(ci[0] <= actual <= ci[1])
        std_dev = float(posterior.std())
        display_name = candidate_registry[key].display_name if key in candidate_registry else key

        candidates_metrics.append(
            {
                "candidate_key": key,
                "display_name": display_name,
                "actual": round(actual, 6),
                "predicted_mean": round(mean_pred, 6),
                "predicted_median": round(median_pred, 6),
                "error": round(error, 6),
                "abs_error": round(abs_error, 6),
                "within_95_ci": in_ci,
                "ci_95_lower": round(ci[0], 6),
                "ci_95_upper": round(ci[1], 6),
                "posterior_std": round(std_dev, 6),
            },
        )

    mae = float(np.mean(all_errors)) if all_errors else 0.0
    rmse = float(np.sqrt(np.mean(np.square(all_errors)))) if all_errors else 0.0
    n_candidates = len(candidates_metrics)
    ci_coverage_95 = (
        sum(1 for cm in candidates_metrics if cm["within_95_ci"]) / n_candidates
        if n_candidates > 0
        else 0.0
    )

    return {
        "candidates": candidates_metrics,
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "ci_coverage_95": round(ci_coverage_95, 6),
    }


def compute_r2_accuracy_from_matrix(
    matrix: RunoffMatrix,
    results_r1: RoundResult,
    results_r2: RoundResult,
    idata_runoff: xr.DataTree | None = None,
    candidate_registry: dict[str, Candidate] | None = None,
) -> dict:
    """Compute runoff accuracy metrics from the probabilistic pairing matrix.

    Identifies the pairing that matches the actual top-two candidates from
    the historical result, then extracts win probability and margin from
    the matrix forecast.  When ``idata_runoff`` is provided, also computes
    95% highest-density intervals and posterior standard deviations.

    Args:
        matrix: :class:`RunoffMatrix` from :func:`estimate_runoff_matrix`.
        results_r1: Round 1 ``RoundResult`` (used to identify top-two).
        results_r2: Round 2 ``RoundResult``.
        idata_runoff: Posterior ``DataTree`` from the K=3 runoff model
            (optional).  Used to compute HDIs and posterior std.
        candidate_registry: Mapping of candidate key to ``Candidate`` metadata.
            When ``None``, falls back to ``get_active_candidates(1)``.

    Returns:
        Dict with pairing-level metrics and aggregate scores.

    """
    if candidate_registry is None:
        candidate_registry = {c.key: c for c in get_active_candidates(1)}

    top_two = results_r1.top_two()
    cand_a_key = top_two[0].candidate_key
    cand_b_key = top_two[1].candidate_key

    try:
        actual_a = results_r2.get_share(cand_a_key)
    except KeyError:
        actual_a = 0.0
    try:
        actual_b = results_r2.get_share(cand_b_key)
    except KeyError:
        actual_b = 0.0
    try:
        actual_rest = results_r2.get_share("rest")
    except KeyError:
        actual_rest = 0.0

    names_a = candidate_registry[cand_a_key].display_name
    names_b_entry = candidate_registry.get(cand_b_key, None)
    names_b = names_b_entry.display_name if names_b_entry else cand_b_key
    names_rest_entry = candidate_registry.get("rest", None)
    names_rest = names_rest_entry.display_name if names_rest_entry else "Rest"

    margin_actual = actual_a - actual_b

    pairing = None
    for pf in matrix.pairings:
        if pf.candidate_first == cand_a_key and pf.candidate_second == cand_b_key:
            pairing = pf
            break

    if pairing is not None:
        prob_a_wins = pairing.prob_first_wins
        mean_margin = pairing.mean_margin
        pred_a = pairing.mean_share_first
        pred_b = pairing.mean_share_second
        pred_rest = pairing.mean_share_rest
    else:
        prob_a_wins = float("nan")
        mean_margin = float("nan")
        pred_a = float("nan")
        pred_b = float("nan")
        pred_rest = float("nan")
        logger.warning(
            "No matrix pairing found for actual top-two %s vs %s",
            cand_a_key,
            cand_b_key,
        )

    margin_error = mean_margin - margin_actual if not np.isnan(mean_margin) else float("nan")

    ci_a_lower = float("nan")
    ci_a_upper = float("nan")
    ci_b_lower = float("nan")
    ci_b_upper = float("nan")
    ci_rest_lower = float("nan")
    ci_rest_upper = float("nan")
    std_a = float("nan")
    std_b = float("nan")
    std_rest = float("nan")
    a_in_ci = False
    b_in_ci = False
    rest_in_ci = False

    if idata_runoff is not None and "p_time" in idata_runoff.posterior:
        p_time = idata_runoff.posterior["p_time"]  # K=3 ordering: [cand_a, cand_b, rest_blanco]
        election_day = p_time[:, :, 0, :]
        a_values = election_day[:, :, 0].to_numpy().flatten()
        b_values = election_day[:, :, 1].to_numpy().flatten()
        rest_values = election_day[:, :, 2].to_numpy().flatten()

        ci_a = az.hdi(a_values, prob=0.95)  # type: ignore
        ci_b = az.hdi(b_values, prob=0.95)  # type: ignore
        ci_rest = az.hdi(rest_values, prob=0.95)  # type: ignore
        ci_a_lower = float(ci_a[0])  # type: ignore
        ci_a_upper = float(ci_a[1])  # type: ignore
        ci_b_lower = float(ci_b[0])  # type: ignore
        ci_b_upper = float(ci_b[1])  # type: ignore
        ci_rest_lower = float(ci_rest[0])  # type: ignore
        ci_rest_upper = float(ci_rest[1])  # type: ignore
        a_in_ci = bool(ci_a[0] <= actual_a <= ci_a[1])  # type: ignore
        b_in_ci = bool(ci_b[0] <= actual_b <= ci_b[1])  # type: ignore
        rest_in_ci = bool(ci_rest[0] <= actual_rest <= ci_rest[1])  # type: ignore
        std_a = float(a_values.std())
        std_b = float(b_values.std())
        std_rest = float(rest_values.std())

    candidates = [
        {
            "candidate_key": cand_a_key,
            "display_name": names_a,
            "actual": round(actual_a, 6),
            "predicted_mean": round(pred_a, 6),
            "predicted_median": round(float("nan"), 6),
            "error": round(pred_a - actual_a, 6) if np.isfinite(pred_a) else float("nan"),
            "abs_error": round(abs(pred_a - actual_a), 6) if np.isfinite(pred_a) else float("nan"),
            "within_95_ci": a_in_ci,
            "ci_95_lower": round(ci_a_lower, 6),
            "ci_95_upper": round(ci_a_upper, 6),
            "posterior_std": round(std_a, 6),
        },
        {
            "candidate_key": cand_b_key,
            "display_name": names_b,
            "actual": round(actual_b, 6),
            "predicted_mean": round(pred_b, 6),
            "predicted_median": round(float("nan"), 6),
            "error": round(pred_b - actual_b, 6) if np.isfinite(pred_b) else float("nan"),
            "abs_error": round(abs(pred_b - actual_b), 6) if np.isfinite(pred_b) else float("nan"),
            "within_95_ci": b_in_ci,
            "ci_95_lower": round(ci_b_lower, 6),
            "ci_95_upper": round(ci_b_upper, 6),
            "posterior_std": round(std_b, 6),
        },
        {
            "candidate_key": "rest",
            "display_name": names_rest,
            "actual": round(actual_rest, 6),
            "predicted_mean": round(pred_rest, 6),
            "predicted_median": round(float("nan"), 6),
            "error": round(pred_rest - actual_rest, 6) if np.isfinite(pred_rest) else float("nan"),
            "abs_error": round(abs(pred_rest - actual_rest), 6)
            if np.isfinite(pred_rest)
            else float("nan"),
            "within_95_ci": rest_in_ci,
            "ci_95_lower": round(ci_rest_lower, 6),
            "ci_95_upper": round(ci_rest_upper, 6),
            "posterior_std": round(std_rest, 6),
        },
    ]

    if np.isfinite(pred_a) and np.isfinite(pred_b) and np.isfinite(pred_rest):
        all_errors = [
            abs(pred_a - actual_a),
            abs(pred_b - actual_b),
            abs(pred_rest - actual_rest),
        ]
        mae = round(float(np.mean(all_errors)), 6)
        rmse = round(float(np.sqrt(np.mean(np.square(all_errors)))), 6)
        ci_coverage_95 = (
            (1 if a_in_ci else 0) + (1 if b_in_ci else 0) + (1 if rest_in_ci else 0)
        ) / 3.0
    else:
        mae = round(float("nan"), 6)
        rmse = round(float("nan"), 6)
        ci_coverage_95 = float("nan")

    return {
        "candidates": candidates,
        "cand_a_key": cand_a_key,
        "cand_b_key": cand_b_key,
        "mae": mae,
        "rmse": rmse,
        "ci_coverage_95": ci_coverage_95,
        "mean_margin": round(mean_margin, 6),
        "actual_margin": round(margin_actual, 6),
        "margin_error": round(margin_error, 6),
        "prob_a_wins": round(prob_a_wins, 6),
    }


def check_convergence(idata: xr.DataTree, label: str) -> tuple[float, bool]:
    """Check R-hat convergence.

    Args:
        idata: Posterior ``DataTree``.
        label: Human-readable label for logging.

    Returns:
        Tuple of ``(max_rhat, converged)``.

    """
    try:
        summary = az.summary(idata)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        max_rhat = float(summary["r_hat"].max())  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        converged = max_rhat < _RHAT_THRESHOLD
        logger.info("%s: max R-hat = %.4f (converged=%s)", label, max_rhat, converged)
    except (ValueError, TypeError, KeyError):
        max_rhat = float("nan")
        converged = False
        logger.warning("Could not compute R-hat for %s", label)
    return max_rhat, converged


def write_header(lines: list[str], config: ModelConfig, year: int = 2022) -> None:
    """Append report header lines."""
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"# {year} Backtesting Accuracy Report\n")
    lines.append(f"**Generated**: {now}\n")
    total = config.mcmc_draws * config.mcmc_chains
    lines.append(
        f"**Configuration**: {config.mcmc_draws} draws x {config.mcmc_chains} "
        f"chains = {total} total samples\n",
    )
    lines.append(
        f"**Seed**: {config.seed}  |  **Target accept**: {config.target_accept}\n",
    )


def write_data_summary(lines: list[str], data: ReportInputs) -> None:
    """Append data summary table."""
    lines.append("## Data Summary\n")
    lines.append("| Metric | Round 1 | Runoff |")
    lines.append("|--------|---------|--------|")
    lines.append(f"| Polls | {data.n_polls_r1} | {data.n_polls_r2} |")
    lines.append(f"| Sampling time | {data.elapsed_r1_s:.0f}s | {data.elapsed_r2_s:.0f}s |")
    r2_rhat_str = f"{data.r2_rhat:.4f}" if np.isfinite(data.r2_rhat) else "N/A"
    lines.append(f"| Max R-hat | {data.r1_rhat:.4f} | {r2_rhat_str} |")
    lines.append(
        "| Converged (R-hat < 1.10) |"
        f" {'Y' if data.r1_converged else 'N'} |"
        f" {'Y' if data.r2_converged else 'N'} |",
    )
    lines.append("")


def format_ci(lower: float, upper: float) -> str:
    """Format a 95% CI interval string, handling NaN values."""
    if not np.isfinite(lower) or not np.isfinite(upper):
        return "N/A"
    return f"[{lower * 100:.1f}%, {upper * 100:.1f}%]"


def write_candidate_table(lines: list[str], metrics: dict) -> None:
    """Append a candidate accuracy table from a metrics dict."""
    lines.append(
        "| Candidate | Actual | Predicted | Error | 95% CI | In CI? |",
    )
    lines.append(
        "|-----------|--------|-----------|-------|--------|--------|",
    )
    for cm in metrics["candidates"]:
        ci_lower = cm["ci_95_lower"]
        ci_upper = cm["ci_95_upper"]
        ci_str = format_ci(ci_lower, ci_upper)
        in_ci = "Y" if cm["within_95_ci"] else "N"
        lines.append(
            f"| {cm['display_name']} | {format_pct(cm['actual'])} |"
            f" {format_pct(cm['predicted_mean'])} |"
            f" {format_pp(cm['error'])} | {ci_str} | {in_ci} |",
        )
    lines.append("")

    lines.append("### Aggregate Metrics\n")
    mae_val = metrics["mae"]
    rmse_val = metrics["rmse"]
    mae_str = f"{mae_val * 100:.2f}pp" if np.isfinite(mae_val) else "N/A"
    rmse_str = f"{rmse_val * 100:.2f}pp" if np.isfinite(rmse_val) else "N/A"
    cov_val = metrics["ci_coverage_95"]
    cov_str = f"{cov_val * 100:.0f}%" if np.isfinite(cov_val) else "N/A"
    lines.append(f"- **MAE**: {mae_str}")
    lines.append(f"- **RMSE**: {rmse_str}")
    n_in_ci = sum(1 for cm in metrics["candidates"] if cm["within_95_ci"])
    lines.append(
        f"- **95% CI coverage**: {cov_str} ({n_in_ci}/{len(metrics['candidates'])} candidates)",
    )
    lines.append("")


def write_runoff_details(lines: list[str], r2_metrics: dict) -> None:
    """Append runoff-specific metrics beyond the candidate table."""
    mean_margin_val = r2_metrics["mean_margin"]
    actual_margin_val = r2_metrics["actual_margin"]
    margin_error_val = r2_metrics["margin_error"]
    prob_a_wins_val = r2_metrics["prob_a_wins"]

    if np.isfinite(mean_margin_val) and np.isfinite(actual_margin_val):
        lines.append(
            f"- **Predicted margin ({r2_metrics['cand_a_key']} -"
            f" {r2_metrics['cand_b_key']}):"
            f" {mean_margin_val * 100:+.2f}pp"
            f" (actual: {actual_margin_val * 100:+.2f}pp)"
            f" [error: {margin_error_val * 100:+.2f}pp]",
        )
    else:
        lines.append(
            f"- **Pairing ({r2_metrics['cand_a_key']} vs"
            f" {r2_metrics['cand_b_key']}) not found in matrix."
            f" Most likely pairings may differ from actual top-two.",
        )
    p_str = f"{prob_a_wins_val * 100:.1f}%" if np.isfinite(prob_a_wins_val) else "N/A"
    lines.append(
        f"- **P({r2_metrics['cand_a_key']} wins): {p_str}",
    )
    lines.append("")


def write_model_parameters_section(lines: list[str], r1_metrics: dict) -> None:
    """Append a section for learned model parameter posteriors."""
    ps = r1_metrics.get("posterior_summaries", {})

    phi_elec = ps.get("phi_elec", {})
    phi_digital = ps.get("phi_digital_base", {})
    internet_rate = ps.get("internet_rate")

    if not phi_elec and not phi_digital and internet_rate is None:
        return

    lines.append("## Model Parameters\n")

    if phi_elec:
        lines.append(
            "### φ_elec (concentration on election result)\n",
        )
        lines.append(
            f"- **Posterior mean**: {phi_elec['mean']:.1f}  **Std**: {phi_elec['std']:.1f}\n",
        )
        lines.append(
            f"- **95% HDI**: [{phi_elec['ci_95_lower']:.1f}, {phi_elec['ci_95_upper']:.1f}]\n",
        )
        lines.append(
            "- **Interpretation**: Higher values indicate the model trusts"
            " the election result more strongly. Values > 100,000 mean the"
            " concentration was fixed (Gamma prior not used).\n",
        )

    if phi_digital:
        lines.append(
            "### φ_digital_base (concentration on digital signals)\n",
        )
        lines.append(
            f"- **Posterior mean**: {phi_digital['mean']:.1f}  **Std**: {phi_digital['std']:.1f}\n",
        )
        lines.append(
            f"- **95% HDI**: [{phi_digital['ci_95_lower']:.1f}, "
            f"{phi_digital['ci_95_upper']:.1f}]\n",
        )
        lines.append(
            "- **Interpretation**: Higher values indicate the model trusts"
            " digital signals (Google Trends) more per observation.\n",
        )

    if internet_rate is not None:
        lines.append(
            "### Internet Access Rate (effective weight)\n",
        )
        lines.append(
            f"- **Population-weighted national rate**: {internet_rate:.2%}\n",
        )
        lines.append(
            "- This rate multiplies the φ_digital_base prior mean,"
            " reducing digital signal influence in low-connectivity"
            " regions.\n",
        )

    lines.append("")


def write_assessment(lines: list[str], data: ReportInputs) -> None:
    """Append overall assessment and interpretation."""
    lines.append("## Overall Assessment\n")

    if not data.r1_converged or not data.r2_converged:
        lines.append(
            "**WARNING**: Chains did not fully converge."
            " Results should be interpreted with caution.\n",
        )

    r1_mae_pp = data.r1_metrics["mae"] * 100
    r2_mae_finite = np.isfinite(data.r2_metrics["mae"])
    r2_mae_pp = data.r2_metrics["mae"] * 100 if r2_mae_finite else float("inf")
    r2_mae_str = f"{data.r2_metrics['mae'] * 100:.2f}pp" if r2_mae_finite else "N/A"
    r2_rmse_str = (
        f"{data.r2_metrics['rmse'] * 100:.2f}pp" if np.isfinite(data.r2_metrics["rmse"]) else "N/A"
    )

    lines.append("| Metric | Round 1 | Runoff |")
    lines.append("|--------|---------|--------|")
    lines.append(f"| MAE | {r1_mae_pp:.2f}pp | {r2_mae_str} |")
    lines.append(
        f"| RMSE | {data.r1_metrics['rmse'] * 100:.2f}pp | {r2_rmse_str} |",
    )
    c1 = "Y" if data.r1_converged else "N"
    c2 = "Y" if data.r2_converged else "N"
    lines.append(f"| Converged | {c1} | {c2} |")
    lines.append("")

    lines.append("### Interpretation\n")
    if r1_mae_pp < _MAE_TIGHT:
        lines.append(
            f"- **Round 1**: The model achieves MAE = {r1_mae_pp:.2f}pp,"
            f" well within the +/-5pp SPEC-06 acceptance target."
            f" The Dirichlet-Multinomial model with reverse-time random walk"
            f" and house effects captures the vote shares accurately.",
        )
    elif r1_mae_pp < _MAE_LOOSE:
        lines.append(
            f"- **Round 1**: The model achieves MAE = {r1_mae_pp:.2f}pp,"
            f" within the +/-5pp SPEC-06 acceptance target."
            f" Accuracy is acceptable but there is room for improvement.",
        )
    else:
        lines.append(
            f"- **Round 1**: MAE = {r1_mae_pp:.2f}pp, exceeding the +/-5pp"
            f" SPEC-06 target. Investigation needed into model specification"
            f" or data quality.",
        )

    if r2_mae_finite and r2_mae_pp < _MAE_TIGHT:
        lines.append(
            f"- **Runoff**: MAE = {r2_mae_pp:.2f}pp, within the +/-3pp"
            f" SPEC-07 target. The probabilistic pairing matrix with"
            f" data-driven transfer rates performs well.",
        )
    elif r2_mae_finite and r2_mae_pp < _MAE_LOOSE:
        lines.append(
            f"- **Runoff**: MAE = {r2_mae_pp:.2f}pp, moderately above the"
            f" +/-3pp SPEC-07 target. The model's accuracy is reasonable but"
            f" not tight.",
        )
    elif r2_mae_finite:
        lines.append(
            f"- **Runoff**: MAE = {r2_mae_pp:.2f}pp, exceeds the +/-3pp target."
            f" The matrix model may need tuning of the transfer rate priors.",
        )
    else:
        lines.append(
            "- **Runoff**: Metrics not available (pairing not found in"
            " matrix). The most likely pairing from the Round 1 posterior"
            " differs from the actual top-two, which is expected when the"
            " Round 1 model does not perfectly match election outcomes.",
        )

    lines.append("")


def generate_report(data: ReportInputs, config: ModelConfig) -> str:
    """Generate a Markdown accuracy report.

    Args:
        data: Grouped inputs for report generation.
        config: Model configuration (used for header metadata).

    Returns:
        Markdown-formatted report string.

    """
    lines: list[str] = []

    write_header(lines, config, year=data.election_year)
    write_data_summary(lines, data)

    lines.append(f"## First Round ({data.election_year} Election)\n")
    write_candidate_table(lines, data.r1_metrics)
    write_model_parameters_section(lines, data.r1_metrics)

    lines.append("## Runoff\n")
    write_candidate_table(lines, data.r2_metrics)
    write_runoff_details(lines, data.r2_metrics)

    write_assessment(lines, data)

    return "\n".join(lines)


def _extract_posterior_summaries(
    idata_r1: xr.DataTree,
    features: pd.DataFrame | None,
) -> dict:
    """Extract model parameter posteriors (phi_elec, phi_digital_base) for report.

    Args:
        idata_r1: Round 1 posterior ``DataTree``.
        features: Optional municipal feature matrix.

    Returns:
        Dict with optional ``phi_elec``, ``phi_digital_base``, and
        ``internet_rate`` keys.

    """
    summaries: dict = {}

    try:
        phi_elec_samples = idata_r1.posterior["phi_elec"].to_numpy().flatten()
        ci = az.hdi(phi_elec_samples, prob=0.95)
        summaries["phi_elec"] = {
            "mean": round(float(phi_elec_samples.mean()), 1),
            "std": round(float(phi_elec_samples.std()), 1),
            "ci_95_lower": round(float(ci[0]), 1),
            "ci_95_upper": round(float(ci[1]), 1),
        }
    except (KeyError, ValueError, TypeError):
        summaries["phi_elec"] = {}

    try:
        phi_digital_samples = idata_r1.posterior["phi_digital_base"].to_numpy().flatten()
        ci = az.hdi(phi_digital_samples, prob=0.95)
        summaries["phi_digital_base"] = {
            "mean": round(float(phi_digital_samples.mean()), 1),
            "std": round(float(phi_digital_samples.std()), 1),
            "ci_95_lower": round(float(ci[0]), 1),
            "ci_95_upper": round(float(ci[1]), 1),
        }
    except (KeyError, ValueError, TypeError):
        summaries["phi_digital_base"] = {}

    if features is not None and not features.empty:
        from co_president.model_round1 import _compute_national_internet_rate  # noqa: PLC0415

        summaries["internet_rate"] = round(
            _compute_national_internet_rate(features),
            4,
        )
    else:
        summaries["internet_rate"] = None

    return summaries


def run_round1_survey(
    config: ModelConfig,
    clean_polls: CleanPolls,
    results_r1: RoundResult,
    digital_signals: pd.DataFrame,
    features: pd.DataFrame | None = None,
    results_dir: Path = Path("results"),
    year: int = 2022,
) -> tuple:
    """Build, sample, and compute metrics for round 1.

    Uses ``build_round1_model`` with election-result likelihood for
    maximum backtest accuracy.

    Args:
        config: Model configuration.
        clean_polls: Cleaned poll data.
        results_r1: Canonical round 1 results.
        features: Optional municipal features matrix.
        digital_signals: Google Trends data.
        results_dir: Directory for saving trace files.
        year: Election year (default 2022).

    Returns:
        Tuple of ``(idata, metrics, rhat, converged, elapsed_s)``.

    """
    import co_president.model_round1 as m1  # noqa: PLC0415

    candidate_registry = {c.key: c for c in get_active_candidates(1, year=year)}

    logger.info("Building round 1 model with election likelihood...")
    model_r1 = m1.build_round1_model(
        clean_polls.round1,
        results_r1,
        config,
        features=features,
        digital_signals=digital_signals,
        year=year,
    )

    logger.info(
        "Sampling round 1 model (%d draws x %d chains, this will take a while)...",
        config.mcmc_draws,
        config.mcmc_chains,
    )
    t0 = datetime.now(UTC)
    idata_r1 = m1.sample_round1(model_r1, config)
    elapsed_r1_s = (datetime.now(UTC) - t0).total_seconds()
    logger.info("Round 1 sampling complete in %.0fs", elapsed_r1_s)

    try:
        ess_bulk = az.ess(idata_r1, method="bulk")
        ess_tail = az.ess(idata_r1, method="tail")
        min_ess_bulk = min(float(v.min()) for v in ess_bulk.values())
        min_ess_tail = min(float(v.min()) for v in ess_tail.values())
        logger.info(
            "Round 1 ESS: bulk=%s, tail=%s",
            f"{min_ess_bulk:.0f}",
            f"{min_ess_tail:.0f}",
        )
    except (ValueError, TypeError):
        logger.warning("Could not compute ESS for Round 1")

    try:
        idata_r1.to_netcdf(str(results_dir / "round1_trace.nc"))  # pyright: ignore[reportUnknownMemberType]
        logger.info("Saved round 1 trace (netcdf)")
    except (ValueError, ImportError):
        logger.info("Skipping round 1 trace save (no netcdf/zarr backend)")

    r1_rhat, r1_converged = check_convergence(idata_r1, "Round 1")
    r1_metrics = compute_r1_accuracy(
        idata_r1,
        results_r1,
        poll_columns=set(clean_polls.round1.columns),
        candidate_registry=candidate_registry,
    )
    r1_metrics["posterior_summaries"] = _extract_posterior_summaries(
        idata_r1,
        features,
    )

    return idata_r1, r1_metrics, r1_rhat, r1_converged, elapsed_r1_s


def run_round2_survey(  # noqa: PLR0913
    config: ModelConfig,
    clean_polls: CleanPolls,
    results_r1: RoundResult,
    results_r2: RoundResult,
    idata_r1: xr.DataTree,
    digital_signals: pd.DataFrame,
    features: pd.DataFrame | None = None,
    year: int = 2022,
) -> tuple:
    """Compute runoff metrics via the probabilistic pairing matrix.

    Runs :func:`~co_president.model_runoff_matrix.estimate_runoff_matrix`
    which combines the Round 1 posterior, head-to-head polls, and data-driven
    transfer rates to estimate each candidate pairing's win probability.

    Args:
        config: Model configuration.
        clean_polls: Cleaned poll data.
        results_r1: Canonical round 1 results.
        results_r2: Canonical round 2 results.
        idata_r1: Round 1 posterior ``DataTree``.
        features: Municipal features for on-demand transfer model training.
        digital_signals: Google Trends data for the runoff K=3 poll
            likelihood.
        year: Election year (default 2022).

    Returns:
        Tuple of ``(idata_runoff, metrics, rhat, converged, elapsed_s)``.

    """
    from co_president.model_runoff_matrix import estimate_runoff_matrix  # noqa: PLC0415

    candidate_registry = {c.key: c for c in get_active_candidates(1, year=year)}
    candidate_keys = sorted(
        set(candidate_registry.keys()) & set(clean_polls.round1.columns),
    )

    logger.info("Computing runoff pairing matrix...")
    t0 = datetime.now(UTC)
    matrix = estimate_runoff_matrix(
        idata_r1,
        (results_r1, results_r2),
        clean_polls.round2,
        config,
        candidate_keys=candidate_keys,
        features=features,
        digital_signals=digital_signals,
        year=year,
    )
    elapsed_r2_s = (datetime.now(UTC) - t0).total_seconds()
    logger.info("Runoff matrix computed in %.0fs", elapsed_r2_s)

    top_two = results_r1.top_two()
    cand_a_key = top_two[0].candidate_key
    cand_b_key = top_two[1].candidate_key

    idata_runoff: xr.DataTree | None = None
    for pf in matrix.pairings:
        if pf.candidate_first == cand_a_key and pf.candidate_second == cand_b_key:
            idata_runoff = pf.idata
            break

    r2_rhat = float("nan")
    r2_converged = False
    if idata_runoff is not None:
        r2_rhat, r2_converged = check_convergence(idata_runoff, "Runoff")
        try:
            ess_bulk = az.ess(idata_runoff, method="bulk")
            ess_tail = az.ess(idata_runoff, method="tail")
            min_ess_bulk = min(float(v.min()) for v in ess_bulk.values())
            min_ess_tail = min(float(v.min()) for v in ess_tail.values())
            logger.info(
                "Runoff ESS: bulk=%s, tail=%s",
                f"{min_ess_bulk:.0f}",
                f"{min_ess_tail:.0f}",
            )
        except (ValueError, TypeError):
            logger.warning("Could not compute ESS for Runoff")

    r2_metrics = compute_r2_accuracy_from_matrix(
        matrix,
        results_r1,
        results_r2,
        idata_runoff=idata_runoff,
        candidate_registry=candidate_registry,
    )

    return idata_runoff, r2_metrics, r2_rhat, r2_converged, elapsed_r2_s


def save_reports(
    r1_metrics: dict,
    r2_metrics: dict,
    total_elapsed: float,
    n_polls_r1: int,
    n_polls_r2: int,
    config: ModelConfig,
    report_path: Path,
    json_path: Path,
    year: int = 2022,
) -> None:
    """Write JSON report to disk.

    Args:
        r1_metrics: Round 1 metrics dict.
        r2_metrics: Runoff metrics dict.
        total_elapsed: Total wall-clock time in seconds.
        n_polls_r1: Number of round 1 polls.
        n_polls_r2: Number of round 2 polls.
        config: Model configuration.
        report_path: Reserved for future Markdown output.
        json_path: Path for the JSON report.
        year: Election year.

    """
    now_str = datetime.now(UTC).isoformat()
    json_data = {
        "generated_at": now_str,
        "config": {
            "mcmc_draws": config.mcmc_draws,
            "mcmc_tune": config.mcmc_tune,
            "mcmc_chains": config.mcmc_chains,
            "seed": config.seed,
            "target_accept": config.target_accept,
        },
        "data": {
            "n_polls_round1": n_polls_r1,
            "n_polls_round2": n_polls_r2,
        },
        "total_time_s": total_elapsed,
        "election_year": year,
    }
    json_path.write_text(json.dumps(json_data, indent=2, default=str), encoding="utf-8")
    logger.info("Saved JSON report to %s", json_path)


def print_summary(
    r1_metrics: dict,
    r2_metrics: dict,
    total_elapsed: float,
    report_path: Path,
    json_path: Path,
    *,
    r1_converged: bool,
    r2_converged: bool,
) -> None:
    """Print a human-readable accuracy summary to stdout."""
    lines: list[str] = []
    lines.append("\n" + "=" * 70)
    lines.append("  BACKTEST ACCURACY REPORT SUMMARY")
    lines.append("=" * 70)
    lines.append(
        f"\n  Round 1  (MAE: {r1_metrics['mae'] * 100:.2f}pp,"
        f"  RMSE: {r1_metrics['rmse'] * 100:.2f}pp,"
        f"  Converged: {'Y' if r1_converged else 'N'})",
    )
    for cm in r1_metrics["candidates"]:
        ci = "Y" if cm["within_95_ci"] else "N"
        lines.append(
            f"    {cm['display_name']:25s}"
            f"  pred={cm['predicted_mean'] * 100:6.2f}%"
            f"  actual={cm['actual'] * 100:6.2f}%"
            f"  error={cm['error'] * 100:+6.2f}pp"
            f"  95%CI={ci}",
        )
    lines.append(
        f"\n  Runoff   (MAE: {r2_metrics['mae'] * 100:.2f}pp,"
        f"  Converged: {'Y' if r2_converged else 'N'})",
    )
    for cm in r2_metrics["candidates"]:
        ci = "Y" if cm["within_95_ci"] else "N"
        lines.append(
            f"    {cm['display_name']:25s}"
            f"  pred={cm['predicted_mean'] * 100:6.2f}%"
            f"  actual={cm['actual'] * 100:6.2f}%"
            f"  error={cm['error'] * 100:+6.2f}pp"
            f"  95%CI={ci}",
        )
    lines.append(f"\n  Full report: {report_path}")
    lines.append(f"  JSON report: {json_path}")
    lines.append(f"  Total time:  {total_elapsed:.0f}s")
    lines.append("=" * 70)
    sys.stdout.write("\n".join(lines))
