"""Run 100K-iteration MCMC for both rounds and produce an accuracy report.

Samples the Round 1 (Dirichlet-Multinomial) and Runoff (K=3) models with
100,000 posterior draws each, then compares posterior means against the
canonical 2022 election results.
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

from co_president.config import (
    FIRST_ROUND_CANDIDATES,
    ModelConfig,
)
from co_president.data import load_and_clean_all, load_canonical_results

if TYPE_CHECKING:
    import xarray as xr

    from co_president.data import CleanPolls, RoundResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_100k_report")

_RHAT_THRESHOLD = 1.10
_MAE_TIGHT = 3.0
_MAE_LOOSE = 5.0

CONFIG_100K = ModelConfig(
    mcmc_draws=100_000,
    mcmc_tune=5_000,
    mcmc_chains=4,
    mcmc_cores=4,
    target_accept=0.95,
    seed=332211,
)

RESULTS_DIR = Path("results")
REPORT_PATH = Path("results/100k_accuracy_report.md")
JSON_PATH = Path("results/100k_accuracy_report.json")


@dataclass
class _ReportInputs:
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


def _format_pct(value: float) -> str:
    """Format a proportion as a percentage string."""
    return f"{value * 100:.2f}%"


def _format_pp(error: float) -> str:
    """Format an error with sign and pp suffix."""
    return f"{error * 100:+.2f}pp"


def _hdi_95(samples: np.ndarray) -> tuple[float, float]:
    """Compute 95% highest-density interval for a 1-D array of samples.

    Args:
        samples: 1-D numpy array of posterior draws.

    Returns:
        Tuple of ``(lower, upper)`` bounds.

    """
    return az.hdi(samples, prob=0.95)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType, reportReturnType]


def _extract_election_day_shares(
    idata: xr.DataTree,
    candidate_keys: list[str],
) -> dict[str, np.ndarray]:
    """Extract election-day (t=0) posterior shares from DataTree.

    Args:
        idata: Posterior ``DataTree`` from a round model.
        candidate_keys: Candidate column keys in order.

    Returns:
        Mapping of candidate key to 1-D numpy array of posterior shares.

    """
    p_time = idata.posterior["p_time"]
    elec_shares = p_time[:, :, 0, :].values
    n_chains, n_draws, n_candidates = elec_shares.shape
    flat = elec_shares.reshape(n_chains * n_draws, n_candidates)
    return {key: flat[:, i] for i, key in enumerate(candidate_keys)}


def _compute_r1_accuracy(
    idata_r1: xr.DataTree,
    results_r1: RoundResult,
) -> dict:
    """Compute round 1 accuracy metrics.

    Args:
        idata_r1: Round 1 posterior ``DataTree``.
        results_r1: Canonical round 1 ``RoundResult``.

    Returns:
        Dict with per-candidate metrics and aggregate scores.

    """
    candidate_keys = sorted(
        set(FIRST_ROUND_CANDIDATES.keys()) & {c.candidate_key for c in results_r1.candidates},
    )
    shares = _extract_election_day_shares(idata_r1, candidate_keys)

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

        candidates_metrics.append(
            {
                "candidate_key": key,
                "display_name": FIRST_ROUND_CANDIDATES[key].display_name,
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

    mae = float(np.mean(all_errors))
    rmse = float(np.sqrt(np.mean(np.square(all_errors))))
    ci_coverage_95 = sum(1 for cm in candidates_metrics if cm["within_95_ci"]) / len(
        candidates_metrics,
    )

    return {
        "candidates": candidates_metrics,
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "ci_coverage_95": round(ci_coverage_95, 6),
    }


def _compute_r2_accuracy(
    idata_runoff: xr.DataTree,
    results_r1: RoundResult,
    results_r2: RoundResult,
) -> dict:
    """Compute runoff (round 2) accuracy metrics.

    Args:
        idata_runoff: Runoff posterior ``DataTree``.
        results_r1: Round 1 ``RoundResult`` (used to identify top-two).
        results_r2: Round 2 ``RoundResult``.

    Returns:
        Dict with candidate metrics and aggregate scores.

    """
    top_two = results_r1.top_two()
    cand_a_key = top_two[0].candidate_key
    cand_b_key = top_two[1].candidate_key

    p_time = idata_runoff.posterior["p_time"]
    elec_shares = p_time[:, :, 0, :].values
    n_chains, n_draws, k3 = elec_shares.shape
    flat = elec_shares.reshape(n_chains * n_draws, k3)

    share_a = flat[:, 0]
    share_b = flat[:, 1]

    try:
        actual_a = results_r2.get_share(cand_a_key)
    except KeyError:
        actual_a = 0.0
    try:
        actual_b = results_r2.get_share(cand_b_key)
    except KeyError:
        actual_b = 0.0

    names_a = FIRST_ROUND_CANDIDATES[cand_a_key].display_name
    names_b = FIRST_ROUND_CANDIDATES.get(
        cand_b_key,
        None,
    )
    names_b = names_b.display_name if names_b else cand_b_key

    mean_a = float(share_a.mean())
    mean_b = float(share_b.mean())
    error_a = mean_a - actual_a
    error_b = mean_b - actual_b
    margin_actual = actual_a - actual_b
    mean_margin = mean_a - mean_b
    ci_a = _hdi_95(share_a)
    ci_b = _hdi_95(share_b)
    prob_a_wins = float((share_a > share_b).mean())

    candidates = [
        {
            "candidate_key": cand_a_key,
            "display_name": names_a,
            "actual": round(actual_a, 6),
            "predicted_mean": round(mean_a, 6),
            "predicted_median": round(float(np.median(share_a)), 6),
            "error": round(error_a, 6),
            "abs_error": round(abs(error_a), 6),
            "within_95_ci": bool(ci_a[0] <= actual_a <= ci_a[1]),
            "ci_95_lower": round(ci_a[0], 6),
            "ci_95_upper": round(ci_a[1], 6),
            "posterior_std": round(float(share_a.std()), 6),
        },
        {
            "candidate_key": cand_b_key,
            "display_name": names_b,
            "actual": round(actual_b, 6),
            "predicted_mean": round(mean_b, 6),
            "predicted_median": round(float(np.median(share_b)), 6),
            "error": round(error_b, 6),
            "abs_error": round(abs(error_b), 6),
            "within_95_ci": bool(ci_b[0] <= actual_b <= ci_b[1]),
            "ci_95_lower": round(ci_b[0], 6),
            "ci_95_upper": round(ci_b[1], 6),
            "posterior_std": round(float(share_b.std()), 6),
        },
    ]

    mae = float(np.mean([abs(error_a), abs(error_b)]))
    rmse = float(np.sqrt(np.mean([abs(error_a) ** 2, abs(error_b) ** 2])))

    return {
        "candidates": candidates,
        "cand_a_key": cand_a_key,
        "cand_b_key": cand_b_key,
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "mean_margin": round(mean_margin, 6),
        "actual_margin": round(margin_actual, 6),
        "margin_error": round(mean_margin - margin_actual, 6),
        "prob_a_wins": round(prob_a_wins, 6),
    }


def _check_convergence(idata: xr.DataTree, label: str) -> tuple[float, bool]:
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


def _write_header(lines: list[str]) -> None:
    """Append report header lines."""
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    lines.append("# 2022 Backtesting Accuracy Report - 100K Iterations\n")
    lines.append(f"**Generated**: {now}\n")
    lines.append("**Configuration**: 100,000 draws x 4 chains = 400,000 total samples\n")
    lines.append(
        f"**Seed**: {CONFIG_100K.seed}  |  **Target accept**: {CONFIG_100K.target_accept}\n",
    )


def _write_data_summary(lines: list[str], data: _ReportInputs) -> None:
    """Append data summary table."""
    lines.append("## Data Summary\n")
    lines.append("| Metric | Round 1 | Runoff |")
    lines.append("|--------|---------|--------|")
    lines.append(f"| Polls | {data.n_polls_r1} | {data.n_polls_r2} |")
    lines.append(f"| Sampling time | {data.elapsed_r1_s:.0f}s | {data.elapsed_r2_s:.0f}s |")
    lines.append(f"| Max R-hat | {data.r1_rhat:.4f} | {data.r2_rhat:.4f} |")
    lines.append(
        "| Converged (R-hat < 1.10) |"
        f" {'Y' if data.r1_converged else 'N'} |"
        f" {'Y' if data.r2_converged else 'N'} |",
    )
    lines.append("")


def _write_candidate_table(lines: list[str], metrics: dict) -> None:
    """Append a candidate accuracy table from a metrics dict."""
    lines.append(
        "| Candidate | Actual | Predicted | Error | 95% CI | In CI? |",
    )
    lines.append(
        "|-----------|--------|-----------|-------|--------|--------|",
    )
    for cm in metrics["candidates"]:
        ci_str = f"[{cm['ci_95_lower'] * 100:.1f}%, {cm['ci_95_upper'] * 100:.1f}%]"
        in_ci = "Y" if cm["within_95_ci"] else "N"
        lines.append(
            f"| {cm['display_name']} | {_format_pct(cm['actual'])} |"
            f" {_format_pct(cm['predicted_mean'])} |"
            f" {_format_pp(cm['error'])} | {ci_str} | {in_ci} |",
        )
    lines.append("")

    lines.append("### Aggregate Metrics\n")
    lines.append(f"- **MAE**: {metrics['mae'] * 100:.2f}pp")
    lines.append(f"- **RMSE**: {metrics['rmse'] * 100:.2f}pp")
    lines.append(
        f"- **95% CI coverage**: {metrics['ci_coverage_95'] * 100:.0f}%"
        f" ({sum(1 for cm in metrics['candidates'] if cm['within_95_ci'])}"
        f"/{len(metrics['candidates'])} candidates)",
    )
    lines.append("")


def _write_runoff_details(lines: list[str], r2_metrics: dict) -> None:
    """Append runoff-specific metrics beyond the candidate table."""
    lines.append(
        f"- **Predicted margin ({r2_metrics['cand_a_key']} -"
        f" {r2_metrics['cand_b_key']}):"
        f" {r2_metrics['mean_margin'] * 100:+.2f}pp"
        f" (actual: {r2_metrics['actual_margin'] * 100:+.2f}pp)"
        f" [error: {r2_metrics['margin_error'] * 100:+.2f}pp]",
    )
    lines.append(
        f"- **P({r2_metrics['cand_a_key']} wins): {r2_metrics['prob_a_wins'] * 100:.1f}%",
    )
    lines.append("")


def _write_assessment(lines: list[str], data: _ReportInputs) -> None:
    """Append overall assessment and interpretation."""
    lines.append("## Overall Assessment\n")

    if not data.r1_converged or not data.r2_converged:
        lines.append(
            "**WARNING**: Chains did not fully converge."
            " Results should be interpreted with caution.\n",
        )

    r1_mae_pp = data.r1_metrics["mae"] * 100
    r2_mae_pp = data.r2_metrics["mae"] * 100

    lines.append("| Metric | Round 1 | Runoff |")
    lines.append("|--------|---------|--------|")
    lines.append(f"| MAE | {r1_mae_pp:.2f}pp | {r2_mae_pp:.2f}pp |")
    lines.append(
        f"| RMSE | {data.r1_metrics['rmse'] * 100:.2f}pp | {data.r2_metrics['rmse'] * 100:.2f}pp |",
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

    if r2_mae_pp < _MAE_TIGHT:
        lines.append(
            f"- **Runoff**: MAE = {r2_mae_pp:.2f}pp, within the +/-3pp"
            f" SPEC-07 target. The K=3 Dirichlet runoff model with"
            f" Round-1-informed prior performs well.",
        )
    elif r2_mae_pp < _MAE_LOOSE:
        lines.append(
            f"- **Runoff**: MAE = {r2_mae_pp:.2f}pp, moderately above the"
            f" +/-3pp SPEC-07 target. The model's accuracy is reasonable but"
            f" not tight.",
        )
    else:
        lines.append(
            f"- **Runoff**: MAE = {r2_mae_pp:.2f}pp, exceeds the +/-3pp target."
            f" The K=3 model may need tuning of the prior from Round 1.",
        )

    lines.append("")


def _generate_report(data: _ReportInputs) -> str:
    """Generate a Markdown accuracy report.

    Args:
        data: Grouped inputs for report generation.

    Returns:
        Markdown-formatted report string.

    """
    lines: list[str] = []

    _write_header(lines)
    _write_data_summary(lines, data)

    lines.append("## First Round (May 29, 2022)\n")
    _write_candidate_table(lines, data.r1_metrics)

    lines.append("## Runoff (June 19, 2022)\n")
    _write_candidate_table(lines, data.r2_metrics)
    _write_runoff_details(lines, data.r2_metrics)

    _write_assessment(lines, data)

    return "\n".join(lines)


def _run_round1_survey(
    config: ModelConfig,
    clean_polls: CleanPolls,
    results_r1: RoundResult,
) -> tuple:
    """Build, sample, and compute metrics for round 1.

    Args:
        config: Model configuration.
        clean_polls: Cleaned poll data.
        results_r1: Canonical round 1 results.

    Returns:
        Tuple of ``(idata, metrics, rhat, converged, elapsed_s)``.

    """
    import co_president.model_round1 as m1  # noqa: PLC0415

    model_r1 = m1.build_round1_model(clean_polls.round1, results_r1, config)

    logger.info("Sampling round 1 model (this will take a while)...")
    t0 = datetime.now(UTC)
    idata_r1 = m1.sample_round1(model_r1, config)
    elapsed_r1_s = (datetime.now(UTC) - t0).total_seconds()
    logger.info("Round 1 sampling complete in %.0fs", elapsed_r1_s)

    try:
        idata_r1.to_netcdf(str(RESULTS_DIR / "round1_trace_100k.nc"))  # pyright: ignore[reportUnknownMemberType]
        logger.info("Saved round 1 trace (netcdf)")
    except (ValueError, ImportError):
        logger.info("Skipping round 1 trace save (no netcdf/zarr backend)")

    r1_rhat, r1_converged = _check_convergence(idata_r1, "Round 1")
    r1_metrics = _compute_r1_accuracy(idata_r1, results_r1)

    return idata_r1, r1_metrics, r1_rhat, r1_converged, elapsed_r1_s


def _run_round2_survey(
    config: ModelConfig,
    clean_polls: CleanPolls,
    results_r1: RoundResult,
    results_r2: RoundResult,
    idata_r1: xr.DataTree,
) -> tuple:
    """Build, sample, and compute metrics for the runoff.

    Args:
        config: Model configuration.
        clean_polls: Cleaned poll data.
        results_r1: Canonical round 1 results.
        results_r2: Canonical round 2 results.
        idata_r1: Round 1 posterior ``DataTree``.

    Returns:
        Tuple of ``(idata, metrics, rhat, converged, elapsed_s)``.

    """
    import co_president.model_runoff_simple as mr  # noqa: PLC0415

    model_runoff = mr.build_runoff_simple_model(
        clean_polls.round2,
        results_r1,
        idata_r1,
        config,
    )

    logger.info("Sampling runoff model (this will take a while)...")
    t0 = datetime.now(UTC)
    idata_runoff = mr.sample_runoff(model_runoff, config)
    elapsed_r2_s = (datetime.now(UTC) - t0).total_seconds()
    logger.info("Runoff sampling complete in %.0fs", elapsed_r2_s)

    try:
        idata_runoff.to_netcdf(str(RESULTS_DIR / "runoff_trace_100k.nc"))  # pyright: ignore[reportUnknownMemberType]
        logger.info("Saved runoff trace (netcdf)")
    except (ValueError, ImportError):
        logger.info("Skipping runoff trace save (no netcdf/zarr backend)")

    r2_rhat, r2_converged = _check_convergence(idata_runoff, "Runoff")
    r2_metrics = _compute_r2_accuracy(idata_runoff, results_r1, results_r2)

    return idata_runoff, r2_metrics, r2_rhat, r2_converged, elapsed_r2_s


def _save_reports(
    total_elapsed: float,
    n_polls_r1: int,
    n_polls_r2: int,
) -> None:
    """Write markdown and JSON reports to disk.

    Args:
        total_elapsed: Total wall-clock time in seconds.
        n_polls_r1: Number of round 1 polls.
        n_polls_r2: Number of round 2 polls.

    """
    now_str = datetime.now(UTC).isoformat()
    json_data = {
        "generated_at": now_str,
        "config": {
            "mcmc_draws": CONFIG_100K.mcmc_draws,
            "mcmc_tune": CONFIG_100K.mcmc_tune,
            "mcmc_chains": CONFIG_100K.mcmc_chains,
            "seed": CONFIG_100K.seed,
            "target_accept": CONFIG_100K.target_accept,
        },
        "data": {
            "n_polls_round1": n_polls_r1,
            "n_polls_round2": n_polls_r2,
        },
        "total_time_s": total_elapsed,
    }
    JSON_PATH.write_text(json.dumps(json_data, indent=2, default=str), encoding="utf-8")
    logger.info("Saved JSON report to %s", JSON_PATH)


def _print_summary(
    r1_metrics: dict,
    r2_metrics: dict,
    total_elapsed: float,
    *,
    r1_converged: bool,
    r2_converged: bool,
) -> None:
    """Print a human-readable accuracy summary to stdout."""
    lines: list[str] = []
    lines.append("\n" + "=" * 70)
    lines.append("  100K ITERATION ACCURACY REPORT SUMMARY")
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
    lines.append(f"\n  Full report: {REPORT_PATH}")
    lines.append(f"  JSON report: {JSON_PATH}")
    lines.append(f"  Total time:  {total_elapsed:.0f}s")
    lines.append("=" * 70)
    sys.stdout.write("\n".join(lines))


def main() -> None:
    """Run 100K-iteration MCMC for both rounds and produce accuracy report."""
    started_at = datetime.now(UTC)
    logger.info("Starting 100K-iteration accuracy benchmark...")

    logger.info("Loading canonical election results...")
    try:
        results_r1, _results_r2 = load_canonical_results()
    except (OSError, ValueError):
        logger.exception("Failed to load election results")
        sys.exit(1)

    logger.info("Loading and cleaning poll data...")
    try:
        clean_polls: CleanPolls = load_and_clean_all()
    except (OSError, ValueError):
        logger.exception("Failed to load poll data")
        sys.exit(1)

    n_polls_r1 = len(clean_polls.round1)
    n_polls_r2 = len(clean_polls.round2)
    logger.info(
        "Loaded %d round-1 polls, %d runoff polls",
        n_polls_r1,
        n_polls_r2,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    idata_r1, r1_metrics, r1_rhat, r1_converged, elapsed_r1_s = _run_round1_survey(
        CONFIG_100K,
        clean_polls,
        results_r1,
    )

    _idata_runoff, r2_metrics, r2_rhat, r2_converged, elapsed_r2_s = _run_round2_survey(
        CONFIG_100K,
        clean_polls,
        results_r1,
        _results_r2,
        idata_r1,
    )

    report_data = _ReportInputs(
        r1_metrics=r1_metrics,
        r2_metrics=r2_metrics,
        r1_rhat=r1_rhat,
        r1_converged=r1_converged,
        r2_rhat=r2_rhat,
        r2_converged=r2_converged,
        elapsed_r1_s=elapsed_r1_s,
        elapsed_r2_s=elapsed_r2_s,
        n_polls_r1=n_polls_r1,
        n_polls_r2=n_polls_r2,
    )

    report_md = _generate_report(report_data)
    REPORT_PATH.write_text(report_md, encoding="utf-8")
    logger.info("Saved report to %s", REPORT_PATH)

    total_elapsed = (datetime.now(UTC) - started_at).total_seconds()
    _save_reports(total_elapsed, n_polls_r1, n_polls_r2)
    _print_summary(
        r1_metrics,
        r2_metrics,
        total_elapsed,
        r1_converged=r1_converged,
        r2_converged=r2_converged,
    )


if __name__ == "__main__":
    main()
