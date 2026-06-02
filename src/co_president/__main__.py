"""SPEC-10: CLI entry point for co-president.

Provides ``python -m co_president <command>`` subcommands for running the
full pipeline, generating reports, and producing plots.

Commands:
    run         Run the full pipeline (or ``--no-sample`` for baseline only)
    validate    Load saved ``InferenceData`` and print validation metrics
    aggregate   Print baseline weighted polling averages (SPEC-05)
    plot        Generate all visualization plots
    config      Print current configuration
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import logging
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any

import arviz as az  # type: ignore[reportMissingTypeStubs]

if TYPE_CHECKING:
    from datetime import date

    import pandas as pd  # type: ignore[reportMissingTypeStubs]

from co_president.config import (
    CONSULTATION_DATE,
    ELECTION_DATE_ROUND1,
    ELECTION_DATE_ROUND2,
    FIRST_ROUND_CANDIDATES,
    POLLSTER_RATINGS,
    ModelConfig,
    get_active_candidates,
)

if TYPE_CHECKING:
    from co_president.config import Candidate
    from co_president.data import CleanPolls, RoundResult
    from co_president.model_round1 import Round1Forecast
    from co_president.model_runoff_matrix import PairingForecast, RunoffMatrix
    from co_president.model_runoff_simple import RunoffForecast
    from co_president.validation import (
        RoundValidation,
    )

logger = logging.getLogger(__name__)

# ruff: noqa: T201 — CLI output uses print() for user-facing text

_MIN_POLLSTERS_R1 = 5
_MIN_POLLSTERS_R2 = 2
_MAX_SHARE_PCT = 100.0
_RHAT_LIMIT = 1.10
_SHARE_TOLERANCE = 0.02


# ═══════════════════════════════════════════════════════════════════════
# Argument parser
# ═══════════════════════════════════════════════════════════════════════


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all subcommands.

    Returns:
        Configured ``ArgumentParser``.

    """
    parser = argparse.ArgumentParser(
        description="co-president 2026 — Bayesian election forecast",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    run_parser = subparsers.add_parser("run", help="Run the full pipeline")
    run_parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Skip MCMC sampling; print baseline weighted averages instead",
    )
    run_parser.add_argument(
        "--config-override",
        nargs="+",
        default=[],
        help="Override ModelConfig fields (KEY=VALUE KEY=VALUE ...)",
    )

    subparsers.add_parser(
        "validate",
        help="Load saved InferenceData and print validation metrics",
    )

    subparsers.add_parser(
        "aggregate",
        help="Print baseline weighted polling averages",
    )

    plot_parser = subparsers.add_parser("plot", help="Generate visualization plots")
    plot_parser.add_argument(
        "--output-dir",
        type=str,
        default="results/",
        help="Directory to save plots (default: results/)",
    )

    download_cedae_parser = subparsers.add_parser(
        "download-cedae",
        help="Download CEDAE election-result CSVs from S3",
    )
    download_cedae_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Target directory (default: data/raw/cedae/)",
    )
    download_cedae_parser.add_argument(
        "--years",
        type=str,
        default="2002,2006,2010,2014,2018",
        help="Comma-separated election years (default: 2002,2006,2010,2014,2018)",
    )
    download_cedae_parser.add_argument(
        "--niveles",
        type=str,
        default="Camara,Presidencia,Senado",
        help="Comma-separated niveles (default: Camara,Presidencia,Senado)",
    )

    subparsers.add_parser("config", help="Print current configuration")

    return parser


# ═══════════════════════════════════════════════════════════════════════
# Config override parsing
# ═══════════════════════════════════════════════════════════════════════


def _parse_config_overrides(overrides: list[str]) -> dict[str, Any]:
    """Parse ``--config-override KEY=VALUE ...`` into a typed dict.

    Args:
        overrides: List of ``KEY=VALUE`` strings.

    Returns:
        Dict mapping field names to typed values.

    Raises:
        ValueError: If a key is not a valid ``ModelConfig`` field or the
            value cannot be cast.

    """
    valid_fields = {f.name: f.type for f in fields(ModelConfig)}
    result: dict[str, Any] = {}
    for override in overrides:
        if "=" not in override:
            msg = f"Invalid override format: {override!r} (expected KEY=VALUE)"
            raise ValueError(msg)
        key, value_str = override.split("=", 1)
        if key not in valid_fields:
            available = ", ".join(sorted(valid_fields))
            msg = f"Unknown config field: {key!r}. Available fields: {available}"
            raise ValueError(msg)
        expected_type = valid_fields[key]
        expected_type_name = (
            expected_type if isinstance(expected_type, str) else expected_type.__name__
        )
        if expected_type_name == "int":
            try:
                result[key] = int(value_str)
            except ValueError:
                msg = f"Cannot cast {value_str!r} to int for field {key!r}"
                raise ValueError(msg) from None
        elif expected_type_name == "float":
            try:
                result[key] = float(value_str)
            except ValueError:
                msg = f"Cannot cast {value_str!r} to float for field {key!r}"
                raise ValueError(msg) from None
        else:
            result[key] = value_str
    return result


# ═══════════════════════════════════════════════════════════════════════
# Data validation
# ═══════════════════════════════════════════════════════════════════════


def _validate_data(
    clean_polls: CleanPolls,
    results_r1: RoundResult,
    results_r2: RoundResult,
    round1_candidates: list[Candidate],
) -> list[str]:
    """Validate poll data and election results before running the pipeline.

    Args:
        clean_polls: A ``CleanPolls`` instance with ``round1`` and ``round2``
            DataFrames.
        results_r1: Round 1 ``RoundResult``.
        results_r2: Round 2 ``RoundResult``.
        round1_candidates: List of ``Candidate`` objects active in round 1.

    Returns:
        List of warning messages (empty = no issues found).

    """
    warnings: list[str] = []

    r1_pollsters = clean_polls.round1["encuestadora"].nunique()  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    if r1_pollsters < _MIN_POLLSTERS_R1:
        warnings.append(
            f"Round 1 has {r1_pollsters} unique pollster(s); at least {_MIN_POLLSTERS_R1} required",
        )
    r2_pollsters = clean_polls.round2["encuestadora"].nunique()  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    if r2_pollsters < _MIN_POLLSTERS_R2:
        warnings.append(
            f"Round 2 has {r2_pollsters} unique pollster(s); at least {_MIN_POLLSTERS_R2} required",
        )

    for candidate in round1_candidates:
        key = candidate.key
        if key not in clean_polls.round1.columns:
            continue
        col = clean_polls.round1[key]  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        if (col < 0).any():  # pyright: ignore[reportUnknownMemberType]
            warnings.append(
                f"Negative vote share for '{key}' in round 1 polls (min={col.min():.2f})",  # pyright: ignore[reportUnknownMemberType]
            )
        if (col > _MAX_SHARE_PCT).any():  # pyright: ignore[reportUnknownMemberType]
            warnings.append(
                f"Vote share > 100% for '{key}' in round 1 polls (max={col.max():.2f})",  # pyright: ignore[reportUnknownMemberType]
            )

    for label, rr in [("Round 1", results_r1), ("Round 2", results_r2)]:
        total = sum(c.vote_share for c in rr.candidates)
        if abs(total - 1.0) > _SHARE_TOLERANCE:
            warnings.append(
                f"{label} candidate shares sum to {total:.4f}, expected ~1.0",
            )

    return warnings


# ═══════════════════════════════════════════════════════════════════════
# Output helpers
# ═══════════════════════════════════════════════════════════════════════


def _print_separator(char: str = "=", width: int = 70) -> None:
    """Print a horizontal separator line."""
    print(char * width)


def _format_pct(value: float) -> str:
    """Format a vote share as a percentage string."""
    return f"{value * 100:6.2f}%"


def _format_error(error: float) -> str:
    """Format an error (predicted - actual) with sign and pp suffix."""
    return f"{error * 100:+7.2f}pp".strip()


def _format_ci(ci: tuple[float, float]) -> str:
    """Format a 95% credible interval."""
    return f"[{ci[0] * 100:5.1f}%, {ci[1] * 100:5.1f}%]"


def _print_table_header() -> None:
    """Print the summary table header with column headings."""
    print(
        "  +" + "-" * 19 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 13 + "+" + "-" * 20 + "+",
    )
    print(
        "  | {:<17s} | {:>8s} | {:>8s} | {:>11s} | {:>18s} |".format(
            "Candidate",
            "Predicted",
            "Actual",
            "Error",
            "95% CI",
        ),
    )
    print(
        "  +" + "-" * 19 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 13 + "+" + "-" * 20 + "+",
    )


def _print_table_footer() -> None:
    """Print the summary table footer separator."""
    print(
        "  +" + "-" * 19 + "+" + "-" * 10 + "+" + "-" * 10 + "+" + "-" * 13 + "+" + "-" * 20 + "+",
    )


# ═══════════════════════════════════════════════════════════════════════
# Command implementations
# ═══════════════════════════════════════════════════════════════════════


def _run_config() -> None:
    """Print the current project configuration."""
    config = ModelConfig()
    _print_separator()
    print("  CO-PRESIDENT 2026 — Configuration")
    _print_separator()
    print(f"  Election Round 1: {ELECTION_DATE_ROUND1}")
    print(f"  Election Round 2: {ELECTION_DATE_ROUND2}")
    print(f"  Consultation:     {CONSULTATION_DATE}")
    print()
    print("  Candidates (Round 1):")
    for key, c in FIRST_ROUND_CANDIDATES.items():
        print(f"    {key:25s}  {c.display_name}")
    print()
    print("  Candidates (Round 2):")
    r2_candidates = get_active_candidates(2)
    for c in r2_candidates:
        print(f"    {c.key:25s}  {c.display_name}")
    print()
    print("  Pollster Ratings:")
    for pollster, rating in sorted(POLLSTER_RATINGS.items(), key=lambda x: -x[1]):
        print(f"    {pollster:25s}  {rating:.2f}")
    print()
    print("  Model Hyperparameters:")
    for f in fields(ModelConfig):
        print(f"    {f.name:40s} = {getattr(config, f.name)}")
    _print_separator()


def _run_aggregate() -> None:
    """Run the baseline weighted average (SPEC-05) and print results."""
    from co_president.aggregation import weighted_average  # noqa: PLC0415
    from co_president.data import load_and_clean_all  # noqa: PLC0415

    try:
        clean_polls = load_and_clean_all()
    except (OSError, ValueError) as e:
        print(f"Error loading poll data: {e}")
        sys.exit(1)

    r1_candidates = get_active_candidates(1)
    r1_keys = [c.key for c in r1_candidates]

    _print_separator()
    print("  BASELINE WEIGHTED AVERAGES")
    _print_separator()
    print(f"\n  First Round (election: {ELECTION_DATE_ROUND1}):")
    print(f"  {'Candidate':25s} {'Weighted Avg':>15s}")
    print(f"  {'-' * 25} {'-' * 15}")
    r1_avg = weighted_average(
        clean_polls.round1,
        r1_keys,
        ELECTION_DATE_ROUND1,
        POLLSTER_RATINGS,
    )
    for key in r1_keys:
        if key in r1_avg:
            print(f"  {key:25s} {_format_pct(r1_avg[key] / 100.0):>15s}")
    total_r1 = sum(r1_avg.get(k, 0.0) for k in r1_keys)
    print(f"  {'Total':25s} {total_r1:14.2f}%")
    print()

    r2_candidates = get_active_candidates(2)
    r2_keys = [c.key for c in r2_candidates]
    print(f"  Runoff (election: {ELECTION_DATE_ROUND2}):")
    print(f"  {'Candidate':25s} {'Weighted Avg':>15s}")
    print(f"  {'-' * 25} {'-' * 15}")
    r2_avg = weighted_average(
        clean_polls.round2,
        r2_keys,
        ELECTION_DATE_ROUND2,
        POLLSTER_RATINGS,
    )
    for key in r2_keys:
        if key in r2_avg:
            print(f"  {key:25s} {_format_pct(r2_avg[key] / 100.0):>15s}")
    total_r2 = sum(r2_avg.get(k, 0.0) for k in r2_keys)
    print(f"  {'Total':25s} {total_r2:14.2f}%")
    _print_separator()


# ═══════════════════════════════════════════════════════════════════════
# Run command helpers
# ═══════════════════════════════════════════════════════════════════════


def _load_and_validate() -> tuple[CleanPolls, RoundResult, RoundResult]:
    """Load canonical results and poll data, validating along the way.

    Returns:
        Tuple of ``(clean_polls, results_r1, results_r2)``.

    """
    from co_president.data import load_and_clean_all, load_canonical_results  # noqa: PLC0415

    logger.info("Loading canonical election results...")
    results_r1, results_r2 = load_canonical_results()

    logger.info("Loading and cleaning poll data...")
    clean_polls = load_and_clean_all()

    round1_candidates = get_active_candidates(1)
    warnings = _validate_data(clean_polls, results_r1, results_r2, round1_candidates)
    for w in warnings:
        logger.warning("Data validation: %s", w)

    return clean_polls, results_r1, results_r2


def _log_and_save_trace(idata: az.InferenceData, path: Path) -> None:
    """Log and save an InferenceData to a netCDF file."""
    logger.info("Saving trace to %s", path)
    idata.to_netcdf(str(path))  # pyright: ignore[reportUnknownMemberType]


def _check_convergence(idata: az.InferenceData, label: str) -> None:
    """Check R-hat convergence and log a warning if it exceeds threshold."""
    try:
        summary = az.summary(idata)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        max_rhat = summary["r_hat"].max()  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        if max_rhat > _RHAT_LIMIT:
            logger.warning(
                "%s: max R-hat = %.3f (> %.2f); convergence may be suspect",
                label,
                max_rhat,  # pyright: ignore[reportUnknownArgumentType]
                _RHAT_LIMIT,
            )
    except (ValueError, TypeError, KeyError):
        logger.warning("Could not compute R-hat for %s trace", label)


def _find_pairing(matrix: RunoffMatrix, pair: tuple[str, str]) -> PairingForecast | None:
    """Return the ``PairingForecast`` for ``pair``, or ``None`` if not found."""
    return next(
        (p for p in matrix.pairings if (p.candidate_first, p.candidate_second) == pair),
        None,
    )


def _compute_runoff_matrix(  # noqa: PLR0913
    idata_r1: az.InferenceData,
    round1_forecast: Round1Forecast,
    results_r1: RoundResult,
    results_r2: RoundResult,
    round2_polls: pd.DataFrame | None,
    config: ModelConfig,
    results_dir: Path,
) -> tuple[RunoffMatrix | None, dict[str, float] | None]:
    """Estimate the runoff matrix, log top pairings, and save as JSON.

    Args:
        idata_r1: Round 1 posterior ``InferenceData``.
        round1_forecast: Round 1 forecast with ``prob_win_outright`` per candidate.
        results_r1: Round 1 ``RoundResult``.
        results_r2: Round 2 ``RoundResult``.
        round2_polls: Clean Round 2 poll DataFrame (or ``None``).
        config: Model hyperparameters.
        results_dir: Directory to save the JSON matrix file.

    Returns:
        Tuple of ``(runoff_matrix, overall_probs)``. Either may be ``None`` on
        failure.

    """
    from dataclasses import asdict  # noqa: PLC0415
    import json  # noqa: PLC0415

    from co_president.model_runoff_matrix import (  # noqa: PLC0415
        estimate_runoff_matrix,
        overall_win_probability,
    )

    try:
        prob_win_outright_dict = {
            fc.candidate_key: fc.prob_win_outright for fc in round1_forecast.candidates
        }
        runoff_matrix = estimate_runoff_matrix(
            idata_r1, (results_r1, results_r2), round2_polls, config
        )
        overall_probs_val = overall_win_probability(runoff_matrix, prob_win_outright_dict)

        top_pairings = runoff_matrix.ordered_by_likelihood[:3]
        logger.info("Top-3 most likely pairings:")
        for pair in top_pairings:
            pairing_obj = _find_pairing(runoff_matrix, pair)
            if pairing_obj is None:
                continue
            logger.info("  %s vs %s: %.1f%%", pair[0], pair[1], pairing_obj.prob_pairing * 100)

        for cand, prob in sorted(overall_probs_val.items(), key=lambda x: x[1], reverse=True):
            logger.info("Overall presidency probability for %s: %.1f%%", cand, prob * 100)

        matrix_path = results_dir / "runoff_matrix.json"
        matrix_dict = asdict(runoff_matrix)
        matrix_dict["ordered_by_likelihood"] = [
            list(p) for p in matrix_dict["ordered_by_likelihood"]
        ]
        matrix_dict["pairings"] = list(matrix_dict["pairings"])
        matrix_path.write_text(json.dumps(matrix_dict, indent=2), encoding="utf-8")
        logger.info("Saved runoff matrix to %s", matrix_path)
    except (ValueError, RuntimeError, OSError, TypeError):
        logger.exception("Runoff matrix computation failed")
        return None, None
    else:
        return runoff_matrix, overall_probs_val


def _run_pipeline_mcmc(
    clean_polls: CleanPolls,
    results_r1: RoundResult,
    results_r2: RoundResult,
    config: ModelConfig,
    results_dir: Path,
) -> None:
    """Run the full MCMC pipeline: round 1, runoff, validation."""
    import co_president.model_round1 as m1  # noqa: PLC0415
    import co_president.model_runoff_simple as mr  # noqa: PLC0415
    import co_president.validation as v  # noqa: PLC0415

    # ── Round 1 ────────────────────────────────────────────────────────
    logger.info("Building round 1 model...")
    round1_model = m1.build_round1_model(clean_polls.round1, results_r1, config)
    logger.info(
        "Sampling round 1 model (%d draws x %d chains)...",
        config.mcmc_draws,
        config.mcmc_chains,
    )
    idata_r1 = m1.sample_round1(round1_model, config)
    _log_and_save_trace(idata_r1, results_dir / "round1_trace.nc")

    r1_candidate_keys = sorted(
        set(m1.FIRST_ROUND_CANDIDATES) & set(clean_polls.round1.columns),
    )
    logger.info("Computing round 1 forecast...")
    round1_forecast = m1.forecast_round1(idata_r1, r1_candidate_keys)
    _check_convergence(idata_r1, "Round 1")

    # ── Runoff ─────────────────────────────────────────────────────────
    try:
        r2_top_two = results_r1.top_two()
    except ValueError:
        logger.warning("Not enough candidates for runoff analysis")
        return
    candidate_a = r2_top_two[0].candidate_key
    candidate_b = r2_top_two[1].candidate_key

    runoff_model = mr.build_runoff_simple_model(
        clean_polls.round2,
        results_r1,
        idata_r1,
        config,
    )
    logger.info("Sampling runoff model...")
    runoff_forecast: RunoffForecast | None = None
    try:
        idata_runoff = mr.sample_runoff(runoff_model, config)
        _log_and_save_trace(idata_runoff, results_dir / "runoff_trace.nc")
        runoff_forecast = mr.forecast_runoff_simple(idata_runoff, candidate_a, candidate_b)
        _check_convergence(idata_runoff, "Runoff")
    except (ValueError, RuntimeError):
        logger.exception("Runoff MCMC sampling failed")

    # ── Validate ───────────────────────────────────────────────────────
    logger.info("Validating round 1 forecast...")
    r1_validation = v.validate_round1(round1_forecast, results_r1)
    r1_brier = v.brier_score_round1(round1_forecast, results_r1)

    r2_validation: RoundValidation | None = None
    if runoff_forecast is not None:
        logger.info("Validating runoff forecast...")
        try:
            r2_validation = v.validate_runoff(runoff_forecast, results_r2)
        except (ValueError, KeyError):
            logger.warning("Runoff validation expected to be partial")

    # ── Runoff matrix & overall win probabilities ──────────────────────
    runoff_matrix, overall_probs = _compute_runoff_matrix(
        idata_r1,
        round1_forecast,
        results_r1,
        results_r2,
        clean_polls.round2,
        config,
        results_dir,
    )

    _print_run_summary_table(
        round1_forecast,
        results_r1,
        r1_validation,
        r1_brier,
        runoff_forecast,
        results_r2,
        r2_validation,
        runoff_matrix,
        overall_probs,
    )

    # ── Rolling forecast (optional) ────────────────────────────────────
    logger.info("Computing rolling forecast (may take a while)...")
    try:
        rolling = v.rolling_forecast(
            clean_polls,
            (results_r1, results_r2),
            config,
            n_snapshots=10,
        )
        for snap_date, snap_forecast in rolling:
            v.save_rolling_snapshot((snap_date, snap_forecast), str(results_dir))
    except (ValueError, RuntimeError, OSError):
        logger.warning("Rolling forecast failed")


def _print_no_sample_table(
    clean_polls: CleanPolls,
    results_r1: RoundResult,
    round1_candidates: list[Candidate],
) -> None:
    """Print the summary table using baseline weighted averages (no MCMC)."""
    from co_president.aggregation import weighted_average  # noqa: PLC0415

    _print_separator()
    print("  CO-PRESIDENT 2026 — 2022 Backtesting (Baseline Weighted Avg)")
    _print_separator()

    r1_keys = [c.key for c in round1_candidates]
    r1_avg = weighted_average(
        clean_polls.round1,
        r1_keys,
        ELECTION_DATE_ROUND1,
        POLLSTER_RATINGS,
    )

    print()
    print(f"  FIRST ROUND ({ELECTION_DATE_ROUND1})")
    _print_table_header()
    for c in round1_candidates:
        pred = r1_avg.get(c.key, 0.0) / 100.0
        try:
            actual = results_r1.get_share(c.key)
        except KeyError:
            actual = 0.0
        error = pred - actual
        print(
            f"  | {c.display_name:17s} | {_format_pct(pred):>8s} | "
            f"{_format_pct(actual):>8s} | {_format_error(error):>11s} | "
            f"{'N/A':>18s} |",
        )
    _print_table_footer()
    print()
    print("  RUNOFF — Baseline only (no runoff polls computed without MCMC)")
    _print_separator()


def _print_run_summary_table(  # noqa: PLR0913
    round1_forecast: Round1Forecast,
    results_r1: RoundResult,
    r1_validation: RoundValidation,
    r1_brier: float,
    runoff_forecast: RunoffForecast | None,
    results_r2: RoundResult,
    r2_validation: RoundValidation | None,
    runoff_matrix: RunoffMatrix | None = None,
    overall_probs: dict[str, float] | None = None,
) -> None:
    """Print the formatted summary table after a full pipeline run."""
    display_names = {key: c.display_name for key, c in FIRST_ROUND_CANDIDATES.items()}
    _print_separator()
    print("  CO-PRESIDENT 2026 — 2022 Backtesting Results")
    _print_separator()

    print()
    print(f"  FIRST ROUND ({results_r1.date})")
    _print_table_header()
    for fc in round1_forecast.candidates:
        try:
            actual = results_r1.get_share(fc.candidate_key)
        except KeyError:
            actual = 0.0
        error = fc.mean_share - actual
        ci_str = _format_ci(fc.ci_95)
        name = display_names.get(fc.candidate_key, fc.candidate_key)
        print(
            f"  | {name:17s} | {_format_pct(fc.mean_share):>8s} | "
            f"{_format_pct(actual):>8s} | {_format_error(error):>11s} | "
            f"{ci_str:>18s} |",
        )
    _print_table_footer()
    print(
        f"  MAE: {r1_validation.mae * 100:.2f}pp  |  "
        f"RMSE: {r1_validation.rmse * 100:.2f}pp  |  "
        f"Brier: {r1_brier:.4f}",
    )
    print(
        f"  Probability of runoff: {round1_forecast.prob_runoff * 100:.1f}%",
    )
    print()

    if runoff_forecast is not None:
        name_a = display_names.get(runoff_forecast.candidate_a_key, runoff_forecast.candidate_a_key)
        name_b = display_names.get(runoff_forecast.candidate_b_key, runoff_forecast.candidate_b_key)
        print(
            f"  RUNOFF ({results_r2.date}) — {name_a} vs. {name_b}",
        )
        print(
            f"  {name_a:20s}: "
            f"{_format_pct(runoff_forecast.mean_share_a)} "
            f"{_format_ci(runoff_forecast.ci_95_a)}  "
            f"-> Win prob: {runoff_forecast.prob_a_wins * 100:.1f}%",
        )
        print(
            f"  {name_b:20s}: "
            f"{_format_pct(runoff_forecast.mean_share_b)} "
            f"{_format_ci(runoff_forecast.ci_95_b)}  "
            f"-> Win prob: {runoff_forecast.prob_b_wins * 100:.1f}%",
        )
        if r2_validation is not None:
            print(
                f"  MAE: {r2_validation.mae * 100:.2f}pp  |  "
                f"Expected margin: {runoff_forecast.mean_margin * 100:+.1f}pp "
                f"{name_a}",
            )
    else:
        print("  RUNOFF: No forecast available (sampling failed)")

    # ── Runoff matrix & overall win probability section ────────────────
    if runoff_matrix is not None and runoff_matrix.pairings:
        print()
        print("  RUNOFF MATRIX — Most likely pairings")
        for i, pair in enumerate(runoff_matrix.ordered_by_likelihood[:3]):
            pairing_obj = _find_pairing(runoff_matrix, pair)
            if pairing_obj is None:
                continue
            name_first = display_names.get(pair[0], pair[0])
            name_second = display_names.get(pair[1], pair[1])
            print(
                f"  {i + 1}. {name_first} vs {name_second}: "
                f"{pairing_obj.prob_pairing * 100:.1f}%  "
                f"({name_first}: {pairing_obj.prob_first_wins * 100:.1f}%)",
            )

        if overall_probs:
            print()
            print("  OVERALL PRESIDENCY PROBABILITY")
            for cand, prob in sorted(overall_probs.items(), key=lambda x: x[1], reverse=True):
                name = display_names.get(cand, cand)
                print(f"  {name:20s}: {prob * 100:5.1f}%")

    _print_separator()


def _cmd_run(args: argparse.Namespace) -> None:
    """Execute the ``run`` subcommand.

    Args:
        args: Parsed CLI arguments.

    """
    config = ModelConfig()
    if args.config_override:
        try:
            overrides = _parse_config_overrides(args.config_override)
            config = ModelConfig(**overrides)
        except ValueError as e:
            print(f"Configuration error: {e}")
            sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # ── Load data ─────────────────────────────────────────────────────
    try:
        clean_polls, results_r1, results_r2 = _load_and_validate()
    except (ValueError, OSError):
        logger.exception("Failed to load data")
        sys.exit(1)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Run (with or without MCMC) ────────────────────────────────────
    if args.no_sample:
        round1_candidates = get_active_candidates(1)
        _print_no_sample_table(clean_polls, results_r1, round1_candidates)
        return

    _run_pipeline_mcmc(clean_polls, results_r1, results_r2, config, results_dir)

    logger.info("Pipeline complete. Results saved to %s/", results_dir)


# ═══════════════════════════════════════════════════════════════════════
# Validate command
# ═══════════════════════════════════════════════════════════════════════


def _load_trace_or_exit(path: Path, label: str) -> az.InferenceData:
    """Load an InferenceData from a netCDF file or exit with error.

    Args:
        path: Path to the .nc file.
        label: Human-readable name for error messages.

    Returns:
        Loaded InferenceData.

    """
    try:
        return az.from_netcdf(str(path))  # pyright: ignore[reportUnknownMemberType]
    except (OSError, ValueError) as e:
        print(f"Error loading {label} trace: {e}")
        sys.exit(1)


def _print_r1_validation(
    r1_validation: RoundValidation,
    r1_brier: float,
) -> None:
    """Print round 1 validation results.

    Args:
        r1_validation: ``RoundValidation`` for round 1.
        r1_brier: Brier score for round 1.

    """
    print()
    print("  Round 1:")
    print(f"    MAE:    {r1_validation.mae * 100:.4f}pp")
    print(f"    RMSE:   {r1_validation.rmse * 100:.4f}pp")
    print(f"    Brier:  {r1_brier:.4f}")
    print(f"    95%% CI coverage: {r1_validation.calibration_95 * 100:.1f}%")
    print(f"    50%% CI coverage: {r1_validation.calibration_50 * 100:.1f}%")
    print()
    print("  Per-candidate Errors:")
    for cv in r1_validation.candidates:
        print(
            f"    {cv.candidate_key:25s}  "
            f"pred={cv.predicted_mean * 100:6.2f}%  "
            f"actual={cv.actual_share * 100:6.2f}%  "
            f"error={cv.error * 100:+7.3f}pp  "
            f"{'✓' if cv.within_95ci else '✗'} 95% CI",
        )


def _print_r2_validation(
    runoff_forecast: RunoffForecast,
    r2_validation: RoundValidation,
    candidate_a: str,
    candidate_b: str,
) -> None:
    """Print runoff validation results.

    Args:
        runoff_forecast: ``RunoffForecast``.
        r2_validation: ``RoundValidation`` for runoff.
        candidate_a: First runoff candidate key.
        candidate_b: Second runoff candidate key.

    """
    print()
    print("  Runoff:")
    print(f"    MAE:    {r2_validation.mae * 100:.4f}pp")
    print(f"    RMSE:   {r2_validation.rmse * 100:.4f}pp")
    print(f"    {candidate_a} mean: {runoff_forecast.mean_share_a * 100:.2f}%")
    print(f"    {candidate_b} mean: {runoff_forecast.mean_share_b * 100:.2f}%")
    print(f"    Win prob {candidate_a}: {runoff_forecast.prob_a_wins * 100:.1f}%")
    print(f"    Win prob {candidate_b}: {runoff_forecast.prob_b_wins * 100:.1f}%")


def _cmd_validate() -> None:
    """Load saved ``InferenceData`` from ``results/`` and print validation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    from co_president.data import load_canonical_results  # noqa: PLC0415
    from co_president.model_round1 import (  # noqa: PLC0415
        forecast_round1,
    )
    from co_president.model_runoff_simple import (  # noqa: PLC0415
        forecast_runoff_simple,
    )
    from co_president.validation import (  # noqa: PLC0415
        brier_score_round1,
        validate_round1,
        validate_runoff,
    )

    results_dir = Path("results")
    round1_path = results_dir / "round1_trace.nc"
    runoff_path = results_dir / "runoff_trace.nc"

    if not round1_path.is_file():
        print(f"Error: round 1 trace not found at {round1_path}")
        print("Run 'python -m co_president run' first to generate traces.")
        sys.exit(1)

    logger.info("Loading results and traces...")
    try:
        results_r1, results_r2 = load_canonical_results()
    except (OSError, ValueError) as e:
        print(f"Error loading election results: {e}")
        sys.exit(1)

    idata_r1 = _load_trace_or_exit(round1_path, "round 1")
    r1_candidate_keys = sorted(FIRST_ROUND_CANDIDATES)

    logger.info("Computing round 1 forecast from trace...")
    try:
        round1_forecast = forecast_round1(idata_r1, r1_candidate_keys)
    except (ValueError, KeyError) as e:
        print(f"Error computing round 1 forecast: {e}")
        sys.exit(1)

    r1_validation = validate_round1(round1_forecast, results_r1)
    r1_brier = brier_score_round1(round1_forecast, results_r1)

    _print_separator()
    print("  VALIDATION REPORT")
    _print_separator()
    _print_r1_validation(r1_validation, r1_brier)

    if runoff_path.is_file():
        try:
            idata_runoff = az.from_netcdf(str(runoff_path))  # pyright: ignore[reportUnknownMemberType]
            r2_top_two = results_r1.top_two()
            if len(r2_top_two) < 2:  # noqa: PLR2004
                logger.warning("Round 1 results have <2 candidates; runoff validation skipped")
                return
            candidate_a = r2_top_two[0].candidate_key
            candidate_b = r2_top_two[1].candidate_key
            runoff_forecast = forecast_runoff_simple(idata_runoff, candidate_a, candidate_b)
            r2_validation = validate_runoff(runoff_forecast, results_r2)
            _print_r2_validation(runoff_forecast, r2_validation, candidate_a, candidate_b)
        except (ValueError, KeyError, OSError) as e:
            logger.warning("Runoff validation skipped: %s", e)
    else:
        print()
        print("  Runoff: no trace found (skipped)")
    _print_separator()


# ═══════════════════════════════════════════════════════════════════════
# Plot command
# ═══════════════════════════════════════════════════════════════════════


def _load_or_generate_snapshots(
    results_r1: RoundResult,
    results_r2: RoundResult,
    out: Path,
) -> list[tuple[date, Round1Forecast]]:
    """Load saved rolling snapshots or generate them from scratch.

    Args:
        results_r1: Round 1 RoundResult.
        results_r2: Round 2 RoundResult.
        out: Output directory path.

    Returns:
        List of (date, Round1Forecast) snapshots.

    """
    from co_president.validation import (  # noqa: PLC0415
        load_rolling_snapshots,
    )

    snapshots = load_rolling_snapshots(str(out))
    if snapshots:
        return snapshots

    logger.info("No saved snapshots found; generating rolling forecast...")
    from co_president.data import load_and_clean_all  # noqa: PLC0415
    from co_president.validation import rolling_forecast, save_rolling_snapshot  # noqa: PLC0415

    try:
        clean_polls = load_and_clean_all()
        snapshots = rolling_forecast(
            clean_polls,
            (results_r1, results_r2),
            ModelConfig(),
            n_snapshots=10,
        )
        for snap_date, snap_forecast in snapshots:
            save_rolling_snapshot((snap_date, snap_forecast), str(out))
    except (ValueError, RuntimeError, OSError):
        logger.exception("Failed to generate rolling forecast")
        snapshots = []
    return snapshots


def _plot_evolution(
    snapshots: list[tuple[date, Round1Forecast]],
    results_r1: RoundResult,
    out: Path,
) -> None:
    """Generate and save the forecast evolution plot."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    from co_president.plotting import plot_forecast_evolution  # noqa: PLC0415

    try:
        fig = plot_forecast_evolution(snapshots, results_r1)
        fig.savefig(str(out / "forecast_evolution.png"), dpi=150, bbox_inches="tight")  # pyright: ignore[reportUnknownMemberType]
        plt.close(fig)
        logger.info("Saved forecast_evolution.png")
    except (ValueError, RuntimeError, OSError) as e:
        logger.warning("Failed to generate forecast evolution plot: %s", e)


def _plot_calibration(
    results_r1: RoundResult,
    out: Path,
    trace_dir: str = "results",
) -> None:
    """Generate and save the calibration plot.

    Args:
        results_r1: Round 1 RoundResult.
        out: Output directory for the plot PNG.
        trace_dir: Directory containing the saved round 1 trace file.

    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    from co_president.plotting import plot_calibration  # noqa: PLC0415
    from co_president.validation import validate_round1  # noqa: PLC0415

    trace_path = Path(trace_dir) / "round1_trace.nc"
    r1_validation: RoundValidation | None = None

    if trace_path.is_file():
        try:
            from co_president.model_round1 import (  # noqa: PLC0415
                forecast_round1,
            )

            idata_r1 = az.from_netcdf(str(trace_path))  # pyright: ignore[reportUnknownMemberType]
            r1_forecast = forecast_round1(idata_r1, sorted(FIRST_ROUND_CANDIDATES))
            r1_validation = validate_round1(r1_forecast, results_r1)
        except (ValueError, KeyError, OSError) as e:
            logger.warning("Could not load trace for calibration: %s", e)

    if r1_validation is not None:
        fig = plot_calibration(r1_validation)
        fig.savefig(str(out / "calibration.png"), dpi=150, bbox_inches="tight")  # pyright: ignore[reportUnknownMemberType]
        plt.close(fig)
        logger.info("Saved calibration.png")
    else:
        logger.warning("Skipping calibration plot (no trace available)")


def _plot_errors(
    snapshots: list[tuple[date, Round1Forecast]],
    results_r1: RoundResult,
    out: Path,
) -> None:
    """Generate and save the error-over-time plot."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    from co_president.plotting import plot_error_over_time  # noqa: PLC0415
    from co_president.validation import compute_rolling_errors  # noqa: PLC0415

    if not snapshots:
        logger.warning("Skipping error over time plot (no snapshots)")
        return

    try:
        rolling_errors = compute_rolling_errors(snapshots, results_r1)
        fig = plot_error_over_time(rolling_errors)
        fig.savefig(str(out / "error_over_time.png"), dpi=150, bbox_inches="tight")  # pyright: ignore[reportUnknownMemberType]
        plt.close(fig)
        logger.info("Saved error_over_time.png")
    except (ValueError, RuntimeError, OSError) as e:
        logger.warning("Failed to generate error over time plot: %s", e)


def _cmd_download_cedae(args: argparse.Namespace) -> None:
    """Execute the ``download-cedae`` subcommand.

    Args:
        args: Parsed CLI arguments.

    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    from co_president.ingestion._download_cedae import download_cedae_elections  # noqa: PLC0415

    dest_dir = Path(args.output_dir) if args.output_dir else None
    years = tuple(int(y.strip()) for y in args.years.split(","))
    niveles = tuple(n.strip() for n in args.niveles.split(","))

    files = download_cedae_elections(dest_dir=dest_dir, years=years, niveles=niveles)
    total_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
    logger.info(
        "Downloaded %d files (%.1f MiB) to %s",
        len(files),
        total_mb,
        files[0].parent if files else "N/A",
    )


def _cmd_plot(output_dir: str) -> None:
    """Generate all visualization plots and save to *output_dir*.

    Args:
        output_dir: Directory path for saving plot PNG files.

    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")

    from co_president.data import load_canonical_results  # noqa: PLC0415

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger.info("Saving plots to %s/", out)

    try:
        results_r1, results_r2 = load_canonical_results()
    except (OSError, ValueError):
        logger.exception("Failed to load election results")
        sys.exit(1)

    snapshots = _load_or_generate_snapshots(results_r1, results_r2, out)

    if snapshots:
        _plot_evolution(snapshots, results_r1, out)
        _plot_errors(snapshots, results_r1, out)
    else:
        logger.warning("Skipping evolution and error plots (no snapshots)")

    _plot_calibration(results_r1, out)
    logger.info("Plot generation complete.")


# ═══════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate subcommand."""
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.command == "config":
            _run_config()
        elif args.command == "aggregate":
            _run_aggregate()
        elif args.command == "run":
            _cmd_run(args)
        elif args.command == "validate":
            _cmd_validate()
        elif args.command == "download-cedae":
            _cmd_download_cedae(args)
        elif args.command == "plot":
            _cmd_plot(args.output_dir)
        else:
            parser.print_help()
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
