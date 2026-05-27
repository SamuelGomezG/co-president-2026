"""SPEC-10: CLI entry point for co-president.

Provides ``python -m co_president <command>`` subcommands.

Commands:
    validate    Run validation diagnostics including ns_nr sensitivity.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
import sys

from co_president.config import (
    CONSULTATION_DATE,
    ELECTION_DATE_ROUND1,
    ELECTION_DATE_ROUND2,
    FIRST_ROUND_CANDIDATES,
    ModelConfig,
)

# ruff: noqa: T201 — CLI output uses print() for user-facing text


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all subcommands.

    Returns:
        Configured ArgumentParser.

    """
    parser = argparse.ArgumentParser(
        description="co-president 2026 — Bayesian election forecast",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    validate_parser = subparsers.add_parser(
        "validate",
        help="Run validation diagnostics including ns_nr sensitivity",
    )
    validate_parser.add_argument(
        "--baseline-forecast",
        type=str,
        required=True,
        help="Path to baseline Round1Forecast JSON file",
    )
    validate_parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="Shift threshold in vote share (default 0.02 = 2pp)",
    )

    subparsers.add_parser(
        "config",
        help="Print current configuration",
    )

    return parser


def _run_validate(args: argparse.Namespace) -> None:
    """Run the ``validate`` subcommand.

    Loads a baseline forecast from a JSON file, runs the ns_nr sensitivity
    diagnostic, and prints results.

    Args:
        args: Parsed command-line arguments.

    """
    import json  # noqa: PLC0415

    from co_president.data_polls import load_and_clean_all  # noqa: PLC0415
    from co_president.data_results import (  # noqa: PLC0415
        load_canonical_results,
    )
    from co_president.model_round1 import Round1Forecast  # noqa: PLC0415
    from co_president.validation import sensitivity_ns_nr  # noqa: PLC0415

    try:
        path = Path(args.baseline_forecast)
        baseline_json = json.loads(path.read_text(encoding="utf-8"))
        baseline_forecast = Round1Forecast.from_dict(baseline_json)

        results_r1, _ = load_canonical_results()
        polls = load_and_clean_all()
        config = ModelConfig()
    except FileNotFoundError:
        print(f"Error: baseline forecast file not found: {args.baseline_forecast}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: invalid JSON in baseline forecast file: {e}")
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        print(f"Error loading data or forecast: {e}")
        sys.exit(1)

    try:
        result = sensitivity_ns_nr(
            baseline_forecast=baseline_forecast,
            polls=polls,
            results=results_r1,
            config=config,
            threshold=args.threshold,
        )
    except Exception as e:  # noqa: BLE001
        print(f"Error running ns_nr sensitivity check: {e}")
        sys.exit(1)

    print("=" * 60)
    print("  NS_NR SENSITIVITY CHECK")
    print("=" * 60)
    print(
        f"  Threshold: {args.threshold * 100:.1f}pp  |  "
        f"Flagged: {result.total_candidates_flagged}/{len(result.flags)}",
    )
    print()
    for flag in result.flags:
        marker = " *** FLAGGED ***" if flag.exceeds_threshold else ""
        print(
            f"  {flag.candidate_key:25s}  "
            f"baseline={flag.baseline_mean * 100:5.2f}%  "
            f"sensitive={flag.sensitive_mean * 100:5.2f}%  "
            f"shift={flag.shift * 100:+6.3f}pp{marker}",
        )
    print()
    if result.has_failures:
        print("  WARNING: Some candidates exceed the shift threshold.")
    else:
        print("  All candidates within threshold.")
    print("=" * 60)


def _run_config() -> None:
    """Print the current project configuration.

    Displays candidate mappings, election dates, and model hyperparameters.

    """
    config = ModelConfig()
    print("CO-PRESIDENT 2026 — Configuration")
    print("=" * 50)
    print(f"  Election Round 1: {ELECTION_DATE_ROUND1}")
    print(f"  Election Round 2: {ELECTION_DATE_ROUND2}")
    print(f"  Consultation:     {CONSULTATION_DATE}")
    print()
    print("  Candidates (Round 1):")
    for key, c in FIRST_ROUND_CANDIDATES.items():
        print(f"    {key:25s}  {c.display_name}")
    print()
    print("  Model Hyperparameters:")
    for f in fields(ModelConfig):
        print(f"    {f.name:40s} = {getattr(config, f.name)}")


def main() -> None:
    """Parse CLI arguments and dispatch to the appropriate subcommand."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "validate":
        _run_validate(args)
    elif args.command == "config":
        _run_config()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
