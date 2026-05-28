"""Tests for SPEC-10: CLI entry point."""

from __future__ import annotations

import subprocess
import sys

import pandas as pd
import pytest

from co_president.config import ELECTION_DATE_ROUND1, ELECTION_DATE_ROUND2, get_active_candidates
from co_president.data import CandidateResult, CleanPolls, RoundResult

# ═══════════════════════════════════════════════════════════════════════
# Module import tests (validate the interface)
# ═══════════════════════════════════════════════════════════════════════


def test_main_module_importable() -> None:
    """The __main__.py module should be importable without errors."""
    import co_president.__main__  # noqa: PLC0415

    assert co_president.__main__ is not None


def test_main_has_entry_point() -> None:
    """The module should expose a main() function."""
    from co_president.__main__ import main  # noqa: PLC0415

    assert callable(main)


# ═══════════════════════════════════════════════════════════════════════
# Subprocess tests (integration-level)
# ═══════════════════════════════════════════════════════════════════════

_PYTHON = sys.executable


def test_config_command_prints_keys() -> None:
    """``co_president config`` must print configuration keys."""
    result = subprocess.run(  # noqa: S603
        [_PYTHON, "-m", "co_president", "config"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0
    # Should contain candidate names
    for candidate_key in ("gustavo_petro", "rodolfo_hernandez", "federico_gutierrez"):
        assert candidate_key in result.stdout, (
            f"Expected candidate key '{candidate_key}' in config output, stdout:\n{result.stdout}"
        )
    # Should contain election dates
    assert "2022-05-29" in result.stdout
    assert "2022-06-19" in result.stdout
    # Should contain pollster ratings section
    assert "Invamer" in result.stdout


def test_aggregate_command_exits_success() -> None:
    """``co_president aggregate`` must exit with code 0."""
    result = subprocess.run(  # noqa: S603
        [_PYTHON, "-m", "co_president", "aggregate"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, (
        f"aggregate command failed with code {result.returncode}, stderr:\n{result.stderr}"
    )
    # Should print weighted averages
    assert "Petro" in result.stdout or "petro" in result.stdout.lower()


def test_run_no_sample_exits_success() -> None:
    """``co_president run --no-sample`` must exit with code 0."""
    result = subprocess.run(  # noqa: S603
        [_PYTHON, "-m", "co_president", "run", "--no-sample"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, (
        f"run --no-sample failed with code {result.returncode}, stderr:\n{result.stderr}"
    )
    # The no-sample output should contain a summary table header
    assert "CO-PRESIDENT" in result.stdout.upper() or "FIRST ROUND" in result.stdout.upper()


# ═══════════════════════════════════════════════════════════════════════
# Data validation tests (unit-level)
# ═══════════════════════════════════════════════════════════════════════


def _make_bad_clean_polls() -> CleanPolls:
    """Return a CleanPolls with deliberately bad data for validation tests."""
    # Need >= 5 pollsters for round1, >= 2 for round2 to satisfy CleanPolls post_init
    bad_round1 = pd.DataFrame(
        {
            "encuestadora": [f"Pollster{i}" for i in range(5)],
            "gustavo_petro": [120.0, 40.0, 40.0, 40.0, 40.0],  # first row > 100%
            "rodolfo_hernandez": [30.0, 28.0, 28.0, 28.0, 28.0],
            "fecha": pd.to_datetime(["2022-05-01"] * 5),
            "muestra": [1000] * 5,
        }
    )
    bad_round2 = pd.DataFrame(
        {
            "encuestadora": ["PollsterX", "PollsterY"],
            "gustavo_petro": [55.0, 54.0],
            "rodolfo_hernandez": [45.0, 46.0],
            "fecha": pd.to_datetime(["2022-06-10", "2022-06-11"]),
            "muestra": [500, 600],
        }
    )
    return CleanPolls(
        round1=bad_round1,
        round2=bad_round2,
        consultation=[],
        all_polls=bad_round1,
    )


def _make_good_clean_polls() -> CleanPolls:
    """Return a CleanPolls with valid data for validation tests."""
    # Need >= 5 pollsters for round1, >= 2 for round2
    pollsters_r1 = [f"Pollster{i}" for i in range(5)]
    good_round1 = pd.DataFrame(
        {
            "encuestadora": pollsters_r1,
            "gustavo_petro": [40.0] * 5,
            "rodolfo_hernandez": [28.0] * 5,
            "federico_gutierrez": [24.0] * 5,
            "sergio_fajardo": [4.4] * 5,
            "ingrid_betancourt": [0.4] * 5,
            "blanco": [2.0] * 5,
            "rest": [1.2] * 5,
            "fecha": pd.to_datetime(["2022-05-01"] * 5),
            "muestra": [1000] * 5,
        }
    )
    good_round2 = pd.DataFrame(
        {
            "encuestadora": ["PollsterA", "PollsterB"],
            "gustavo_petro": [50.0, 51.0],
            "rodolfo_hernandez": [47.0, 46.0],
            "blanco": [3.0, 3.0],
            "fecha": pd.to_datetime(["2022-06-10", "2022-06-11"]),
            "muestra": [500, 600],
        }
    )
    return CleanPolls(
        round1=good_round1,
        round2=good_round2,
        consultation=[],
        all_polls=good_round1,
    )


def _make_good_results_round1() -> RoundResult:
    """Return a valid RoundResult for round 1."""
    return RoundResult(
        round_number=1,
        date=ELECTION_DATE_ROUND1,
        total_valid_votes=29_900_000,
        total_votes_incl_blank=30_700_000,
        registered_voters=39_000_000,
        polling_stations=110_000,
        candidates=(
            CandidateResult("gustavo_petro", 12_300_000, 0.4034),
            CandidateResult("rodolfo_hernandez", 8_640_000, 0.2815),
            CandidateResult("federico_gutierrez", 7_330_000, 0.2389),
            CandidateResult("sergio_fajardo", 1_350_000, 0.0439),
            CandidateResult("ingrid_betancourt", 123_000, 0.0040),
            CandidateResult("rest", 392_000, 0.0128),
            CandidateResult("blanco", 800_000, 0.0155),
        ),
        blank_votes=800_000,
        null_votes=300_000,
        unmarked_votes=50_000,
    )


def test_validate_data_catches_negative_shares() -> None:
    """Validation must flag candidate shares < 0%."""
    from co_president.__main__ import _validate_data  # noqa: PLC0415

    polls = _make_good_clean_polls()
    polls.round1.loc[0, "gustavo_petro"] = -5.0
    results_r1 = _make_good_results_round1()
    results_r2 = (
        _make_good_results_round1()
    )  # not actually used for round 2, just structurally valid

    warnings = _validate_data(polls, results_r1, results_r2, get_active_candidates(1))
    assert any("negative" in w.lower() or "-5" in w for w in warnings), (
        f"Expected a warning about negative share, got: {warnings}"
    )


def test_validate_data_catches_shares_over_100() -> None:
    """Validation must flag candidate shares > 100%."""
    from co_president.__main__ import _validate_data  # noqa: PLC0415

    polls = _make_bad_clean_polls()
    results_r1 = _make_good_results_round1()
    results_r2 = _make_good_results_round1()

    warnings = _validate_data(polls, results_r1, results_r2, get_active_candidates(1))
    assert any("120" in w or "100" in w or "exceed" in w.lower() for w in warnings), (
        f"Expected a warning about >100% share, got: {warnings}"
    )


def test_validate_data_passes_good_data() -> None:
    """Validation should produce no warnings for clean data."""
    from co_president.__main__ import _validate_data  # noqa: PLC0415

    polls = _make_good_clean_polls()
    results_r1 = _make_good_results_round1()
    # Need a valid round 2 result too
    results_r2 = RoundResult(
        round_number=2,
        date=ELECTION_DATE_ROUND2,
        total_valid_votes=29_900_000,
        total_votes_incl_blank=30_700_000,
        registered_voters=39_000_000,
        polling_stations=110_000,
        candidates=(
            CandidateResult("gustavo_petro", 15_500_000, 0.5044),
            CandidateResult("rodolfo_hernandez", 14_500_000, 0.4726),
            CandidateResult("blanco", 700_000, 0.0230),
        ),
        blank_votes=700_000,
        null_votes=200_000,
        unmarked_votes=30_000,
    )

    warnings = _validate_data(polls, results_r1, results_r2, get_active_candidates(1))
    assert len(warnings) == 0, f"Expected no warnings for clean data, got: {warnings}"


# ═══════════════════════════════════════════════════════════════════════
# Config override parsing tests
# ═══════════════════════════════════════════════════════════════════════


def test_parse_config_overrides_valid() -> None:
    """Valid config overrides must be parsed correctly."""
    from co_president.__main__ import _parse_config_overrides  # noqa: PLC0415

    overrides = _parse_config_overrides(["mcmc_draws=500", "target_accept=0.99"])
    assert overrides["mcmc_draws"] == 500
    assert overrides["target_accept"] == 0.99


def test_parse_config_overrides_invalid_key() -> None:
    """An invalid configuration key must raise ValueError."""
    from co_president.__main__ import _parse_config_overrides  # noqa: PLC0415

    with pytest.raises(ValueError, match="mcmc_unicorns"):
        _parse_config_overrides(["mcmc_unicorns=100"])


def test_parse_config_overrides_type_cast() -> None:
    """Values must be cast to the annotated type (int vs float)."""
    from co_president.__main__ import _parse_config_overrides  # noqa: PLC0415

    result = _parse_config_overrides(["mcmc_chains=2", "seed=42"])
    assert result["mcmc_chains"] == 2
    assert isinstance(result["mcmc_chains"], int)
    assert result["seed"] == 42
    assert isinstance(result["seed"], int)


# ═══════════════════════════════════════════════════════════════════════
# CLI argument parsing tests
# ═══════════════════════════════════════════════════════════════════════


def test_cli_parser_accepts_run() -> None:
    """The argument parser must accept ``run`` as a subcommand."""
    from co_president.__main__ import _build_parser  # noqa: PLC0415

    parser = _build_parser()
    args = parser.parse_args(["run"])
    assert args.command == "run"
    assert not args.no_sample
    assert args.config_override == []


def test_cli_parser_accepts_run_no_sample() -> None:
    """Parser must accept ``run --no-sample``."""
    from co_president.__main__ import _build_parser  # noqa: PLC0415

    parser = _build_parser()
    args = parser.parse_args(["run", "--no-sample"])
    assert args.command == "run"
    assert args.no_sample


def test_cli_parser_accepts_config_overrides() -> None:
    """Parser must accept ``--config-override`` with multiple values."""
    from co_president.__main__ import _build_parser  # noqa: PLC0415

    parser = _build_parser()
    args = parser.parse_args(
        [
            "run",
            "--config-override",
            "mcmc_draws=500",
            "target_accept=0.99",
        ]
    )
    assert args.command == "run"
    assert args.config_override == ["mcmc_draws=500", "target_accept=0.99"]


def test_cli_parser_accepts_validate() -> None:
    """Parser must accept ``validate`` as a subcommand."""
    from co_president.__main__ import _build_parser  # noqa: PLC0415

    parser = _build_parser()
    args = parser.parse_args(["validate"])
    assert args.command == "validate"


def test_cli_parser_accepts_aggregate() -> None:
    """Parser must accept ``aggregate`` as a subcommand."""
    from co_president.__main__ import _build_parser  # noqa: PLC0415

    parser = _build_parser()
    args = parser.parse_args(["aggregate"])
    assert args.command == "aggregate"


def test_cli_parser_accepts_plot() -> None:
    """Parser must accept ``plot`` with optional output dir."""
    from co_president.__main__ import _build_parser  # noqa: PLC0415

    parser = _build_parser()
    args = parser.parse_args(["plot", "--output-dir", "my_results"])
    assert args.command == "plot"
    assert args.output_dir == "my_results"


def test_cli_parser_plot_default_output_dir() -> None:
    """Parser defaults ``--output-dir`` to ``results/``."""
    from co_president.__main__ import _build_parser  # noqa: PLC0415

    parser = _build_parser()
    args = parser.parse_args(["plot"])
    assert args.command == "plot"
    assert args.output_dir == "results/"


def test_cli_parser_accepts_config() -> None:
    """Parser must accept ``config`` as a subcommand."""
    from co_president.__main__ import _build_parser  # noqa: PLC0415

    parser = _build_parser()
    args = parser.parse_args(["config"])
    assert args.command == "config"
