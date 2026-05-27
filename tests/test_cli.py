"""Tests for SPEC-10: CLI entry point (initial scaffolding)."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile

from co_president.config import FIRST_ROUND_CANDIDATES
from co_president.model_round1 import CandidateForecast, Round1Forecast


def _write_sample_forecast(output_dir: str) -> Path:
    """Write a sample forecast JSON for use as --baseline-forecast argument.

    Args:
        output_dir: Directory to write the file in.

    Returns:
        Path to the written JSON file.

    """
    required_keys = {"gustavo_petro", "rodolfo_hernandez"}
    missing = required_keys - set(FIRST_ROUND_CANDIDATES)
    if missing:
        msg = f"Required candidate key(s) not found in FIRST_ROUND_CANDIDATES: {sorted(missing)}"
        raise AssertionError(msg)
    keys = sorted(required_keys & set(FIRST_ROUND_CANDIDATES))
    forecast = Round1Forecast(
        candidates=[
            CandidateForecast(
                candidate_key=k,
                mean_share=0.40,
                median_share=0.40,
                ci_50=(0.38, 0.42),
                ci_95=(0.35, 0.45),
                prob_first=0.5,
                prob_second=0.3,
                prob_top_two=0.8,
                prob_win_outright=0.1,
            )
            for k in keys
        ],
        prob_runoff=0.9,
        round_number=1,
    )
    path = Path(output_dir) / "baseline_forecast.json"
    path.write_text(forecast.to_json())
    return path


class TestValidateCommand:
    """Tests for the ``validate`` CLI subcommand."""

    def test_validate_help_prints(self) -> None:
        """``validate --help`` prints usage information."""
        result = subprocess.run(  # noqa: PLW1510
            [sys.executable, "-m", "co_president", "validate", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "threshold" in result.stdout

    def test_validate_requires_baseline(self) -> None:
        """``validate`` without --baseline-forecast exits with error."""
        result = subprocess.run(  # noqa: PLW1510
            [sys.executable, "-m", "co_president", "validate"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "baseline-forecast" in result.stderr

    def test_validate_with_missing_file(self) -> None:
        """``validate`` with a non-existent file path raises an error."""
        result = subprocess.run(  # noqa: PLW1510
            [
                sys.executable,
                "-m",
                "co_president",
                "validate",
                "--baseline-forecast",
                "/nonexistent/path.json",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0


class TestWriteSampleForecast:
    """Tests for the ``_write_sample_forecast`` helper."""

    def test_roundtrip(self) -> None:
        """Written JSON roundtrips through Round1Forecast.from_json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_sample_forecast(tmpdir)
            assert path.exists()
            raw = path.read_text(encoding="utf-8")
            restored = Round1Forecast.from_json(raw)
        assert len(restored.candidates) > 0
        assert restored.round_number == 1
        assert isinstance(restored.prob_runoff, float)


class TestConfigCommand:
    """Tests for the ``config`` CLI subcommand."""

    def test_config_prints_output(self) -> None:
        """``config`` prints configuration to stdout."""
        result = subprocess.run(  # noqa: PLW1510
            [sys.executable, "-m", "co_president", "config"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "CO-PRESIDENT" in result.stdout
        assert "Election" in result.stdout
        assert "Candidates" in result.stdout

    def test_config_contains_model_params(self) -> None:
        """``config`` output includes ModelConfig fields."""
        result = subprocess.run(  # noqa: PLW1510
            [sys.executable, "-m", "co_president", "config"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "mcmc_draws" in result.stdout
        assert "seed" in result.stdout
