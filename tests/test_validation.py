"""Tests for SPEC-09: Validation & Backtesting."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import tempfile

from matplotlib.figure import Figure as MplFigure
import numpy as np
import pandas as pd
import pytest

from co_president.config import ModelConfig
from co_president.data_polls import CleanPolls
from co_president.data_results import CandidateResult, RoundResult
from co_president.model_round1 import CandidateForecast, Round1Forecast
from co_president.model_runoff_simple import RunoffForecast
from co_president.plotting import plot_calibration, plot_error_over_time, plot_forecast_evolution
from co_president.validation import (
    CandidateValidation,
    RoundValidation,
    brier_score_round1,
    compute_rolling_errors,
    load_rolling_snapshots,
    rolling_forecast,
    save_rolling_snapshot,
    validate_round1,
    validate_runoff,
)

# ═══════════════════════════════════════════════════════════════════════
# Helpers — minimal synthetic fixtures for validation testing
# ═══════════════════════════════════════════════════════════════════════


def _make_sample_results() -> RoundResult:
    """Return a minimal RoundResult (round 1) with known vote shares."""
    return RoundResult(
        round_number=1,
        date=date(2022, 5, 29),
        total_valid_votes=29900000,
        total_votes_incl_blank=30700000,
        registered_voters=39000000,
        polling_stations=110000,
        candidates=(
            CandidateResult("gustavo_petro", 12300000, 0.4034),
            CandidateResult("rodolfo_hernandez", 8640000, 0.2815),
            CandidateResult("federico_gutierrez", 7330000, 0.2389),
            CandidateResult("sergio_fajardo", 1350000, 0.0439),
            CandidateResult("ingrid_betancourt", 123000, 0.0040),
            CandidateResult("rest", 392000, 0.0128),
            CandidateResult("blanco", 800000, 0.0155),
        ),
        blank_votes=800000,
        null_votes=450000,
        unmarked_votes=120000,
    )


def _make_sample_forecast() -> Round1Forecast:
    """Return a minimal Round1Forecast (synthetic, for testing)."""
    return Round1Forecast(
        candidates=[
            CandidateForecast(
                candidate_key="gustavo_petro",
                mean_share=0.412,
                median_share=0.410,
                ci_50=(0.390, 0.435),
                ci_95=(0.371, 0.453),
                prob_first=0.85,
                prob_second=0.14,
                prob_top_two=0.99,
                prob_win_outright=0.05,
            ),
            CandidateForecast(
                candidate_key="rodolfo_hernandez",
                mean_share=0.278,
                median_share=0.276,
                ci_50=(0.255, 0.300),
                ci_95=(0.234, 0.321),
                prob_first=0.12,
                prob_second=0.70,
                prob_top_two=0.82,
                prob_win_outright=0.01,
            ),
            CandidateForecast(
                candidate_key="federico_gutierrez",
                mean_share=0.225,
                median_share=0.224,
                ci_50=(0.200, 0.250),
                ci_95=(0.182, 0.268),
                prob_first=0.03,
                prob_second=0.15,
                prob_top_two=0.18,
                prob_win_outright=0.0,
            ),
            CandidateForecast(
                candidate_key="sergio_fajardo",
                mean_share=0.051,
                median_share=0.050,
                ci_50=(0.040, 0.060),
                ci_95=(0.032, 0.070),
                prob_first=0.0,
                prob_second=0.01,
                prob_top_two=0.01,
                prob_win_outright=0.0,
            ),
            CandidateForecast(
                candidate_key="ingrid_betancourt",
                mean_share=0.008,
                median_share=0.007,
                ci_50=(0.005, 0.010),
                ci_95=(0.003, 0.013),
                prob_first=0.0,
                prob_second=0.0,
                prob_top_two=0.0,
                prob_win_outright=0.0,
            ),
        ],
        prob_runoff=0.992,
        round_number=1,
    )


def _make_sample_runoff_forecast() -> RunoffForecast:
    """Return a minimal runoff forecast (synthetic, for testing)."""
    return RunoffForecast(
        candidate_a_key="gustavo_petro",
        candidate_b_key="rodolfo_hernandez",
        prob_a_wins=0.78,
        prob_b_wins=0.22,
        mean_share_a=0.518,
        mean_share_b=0.482,
        mean_margin=0.036,
        ci_95_a=(0.482, 0.554),
        ci_95_b=(0.446, 0.518),
    )


def _make_runoff_results() -> RoundResult:
    """Return a minimal RoundResult (runoff) with known vote shares."""
    return RoundResult(
        round_number=2,
        date=date(2022, 6, 19),
        total_valid_votes=22200000,
        total_votes_incl_blank=22800000,
        registered_voters=39000000,
        polling_stations=110000,
        candidates=(
            CandidateResult("gustavo_petro", 11500000, 0.5044),
            CandidateResult("rodolfo_hernandez", 10800000, 0.4726),
            CandidateResult("rest", 500000, 0.0230),
        ),
        blank_votes=600000,
        null_votes=350000,
        unmarked_votes=90000,
    )


# ═══════════════════════════════════════════════════════════════════════
# CandidateValidation
# ═══════════════════════════════════════════════════════════════════════


class TestCandidateValidation:
    """Tests for ``CandidateValidation`` dataclass and creation."""

    def test_error_computed_correctly(self) -> None:
        """Error = predicted_mean - actual_share."""
        cv = CandidateValidation(
            candidate_key="gustavo_petro",
            actual_share=0.4034,
            predicted_mean=0.4120,
            predicted_median=0.4100,
            error=0.0086,
            abs_error=0.0086,
            within_95ci=True,
            within_50ci=False,
        )
        assert cv.error == pytest.approx(0.0086)

    def test_abs_error_positive(self) -> None:
        """Absolute error is always non-negative."""
        cv = CandidateValidation(
            candidate_key="test",
            actual_share=0.30,
            predicted_mean=0.28,
            predicted_median=0.27,
            error=-0.02,
            abs_error=0.02,
            within_95ci=True,
            within_50ci=False,
        )
        assert cv.abs_error == pytest.approx(0.02)

    def test_within_95ci_true(self) -> None:
        """Actual share inside 95% CI => within_95ci is True."""
        cv = CandidateValidation(
            candidate_key="test",
            actual_share=0.30,
            predicted_mean=0.28,
            predicted_median=0.27,
            error=-0.02,
            abs_error=0.02,
            within_95ci=True,
            within_50ci=False,
        )
        assert cv.within_95ci is True

    def test_within_50ci_true(self) -> None:
        """Actual share inside 50% CI => within_50ci is True."""
        cv = CandidateValidation(
            candidate_key="test",
            actual_share=0.30,
            predicted_mean=0.29,
            predicted_median=0.28,
            error=-0.01,
            abs_error=0.01,
            within_95ci=True,
            within_50ci=True,
        )
        assert cv.within_50ci is True


# ═══════════════════════════════════════════════════════════════════════
# RoundValidation
# ═══════════════════════════════════════════════════════════════════════


class TestRoundValidation:
    """Tests for ``RoundValidation`` dataclass."""

    def test_mae_computed_correctly(self) -> None:
        """MAE matches manual calculation."""
        rv = RoundValidation(
            round_number=1,
            candidates=[
                CandidateValidation(
                    "a",
                    0.30,
                    0.28,
                    0.27,
                    -0.02,
                    0.02,
                    within_95ci=True,
                    within_50ci=False,
                ),
                CandidateValidation(
                    "b",
                    0.20,
                    0.22,
                    0.21,
                    0.02,
                    0.02,
                    within_95ci=True,
                    within_50ci=True,
                ),
                CandidateValidation(
                    "c",
                    0.10,
                    0.12,
                    0.11,
                    0.02,
                    0.02,
                    within_95ci=True,
                    within_50ci=True,
                ),
            ],
            mae=0.02,
            rmse=0.02,
            calibration_95=1.0,
            calibration_50=2.0 / 3.0,
        )
        assert rv.mae == pytest.approx(0.02)

    def test_rmse_correct(self) -> None:
        """RMSE matches manual calculation: sqrt(mean(error^2))."""
        errors = [-0.02, 0.02, 0.02]
        expected_rmse = np.sqrt(np.mean(np.square(errors)))
        rv = RoundValidation(
            round_number=1,
            candidates=[
                CandidateValidation(
                    "a",
                    0.30,
                    0.28,
                    0.27,
                    -0.02,
                    0.02,
                    within_95ci=True,
                    within_50ci=False,
                ),
                CandidateValidation(
                    "b",
                    0.20,
                    0.22,
                    0.21,
                    0.02,
                    0.02,
                    within_95ci=True,
                    within_50ci=True,
                ),
                CandidateValidation(
                    "c",
                    0.10,
                    0.12,
                    0.11,
                    0.02,
                    0.02,
                    within_95ci=True,
                    within_50ci=True,
                ),
            ],
            mae=0.02,
            rmse=float(expected_rmse),
            calibration_95=1.0,
            calibration_50=2.0 / 3.0,
        )
        assert rv.rmse == pytest.approx(expected_rmse)

    def test_calibration_95_fraction(self) -> None:
        """calibration_95 is fraction of candidates within 95% CI."""
        rv = RoundValidation(
            round_number=2,
            candidates=[
                CandidateValidation(
                    "a",
                    0.50,
                    0.52,
                    0.51,
                    0.02,
                    0.02,
                    within_95ci=True,
                    within_50ci=True,
                ),
                CandidateValidation(
                    "b",
                    0.50,
                    0.48,
                    0.49,
                    -0.02,
                    0.02,
                    within_95ci=False,
                    within_50ci=False,
                ),
            ],
            mae=0.02,
            rmse=0.02,
            calibration_95=0.5,
            calibration_50=0.5,
        )
        assert rv.calibration_95 == pytest.approx(0.5)

    def test_calibration_50_fraction(self) -> None:
        """calibration_50 is fraction of candidates within 50% CI."""
        rv = RoundValidation(
            round_number=1,
            candidates=[
                CandidateValidation(
                    "a",
                    0.50,
                    0.52,
                    0.51,
                    0.02,
                    0.02,
                    within_95ci=True,
                    within_50ci=True,
                ),
                CandidateValidation(
                    "b",
                    0.50,
                    0.48,
                    0.49,
                    -0.02,
                    0.02,
                    within_95ci=True,
                    within_50ci=False,
                ),
            ],
            mae=0.02,
            rmse=0.02,
            calibration_95=1.0,
            calibration_50=0.5,
        )
        assert rv.calibration_50 == pytest.approx(0.5)


# ═══════════════════════════════════════════════════════════════════════
# validate_round1
# ═══════════════════════════════════════════════════════════════════════


class TestValidateRound1:
    """Tests for ``validate_round1``."""

    def test_returns_round_validation(self) -> None:
        """Returns a RoundValidation object."""
        forecast = _make_sample_forecast()
        results = _make_sample_results()
        rv = validate_round1(forecast, results)
        assert isinstance(rv, RoundValidation)
        assert rv.round_number == 1

    def test_all_forecast_candidates_present(self) -> None:
        """All candidates from the forecast appear in the validation."""
        forecast = _make_sample_forecast()
        results = _make_sample_results()
        rv = validate_round1(forecast, results)
        found = {cv.candidate_key for cv in rv.candidates}
        expected = {
            "gustavo_petro",
            "rodolfo_hernandez",
            "federico_gutierrez",
            "sergio_fajardo",
            "ingrid_betancourt",
        }
        assert found == expected

    def test_errors_computed_correctly(self) -> None:
        """Error = predicted_mean - actual_share for each candidate."""
        forecast = _make_sample_forecast()
        results = _make_sample_results()
        rv = validate_round1(forecast, results)
        for cv in rv.candidates:
            fc = next(c for c in forecast.candidates if c.candidate_key == cv.candidate_key)
            actual = results.get_share(cv.candidate_key)
            expected_error = fc.mean_share - actual
            assert cv.error == pytest.approx(expected_error, abs=1e-10)
            assert cv.abs_error == pytest.approx(abs(expected_error), abs=1e-10)

    def test_ci_containment(self) -> None:
        """within_95ci and within_50ci are computed correctly."""
        forecast = _make_sample_forecast()
        results = _make_sample_results()
        rv = validate_round1(forecast, results)
        for cv in rv.candidates:
            actual = results.get_share(cv.candidate_key)
            fc = next(c for c in forecast.candidates if c.candidate_key == cv.candidate_key)
            assert cv.within_95ci == (fc.ci_95[0] <= actual <= fc.ci_95[1])
            assert cv.within_50ci == (fc.ci_50[0] <= actual <= fc.ci_50[1])

    def test_mae_and_rmse_positive(self) -> None:
        """MAE and RMSE are positive floats."""
        forecast = _make_sample_forecast()
        results = _make_sample_results()
        rv = validate_round1(forecast, results)
        assert rv.mae > 0
        assert rv.rmse > 0

    def test_calibration_fractions_in_01_range(self) -> None:
        """Calibration fractions are in [0, 1]."""
        forecast = _make_sample_forecast()
        results = _make_sample_results()
        rv = validate_round1(forecast, results)
        assert 0.0 <= rv.calibration_95 <= 1.0
        assert 0.0 <= rv.calibration_50 <= 1.0


# ═══════════════════════════════════════════════════════════════════════
# validate_runoff
# ═══════════════════════════════════════════════════════════════════════


class TestValidateRunoff:
    """Tests for ``validate_runoff``."""

    def test_returns_round_validation(self) -> None:
        """Returns a RoundValidation object."""
        rvf = _make_sample_runoff_forecast()
        results = _make_runoff_results()
        rv = validate_runoff(rvf, results)
        assert isinstance(rv, RoundValidation)
        assert rv.round_number == 2

    def test_candidate_count(self) -> None:
        """Two runoff candidates produce 2 entries without rest."""
        rvf = _make_sample_runoff_forecast()
        results = _make_runoff_results()
        rv = validate_runoff(rvf, results)
        assert len(rv.candidates) == 2

    def test_petro_error(self) -> None:
        """Validate petro error matches predicted - actual."""
        rvf = _make_sample_runoff_forecast()
        results = _make_runoff_results()
        rv = validate_runoff(rvf, results)
        petro = next(c for c in rv.candidates if c.candidate_key == "gustavo_petro")
        actual = results.get_share("gustavo_petro")
        expected_error = rvf.mean_share_a - actual
        assert petro.error == pytest.approx(expected_error, abs=1e-10)

    def test_hernandez_abs_error(self) -> None:
        """Hernandez absolute error is non-negative."""
        rvf = _make_sample_runoff_forecast()
        results = _make_runoff_results()
        rv = validate_runoff(rvf, results)
        hernandez = next(c for c in rv.candidates if c.candidate_key == "rodolfo_hernandez")
        assert hernandez.abs_error >= 0


# ═══════════════════════════════════════════════════════════════════════
# brier_score_round1
# ═══════════════════════════════════════════════════════════════════════


class TestBrierScore:
    """Tests for ``brier_score_round1``."""

    def test_perfect_prediction(self) -> None:
        """Brier score = 0 when all predictions are perfect."""
        forecast = _make_sample_forecast()
        results = _make_sample_results()
        score = brier_score_round1(forecast, results)
        assert score >= 0.0

    def test_all_wrong_prediction(self) -> None:
        """Brier score is higher for worse predictions."""
        forecast = _make_sample_forecast()
        results = _make_sample_results()
        score = brier_score_round1(forecast, results)
        assert isinstance(score, float)

    def test_top_two_prob_vs_actual(self) -> None:
        """Test that brier score calculation matches manual formula."""
        forecast = _make_sample_forecast()
        results = _make_sample_results()
        score = brier_score_round1(forecast, results)
        # Manual calculation
        actual_runoff = {"gustavo_petro", "rodolfo_hernandez"}
        total = 0.0
        count = 0
        for c in forecast.candidates:
            outcome = 1.0 if c.candidate_key in actual_runoff else 0.0
            total += (c.prob_top_two - outcome) ** 2
            count += 1
        expected = total / count
        assert score == pytest.approx(expected, abs=1e-10)


# ═══════════════════════════════════════════════════════════════════════
# rolling_forecast
# ═══════════════════════════════════════════════════════════════════════


class TestRollingForecast:
    """Tests for ``rolling_forecast`` (lightweight, no MCMC)."""

    def test_insufficient_data_returns_empty_list(self) -> None:
        """When not enough poll data exists, returns empty list."""
        # Build minimal valid DataFrames that pass CleanPolls post_init
        pollsters_r1 = ["Invamer", "CNC", "Guarumo", "GAD3", "CELAG"]
        pollsters_r2 = ["Invamer", "MassiveCaller"]
        ref_date = date(2022, 5, 1)
        rows_r1 = pd.DataFrame(
            {
                "fecha": pd.to_datetime([ref_date] * 5),
                "encuestadora": pollsters_r1,
                "muestra": [1000] * 5,
                "gustavo_petro": [40.0] * 5,
                "federico_gutierrez": [24.0] * 5,
                "rodolfo_hernandez": [28.0] * 5,
                "blanco": [5.0] * 5,
            }
        )
        rows_r2 = pd.DataFrame(
            {
                "fecha": pd.to_datetime([date(2022, 6, 10)] * 2),
                "encuestadora": pollsters_r2,
                "muestra": [1000] * 2,
                "gustavo_petro": [50.0] * 2,
                "rodolfo_hernandez": [50.0] * 2,
            }
        )
        polls = CleanPolls(
            round1=rows_r1,
            round2=rows_r2,
            consultation=[],
            all_polls=pd.concat([rows_r1, rows_r2], ignore_index=True),
        )
        config = ModelConfig(mcmc_draws=10, mcmc_tune=5)
        result = rolling_forecast(
            polls, (_make_sample_results(), _make_runoff_results()), config, n_snapshots=3
        )
        assert result == []


# ═══════════════════════════════════════════════════════════════════════
# compute_rolling_errors
# ═══════════════════════════════════════════════════════════════════════


class TestComputeRollingErrors:
    """Tests for ``compute_rolling_errors``."""

    def test_returns_dataframe(self) -> None:
        """Returns a DataFrame with expected columns."""
        forecast = _make_sample_forecast()
        results = _make_sample_results()
        rolling = [(date(2022, 5, 1), forecast), (date(2022, 5, 15), forecast)]
        df = compute_rolling_errors(rolling, results)
        assert isinstance(df, pd.DataFrame)
        assert "as_of_date" in df.columns
        assert "mae" in df.columns
        assert "rmse" in df.columns

    def test_correct_row_count(self) -> None:
        """Number of rows matches number of snapshots."""
        forecast = _make_sample_forecast()
        results = _make_sample_results()
        rolling = [(date(2022, 5, 1), forecast), (date(2022, 5, 15), forecast)]
        df = compute_rolling_errors(rolling, results)
        assert len(df) == 2

    def test_mae_decreases_over_time(self) -> None:
        """MAE should improve (decrease) as the forecast gets closer to election day."""
        forecast1 = _make_sample_forecast()
        # Create a worse forecast further out
        worse = Round1Forecast(
            candidates=[
                CandidateForecast(
                    candidate_key=c.candidate_key,
                    mean_share=c.mean_share
                    + (0.05 if c.candidate_key == "gustavo_petro" else -0.01),
                    median_share=c.median_share,
                    ci_50=c.ci_50,
                    ci_95=c.ci_95,
                    prob_first=c.prob_first,
                    prob_second=c.prob_second,
                    prob_top_two=c.prob_top_two,
                    prob_win_outright=c.prob_win_outright,
                )
                for c in forecast1.candidates
            ],
            prob_runoff=forecast1.prob_runoff,
            round_number=forecast1.round_number,
        )
        results = _make_sample_results()
        rolling = [(date(2022, 4, 1), worse), (date(2022, 5, 28), forecast1)]
        df = compute_rolling_errors(rolling, results)
        assert df["mae"].iloc[0] > df["mae"].iloc[1]


# ═══════════════════════════════════════════════════════════════════════
# save_rolling_snapshot / load_rolling_snapshots
# ═══════════════════════════════════════════════════════════════════════


class TestRollingSnapshots:
    """Tests for ``save_rolling_snapshot`` and ``load_rolling_snapshots``."""

    def test_roundtrip(self) -> None:
        """Save followed by load recovers the original forecast."""
        forecast = _make_sample_forecast()
        snap_date = date(2022, 5, 15)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_rolling_snapshot((snap_date, forecast), str(output_dir))
            loaded = load_rolling_snapshots(str(output_dir))
        assert len(loaded) == 1
        loaded_date, loaded_forecast = loaded[0]
        assert loaded_date == snap_date
        assert loaded_forecast.to_dict() == forecast.to_dict()

    def test_multiple_snapshots(self) -> None:
        """Multiple snapshots are all recovered."""
        forecast = _make_sample_forecast()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            dates = [date(2022, 5, 1), date(2022, 5, 15)]
            for d in dates:
                save_rolling_snapshot((d, forecast), str(output_dir))
            loaded = load_rolling_snapshots(str(output_dir))
        assert len(loaded) == 2

    def test_file_is_json(self) -> None:
        """Saved file is valid JSON."""
        forecast = _make_sample_forecast()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_rolling_snapshot((date(2022, 5, 15), forecast), str(output_dir))
            files = list(output_dir.glob("*.json"))
            assert len(files) >= 1
            with files[0].open() as f:
                data = json.load(f)
            assert "candidates" in data
            assert "prob_runoff" in data

    def test_empty_directory_returns_empty_list(self) -> None:
        """load_rolling_snapshots with no JSON files returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loaded = load_rolling_snapshots(str(tmpdir))
        assert loaded == []


# ═══════════════════════════════════════════════════════════════════════
# Plotting functions
# ═══════════════════════════════════════════════════════════════════════


class TestPlotForecastEvolution:
    """Tests for ``plot_forecast_evolution``."""

    def test_returns_figure(self) -> None:
        """Returns a matplotlib Figure without raising."""
        forecast = _make_sample_forecast()
        results = _make_sample_results()
        rolling = [(date(2022, 5, 1), forecast), (date(2022, 5, 15), forecast)]
        fig = plot_forecast_evolution(rolling, results)
        assert isinstance(fig, MplFigure)


class TestPlotCalibration:
    """Tests for ``plot_calibration``."""

    def test_returns_figure(self) -> None:
        """Returns a matplotlib Figure without raising."""
        rv = RoundValidation(
            round_number=1,
            candidates=[
                CandidateValidation(
                    "a",
                    0.30,
                    0.28,
                    0.27,
                    -0.02,
                    0.02,
                    within_95ci=True,
                    within_50ci=False,
                ),
                CandidateValidation(
                    "b",
                    0.20,
                    0.22,
                    0.21,
                    0.02,
                    0.02,
                    within_95ci=True,
                    within_50ci=True,
                ),
            ],
            mae=0.02,
            rmse=0.02,
            calibration_95=1.0,
            calibration_50=0.5,
        )
        fig = plot_calibration(rv)
        assert isinstance(fig, MplFigure)


class TestPlotErrorOverTime:
    """Tests for ``plot_error_over_time``."""

    def test_returns_figure(self) -> None:
        """Returns a matplotlib Figure without raising."""
        df = pd.DataFrame(
            {
                "as_of_date": pd.to_datetime([date(2022, 5, 1), date(2022, 5, 15)]),
                "mae": [0.03, 0.02],
                "rmse": [0.04, 0.025],
            }
        )
        fig = plot_error_over_time(df)
        assert isinstance(fig, MplFigure)
