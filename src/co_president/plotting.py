"""SPEC-09: Visualization functions for model output.

Produces matplotlib figures for forecast evolution, calibration,
and error-over-time plots.
"""

# pyright: reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportMissingTypeArgument=false
from __future__ import annotations

from itertools import cycle
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

if TYPE_CHECKING:
    from datetime import date

    from matplotlib.figure import Figure

    from co_president.data_results import RoundResult
    from co_president.model_round1 import Round1Forecast
    from co_president.validation import RoundValidation

_PLOT_STYLE = "seaborn-v0_8"

# Distinct (linestyle, marker) pairs for monochrome readability.
# Applied in order; cycles if there are more candidates than styles.
_LINE_MARKER_STYLES: list[tuple[str | tuple[int, tuple[int, ...]], str]] = [
    ("solid", "o"),
    ("dashed", "s"),
    ("dotted", "^"),
    ("dashdot", "D"),
    ((0, (1, 2)), "v"),
    ((0, (3, 1, 1, 1)), "p"),
    ((0, (5, 2)), "h"),
]


def plot_forecast_evolution(
    rolling: list[tuple[date, Round1Forecast]],
    results: RoundResult,
) -> Figure:
    """Plot predicted mean vote share over time with 95% CI and actual results.

    Args:
        rolling: List of (cutoff_date, forecast) tuples.
        results: Actual election results (horizontal reference lines).

    Returns:
        Matplotlib Figure.

    """
    with plt.style.context(_PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Collect candidate keys from the first forecast
        if not rolling:
            ax.set_title("No rolling forecast data available")
            return fig

        candidates = [c.candidate_key for c in rolling[0][1].candidates]

        style_cycle = cycle(_LINE_MARKER_STYLES)

        for candidate_key in candidates:
            ts: list[float] = []
            means: list[float] = []
            ci_low: list[float] = []
            ci_high: list[float] = []
            for cutoff_date, forecast in rolling:
                for fc in forecast.candidates:
                    if fc.candidate_key == candidate_key:
                        ts.append(cutoff_date.toordinal())
                        means.append(fc.mean_share * 100)
                        ci_low.append(fc.ci_95[0] * 100)
                        ci_high.append(fc.ci_95[1] * 100)
                        break
            if not ts:
                continue
            linestyle, marker = next(style_cycle)
            ax.fill_between(ts, ci_low, ci_high, alpha=0.2)
            ax.plot(
                ts,
                means,
                linestyle=linestyle,
                marker=marker,
                markersize=4,
                label=candidate_key.replace("_", " ").title(),
                linewidth=2,
            )

        # Actual results as horizontal dashed lines
        for c in results.candidates:
            ax.axhline(
                y=c.vote_share * 100,
                linestyle="--",
                color="gray",
                alpha=0.6,
                linewidth=0.8,
            )

        ax.set_xlabel("Date")
        ax.set_ylabel("Vote Share (%)")
        ax.set_title("Forecast Evolution Over Time")
        ax.legend(loc="best")
        fig.tight_layout()
    return fig


def plot_calibration(
    validation: RoundValidation,
) -> Figure:
    """Plot predicted vs actual vote share with error bars and 45-degree line.

    Error bars show the absolute prediction error magnitude (not credible
    intervals) since RoundValidation does not carry per-candidate CI data.

    Args:
        validation: Validated results for one round.

    Returns:
        Matplotlib Figure.

    """
    with plt.style.context(_PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 8))

        predicted = [cv.predicted_mean * 100 for cv in validation.candidates]
        actual = [cv.actual_share * 100 for cv in validation.candidates]
        labels = [cv.candidate_key.replace("_", " ").title() for cv in validation.candidates]

        # Compute xerr from the difference between predicted and actual
        xerr_lower = [abs(p - a) for p, a in zip(predicted, actual, strict=True)]
        xerr_values = [xerr_lower, xerr_lower]

        ax.errorbar(
            predicted,
            actual,
            xerr=xerr_values,
            fmt="o",
            capsize=4,
            markersize=6,
        )

        # 45-degree reference line (y=x)
        all_vals = predicted + actual
        lims = [min(all_vals) - 1, max(all_vals) + 1]
        ax.plot(lims, lims, "k--", alpha=0.5, label="Perfect calibration")

        for i, label in enumerate(labels):
            ax.annotate(label, (predicted[i], actual[i]), fontsize=8)

        ax.set_xlabel("Predicted Vote Share (%)")
        ax.set_ylabel("Actual Vote Share (%)")
        ax.set_title(f"Calibration Plot — Round {validation.round_number}")
        ax.legend(loc="lower right")
        ax.grid(visible=True, alpha=0.3)
        fig.tight_layout()
    return fig


def plot_error_over_time(
    rolling_errors: pd.DataFrame,
) -> Figure:
    """Plot MAE and RMSE over the rolling forecast timeline.

    Args:
        rolling_errors: DataFrame with ``as_of_date``, ``mae``, ``rmse`` columns.

    Returns:
        Matplotlib Figure.

    """
    with plt.style.context(_PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 6))

        dates = pd.to_datetime(rolling_errors["as_of_date"])
        ax.plot(dates, rolling_errors["mae"] * 100, marker="o", label="MAE")
        ax.plot(dates, rolling_errors["rmse"] * 100, marker="s", label="RMSE")

        ax.set_xlabel("Date")
        ax.set_ylabel("Error (percentage points)")
        ax.set_title("Forecast Error Over Time")
        ax.legend(loc="best")
        ax.grid(visible=True, alpha=0.3)
        fig.tight_layout()
    return fig
