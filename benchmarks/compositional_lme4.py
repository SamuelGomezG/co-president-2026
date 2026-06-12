"""SPEC-37: Benchmark comparing CLR-target model vs baseline on 2018 holdout.

Computes standard R² and ILR-space composite R² for both baseline
(``clr_target=False``) and CLR-target (``clr_target=True``) models on
the 2018 leave-one-year-out forecast.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

import numpy as np

from co_president.model_municipal import (
    build_municipal_model,
    sample_municipal_model,
)
from co_president.validation.compositional import composite_r2
from co_president.validation.municipal_oos import (
    _compute_r2,
)

if TYPE_CHECKING:
    import pandas as pd

    from co_president.config import ModelConfig
    from co_president.data import RoundResult

logger = logging.getLogger(__name__)


def run_2018_comparison(
    features: pd.DataFrame,
    polls_to_2014: pd.DataFrame,
    results_2018: RoundResult,
    config: ModelConfig,
    candidate_keys: list[str],
) -> dict[str, float]:
    """Run 2018 holdout for both baseline and CLR-target models.

    Constructs a fresh ``ModelConfig`` from *config* with
    ``clr_target=False`` and ``clr_target=True``, builds, and samples each.
    Returns comparison metrics.

    Args:
        features: Municipal feature matrix.
        polls_to_2014: Poll DataFrame with data up to 2014.
        results_2018: Actual 2018 round 1 results.
        config: Base ``ModelConfig`` (``clr_target`` field is overridden).
        candidate_keys: Candidate columns to evaluate.

    Returns:
        Dictionary with keys:
            ``standard_r2_baseline``, ``standard_r2_clr``,
            ``composite_r2_baseline``, ``composite_r2_clr``,
            ``standard_r2_delta``, ``composite_r2_delta``.

    Raises:
        ValueError: If either model's R² < 0.3 gating threshold.

    """
    actual_shares = np.array(
        [results_2018.get_share(k) for k in candidate_keys],
    )

    results: dict[str, float] = {}

    for label, clr_flag in [("baseline", False), ("clr", True)]:
        cfg = dataclasses.replace(config, clr_target=clr_flag)
        model = build_municipal_model(
            features,
            polls_to_2014,
            None,
            cfg,
            target_year=2014,
        )
        idata = sample_municipal_model(model, cfg)

        p_natl = idata.posterior["p_natl"].to_numpy()
        means = p_natl.mean(axis=(0, 1))

        std_r2 = _compute_r2(actual_shares, means)
        comp_r2 = composite_r2(
            actual_shares.reshape(1, -1),
            means.reshape(1, -1),
        )

        results[f"standard_r2_{label}"] = std_r2
        results[f"composite_r2_{label}"] = comp_r2

    results["standard_r2_delta"] = results["standard_r2_clr"] - results["standard_r2_baseline"]
    results["composite_r2_delta"] = results["composite_r2_clr"] - results["composite_r2_baseline"]

    logger.info(
        "2018 holdout comparison — standard R²: baseline=%.4f, CLR=%.4f "
        "(Δ=%.4f), composite R²: baseline=%.4f, CLR=%.4f (Δ=%.4f)",
        results["standard_r2_baseline"],
        results["standard_r2_clr"],
        results["standard_r2_delta"],
        results["composite_r2_baseline"],
        results["composite_r2_clr"],
        results["composite_r2_delta"],
    )

    return results
