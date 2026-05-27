"""SPEC-06: Dirichlet-Multinomial + reverse-time RW (1st round)."""

# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
from __future__ import annotations

from typing import TYPE_CHECKING

import pymc as pm  # type: ignore[reportMissingTypeStubs]

from co_president.config import FIRST_ROUND_CANDIDATES

if TYPE_CHECKING:
    import pandas as pd

    from co_president.config import ModelConfig
    from co_president.data_results import RoundResult


def build_round1_model(
    polls: pd.DataFrame, results: RoundResult | None, config: ModelConfig
) -> pm.Model:
    """Build the PyMC model graph for the first round.

    Args:
        polls: DataFrame containing clean poll data.
        results: Canonical election results for validation, if available.
        config: Model hyperparameters.

    Returns:
        pm.Model: Constructed PyMC model.

    Examples:
        >>> config = ModelConfig()
        >>> model = build_round1_model(polls, None, config)

    """
    _ = results  # Reserved for future backtest/forecast mode
    num_candidates = len(FIRST_ROUND_CANDIDATES)
    pollsters = polls["encuestadora"].unique()
    num_pollsters = len(pollsters)

    with pm.Model() as model:
        # Priors
        pm.Normal("sigma_rw", mu=0, sigma=config.random_walk_sigma_prior)
        pm.HalfNormal("sigma_house", sigma=config.house_effect_sigma_prior)

        # theta[0] (election day)
        pm.Normal("theta", mu=0, sigma=1, shape=num_candidates)

        # House effects
        raw_house = pm.Normal("raw_house", mu=0, sigma=1, shape=(num_pollsters, num_candidates))
        pm.Deterministic(
            "house_effects",
            raw_house - raw_house.mean(axis=0, keepdims=True),
        )

        # p_adj placeholder
        pm.Deterministic("p_adj", pm.math.zeros(num_candidates))

    return model
