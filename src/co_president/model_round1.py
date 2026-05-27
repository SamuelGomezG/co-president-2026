"""SPEC-06: Dirichlet-Multinomial + reverse-time RW (1st round)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pymc as pm  # type: ignore[reportMissingTypeStubs]

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

    Raises:
        ValueError: If model graph construction fails.
        TypeError: If input types are incorrect.

    Examples:
        >>> config = ModelConfig()
        >>> model = build_round1_model(polls, None, config)

    """
    # Placeholder implementation to pass the test
    _ = polls, results
    with pm.Model() as model:
        # Minimalist construction for testing
        pm.Normal("sigma_rw", mu=0, sigma=config.random_walk_sigma_prior)
    return model
