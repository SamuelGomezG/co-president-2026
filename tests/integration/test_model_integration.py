"""SPEC-06/SPEC-07: Integration tests for Bayesian models with real CSV data.

These tests load full production CSV datasets and are marked with
``@pytest.mark.integration``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from co_president.config import ModelConfig
from co_president.data import load_and_clean_all
from co_president.model_round1 import build_round1_model

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.integration
class TestRound1ModelIntegration:
    """Integration tests for the full round 1 model pipeline with real data."""

    def test_build_round1_integration(self, data_dir: Path) -> None:
        """Verify model builds successfully with real 2022 round 1 data."""
        clean = load_and_clean_all(data_dir)
        polls = clean.round1

        config = ModelConfig()
        model = build_round1_model(polls, None, config, digital_signals=pd.DataFrame())

        assert model is not None
        assert "theta" in model.named_vars or any("theta" in name for name in model.named_vars)
        assert "p_time" in model.named_vars
        assert "p_adj" in model.named_vars
        assert "house_effects" in model.named_vars
        assert "poll_likelihood" in model.named_vars
