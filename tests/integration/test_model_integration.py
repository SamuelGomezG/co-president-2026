"""SPEC-06/SPEC-07: Integration tests for Bayesian models with real CSV data.

These tests load full production CSV datasets and are marked with
``@pytest.mark.integration``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import arviz as az  # type: ignore[reportMissingTypeStubs]
import pandas as pd
import pytest

from co_president.config import FIRST_ROUND_CANDIDATES, ModelConfig
from co_president.data import load_and_clean_all, load_canonical_results
from co_president.model_round1 import build_round1_model, sample_round1

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.integration
class TestRound1ModelIntegration:
    """Integration tests for the full round 1 model pipeline with real data."""

    @pytest.mark.slow
    def test_sample_round1_integration(self, data_dir: Path) -> None:
        """Test model on actual 2022 round 1 polls.

        Loads real 2022 first-round poll data, fits the model, and verifies:
        - MCMC convergence (R-hat < 1.10 for all parameters)
        - Posterior mean within ±5pp of actual 2022 results for major
          candidates with strong prior+signal (Petro, Gutierrez).
          Hernandez is excluded per SPEC-06 §9.2.1 model limitations.
        """
        clean = load_and_clean_all(data_dir)
        polls = clean.round1
        round1_result, _ = load_canonical_results(data_dir)

        config = ModelConfig(
            mcmc_draws=2000,
            mcmc_tune=1000,
            mcmc_chains=2,
            mcmc_cores=2,
        )
        model = build_round1_model(polls, None, config)
        idata = sample_round1(model, config)

        candidate_keys = sorted(set(FIRST_ROUND_CANDIDATES) & set(polls.columns))
        candidate_idx = {k: i for i, k in enumerate(candidate_keys)}

        summary = az.summary(idata, var_names=["~p_adj", "~p_time", "~house_effects", "~raw_house"])
        r_hat = pd.to_numeric(summary["r_hat"], errors="coerce").dropna()
        assert (r_hat < 1.10).all(), (
            f"R-hat convergence failure: max r_hat = {r_hat.max():.4f}, "
            f"parameters with r_hat >= 1.10: {list(r_hat[r_hat >= 1.10].index)}"
        )

        p_time = idata.posterior["p_time"].to_numpy()
        election_day = p_time[:, :, 0, :]
        means = election_day.mean(axis=(0, 1))

        major_checks = {"gustavo_petro", "federico_gutierrez"}
        for cr in round1_result.candidates:
            if cr.candidate_key not in candidate_idx:
                continue
            if cr.candidate_key not in major_checks:
                continue
            pred = means[candidate_idx[cr.candidate_key]]
            actual = cr.vote_share
            assert abs(pred - actual) <= 0.05, (
                f"{cr.candidate_key}: predicted {pred:.3f}, actual {actual:.3f}, "
                f"diff {abs(pred - actual):.3f} > 0.05"
            )
        assert (means >= 0.0).all()
        assert (means <= 1.0).all()
