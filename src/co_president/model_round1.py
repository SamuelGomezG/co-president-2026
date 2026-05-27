"""SPEC-06: Dirichlet-Multinomial + reverse-time RW (1st round)."""

# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportMissingTypeArgument=false, reportIndexIssue=false, reportOperatorIssue=false
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pymc as pm  # type: ignore[reportMissingTypeStubs]

from co_president.config import (
    CONSULTATION_VOTES,
    ELECTION_DATE_ROUND1,
    FIRST_ROUND_CANDIDATES,
    consultation_log_share_prior,
)

if TYPE_CHECKING:
    from co_president.config import ModelConfig
    from co_president.data_results import RoundResult


def build_round1_model(  # noqa: PLR0915
    polls: pd.DataFrame, results: RoundResult | None, config: ModelConfig
) -> pm.Model:
    """Build the PyMC model graph for the first round.

    Constructs a Dirichlet-Multinomial observation model with a reverse-time
    random walk for latent vote intention and hierarchical house effects.

    Args:
        polls: DataFrame containing clean poll data. Must include columns for
            each active candidate's vote share (%), ``fecha`` (dates),
            ``encuestadora`` (pollster name), and ``muestra`` (sample size).
        results: Canonical election results for validation, if available.
            When provided, an election-day likelihood term is included.
        config: Model hyperparameters.

    Returns:
        pm.Model: Constructed PyMC model.

    Raises:
        ValueError: If ``polls`` is empty or missing required columns.

    Examples:
        >>> config = ModelConfig()
        >>> model = build_round1_model(polls, None, config)

    """
    # Convert fecha to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(polls["fecha"]):
        polls = polls.copy()
        polls["fecha"] = pd.to_datetime(polls["fecha"])

    # Determine candidate keys from DataFrame columns
    candidate_keys = sorted(set(FIRST_ROUND_CANDIDATES) & set(polls.columns))
    n_candidates = len(candidate_keys)
    if n_candidates == 0:
        msg = "No candidate columns found in polls DataFrame"
        raise ValueError(msg)

    # Build time index mapping: 0 = election day, n_time_points-1 = farthest back
    polls = polls.copy()
    polls["days_before"] = (pd.Timestamp(ELECTION_DATE_ROUND1) - polls["fecha"]).dt.days
    unique_days = sorted(polls["days_before"].unique())
    n_time_points = len(unique_days)
    day_to_idx = {day: i for i, day in enumerate(unique_days)}
    polls["time_idx"] = polls["days_before"].map(day_to_idx)

    # Build pollster index mapping
    unique_pollsters = polls["encuestadora"].unique()
    n_pollsters = len(unique_pollsters)
    pollster_to_idx = {name: i for i, name in enumerate(unique_pollsters)}
    polls["pollster_idx"] = polls["encuestadora"].map(pollster_to_idx)

    # Observed poll counts from percentage shares
    sample_sizes = polls["muestra"].to_numpy().astype(int)
    observed_counts = np.round(
        polls[candidate_keys].to_numpy() / 100.0 * sample_sizes[:, np.newaxis],
    ).astype(int)

    # Index arrays
    time_indices = polls["time_idx"].to_numpy().astype(int)
    pollster_indices = polls["pollster_idx"].to_numpy().astype(int)

    # Consultation log-share prior for theta[T-1] (earliest time point)
    log_prior = consultation_log_share_prior()

    # Default log-share for candidates without consultation data
    nonzero_total = sum(v for v in CONSULTATION_VOTES.values() if v > 0)
    if nonzero_total > 0:
        nonzero_shares = [v / nonzero_total for v in CONSULTATION_VOTES.values() if v > 0]
        default_log_share = float(np.log(min(nonzero_shares) / 2.0))
    else:
        default_log_share = -1.0

    prior_mean = np.array(
        [log_prior.get(k, default_log_share) for k in candidate_keys],
        dtype=float,
    )

    with pm.Model() as model:
        sigma_rw = pm.HalfNormal(
            "sigma_rw",
            sigma=config.random_walk_sigma_prior,
        )
        sigma_house = pm.HalfNormal(
            "sigma_house",
            sigma=config.house_effect_sigma_prior,
        )
        phi_poll = pm.Gamma(
            "phi_poll",
            alpha=2,
            beta=2.0 / config.concentration_poll_prior_mean,
        )

        # Reverse-time random walk
        theta_rev: list = []
        for t_idx in range(n_time_points - 1, -1, -1):
            if t_idx == n_time_points - 1:
                theta_t = pm.Normal(
                    f"theta_{t_idx}",
                    mu=prior_mean,
                    sigma=config.consultation_prior_strength,
                    shape=n_candidates,
                )
            else:
                theta_t = pm.Normal(
                    f"theta_{t_idx}",
                    mu=theta_rev[-1],
                    sigma=sigma_rw,
                    shape=n_candidates,
                )
            theta_rev.append(theta_t)

        theta_stacked = pm.math.stack(list(reversed(theta_rev)), axis=0)  # noqa: PD013

        # House effects (zero-sum constrained)
        raw_house = pm.Normal(
            "raw_house",
            mu=0,
            sigma=sigma_house,
            shape=(n_pollsters, n_candidates),
        )
        house_effects = pm.Deterministic(
            "house_effects",
            raw_house - raw_house.mean(axis=0, keepdims=True),
        )

        # Poll observation model
        theta_selected = theta_stacked[time_indices]
        house_selected = house_effects[pollster_indices]
        theta_adj = theta_selected + house_selected

        p_adj = pm.Deterministic("p_adj", pm.math.softmax(theta_adj, axis=-1))
        alpha_poll = p_adj * phi_poll

        pm.DirichletMultinomial(
            "poll_likelihood",
            n=sample_sizes,
            a=alpha_poll,
            observed=observed_counts,
        )

        # Election result likelihood (optional)
        if results is not None:
            p_elec = pm.Deterministic(
                "p_elec",
                pm.math.softmax(theta_stacked[0], axis=-1),
            )
            phi_elec = pm.Gamma(
                "phi_elec",
                alpha=5,
                beta=5.0 / config.concentration_election_prior_mean,
            )
            alpha_elec = p_elec * phi_elec

            vote_dict = {c.candidate_key: c.votes for c in results.candidates}
            election_counts = np.array(
                [vote_dict.get(k, 0) for k in candidate_keys],
                dtype=int,
            )

            pm.DirichletMultinomial(
                "election_likelihood",
                n=results.total_votes_incl_blank,
                a=alpha_elec,
                observed=election_counts,
            )

    return model
