"""SPEC-07: Dirichlet-Multinomial (K=3) runoff model.

Builds and samples a Bayesian runoff election forecast between the top two
first-round candidates. The model uses a Dirichlet-Multinomial likelihood
with a reverse-time random walk and hierarchical pollster house effects,
mirroring the structure of ``model_round1`` but with ``K=3`` (candidate A,
candidate B, rest+blanco).

Public functions: :func:`build_runoff_simple_model`, :func:`sample_runoff`,
:func:`forecast_runoff_simple`, and :class:`RunoffForecast`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import arviz as az  # type: ignore[reportMissingTypeStubs]
import numpy as np
import pandas as pd
import pymc as pm  # type: ignore[reportMissingTypeStubs]

from co_president.config import (
    ELECTION_DATE_ROUND2,
    FIRST_ROUND_CANDIDATES,
)

if TYPE_CHECKING:
    from co_president.config import ModelConfig
    from co_president.data import RoundResult


@dataclass(frozen=True)
class RunoffForecast:
    """Forecast for a runoff election between two candidates.

    Attributes:
        candidate_a_key: Unique identifier for candidate A.
        candidate_b_key: Unique identifier for candidate B.
        prob_a_wins: Probability candidate A receives more votes than B.
        prob_b_wins: Probability candidate B receives more votes than A.
        mean_share_a: Posterior mean vote share for candidate A.
        mean_share_b: Posterior mean vote share for candidate B.
        mean_margin: Mean margin (mean_share_a - mean_share_b).
        ci_95_a: 95% credible interval for candidate A's vote share.
        ci_95_b: 95% credible interval for candidate B's vote share.

    """

    candidate_a_key: str
    candidate_b_key: str
    prob_a_wins: float
    prob_b_wins: float
    mean_share_a: float
    mean_share_b: float
    mean_margin: float
    ci_95_a: tuple[float, float]
    ci_95_b: tuple[float, float]


def build_runoff_simple_model(  # noqa: PLR0915
    polls: pd.DataFrame,
    results: RoundResult,
    round1_idata: az.InferenceData | None,
    config: ModelConfig,
) -> pm.Model:
    """Build the PyMC model graph for the runoff (K=3).

    Constructs a Dirichlet-Multinomial observation model with a reverse-time
    random walk and hierarchical pollster house effects, for the head-to-head
    runoff between the top two candidates from the first round. The third
    category (rest+blanco) serves as the reference category.

    Args:
        polls: DataFrame containing clean Round 2 poll data. Must include
            columns for the two runoff candidates' vote shares (%), ``fecha``,
            ``encuestadora``, and ``muestra``.
        results: First-round election result used to identify the top two
            candidates and to compute the prior for ``theta[T-1]``.
        round1_idata: Posterior samples from the Round 1 model. If provided,
            the prior for ``theta[T-1]`` uses the posterior means of the
            top-two candidates' vote shares. If ``None``, the Round 1 actual
            election results are used as an informed prior instead (the
            fallback path for when the Round 1 model hasn't been run).
        config: Model hyperparameters.

    Returns:
        pm.Model: Constructed PyMC model.

    Raises:
        ValueError: If runoff candidates are not found in ``polls`` columns,
            or if all polls have NaN candidate shares and no data remains.

    Examples:
        >>> from co_president.config import ModelConfig
        >>> config = ModelConfig()
        >>> model = build_runoff_simple_model(polls, results, None, config)
        >>> list(model.named_vars.keys())
        ['sigma_rw', 'sigma_house', ...]

    """
    polls = polls.copy()

    if not pd.api.types.is_datetime64_any_dtype(polls["fecha"]):
        polls["fecha"] = pd.to_datetime(polls["fecha"])

    # Identify runoff candidates from first-round result
    top_two = results.top_two()
    cand_a_key = top_two[0].candidate_key
    cand_b_key = top_two[1].candidate_key

    # Build K=3 columns: [cand_a, cand_b, rest_blanco]
    candidate_keys = sorted(set(FIRST_ROUND_CANDIDATES) & set(polls.columns))

    if cand_a_key not in polls.columns or cand_b_key not in polls.columns:
        msg = f"Runoff candidates {cand_a_key}, {cand_b_key} not found in polls"
        raise ValueError(msg)

    other_keys = [k for k in candidate_keys if k not in (cand_a_key, cand_b_key)]

    work = polls[["fecha", "encuestadora", "muestra", cand_a_key, cand_b_key]].copy()
    work["rest_blanco"] = polls[other_keys].sum(axis=1, min_count=1)
    k_runoff_cols: list[str] = [cand_a_key, cand_b_key, "rest_blanco"]
    n_candidates = 3

    # Drop rows with NaN in candidate columns or muestra
    nan_mask = work[[*k_runoff_cols, "muestra"]].isna().any(axis=1)
    if nan_mask.any():
        work = work.loc[~nan_mask].copy()
        if len(work) == 0:
            msg = "All runoff polls have NaN candidate shares; no data remains"
            raise ValueError(msg)

    # Time index anchored on runoff election date
    work["days_before"] = (pd.Timestamp(ELECTION_DATE_ROUND2) - work["fecha"]).dt.days
    unique_days = sorted(work["days_before"].unique())
    n_time_points = len(unique_days)
    day_to_idx = {day: i for i, day in enumerate(unique_days)}
    work["time_idx"] = work["days_before"].map(day_to_idx)

    # Pollster index
    unique_pollsters = work["encuestadora"].unique()
    n_pollsters = len(unique_pollsters)
    pollster_to_idx = {name: i for i, name in enumerate(unique_pollsters)}
    work["pollster_idx"] = work["encuestadora"].map(pollster_to_idx)

    # Observed counts
    sample_sizes = work["muestra"].to_numpy().astype(int)
    observed_counts = np.round(
        work[k_runoff_cols].to_numpy() / 100.0 * sample_sizes[:, np.newaxis],
    ).astype(int)
    effective_n = observed_counts.sum(axis=1)

    # Sample-size-dependent concentration multiplier
    eps = 1e-8
    mean_effective_n = effective_n.mean()
    numerator = np.log(effective_n + 1 + eps)
    denominator = np.log(mean_effective_n + 1 + eps)
    effective_n_multiplier = np.maximum(numerator / denominator, eps)[:, np.newaxis]

    time_indices = work["time_idx"].to_numpy().astype(int)
    pollster_indices = work["pollster_idx"].to_numpy().astype(int)

    # Compute prior for theta[T-1] from round1 posterior or actual results
    # Note: round1_idata.posterior["p_time"] is assumed to have candidate
    # probabilities ordered by sorted(FIRST_ROUND_CANDIDATES.keys()), which
    # matches the candidate_keys order used in build_round1_model. The indices
    # a_idx / b_idx rely on this ordering to correctly extract posterior means
    # for the two runoff candidates.
    if round1_idata is not None:
        candidate_order = sorted(FIRST_ROUND_CANDIDATES.keys())
        a_idx = candidate_order.index(cand_a_key)
        b_idx = candidate_order.index(cand_b_key)

        p_time = round1_idata.posterior["p_time"]
        election_day = p_time[:, :, 0, :]
        mean_shares = election_day.mean(dim=("chain", "draw")).to_numpy()

        p_a = max(float(mean_shares[a_idx]), eps)
        p_b = max(float(mean_shares[b_idx]), eps)
        p_rest_runoff = max(1.0 - p_a - p_b, eps)
    else:
        p_a = max(results.get_share(cand_a_key), eps)
        p_b = max(results.get_share(cand_b_key), eps)
        p_rest_runoff = max(1.0 - p_a - p_b, eps)

    prior_mean_a = float(np.log(p_a / p_rest_runoff))
    prior_mean_b = float(np.log(p_b / p_rest_runoff))
    prior_mean_free = np.array([prior_mean_a, prior_mean_b], dtype=float)

    with pm.Model() as model:  # type: ignore
        sigma_rw = pm.HalfNormal(  # type: ignore
            "sigma_rw",
            sigma=config.random_walk_sigma_prior,
        )
        sigma_house = pm.HalfNormal(  # type: ignore
            "sigma_house",
            sigma=config.house_effect_sigma_prior,
        )
        phi_poll = pm.Gamma(  # type: ignore
            "phi_poll",
            alpha=2,
            beta=2.0 / config.concentration_poll_prior_mean,
        )
        phi_poll_n = pm.Deterministic(  # type: ignore
            "phi_poll_n",
            phi_poll * effective_n_multiplier,
        )

        # Reverse-time random walk with reference parameterization
        # theta has shape (T, K) where K=3: [theta_A, theta_B, 0 (rest reference)]
        theta_rev = []  # type: ignore[reportMissingTypeArgument]
        for t_idx in range(n_time_points - 1, -1, -1):
            if t_idx == n_time_points - 1:
                theta_free = pm.Normal(  # type: ignore
                    f"theta_r_{t_idx}",
                    mu=prior_mean_free,
                    sigma=0.5,
                    shape=2,
                )
            else:
                theta_free = pm.Normal(  # type: ignore
                    f"theta_r_{t_idx}",
                    mu=theta_rev[-1],  # type: ignore
                    sigma=sigma_rw,
                    shape=2,
                )
            theta_rev.append(theta_free)  # type: ignore

        theta_free_stacked = pm.math.stack(list(reversed(theta_rev)), axis=0)  # type: ignore
        zero_rest = pm.math.zeros((n_time_points, 1))  # type: ignore
        theta = pm.math.concatenate([theta_free_stacked, zero_rest], axis=-1)  # type: ignore

        # Latent vote share probabilities per time point (K=3)
        pm.Deterministic("p_time", pm.math.softmax(theta, axis=-1))  # type: ignore

        # House effects (zero-sum constrained, K=3)
        raw_house = pm.Normal(  # type: ignore
            "raw_house",
            mu=0,
            sigma=sigma_house,
            shape=(n_pollsters, n_candidates),
        )
        house_effects = pm.Deterministic(  # type: ignore
            "house_effects",
            raw_house - raw_house.mean(axis=0, keepdims=True),  # type: ignore
        )

        # Poll observation model
        theta_selected = theta[time_indices]  # type: ignore
        house_selected = house_effects[pollster_indices]  # type: ignore
        theta_adj = theta_selected + house_selected  # type: ignore

        p_adj = pm.Deterministic("p_adj", pm.math.softmax(theta_adj, axis=-1))  # type: ignore
        alpha_poll = pm.math.maximum(p_adj * phi_poll_n, eps)  # type: ignore

        pm.DirichletMultinomial(  # type: ignore
            "poll_likelihood",
            n=effective_n,
            a=alpha_poll,
            observed=observed_counts,
        )

    return model


def sample_runoff(model: pm.Model, config: ModelConfig) -> az.InferenceData:
    """Sample posterior draws from a built runoff model via NUTS.

    Args:
        model: A built ``pm.Model`` from :func:`build_runoff_simple_model`.
        config: Model hyperparameters (draws, tune, chains, etc.).

    Returns:
        az.InferenceData: Posterior and sampling stats.

    Raises:
        RuntimeError: If NUTS sampling fails to initialise or diverges.

    Examples:
        >>> model = build_runoff_simple_model(polls, results, None, ModelConfig())
        >>> idata = sample_runoff(model, ModelConfig(mcmc_draws=200, mcmc_tune=100))

    """
    with model:
        return pm.sample(  # type: ignore
            draws=config.mcmc_draws,
            tune=config.mcmc_tune,
            chains=config.mcmc_chains,
            cores=config.mcmc_cores,
            target_accept=config.target_accept,
            random_seed=config.seed,
        )


def forecast_runoff_simple(
    idata: az.InferenceData,
    candidate_a: str,
    candidate_b: str,
) -> RunoffForecast:
    """Compute a runoff forecast from posterior samples.

    Extracts election-day (time=0) vote shares for candidate A, candidate B,
    and rest+blanco, then computes win probabilities, mean shares, mean margin,
    and 95% credible intervals.

    Args:
        idata: Posterior samples from :func:`build_runoff_simple_model`.
        candidate_a: Key of the first runoff candidate.
        candidate_b: Key of the second runoff candidate.

    Returns:
        RunoffForecast with computed forecast statistics.

    Examples:
        >>> idata = sample_runoff(model, ModelConfig())
        >>> forecast = forecast_runoff_simple(idata, "gustavo_petro", "rodolfo_hernandez")
        >>> forecast.prob_a_wins
        0.65

    """
    p_time = idata.posterior["p_time"]  # (chain, draw, time, candidate=3)
    election_day = p_time[:, :, 0, :]  # (chain, draw, 3) -> [A, B, rest]

    a_values = election_day[:, :, 0].to_numpy().flatten()
    b_values = election_day[:, :, 1].to_numpy().flatten()

    prob_a_wins = float((a_values > b_values).mean())
    prob_b_wins = float((b_values > a_values).mean())

    mean_share_a = float(a_values.mean())
    mean_share_b = float(b_values.mean())
    mean_margin = mean_share_a - mean_share_b

    ci_95_a_result = az.hdi(a_values, prob=0.95)  # type: ignore
    ci_95_b_result = az.hdi(b_values, prob=0.95)  # type: ignore
    ci_95_a = (float(ci_95_a_result[0]), float(ci_95_a_result[1]))  # type: ignore
    ci_95_b = (float(ci_95_b_result[0]), float(ci_95_b_result[1]))  # type: ignore

    return RunoffForecast(
        candidate_a_key=candidate_a,
        candidate_b_key=candidate_b,
        prob_a_wins=prob_a_wins,
        prob_b_wins=prob_b_wins,
        mean_share_a=mean_share_a,
        mean_share_b=mean_share_b,
        mean_margin=mean_margin,
        ci_95_a=ci_95_a,
        ci_95_b=ci_95_b,
    )
