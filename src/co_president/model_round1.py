"""SPEC-06: Dirichlet-Multinomial + reverse-time RW (1st round)."""

# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportMissingTypeArgument=false, reportIndexIssue=false, reportOperatorIssue=false
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import TYPE_CHECKING, Any, Literal

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
    import arviz as az  # type: ignore[reportMissingTypeStubs]

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
    # Make a working copy to avoid mutating the input
    polls = polls.copy()

    # Convert fecha to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(polls["fecha"]):
        polls["fecha"] = pd.to_datetime(polls["fecha"])

    # Determine candidate keys from DataFrame columns
    candidate_keys = sorted(set(FIRST_ROUND_CANDIDATES) & set(polls.columns))
    n_candidates = len(candidate_keys)
    if n_candidates == 0:
        msg = "No candidate columns found in polls DataFrame"
        raise ValueError(msg)

    # Drop rows where any candidate share is NaN — those polls cannot
    # contribute useful likelihood information for all candidates
    nan_mask = polls[[*candidate_keys, "muestra"]].isna().any(axis=1)
    if nan_mask.any():
        polls = polls.loc[~nan_mask].copy()
        # Recalculate after filtering
        if len(polls) == 0:
            msg = "All polls have NaN candidate shares; no data remains"
            raise ValueError(msg)

    # Build time index mapping: 0 = election day, n_time_points-1 = farthest back
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

    # Observed poll counts from percentage shares.
    # The DirichletMultinomial trial count is the raw sample size (SPEC-06 §9.1),
    # not the sum of rounded per-candidate counts.
    sample_sizes = polls["muestra"].to_numpy().astype(int)
    observed_counts = np.round(
        polls[candidate_keys].to_numpy() / 100.0 * sample_sizes[:, np.newaxis],
    ).astype(int)

    # Sample-size-dependent concentration multiplier:
    # Larger polls contribute more to the concentration parameter via log-based
    # scaling (phi_poll_n = phi_poll * log(N+1) / log(mean_N+1), sublinear to
    # avoid over-weighting extremely large samples)
    eps = 1e-8
    mean_sample_size = sample_sizes.mean()
    numerator = np.log(sample_sizes + 1 + eps)
    denominator = np.log(mean_sample_size + 1 + eps)
    sample_size_multiplier = np.maximum(numerator / denominator, eps)[:, np.newaxis]

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
        phi_poll_n = pm.Deterministic(
            "phi_poll_n",
            phi_poll * sample_size_multiplier,
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

        # Latent vote share probabilities per time point
        # (used for prior predictive validation and plotting)
        pm.Deterministic("p_time", pm.math.softmax(theta_stacked, axis=-1))

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
        alpha_poll = pm.math.maximum(p_adj * phi_poll_n, eps)

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
                n=election_counts.sum(),
                a=alpha_elec,
                observed=election_counts,
            )

    return model


def sample_round1(model: pm.Model, config: ModelConfig) -> az.InferenceData:
    """Sample posterior draws from a built first-round model via NUTS.

    Runs MCMC sampling with the hyperparameters specified in *config*.

    Args:
        model: A built ``pm.Model`` from :func:`build_round1_model`.
        config: Model hyperparameters (draws, tune, chains, etc.).

    Raises:
        ValueError: If ``sample_round1`` receives invalid sampling inputs and
            PyMC's NUTS/MCMC sampler rejects them.
        RuntimeError: If ``sample_round1`` fails during NUTS/MCMC sampling
            initialization or execution.

    Returns:
        az.InferenceData: Posterior and sampling stats only. Posterior
        predictive samples require ``pm.sample_posterior_predictive``.

    Examples:
        >>> model = build_round1_model(polls, None, ModelConfig())
        >>> idata = sample_round1(model, ModelConfig(mcmc_draws=200, mcmc_tune=100))

    """
    with model:
        return pm.sample(
            draws=config.mcmc_draws,
            tune=config.mcmc_tune,
            chains=config.mcmc_chains,
            cores=config.mcmc_cores,
            target_accept=config.target_accept,
            random_seed=config.seed,
        )


@dataclass(frozen=True)
class CandidateForecast:
    """Forecast for a single candidate in the first round.

    Attributes:
        candidate_key: Candidate identifier key.
        mean_share: Posterior mean vote share.
        median_share: Posterior median vote share.
        ci_50: 50% credible interval for the vote share.
        ci_95: 95% credible interval for the vote share.
        prob_first: Probability the candidate finishes first.
        prob_second: Probability the candidate finishes second.
        prob_top_two: Probability the candidate finishes in the top two.
        prob_win_outright: Probability the candidate wins outright (>50%).

    """

    candidate_key: str
    mean_share: float
    median_share: float
    ci_50: tuple[float, float]
    ci_95: tuple[float, float]
    prob_first: float
    prob_second: float
    prob_top_two: float
    prob_win_outright: float


@dataclass(frozen=True)
class Round1Forecast:
    """Forecast for the full first-round election.

    Attributes:
        candidates: Per-candidate forecast objects.
        prob_runoff: Probability of a runoff (no candidate wins outright).
        round_number: Round identifier, always 1.

    """

    candidates: list[CandidateForecast]
    prob_runoff: float
    round_number: Literal[1] = 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize the Round1Forecast to a plain dictionary.

        Uses ``dataclasses.asdict`` to recursively convert all fields,
        including nested ``CandidateForecast`` objects. CI tuples
        (``ci_50``, ``ci_95``) become lists in the output dict.

        Returns:
            dict[str, Any]: Dictionary representation with str keys.

        Examples:
            >>> f = Round1Forecast(candidates=[], prob_runoff=0.5)
            >>> d = f.to_dict()
            >>> isinstance(d["candidates"], list)
            True

        """
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Round1Forecast:
        """Reconstruct a Round1Forecast from a dictionary.

        Handles type-casting of CI fields (lists back to tuples,
        ints back to floats). Recursively reconstructs nested
        ``CandidateForecast`` objects.

        Args:
            d: Dictionary produced by ``to_dict()``.

        Returns:
            Round1Forecast: Reconstructed dataclass instance.

        Raises:
            ValueError: If the dict is missing required keys or has
                malformed CI fields.

        """
        try:
            candidates = []
            for c in d["candidates"]:
                ci_50: tuple[float, float] = tuple(c["ci_50"])
                ci_95: tuple[float, float] = tuple(c["ci_95"])
                candidates.append(
                    CandidateForecast(
                        candidate_key=str(c["candidate_key"]),
                        mean_share=float(c["mean_share"]),
                        median_share=float(c["median_share"]),
                        ci_50=ci_50,
                        ci_95=ci_95,
                        prob_first=float(c["prob_first"]),
                        prob_second=float(c["prob_second"]),
                        prob_top_two=float(c["prob_top_two"]),
                        prob_win_outright=float(c["prob_win_outright"]),
                    )
                )
            return Round1Forecast(
                candidates=candidates,
                prob_runoff=d["prob_runoff"],
                round_number=d["round_number"],
            )
        except (KeyError, TypeError) as exc:
            msg = f"Malformed Round1Forecast dict: missing keys or bad CI fields ({exc})"
            raise ValueError(msg) from exc

    def to_json(self) -> str:
        """Serialize the Round1Forecast to a JSON string.

        Returns:
            str: JSON-encoded string via ``json.dumps(self.to_dict())``.

        """
        return json.dumps(self.to_dict())

    @staticmethod
    def from_json(s: str) -> Round1Forecast:
        """Deserialize a Round1Forecast from a JSON string.

        Args:
            s: JSON string produced by ``to_json()``.

        Returns:
            Round1Forecast: Reconstructed dataclass instance.

        Raises:
            json.JSONDecodeError: If the string is not valid JSON.
            ValueError: If the JSON is valid but missing required keys or
                has malformed CI fields.

        """
        return Round1Forecast.from_dict(json.loads(s))


def _percentile_tuple(vals: np.ndarray, q: list[float]) -> tuple[float, float]:
    """Compute percentile values and return as an explicitly sized tuple.

    Args:
        vals: Input array.
        q: Percentile values to compute (must be a 2-element list).

    Returns:
        A 2-tuple of floats for the specified percentiles.

    Raises:
        ValueError: If ``q`` does not have exactly two elements.

    """
    if len(q) != 2:  # noqa: PLR2004
        msg = f"q must have exactly 2 elements, got {len(q)}"
        raise ValueError(msg)
    result = np.percentile(vals, q)
    return (float(result[0]), float(result[1]))


def forecast_round1(
    idata: az.InferenceData,
    candidates: list[str],
) -> Round1Forecast:
    """Compute first-round forecast from posterior samples.

    Extracts election-day (time=0) vote share posteriors, computes mean,
    median, credible intervals, and ranking-based probabilities for each
    candidate.

    Args:
        idata: Posterior samples from :func:`build_round1_model`.
        candidates: Ordered list of candidate keys (must match the model's
            candidate column order).

    Returns:
        Round1Forecast with per-candidate forecasts.

    Examples:
        >>> from co_president.config import FIRST_ROUND_CANDIDATES, ModelConfig
        >>> model = build_round1_model(polls, None, ModelConfig())
        >>> idata = sample_round1(model, ModelConfig(mcmc_draws=200, mcmc_tune=100))
        >>> candidate_keys = sorted(set(FIRST_ROUND_CANDIDATES) & set(polls.columns))
        >>> forecast = forecast_round1(idata, candidate_keys)
        >>> forecast.prob_runoff
        0.99

    """
    p_time = idata.posterior["p_time"]  # (chain, draw, time, candidate)
    ndim = p_time.ndim
    if ndim != 4:  # noqa: PLR2004
        msg = f"p_time must have 4 dimensions (chain, draw, time, candidate), got {ndim}"
        raise ValueError(msg)
    last_dim = p_time.shape[-1]
    if last_dim != len(candidates):
        msg = (
            f"p_time candidate dimension ({last_dim}) does not match "
            f"candidates list length ({len(candidates)})"
        )
        raise ValueError(msg)
    n_candidates = len(candidates)
    if n_candidates < 2:  # noqa: PLR2004
        msg = f"forecast_round1 requires at least 2 candidates, got {n_candidates}"
        raise ValueError(msg)

    election_day = p_time[:, :, 0, :]  # (chain, draw, candidate)

    n_total = election_day.shape[0] * election_day.shape[1]
    shares = election_day.to_numpy().reshape(n_total, n_candidates)

    # Compute ranks for probability calculations (n_candidates >= 2 guaranteed)
    ranks = np.argsort(-shares, axis=1)

    candidate_forecasts: list[CandidateForecast] = []
    for i, key in enumerate(candidates):
        vals = shares[:, i]

        candidate_forecasts.append(
            CandidateForecast(
                candidate_key=key,
                mean_share=float(vals.mean()),
                median_share=float(np.median(vals)),
                ci_50=_percentile_tuple(vals, [25, 75]),
                ci_95=_percentile_tuple(vals, [2.5, 97.5]),
                prob_first=float((ranks[:, 0] == i).mean()),
                prob_second=float((ranks[:, 1] == i).mean()),
                prob_top_two=float(np.any(ranks[:, :2] == i, axis=1).mean()),
                prob_win_outright=float((vals > 0.5).mean()),  # noqa: PLR2004
            )
        )

    no_outright = (~np.any(shares > 0.5, axis=1)).mean()  # noqa: PLR2004
    prob_runoff = float(no_outright)

    return Round1Forecast(
        candidates=candidate_forecasts,
        prob_runoff=prob_runoff,
        round_number=1,
    )
