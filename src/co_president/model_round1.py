"""SPEC-06: Dirichlet-Multinomial + reverse-time RW (1st round)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from typing import TYPE_CHECKING, Any, Literal

import arviz as az  # type: ignore[reportMissingTypeStubs]
import numpy as np
import pandas as pd
import pymc as pm  # type: ignore[reportMissingTypeStubs]
import pytensor.tensor as pt

from co_president.config import (
    ModelConfig,
    consultation_log_share_prior,
    get_active_candidates,
    get_consultation_votes,
    get_election_date,
)
from co_president.data_polls import merge_digital_signals
from co_president.fundamentals.compositional import (
    _apply_zero_floor,  # type: ignore[reportPrivateUsage]
)
from co_president.model_municipal import (
    build_municipal_model,
    sample_municipal_model,
)

if TYPE_CHECKING:
    from xarray import DataTree

    from co_president.data import RoundResult

logger = logging.getLogger(__name__)

# Threshold above which concentration_election_prior_mean is treated as a
# fixed constant instead of a learned Gamma parameter.  Values > 100,000
# indicate the user wants to turn off the Gamma and fix phi_elec.
_PHI_ELEC_FIXED_THRESHOLD = 0

# Digital signal Beta decay peaks at T-1 (day before election) per SciELO 2023.
# Formula: exp(-abs(days_from_elec - 1.0) / 3.0).  T-1=1.0 (peak), T-0=0.72,
# T-7=0.135.  SciELO: T-1 has 1.86pp error, election day 6.56pp (bots distort).
_DIGITAL_SIGNAL_DECAY_DAYS = 3.0

__all__ = [
    "CandidateForecast",
    "Round1Forecast",
    "build_round1_model",
    "forecast_round1",
    "predict_year",
    "sample_round1",
]


def _extract_municipal_prior(
    features: pd.DataFrame,
    config: ModelConfig,
    candidate_keys: list[str],
    target_year: int = 2022,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract municipal model posterior mean and std as structural prior.

    Trains the municipal hierarchical model (SPEC-22) on the full feature
    matrix in prior-only mode (no poll likelihood), then computes the
    turnout-weighted national vote share posterior for each candidate.
    Returns the CLR-space posterior mean and standard deviation for use
    as a Normal prior on the first time-point theta.

    Args:
        features: Municipal feature matrix (M rows x F columns).
        config: Model hyperparameters.
        candidate_keys: Active candidate column names.
        target_year: Historical election year for CLR left share.

    Returns:
        (mu_logit, sigma_logit): Arrays shape (K,) where K = len(candidate_keys).
        mu_logit[k] is the CLR-scale posterior mean for candidate k.
        sigma_logit[k] is the CLR-scale posterior std for candidate k.

    Raises:
        ValueError: If features is empty or missing required columns.

    """
    _eps = 1e-10

    polls = pd.DataFrame(
        columns=[*candidate_keys, "fecha", "encuestadora", "muestra"],
    )

    model = build_municipal_model(
        features,
        polls,
        results=None,
        config=config,
        target_year=target_year,
        prior_only=True,
    )

    muni_draws = max(config.mcmc_draws // 4, 200)
    muni_config = ModelConfig(
        mcmc_draws=muni_draws,
        mcmc_tune=500,
        mcmc_chains=config.mcmc_chains,
        mcmc_cores=min(config.mcmc_cores, 2),
        target_accept=0.9,
        seed=config.seed,
        nuts_sampler=config.nuts_sampler,
    )
    idata = sample_municipal_model(model, muni_config)

    p_natl = idata.posterior["p_natl"].to_numpy()  # (chain, draw, K)
    log_p = np.log(p_natl + _eps)
    log_mean = np.mean(log_p, axis=-1, keepdims=True)
    clr_p = log_p - log_mean  # (chain, draw, K)

    mu_logit = np.asarray(clr_p.mean(axis=(0, 1)))
    sigma_logit = np.asarray(clr_p.std(axis=(0, 1)))
    return mu_logit, sigma_logit


def _compute_national_internet_rate(
    features: pd.DataFrame,
    *,
    alpha: float = 0.3,
) -> float:
    """Compute internet-penetration-adjusted digital signal weight.

    Uses soft deflation: weight = 1 - alpha*(1 - rate).
    alpha=0 gives no adjustment (SciELO approach).  alpha=1 gives full
    deflation by penetration.  alpha=0.3 gives gentle correction: at
    rate=0.36 the weight is 0.808 (2.2x stronger than full deflation).

    Args:
        features: Municipal feature matrix with ``internet_access_rate`` and
            ``pop_2022`` columns.
        alpha: Soft-deflation parameter in [0, 1].

    Returns:
        Float in [0, 1] representing the adjusted digital signal weight.

    Raises:
        ValueError: If features is empty or missing required columns.

    """
    required = {"internet_access_rate", "pop_2022"}
    missing = required - set(features.columns)
    if missing:
        msg = f"features missing columns for internet rate: {sorted(missing)}"
        raise ValueError(msg)
    pop = features["pop_2022"].fillna(0).to_numpy()
    rate = features["internet_access_rate"].fillna(0.0).to_numpy()
    if pop.sum() == 0:
        return 0.0
    national_rate = float(np.average(rate, weights=pop))
    return float(1.0 - alpha * (1.0 - national_rate))


def build_round1_model(  # noqa: C901, PLR0912, PLR0913, PLR0915
    polls: pd.DataFrame,
    results: RoundResult | None,
    config: ModelConfig,
    *,
    no_house_effects: bool = False,
    features: pd.DataFrame | None = None,
    digital_signals: pd.DataFrame,
    year: int = 2022,
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
        no_house_effects: If True, build a simplified model without house
            effects and with a fixed concentration parameter. Defaults to
            False.
        features: Optional municipal feature matrix. When provided, the
            first time-point theta gets a structural prior from the
            municipal hierarchical model instead of the flat consultation
            prior.
        digital_signals: DataFrame with ``fecha`` and per-candidate
            columns containing digital signal values (e.g., Google Trends).
            When provided, a separate Beta observation layer is added.
        year: Election year (default 2022).

    Returns:
        pm.Model: Constructed PyMC model.

    Raises:
        ValueError: If ``polls`` is empty or missing required columns.

    Examples:
        >>> config = ModelConfig()
        >>> model = build_round1_model(polls, None, config, digital_signals=pd.DataFrame())

    """
    # Make a working copy to avoid mutating the input
    polls = polls.copy()

    # Convert fecha to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(polls["fecha"]):
        polls["fecha"] = pd.to_datetime(polls["fecha"])

    # Determine candidate keys from DataFrame columns
    candidate_keys = sorted(
        {c.key for c in get_active_candidates(1, year=year)} & set(polls.columns)
    )
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
    election_day_r1 = get_election_date(year, 1)
    polls["days_before"] = (pd.Timestamp(election_day_r1) - polls["fecha"]).dt.days
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
    sample_sizes: np.ndarray = polls["muestra"].to_numpy().astype(int)
    raw_probs: np.ndarray = polls[candidate_keys].to_numpy() / 100.0
    raw_probs = raw_probs / raw_probs.sum(axis=1, keepdims=True)
    observed_counts: np.ndarray = np.round(
        raw_probs * sample_sizes[:, np.newaxis],
    ).astype(int)

    row_sums: np.ndarray = observed_counts.sum(axis=1)
    diff: np.ndarray = sample_sizes - row_sums
    if not np.all(diff == 0):
        n_candidates = len(candidate_keys)
        if np.any(np.abs(diff) > n_candidates):
            logger.warning(
                "Rounding mismatch >%d votes in %d row(s); "
                "largest |diff| = %d. Some candidate categories may "
                "be missing from the poll DataFrame.",
                n_candidates,
                int((np.abs(diff) > n_candidates).sum()),
                int(np.abs(diff).max()),
            )
        max_idx = np.argmax(observed_counts, axis=1)
        observed_counts[np.arange(len(observed_counts)), max_idx] += diff

    # Zero floor — replace 0 with 1, redistributing from largest columns
    # to preserve row sum (avoids log(0) in Dirichlet-Multinomial).
    _apply_zero_floor(observed_counts)

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
    log_prior = consultation_log_share_prior(year)
    consultation_votes = get_consultation_votes(year)

    # Default log-share for candidates without consultation data
    nonzero_total = sum(v for v in consultation_votes.values() if v > 0)
    if nonzero_total > 0:
        nonzero_shares = [v / nonzero_total for v in consultation_votes.values() if v > 0]
        default_log_share = float(np.log(min(nonzero_shares) / 2.0))
    else:
        default_log_share = -1.0

    prior_mean = np.array(
        [log_prior.get(k, default_log_share) for k in candidate_keys],
        dtype=float,
    )

    # Municipal structural prior: extract OUTSIDE the pm.Model() context
    # to prevent municipal RVs from leaking into the round 1 trace.
    if features is not None and len(features) > 0:
        logger.info(
            "Extracting municipal prior from %d municipalities...",
            len(features),
        )
        theta_mu, theta_sigma = _extract_municipal_prior(
            features,
            config,
            candidate_keys,
            target_year=year,
        )
    else:
        theta_mu = prior_mean
        theta_sigma = config.consultation_prior_strength

    # Ensure theta_sigma shape matches theta_mu (guard against municipal
    # model returning wrong candidate count for non-2022 years)
    if not isinstance(theta_sigma, float | int):
        theta_sigma = np.asarray(theta_sigma, dtype=float)
        if theta_sigma.shape != theta_mu.shape:
            theta_sigma = np.full_like(theta_mu, float(config.consultation_prior_strength))
    if theta_mu.shape != (n_candidates,):
        logger.warning(
            "Municipal prior mu shape %s != n_candidates %d; using flat prior",
            theta_mu.shape,
            n_candidates,
        )
        theta_mu = prior_mean
        theta_sigma = float(config.consultation_prior_strength)

    # Override blanco prior: shrink toward historical blank vote rate (~2.5%)
    # instead of the poll-implied ~10.6%, which is systematically inflated in
    # Colombian polls relative to actual blank voting.
    _r1_historical_blanco = 0.020
    if "blanco" in candidate_keys:
        _b_idx = candidate_keys.index("blanco")
        theta_mu[_b_idx] = np.log(_r1_historical_blanco)
        if isinstance(theta_sigma, float | int):
            theta_sigma = np.full(len(candidate_keys), float(theta_sigma))
        else:
            theta_sigma = theta_sigma.copy()
        theta_sigma[_b_idx] = 0.3

    with pm.Model() as model:  # type: ignore
        sigma_rw = pm.HalfNormal(  # type: ignore
            "sigma_rw",
            sigma=config.random_walk_sigma_prior,
        )

        # Non-centered random walk: breaks posterior correlation between
        # adjacent theta_t, enabling fast NUTS mixing (R-hat < 1.01).
        theta_0 = pm.Normal(  # type: ignore
            "theta_0",
            mu=theta_mu,
            sigma=theta_sigma,
            shape=n_candidates,
        )
        rw_raw = pm.Normal(  # type: ignore
            "rw_raw",
            mu=0,
            sigma=1,
            shape=(n_time_points - 1, n_candidates),
        )
        theta_increments = sigma_rw * rw_raw  # type: ignore[reportUnknownVariableType]
        theta_cumulative = pt.cumsum(theta_increments, axis=0)  # type: ignore[reportUnknownVariableType]
        theta_rest = theta_0[None, :] + theta_cumulative  # type: ignore[reportUnknownVariableType]
        theta_stacked = pt.concatenate(  # type: ignore[reportUnknownVariableType]
            [theta_0[None, :], theta_rest], axis=0
        )

        if not no_house_effects:
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
                phi_poll * sample_size_multiplier,
            )

            # Latent vote share probabilities per time point
            # (used for prior predictive validation and plotting)
            pm.Deterministic("p_time", pm.math.softmax(theta_stacked, axis=-1))  # type: ignore

            # House effects (zero-sum constrained)
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

            # Poll observation model with house effects
            theta_selected = theta_stacked[time_indices]  # type: ignore
            house_selected = house_effects[pollster_indices]  # type: ignore
            theta_adj = theta_selected + house_selected  # type: ignore

            p_adj = pm.Deterministic("p_adj", pm.math.softmax(theta_adj, axis=-1))  # type: ignore
            alpha_poll = pm.math.maximum(p_adj * phi_poll_n, eps)  # type: ignore
        else:
            # Simplified model without house effects or phi_poll
            # Latent vote share probabilities per time point
            pm.Deterministic("p_time", pm.math.softmax(theta_stacked, axis=-1))  # type: ignore
            theta_selected = theta_stacked[time_indices]  # type: ignore
            p_adj = pm.Deterministic("p_adj", pm.math.softmax(theta_selected, axis=-1))  # type: ignore
            phi_poll_fixed = float(config.concentration_poll_prior_mean)
            alpha_poll = pm.math.maximum(  # type: ignore
                p_adj * phi_poll_fixed * sample_size_multiplier,  # type: ignore
                eps,
            )

        pm.DirichletMultinomial(  # type: ignore
            "poll_likelihood",
            n=sample_sizes,
            a=alpha_poll,
            observed=observed_counts,
        )

        # Election result likelihood (optional)
        if results is not None:
            p_elec = pm.Deterministic(  # type: ignore
                "p_elec",
                pm.math.softmax(theta_stacked[0], axis=-1),  # type: ignore
            )

            vote_dict = {c.candidate_key: c.votes for c in results.candidates}
            election_counts = np.array(
                [vote_dict.get(k, 0) for k in candidate_keys],
                dtype=int,
            )

            if config.concentration_election_prior_mean > _PHI_ELEC_FIXED_THRESHOLD:
                phi_elec = config.concentration_election_prior_mean  # type: ignore[assignment]
                alpha_elec = p_elec * phi_elec  # type: ignore[operator]

                pm.DirichletMultinomial(  # type: ignore[reportUnknownMemberType]
                    "election_likelihood",
                    n=election_counts.sum(),
                    a=alpha_elec,  # type: ignore[reportUnknownArgumentType]
                    observed=election_counts,
                )
            else:
                phi_elec = pm.Gamma(  # type: ignore
                    "phi_elec",
                    alpha=config.concentration_election_prior_shape,
                    beta=config.concentration_election_prior_shape
                    / config.concentration_election_prior_mean,
                )
                alpha_elec = p_elec * phi_elec  # type: ignore

                total_votes_scaled = int(config.concentration_election_votes_scale)
                election_counts_sum = int(election_counts.sum())
                observed_scaled = np.round(
                    election_counts.astype(float) / election_counts_sum * total_votes_scaled,
                ).astype(int)
                n_scaled = int(observed_scaled.sum())

                pm.DirichletMultinomial(  # type: ignore
                    "election_likelihood",
                    n=n_scaled,
                    a=alpha_elec,
                    observed=observed_scaled,
                )

        # Digital signal Beta observation layer
        if not digital_signals.empty:
            _ds_cols = [c for c in candidate_keys if c in digital_signals.columns]
            if _ds_cols:
                ds_merged = merge_digital_signals(polls, digital_signals, candidate_keys)
                if ds_merged.empty:
                    msg = (
                        "merge_digital_signals returned empty DataFrame. "
                        "Time indices misaligned between polls and digital signals."
                    )
                    raise ValueError(msg)
                ds_time_indices = ds_merged["days_before"].map(day_to_idx).to_numpy(dtype=int)
                ds_values = np.clip(  # type: ignore[assignment]
                    ds_merged[_ds_cols].to_numpy(dtype=float),
                    1e-10,
                    1 - 1e-10,
                )
                ds_candidate_indices = [candidate_keys.index(c) for c in _ds_cols]

                _internet_rate = 1.0
                if features is not None and not features.empty:
                    _internet_rate = _compute_national_internet_rate(features)

                phi_digital_base = pm.Gamma(  # type: ignore[reportUnknownMemberType]
                    "phi_digital_base",
                    alpha=5.0,
                    beta=5.0 / 100.0,
                )

                days_from_elec = ds_merged["days_before"].to_numpy(dtype=float)
                _decay = pm.math.exp(  # type: ignore[reportUnknownMemberType]
                    -pm.math.abs(days_from_elec - 1.0) / _DIGITAL_SIGNAL_DECAY_DAYS,  # type: ignore[reportUnknownArgumentType]
                )
                phi_digital_t = phi_digital_base * _decay * _internet_rate  # type: ignore[reportUnknownVariableType]

                _p_time = model["p_time"]  # type: ignore[reportUnknownVariableType]
                for _j, _c_idx in enumerate(ds_candidate_indices):
                    _p_c = _p_time[ds_time_indices, _c_idx]  # type: ignore[reportIndexIssue,reportUnknownVariableType]
                    _p_c_safe = pm.math.clip(_p_c, 1e-10, 1 - 1e-10)  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]

                    _alpha_beta = _p_c_safe * phi_digital_t  # type: ignore[reportUnknownVariableType]
                    _beta_beta = (1.0 - _p_c_safe) * phi_digital_t  # type: ignore[reportUnknownVariableType]

                    pm.Beta(  # type: ignore[reportUnknownMemberType]
                        f"ds_{candidate_keys[_c_idx]}",
                        alpha=pm.math.maximum(_alpha_beta, 1e-10),  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                        beta=pm.math.maximum(_beta_beta, 1e-10),  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                        observed=ds_values[:, _j],
                    )

    return model


def sample_round1(model: pm.Model, config: ModelConfig) -> DataTree:
    """Sample posterior draws from a built round 1 model via NUTS.

    Args:
        model: A built ``pm.Model`` from :func:`build_round1_model`.
        config: Model hyperparameters (draws, tune, chains, etc.).

    Returns:
        DataTree: Posterior and sampling stats.

    Raises:
        RuntimeError: If NUTS sampling fails to initialise or diverges.

    Examples:
        >>> model = build_round1_model(polls, results, ModelConfig(),
        ...     digital_signals=pd.DataFrame())
        >>> idata = sample_round1(model, ModelConfig(mcmc_draws=200, mcmc_tune=100))

    """
    with model:
        return pm.sample(  # type: ignore
            draws=config.mcmc_draws,
            tune=config.mcmc_tune,
            chains=config.mcmc_chains,
            cores=config.mcmc_cores,
            target_accept=config.target_accept,
            init="adapt_diag",  # horseshoe → diagonal-dominant; adapt_full O(N³)
            max_treedepth=12,
            random_seed=config.seed,
            nuts_sampler=config.nuts_sampler,
            mp_ctx="spawn",
        )


def forecast_round1(
    idata: DataTree,
    candidate_keys: list[str],
) -> Round1Forecast:
    """Compute first-round forecast from posterior samples.

    Extracts election-day ``p_time`` from a single-election posterior and
    computes mean, median, credible intervals, and ranking-based probabilities.

    Args:
        idata: Posterior from :func:`sample_round1`.
        candidate_keys: Ordered list of candidate keys (must match the model's
            candidate column order).

    Returns:
        Round1Forecast for the election.

    Raises:
        ValueError: If ``p_time`` is missing or has wrong dimensions.

    Examples:
        >>> model = build_round1_model(polls, results, ModelConfig(),
        ...     digital_signals=pd.DataFrame())
        >>> idata = sample_round1(model, ModelConfig())
        >>> forecast = forecast_round1(idata, sorted(FIRST_ROUND_CANDIDATES.keys()))

    """
    if "p_time" not in idata.posterior:
        msg = (
            f"Posterior does not contain 'p_time'.  Available variables: "
            f"{list(idata.posterior.data_vars)}"
        )
        raise ValueError(msg)

    p_time = idata.posterior["p_time"]
    ndim = p_time.ndim
    if ndim != 4:  # noqa: PLR2004
        msg = f"p_time must have 4 dimensions (chain, draw, time, candidate), got {ndim}"
        raise ValueError(msg)
    last_dim = p_time.shape[-1]
    if last_dim != len(candidate_keys):
        msg = (
            f"p_time candidate dimension ({last_dim}) does not match "
            f"candidates list length ({len(candidate_keys)})"
        )
        raise ValueError(msg)
    n_candidates = len(candidate_keys)
    if n_candidates < 2:  # noqa: PLR2004
        msg = f"forecast_round1 requires at least 2 candidates, got {n_candidates}"
        raise ValueError(msg)

    election_day = p_time[:, :, 0, :]  # (chain, draw, candidate)
    n_total = election_day.shape[0] * election_day.shape[1]
    shares = election_day.to_numpy().reshape(n_total, n_candidates)

    ranks = np.argsort(-shares, axis=1)
    ci_50_all = az.hdi(shares, prob=0.5, axis=0)  # type: ignore
    ci_95_all = az.hdi(shares, prob=0.95, axis=0)  # type: ignore

    candidate_forecasts: list[CandidateForecast] = []
    for i, key in enumerate(candidate_keys):
        vals = shares[:, i]
        ci_50 = (float(ci_50_all[i, 0]), float(ci_50_all[i, 1]))  # type: ignore
        ci_95 = (float(ci_95_all[i, 0]), float(ci_95_all[i, 1]))  # type: ignore
        candidate_forecasts.append(
            CandidateForecast(
                candidate_key=key,
                mean_share=float(vals.mean()),
                median_share=float(np.median(vals)),
                ci_50=ci_50,
                ci_95=ci_95,
                prob_first=float((ranks[:, 0] == i).mean()),
                prob_second=float((ranks[:, 1] == i).mean()),
                prob_top_two=float(np.any(ranks[:, :2] == i, axis=1).mean()),
                prob_win_outright=float((vals > 0.5).mean()),  # noqa: PLR2004
            ),
        )

    no_outright = (~np.any(shares > 0.5, axis=1)).mean()  # noqa: PLR2004
    prob_runoff = float(no_outright)

    return Round1Forecast(
        candidates=candidate_forecasts,
        prob_runoff=prob_runoff,
        round_number=1,
    )


@dataclass(frozen=True)
class CandidateForecast:
    """Forecast for a single candidate in the first round."""

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
    """Forecast for the full first-round election."""

    candidates: list[CandidateForecast]
    prob_runoff: float
    round_number: Literal[1] = 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Round1Forecast:
        """Reconstruct from dictionary."""
        try:
            candidates = [
                CandidateForecast(
                    candidate_key=str(c["candidate_key"]),
                    mean_share=float(c["mean_share"]),
                    median_share=float(c["median_share"]),
                    ci_50=tuple(c["ci_50"]),
                    ci_95=tuple(c["ci_95"]),
                    prob_first=float(c["prob_first"]),
                    prob_second=float(c["prob_second"]),
                    prob_top_two=float(c["prob_top_two"]),
                    prob_win_outright=float(c["prob_win_outright"]),
                )
                for c in d["candidates"]
            ]
            return Round1Forecast(
                candidates=candidates,
                prob_runoff=d["prob_runoff"],
                round_number=d["round_number"],
            )
        except (KeyError, TypeError) as exc:
            msg = f"Malformed Round1Forecast dict: missing keys or bad CI fields ({exc})"
            raise ValueError(msg) from exc

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @staticmethod
    def from_json(s: str) -> Round1Forecast:
        """Deserialize from JSON string."""
        return Round1Forecast.from_dict(json.loads(s))


def predict_year(
    year: int,
    idata: DataTree,
    candidates: list[str],
) -> Round1Forecast:
    """Compute first-round forecast for a specific year from multi-election posterior.

    Extracts the ``{year}_p_time`` variable from the posterior and computes
    mean, median, credible intervals, and ranking-based probabilities.

    Args:
        year: Election year to extract forecast for.
        idata: Posterior samples from :func:`build_multi_election_model`.
        candidates: Ordered list of candidate keys (must match the model's
            candidate column order).

    Returns:
        Round1Forecast for the specified year.

    Raises:
        ValueError: If ``{year}_p_time`` is missing or has wrong dimensions.

    Examples:
        >>> model = build_multi_election_model(polls_by_year, {}, ModelConfig())
        >>> idata = pm.sample(model, draws=200, tune=100)
        >>> candidate_keys = sorted(set(FIRST_ROUND_CANDIDATES) & set(polls_by_year[2022].columns))
        >>> forecast = predict_year(2022, idata, candidate_keys)

    """
    var_name = f"{year}_p_time"
    if var_name not in idata.posterior:
        msg = (
            f"Posterior does not contain '{var_name}'.  Available variables: "
            f"{list(idata.posterior.data_vars)}"
        )
        raise ValueError(msg)

    p_time = idata.posterior[var_name]
    ndim = p_time.ndim
    if ndim != 4:  # noqa: PLR2004
        msg = f"{var_name} must have 4 dimensions (chain, draw, time, candidate), got {ndim}"
        raise ValueError(msg)
    last_dim = p_time.shape[-1]
    if last_dim != len(candidates):
        msg = (
            f"{var_name} candidate dimension ({last_dim}) does not match "
            f"candidates list length ({len(candidates)})"
        )
        raise ValueError(msg)
    n_candidates = len(candidates)
    if n_candidates < 2:  # noqa: PLR2004
        msg = f"predict_year requires at least 2 candidates, got {n_candidates}"
        raise ValueError(msg)

    election_day = p_time[:, :, 0, :]  # (chain, draw, candidate)
    n_total = election_day.shape[0] * election_day.shape[1]
    shares = election_day.to_numpy().reshape(n_total, n_candidates)

    # Compute ranks
    ranks = np.argsort(-shares, axis=1)

    # Highest Density Intervals
    ci_50_all = az.hdi(shares, prob=0.5, axis=0)  # type: ignore
    ci_95_all = az.hdi(shares, prob=0.95, axis=0)  # type: ignore

    candidate_forecasts: list[CandidateForecast] = []
    for i, key in enumerate(candidates):
        vals = shares[:, i]

        ci_50 = (float(ci_50_all[i, 0]), float(ci_50_all[i, 1]))  # type: ignore
        ci_95 = (float(ci_95_all[i, 0]), float(ci_95_all[i, 1]))  # type: ignore

        candidate_forecasts.append(
            CandidateForecast(
                candidate_key=key,
                mean_share=float(vals.mean()),
                median_share=float(np.median(vals)),
                ci_50=ci_50,
                ci_95=ci_95,
                prob_first=float((ranks[:, 0] == i).mean()),
                prob_second=float((ranks[:, 1] == i).mean()),
                prob_top_two=float(np.any(ranks[:, :2] == i, axis=1).mean()),
                prob_win_outright=float((vals > 0.5).mean()),  # noqa: PLR2004
            ),
        )

    no_outright = (~np.any(shares > 0.5, axis=1)).mean()  # noqa: PLR2004
    prob_runoff = float(no_outright)

    return Round1Forecast(
        candidates=candidate_forecasts,
        prob_runoff=prob_runoff,
        round_number=1,
    )
