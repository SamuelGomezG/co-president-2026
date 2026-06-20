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
import logging
from typing import TYPE_CHECKING

import arviz as az  # type: ignore[reportMissingTypeStubs]
import numpy as np
import pandas as pd
import pymc as pm  # type: ignore[reportMissingTypeStubs]

from co_president.config import (
    ELECTION_DATE_ROUND2,
    FIRST_ROUND_CANDIDATES,
)
from co_president.data_polls import merge_digital_signals
from co_president.fundamentals.compositional import (
    _apply_zero_floor,  # type: ignore[reportPrivateUsage]
)
from co_president.model_round1 import (
    _compute_national_internet_rate,  # type: ignore[reportPrivateUsage]
)

logger = logging.getLogger(__name__)

# Threshold above which concentration_election_prior_mean triggers a fixed
# DirichletMultinomial election likelihood (matching model_round1.py).
_PHI_ELEC_FIXED_THRESHOLD = 100000

# Digital signal Beta decay peaks at T-1 (day before election) per SciELO 2023.
# Formula: exp(-abs(days_from_elec - 1.0) / 3.0).  T-1=1.0 (peak), T-0=0.72,
# T-7=0.135.  SciELO: T-1 has 1.86pp error, election day 6.56pp (bots distort).
_DIGITAL_SIGNAL_DECAY_DAYS = 3.0

if TYPE_CHECKING:
    from xarray import DataTree

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
        median_share_a: Posterior median vote share for candidate A.
        median_share_b: Posterior median vote share for candidate B.
        mean_margin: Mean margin (mean_share_a - mean_share_b).
        ci_50_a: 50% credible interval for candidate A's vote share.
        ci_50_b: 50% credible interval for candidate B's vote share.
        ci_95_a: 95% credible interval for candidate A's vote share.
        ci_95_b: 95% credible interval for candidate B's vote share.

    """

    candidate_a_key: str
    candidate_b_key: str
    prob_a_wins: float
    prob_b_wins: float
    mean_share_a: float
    mean_share_b: float
    median_share_a: float
    median_share_b: float
    mean_margin: float
    ci_50_a: tuple[float, float]
    ci_50_b: tuple[float, float]
    ci_95_a: tuple[float, float]
    ci_95_b: tuple[float, float]
    mean_share_rest: float = float("nan")
    median_share_rest: float = float("nan")
    ci_50_rest: tuple[float, float] = (float("nan"), float("nan"))
    ci_95_rest: tuple[float, float] = (float("nan"), float("nan"))


def build_runoff_simple_model(  # noqa: C901, PLR0912, PLR0913, PLR0915
    polls: pd.DataFrame,
    results: RoundResult,
    round1_idata: DataTree | None,
    config: ModelConfig,
    *,
    features: pd.DataFrame | None = None,
    digital_signals: pd.DataFrame,
    round2_result: RoundResult | None = None,
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
        features: Optional municipal feature matrix. When provided, the
            national internet access rate is used to weight the digital
            signal concentration prior.
        digital_signals: DataFrame with ``fecha`` and per-candidate
            columns containing digital signal values (e.g., Google Trends).
            A separate Beta observation layer is added.
        round2_result: Actual runoff election result. When provided and
            ``config.concentration_election_prior_mean > 100000``, adds a
            fixed-concentration DirichletMultinomial likelihood for the
            election-day posterior (goodness-of-fit backtest mode).

    Returns:
        pm.Model: Constructed PyMC model.

    Raises:
        ValueError: If runoff candidates are not found in ``polls`` columns,
            or if all polls have NaN candidate shares and no data remains.

    Examples:
        >>> from co_president.config import ModelConfig
        >>> config = ModelConfig()
        >>> model = build_runoff_simple_model(polls, results, None, config,
        ...     digital_signals=pd.DataFrame())
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
    sample_sizes: np.ndarray = work["muestra"].to_numpy().astype(int)
    raw_probs: np.ndarray = work[k_runoff_cols].to_numpy() / 100.0
    raw_probs = raw_probs / raw_probs.sum(axis=1, keepdims=True)
    observed_counts: np.ndarray = np.round(
        raw_probs * sample_sizes[:, np.newaxis],
    ).astype(int)
    effective_n = observed_counts.sum(axis=1)

    # Zero floor — replace 0 with 1, redistributing from largest columns
    # to preserve row sum (avoids log(0) in Dirichlet-Multinomial).
    _apply_zero_floor(observed_counts)

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

        # Digital signal Beta observation layer
        if not digital_signals.empty:
            _ds_cols = [c for c in k_runoff_cols if c in digital_signals.columns]
            if _ds_cols:
                _runoff_candidate_keys = [cand_a_key, cand_b_key]
                ds_merged = merge_digital_signals(
                    work,
                    digital_signals,
                    _runoff_candidate_keys,
                )
                _ds_cols = [c for c in _ds_cols if c in ds_merged.columns]
                ds_values = np.clip(  # type: ignore[assignment]
                    ds_merged[_ds_cols].to_numpy(dtype=float),
                    1e-10,
                    1 - 1e-10,
                )  # (T, M)
                _p_time = model["p_time"]  # type: ignore[reportUnknownVariableType]  # (T, 3)

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

                for _j, _ck in enumerate(_ds_cols):
                    _c_idx = k_runoff_cols.index(_ck)
                    _p_c = _p_time[:, _c_idx]  # type: ignore[reportIndexIssue,reportUnknownVariableType]
                    _p_c_safe = pm.math.clip(_p_c, 1e-10, 1 - 1e-10)  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]

                    _alpha_beta = _p_c_safe * phi_digital_t  # type: ignore[reportUnknownVariableType]
                    _beta_beta = (1.0 - _p_c_safe) * phi_digital_t  # type: ignore[reportUnknownVariableType]

                    pm.Beta(  # type: ignore[reportUnknownMemberType]
                        f"ds_{_ck}",
                        alpha=pm.math.maximum(_alpha_beta, 1e-10),  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                        beta=pm.math.maximum(_beta_beta, 1e-10),  # type: ignore[reportUnknownMemberType,reportUnknownArgumentType]
                        observed=ds_values[:, _j],
                    )

        # ── Runoff election likelihood (optional, goodness-of-fit) ────
        if round2_result is not None:
            p_elec = pm.Deterministic(  # type: ignore
                "p_elec",
                pm.math.softmax(theta[0], axis=-1),  # type: ignore
            )
            _vote_dict = {c.candidate_key: c.votes for c in round2_result.candidates}
            _elec_counts = np.array(
                [_vote_dict.get(k, 0) for k in k_runoff_cols[:2]],
                dtype=int,
            )
            _elec_rest = round2_result.total_valid_votes - _elec_counts.sum()
            _elec_counts = np.append(_elec_counts, max(_elec_rest, 0))

            if config.concentration_election_prior_mean > _PHI_ELEC_FIXED_THRESHOLD:
                phi_elec = config.concentration_election_prior_mean  # type: ignore[assignment]
                alpha_elec = p_elec * phi_elec  # type: ignore

                pm.DirichletMultinomial(  # type: ignore
                    "election_likelihood",
                    n=_elec_counts.sum(),
                    a=pm.math.maximum(alpha_elec, eps),  # type: ignore[reportUnknownMemberType]
                    observed=_elec_counts,
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
                _elec_sum = int(_elec_counts.sum())
                _observed_scaled = np.round(
                    _elec_counts.astype(float) / _elec_sum * total_votes_scaled,
                ).astype(int)

                n_scaled = int(_observed_scaled.sum())

                pm.DirichletMultinomial(  # type: ignore
                    "election_likelihood",
                    n=n_scaled,
                    a=pm.math.maximum(alpha_elec, eps),  # type: ignore[reportUnknownMemberType]
                    observed=_observed_scaled,
                )

    return model


def sample_runoff(model: pm.Model, config: ModelConfig) -> DataTree:
    """Sample posterior draws from a built runoff model via NUTS.

    Args:
        model: A built ``pm.Model`` from :func:`build_runoff_simple_model`.
        config: Model hyperparameters (draws, tune, chains, etc.).

    Returns:
        DataTree: Posterior and sampling stats.

    Raises:
        RuntimeError: If NUTS sampling fails to initialise or diverges.

    Examples:
        >>> model = build_runoff_simple_model(polls, results, None, ModelConfig(),
        ...     digital_signals=pd.DataFrame())
        >>> idata = sample_runoff(model, ModelConfig(mcmc_draws=200, mcmc_tune=100))

    """
    with model:
        return pm.sample(  # type: ignore
            draws=config.mcmc_draws,
            tune=config.mcmc_tune,
            chains=config.mcmc_chains,
            cores=config.mcmc_cores,
            target_accept=config.target_accept,
            init="adapt_diag",
            max_treedepth=12,
            random_seed=config.seed,
            nuts_sampler=config.nuts_sampler,
            mp_ctx="spawn",
        )


def forecast_runoff_simple(
    idata: DataTree,
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
    rest_values = election_day[:, :, 2].to_numpy().flatten()

    prob_a_wins = float((a_values > b_values).mean())
    prob_b_wins = float((b_values > a_values).mean())

    mean_share_a = float(a_values.mean())
    mean_share_b = float(b_values.mean())
    median_share_a = float(np.median(a_values))
    median_share_b = float(np.median(b_values))
    mean_margin = mean_share_a - mean_share_b

    mean_share_rest = float(rest_values.mean())
    median_share_rest = float(np.median(rest_values))

    ci_50_a_result = az.hdi(a_values, prob=0.5)  # type: ignore
    ci_50_b_result = az.hdi(b_values, prob=0.5)  # type: ignore
    ci_50_a = (float(ci_50_a_result[0]), float(ci_50_a_result[1]))  # type: ignore
    ci_50_b = (float(ci_50_b_result[0]), float(ci_50_b_result[1]))  # type: ignore

    ci_95_a_result = az.hdi(a_values, prob=0.95)  # type: ignore
    ci_95_b_result = az.hdi(b_values, prob=0.95)  # type: ignore
    ci_95_a = (float(ci_95_a_result[0]), float(ci_95_a_result[1]))  # type: ignore
    ci_95_b = (float(ci_95_b_result[0]), float(ci_95_b_result[1]))  # type: ignore

    ci_50_rest_result = az.hdi(rest_values, prob=0.5)  # type: ignore
    ci_95_rest_result = az.hdi(rest_values, prob=0.95)  # type: ignore
    ci_50_rest = (float(ci_50_rest_result[0]), float(ci_50_rest_result[1]))  # type: ignore
    ci_95_rest = (float(ci_95_rest_result[0]), float(ci_95_rest_result[1]))  # type: ignore

    return RunoffForecast(
        candidate_a_key=candidate_a,
        candidate_b_key=candidate_b,
        prob_a_wins=prob_a_wins,
        prob_b_wins=prob_b_wins,
        mean_share_a=mean_share_a,
        mean_share_b=mean_share_b,
        median_share_a=median_share_a,
        median_share_b=median_share_b,
        mean_margin=mean_margin,
        ci_50_a=ci_50_a,
        ci_50_b=ci_50_b,
        ci_95_a=ci_95_a,
        ci_95_b=ci_95_b,
        mean_share_rest=mean_share_rest,
        median_share_rest=median_share_rest,
        ci_50_rest=ci_50_rest,
        ci_95_rest=ci_95_rest,
    )
