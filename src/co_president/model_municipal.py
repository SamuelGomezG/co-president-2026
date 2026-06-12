"""SPEC-22: 3-layer logistic-normal / Dirichlet-Multinomial municipal hierarchical model.

Architecture
------------
Layer A — Municipal prior (logistic-normal):
    logit(p_mk) = alpha_k + Σ beta_g[k] · z_feature_g_m + sigma_m[k] · mu_m_raw[m, k]

Layer B — National poll likelihood (Dirichlet-Multinomial):
    p_natl_n = Σ_m (w_m · softmax(logit_p_mk))  where w = turnout-weighted pop

Layer C — Election rollup likelihood (Dirichlet-Multinomial, optional):
    Same turnout-weighted rollup applied to election-day result.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pymc as pm  # type: ignore[reportMissingTypeStubs]

from co_president.config import (
    ELECTION_DATE_ROUND1,
    FIRST_ROUND_CANDIDATES,
)
from co_president.fundamentals.features import clr

if TYPE_CHECKING:
    from xarray import DataTree

    from co_president.config import ModelConfig
    from co_president.data import RoundResult

logger = logging.getLogger(__name__)

__all__ = [
    "HierarchicalForecast",
    "MunicipalForecast",
    "build_municipal_model",
    "compute_effective_pop",
    "sample_municipal_model",
]

_EPSILON = 1e-10
_EXTRACT_ROUND = 1


# ═══════════════════════════════════════════════════════════════════════
# Feature helpers
# ═══════════════════════════════════════════════════════════════════════


def compute_effective_pop(pop: np.ndarray, turnout: np.ndarray) -> np.ndarray:
    """Turnout-weighted effective population.

    Corrects the inhabitants-vs-voters bias where high-abstention zones
    (e.g. Chocó, Catatumbo) would be over-weighted by raw population
    counts::

        effective_pop_m = pop_m · turnout_m / mean(turnout)

    Args:
        pop: Raw population counts per municipality (M,).
        turnout: Historical turnout rate per municipality (M,) in [0, 1].

    Returns:
        Effective population array (M,) with dtype float64.

    Examples:
        >>> compute_effective_pop(np.array([100, 200]), np.array([0.5, 0.8]))
        array([ 76.92307692, 246.15384615])
        >>> compute_effective_pop(np.array([100, 200]), np.array([0.0, 0.0]))
        array([100., 200.])

    """
    pop_f = pop.astype(float)
    mean_turnout = turnout.mean()
    if mean_turnout <= _EPSILON:
        return pop_f
    return pop_f * turnout / mean_turnout


def _extract_clr_left_share(features: pd.DataFrame, target_year: int = 2022) -> np.ndarray:
    """Extract CLR-transformed left vote share from the most recent election.

    Uses *target_year* round-1 ``historical`` record.  Falls back to 0.0
    (neutral CLR) when no record is available.
    """
    n = len(features)
    result = np.full(n, 0.0, dtype=float)
    for i in range(n):
        hist = features.iloc[i].get("historical")
        if isinstance(hist, tuple) and len(hist) > 0:  # pyright: ignore[reportUnknownArgumentType]
            for rec in hist:  # type: ignore[reportUnknownVariableType]
                if rec.year == target_year and rec.round == _EXTRACT_ROUND:  # type: ignore[reportUnknownMemberType]
                    clr_left = float(rec.clr_shares()[0])  # type: ignore[reportUnknownMemberType]
                    result[i] = clr_left
                    break
    return result


def _clr_nbi_array(nbi: np.ndarray) -> np.ndarray:
    """CLR of the ``(nbi_rate, 1 - nbi_rate)`` binary composition."""
    return np.array([clr((r, 1.0 - r))[0] for r in nbi])


# ═══════════════════════════════════════════════════════════════════════
# Prior helpers
# ═══════════════════════════════════════════════════════════════════════


def _horseshoe_beta(
    name: str,
    tau: object,
    sigma: float,
    shape: int,
) -> object:
    """Build a horseshoe-prior beta coefficient with non-centered parameterization.

    Args:
        name: Variable name prefix.
        tau: Global shrinkage parameter (shared across all beta groups).
        sigma: Prior scale (``beta_coefficient_prior_sigma``).
        shape: Number of candidates.

    Returns:
        Deterministic ``name`` tensor of shape ``(shape,)`` with the
        horseshoe-shrunk coefficient.

    """
    lam = pm.HalfCauchy(f"{name}_lam", beta=1.0, shape=shape)  # type: ignore[reportUnknownMemberType]
    z = pm.Normal(f"{name}_z", mu=0, sigma=1.0, shape=shape)  # type: ignore[reportUnknownMemberType]
    return pm.Deterministic(name, z * tau * lam * sigma)  # type: ignore[reportUnknownMemberType]


# ═══════════════════════════════════════════════════════════════════════
# Model building
# ═══════════════════════════════════════════════════════════════════════


def build_municipal_model(  # noqa: C901, PLR0912, PLR0915
    features: pd.DataFrame,
    polls: pd.DataFrame,
    results: RoundResult | None,
    config: ModelConfig,
    target_year: int = 2022,
) -> pm.Model:
    """Build the 3-layer municipal hierarchical PyMC model.

    Args:
        features: Municipal feature matrix from :func:`load_features`.
            Must contain ``codigo_municipio``, ``pop_2022``,
            ``historical_turnout_m``, ``pct_afro_colombian``, ``nbi_rate``,
            ``pct_rural_disperso``, ``years_schooling_promedio``,
            ``high_risk_flag``, and ``historical``.
        polls: Clean poll DataFrame with candidate columns (percentage
            shares), ``fecha``, ``encuestadora``, ``muestra``.
        results: Election result for validation (optional).
        config: Model hyperparameters.
        target_year: Election year to extract ``clr_left_share`` from the
            ``historical`` column.  Default 2022.

    Returns:
        Constructed ``pm.Model``.

    Raises:
        ValueError: If *polls* is empty, *features* is empty, or required
            columns are missing.
        ValueError: If *features* has fewer than 2 municipalities or no
            candidate columns overlap with *polls*.

    """
    polls = polls.copy()

    # ── Validate inputs ───────────────────────────────────────────────
    _required_columns = (
        "codigo_municipio",
        "pop_2022",
        "historical_turnout_m",
        "pct_afro_colombian",
        "nbi_rate",
        "pct_rural_disperso",
        "years_schooling_promedio",
        "high_risk_flag",
        "historical",
    )
    _missing = [col for col in _required_columns if col not in features.columns]
    if _missing:
        msg = f"features is missing required columns: {_missing}. Run `make fundamentals` first."
        raise ValueError(msg)

    if len(features) < 2:  # noqa: PLR2004
        msg = f"features must have at least 2 municipalities, got {len(features)}"
        raise ValueError(msg)

    if len(polls) == 0:
        msg = "polls DataFrame is empty"
        raise ValueError(msg)

    _required_poll_cols = ("fecha", "encuestadora", "muestra")
    _missing_poll = [c for c in _required_poll_cols if c not in polls.columns]
    if _missing_poll:
        msg = f"polls is missing required columns: {_missing_poll}"
        raise ValueError(msg)

    # Candidate keys from DataFrame columns
    candidate_keys = sorted(set(FIRST_ROUND_CANDIDATES) & set(polls.columns))
    n_candidates = len(candidate_keys)
    if n_candidates == 0:
        msg = "No candidate columns found in polls DataFrame"
        raise ValueError(msg)

    n_municipalities = len(features)

    # ── Feature extraction ────────────────────────────────────────────
    pop = features["pop_2022"].to_numpy().astype(float)
    turnout = features["historical_turnout_m"].to_numpy()
    if config.enable_population_weighting:
        effective_pop = compute_effective_pop(pop, turnout)
    else:
        effective_pop = pop.copy()
    pop_weights = effective_pop / effective_pop.sum()

    # Feature arrays (M,)
    clr_left = _extract_clr_left_share(features, target_year)
    pct_afro = features["pct_afro_colombian"].to_numpy()
    nbi_rate = features["nbi_rate"].to_numpy()
    clr_nbi = _clr_nbi_array(nbi_rate)
    pct_rural = features["pct_rural_disperso"].to_numpy()
    years_schooling = features["years_schooling_promedio"].to_numpy()
    high_risk = features["high_risk_flag"].to_numpy().astype(float)

    # Standardize continuous features
    def _z(x: np.ndarray) -> np.ndarray:
        return (x - x.mean()) / (x.std() + _EPSILON)

    clr_left_z = _z(clr_left)
    pct_afro_z = _z(pct_afro)
    clr_nbi_z = _z(clr_nbi)
    pct_rural_z = _z(pct_rural)
    years_schooling_z = _z(years_schooling)
    # high_risk is binary, no standardization

    # ── Poll processing (mirrors model_round1.py) ─────────────────────
    if not pd.api.types.is_datetime64_any_dtype(polls["fecha"]):
        polls["fecha"] = pd.to_datetime(polls["fecha"])

    # Drop rows with NaN candidate shares
    nan_mask = polls[[*candidate_keys, "muestra"]].isna().any(axis=1)
    if nan_mask.any():
        polls = polls.loc[~nan_mask].copy()
        if len(polls) == 0:
            msg = "All polls have NaN candidate shares; no data remains"
            raise ValueError(msg)

    n_polls = len(polls)

    # Time index: all polls at election day for the municipal model
    # (the fundamentals are time-invariant; all polls mapped to election day)
    polls["days_before"] = (pd.Timestamp(ELECTION_DATE_ROUND1) - polls["fecha"]).dt.days
    unique_days = sorted(polls["days_before"].unique())
    day_to_idx = {day: i for i, day in enumerate(unique_days)}
    polls["time_idx"] = polls["days_before"].map(day_to_idx)

    # Pollster index
    unique_pollsters = polls["encuestadora"].unique()
    n_pollsters = len(unique_pollsters)
    pollster_to_idx = {name: i for i, name in enumerate(unique_pollsters)}
    polls["pollster_idx"] = polls["encuestadora"].map(pollster_to_idx)

    # Observed counts from percentage shares
    sample_sizes = polls["muestra"].to_numpy().astype(int)
    observed_counts = np.round(
        polls[candidate_keys].to_numpy() / 100.0 * sample_sizes[:, np.newaxis],
    ).astype(int)

    # Rounding correction (same as model_round1.py)
    row_sums = observed_counts.sum(axis=1)
    diff = sample_sizes - row_sums
    if not np.all(diff == 0):
        if np.any(np.abs(diff) > n_candidates):
            logger.warning(
                "Rounding mismatch >%d votes in %d row(s); largest |diff| = %d",
                n_candidates,
                int((np.abs(diff) > n_candidates).sum()),
                int(np.abs(diff).max()),
            )
        max_idx = np.argmax(observed_counts, axis=1)
        observed_counts[np.arange(n_polls), max_idx] += diff

    # Sample-size-dependent concentration multiplier
    mean_sample_size = sample_sizes.mean()
    eps = 1e-8
    numerator = np.log(sample_sizes + 1 + eps)
    denominator = np.log(mean_sample_size + 1 + eps)
    sample_size_multiplier = np.maximum(numerator / denominator, eps)[:, np.newaxis]

    # Index arrays
    pollster_indices = polls["pollster_idx"].to_numpy().astype(int)

    # ── Build the model ───────────────────────────────────────────────
    with pm.Model() as model:  # type: ignore
        # ── Layer A: Municipal prior ──────────────────────────────────

        # Intercept per candidate
        alpha = pm.Normal("alpha", mu=0, sigma=1.0, shape=n_candidates)  # type: ignore

        # Shared beta coefficients (one per candidate per feature group)
        # Normal(0, 0.5) default; Horseshoe option for aggressive shrinkage.
        _sigma = config.beta_coefficient_prior_sigma
        if config.use_horseshoe_prior:
            tau_hs = pm.HalfCauchy("tau_horseshoe", beta=1.0)  # type: ignore[reportUnknownMemberType]
            beta_historical = _horseshoe_beta("beta_historical", tau_hs, _sigma, n_candidates)  # type: ignore[reportUnknownVariableType]
            beta_ethnicity = _horseshoe_beta("beta_ethnicity", tau_hs, _sigma, n_candidates)  # type: ignore[reportUnknownVariableType]
            beta_poverty = _horseshoe_beta("beta_poverty", tau_hs, _sigma, n_candidates)  # type: ignore[reportUnknownVariableType]
            beta_rural = _horseshoe_beta("beta_rural", tau_hs, _sigma, n_candidates)  # type: ignore[reportUnknownVariableType]
            beta_education = _horseshoe_beta("beta_education", tau_hs, _sigma, n_candidates)  # type: ignore[reportUnknownVariableType]
            beta_risk = _horseshoe_beta("beta_risk", tau_hs, _sigma, n_candidates)  # type: ignore[reportUnknownVariableType]
        else:
            beta_historical = pm.Normal("beta_historical", mu=0, sigma=_sigma, shape=n_candidates)  # type: ignore[reportUnknownMemberType]
            beta_ethnicity = pm.Normal("beta_ethnicity", mu=0, sigma=_sigma, shape=n_candidates)  # type: ignore[reportUnknownMemberType]
            beta_poverty = pm.Normal("beta_poverty", mu=0, sigma=_sigma, shape=n_candidates)  # type: ignore[reportUnknownMemberType]
            beta_rural = pm.Normal("beta_rural", mu=0, sigma=_sigma, shape=n_candidates)  # type: ignore[reportUnknownMemberType]
            beta_education = pm.Normal("beta_education", mu=0, sigma=_sigma, shape=n_candidates)  # type: ignore[reportUnknownMemberType]
            beta_risk = pm.Normal("beta_risk", mu=0, sigma=_sigma, shape=n_candidates)  # type: ignore[reportUnknownMemberType]

        # Non-centered municipal-level random effects
        sigma_m = pm.HalfNormal("sigma_m", sigma=config.sigma_m_prior, shape=n_candidates)  # type: ignore
        mu_m_raw = pm.Normal(  # type: ignore
            "mu_m_raw", mu=0, sigma=config.pool_alpha, shape=(n_municipalities, n_candidates)
        )

        # Linear predictor: logit(p_mk) = alpha_k + Σ beta_g[k] · z_m + sigma_m[k] · mu_raw[m,k]
        logit_p = (  # type: ignore[operator, index]
            alpha[None, :]
            + beta_historical[None, :] * clr_left_z[:, None]  # type: ignore[index]
            + beta_ethnicity[None, :] * pct_afro_z[:, None]  # type: ignore[index]
            + beta_poverty[None, :] * clr_nbi_z[:, None]  # type: ignore[index]
            + beta_rural[None, :] * pct_rural_z[:, None]  # type: ignore[index]
            + beta_education[None, :] * years_schooling_z[:, None]  # type: ignore[index]
            + beta_risk[None, :] * high_risk[:, None]  # type: ignore[index]
            + sigma_m[None, :] * mu_m_raw
        )

        p_municipal = pm.Deterministic(  # type: ignore
            "p_municipal",
            pm.math.softmax(logit_p, axis=-1),  # type: ignore
        )

        # ── Layer B: National poll likelihood ─────────────────────────

        # Turnout-weighted national rollup (n_candidates,)
        natl_base = (pop_weights[:, None] * p_municipal).sum(axis=0)  # type: ignore
        p_natl = pm.Deterministic("p_natl", natl_base)  # type: ignore

        # House effects
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

        house_selected = house_effects[pollster_indices]  # type: ignore

        # Poll likelihood: p_natl serves as base, adjusted by house effects
        p_poll = pm.Deterministic(  # type: ignore
            "p_poll",
            pm.math.softmax(  # type: ignore
                pm.math.log(p_natl + _EPSILON)[None, :] + house_selected,  # type: ignore
                axis=-1,
            ),
        )

        alpha_poll = pm.math.maximum(p_poll * phi_poll_n, _EPSILON)  # type: ignore

        pm.DirichletMultinomial(  # type: ignore
            "poll_likelihood",
            n=sample_sizes,
            a=alpha_poll,
            observed=observed_counts,
        )

        # ── Layer C: Election rollup likelihood (optional) ────────────
        if results is not None:
            p_elec = pm.Deterministic("p_elec", p_natl)  # type: ignore
            phi_elec = pm.Gamma(  # type: ignore
                "phi_elec",
                alpha=5,
                beta=5.0 / config.concentration_election_prior_mean,
            )
            alpha_elec = p_elec * phi_elec  # type: ignore

            vote_dict = {c.candidate_key: c.votes for c in results.candidates}
            election_counts = np.array(
                [vote_dict.get(k, 0) for k in candidate_keys],
                dtype=int,
            )

            pm.DirichletMultinomial(  # type: ignore
                "election_likelihood",
                n=election_counts.sum(),
                a=alpha_elec,
                observed=election_counts,
            )

    return model


# ═══════════════════════════════════════════════════════════════════════
# Sampling
# ═══════════════════════════════════════════════════════════════════════


def sample_municipal_model(
    model: pm.Model,
    config: ModelConfig,
) -> DataTree:
    """Sample posterior draws from a built municipal model via NUTS.

    Args:
        model: A built ``pm.Model`` from :func:`build_municipal_model`.
        config: Model hyperparameters (draws, tune, chains, etc.).

    Returns:
        ``DataTree`` with posterior and sampling stats.

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


# ═══════════════════════════════════════════════════════════════════════
# Forecast dataclasses
# ═══════════════════════════════════════════════════════════════════════


from dataclasses import dataclass  # noqa: E402 — import after model code


@dataclass(frozen=True)
class MunicipalForecast:
    """Per-municipality forecast from the hierarchical model.

    Attributes:
        codigo_municipio: 5-digit DANE municipality code.
        p_municipal_mean: Mean municipal vote share for each candidate.
        turnout_mean: Mean turnout from historical data.

    """

    codigo_municipio: str
    p_municipal_mean: dict[str, float]
    turnout_mean: float


@dataclass(frozen=True)
class HierarchicalForecast:
    """Full hierarchical forecast: national + per-municipality.

    Attributes:
        national_p_mean: National-level vote share mean per candidate.
        municipal: Per-municipality forecasts.
        effective_num_municipalities: Approximate effective sample size
            accounting for shrinkage.

    """

    national_p_mean: dict[str, float]
    municipal: tuple[MunicipalForecast, ...]
    effective_num_municipalities: int
