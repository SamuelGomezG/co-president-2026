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
from co_president.fundamentals.compositional import (
    _apply_zero_floor,  # type: ignore[reportPrivateUsage]
)
from co_president.fundamentals.features import clr

if TYPE_CHECKING:
    import pytensor.tensor as pt
    from xarray import DataTree

    from co_president.config import ModelConfig
    from co_president.data import RoundResult

logger = logging.getLogger(__name__)

_EMPTY_DF: pd.DataFrame = pd.DataFrame()

__all__ = [
    "HierarchicalForecast",
    "MunicipalForecast",
    "build_municipal_model",
    "compute_effective_num_municipalities",
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


def _horseshoe_beta(  # type: ignore[reportUnknownParameterType, reportMissingTypeArgument, reportPrivateImportUsage]
    name: str,
    tau: pt.TensorVariable,  # type: ignore[reportMissingTypeArgument, reportPrivateImportUsage]
    sigma: float,
    shape: int,
) -> pt.TensorVariable:  # type: ignore[reportMissingTypeArgument, reportPrivateImportUsage]
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


def build_municipal_model(  # noqa: C901, PLR0912, PLR0913, PLR0915
    features: pd.DataFrame,
    polls: pd.DataFrame,
    results: RoundResult | None,
    config: ModelConfig,
    target_year: int = 2022,
    *,
    digital_signals: pd.DataFrame = _EMPTY_DF,
    prior_only: bool = False,
) -> pm.Model:
    """Build the 3-layer municipal hierarchical PyMC model.

    Args:
        features: Municipal feature matrix from :func:`load_features`.
            Must contain ``codigo_municipio``, ``pop_2022``,
            ``historical_turnout_m``, ``pct_afro_colombian``, ``nbi_rate``,
            ``pct_rural_disperso``, ``years_schooling_promedio``,
            ``high_risk_flag``, ``is_pdet``, ``pct_indigenous``,
            ``internet_access_rate``, and ``historical``.
        polls: Clean poll DataFrame with candidate columns (percentage
            shares), ``fecha``, ``encuestadora``, ``muestra``.
        results: Election result for validation (optional).
        config: Model hyperparameters.
        target_year: Election year to extract ``clr_left_share`` from the
            ``historical`` column.  Default 2022.
        digital_signals: Optional DataFrame with ``fecha`` and per-candidate
            columns containing digital signal values (e.g., Google Trends
            favorable propensity).  Merged into poll likelihood as an
            additional linear term.  Backward-filled via ``merge_asof`` to
            poll dates when signal dates are sparse.
        prior_only: When True, builds Layer A (municipal prior) only,
            skipping Layer B (poll likelihood) and Layer C (election
            rollup).  Used for extracting the structural prior without
            observation data.

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
        "is_pdet",
        "pct_indigenous",
        "internet_access_rate",
        "historical",
    )
    _missing = [col for col in _required_columns if col not in features.columns]
    if _missing:
        msg = f"features is missing required columns: {_missing}. Run `make fundamentals` first."
        raise ValueError(msg)

    if len(features) < 2:  # noqa: PLR2004
        msg = f"features must have at least 2 municipalities, got {len(features)}"
        raise ValueError(msg)

    # Candidate keys from DataFrame columns (determined before empty check
    # since prior_only mode passes empty polls with correct columns)
    candidate_keys = sorted(set(FIRST_ROUND_CANDIDATES) & set(polls.columns))
    n_candidates = len(candidate_keys)
    if n_candidates == 0:
        msg = "No candidate columns found in polls DataFrame"
        raise ValueError(msg)

    if not prior_only:
        if len(polls) == 0:
            msg = "polls DataFrame is empty"
            raise ValueError(msg)

        _required_poll_cols = ("fecha", "encuestadora", "muestra")
        _missing_poll = [c for c in _required_poll_cols if c not in polls.columns]
        if _missing_poll:
            msg = f"polls is missing required columns: {_missing_poll}"
            raise ValueError(msg)

    n_municipalities = len(features)

    # ── Feature extraction ────────────────────────────────────────────
    pop = features["pop_2022"].to_numpy().astype(float)
    turnout = features["historical_turnout_m"].to_numpy().copy()
    if np.isnan(pop).any():
        _fill_pop = float(np.nanmean(pop))
        pop[np.isnan(pop)] = _fill_pop
    if np.isnan(turnout).any():
        _fill_turn = float(np.nanmean(turnout))
        turnout[np.isnan(turnout)] = _fill_turn
    if config.enable_population_weighting:
        effective_pop = compute_effective_pop(pop, turnout)
    else:
        effective_pop = pop.copy()
    pop_weights = effective_pop / effective_pop.sum()

    # Feature arrays (M,)
    clr_left = _extract_clr_left_share(features, target_year)
    pct_afro = features["pct_afro_colombian"].to_numpy().copy()
    nbi_rate = features["nbi_rate"].to_numpy().copy()
    # Fill NaN before CLR transform
    _nbi_nan = np.isnan(nbi_rate)
    if _nbi_nan.any():
        nbi_rate[_nbi_nan] = float(np.nanmean(nbi_rate))
    clr_nbi = _clr_nbi_array(nbi_rate)
    pct_rural = features["pct_rural_disperso"].to_numpy().copy()
    years_schooling = features["years_schooling_promedio"].to_numpy().copy()
    high_risk = features["high_risk_flag"].to_numpy().astype(float)
    is_pdet_col = features["is_pdet"].to_numpy().astype(float)
    pct_indigenous_col = features["pct_indigenous"].to_numpy().copy()
    internet_access = features["internet_access_rate"].to_numpy().copy()

    # Fill NaN with feature mean for continuous arrays (pandas may have missing values)
    for _arr in (pct_afro, pct_rural, years_schooling, pct_indigenous_col, internet_access):
        _mask = np.isnan(_arr)
        if _mask.any():
            _arr[_mask] = float(np.nanmean(_arr))

    # Standardize continuous features
    def _z(x: np.ndarray) -> np.ndarray:
        return (x - x.mean()) / (x.std() + _EPSILON)

    clr_left_z = _z(clr_left)
    pct_afro_z = _z(pct_afro)
    clr_nbi_z = _z(clr_nbi)
    pct_rural_z = _z(pct_rural)
    years_schooling_z = _z(years_schooling)
    pct_indigenous_z = _z(pct_indigenous_col)
    internet_access_z = _z(internet_access)
    # high_risk, is_pdet are binary, no standardization

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

    has_digital_signal = False
    ds_values: np.ndarray | None = None

    n_polls = len(polls)

    # Pre-initialize poll-computed variables (used in posterior block below;
    # pyright can't infer n_polls>0 from not prior_only guard).
    n_time_points: int = 1
    n_pollsters: int = 1
    sample_sizes: np.ndarray = np.array([1], dtype=int)
    observed_counts: np.ndarray = np.zeros((1, n_candidates), dtype=int)
    sample_size_multiplier: np.ndarray = np.ones((1, n_candidates))
    pollster_indices: np.ndarray = np.zeros(1, dtype=int)
    time_indices: np.ndarray = np.zeros(1, dtype=int)

    if n_polls > 0:
        # Time index: all polls at election day for the municipal model
        # (the fundamentals are time-invariant; all polls mapped to election day)
        polls["days_before"] = (pd.Timestamp(ELECTION_DATE_ROUND1) - polls["fecha"]).dt.days
        unique_days = sorted(polls["days_before"].unique())
        n_time_points = len(unique_days)
        day_to_idx = {day: i for i, day in enumerate(unique_days)}
        polls["time_idx"] = polls["days_before"].map(day_to_idx)

        # Pollster index
        unique_pollsters = polls["encuestadora"].unique()
        n_pollsters = len(unique_pollsters)
        pollster_to_idx = {name: i for i, name in enumerate(unique_pollsters)}
        polls["pollster_idx"] = polls["encuestadora"].map(pollster_to_idx)

        # Observed counts from percentage shares
        sample_sizes: np.ndarray = polls["muestra"].to_numpy().astype(int)
        raw_probs: np.ndarray = polls[candidate_keys].to_numpy() / 100.0
        raw_probs = raw_probs / raw_probs.sum(axis=1, keepdims=True)
        observed_counts: np.ndarray = np.round(
            raw_probs * sample_sizes[:, np.newaxis],
        ).astype(int)

        row_sums: np.ndarray = observed_counts.sum(axis=1)
        diff: np.ndarray = sample_sizes - row_sums
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

        # Zero floor — replace 0 with 1, redistributing from largest columns
        # to preserve row sum (avoids log(0) in Dirichlet-Multinomial).
        _apply_zero_floor(observed_counts)

        # Sample-size-dependent concentration multiplier
        mean_sample_size = sample_sizes.mean()
        eps = 1e-8
        numerator = np.log(sample_sizes + 1 + eps)
        denominator = np.log(mean_sample_size + 1 + eps)
        sample_size_multiplier = np.maximum(numerator / denominator, eps)[:, np.newaxis]

        # Index arrays
        pollster_indices = polls["pollster_idx"].to_numpy().astype(int)
        time_indices: np.ndarray = polls["time_idx"].to_numpy().astype(int)

        # ── Digital signals (optional) ────────────────────────────────────
        if not digital_signals.empty:
            _ds_cols = [c for c in candidate_keys if c in digital_signals.columns]
            if _ds_cols:
                # Build per-date mean signal (handle duplicate dates)
                ds_subset = digital_signals[["fecha", *_ds_cols]].copy()
                ds_subset["fecha"] = pd.to_datetime(ds_subset["fecha"])
                ds_deduped = (
                    ds_subset.groupby("fecha")[_ds_cols].mean().reset_index().sort_values("fecha")
                )

                # Chronologically safe merge: each poll gets the most recent prior signal
                _polls_sorted = polls[["fecha"]].sort_values("fecha").reset_index()
                _polls_sorted["fecha"] = _polls_sorted["fecha"].astype("datetime64[us]")
                ds_deduped["fecha"] = ds_deduped["fecha"].astype("datetime64[us]")
                _merged = pd.merge_asof(
                    _polls_sorted,
                    ds_deduped,
                    on="fecha",
                    direction="backward",
                )
                _merged = _merged.set_index("index").sort_index()

                # Build (n_polls, n_candidates) array, padding missing cols with 0
                _vals = np.zeros((n_polls, n_candidates))
                for i, ck in enumerate(candidate_keys):
                    if ck in _ds_cols:
                        _vals[:, i] = _merged[ck].fillna(0.0).to_numpy()
                ds_values = _vals
                has_digital_signal = True
                logger.info(
                    "Merged digital-signal columns %s across %d polls",
                    _ds_cols,
                    n_polls,
                )
            else:
                logger.warning(
                    "digital_signals provided but no candidate columns (%s) found; ignoring",
                    candidate_keys,
                )

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
            beta_pdet = _horseshoe_beta("beta_pdet", tau_hs, _sigma, n_candidates)  # type: ignore[reportUnknownVariableType]
            beta_indigenous = _horseshoe_beta("beta_indigenous", tau_hs, _sigma, n_candidates)  # type: ignore[reportUnknownVariableType]
            beta_internet = _horseshoe_beta("beta_internet", tau_hs, _sigma, n_candidates)  # type: ignore[reportUnknownVariableType]
        else:
            beta_historical = pm.Normal("beta_historical", mu=0, sigma=_sigma, shape=n_candidates)  # type: ignore[reportUnknownMemberType]
            beta_ethnicity = pm.Normal("beta_ethnicity", mu=0, sigma=_sigma, shape=n_candidates)  # type: ignore[reportUnknownMemberType]
            beta_poverty = pm.Normal("beta_poverty", mu=0, sigma=_sigma, shape=n_candidates)  # type: ignore[reportUnknownMemberType]
            beta_rural = pm.Normal("beta_rural", mu=0, sigma=_sigma, shape=n_candidates)  # type: ignore[reportUnknownMemberType]
            beta_education = pm.Normal("beta_education", mu=0, sigma=_sigma, shape=n_candidates)  # type: ignore[reportUnknownMemberType]
            beta_risk = pm.Normal("beta_risk", mu=0, sigma=_sigma, shape=n_candidates)  # type: ignore[reportUnknownMemberType]
            beta_pdet = pm.Normal("beta_pdet", mu=0, sigma=_sigma, shape=n_candidates)  # type: ignore[reportUnknownMemberType]
            beta_indigenous = pm.Normal("beta_indigenous", mu=0, sigma=_sigma, shape=n_candidates)  # type: ignore[reportUnknownMemberType]
            beta_internet = pm.Normal("beta_internet", mu=0, sigma=_sigma, shape=n_candidates)  # type: ignore[reportUnknownMemberType]

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
            + beta_pdet[None, :] * is_pdet_col[:, None]  # type: ignore[index]
            + beta_indigenous[None, :] * pct_indigenous_z[:, None]  # type: ignore[index]
            + beta_internet[None, :] * internet_access_z[:, None]  # type: ignore[index]
            + sigma_m[None, :] * mu_m_raw
        )

        p_municipal = pm.Deterministic(  # type: ignore
            "p_municipal",
            pm.math.softmax(logit_p, axis=-1),  # type: ignore
        )

        if config.clr_target:
            log_p_muni = pm.math.log(p_municipal + _EPSILON)  # type: ignore[operator]
            log_geom_mean = pm.math.mean(log_p_muni, axis=-1, keepdims=True)  # type: ignore
            pm.Deterministic("p_municipal_clr", log_p_muni - log_geom_mean)  # type: ignore

        # ── Layer B: National poll likelihood ─────────────────────────

        # Turnout-weighted national rollup (n_candidates,)
        natl_base = (pop_weights[:, None] * p_municipal).sum(axis=0)  # type: ignore
        p_natl = pm.Deterministic("p_natl", natl_base)  # type: ignore

        if config.clr_target:
            log_p_natl = pm.math.log(p_natl + _EPSILON)  # type: ignore[operator, reportUnknownMemberType, reportUnknownArgumentType]
            log_geom_natl = pm.math.mean(log_p_natl, axis=-1, keepdims=True)  # type: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            p_natl_clr = pm.Deterministic("p_natl_clr", log_p_natl - log_geom_natl)  # type: ignore[reportUnknownVariableType, reportUnknownArgumentType]

        if not prior_only:
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
            # and reverse-time RW for sentiment drift.
            log_base = pm.math.log(p_natl + _EPSILON)[None, :] + house_selected  # type: ignore

            if n_time_points > 1:
                sigma_rw = pm.HalfNormal(  # type: ignore
                    "sigma_rw_muni",
                    sigma=config.random_walk_sigma_prior,
                )

                rw_list: list = [pm.Deterministic("delta_muni_0", pm.math.zeros(n_candidates))]  # type: ignore[reportUnknownVariableType, reportUnknownMemberType]
                for t_idx in range(1, n_time_points):
                    rw_list.append(  # type: ignore[reportUnknownMemberType]
                        pm.Normal(  # type: ignore
                            f"delta_muni_{t_idx}",
                            mu=rw_list[-1],  # type: ignore[reportUnknownArgumentType]
                            sigma=sigma_rw,
                            shape=n_candidates,
                        ),
                    )
                delta_time = pm.Deterministic("delta_time", pm.math.stack(rw_list))  # type: ignore
                log_base = log_base + delta_time[time_indices]  # type: ignore

            if has_digital_signal and ds_values is not None:
                beta_digital = pm.Normal(  # type: ignore[reportUnknownMemberType]
                    "beta_digital",
                    mu=0,
                    sigma=config.beta_coefficient_prior_sigma,
                    shape=n_candidates,
                )
                log_base = log_base + beta_digital[None, :] * ds_values  # type: ignore[operator]

            p_poll = pm.Deterministic(  # type: ignore
                "p_poll",
                pm.math.softmax(log_base, axis=-1),  # type: ignore
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
                vote_dict = {c.candidate_key: c.votes for c in results.candidates}
                election_counts = np.array(
                    [vote_dict.get(k, 0) for k in candidate_keys],
                    dtype=int,
                )

                if config.clr_target:
                    eps_obs = 1e-10
                    election_probs = election_counts.astype(float) / election_counts.sum()
                    log_election = np.log(election_probs + eps_obs)
                    clr_election_obs = log_election - log_election.mean()

                    sigma_clr = pm.HalfNormal("sigma_clr_elec", sigma=0.5, shape=n_candidates)  # type: ignore
                    for k_idx in range(n_candidates):
                        pm.Normal(  # type: ignore[reportPossiblyUnboundVariable, reportIndexIssue, reportCallIssue, reportArgumentType, reportUnknownMemberType]
                            f"election_clr_{k_idx}",
                            mu=p_natl_clr[..., k_idx],  # type: ignore[reportIndexIssue, reportCallIssue, reportArgumentType, reportPossiblyUnboundVariable]
                            sigma=sigma_clr[k_idx],  # type: ignore[reportIndexIssue, reportCallIssue, reportArgumentType]
                            observed=clr_election_obs[k_idx],
                        )
                else:
                    p_elec = pm.Deterministic("p_elec", p_natl)  # type: ignore
                    phi_elec = pm.Gamma(  # type: ignore
                        "phi_elec",
                        alpha=config.concentration_election_prior_shape,
                        beta=config.concentration_election_prior_shape
                        / config.concentration_election_prior_mean,
                    )
                    alpha_elec = pm.math.maximum(  # type: ignore[reportUnknownMemberType]
                        p_elec * phi_elec,  # type: ignore[operator]
                        _EPSILON,
                    )

                    total_votes_scaled = int(config.concentration_election_votes_scale)
                    _elec_sum = int(election_counts.sum())
                    observed_scaled = np.round(
                        election_counts.astype(float) / _elec_sum * total_votes_scaled,
                    ).astype(int)

                    pm.DirichletMultinomial(  # type: ignore
                        "election_likelihood",
                        n=int(observed_scaled.sum()),
                        a=alpha_elec,
                        observed=observed_scaled,
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
            init="adapt_diag",  # horseshoe → diagonal-dominant; adapt_full O(N³)
            random_seed=config.seed,
            nuts_sampler=config.nuts_sampler,
            mp_ctx="spawn",
        )


# ═══════════════════════════════════════════════════════════════════════
# Posterior diagnostics
# ═══════════════════════════════════════════════════════════════════════


def compute_effective_num_municipalities(
    idata: DataTree,
    pool_alpha: float,
) -> int:
    """Compute effective number of municipalities from posterior shrinkage weights.

    Shrinkage weight for municipality *m*:
        ``w_m = 1 - Var(mu_m_raw | data) / Var(prior)``

    Effective municipalities = ``sum(w_m)``, which should be << 1 122
    if shrinkage is working well.

    Args:
        idata: ArviZ ``DataTree`` with ``mu_m_raw`` posterior.
        pool_alpha: Prior standard deviation of ``mu_m_raw`` (``pool_alpha``
            from ``ModelConfig``).

    Returns:
        Effective number of municipalities (integer, rounded down).

    Raises:
        KeyError: If ``mu_m_raw`` is missing from the posterior.

    """
    prior_var = pool_alpha**2
    mu_samples = idata.posterior["mu_m_raw"].to_numpy()  # (chain, draw, M, K)
    posterior_var = mu_samples.var(axis=(0, 1), ddof=1).mean(axis=-1)  # (M,) avg over candidates

    _eps = 1e-12
    shrinkage = 1.0 - posterior_var / (prior_var + _eps)
    shrinkage = np.clip(shrinkage, 0.0, 1.0)
    effective = float(np.round(shrinkage.sum()))  # type: ignore[arg-type]
    return int(effective)


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
