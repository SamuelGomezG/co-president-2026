"""SPEC-30: Data-driven transfer rate estimation model.

Replaces hand-tuned ``TRANSFER_*`` constants with a Bayesian ecological
inference model that estimates runoff voter transfer rates from historical
election data (R1→R2 shifts) and municipal demographics.

The model pools data across all historical runoff years (2010, 2014, 2018)
and estimates per-eliminated-candidate transfer probabilities as a function
of demographic and legislative covariates.

Public functions
----------------
:func:`build_transfer_model` — construct the PyMC ecological inference model.
:func:`sample_transfer_rates` — return per-draw transfer rate estimates.
:func:`map_moe_party_to_canonical` — MOE→canonical party crosswalk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import arviz as az  # type: ignore[reportMissingTypeStubs]
import numpy as np
import pandas as pd
import pymc as pm  # type: ignore[reportMissingTypeStubs]

if TYPE_CHECKING:
    import xarray as xr

    from co_president.config import ModelConfig

from co_president.config import COALITION_TO_CANONICAL_WEIGHTS

logger = logging.getLogger(__name__)

__all__ = [
    "build_transfer_model",
    "map_moe_party_to_canonical",
    "sample_transfer_rates",
]

_TRANSFER_POSTERIOR_PATH: str = "results/transfer_posterior.nc"
_HISTORICAL_RESULTS_PATH: str = "data/fundamentals/historical_results.csv"
_EPSILON: float = 1e-12
_MIN_VALID_OBS: int = 5

# Global list of eliminated candidates in runoff elections we estimate rates for
_ELIMINATED: tuple[str, ...] = (
    "sergio_fajardo",
    "federico_gutierrez",
    "ingrid_betancourt",
    "rest",
    "blanco",
)

_FEATURE_COLS: tuple[str, ...] = (
    "pct_afro_colombian",
    "nbi_rate",
    "pct_rural_disperso",
    "camara_left_share",
    "senado_left_share",
)


def map_moe_party_to_canonical(
    coalition_line: str,
    dept: str | None = None,
) -> dict[str, float]:
    """Map MOE 2022 coalition line to canonical party weights.

    Looks up ``COALITION_TO_CANONICAL_WEIGHTS`` for the given coalition line
    and optional department.  Falls back to the ``__national__`` entry when no
    department-specific weights exist.

    Args:
        coalition_line: Party/coalition name from MOE ``nomparti`` column.
        dept: Optional department name for per-department lookup.

    Returns:
        Dict mapping canonical party name to weight fraction.

    Examples:
        >>> map_moe_party_to_canonical("COALICION PACTO HISTORICO")
        {'Colombia Humana (15)': 0.25, ...}

    """
    entry = COALITION_TO_CANONICAL_WEIGHTS.get(coalition_line)
    if entry is None:
        return {coalition_line: 1.0}

    if dept is not None and dept in entry:
        dept_entry = entry[dept]
        if isinstance(dept_entry, dict):
            return dept_entry

    national = entry.get("__national__")
    if isinstance(national, dict):
        return national
    return {coalition_line: 1.0}


def _standardize_col(
    series: pd.Series,
) -> np.ndarray:
    """Return z-scored numpy array, handling zero-std edge case."""
    arr = series.to_numpy(dtype=float)
    mu = arr.mean()
    sd = arr.std()
    if sd < _EPSILON:
        return np.zeros_like(arr)
    return (arr - mu) / sd


def _build_obs_list(
    historical_r1r2: dict[int, tuple[pd.DataFrame, pd.DataFrame]],
    features: pd.DataFrame,
    feature_names: list[str],
) -> list[dict[str, np.ndarray]]:
    """Build stacked observation list from historical R1/R2 data.

    Merges municipal features onto historical R1→R2 data by ``codmpio``
    so feature vectors are automatically aligned with observations.

    Args:
        historical_r1r2: Mapping of election year to ``(round1_df, round2_df)``.
        features: Municipal feature DataFrame with ``codigo_municipio``
            and columns in *feature_names*.
        feature_names: Subset of feature columns to use.

    Returns:
        List of observation dicts, one per (eliminated, year) group.

    Raises:
        ValueError: If no valid observations can be constructed.

    """
    _features_lookup = features[["codigo_municipio", *feature_names]].copy()
    _features_lookup.columns = ["codmpio", *feature_names]
    _features_lookup["codmpio"] = _features_lookup["codmpio"].astype(str).str.zfill(5)

    obs_list: list[dict[str, np.ndarray]] = []
    for year, (r1, r2) in historical_r1r2.items():
        r1c = r1.copy()
        r2c = r2.copy()
        r1c["codmpio"] = r1c["codmpio"].astype(str).str.zfill(5)
        r2c["codmpio"] = r2c["codmpio"].astype(str).str.zfill(5)
        merged = r1c.merge(r2c, on="codmpio", suffixes=("_r1", "_r2"), how="inner")
        merged = merged.merge(_features_lookup, on="codmpio", how="inner")

        if merged.empty:
            continue

        r2_cols = [c for c in merged.columns if c.endswith("_r2")]

        if not r2_cols:
            continue

        left_key = r2_cols[0]

        left_r2 = merged[left_key].to_numpy(dtype=float) / 100.0
        left_r1 = (
            merged.get(
                left_key.replace("_r2", "_r1"),
                pd.Series(np.zeros(len(merged))),
            ).to_numpy(dtype=float)
            / 100.0
        )

        # Build list of all R1-only columns (eliminated candidates)
        _all_r1_candidate_cols = [
            c
            for c in merged.columns
            if c != "codmpio"
            and not c.endswith("_r2")
            and c not in feature_names
            and pd.api.types.is_numeric_dtype(merged[c].dtype)
        ]

        for elim in _ELIMINATED:
            elim_col = [c for c in _all_r1_candidate_cols if elim in str(c)]
            if not elim_col:
                continue
            elim_r1 = merged[elim_col[0]].to_numpy(dtype=float) / 100.0

            valid = ~np.isnan(left_r2) & ~np.isnan(elim_r1) & (elim_r1 > 0)
            if valid.sum() < _MIN_VALID_OBS:
                continue

            x_mat = np.column_stack(
                [_standardize_col(merged[col].loc[valid]) for col in feature_names],
            )

            obs_list.append(
                {  # type: ignore[reportArgumentType]
                    "elim": elim,
                    "year": year,
                    "left_r2": left_r2[valid],
                    "left_r1": left_r1[valid],
                    "elim_r1": elim_r1[valid],
                    "features": x_mat,
                },
            )

    if not obs_list:
        msg = "No valid observations could be constructed from historical_r1r2"
        raise ValueError(msg)

    return obs_list


def build_transfer_model(
    features: pd.DataFrame,
    historical_r1r2: dict[int, tuple[pd.DataFrame, pd.DataFrame]],
    camara: pd.DataFrame | None = None,  # noqa: ARG001
    senado: pd.DataFrame | None = None,  # noqa: ARG001
    config: ModelConfig | None = None,
) -> pm.Model:
    r"""Build ecological inference model for runoff transfer rates.

    For each eliminated candidate :math:`c`, estimates the probability that
    their round-1 voters transfer to the left-wing runoff candidate:

    .. math::
        \text{logit}(\beta_{c \to A_m}) = \gamma_{0c}
            + \gamma_\text{ethn} \cdot \text{pct\_afro}_m
            + \gamma_\text{poverty} \cdot \text{nbi\_rate}_m
            + \gamma_\text{rural} \cdot \text{pct\_rural\_disperso}_m
            + \gamma_\text{cámara} \cdot \text{cámara\_share}_m
            + \gamma_\text{senado} \cdot \text{senado\_share}_m

    The likelihood uses an ecological inference approximation: expected
    round-2 share for the left candidate equals the round-1 left share plus
    the sum of eliminated-candidate shares weighted by their transfer rates.

    Args:
        features: DataFrame indexed by municipality with columns
            ``pct_afro_colombian``, ``nbi_rate``, ``pct_rural_disperso``,
            ``camara_left_share``, ``senado_left_share``.
        historical_r1r2: Mapping of election year to
            ``(round1_df, round2_df)``. Each DataFrame has a ``codmpio``
            column and per-candidate vote-share columns.  Round 2 DataFrames
            must have columns for both runoff candidates.
        camara: Not used directly; ``camara_left_share`` must be in features.
        senado: Not used directly; ``senado_left_share`` must be in features.
        config: Model hyperparameters (uses
            ``beta_coefficient_prior_sigma`` for $\\gamma$ priors).

    Returns:
        pm.Model: Constructed PyMC transfer model.

    Raises:
        ValueError: If features is empty or historical_r1r2 is empty.

    """
    if features.empty:
        msg = "features DataFrame is empty"
        raise ValueError(msg)
    if not historical_r1r2:
        msg = "historical_r1r2 dict is empty"
        raise ValueError(msg)

    sigma_prior = config.beta_coefficient_prior_sigma if config is not None else 0.5

    # Z-score each feature column for numerical stability
    feature_names = [c for c in _FEATURE_COLS if c in features.columns]
    if not feature_names:
        msg = f"features must contain at least one of {_FEATURE_COLS}"
        raise ValueError(msg)

    n_features = len(feature_names)
    obs_list = _build_obs_list(historical_r1r2, features, feature_names)

    with pm.Model() as model:
        # Hierarchical priors: global means + per-candidate offsets
        gamma_mu = pm.Normal("gamma_mu", 0, sigma_prior, shape=n_features)  # type: ignore
        gamma_0_mu = pm.Normal("gamma_0_mu", 0, sigma_prior)  # type: ignore

        gamma_0_offset = pm.Normal("gamma_0_offset", 0, 0.25, shape=len(_ELIMINATED))  # type: ignore
        gamma_offset = pm.Normal("gamma_offset", 0, 0.25, shape=(len(_ELIMINATED), n_features))  # type: ignore

        gamma_0 = gamma_0_mu + gamma_0_offset  # type: ignore
        gamma_coefs = gamma_mu + gamma_offset  # type: ignore

        for o in obs_list:
            c_idx = _ELIMINATED.index(o["elim"])

            g0 = gamma_0[c_idx]  # type: ignore
            g_coef = gamma_coefs[c_idx]  # type: ignore
            x_o = o["features"]

            logit_beta = g0 + pm.math.dot(x_o, g_coef)  # type: ignore
            beta = pm.Deterministic(  # type: ignore
                f"beta_{o['elim']}_{o['year']}",
                pm.math.sigmoid(logit_beta),  # type: ignore
            )

            # Expected R2 share for left candidate
            expected = o["left_r1"] + o["elim_r1"] * beta

            # Likelihood
            pm.Normal(  # type: ignore
                f"obs_{o['elim']}_{o['year']}",
                mu=expected,
                sigma=pm.HalfNormal(f"sigma_{o['elim']}_{o['year']}", 0.1),  # type: ignore
                observed=o["left_r2"],
            )

    return model


def _load_historical_r1r2(
    exclude_year: int = 2022,
) -> dict[int, tuple[pd.DataFrame, pd.DataFrame]]:
    """Load historical R1→R2 data for runoff years from disk.

    Reads ``historical_results.csv``, pivots from long to wide format,
    and returns a dict mapping runoff year to ``(round1_df, round2_df)``.
    Each DataFrame has a ``codmpio`` column and per-candidate vote-share
    columns.  Voting-record categories (blank, null, unmarked) are excluded
    from the Round 2 DataFrames.

    Args:
        exclude_year: Year to exclude (default 2022, the target year).

    Returns:
        Dict mapping year to ``(r1, r2)`` DataFrames.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If no runoff years can be constructed.

    """
    path = Path(_HISTORICAL_RESULTS_PATH)
    if not path.exists():
        msg = f"Historical results not found at {_HISTORICAL_RESULTS_PATH}"
        raise FileNotFoundError(msg)

    raw = pd.read_csv(path)
    raw_cols = [str(c) for c in raw.columns]
    if "codigo_municipio" not in raw_cols:
        msg = "historical_results.csv must contain 'codigo_municipio' column"
        raise ValueError(msg)

    # Rename to match _build_obs_list expected column name
    df: pd.DataFrame = raw.rename(columns={"codigo_municipio": "codmpio"})

    # Runoff years in the data
    runoff_round = 2
    unique_years = df["year"].unique()  # type: ignore[reportUnknownMemberType]
    runoff_years = sorted(
        y
        for y in unique_years
        if runoff_round in df[df["year"] == y]["round"].to_numpy()  # type: ignore[reportUnknownMemberType]
    )
    runoff_years = [y for y in runoff_years if y != exclude_year]

    if not runoff_years:
        msg = f"No historical runoff years available (excluded {exclude_year})"
        raise ValueError(msg)

    # Columns to exclude from R2 (voting-record categories)
    r2_exclude_patterns = ("BLANCO", "NULO", "MARCAD", "TARJETAS")

    result: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for year in runoff_years:
        year_df: pd.DataFrame = df[df["year"] == year].copy()  # type: ignore[reportUnknownMemberType]
        r1_long: pd.DataFrame = year_df[year_df["round"] == 1]  # type: ignore[reportUnknownMemberType]
        r2_long: pd.DataFrame = year_df[year_df["round"] == 2]  # type: ignore[reportUnknownMemberType]  # noqa: PLR2004

        if r1_long.empty or r2_long.empty:  # type: ignore[reportUnknownMemberType]
            continue

        # Pivot to wide: codmpio x candidate, values=vote_share
        r1_pivot: pd.DataFrame = r1_long.pivot_table(  # type: ignore[reportUnknownMemberType]
            index="codmpio",
            columns="candidate",
            values="vote_share",
            aggfunc="first",
        )
        r2_pivot: pd.DataFrame = r2_long.pivot_table(  # type: ignore[reportUnknownMemberType]
            index="codmpio",
            columns="candidate",
            values="vote_share",
            aggfunc="first",
        )
        r1_wide: pd.DataFrame = r1_pivot.reset_index()  # type: ignore[reportUnknownMemberType]
        r2_wide: pd.DataFrame = r2_pivot.reset_index()  # type: ignore[reportUnknownMemberType]

        # Exclude voting-record columns from R2
        r2_cols: list[str] = []  # type: ignore[reportUnknownMemberType]
        for c in r2_wide.columns:  # type: ignore[reportUnknownMemberType]
            c_str = str(c)  # type: ignore[reportUnknownArgumentType]
            if c_str != "codmpio" and any(p in c_str.upper() for p in r2_exclude_patterns):
                r2_cols.append(c_str)
        r2_wide = r2_wide.drop(columns=r2_cols)  # type: ignore[reportUnknownMemberType]

        # Ensure at least 2 candidates remain in R2
        r2_cands: list[str] = [c for c in r2_wide.columns if c != "codmpio"]  # type: ignore[reportUnknownMemberType]
        if len(r2_cands) < 2:  # noqa: PLR2004
            continue

        result[year] = (r1_wide, r2_wide)

    if not result:
        msg = f"Could not construct any historical R1→R2 pairs from runoff years {runoff_years}"
        raise ValueError(msg)

    return result


def _build_historical_transfer_prior(
    eliminated: tuple[str, ...],
    seed: int = 332211,
) -> dict[tuple[str, str], np.ndarray]:
    """Generate fallback transfer rates from historical Dirichlet-Categorical prior.

    Uses the empirically calibrated aggregate split from
    ``derive_transfer_constants.py`` (~27% to Petro, ~73% to Hernandez) as
    the concentration for a Dirichlet prior.  Returns per-candidate-pair
    transfer samples without running NUTS (used when the full model fails
    to converge).

    Args:
        eliminated: Tuple of eliminated candidate keys.
        seed: RNG seed for reproducibility.

    Returns:
        Dict mapping ``(eliminated, target)`` to 1-D array of transfer
        rate samples matching the historical data distribution.

    """
    rng = np.random.default_rng(seed)
    n_draws = 4_000

    # Aggregate historical split ~27% Petro, ~73% Hernández.
    # Dirichlet alpha [3.70, 8.30] gives sum=12 (matching expected
    # concentration of original gamma(5,1)+1 prior) and mode at 27/73.
    alpha_petro = 3.70
    alpha_hernandez = 8.30
    alpha = np.array([alpha_petro, alpha_hernandez])

    results: dict[tuple[str, str], np.ndarray] = {}
    for elim in eliminated:
        raw = rng.dirichlet(alpha, size=n_draws)
        results[(elim, "gustavo_petro")] = raw[:, 0]
        results[(elim, "rodolfo_hernandez")] = raw[:, 1]

    return results


def _load_transfer_posterior() -> xr.DataTree | None:
    """Load cached transfer posterior from NetCDF, or None if unavailable."""
    path = Path(_TRANSFER_POSTERIOR_PATH)
    if not path.exists():
        logger.info("No cached transfer posterior at %s", _TRANSFER_POSTERIOR_PATH)
        return None
    # TRY300 note: return in else block per ruff convention
    try:
        idata = az.from_netcdf(str(path))  # type: ignore[reportUnknownVariableType]
    except (ValueError, FileNotFoundError, ImportError):
        logger.warning("Failed to load transfer posterior from %s", _TRANSFER_POSTERIOR_PATH)
        return None
    else:
        logger.info("Loaded transfer posterior from %s", _TRANSFER_POSTERIOR_PATH)
        return idata  # pyright: ignore[reportUnknownVariableType]


def sample_transfer_rates(
    features: pd.DataFrame | None = None,
    config: ModelConfig | None = None,
    n_draws: int = 4000,
) -> dict[tuple[str, str], np.ndarray]:
    """Return per-draw transfer rate estimates for the runoff model.

    Resolution order:
    1. Load cached PyMC posterior from ``transfer_posterior.nc``.
    2. If ``features`` is provided and no cached posterior, train a
       hierarchical ecological inference model on-demand using historical
       runoff data (2010, 2014, 2018 R1→R2 shifts).
    3. Fall back to a calibrated Dirichlet prior with ~27/73 Petro/Hernández
       aggregate split.

    Args:
        features: Municipal features DataFrame with at least one of
            ``_FEATURE_COLS``.  Required for on-demand training.
        config: ModelConfig (or ``None`` for defaults).
        n_draws: Number of posterior draws to return (ignored when cached
            posterior is loaded; defaults to 4000).

    Returns:
        Dict mapping ``(eliminated_candidate_key, target_candidate_key)`` to
        a 1-D numpy array of ``n_draws`` transfer rate samples in ``[0, 1]``.

    """
    # Step 1: try cached posterior
    idata = _load_transfer_posterior()
    if idata is not None:
        try:
            return _extract_transfer_rates_from_idata(idata)
        except (KeyError, ValueError) as exc:
            logger.warning(
                "Failed to extract transfer rates from cached posterior: %s. "
                "Trying on-demand training.",
                exc,
            )

    # Step 2: on-demand training when features are provided
    if features is not None and not features.empty and config is not None:
        try:
            logger.info("Attempting on-demand transfer model training...")
            historical_r1r2 = _load_historical_r1r2(
                exclude_year=config.target_year,
            )
            model = build_transfer_model(features, historical_r1r2, config=config)
            with model:
                idata_train = pm.sample(  # type: ignore[reportUnknownMemberType]
                    draws=min(n_draws // 2, 1000),
                    tune=500,
                    chains=2,
                    cores=1,
                    random_seed=332211,
                    progressbar=False,
                )
            # Cache the posterior for future runs
            try:
                idata_train.to_netcdf(  # type: ignore[reportUnknownMemberType]
                    str(Path(_TRANSFER_POSTERIOR_PATH)),
                )
                logger.info("Saved transfer posterior to %s", _TRANSFER_POSTERIOR_PATH)
            except (OSError, ImportError):
                logger.warning("Could not save transfer posterior to disk")

            return _extract_transfer_rates_from_idata(idata_train)  # type: ignore[reportUnknownArgumentType]
        except (FileNotFoundError, ValueError, RuntimeError, KeyError, IndexError) as exc:
            logger.warning(
                "On-demand transfer model training failed: %s. Falling back to historical prior.",
                exc,
            )

    # Step 3: calibrated fallback
    logger.info("Using historical Dirichlet-Categorical fallback for transfer rates")
    return _build_historical_transfer_prior(_ELIMINATED)


def _extract_transfer_rates_from_idata(
    idata: xr.DataTree,
) -> dict[tuple[str, str], np.ndarray]:
    """Extract per-candidate-pair transfer rates from cached InferenceData.

    Args:
        idata: Cached posterior DataTree containing beta_* samples.

    Returns:
        Dict mapping ``(eliminated, target)`` to array of shape ``(n_draws,)``.

    Raises:
        ValueError: If expected variables are missing from the posterior.

    """
    posterior = getattr(idata, "posterior", None)
    if posterior is None:
        msg = "Cached posterior has no 'posterior' group"
        raise ValueError(msg)

    # Look for beta variables (shape: chain, draw, municipality)
    beta_vars = [v for v in list(posterior.data_vars) if v.startswith("beta_")]
    if not beta_vars:
        msg = "No beta_* variables found in cached posterior"
        raise ValueError(msg)

    results: dict[tuple[str, str], np.ndarray] = {}

    for elim in _ELIMINATED:
        matching = [v for v in beta_vars if elim in v]
        if not matching:
            continue

        # Average across years and municipalities to get aggregate transfer rate
        beta_samples = posterior[matching[0]].to_numpy()
        n_chains, n_draw_local = beta_samples.shape[0], beta_samples.shape[1]
        flat = beta_samples.reshape(n_chains * n_draw_local, -1).mean(axis=1)

        results[(elim, "gustavo_petro")] = flat
        results[(elim, "rodolfo_hernandez")] = 1.0 - flat

    if not results:
        msg = "Could not extract any transfer rates from cached posterior"
        raise ValueError(msg)

    return results
