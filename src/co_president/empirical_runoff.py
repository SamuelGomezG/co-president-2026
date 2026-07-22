"""SPEC-34: Empirical Beta posterior estimation from CNE runoff pairings.

Fits Beta distributions to CNE 2026 head-to-head runoff polling data and
provides sampling functions that can replace the ecological-inference transfer
model when sufficient empirical data is available (n_eff >= 50 per pairing).

Public functions
----------------
:func:`get_empirical_runoff_betas` — orchestrator: load cached or rebuild.
:func:`load_runoff_raw` — load runoff pairing data from CNE bundles.
:func:`compute_per_pairing_stats` — aggregate effective sample sizes per pairing.
:func:`fit_empirical_beta_per_pairing` — fit Beta(alpha, beta) per pairing.
:func:`sample_empirical_transfer` — draw from fitted empirical Beta.
:func:`cache_empirical_posterior` — save to netCDF.
:func:`load_empirical_posterior` — load from netCDF.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from co_president.paths import resolve_data_dir

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "cache_empirical_posterior",
    "compute_per_pairing_stats",
    "fit_empirical_beta_per_pairing",
    "get_empirical_runoff_betas",
    "load_empirical_posterior",
    "load_runoff_raw",
    "sample_empirical_transfer",
]

_N_EFF_THRESHOLD: float = 50.0
_CACHE_FILENAME_TEMPLATE: str = "empirical_runoff_posterior_{year}.nc"


def load_runoff_raw(data_dir: Path | None = None) -> pd.DataFrame:
    """Load runoff pairing data from CNE 2026 bundles.

    Reads ``data/2026-polls/_processed/2026_runoff_pairings.parquet`` if
    it exists.  Otherwise calls :func:`~co_president.data_cne_2026.build_cne_2026_tables`
    to build it on-demand.

    Args:
        data_dir: Optional override for the project data directory.

    Returns:
        DataFrame with columns: fecha, encuestadora, field_end, candidate_a,
        candidate_b, n_a, n_b, n_total, effective_n.  Empty DataFrame if no
        runoff pairings are available.

    """
    from co_president.data_cne_2026 import build_cne_2026_tables  # noqa: PLC0415

    root = resolve_data_dir(data_dir)
    processed_dir = root / "2026-polls" / "_processed"
    parquet_path = processed_dir / "2026_runoff_pairings.parquet"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    logger.info("Runoff pairings parquet not found; building CNE 2026 tables on-demand...")
    _topline, runoff, _report = build_cne_2026_tables(data_dir=data_dir)
    return runoff


def _canonical_pair(a: str, b: str) -> tuple[str, str]:
    """Return a canonical ordering for a candidate pair (sorted)."""
    return (a, b) if a <= b else (b, a)


def compute_per_pairing_stats(
    runoff_df: pd.DataFrame,
    year: int = 2026,  # noqa: ARG001
) -> pd.DataFrame:
    """Aggregate effective sample sizes per runoff pairing.

    Groups by symmetric ``(candidate_a, candidate_b)`` pairs and computes
    total effective sample size, wins for each candidate.

    Args:
        runoff_df: Long-format DataFrame from :func:`load_runoff_raw` with
            columns ``candidate_a``, ``candidate_b``, ``effective_n``.
        year: Election year (reserved for future use).

    Returns:
        DataFrame with columns: candidate_a, candidate_b, n_a, n_b,
        n_total, n_eff.  ``n_a`` and ``n_b`` sum to ``n_total``, and
        ``n_eff`` is the sum of ``effective_n`` across rows for the pairing.

    """
    if runoff_df.empty:
        return pd.DataFrame(
            columns=["candidate_a", "candidate_b", "n_a", "n_b", "n_total", "n_eff"],
        )

    df = runoff_df.copy()
    df["effective_n"] = pd.to_numeric(df["effective_n"], errors="coerce")

    ca = df["candidate_a"].astype(str).to_numpy()
    cb = df["candidate_b"].astype(str).to_numpy()
    pair_tuples = [_canonical_pair(str(ca[i]), str(cb[i])) for i in range(len(ca))]
    df["pair_a"] = [p[0] for p in pair_tuples]
    df["pair_b"] = [p[1] for p in pair_tuples]

    grouped = df.groupby(["pair_a", "pair_b"], as_index=False).agg(
        n_total=("n_total", "sum"),
        n_eff=("effective_n", "sum"),
    )

    # Count wins for each candidate based on original ordering
    rows: list[dict[str, str | float]] = []
    for _, row in grouped.iterrows():
        a = str(row["pair_a"])
        b = str(row["pair_b"])

        mask = ((df["candidate_a"] == a) & (df["candidate_b"] == b)) | (
            (df["candidate_a"] == b) & (df["candidate_b"] == a)
        )
        subset = df.loc[mask]

        n_a_total = float(
            subset.loc[subset["candidate_a"] == a, "n_total"].sum()
            + subset.loc[subset["candidate_b"] == a, "n_total"].sum()
        )
        n_b_total = float(row["n_total"]) - n_a_total

        rows.append(
            {
                "candidate_a": a,
                "candidate_b": b,
                "n_a": n_a_total,
                "n_b": n_b_total,
                "n_total": float(row["n_total"]),
                "n_eff": float(row["n_eff"]),
            }
        )

    result = pd.DataFrame(
        rows, columns=["candidate_a", "candidate_b", "n_a", "n_b", "n_total", "n_eff"]
    )
    return result.sort_values("n_eff", ascending=False, ignore_index=True)


def fit_empirical_beta_per_pairing(
    pairing_stats: pd.DataFrame,
) -> dict[tuple[str, str], tuple[float, float]]:
    """Fit Beta(alpha, beta) per pairing using Jeffreys prior.

    For each pairing computes ``Beta(alpha = n_a + 0.5, beta = n_b + 0.5)``
    where ``n_a`` is the total weighted count for candidate_a and ``n_b`` for
    candidate_b.  The Jeffreys Beta(0.5, 0.5) prior provides a weakly
    informative regularisation that centres on 0.5 when no data is available.

    Args:
        pairing_stats: DataFrame from :func:`compute_per_pairing_stats` with
            columns ``candidate_a``, ``candidate_b``, ``n_a``, ``n_b``.

    Returns:
        Dict mapping ``(candidate_a, candidate_b)`` to ``(alpha, beta)``.

    """
    posteriors: dict[tuple[str, str], tuple[float, float]] = {}
    for _, row in pairing_stats.iterrows():
        a = str(row["candidate_a"])
        b = str(row["candidate_b"])
        key = _canonical_pair(a, b)
        n_a = float(row["n_a"])
        n_b = float(row["n_b"])
        posteriors[key] = (n_a + 0.5, n_b + 0.5)
    return posteriors


def sample_empirical_transfer(
    beta_posterior: tuple[float, float],
    candidate_a: str,  # noqa: ARG001
    candidate_b: str,  # noqa: ARG001
    n_draws: int = 1000,
) -> np.ndarray:
    """Sample from the empirical Beta posterior for a given pairing.

    Args:
        beta_posterior: ``(alpha, beta)`` tuple from
            :func:`fit_empirical_beta_per_pairing`.
        candidate_a: First candidate key (for interface compatibility;
            not used in sampling).
        candidate_b: Second candidate key (for interface compatibility;
            not used in sampling).
        n_draws: Number of draws to generate.

    Returns:
        1-D numpy array of shape ``(n_draws,)`` with values in ``[0, 1]``
        representing ``P(candidate_a wins)``.

    """
    alpha, beta = beta_posterior
    rng = np.random.default_rng()
    return rng.beta(alpha, beta, size=n_draws)


def cache_empirical_posterior(
    beta_posteriors: dict[tuple[str, str], tuple[float, float]],
    year: int = 2026,
    data_dir: Path | None = None,
) -> Path:
    """Save empirical Beta posteriors to a NetCDF cache file.

    Uses xarray.Dataset with variables ``alpha`` and ``beta`` keyed by
    concatenated candidate pair string.

    Args:
        beta_posteriors: Dict mapping ``(candidate_a, candidate_b)`` to
            ``(alpha, beta)``.
        year: Election year for filename.
        data_dir: Optional override for the project data directory.

    Returns:
        Path to the written cache file.

    Raises:
        ImportError: If xarray is not installed.

    """
    import xarray as xr  # noqa: PLC0415

    out_dir = resolve_data_dir(data_dir) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / _CACHE_FILENAME_TEMPLATE.format(year=year)

    pair_names: list[str] = []
    alphas: list[float] = []
    betas: list[float] = []
    for (ca, cb), (alpha, beta_) in sorted(beta_posteriors.items()):
        pair_names.append(f"{ca}__vs__{cb}")
        alphas.append(alpha)
        betas.append(beta_)

    ds = xr.Dataset(
        {
            "alpha": (["pairing"], alphas),
            "beta": (["pairing"], betas),
        },
        coords={"pairing": pair_names},
        attrs={
            "description": "Empirical Beta posterior parameters for 2026 runoff pairings",
            "year": str(year),
            "prior": "Jeffreys Beta(0.5, 0.5)",
        },
    )
    ds.to_netcdf(str(cache_path))  # pyright: ignore[reportUnknownMemberType]
    logger.info("Cached %d empirical posteriors to %s", len(pair_names), cache_path)
    return cache_path


def load_empirical_posterior(
    year: int = 2026,
    data_dir: Path | None = None,
) -> dict[tuple[str, str], tuple[float, float]] | None:
    """Load empirical Beta posteriors from a NetCDF cache file.

    Args:
        year: Election year for filename.
        data_dir: Optional override for the project data directory.

    Returns:
        Dict mapping ``(candidate_a, candidate_b)`` to ``(alpha, beta)``,
        or ``None`` if the cache file does not exist or cannot be read.

    """
    import xarray as xr  # noqa: PLC0415

    results_dir = resolve_data_dir(data_dir) / "results"
    cache_path = results_dir / _CACHE_FILENAME_TEMPLATE.format(year=year)

    if not cache_path.exists():
        logger.info("No empirical posterior cache at %s", cache_path)
        return None

    try:
        ds = xr.open_dataset(str(cache_path))  # pyright: ignore[reportUnknownMemberType]
    except (ValueError, FileNotFoundError, OSError) as exc:
        logger.warning("Failed to open empirical posterior cache: %s", exc)
        return None

    try:
        result: dict[tuple[str, str], tuple[float, float]] = {}
        for i, pair_name in enumerate(ds["pairing"].to_numpy()):
            name_str = str(pair_name)
            parts = name_str.split("__vs__")
            if len(parts) != 2:  # noqa: PLR2004
                logger.warning("Unexpected pairing name format: %r", name_str)
                continue
            a, b = parts
            alpha = float(ds["alpha"].to_numpy()[i])
            beta_val = float(ds["beta"].to_numpy()[i])
            result[(a, b)] = (alpha, beta_val)
        return result or None
    finally:
        ds.close()


def get_empirical_runoff_betas(
    year: int = 2026,
    *,
    force_rebuild: bool = False,
    data_dir: Path | None = None,
) -> dict[tuple[str, str], tuple[float, float]]:
    """Orchestrate empirical Beta posterior estimation for runoff pairings.

    Loads cached posteriors, or rebuilds from raw CNE data if not available
    or ``force_rebuild`` is True.

    Args:
        year: Election year (default 2026).
        force_rebuild: If ``True``, bypass cache and recompute from raw data.
        data_dir: Optional override for the project data directory.

    Returns:
        Dict mapping ``(candidate_a, candidate_b)`` to ``(alpha, beta)``.
        Empty dict when no runoff data is available.

    """
    if not force_rebuild:
        cached = load_empirical_posterior(year=year, data_dir=data_dir)
        if cached is not None:
            logger.info("Loaded %d cached empirical posteriors for %d", len(cached), year)
            return cached

    logger.info("Building empirical posteriors from raw CNE 2026 data...")
    runoff_df = load_runoff_raw(data_dir=data_dir)
    if runoff_df.empty:
        logger.warning("No runoff pairing data available; returning empty dict")
        return {}

    stats = compute_per_pairing_stats(runoff_df, year=year)
    if stats.empty:
        logger.warning("No valid pairing statistics; returning empty dict")
        return {}

    posteriors = fit_empirical_beta_per_pairing(stats)
    cache_empirical_posterior(posteriors, year=year, data_dir=data_dir)
    return posteriors
