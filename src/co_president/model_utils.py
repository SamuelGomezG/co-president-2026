"""SPEC-06: Shared utilities for model inference and posterior extraction.

Provides functions that abstract away differences between model types
(e.g., ``p_time`` vs ``p_natl``) so downstream code (runoff matrix,
accuracy reports) works with any model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from xarray import DataTree

__all__ = [
    "extract_election_day_shares",
    "get_election_day_array",
]


def get_election_day_array(idata: DataTree) -> np.ndarray:
    """Extract election-day vote shares from a model posterior.

    Handles both the round1 model (which stores ``p_time`` with a time
    dimension) and the municipal model (which stores ``p_natl`` without
    a time dimension).

    Args:
        idata: Posterior ``DataTree`` from any model.

    Returns:
        Array of shape ``(chain, draw, candidate)`` with election-day
        vote shares.

    Raises:
        KeyError: If neither ``p_time`` nor ``p_natl`` is found in the
            posterior.

    """
    if "p_time" in idata.posterior:
        return idata.posterior["p_time"][:, :, 0, :].to_numpy()
    if "p_natl" in idata.posterior:
        return idata.posterior["p_natl"].to_numpy()
    msg = "Posterior contains neither p_time nor p_natl"
    raise KeyError(msg)


def extract_election_day_shares(
    idata: DataTree,
    candidate_keys: list[str],
) -> dict[str, np.ndarray]:
    """Extract election-day posterior shares into per-candidate arrays.

    Args:
        idata: Posterior ``DataTree`` from a round model.
        candidate_keys: Candidate column keys in order matching the
            posterior's candidate dimension.

    Returns:
        Mapping of candidate key to 1-D numpy array of posterior shares
        (flattened over all chains and draws).

    """
    elec_shares = get_election_day_array(idata)  # (chain, draw, candidate)
    n_chains, n_draws, n_candidates = elec_shares.shape
    flat = elec_shares.reshape(n_chains * n_draws, n_candidates)
    return {key: flat[:, i] for i, key in enumerate(candidate_keys)}
