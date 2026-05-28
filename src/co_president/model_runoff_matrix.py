"""SPEC-08: Full probabilistic pairing matrix for runoff.

From the Round 1 posterior, computes the probability of every possible runoff
pairing and, for each pairing, estimates who would win using either head-to-head
poll data or a transfer heuristic based on coalition alignment.

Public functions
----------------
:func:`compute_top_two_probabilities` — ordered top-two pairing probabilities.
:func:`estimate_runoff_matrix` — full matrix with transfer-heuristic outcomes.
:func:`overall_win_probability` — aggregate each candidate's presidency chance.

Public dataclasses
------------------
:class:`PairingForecast` — forecast for one specific runoff pairing.
:class:`RunoffMatrix` — collection of all plausible pairings.
"""

# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportMissingTypeArgument=false, reportIndexIssue=false, reportOperatorIssue=false
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from co_president.config import (
    FIRST_ROUND_CANDIDATES,
    TRANSFER_BLANCO_SPLIT,
    TRANSFER_FAJARDO_HERNANDEZ,
    TRANSFER_FAJARDO_PETRO,
    TRANSFER_GUTIERREZ_HERNANDEZ,
    TRANSFER_GUTIERREZ_PETRO,
)

if TYPE_CHECKING:
    import arviz as az  # type: ignore[reportMissingTypeStubs]
    import pandas as pd

    from co_president.config import ModelConfig
    from co_president.data_results import RoundResult


@dataclass(frozen=True)
class PairingForecast:
    """Forecast for a specific runoff pairing.

    Attributes:
        candidate_first: Key of the candidate who finished first in Round 1.
        candidate_second: Key of the candidate who finished second in Round 1.
        prob_pairing: Probability that this exact pairing occurs.
        prob_first_wins: Probability the first-placed candidate wins the runoff.
        prob_second_wins: Probability the second-placed candidate wins the runoff.
        mean_margin: Expected vote margin (first - second) in the runoff.

    """

    candidate_first: str
    candidate_second: str
    prob_pairing: float
    prob_first_wins: float
    prob_second_wins: float
    mean_margin: float


@dataclass(frozen=True)
class RunoffMatrix:
    """Complete probabilistic runoff pairing matrix.

    Attributes:
        pairings: Tuple of :class:`PairingForecast` for all plausible pairings.
        prob_runoff: Total probability of a runoff
            (sum of all pairing probabilities).
        ordered_by_likelihood: Pairing tuples sorted from most to least likely.

    """

    pairings: tuple[PairingForecast, ...]
    prob_runoff: float
    ordered_by_likelihood: tuple[tuple[str, str], ...]


_MIN_PAIRING_PROB: float = 0.01


def _get_candidate_order() -> list[str]:
    """Return the canonical candidate ordering matching the Round 1 model."""
    return sorted(FIRST_ROUND_CANDIDATES.keys())


def compute_top_two_probabilities(
    round1_idata: az.InferenceData,
    candidates: list[str],
) -> dict[tuple[str, str], float]:
    """Compute probability of each ordered top-two pairing from posterior.

    Extracts election-day (``time=0``) vote shares from the Round 1 posterior
    and, for every posterior draw, records which two candidates come first and
    second. Frequencies across all draws give the pairing probabilities.

    Args:
        round1_idata: Posterior from :func:`sample_round1`.
        candidates: Candidate keys eligible for top-two consideration
            (e.g. excludes ``rest`` and ``blanco``).

    Returns:
        Dictionary mapping ``(first_place, second_place)`` to probability.

    Examples:
        >>> idata = sample_round1(model, config)
        >>> probs = compute_top_two_probabilities(idata, ["gustavo_petro", ...])
        >>> probs[("gustavo_petro", "rodolfo_hernandez")]
        0.92

    """
    p_time = round1_idata.posterior["p_time"]  # (chain, draw, time, candidate)
    election_day = p_time[:, :, 0, :].to_numpy()  # (chain, draw, candidate)

    candidate_order = _get_candidate_order()
    candidate_idx: dict[str, int] = {k: i for i, k in enumerate(candidate_order)}

    active_indices = [candidate_idx[c] for c in candidates]
    active_shares = election_day[:, :, active_indices]  # (chain, draw, n_active)

    flat_shares = active_shares.reshape(-1, len(candidates))

    # Vectorised ranking: argsort descending across all draws at once
    sorted_idx = np.argsort(-flat_shares, axis=1)  # (draw, n_candidates)
    first_idx = sorted_idx[:, 0]
    second_idx = sorted_idx[:, 1]

    # Encode each (first, second) pair as a single integer
    pair_ids = first_idx * len(candidates) + second_idx
    unique_ids, counts = np.unique(pair_ids, return_counts=True)

    total_draws = flat_shares.shape[0]
    return {
        (candidates[int(uid // len(candidates))], candidates[int(uid % len(candidates))]): count
        / total_draws
        for uid, count in zip(unique_ids, counts, strict=True)
    }


_TRANSFER_MAP: dict[str, dict[str, float]] = {
    "sergio_fajardo": {
        "gustavo_petro": TRANSFER_FAJARDO_PETRO,
        "rodolfo_hernandez": TRANSFER_FAJARDO_HERNANDEZ,
    },
    "federico_gutierrez": {
        "rodolfo_hernandez": TRANSFER_GUTIERREZ_HERNANDEZ,
        "gustavo_petro": TRANSFER_GUTIERREZ_PETRO,
    },
}


def _get_transfer_fraction(
    eliminated: str,
    target: str,
    share_target: float,
    share_other: float,
) -> float:
    """Return the fraction of an eliminated candidate's votes for *target*.

    Uses the transfer-heuristic constants from ``config``. For candidates
    without specific constants (e.g. ``ingrid_betancourt``, ``rest``), votes
    are split proportionally to the target's and other's base shares.

    Args:
        eliminated: Key of the eliminated candidate.
        target: Key of the candidate receiving votes.
        share_target: Base vote share of the target candidate in this draw.
        share_other: Base vote share of the other runoff candidate.

    Returns:
        Fraction in ``[0, 1]``.

    """
    if eliminated == "blanco":
        return TRANSFER_BLANCO_SPLIT

    elim_map = _TRANSFER_MAP.get(eliminated)
    if elim_map is not None:
        fraction = elim_map.get(target)
        if fraction is not None:
            return fraction

    # For ingrid_betancourt, rest, or pairings not covered by the
    # constants above: proportional split based on base shares.
    denom = share_target + share_other
    if denom > 0:
        return share_target / denom
    return 0.5


def _compute_transfer_shares(
    election_day: np.ndarray,
    all_candidate_keys: list[str],
    first: str,
    second: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized runoff shares for each posterior draw.

    Args:
        election_day: NumPy array of shape ``(chain, draw, n_candidates)``
            with election-day vote shares (summing to 1 along axis=-1).
        all_candidate_keys: Full candidate ordering matching the last axis.
        first: Key of the first-placed runoff candidate.
        second: Key of the second-placed runoff candidate.

    Returns:
        Tuple of arrays ``(shares_first, shares_second)`` with shape
        ``(n_draws,)``.

    """
    key_to_idx = {k: i for i, k in enumerate(all_candidate_keys)}
    first_idx = key_to_idx[first]
    second_idx = key_to_idx[second]

    eliminated = [k for k in all_candidate_keys if k not in (first, second)]

    flat_shares = election_day.reshape(-1, len(all_candidate_keys))
    n_total = flat_shares.shape[0]
    shares_first = np.empty(n_total, dtype=float)
    shares_second = np.empty(n_total, dtype=float)

    for draw_idx in range(n_total):
        share_first = float(flat_shares[draw_idx, first_idx])
        share_second = float(flat_shares[draw_idx, second_idx])

        for elim in eliminated:
            elim_share = float(flat_shares[draw_idx, key_to_idx[elim]])
            if elim_share <= 0:
                continue

            transfer_to_first = _get_transfer_fraction(
                elim,
                first,
                share_first,
                share_second,
            )
            share_first += elim_share * transfer_to_first
            share_second += elim_share * (1.0 - transfer_to_first)

        total = share_first + share_second
        if total == 0:
            total = 1.0  # Both shares are 0; keep them at 0
        share_first /= total
        share_second /= total

        shares_first[draw_idx] = share_first
        shares_second[draw_idx] = share_second

    return shares_first, shares_second


def _compute_transfer_outcome(
    election_day: np.ndarray,
    all_candidate_keys: list[str],
    first: str,
    second: str,
) -> tuple[float, float]:
    """Compute runoff outcome for a pairing using the transfer heuristic.

    Iterates over all posterior draws, redistributes eliminated candidates'
    vote shares to the two runoff finalists, and returns the probability that
    the first-placed candidate wins and the expected margin.

    Args:
        election_day: NumPy array of shape ``(chain, draw, n_candidates)``
            with election-day vote shares (summing to 1 along axis=-1).
        all_candidate_keys: Full candidate ordering matching the last axis.
        first: Key of the first-placed runoff candidate.
        second: Key of the second-placed runoff candidate.

    Returns:
        Tuple of ``(prob_first_wins, mean_margin)``.

    """
    shares_first, shares_second = _compute_transfer_shares(
        election_day,
        all_candidate_keys,
        first,
        second,
    )
    prob_first_wins = float((shares_first > shares_second).mean())
    mean_margin = float((shares_first - shares_second).mean())
    return prob_first_wins, mean_margin


def estimate_runoff_matrix(
    round1_idata: az.InferenceData,
    results: tuple[RoundResult, RoundResult],  # noqa: ARG001
    round2_polls: pd.DataFrame | None,  # noqa: ARG001
    config: ModelConfig,  # noqa: ARG001
) -> RunoffMatrix:
    """Compute a full probabilistic runoff matrix.

    Uses the Round 1 posterior to identify plausible pairings, then for each
    pairing estimates the runoff outcome via a **transfer heuristic** that
    redistributes eliminated candidates' votes based on coalition alignment.

    TODO(SPEC-08): When ``round2_polls`` contains head-to-head polls for the
    pairing, use :func:`~co_president.model_runoff_simple.build_runoff_simple_model`
    instead of the heuristic. The parameters ``results``, ``round2_polls``, and
    ``config`` are accepted now to keep the API stable for that future path.

    Args:
        round1_idata: Posterior from :func:`~co_president.model_round1.sample_round1`.
        results: Canonical election results ``(round1, round2)``.
        round2_polls: Clean Round 2 poll DataFrame, or ``None`` to use the
            transfer heuristic for all pairings.
        config: Model hyperparameters.

    Returns:
        :class:`RunoffMatrix` containing all plausible pairings.

    Examples:
        >>> matrix = estimate_runoff_matrix(idata, (r1, r2), None, config)
        >>> matrix.ordered_by_likelihood[0]
        ('gustavo_petro', 'rodolfo_hernandez')

    """
    candidates = [k for k in _get_candidate_order() if k not in ("rest", "blanco")]

    top_two_probs = compute_top_two_probabilities(round1_idata, candidates)

    p_time = round1_idata.posterior["p_time"].to_numpy()
    election_day = p_time[:, :, 0, :]

    all_candidate_keys = _get_candidate_order()

    pairings: list[PairingForecast] = []
    for (first, second), prob in sorted(top_two_probs.items(), key=lambda x: x[1], reverse=True):
        if prob < _MIN_PAIRING_PROB:
            break

        # Use the transfer heuristic when no head-to-head polls are available.
        prob_first_wins, mean_margin = _compute_transfer_outcome(
            election_day,
            all_candidate_keys,
            first,
            second,
        )

        pairings.append(
            PairingForecast(
                candidate_first=first,
                candidate_second=second,
                prob_pairing=prob,
                prob_first_wins=prob_first_wins,
                prob_second_wins=1.0 - prob_first_wins,
                mean_margin=mean_margin,
            ),
        )

    total_prob = sum(p.prob_pairing for p in pairings)
    ordered = tuple(
        (p.candidate_first, p.candidate_second)
        for p in sorted(pairings, key=lambda x: x.prob_pairing, reverse=True)
    )

    return RunoffMatrix(
        pairings=tuple(pairings),
        prob_runoff=total_prob,
        ordered_by_likelihood=ordered,
    )


def overall_win_probability(
    matrix: RunoffMatrix,
    prob_win_outright: dict[str, float],
) -> dict[str, float]:
    """Compute each candidate's total probability of becoming president.

    The total probability is the sum of the candidate's outright Round 1 win
    probability and, for every pairing containing the candidate, the product
    of the pairing probability and the candidate's win probability within
    that pairing.

    Args:
        matrix: :class:`RunoffMatrix` from :func:`estimate_runoff_matrix`.
        prob_win_outright: Per-candidate probability of winning outright in
            Round 1 (from :class:`~co_president.model_round1.Round1Forecast`).

    Returns:
        Mapping of candidate key to total presidency probability.

    Examples:
        >>> idata = sample_round1(model, config)
        >>> forecast = forecast_round1(idata, candidates)
        >>> matrix = estimate_runoff_matrix(idata, results, None, config)
        >>> probs = overall_win_probability(matrix, {
        ...     c.candidate_key: c.prob_win_outright for c in forecast.candidates
        ... })

    """
    result: dict[str, float] = dict(prob_win_outright)

    for pairing in matrix.pairings:
        first_contrib = pairing.prob_pairing * pairing.prob_first_wins
        result[pairing.candidate_first] = result.get(pairing.candidate_first, 0.0) + first_contrib

        second_contrib = pairing.prob_pairing * pairing.prob_second_wins
        result[pairing.candidate_second] = (
            result.get(pairing.candidate_second, 0.0) + second_contrib
        )

    return result
