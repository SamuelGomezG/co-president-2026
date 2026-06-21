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

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from co_president.config import get_active_candidates
from co_president.data import CandidateResult, RoundResult
from co_president.model_runoff_simple import (
    build_runoff_simple_model,
    forecast_runoff_simple,
    sample_runoff,
)
from co_president.model_utils import get_election_day_array

if TYPE_CHECKING:
    from xarray import DataTree

    from co_president.config import ModelConfig

logger = logging.getLogger(__name__)


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
        mean_share_first: Predicted vote share of candidate_first.
        mean_share_second: Predicted vote share of candidate_second.
        idata: Posterior ``DataTree`` from the K=3 runoff model, or ``None``
            when the transfer heuristic was used instead.

    """

    candidate_first: str
    candidate_second: str
    prob_pairing: float
    prob_first_wins: float
    prob_second_wins: float
    mean_margin: float
    mean_share_first: float = float("nan")
    mean_share_second: float = float("nan")
    mean_share_rest: float = float("nan")
    idata: DataTree | None = None


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
_MIN_PAIRED_POLLS: int = 3


def _get_candidate_order(year: int = 2022) -> list[str]:
    """Return the canonical candidate ordering matching the Round 1 model."""
    return sorted(c.key for c in get_active_candidates(1, year=year))


def compute_top_two_probabilities(
    round1_idata: DataTree,
    candidates: list[str],
    candidate_keys: list[str] | None = None,
    year: int = 2022,
) -> dict[tuple[str, str], float]:
    """Compute probability of each ordered top-two pairing from posterior.

    Extracts election-day (``time=0``) vote shares from the Round 1 posterior
    and, for every posterior draw, records which two candidates come first and
    second. Frequencies across all draws give the pairing probabilities.

    Args:
        round1_idata: Posterior from :func:`sample_round1`.
        candidates: Candidate keys eligible for top-two consideration
            (e.g. excludes ``rest`` and ``blanco``).
        candidate_keys: Actual candidate ordering used to build the model.
            When ``None``, inferred from ``_get_candidate_order()``
            truncated to the posterior dimension (legacy fallback).
        year: Election year (default 2022).

    Returns:
        Dictionary mapping ``(first_place, second_place)`` to probability.

    Examples:
        >>> idata = sample_round1(model, config)
        >>> probs = compute_top_two_probabilities(idata, ["gustavo_petro", ...])
        >>> probs[("gustavo_petro", "rodolfo_hernandez")]
        0.92

    """
    election_day = get_election_day_array(round1_idata)  # (chain, draw, candidate)

    # The posterior dimension may be smaller than FIRST_ROUND_CANDIDATES
    # (e.g., rest excluded when missing from poll columns).  Use only the
    # number of columns present in the posterior array.
    n_posterior = election_day.shape[-1]
    if candidate_keys is not None:
        candidate_order = candidate_keys
    else:
        full_order = _get_candidate_order(year)
        candidate_order = full_order[:n_posterior] if len(full_order) >= n_posterior else full_order

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


_DEFAULT_TRANSFER: float = 0.5


def _transfer_rate_for_pairing(
    transfer_rates: dict[tuple[str, str], np.ndarray] | None,
    elim: str,
    first: str,
    second: str,
    draw_idx: int,
) -> float:
    """Return fraction of ``elim`` voters transferring to ``first``.

    Tries the direct key ``(elim, first)`` first.  Falls back to
    ``1 - transfer_rates[(elim, second)]`` when only the inverse key is
    available.  Returns :data:`_DEFAULT_TRANSFER` when neither key exists.

    The transfer rate arrays may be shorter than the number of posterior
    draws; uses modular indexing to cycle through them.

    Args:
        transfer_rates: Dict mapping ``(eliminated, target)`` to per-draw
            transfer rate arrays, or ``None`` to use default.
        elim: Eliminated candidate key.
        first: First-placed runoff candidate key.
        second: Second-placed runoff candidate key.
        draw_idx: Index into the per-draw arrays.

    Returns:
        Transfer fraction in ``[0, 1]``.

    """
    if transfer_rates is None:
        return _DEFAULT_TRANSFER

    direct = transfer_rates.get((elim, first))
    if direct is not None:
        return float(direct[draw_idx % len(direct)])

    inverse = transfer_rates.get((elim, second))
    if inverse is not None:
        return 1.0 - float(inverse[draw_idx % len(inverse)])

    return _DEFAULT_TRANSFER


def _compute_transfer_shares(
    election_day: np.ndarray,
    all_candidate_keys: list[str],
    first: str,
    second: str,
    transfer_rates: dict[tuple[str, str], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized runoff shares for each posterior draw.

    Args:
        election_day: NumPy array of shape ``(chain, draw, n_candidates)``
            with election-day vote shares (summing to 1 along axis=-1).
        all_candidate_keys: Full candidate ordering matching the last axis.
        first: Key of the first-placed runoff candidate.
        second: Key of the second-placed runoff candidate.
        transfer_rates: Optional dict of per-draw transfer rate arrays
            from :func:`~co_president.model_transfer.sample_transfer_rates`.

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

            transfer_to_first = _transfer_rate_for_pairing(
                transfer_rates,
                elim,
                first,
                second,
                draw_idx,
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
    transfer_rates: dict[tuple[str, str], np.ndarray] | None = None,
) -> tuple[float, float, float, float]:
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
        transfer_rates: Optional dict of per-draw transfer rate arrays.

    Returns:
        Tuple of ``(prob_first_wins, mean_margin, mean_share_first,
        mean_share_second)``.

    """
    shares_first, shares_second = _compute_transfer_shares(
        election_day,
        all_candidate_keys,
        first,
        second,
        transfer_rates,
    )
    prob_first_wins = float((shares_first > shares_second).mean())
    mean_margin = float((shares_first - shares_second).mean())
    mean_share_first = float(shares_first.mean())
    mean_share_second = float(shares_second.mean())
    return prob_first_wins, mean_margin, mean_share_first, mean_share_second


def _filter_polls_for_pairing(
    polls: pd.DataFrame,
    first: str,
    second: str,
) -> pd.DataFrame | None:
    """Filter ``round2_polls`` to rows relevant for a ``(first, second)`` pairing.

    Returns ``None`` if fewer than 3 rows remain after removing rows with NaN
    candidate shares, or if the required columns are missing.

    Args:
        polls: Clean Round 2 poll DataFrame.
        first: Key of the first-placed runoff candidate.
        second: Key of the second-placed runoff candidate.

    Returns:
        Filtered DataFrame with at least 3 rows, or ``None``.

    """
    required = {"fecha", "encuestadora", "muestra", first, second}
    missing = required - set(polls.columns)
    if missing:
        return None

    filtered = polls.dropna(subset=[first, second])
    if len(filtered) < _MIN_PAIRED_POLLS:
        return None

    return filtered


_HISTORICAL_RUNOFF_REST_RATE = 0.023


def _run_runoff_model_for_pairing(  # noqa: PLR0913
    polls: pd.DataFrame,
    round1_result: RoundResult,
    round1_idata: DataTree,
    config: ModelConfig,
    pairing: tuple[str, str],
    digital_signals: pd.DataFrame,
    round2_result: RoundResult | None = None,
    features: pd.DataFrame | None = None,
    transfer_rates: dict[tuple[str, str], np.ndarray] | None = None,
    all_candidate_keys: list[str] | None = None,
    year: int = 2022,
) -> tuple[float, float, float, float, float, DataTree | None]:
    """Build, sample, and forecast a K=3 runoff model for a given pairing.

    Constructs a synthetic :class:`RoundResult` that positions the pairing's
    candidates as the top two so that
    :func:`~co_president.model_runoff_simple.build_runoff_simple_model`
    builds the model graph targeting this specific pairing.

    Args:
        polls: Filtered head-to-head poll DataFrame (``>=3`` rows).
        round1_result: Round 1 election result (used for candidate shares).
        round1_idata: Round 1 posterior for the informed prior.
        config: Model hyperparameters.
        pairing: Tuple of ``(first, second)`` candidate keys.
        round2_result: Actual Round 2 election result (for election
            likelihood, goodness-of-fit backtest).
        features: Municipal feature matrix for internet rate weighting.
        digital_signals: Google Trends data for the runoff model's
            poll likelihood.
        transfer_rates: Optional per-draw transfer rate dict from
            :func:`~co_president.model_transfer.sample_transfer_rates`.
            When provided, the prior mean uses transfer-implied runoff
            shares instead of the Round 1 vote shares.
        all_candidate_keys: Full candidate ordering matching the
            Round 1 posterior.  Required when ``transfer_rates`` is provided.
        year: Election year (default 2022).

    Returns:
        Tuple of ``(prob_first_wins, mean_margin, share_first, share_second,
        share_rest, idata)`` where *idata* is the posterior ``DataTree`` from
        the K=3 runoff model (or ``None`` if no model was sampled).

    """
    first, second = pairing

    if transfer_rates is not None and all_candidate_keys is not None:
        election_day = get_election_day_array(round1_idata)
        shares_first, shares_second = _compute_transfer_shares(
            election_day,
            all_candidate_keys,
            first,
            second,
            transfer_rates,
        )
        first_share = float(shares_first.mean())
        second_share = float(shares_second.mean())
    else:
        first_share = round1_result.get_share(first)
        second_share = round1_result.get_share(second)

    synthetic_candidates = (
        CandidateResult(first, int(first_share * 100_000), first_share),
        CandidateResult(second, int(second_share * 100_000), second_share),
    )
    synthetic_result = RoundResult(
        round_number=1,
        date=round1_result.date,
        total_valid_votes=int((first_share + second_share) * 100_000),
        total_votes_incl_blank=int((first_share + second_share) * 100_000),
        registered_voters=round1_result.registered_voters,
        polling_stations=round1_result.polling_stations,
        candidates=synthetic_candidates,
        blank_votes=0,
        null_votes=0,
        unmarked_votes=0,
    )

    model = build_runoff_simple_model(
        polls,
        synthetic_result,
        round1_idata,
        config,
        features=features,
        digital_signals=digital_signals,
        round2_result=round2_result,
        year=year,
    )
    idata = sample_runoff(model, config)
    forecast = forecast_runoff_simple(idata, first, second)

    return (
        forecast.prob_a_wins,
        forecast.mean_margin,
        forecast.mean_share_a,
        forecast.mean_share_b,
        forecast.mean_share_rest,
        idata,
    )


def estimate_runoff_matrix(  # noqa: PLR0913
    round1_idata: DataTree,
    results: tuple[RoundResult, RoundResult],
    round2_polls: pd.DataFrame | None,
    config: ModelConfig,
    digital_signals: pd.DataFrame,
    candidate_keys: list[str] | None = None,
    features: pd.DataFrame | None = None,
    year: int = 2022,
) -> RunoffMatrix:
    """Compute a full probabilistic runoff matrix.

    Uses the Round 1 posterior to identify plausible pairings, then for each
    pairing estimates the runoff outcome using one of two paths:

    1. **Model path**: If ``round2_polls`` contains 3+ head-to-head polls for
       the candidate pairing, runs the K=3
       :func:`~co_president.model_runoff_simple.build_runoff_simple_model`
       and extracts win probabilities from the posterior.
    2. **Heuristic path**: Otherwise, uses the transfer heuristic that
       redistributes eliminated candidates' votes using data-driven transfer
       rates from :func:`~co_president.model_transfer.sample_transfer_rates`.

    Args:
        round1_idata: Posterior from :func:`~co_president.model_round1.sample_round1`.
        results: Canonical election results ``(round1, round2)``.
        round2_polls: Clean Round 2 poll DataFrame, or ``None`` to use the
            transfer heuristic for all pairings.
        config: Model hyperparameters.
        candidate_keys: Actual candidate ordering used to build the model.
            When ``None``, inferred from ``_get_candidate_order()`` truncated
            to the posterior dimension (legacy fallback).
        features: Municipal features DataFrame for on-demand transfer model
            training.  When ``None``, uses the calibrated Dirichlet prior.
        digital_signals: Google Trends (or other digital signal) data for
            the runoff K=3 poll likelihood.  Passed through to
            :func:`build_runoff_simple_model`.
        year: Election year (default 2022).

    Returns:
        :class:`RunoffMatrix` containing all plausible pairings.

    Examples:
        >>> matrix = estimate_runoff_matrix(idata, (r1, r2), None, config,
        ...     digital_signals=pd.DataFrame())
        >>> matrix.ordered_by_likelihood[0]
        ('gustavo_petro', 'rodolfo_hernandez')

    """
    from co_president.model_transfer import sample_transfer_rates  # noqa: PLC0415

    # Determine candidate ordering
    election_day = get_election_day_array(round1_idata)  # (chain, draw, candidate)
    n_posterior = election_day.shape[-1]

    if candidate_keys is not None:
        all_candidate_keys = candidate_keys
    else:
        all_candidate_keys = _get_candidate_order(year)[:n_posterior]

    candidates = [k for k in all_candidate_keys if k not in ("rest", "blanco")]

    top_two_probs = compute_top_two_probabilities(
        round1_idata, candidates, candidate_keys, year=year
    )

    round1_result, _round2_result = results

    # Load data-driven transfer rates (falls back to historical Dirichlet
    # prior when no cached PyMC posterior is available).  Pass features
    # for on-demand training when available.
    transfer_rates = sample_transfer_rates(features=features, config=config)

    pairings: list[PairingForecast] = []
    for (first, second), prob in sorted(top_two_probs.items(), key=lambda x: x[1], reverse=True):
        if prob < _MIN_PAIRING_PROB:
            break

        # If head-to-head polls are available for this pairing, use the
        # K=3 runoff model instead of the transfer heuristic.
        if round2_polls is not None:
            pairing_polls = _filter_polls_for_pairing(round2_polls, first, second)
        else:
            pairing_polls = None

        idata_runoff: DataTree | None = None
        if pairing_polls is not None:
            # Use digital signals in the runoff model only when explicitly
            # enabled via config (off by default — digital signals were
            # systematically biased toward Rodolfo in 2022).
            _ds_runoff = digital_signals if config.use_digital_signals_runoff else pd.DataFrame()
            prob_first_wins, mean_margin, share_first, share_second, share_rest, idata_runoff = (
                _run_runoff_model_for_pairing(
                    pairing_polls,
                    round1_result,
                    round1_idata,
                    config,
                    (first, second),
                    round2_result=None,
                    features=features,
                    digital_signals=_ds_runoff,
                    transfer_rates=transfer_rates,
                    all_candidate_keys=all_candidate_keys,
                    year=year,
                )
            )
        else:
            prob_first_wins, mean_margin, share_first, share_second = _compute_transfer_outcome(
                election_day,
                all_candidate_keys,
                first,
                second,
                transfer_rates,
            )
            share_rest = float("nan")

        pairings.append(
            PairingForecast(
                candidate_first=first,
                candidate_second=second,
                prob_pairing=prob,
                prob_first_wins=prob_first_wins,
                prob_second_wins=1.0 - prob_first_wins,
                mean_margin=mean_margin,
                mean_share_first=share_first,
                mean_share_second=share_second,
                mean_share_rest=share_rest,
                idata=idata_runoff,
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
        >>> matrix = estimate_runoff_matrix(idata, results, None, config,
        ...     digital_signals=pd.DataFrame())
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
