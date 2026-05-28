"""SPEC-02: Configuration & Constants.

Candidate maps, dates, pollster ratings, and model hyperparameters. All
downstream modules import from this module rather than hardcoding values.

Transfer-heuristic constants for the runoff vote flow
-----------------------------------------------------
Aggregate analysis of 8 pollsters' round-1 to round-2 deltas shows:
  ~73% of eliminated-candidate votes flow to Hernandez
  ~27% flow to Petro

Per-candidate constants were calibrated to match this aggregate split.
Each transfer row sums to 1.0 (e.g. Fajardo's voters split between
Petro and Hernandez). The derivation is documented in
``notebooks/derive_transfer_constants.py``.

Ecological inference limitation: per-candidate transfer rates cannot be
identified from aggregate data alone. The constants below are heuristics,
not empirically identified parameters.
"""

from __future__ import annotations

import contextlib
import csv
from dataclasses import dataclass
from datetime import date
import functools
import math
from pathlib import Path
import statistics
from typing import Literal
import unicodedata

from co_president.paths import resolve_data_dir

__all__ = [
    "COALITION_TO_CANDIDATE",
    "COMPUTED_CONSULTATION_PRIOR_STRENGTHS",
    "CONSULTATION_DATE",
    "CONSULTATION_KEY_MAP",
    "CONSULTATION_VOTES",
    "ELECTION_DATE_ROUND1",
    "ELECTION_DATE_ROUND2",
    "FIRST_ROUND_CANDIDATES",
    "POLLSTER_RATINGS",
    "TRANSFER_BLANCO_SPLIT",
    "TRANSFER_FAJARDO_HERNANDEZ",
    "TRANSFER_FAJARDO_PETRO",
    "TRANSFER_GUTIERREZ_HERNANDEZ",
    "TRANSFER_GUTIERREZ_PETRO",
    "Candidate",
    "ModelConfig",
    "compute_consultation_prior_strength",
    "consultation_log_share_prior",
    "get_active_candidates",
    "get_candidate_column_map",
    "get_computed_consultation_prior_strengths",
    "pollster_weight_formula",
    "validate_consultation_prior_means",
]


@dataclass(frozen=True)
class Candidate:
    r"""A presidential candidate across the election cycle.

    Attributes:
        key: Internal key matching column names in polls.
        display_name: Human-readable name.
        coalition: Coalition the candidate runs under (e.g. "Pacto Histórico").
        first_round: Whether the candidate ran in the first round.
        runoff: Whether the candidate made the runoff.

    """

    key: str
    display_name: str
    coalition: str | None
    first_round: bool = True
    runoff: bool = False


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for the Bayesian election model.

    Attributes:
        random_walk_sigma_prior: HalfNormal sigma for RW step size.
        concentration_poll_prior_mean: Gamma mean for poll concentration.
        concentration_election_prior_mean: Gamma mean for election concentration.
        house_effect_sigma_prior: HalfNormal sigma for house effects.
        mcmc_draws: Number of posterior draws per chain.
        mcmc_tune: Number of tuning/warmup draws per chain.
        mcmc_chains: Number of chains.
        mcmc_cores: Number of CPU cores for parallel chains.
        target_accept: NUTS target acceptance rate.
        seed: RNG seed for reproducibility.
        time_decay_half_life_days: Days for poll weight to halve.
        consultation_prior_strength: Sigma for Normal prior on theta[T-1].
            Can be overridden by candidate-specific values in
            ``COMPUTED_CONSULTATION_PRIOR_STRENGTHS``.
        consultation_prior_strength_override: Candidate-specific override of
            ``consultation_prior_strength``. Keys are candidate keys, values
            are standard-deviation strengths. When provided, these take
            precedence over ``COMPUTED_CONSULTATION_PRIOR_STRENGTHS``.

    """

    random_walk_sigma_prior: float = 0.5
    concentration_poll_prior_mean: float = 5.0
    concentration_election_prior_mean: float = 50.0
    house_effect_sigma_prior: float = 1.0
    mcmc_draws: int = 4000
    mcmc_tune: int = 1000
    mcmc_chains: int = 4
    mcmc_cores: int = 4
    target_accept: float = 0.95
    seed: int = 332211
    time_decay_half_life_days: float = 30.0
    consultation_prior_strength: float = 0.5
    consultation_prior_strength_override: dict[str, float] | None = None


FIRST_ROUND_CANDIDATES: dict[str, Candidate] = {
    "gustavo_petro": Candidate(
        key="gustavo_petro",
        display_name="Gustavo Petro",
        coalition="Pacto Histórico",
        first_round=True,
        runoff=True,
    ),
    "federico_gutierrez": Candidate(
        key="federico_gutierrez",
        display_name="Federico Gutiérrez",
        coalition="Equipo por Colombia",
        first_round=True,
        runoff=False,
    ),
    "rodolfo_hernandez": Candidate(
        key="rodolfo_hernandez",
        display_name="Rodolfo Hernández",
        coalition="Liga de Gobernantes Anticorrupción",
        first_round=True,
        runoff=True,
    ),
    "sergio_fajardo": Candidate(
        key="sergio_fajardo",
        display_name="Sergio Fajardo",
        coalition="Centro Esperanza",
        first_round=True,
        runoff=False,
    ),
    "ingrid_betancourt": Candidate(
        key="ingrid_betancourt",
        display_name="Ingrid Betancourt",
        coalition=None,  # Independent; initially in Centro Esperanza, withdrew after consultation
        first_round=True,
        runoff=False,
    ),
    "rest": Candidate(
        key="rest",
        display_name="Otros",
        coalition=None,
        first_round=True,
        runoff=True,  # Aggregates blanco + minor candidates in runoff (SPEC-07 K=3)
    ),
    "blanco": Candidate(
        key="blanco",
        display_name="Voto en Blanco",
        coalition=None,
        first_round=True,
        runoff=False,  # Merged with rest for runoff; not a separate entity in K=3
    ),
}

# "nulos" and "no_marcados" are mapped for the data loader (SPEC-03) to
# recognise but are NOT candidates in FIRST_ROUND_CANDIDATES — they are
# tracked separately and excluded from valid vote share per the spec.
# The data loader must filter them before candidate lookups.
COALITION_TO_CANDIDATE: dict[str, str] = {
    "COALICION PACTO HISTORICO": "gustavo_petro",
    "COALICION EQUIPO POR COLOMBIA": "federico_gutierrez",
    "LIGA DE GOBERNANTES ANTICORRUPCION": "rodolfo_hernandez",
    "COALICION CENTRO ESPERANZA": "sergio_fajardo",
    "VOTOS EN BLANCO": "blanco",
    "VOTOS NULOS": "nulos",
    "VOTOS NO MARCADOS": "no_marcados",
    "COLOMBIA JUSTA LIBRES": "rest",
    "PARTIDO MOVIMIENTO DE SALVACION NACIONAL": "rest",
    "COLOMBIA PIENSA EN GRANDE": "rest",
    "PARTIDO VERDE OXIGENO": "rest",
}

ELECTION_DATE_ROUND1: date = date(2022, 5, 29)
ELECTION_DATE_ROUND2: date = date(2022, 6, 19)
CONSULTATION_DATE: date = date(2022, 3, 13)

_ROUND_FIRST = 1
_ROUND_SECOND = 2

CONSULTATION_VOTES: dict[str, int] = {
    "gustavo_petro": 5_500_000,
    "federico_gutierrez": 3_800_000,
    "sergio_fajardo": 1_800_000,
    "rodolfo_hernandez": 0,
    "ingrid_betancourt": 0,
}

CONSULTATION_KEY_MAP: dict[str, str] = {
    "Gustavo Petro": "gustavo_petro",
    "Federico Gutierrez": "federico_gutierrez",
    "Sergio Fajardo": "sergio_fajardo",
    "Ingrid Betancourt": "ingrid_betancourt",
    "Rodolfo Hernández": "rodolfo_hernandez",
}


def _normalize_name(name: str) -> str:
    """Normalize name for case-insensitive, accent-insensitive comparison."""
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii").casefold()


_NORMALIZED_CONSULTATION_MAP: dict[str, str] = {
    _normalize_name(k): v for k, v in CONSULTATION_KEY_MAP.items()
}


# Note: Values are approximate (±200K) and serve as rough proxies for
# coalition base support. The model can deviate if poll data disagrees.


def consultation_log_share_prior() -> dict[str, float]:
    """Compute log-scale prior means from consultation vote shares.

    For candidates with non-zero consultation votes, returns
    ``log(votes / total_votes)``. For candidates with zero votes, returns
    ``log(min_nonzero_share / 2)`` as a small placeholder.
    If all candidates have zero consultation votes, returns -1.0 for every
    candidate as a uniform fallback.

    These values are used as the means of the Normal prior on ``theta[T-1]``
    (the earliest time point in the reverse-time random walk). Setting
    ``theta`` to the log of consultation shares centers the softmax-transformed
    probabilities around those shares.

    Returns:
        Mapping of candidate key to log-share value (always finite).

    Examples:
        >>> prior = consultation_log_share_prior()
        >>> len(prior)
        5
        >>> prior["gustavo_petro"] > prior["rodolfo_hernandez"]
        True

    """
    nonzero = {k: v for k, v in CONSULTATION_VOTES.items() if v > 0}
    total = sum(nonzero.values())
    if total == 0:
        return dict.fromkeys(CONSULTATION_VOTES, -1.0)
    min_share = min(v / total for v in nonzero.values())
    shares: dict[str, float] = {}
    for k, v in CONSULTATION_VOTES.items():
        if v > 0:
            shares[k] = math.log(v / total)
        else:
            shares[k] = math.log(min_share / 2.0)
    return shares


def compute_consultation_prior_strength() -> dict[str, float]:
    """Compute candidate-specific prior strengths from consultation polls.

    Reads ``consultas.csv`` from the project data directory, groups poll
    results by candidate, and computes the standard deviation of ``int_voto``
    values (expressed as proportions in [0,1]). Enforces a minimum standard
    deviation floor of 0.10. For candidates without consultation data, uses a
    fallback of the mean strength * 1.5.

    The result is memoized via :func:`get_computed_consultation_prior_strengths`
    so repeated calls incur no I/O after the first.

    Returns:
        Mapping of candidate key to prior standard deviation.

    Raises:
        ValueError: If any key in ``CONSULTATION_KEY_MAP`` produced zero rows,
            indicating a name mismatch between the CSV and the key map.

    Examples:
        >>> strengths = compute_consultation_prior_strength()
        >>> isinstance(strengths, dict)
        True
        >>> all(v >= 0.10 for v in strengths.values())
        True

    """
    data_dir = resolve_data_dir(None)
    csv_path = data_dir / "2022-polls" / "consultas.csv"
    strengths: dict[str, list[float]] = {}

    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = _normalize_name(row["candidato"])
            key = _NORMALIZED_CONSULTATION_MAP.get(name)
            if key is not None:
                strengths.setdefault(key, []).append(float(row["int_voto"]) / 100.0)

    results: dict[str, float] = {}
    for key, values in strengths.items():
        if len(values) > 1:
            results[key] = max(0.10, statistics.stdev(values))
        else:
            results[key] = 0.10

    # Fallback for candidates with zero consultation votes (independents)
    if results:
        mean_strength = statistics.mean(results.values())
        for key in CONSULTATION_VOTES:
            if key not in results:
                results[key] = mean_strength * 1.5

    # Candidates with non-zero consultation votes are expected in the CSV;
    # those with zero votes (independents) are not.
    expected = [k for k, v in CONSULTATION_VOTES.items() if v > 0]
    missing = [k for k in expected if k not in strengths]
    if missing:
        msg = f"No consultation data found for candidates: {missing}"
        raise ValueError(msg)

    return results


def validate_consultation_prior_means(
    means: dict[str, float],
    consultas_path: str | None = None,
) -> None:
    """Validate prior means against ``consultas.csv`` polling ranges.

    Reads ``consultas.csv``, groups by candidate key (using
    ``CONSULTATION_KEY_MAP``), and computes the min/max ``int_voto/100`` for
    each candidate. Every mean must fall within ``[min, max]`` of that
    candidate's polling range.

    Args:
        means: Mapping of candidate key to prior mean (proportion in [0,1]).
        consultas_path: Override path to ``consultas.csv``. If ``None``,
            resolves via ``resolve_data_dir``.

    Returns:
        None. Raises on validation failure.

    Raises:
        ValueError: If any mean falls outside its candidate's polling range,
            or if a candidate key has no polling data.

    Examples:
        >>> validate_consultation_prior_means({"gustavo_petro": 0.77})
        >>> validate_consultation_prior_means({"gustavo_petro": 0.99})
        Traceback (most recent call last):
            ...
        ValueError: Prior mean for ...

    """
    if consultas_path is None:
        data_dir = resolve_data_dir(None)
        consultas_path = str(data_dir / "2022-polls" / "consultas.csv")

    ranges: dict[str, list[float]] = {}
    with Path(consultas_path).open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = _normalize_name(row["candidato"])
            key = _NORMALIZED_CONSULTATION_MAP.get(name)
            if key is not None:
                ranges.setdefault(key, []).append(float(row["int_voto"]) / 100.0)

    for key, mean in means.items():
        if key not in ranges:
            msg = f"No consultation data found for candidate '{key}'"
            raise ValueError(msg)
        vals = ranges[key]
        lo, hi = min(vals), max(vals)
        if not (lo <= mean <= hi):
            msg = (
                f"Prior mean for '{key}' ({mean:.4f}) is outside polling range [{lo:.4f}, {hi:.4f}]"
            )
            raise ValueError(msg)


@functools.cache
def get_computed_consultation_prior_strengths() -> dict[str, float]:
    """Memoized wrapper around ``compute_consultation_prior_strength``.

    On first call, reads ``consultas.csv``, computes candidate-specific
    standard deviations, and caches the result. Subsequent calls return the
    cached dict without I/O.

    Returns:
        Mapping of candidate key to prior standard deviation, computed from
        ``consultas.csv``.

    Examples:
        >>> strengths = get_computed_consultation_prior_strengths()
        >>> isinstance(strengths, dict)
        True
        >>> strengths is get_computed_consultation_prior_strengths()
        True

    """
    return compute_consultation_prior_strength()


def _get_strengths_safe() -> dict[str, float]:
    """Compute consultation prior strengths, returning empty dict on failure."""
    with contextlib.suppress(FileNotFoundError, KeyError, ValueError):
        return get_computed_consultation_prior_strengths()
    return {}


COMPUTED_CONSULTATION_PRIOR_STRENGTHS: dict[str, float] = dict(_get_strengths_safe())


POLLSTER_RATINGS: dict[str, float] = {
    "Invamer": 10.00,
    "CNC": 8.10,
    "GAD3": 8.10,
    "Guarumo": 8.00,
    "TYSE": 6.30,
    "CELAG": 5.90,
    "AtlasIntel": 5.40,
    "MassiveCaller": 5.10,
    "Medilab": 3.80,
    "YanHaas": 3.60,
    "CifrasYConceptos": 3.60,
    "Datexco": 3.23,
    "Mosqueteros": 1.00,
}


def pollster_weight_formula(rating: float) -> float:
    """Convert pollster rating to a weight factor.

    The formula is ``weight = rating * 0.02 + 0.8``, constraining weights
    to the range [0.8, 1.0].

    Args:
        rating: La Silla Vacía pollster quality rating (0-10 scale).

    Returns:
        Weight factor between 0.8 and 1.0.

    Examples:
        >>> pollster_weight_formula(10.0)
        1.0
        >>> pollster_weight_formula(0.0)
        0.8
        >>> pollster_weight_formula(5.0)
        0.9

    """
    # The clamp is defensive: all known ratings are in [0,10], but
    # guaranteeing [0.8, 1.0] prevents silent downstream errors in
    # weighted averages from out-of-range inputs.
    return max(0.8, min(1.0, rating * 0.02 + 0.8))


def get_default_pollster_weight() -> float:
    """Return a default weight for pollsters not in POLLSTER_RATINGS.

    Computes the median of all ``POLLSTER_RATINGS`` values and applies the
    ``pollster_weight_formula`` to produce a sensible fallback weight
    between 0.8 and 1.0.

    Returns:
        Default weight factor for unrated pollsters.

    Raises:
        ValueError: If POLLSTER_RATINGS is empty.

    Examples:
        >>> get_default_pollster_weight()
        0.908

    """
    if not POLLSTER_RATINGS:
        msg = "POLLSTER_RATINGS cannot be empty"
        raise ValueError(msg)
    median_rating = statistics.median(POLLSTER_RATINGS.values())
    return pollster_weight_formula(median_rating)


def get_active_candidates(
    round_number: Literal[1, 2],
) -> list[Candidate]:
    """Return candidates active in a given election round.

    Args:
        round_number: 1 for first round, 2 for runoff.

    Returns:
        List of Candidate objects active in that round.

    Raises:
        ValueError: If round_number is not 1 or 2.

    Examples:
        >>> round1 = get_active_candidates(1)
        >>> len(round1)
        7
        >>> round2 = get_active_candidates(2)
        >>> len(round2)
        3
        >>> all(c.runoff for c in round2)
        True

    """
    if round_number == _ROUND_FIRST:
        return [c for c in FIRST_ROUND_CANDIDATES.values() if c.first_round]
    if round_number == _ROUND_SECOND:
        return [c for c in FIRST_ROUND_CANDIDATES.values() if c.runoff]
    msg = f"round_number must be 1 or 2, got {round_number!r}"
    raise ValueError(msg)


def get_candidate_column_map() -> dict[str, str]:
    """Return mapping from CSV column names to candidate keys.

    All keys in ``FIRST_ROUND_CANDIDATES`` are identity-mapped (CSV column
    name equals candidate key).

    Note:
        Currently an identity mapping (CSV column names equal candidate
        keys). If the actual CSV data uses different column names, this
        function must be updated.

    Returns:
        Dict mapping CSV column name to candidate key.

    Examples:
        >>> column_map = get_candidate_column_map()
        >>> column_map["gustavo_petro"]
        'gustavo_petro'
        >>> len(column_map)
        7

    """
    return {key: key for key in FIRST_ROUND_CANDIDATES}


# Transfer-heuristic constants were calibrated to match the aggregate split
# across 8 pollsters (~27% to Petro, ~73% to Hernandez). See
# ``notebooks/derive_transfer_constants.py`` for the derivation.
# NOTE: These values invert and adjust the SPEC-02 placeholder values.
TRANSFER_FAJARDO_PETRO: float = 0.40
TRANSFER_FAJARDO_HERNANDEZ: float = 0.60
TRANSFER_GUTIERREZ_HERNANDEZ: float = 0.87
TRANSFER_GUTIERREZ_PETRO: float = 0.13
# Blank votes are kept at a 50/50 split pending calibration.
TRANSFER_BLANCO_SPLIT: float = 0.50
