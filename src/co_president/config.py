"""SPEC-02: Configuration & Constants.

Candidate maps, dates, pollster ratings, and model hyperparameters. All
downstream modules import from this module rather than hardcoding values.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
from typing import Literal

__all__ = [
    "COALITION_TO_CANDIDATE",
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
    "consultation_log_share_prior",
    "get_active_candidates",
    "get_candidate_column_map",
    "pollster_weight_formula",
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


TRANSFER_FAJARDO_PETRO: float = 0.60
TRANSFER_FAJARDO_HERNANDEZ: float = 0.40
TRANSFER_GUTIERREZ_HERNANDEZ: float = 0.70
TRANSFER_GUTIERREZ_PETRO: float = 0.30
TRANSFER_BLANCO_SPLIT: float = 0.5

CONSULTATION_KEY_MAP: dict[str, str] = {
    "Gustavo Petro": "gustavo_petro",
    "Federico Gutiérrez": "federico_gutierrez",
    "Sergio Fajardo": "sergio_fajardo",
    "Ingrid Betancourt": "ingrid_betancourt",
    "Rodolfo Hernández": "rodolfo_hernandez",
}
