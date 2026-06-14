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

from dataclasses import dataclass
from datetime import date
import math
import statistics
from typing import Literal

__all__ = [
    "COALITION_TO_CANDIDATE",
    "CONSULTATION_DATE",
    "CONSULTATION_KEY_MAP",
    "CONSULTATION_VOTES",
    "ELECTION_DATES",
    "ELECTION_DATE_ROUND1",
    "ELECTION_DATE_ROUND2",
    "FIRST_ROUND_CANDIDATES",
    "FIRST_ROUND_CANDIDATES_2026",
    "HISTORICAL_CANDIDATE_IDEOLOGY",
    "HISTORICAL_ROUND2_IDEOLOGY",
    "HISTORICAL_TURNOUT_SOURCE",
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
        consultation_prior_strength_override: Candidate-specific override.
        fundamentals_mode: Municipal model mode (off/prior_only/joint).
        beta_coefficient_prior_sigma: Std dev for beta coefficient priors.
        use_horseshoe_prior: Horseshoe prior on beta coefficients (default
            Normal(0, 0.5)).  Enables local shrinkage per coefficient while
            preserving signal coefficients.
        sigma_m_prior: HalfNormal sigma for municipal random-effect scale.
        pool_alpha: Global shrinkage on non-centered mu_m_raw.
        enable_population_weighting: Turnout-weighted effective population.
        clr_target: Use CLR-transformed targets in municipal model.

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

    # SPEC-22: Municipal hierarchical model
    fundamentals_mode: Literal["off", "prior_only", "joint"] = "prior_only"
    beta_coefficient_prior_sigma: float = 0.5
    use_horseshoe_prior: bool = False
    sigma_m_prior: float = 0.3
    pool_alpha: float = 0.95
    enable_population_weighting: bool = True
    clr_target: bool = False

    @property
    def computed_consultation_prior_strengths(self) -> dict[str, float]:
        """Lazily computed candidate-specific prior strengths.

        Calls ``get_computed_consultation_prior_strengths`` from the
        ``co_president.data`` facade on first access (not at module
        import time).  Merges with ``consultation_prior_strength_override``,
        which takes precedence.  Returns empty dict when ``consultas.csv``
        is missing or cannot be read.

        Returns:
            Mapping of candidate key to prior standard deviation,
            or empty dict if unavailable.

        """
        from co_president.data import get_computed_consultation_prior_strengths  # noqa: PLC0415, I001

        try:
            computed = get_computed_consultation_prior_strengths().copy()
        except (FileNotFoundError, KeyError, ValueError):
            computed = {}

        if self.consultation_prior_strength_override is not None:
            computed.update(self.consultation_prior_strength_override)
        return computed


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

# Placeholder for 2026 first-round candidates.  Official candidate list is
# not yet published.  Once known, replace with real Candidate objects.
# Until populated, forecasts default to FIRST_ROUND_CANDIDATES for
# candidate-agnostic feature space.
FIRST_ROUND_CANDIDATES_2026: dict[str, Candidate] = {}

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

# Election dates for the first round of each presidential election since 2002.
# Used by ``build_multi_election_model`` (SPEC-40) for per-year time indexing.
ELECTION_DATES: dict[int, date] = {
    2002: date(2002, 5, 26),
    2006: date(2006, 5, 28),
    2010: date(2010, 5, 30),
    2014: date(2014, 5, 25),
    2018: date(2018, 5, 27),
    2022: date(2022, 5, 29),
    2026: date(2026, 5, 31),
}

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
    year: int = 2022,
) -> list[Candidate]:
    """Return candidates active in a given election round.

    Args:
        round_number: 1 for first round, 2 for runoff.
        year: Election year (default 2022).  When 2026, uses
            ``FIRST_ROUND_CANDIDATES_2026`` if populated, otherwise
            falls back to ``FIRST_ROUND_CANDIDATES``.

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
    candidates = (
        FIRST_ROUND_CANDIDATES_2026
        if year == 2026 and FIRST_ROUND_CANDIDATES_2026  # noqa: PLR2004
        else FIRST_ROUND_CANDIDATES
    )
    if round_number == _ROUND_FIRST:
        return [c for c in candidates.values() if c.first_round]
    if round_number == _ROUND_SECOND:
        return [c for c in candidates.values() if c.runoff]
    msg = f"round_number must be 1 or 2, got {round_number!r}"
    raise ValueError(msg)


def get_candidate_column_map(year: int = 2022) -> dict[str, str]:
    """Return mapping from CSV column names to candidate keys.

    All keys in ``FIRST_ROUND_CANDIDATES`` are identity-mapped (CSV column
    name equals candidate key).

    Note:
        Currently an identity mapping (CSV column names equal candidate
        keys). If the actual CSV data uses different column names, this
        function must be updated.

    Args:
        year: Election year (default 2022).  When 2026, uses
            ``FIRST_ROUND_CANDIDATES_2026`` if populated.

    Returns:
        Dict mapping CSV column name to candidate key.

    Examples:
        >>> column_map = get_candidate_column_map()
        >>> column_map["gustavo_petro"]
        'gustavo_petro'
        >>> len(column_map)
        7

    """
    candidates = (
        FIRST_ROUND_CANDIDATES_2026
        if year == 2026 and FIRST_ROUND_CANDIDATES_2026  # noqa: PLR2004
        else FIRST_ROUND_CANDIDATES
    )
    return {key: key for key in candidates}


# Historical candidate ideology registry for the 2002-2022 elections.
# Maps election year -> {left candidate key, right candidate key}.
# Used by ``load_features()`` to construct ``HistoricalRecord`` objects.
HISTORICAL_CANDIDATE_IDEOLOGY: dict[int, dict[str, str]] = {
    2002: {"left": "horacio_serpa", "right": "alvaro_uribe"},
    2006: {"left": "carlos_gaviria", "right": "alvaro_uribe"},
    2010: {"left": "gustavo_petro", "right": "juan_manuel_santos"},
    2014: {"left": "clara_lopez", "right": "oscar_ivan_zuluaga"},
    2018: {"left": "gustavo_petro", "right": "ivan_duque"},
    2022: {"left": "gustavo_petro", "right": "federico_gutierrez"},
}

# Round-2 overrides: when the runoff right-wing candidate differs from the
# round-1 right-wing candidate (e.g. 2022: Rodolfo Hernandez).
HISTORICAL_ROUND2_IDEOLOGY: dict[int, dict[str, str]] = {
    2022: {"left": "gustavo_petro", "right": "rodolfo_hernandez"},
}

# Source definition for historical turnout data.
# Each entry maps an election year to the primary data source for municipal
# turnout (registered voters / census) for that year.
# - 2022: MOE ``censo`` field from ``moe_camara_territorial_2022.csv``
#   (primary, full municipal coverage).
# - 2002-2018: Registraduría ``censo_electoral`` field from CEDAE
#   ``*_presidencia.dta.csv.gz`` files.
HISTORICAL_TURNOUT_SOURCE: dict[int, str] = {
    2022: "MOE_2022_censo",
    2018: "CEDAE_2018_censo_electoral",
    2014: "CEDAE_2014_censo_electoral",
    2010: "CEDAE_2010_censo_electoral",
    2006: "CEDAE_2006_censo_electoral",
    2002: "CEDAE_2002_censo_electoral",
}

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
