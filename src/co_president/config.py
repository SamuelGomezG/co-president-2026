"""SPEC-02: Configuration & Constants.

Candidate maps, dates, pollster ratings, and model hyperparameters. All
downstream modules import from this module rather than hardcoding values.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math
import statistics
from typing import Literal

__all__ = [
    "BELEN_DE_BAJIRA_CODE",
    "BOGOTA_LOCALIDADES",
    "COALITION_TO_CANDIDATE",
    "COALITION_TO_CANONICAL_WEIGHTS",
    "CONSULTATION_DATE",
    "CONSULTATION_KEY_MAP",
    "CONSULTATION_VOTES",
    "ELECTION_DATES",
    "ELECTION_DATE_ROUND1",
    "ELECTION_DATE_ROUND2",
    "EXPECTED_MUNICIPALITIES",
    "EXPECTED_MUNICIPALITIES_POPULATION_INCL_ANM",
    "EXPECTED_MUNICIPALITIES_WITH_LOCALIDADES",
    "FIRST_ROUND_CANDIDATES",
    "FIRST_ROUND_CANDIDATES_2026",
    "HISTORICAL_CANDIDATE_IDEOLOGY",
    "HISTORICAL_CANDIDATE_IDEOLOGY_5CLASS",
    "HISTORICAL_ROUND2_IDEOLOGY",
    "HISTORICAL_TURNOUT_SOURCE",
    "POLLSTER_RATINGS",
    "Candidate",
    "ModelConfig",
    "consultation_log_share_prior",
    "get_active_candidates",
    "get_candidate_column_map",
    "get_ideology",
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
    withdrawal_date: date | None = None


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
        nuts_sampler: External NUTS sampler. Options: ``"numpyro"`` (JAX),
            ``"blackjax"`` (JAX alternative), ``"nutpie"`` (Rust NUTS),
            ``"pymc"`` (PyMC default), or ``None`` (auto-select PyMC default).
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
    concentration_election_prior_mean: float = 5000.0
    concentration_election_votes_scale: int = 1000000
    concentration_election_prior_shape: float = 0.75
    concentration_t1_boost: float = 0.0
    use_digital_signals_runoff: bool = False
    house_effect_sigma_prior: float = 1.0
    mcmc_draws: int = 4000
    mcmc_tune: int = 1000
    mcmc_chains: int = 4
    mcmc_cores: int = 4
    target_accept: float = 0.95
    nuts_sampler: Literal["pymc", "nutpie", "numpyro", "blackjax"] | None = None
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

    # SPEC-30: Transfer rate estimation model
    transfer_rhat_threshold: float = 1.10

    # Phase 4: Target election year for backtesting
    target_year: int = 2022

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
        withdrawal_date=date(2022, 5, 20),
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


# Bogotá D.C. localidad code to name mapping (DANE DIVIPOLA localidad codes).
# Codes 01-20 are the official 20 localidades.  Code 99 is the catch-all for
# polling stations without a localidad assignment (SIN COMUNA).
BOGOTA_LOCALIDADES: dict[str, str] = {
    "01": "Usaquén",
    "02": "Chapinero",
    "03": "Santa Fe",
    "04": "San Cristóbal",
    "05": "Usme",
    "06": "Tunjuelito",
    "07": "Bosa",
    "08": "Kennedy",
    "09": "Fontibón",
    "10": "Engativá",
    "11": "Suba",
    "12": "Barrios Unidos",
    "13": "Teusaquillo",
    "14": "Los Mártires",
    "15": "Antonio Nariño",
    "16": "Puente Aranda",
    "17": "La Candelaria",
    "18": "Rafael Uribe Uribe",
    "19": "Ciudad Bolívar",
    "20": "Sumapaz",
    "99": "BOGOTÁ D.C. - SIN COMUNA",
}

# Canonical municipality count per DIVIPOLA master catalog
# (data/fundamentals/divipola_master.csv).  Colombia has 1,122
# municipalities.  This is the pre-Bogotá-disaggregation count:
# Bogotá (11001) counts as a single municipality.
EXPECTED_MUNICIPALITIES: int = 1_122

# Post-Bogotá-disaggregation count used by features.py after
# build_feature_matrix replaces Bogotá (11001) with 21 localidad
# codes (01-20 + 99 catch-all).  1,122 - 1 + 21 + 1 (Belén de Bajirá) = 1,143.
EXPECTED_MUNICIPALITIES_WITH_LOCALIDADES: int = 1_143

# Population vintage including Archipiélago de San Andrés, Providencia y
# Santa Catalina (ANM, codigo_municipio 88xxx).
#   1,122 + 1 ANM + 1 (Belén de Bajirá) = 1,124.
EXPECTED_MUNICIPALITIES_POPULATION_INCL_ANM: int = 1_124

BELEN_DE_BAJIRA_CODE: str = "27086"


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
    as_of: date | None = None,
) -> list[Candidate]:
    """Return candidates active in a given election round.

    Args:
        round_number: 1 for first round, 2 for runoff.
        year: Election year (default 2022).  When 2026, uses
            ``FIRST_ROUND_CANDIDATES_2026`` if populated, otherwise
            falls back to ``FIRST_ROUND_CANDIDATES``.
        as_of: Optional date filter.  When provided, excludes candidates
            whose ``withdrawal_date`` is before this date (i.e., candidates
            who had already withdrawn by ``as_of``).

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
        active = [c for c in candidates.values() if c.first_round]
    elif round_number == _ROUND_SECOND:
        active = [c for c in candidates.values() if c.runoff]
    else:
        msg = f"round_number must be 1 or 2, got {round_number!r}"
        raise ValueError(msg)
    if as_of is not None:
        active = [c for c in active if c.withdrawal_date is None or as_of < c.withdrawal_date]
    return active


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


# Ideology classes for left/right grouping in 2-class derivation.
_IDEOLOGY_LEFT = frozenset({"Izquierda", "Centro_Izquierda"})
_IDEOLOGY_RIGHT = frozenset({"Derecha", "Centro_Derecha"})
_ROUND_TWO = 2


def _first_candidate(
    mapping: dict[str, str],
    target_classes: frozenset[str],
) -> str | None:
    """First candidate in *mapping* with class in *target_classes*, preferring canonical names."""
    canon: str | None = None
    raw: str | None = None
    for k, v in mapping.items():
        if v in target_classes:
            is_canon = "_" in k and k.islower()
            if is_canon:
                if canon is None:
                    canon = k
            elif raw is None:
                raw = k
        if canon is not None and raw is not None:
            break
    return canon or raw


def _first_centro(mapping: dict[str, str]) -> str | None:
    """Return first Centro candidate in *mapping*, preferring canonical names."""
    canon: str | None = None
    raw: str | None = None
    for k, v in mapping.items():
        if v == "Centro":
            if "_" in k and k.islower():
                if canon is None:
                    canon = k
            elif raw is None:
                raw = k
        if canon is not None and raw is not None:
            break
    return canon or raw


def _derive_ideology_2class() -> dict[int, dict[str, str]]:
    """Auto-derive 2-class ideology from the 5-class master map.

    Centro candidates are excluded from automatic derivation per Option A.
    If a year has no left or right candidate from non-Centro classes, the
    first Centro candidate is used as a fallback for the missing side.
    Canonical (lowercase_snake_case) names are preferred over raw MMV names.

    Returns:
        Dict mapping year -> ``{"left": candidate, "right": candidate}``.

    """
    result: dict[int, dict[str, str]] = {}
    for (year, round_num), round_map in HISTORICAL_CANDIDATE_IDEOLOGY_5CLASS.items():
        if round_num != 1:
            continue
        left = _first_candidate(round_map, _IDEOLOGY_LEFT)
        right = _first_candidate(round_map, _IDEOLOGY_RIGHT)
        centro = _first_centro(round_map) if left is None or right is None else None
        if left is None:
            left = centro
        if right is None:
            right = centro
        if left is not None and right is not None:
            result[year] = {"left": left, "right": right}
    return result


# 5-class ideological classification for each presidential election (2002-2022).
# Maps (year, round) -> {candidate: ideology_class}.
# Expert-derived mapping based on Colombian political tradition.
# Classes: Izquierda, Centro_Izquierda, Centro, Centro_Derecha, Derecha.
# Both canonical (lowercase_underscore) and raw MMV name variants are included
# to support different column-name conventions in the feature matrix.
# Canonical entries are ordered by approximate vote power so that the first
# left/right match produces the expected 2-class derivation.
HISTORICAL_CANDIDATE_IDEOLOGY_5CLASS: dict[tuple[int, int], dict[str, str]] = {
    (2002, 1): {
        "alvaro_uribe": "Derecha",
        "horacio_serpa": "Centro",
        "luis_eduardo_garzon": "Izquierda",
        "ingrid_betancourt": "Centro",
        "noemi_sanin": "Centro_Derecha",
    },
    (2006, 1): {
        "alvaro_uribe": "Derecha",
        "carlos_gaviria": "Izquierda",
        "horacio_serpa": "Centro",
        "antanas_mockus": "Centro",
    },
    (2010, 1): {
        "juan_manuel_santos": "Centro_Derecha",
        "antanas_mockus": "Centro",
        "gustavo_petro": "Izquierda",
        "noemi_sanin": "Centro_Derecha",
    },
    (2010, 2): {
        "juan_manuel_santos": "Centro_Derecha",
        "antanas_mockus": "Centro",
    },
    (2014, 1): {
        "juan_manuel_santos": "Centro_Derecha",
        "oscar_ivan_zuluaga": "Derecha",
        "enrique_penalosa": "Centro",
        "LOPEZ": "Izquierda",
        "clara_lopez": "Izquierda",
    },
    (2014, 2): {
        "juan_manuel_santos": "Centro_Derecha",
        "oscar_ivan_zuluaga": "Derecha",
    },
    (2018, 1): {
        "ivan_duque": "Derecha",
        "gustavo_petro": "Izquierda",
        "sergio_fajardo": "Centro",
        "DE LA CALLE": "Centro_Izquierda",
    },
    (2018, 2): {
        "ivan_duque": "Derecha",
        "gustavo_petro": "Izquierda",
    },
    (2022, 1): {
        "GUSTAVO PETRO": "Izquierda",
        "gustavo_petro": "Izquierda",
        "RODOLFO HERNÁNDEZ": "Derecha",
        "rodolfo_hernandez": "Derecha",
        "FEDERICO GUTIÉRREZ": "Centro_Derecha",
        "federico_gutierrez": "Centro_Derecha",
        "SERGIO FAJARDO": "Centro",
        "sergio_fajardo": "Centro",
        "INGRID BETANCOURT": "Centro",
        "ingrid_betancourt": "Centro",
        "JOHN MILTON RODRÍGUEZ": "Derecha",
        "LUIS PÉREZ": "Centro_Derecha",
    },
    (2022, 2): {
        "gustavo_petro": "Izquierda",
        "rodolfo_hernandez": "Derecha",
        "GUSTAVO PETRO": "Izquierda",
        "RODOLFO HERNÁNDEZ": "Derecha",
    },
}


# Auto-derived 2-class ideology from the 5-class master map.
# Centro candidates excluded; canonical names preferred over raw MMV.
HISTORICAL_CANDIDATE_IDEOLOGY: dict[int, dict[str, str]] = _derive_ideology_2class()

# Round-2 overrides: when the runoff right-wing candidate differs from the
# round-1 right-wing candidate (e.g. 2022: Rodolfo Hernandez).
HISTORICAL_ROUND2_IDEOLOGY: dict[int, dict[str, str]] = {
    2022: {"left": "gustavo_petro", "right": "rodolfo_hernandez"},
}


def get_ideology(year: int, round_num: int) -> dict[str, str]:
    """Return the left/right candidate mapping for a given year and round.

    Uses ``HISTORICAL_ROUND2_IDEOLOGY`` for round 2 when available (e.g. 2022
    R2 uses Rodolfo Hernández instead of the round-1 right candidate).  Falls
    back to ``HISTORICAL_CANDIDATE_IDEOLOGY`` (auto-derived from 5-class map)
    otherwise.

    Args:
        year: Election year.
        round_num: Election round (1 or 2).

    Returns:
        Dict with ``"left"`` and ``"right"`` keys, or empty dict if the year
        is not in the registry.

    """
    if round_num == _ROUND_TWO and year in HISTORICAL_ROUND2_IDEOLOGY:
        return HISTORICAL_ROUND2_IDEOLOGY[year]
    return HISTORICAL_CANDIDATE_IDEOLOGY.get(year, {})


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

# Per-department coalition→canonical-party weight table (SPEC-30).
# Maps MOE 2022 coalition lines to the constituent parties they contain,
# with vote-share weights.  Department-level entries override the
# ``__national__`` default.  Generated from MOE 2022 legislative data.
# Used by ``map_moe_party_to_canonical()`` to disaggregate coalition totals
# into canonical party-level shares for the transfer model features.
# NOTE: This is a static snapshot; regenerate via
# ``notebooks/generate_coalition_crosswalk.py`` after data updates.
# Partido 1 = Liberal, 2 = Conservador, 3 = Cambio Radical, 4 = Alianza Verde,
# 8 = Partido de la U, 11 = Centro Democrático, 12 = MIRA,
# 13 = Polo Democrático Alternativo, 14 = Unión Patriótica,
# 15 = Colombia Humana, 16 = Nuevo Liberalismo, 17 = Partido Comunes,
# 18 = Colombia Justa Libres.
COALITION_TO_CANONICAL_WEIGHTS: dict[str, dict[str, float | dict[str, float]]] = {
    "COALICION PACTO HISTORICO": {
        "__national__": {
            "Colombia Humana (15)": 0.25,
            "Polo Democrático (13)": 0.25,
            "Unión Patriótica (14)": 0.20,
            "Partido Comunes (17)": 0.15,
            "Partido del Trabajo": 0.10,
            "Movimiento Progresistas": 0.05,
        },
    },
    "COALICION EQUIPO POR COLOMBIA": {
        "__national__": {
            "Conservador (2)": 0.30,
            "Centro Democrático (11)": 0.30,
            "Cambio Radical (3)": 0.20,
            "MIRA (12)": 0.10,
            "Colombia Justa Libres (18)": 0.10,
        },
    },
    "COALICION CENTRO ESPERANZA": {
        "__national__": {
            "Alianza Verde (4)": 0.40,
            "Nuevo Liberalismo (16)": 0.25,
            "Liberal (1)": 0.20,
            "Partido de la U (8)": 0.15,
        },
    },
    "LIGA DE GOBERNANTES ANTICORRUPCION": {
        "__national__": {
            "Liga Anticorrupción": 1.0,
        },
    },
    "PARTIDO VERDE OXIGENO": {
        "__national__": {
            "Verde Oxígeno": 1.0,
        },
    },
    "COLOMBIA JUSTA LIBRES": {
        "__national__": {
            "Colombia Justa Libres (18)": 1.0,
        },
    },
    "PARTIDO MOVIMIENTO DE SALVACION NACIONAL": {
        "__national__": {
            "Movimiento Salvación Nacional": 1.0,
        },
    },
    "COLOMBIA PIENSA EN GRANDE": {
        "__national__": {
            "Colombia Piensa en Grande": 1.0,
        },
    },
    "COALICION ALIANZA VERDE Y CENTRO ESPERANZA": {
        "__national__": {
            "Alianza Verde (4)": 0.50,
            "Nuevo Liberalismo (16)": 0.30,
            "Liberal (1)": 0.20,
        },
    },
    "COALICION MIRA": {
        "__national__": {
            "MIRA (12)": 0.60,
            "Colombia Justa Libres (18)": 0.40,
        },
    },
}
