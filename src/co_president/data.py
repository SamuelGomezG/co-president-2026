"""SPEC-03+04: Data pipeline facade — re-exports from internal submodules.

Provides a single import surface for all data-related types and functions.
Downstream code should import from this module, not from the internal
submodules (``data_polls``, ``data_results``).
"""

from __future__ import annotations

from co_president.data_polls import (
    CandidateShares,
    CleanPolls,
    ConsultationPoll,
    PollRow,
    compute_consultation_prior_strength,
    deduplicate_polls,
    fix_invamer_date,
    get_computed_consultation_prior_strengths,
    infer_round_number,
    load_and_clean_all,
    load_raw_consultas,
    load_raw_polls,
    map_consultation_name_to_key,
    normalize_undecided,
    parse_consultations,
    retain_active_candidates,
    validate_consultation_prior_means,
)
from co_president.data_results import (
    CandidateResult,
    RoundResult,
    consolidate_round,
    cross_validate,
    load_canonical_results,
    load_moe_round1,
    load_moe_round2,
    load_participation_round1,
    load_participation_round2,
    load_registraduria_round1,
    load_registraduria_round2,
)

__all__ = [
    "CandidateResult",
    "CandidateShares",
    "CleanPolls",
    "ConsultationPoll",
    "PollRow",
    "RoundResult",
    "compute_consultation_prior_strength",
    "consolidate_round",
    "cross_validate",
    "deduplicate_polls",
    "fix_invamer_date",
    "get_computed_consultation_prior_strengths",
    "infer_round_number",
    "load_and_clean_all",
    "load_canonical_results",
    "load_moe_round1",
    "load_moe_round2",
    "load_participation_round1",
    "load_participation_round2",
    "load_raw_consultas",
    "load_raw_polls",
    "load_registraduria_round1",
    "load_registraduria_round2",
    "map_consultation_name_to_key",
    "normalize_undecided",
    "parse_consultations",
    "retain_active_candidates",
    "validate_consultation_prior_means",
]
