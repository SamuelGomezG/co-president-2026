"""SPEC-38: Google Trends keywords for Colombian presidential election.

Maps internal candidate keys to Spanish-language search queries
for Google Trends search interest extraction.
"""

from __future__ import annotations

CANDIDATE_QUERY_MAP_2026: dict[str, str] = {
    "gustavo_petro": "Petro",
    "federico_gutierrez": "Fico",
    "rodolfo_hernandez": "Rodolfo",
    "sergio_fajardo": "Fajardo",
    "ingrid_betancourt": "Ingrid",
}

CANDIDATE_QUERY_MAP_2022: dict[str, str] = {
    "gustavo_petro": "Petro",
    "rodolfo_hernandez": "Rodolfo",
    "federico_gutierrez": "Fico",
    "sergio_fajardo": "Fajardo",
}

CANDIDATE_QUERY_MAP_2022_RUNOFF: dict[str, str] = {
    "gustavo_petro": "Petro",
    "rodolfo_hernandez": "Rodolfo",
}

CANDIDATE_QUERY_MAPS: dict[str, dict[str, str]] = {
    "2022": CANDIDATE_QUERY_MAP_2022,
    "2022_runoff": CANDIDATE_QUERY_MAP_2022_RUNOFF,
    "2026": CANDIDATE_QUERY_MAP_2026,
}

__all__ = [
    "CANDIDATE_QUERY_MAPS",
    "CANDIDATE_QUERY_MAP_2022",
    "CANDIDATE_QUERY_MAP_2022_RUNOFF",
    "CANDIDATE_QUERY_MAP_2026",
]
