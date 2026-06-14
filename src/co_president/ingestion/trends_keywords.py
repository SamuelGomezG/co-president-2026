"""SPEC-38: Google Trends keywords for Colombian presidential election.

Maps internal candidate keys to Spanish-language search queries
for Google Trends search interest extraction.
"""

from __future__ import annotations

CANDIDATE_QUERY_MAP_2026: dict[str, str] = {
    "gustavo_petro": "Gustavo Petro",
    "federico_gutierrez": "Federico Guti\u00e9rrez",
    "rodolfo_hernandez": "Rodolfo Hern\u00e1ndez",
    "sergio_fajardo": "Sergio Fajardo",
    "ingrid_betancourt": "Ingrid Betancourt",
}

CANDIDATE_QUERY_MAP_2022: dict[str, str] = {
    "gustavo_petro": "Gustavo Petro",
    "rodolfo_hernandez": "Rodolfo Hern\u00e1ndez",
    "federico_gutierrez": "Federico Guti\u00e9rrez",
    "sergio_fajardo": "Sergio Fajardo",
}

CANDIDATE_QUERY_MAPS: dict[str, dict[str, str]] = {
    "2022": CANDIDATE_QUERY_MAP_2022,
    "2026": CANDIDATE_QUERY_MAP_2026,
}

__all__ = [
    "CANDIDATE_QUERY_MAPS",
    "CANDIDATE_QUERY_MAP_2022",
    "CANDIDATE_QUERY_MAP_2026",
]
