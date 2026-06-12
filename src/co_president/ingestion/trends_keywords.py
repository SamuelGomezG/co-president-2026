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

__all__ = [
    "CANDIDATE_QUERY_MAP_2026",
]
