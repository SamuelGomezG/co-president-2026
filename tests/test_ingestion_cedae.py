"""SPEC-12.2: Tests for CEDAE election results ingestion."""

from __future__ import annotations

from co_president.ingestion._download_cedae import _cedae_candidate_key

__all__: list[str] = []


class TestCedaeCandidateKey:
    """Tests for ``_cedae_candidate_key``."""

    def test_known_candidate_returns_mapped_key(self) -> None:
        row = {
            "nombres": "GUSTAVO FRANCISCO",
            "primer_apellido": "PETRO",
            "segundo_apellido": "URREGO",
        }
        assert _cedae_candidate_key(row) == "gustavo_petro"

    def test_unknown_candidate_returns_unknown_prefix(self) -> None:
        row = {
            "nombres": "JOHN",
            "primer_apellido": "DOE",
            "segundo_apellido": "",
        }
        result = _cedae_candidate_key(row)
        assert result.startswith("UNKNOWN")

    def test_unknown_without_primer(self) -> None:
        row = {
            "nombres": "SOMEONE",
            "primer_apellido": "",
            "segundo_apellido": "",
        }
        assert _cedae_candidate_key(row) == "UNKNOWN_SOMEONE"

    def test_unknown_all_empty(self) -> None:
        row = {
            "nombres": "",
            "primer_apellido": "",
            "segundo_apellido": "",
        }
        assert _cedae_candidate_key(row) == "UNKNOWN"
