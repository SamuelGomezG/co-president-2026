"""SPEC-31: Tests for 2026 CNE microdata ingestion."""

from __future__ import annotations

from datetime import date
import sys
from typing import TYPE_CHECKING, Self
import zipfile

import pandas as pd
import pytest

from co_president.data_cne_2026 import (
    CANDIDATE_KEY_MAP_2026,
    extract_pdf_topline,
    extract_runoff_pairings,
    extract_topline,
    load_atlas_intel,
    load_bundle,
    normalize_candidate_labels,
)

if TYPE_CHECKING:
    from pathlib import Path


def _make_fake_pdfplumber(text: str) -> object:
    """Build a minimal fake pdfplumber module that returns *text*."""

    class FakePage:
        def extract_text(self) -> str:
            return text

    class FakePdf:
        def __init__(self) -> None:
            self.pages = [FakePage()]

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    class FakePdfplumber:
        @staticmethod
        def open(_path: object) -> FakePdf:
            return FakePdf()

    return FakePdfplumber()


class TestNormalizeCandidateLabels:
    """Tests for the canonical candidate label normalizer."""

    def test_maps_known_variants(self) -> None:
        """Known display names and variants map to canonical keys."""
        series = pd.Series(
            [
                "Iván Cepeda",
                "Paloma Valencia",
                "Abelardo de la Espriella",
                "Sergio Fajardo",
                "Voto en Blanco",
                "No sabe / no responde",
                "Otro candidato",
            ],
        )
        expected = pd.Series(
            ["cepeda", "valencia", "de_la_espriella", "fajardo", "blanco", "ns_nr", "rest"],
        )
        result = normalize_candidate_labels(series)
        pd.testing.assert_series_equal(result, expected)

    def test_unknown_label_raises(self) -> None:
        """Unknown candidate labels raise ValueError."""
        series = pd.Series(["Iván Cepeda", "Candidato Desconocido"])
        with pytest.raises(ValueError, match="Unmapped candidate label"):
            normalize_candidate_labels(series)

    def test_empty_and_na_preserved(self) -> None:
        """Empty strings and NA are preserved as NA."""
        series = pd.Series(["Iván Cepeda", "", None])
        result = normalize_candidate_labels(series)
        assert result.iloc[0] == "cepeda"
        assert pd.isna(result.iloc[1])
        assert pd.isna(result.iloc[2])


class TestAtlasIntelLoader:
    """Tests for the Atlas Intel CSV loader."""

    def test_loads_synthetic_atlas_csv(self, tmp_path: Path) -> None:
        """A tiny zipped CSV with vote_r1 and weight loads correctly."""
        csv_data = (
            "user_id,weight,presidential_election_2026\n"
            "1,1.5,Iván Cepeda\n"
            "2,1.0,Paloma Valencia\n"
            "3,0.5,Voto en Blanco\n"
            "4,1.0,No sabe / no responde\n"
        )
        zip_path = tmp_path / "atlas_intel_01012026_02012026.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("Base de Dados Atlas.csv", csv_data)

        df = load_atlas_intel(zip_path)
        expected_cols = {
            "fecha",
            "weight",
            "candidate_r1",
            "candidate_r2",
            "firm",
            "field_end",
            "raw_source",
        }
        assert set(df.columns) == expected_cols
        assert len(df) == 4
        assert df["weight"].tolist() == [1.5, 1.0, 0.5, 1.0]

    def test_extract_topline_from_synthetic_atlas(self, tmp_path: Path) -> None:
        """Weighted topline percentages sum to 100 for synthetic data."""
        csv_data = (
            "user_id,weight,presidential_election_2026\n"
            "1,2.0,Iván Cepeda\n"
            "2,1.0,Paloma Valencia\n"
            "3,1.0,Voto en Blanco\n"
        )
        zip_path = tmp_path / "atlas_intel_01012026_02012026.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("Base de Dados Atlas.csv", csv_data)

        df = load_atlas_intel(zip_path)
        field_end = date(2026, 1, 2)
        topline = extract_topline(
            df, date_col="field_end", candidate_col="candidate_r1", weight_col="weight"
        )

        assert not topline.empty
        assert topline["encuestadora"].iloc[0] == "atlas_intel"
        assert topline["field_end"].iloc[0] == field_end
        # weights: 2.0 cepeda, 1.0 valencia, 1.0 blanco → 50/25/25
        assert topline.loc[topline["field_end"] == field_end, "cepeda"].iloc[0] == pytest.approx(
            50.0
        )
        assert topline.loc[topline["field_end"] == field_end, "valencia"].iloc[0] == pytest.approx(
            25.0
        )
        assert topline.loc[topline["field_end"] == field_end, "blanco"].iloc[0] == pytest.approx(
            25.0
        )


class TestRunoffPairings:
    """Tests for runoff pairing extraction."""

    def test_extract_runoff_pairings_named_columns(self) -> None:
        """Named runoff columns produce correct weighted pairing counts."""
        runoff_df = pd.DataFrame(
            {
                "candidate_a": ["cepeda", "cepeda", "cepeda", "ns_nr"],
                "candidate_b": ["valencia", "valencia", "valencia", "valencia"],
                "weight": [1.0, 1.0, 2.0, 1.0],
                "field_end": [date(2026, 5, 1)] * 4,
                "firm": ["atlas_intel"] * 4,
            },
        )
        pairings = extract_runoff_pairings(runoff_df)

        assert len(pairings) == 1
        row = pairings.iloc[0]
        assert row["candidate_a"] == "cepeda"
        assert row["candidate_b"] == "valencia"
        assert row["n_a"] == pytest.approx(4.0)
        assert row["n_b"] == pytest.approx(4.0)
        assert row["n_total"] == 3
        assert row["effective_n"] > 0

    def test_no_runoff_data_returns_empty(self) -> None:
        """DataFrame without runoff columns returns empty pairings."""
        runoff_df = pd.DataFrame(
            {
                "firm": ["atlas_intel", "atlas_intel"],
                "field_end": [date(2026, 5, 1), date(2026, 5, 1)],
                "candidate_a": ["cepeda", "valencia"],
                "candidate_b": [None, None],
                "weight": [1.0, 1.0],
            }
        )
        pairings = extract_runoff_pairings(runoff_df)
        assert pairings.empty


class TestLoadBundle:
    """Tests for the bundle dispatcher."""

    def test_unknown_extension_raises(self, tmp_path: Path) -> None:
        """A bundle without a registered firm raises ValueError."""
        zip_path = tmp_path / "unknown_firm_01012026_02012026.zip"
        zip_path.write_bytes(b"PK")
        with pytest.raises(ValueError, match="No loader for firm"):
            load_bundle(zip_path)


class TestPdfExtraction:
    """Tests for PDF topline extraction fallback."""

    def test_extract_pdf_topline_mock(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """PDF extraction parses candidate percentages from text."""
        pdf_path = tmp_path / "report.pdf"
        fake_text = "Primera vuelta\nIván Cepeda 34.2%\nPaloma Valencia 28.2%\nBlanco 5.7%\n"
        monkeypatch.setitem(sys.modules, "pdfplumber", _make_fake_pdfplumber(fake_text))
        result = extract_pdf_topline(pdf_path)

        # PDF extraction returns None (stub — not yet implemented)
        assert result is None

    def test_extract_pdf_topline_unrecoverable_returns_none(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """PDF extraction returns None when no candidate percentages are found."""
        pdf_path = tmp_path / "report.pdf"
        monkeypatch.setitem(
            sys.modules,
            "pdfplumber",
            _make_fake_pdfplumber("Some unrelated text without candidates"),
        )
        result = extract_pdf_topline(pdf_path)
        assert result is None


class TestCandidateKeyMap:
    """Tests for the public candidate key map."""

    def test_keys_are_canonical(self) -> None:
        """All mapped values are valid canonical keys."""
        expected = {
            "cepeda",
            "de_la_espriella",
            "valencia",
            "fajardo",
            "claudia_lopez",
            "rest",
            "blanco",
            "ns_nr",
        }
        assert set(CANDIDATE_KEY_MAP_2026.values()).issubset(expected)
