"""SPEC-39: Tests for Twitter Spanish tweet normalization."""

from __future__ import annotations

import pandas as pd

from co_president.ingestion.twitter_preprocess import normalize_spanish_tweets

__all__: list[str] = []


# ═══════════════════════════════════════════════════════════════════
# Normalize tests
# ═══════════════════════════════════════════════════════════════════


class TestNormalizeSpanishTweets:
    """normalize_spanish_tweets returns cleaned text DataFrame."""

    def test_returns_dataframe_with_expected_columns(self) -> None:
        """Output has original_text and clean_text columns."""
        result = normalize_spanish_tweets(["Hola mundo"])
        assert list(result.columns) == ["original_text", "clean_text"]

    def test_removes_urls(self) -> None:
        """URLs are replaced with 'url' token."""
        result = normalize_spanish_tweets(["Visita https://example.com ahora"])
        assert "url" in result["clean_text"].iloc[0]

    def test_removes_mentions(self) -> None:
        """Mentions are replaced with 'usuario' token."""
        result = normalize_spanish_tweets(["Hola @usuario123"])
        assert "usuario" in result["clean_text"].iloc[0]

    def test_splits_camelcase_hashtags(self) -> None:
        """CamelCase hashtags split into separate words."""
        result = normalize_spanish_tweets(["#ReformaLaboralYa"])
        clean = result["clean_text"].iloc[0]
        assert "reforma" in clean
        assert "laboral" in clean

    def test_shortens_repeated_chars(self) -> None:
        """Repeated characters shortened to 2."""
        result = normalize_spanish_tweets(["noooooo"])
        assert result["clean_text"].iloc[0] == "noo"

    def test_normalizes_laughter(self) -> None:
        """Laughter patterns normalized to 'jaja'."""
        result = normalize_spanish_tweets(["jajajajaja"])
        assert result["clean_text"].iloc[0] == "jaja"

    def test_reduces_repeated_punctuation(self) -> None:
        """Repeated punctuation reduced to single mark."""
        result = normalize_spanish_tweets(["Qué bien!!!"])
        assert result["clean_text"].iloc[0] == "qué bien!"

    def test_lowercases_all_caps_words(self) -> None:
        """All-caps words of 3+ chars are lowercased."""
        result = normalize_spanish_tweets(["ESTO es INCREÍBLE"])
        clean = result["clean_text"].iloc[0]
        assert "esto" in clean
        assert "increíble" in clean

    def test_removes_digits(self) -> None:
        """Digit sequences replaced with 'numero' token."""
        result = normalize_spanish_tweets(["Gana 5000 votos"])
        assert "numero" in result["clean_text"].iloc[0]

    def test_replaces_emoticons(self) -> None:
        """ASCII emoticons replaced with Spanish tokens."""
        result = normalize_spanish_tweets(["Me encanta :)", "Qué triste :("])
        assert "sonrisa" in result["clean_text"].iloc[0]
        assert "tristeza" in result["clean_text"].iloc[1]

    def test_lowercases_output(self) -> None:
        """Output text is fully lowercased."""
        result = normalize_spanish_tweets(["HOLA MUNDO"])
        assert result["clean_text"].iloc[0] == "hola mundo"

    def test_collapses_whitespace(self) -> None:
        """Multiple spaces are collapsed to single space."""
        result = normalize_spanish_tweets(["Hola    mundo   cruel"])
        assert result["clean_text"].iloc[0] == "hola mundo cruel"

    def test_accepts_pandas_series(self) -> None:
        """Accepts pd.Series input."""
        series = pd.Series(["Tweet 1", "Tweet 2"])
        result = normalize_spanish_tweets(series)
        assert len(result) == 2
        assert "clean_text" in result.columns

    def test_original_text_preserved(self) -> None:
        """original_text column holds the raw input unchanged."""
        raw = ["Tweet Original!"]
        result = normalize_spanish_tweets(raw)
        assert result["original_text"].iloc[0] == "Tweet Original!"

    def test_empty_input(self) -> None:
        """Empty list returns empty DataFrame with correct columns."""
        result = normalize_spanish_tweets([])
        assert len(result) == 0
        assert list(result.columns) == ["original_text", "clean_text"]
