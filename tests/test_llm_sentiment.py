"""SPEC-39: Tests for LLM-based sentiment classification."""

from __future__ import annotations

import pandas as pd

from co_president.ingestion.llm_sentiment import classify_with_gpt

__all__: list[str] = []


# ═══════════════════════════════════════════════════════════════════
# classify_with_gpt tests (uses mock classifier when no API key)
# ═══════════════════════════════════════════════════════════════════


class TestClassifyWithGpt:
    """classify_with_gpt returns sentiment label DataFrame."""

    def test_returns_dataframe_with_expected_columns(self) -> None:
        """Output has label and explanation columns."""
        result = classify_with_gpt(["Un tweet cualquiera"])
        assert list(result.columns) == ["label", "explanation"]

    def test_returns_expected_number_of_rows(self) -> None:
        """Output row count matches input."""
        tweets = ["Tweet 1", "Tweet 2", "Tweet 3"]
        result = classify_with_gpt(tweets)
        assert len(result) == 3

    def test_label_is_valid(self) -> None:
        """Every label is one of the four valid categories."""
        tweets = ["Buen candidato", "Malo", "Qué hora es?", "Compré pan"]
        result = classify_with_gpt(tweets)
        valid = {"positive", "negative", "neutral", "irrelevant"}
        assert result["label"].isin(valid).all()

    def test_positive_keyword_detected(self) -> None:
        """Tweets with 'bueno' keywords get positive label."""
        result = classify_with_gpt(["Qué bueno este candidato"])
        assert result["label"].iloc[0] == "positive"

    def test_negative_keyword_detected(self) -> None:
        """Tweets with 'odio' keywords get negative label."""
        result = classify_with_gpt(["Odio esta propuesta"])
        assert result["label"].iloc[0] == "negative"

    def test_neutral_keyword_detected(self) -> None:
        """Neutral tweets get neutral label."""
        result = classify_with_gpt(["La reunión es a las 3"])
        assert result["label"].iloc[0] == "neutral"

    def test_accepts_pandas_series(self) -> None:
        """Accepts pd.Series input."""
        series = pd.Series(["Tweet uno", "Tweet dos"])
        result = classify_with_gpt(series)
        assert len(result) == 2

    def test_empty_input(self) -> None:
        """Empty list returns empty DataFrame with correct columns."""
        result = classify_with_gpt([])
        assert len(result) == 0
        assert list(result.columns) == ["label", "explanation"]
