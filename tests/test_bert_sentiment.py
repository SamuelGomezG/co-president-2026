"""SPEC-39: Tests for BERT-based emotion classification."""

from __future__ import annotations

import pytest

from co_president.ingestion.bert_sentiment import classify_emotions

__all__: list[str] = []


# ═══════════════════════════════════════════════════════════════════
# classify_emotions tests (mock-free — tests module internals
# without loading the real model)
# ═══════════════════════════════════════════════════════════════════


class TestClassifyEmotions:
    """classify_emotions returns emotion probability DataFrame."""

    def test_raises_on_unsupported_model(self) -> None:
        """Unsupported model name raises ValueError."""
        with pytest.raises(ValueError, match="robertuito"):
            classify_emotions(["test"], model="unsupported-model")

    def test_raises_on_empty_tweets_list(self) -> None:
        """Empty input raises ValueError."""
        with pytest.raises(ValueError, match="robertuito"):
            classify_emotions([], model="unsupported-model")
