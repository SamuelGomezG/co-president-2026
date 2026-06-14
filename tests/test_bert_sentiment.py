"""SPEC-39: Tests for BERT-based emotion classification."""

from __future__ import annotations

import pandas as pd
import pytest

from co_president.ingestion.bert_sentiment import _EMOTION_LABELS, classify_emotions

__all__: list[str] = []


# ═══════════════════════════════════════════════════════════════════
# classify_emotions tests
# ═══════════════════════════════════════════════════════════════════


class TestClassifyEmotions:
    """classify_emotions returns emotion probability DataFrame."""

    def test_raises_on_unsupported_model(self) -> None:
        """Unsupported model name raises ValueError."""
        with pytest.raises(ValueError, match="robertuito"):
            classify_emotions(["test"], model="unsupported-model")

    def test_valid_input_returns_dataframe_with_expected_columns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Valid input returns DataFrame with emotion probability columns."""
        mock_result = pd.DataFrame(
            {k: [v] for k, v in zip(_EMOTION_LABELS, [0.8, 0.1, 0.05, 0.05], strict=True)}
        )
        captured: list[str] = []

        def _mock_classify(tweets: list[str]) -> pd.DataFrame:
            captured.extend(tweets)
            return mock_result

        monkeypatch.setattr(
            "co_president.ingestion.bert_sentiment._classify_with_pysentimiento",
            _mock_classify,
        )
        tweets = ["¡Qué feliz estoy!"]
        result = classify_emotions(tweets)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == _EMOTION_LABELS
        assert len(result) == 1
        for col in _EMOTION_LABELS:
            assert 0.0 <= result.loc[0, col] <= 1.0
        assert captured == tweets

    def test_empty_list_with_valid_model_returns_empty_dataframe(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty input with valid model returns empty DataFrame with correct columns."""
        mock_result = pd.DataFrame(columns=_EMOTION_LABELS)
        monkeypatch.setattr(
            "co_president.ingestion.bert_sentiment._classify_with_pysentimiento",
            lambda _: mock_result,
        )
        result = classify_emotions([])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == _EMOTION_LABELS
