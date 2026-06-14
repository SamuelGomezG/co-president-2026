"""SPEC-39: Tests for weekly sentiment aggregation."""

from __future__ import annotations

import pandas as pd
import pytest

from co_president.ingestion.sentiment_series import sentiment_series

__all__: list[str] = []


# ═══════════════════════════════════════════════════════════════════
# Synthetic-data helpers
# ═══════════════════════════════════════════════════════════════════


def _make_bert_df(n: int) -> pd.DataFrame:
    """Build synthetic emotion probability DataFrame."""
    return pd.DataFrame(
        {
            "joy": [0.8 if i % 2 == 0 else 0.2 for i in range(n)],
            "fear": [0.1 for _ in range(n)],
            "disgust": [0.05 for _ in range(n)],
            "sadness": [0.05 for _ in range(n)],
        },
    )


def _make_llm_df(n: int) -> pd.DataFrame:
    """Build synthetic LLM label DataFrame."""
    labels: list[str] = []
    for i in range(n):
        if i % 3 == 0:
            labels.append("positive")
        elif i % 3 == 1:
            labels.append("negative")
        else:
            labels.append("neutral")
    return pd.DataFrame({"label": labels, "explanation": [""] * n})


# ═══════════════════════════════════════════════════════════════════
# Sentiment series tests
# ═══════════════════════════════════════════════════════════════════


class TestSentimentSeries:
    """sentiment_series aggregates per-tweet emotions into weekly series."""

    def test_returns_dataframe(self) -> None:
        """Return type is a DataFrame."""
        bert = _make_bert_df(6)
        llm = _make_llm_df(6)
        result = sentiment_series(bert, llm)
        assert isinstance(result, pd.DataFrame)

    def test_default_columns_present(self) -> None:
        """Default output has expected aggregation columns."""
        bert = _make_bert_df(6)
        llm = _make_llm_df(6)
        result = sentiment_series(bert, llm)
        expected = {
            "candidate",
            "week",
            "joy_prop",
            "fear_prop",
            "disgust_prop",
            "sadness_prop",
            "positive_prop",
            "negative_prop",
            "neutral_prop",
            "irrelevant_prop",
            "n_tweets",
        }
        assert expected.issubset(set(result.columns))

    def test_single_week_when_no_timestamps(self) -> None:
        """Without timestamps, all tweets grouped into '2026-W01'."""
        bert = _make_bert_df(6)
        llm = _make_llm_df(6)
        result = sentiment_series(bert, llm)
        assert (result["week"] == "2026-W01").all()

    def test_candidate_all_when_no_names(self) -> None:
        """Without candidate_names, all rows have candidate='all'."""
        bert = _make_bert_df(6)
        llm = _make_llm_df(6)
        result = sentiment_series(bert, llm)
        assert (result["candidate"] == "all").all()

    def test_raises_on_length_mismatch(self) -> None:
        """Different length bert_df and llm_df raises ValueError."""
        bert = _make_bert_df(5)
        llm = _make_llm_df(3)
        with pytest.raises(ValueError, match="same length"):
            sentiment_series(bert, llm)

    def test_emotion_proportions_in_unit_interval(self) -> None:
        """Emotion proportion columns contain values in [0, 1]."""
        bert = _make_bert_df(12)
        llm = _make_llm_df(12)
        result = sentiment_series(bert, llm)
        for col in ["joy_prop", "fear_prop", "disgust_prop", "sadness_prop"]:
            assert result[col].between(0, 1).all()

    def test_label_proportions_sum_to_one(self) -> None:
        """Label proportion columns sum to ~1 per row."""
        bert = _make_bert_df(12)
        llm = _make_llm_df(12)
        result = sentiment_series(bert, llm)
        label_cols = [
            "positive_prop",
            "negative_prop",
            "neutral_prop",
            "irrelevant_prop",
        ]
        assert result[label_cols].sum(axis=1).between(0.99, 1.01).all()

    def test_n_tweets_is_positive(self) -> None:
        """n_tweets is > 0 for all rows."""
        bert = _make_bert_df(12)
        llm = _make_llm_df(12)
        result = sentiment_series(bert, llm)
        assert (result["n_tweets"] > 0).all()

    def test_weekly_partitioning(self) -> None:
        """Timestamps are correctly grouped by ISO week."""
        bert = _make_bert_df(4)
        llm = _make_llm_df(4)
        timestamps = pd.Series(
            [
                pd.Timestamp("2026-01-01"),
                pd.Timestamp("2026-01-05"),
                pd.Timestamp("2026-01-12"),
                pd.Timestamp("2026-01-19"),
            ],
        )
        result = sentiment_series(bert, llm, timestamps=timestamps)
        expected_weeks = {"2026-W01", "2026-W02", "2026-W03", "2026-W04"}
        assert set(result["week"]) == expected_weeks

    def test_candidate_attribution(self) -> None:
        """Candidate names are correctly extracted from tweet text."""
        bert = _make_bert_df(4)
        llm = _make_llm_df(4)
        tweet_texts = pd.Series(
            [
                "Apoyo a Petro presidente",
                "Voy con Fico",
                "Noticia cualquiera",
                "Petro es el mejor",
            ],
        )
        candidate_names = {
            "petro": ["petro", "gustavo petro"],
            "fico": ["fico", "federico gutiérrez"],
        }
        result = sentiment_series(
            bert,
            llm,
            tweet_texts=tweet_texts,
            candidate_names=candidate_names,
        )
        candidates = set(result["candidate"])
        assert "petro" in candidates
        assert "fico" in candidates
        assert "all" in candidates  # the unknown tweet
