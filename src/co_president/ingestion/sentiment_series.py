"""SPEC-39: Weekly sentiment aggregation from BERT and LLM outputs.

Combines emotion probabilities from ``bert_sentiment.classify_emotions`` and
sentiment labels from ``llm_sentiment.classify_with_gpt`` into weekly
candidate-level time series suitable for downstream modelling.

The series DataFrame can be joined with polling data for multi-signal
forecasting (roadmap §6.2).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["sentiment_series"]

logger = logging.getLogger(__name__)


def _make_label_fn(label: str) -> Callable[[pd.Series], float]:
    """Return a Series → float function computing label proportion."""

    def _prop(s: pd.Series) -> float:
        return float((s == label).mean())

    return _prop


def _infer_candidate(text: str, candidate_names: dict[str, list[str]]) -> str | None:
    """Heuristically determine which candidate a tweet mentions.

    Args:
        text: Tweet text (lowercased).
        candidate_names: Map from candidate key to list of known name
            variants (e.g. ``"petro": ["petro", "gustavo petro"]``).

    Returns:
        Candidate key or ``None`` if no known candidate is detected.

    """
    text_lower = text.lower()
    for candidate, names in candidate_names.items():
        for name in names:
            if name.lower() in text_lower:
                return candidate
    return None


def sentiment_series(
    bert_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    *,
    tweet_texts: pd.Series | None = None,
    timestamps: pd.Series | None = None,
    candidate_names: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Aggregate per-tweet emotion and sentiment into weekly candidate series.

    Args:
        bert_df: Output of ``classify_emotions`` — DataFrame with emotion
            probability columns (``joy``, ``fear``, ``disgust``,
            ``sadness``).  Rows correspond one-to-one with ``llm_df`` rows.
        llm_df: Output of ``classify_with_gpt`` — DataFrame with ``label``
            and ``explanation`` columns.
        tweet_texts: Optional ``Series`` of tweet texts used for candidate
            mention detection.  Required when no ``candidate_names``
            heuristic is possible.
        timestamps: Optional ``Series`` of tweet timestamps
            (``pd.Timestamp`` compatible).  If omitted, all tweets are
            treated as one week.
        candidate_names: Optional map from candidate key to name variants
            for heuristic candidate detection.  If ``None``, no
            candidate-level breakdown is performed.

    Returns:
        DataFrame with columns:
        - ``candidate`` — candidate key (or ``"all"``).
        - ``week`` — ISO week string (``YYYY-Www``).
        - ``joy_prop``, ``fear_prop``, ``disgust_prop``, ``sadness_prop``
          — mean probability per emotion for the candidate-week.
        - ``positive_prop``, ``negative_prop``, ``neutral_prop``,
          ``irrelevant_prop`` — fraction of LLM labels per candidate-week.
        - ``n_tweets`` — number of tweets in the group.

    Raises:
        ValueError: If ``bert_df`` and ``llm_df`` have different lengths.

    """
    if len(bert_df) != len(llm_df):
        msg = (
            f"bert_df ({len(bert_df)} rows) and llm_df ({len(llm_df)} rows) "
            "must have the same length"
        )
        raise ValueError(msg)

    # Build the base frame
    base = pd.concat(
        [
            bert_df[["joy", "fear", "disgust", "sadness"]].reset_index(drop=True),
            llm_df[["label"]].reset_index(drop=True),
        ],
        axis=1,
    )

    # Candidate attribution
    if candidate_names is not None and tweet_texts is not None:
        texts = tweet_texts.reset_index(drop=True)
        base["candidate"] = [_infer_candidate(t, candidate_names) for t in texts]
        base["candidate"] = base["candidate"].fillna("all")
    else:
        base["candidate"] = "all"

    # Week assignment
    if timestamps is not None:
        iso = pd.to_datetime(timestamps.reset_index(drop=True)).dt.isocalendar()
        base["week"] = iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)
    else:
        base["week"] = "2026-W01"

    emotion_cols = ["joy", "fear", "disgust", "sadness"]

    # Aggregate
    grouped = base.groupby(["candidate", "week"], as_index=False)
    aggregated = grouped.agg(
        **{f"{col}_prop": (col, "mean") for col in emotion_cols},
        **{
            f"{label}_prop": ("label", _make_label_fn(label))
            for label in ["positive", "negative", "neutral", "irrelevant"]
        },
        n_tweets=("label", "count"),
    )

    logger.info(
        "Aggregated %d tweets into %d candidate-week series rows",
        len(base),
        len(aggregated),
    )
    return aggregated
