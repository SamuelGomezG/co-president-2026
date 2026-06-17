"""SPEC-39: BERT-based emotion classification for Spanish tweets.

Wraps the RoBERTuito model (``pysentimiento/robertuito-base-emotion``) from
HuggingFace to classify tweets into four basic emotions: joy, fear, disgust,
and sadness.  Targets F1 ≥ 0.65 on the arXiv 2407.07258 test set.

The module uses the ``pysentimiento`` convenience API when available and falls
back to raw ``transformers`` if the ``pysentimiento`` model does not exist.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence


__all__ = ["classify_emotions"]

logger = logging.getLogger(__name__)

_EMOTION_LABELS: list[str] = ["joy", "fear", "disgust", "sadness"]


def _classify_with_pysentimiento(tweets: list[str]) -> pd.DataFrame:
    """Classify with pysentimiento's ``create_analyzer``."""
    try:
        from pysentimiento import (  # type: ignore[reportMissingTypeStubs]  # noqa: PLC0415
            create_analyzer,  # type: ignore[reportUnknownVariableType]
        )
    except ImportError:
        msg = "pysentimiento not available"
        raise RuntimeError(msg) from None
    analyzer = create_analyzer(task="emotion", lang="es")
    results: list[dict[str, Any]] = []
    for tweet in tweets:
        probas: dict[str, float] = dict.fromkeys(_EMOTION_LABELS, 0.0)
        try:
            pred = analyzer.predict(tweet)  # type: ignore[reportUnknownVariableType]
            if hasattr(pred, "probas"):  # type: ignore[reportUnknownArgumentType]
                raw = pred.probas  # type: ignore[reportUnknownVariableType, union-attr]
                if isinstance(raw, dict):
                    probas = {k: float(v) for k, v in raw.items()}  # type: ignore[reportUnknownVariableType]
        except Exception:
            logger.exception(
                "pysentimiento failed for tweet, using zero probs", extra={"tweet_len": len(tweet)}
            )
        row = {label: probas.get(label, 0.0) for label in _EMOTION_LABELS}
        for extra in ("anger", "surprise"):
            if extra in probas:
                row[extra] = probas[extra]
        results.append(row)
    return pd.DataFrame(results, index=pd.RangeIndex(len(tweets)))


def _classify_with_transformers(tweets: list[str]) -> pd.DataFrame:
    """Classify with raw HuggingFace ``transformers`` pipeline."""
    try:
        from transformers import pipeline  # noqa: PLC0415
    except ImportError:
        msg = "transformers not available"
        raise RuntimeError(msg) from None
    classifier = pipeline(  # type: ignore[no-untyped-def]
        "text-classification",
        model="pysentimiento/robertuito-base-emotion",
        tokenizer="pysentimiento/robertuito-base-emotion",
        top_k=None,
    )
    results: list[dict[str, Any]] = []
    for tweet in tweets:
        probas: dict[str, float] = dict.fromkeys(_EMOTION_LABELS, 0.0)
        try:
            output = classifier(tweet)
            if isinstance(output, list):  # type: ignore[reportUnnecessaryIsInstance]
                probas = {item["label"].lower(): item["score"] for item in output}
            else:
                probas = {output["label"].lower(): output["score"]}  # type: ignore[reportIndexIssue]
        except Exception:
            logger.exception(
                "transformers failed for tweet, using zero probs", extra={"tweet_len": len(tweet)}
            )
        row = {label: probas.get(label, 0.0) for label in _EMOTION_LABELS}
        for extra in ("anger", "surprise"):
            if extra in probas:
                row[extra] = probas[extra]
        results.append(row)
    return pd.DataFrame(results, index=pd.RangeIndex(len(tweets)))


def classify_emotions(
    tweets: Sequence[str] | pd.Series,
    model: str = "robertuito",
) -> pd.DataFrame:
    """Classify tweets into emotion categories using a BERT-based model.

    Args:
        tweets: Iterable of clean tweet strings.
        model: Model name.  Currently only ``"robertuito"`` is supported.

    Returns:
        DataFrame with emotion columns (``joy``, ``fear``, ``disgust``,
        ``sadness``) and optional ``anger`` / ``surprise`` if the model
        outputs them.  Each value is a probability in [0, 1].

    Raises:
        ValueError: If an unsupported model name is provided.

    """
    supported = {"robertuito"}
    if model not in supported:
        msg = f"Unsupported model '{model}'; choose from {supported}"
        raise ValueError(msg)

    input_list: list[str] = tweets.tolist() if isinstance(tweets, pd.Series) else list(tweets)

    try:
        logger.info("Attempting pysentimiento classifier for %d tweets", len(input_list))
        result = _classify_with_pysentimiento(input_list)
    except ImportError:
        logger.info("pysentimiento not installed, falling back to transformers")
        result = _classify_with_transformers(input_list)
    except (RuntimeError, ValueError) as exc:
        logger.warning("pysentimiento failed (%s), falling back to transformers", exc)
        result = _classify_with_transformers(input_list)

    # Ensure at least the 4 base emotion columns exist
    for label in _EMOTION_LABELS:
        if label not in result.columns:
            result[label] = 0.0

    logger.info(
        "Classified %d tweets into %d emotion categories", len(input_list), len(result.columns)
    )
    return result[_EMOTION_LABELS + [c for c in result.columns if c not in _EMOTION_LABELS]]
