"""SPEC-39: LLM-based sentiment analysis for Spanish tweets.

Wraps OpenAI's GPT-4o-mini (or compatible API) to classify Colombian
electoral tweets into sentiment labels with free-text explanations.

The prompt is designed to capture Colombia-specific political sentiment
including candidate references, policy positions, and regional context.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

    from openai import OpenAI as _OpenAIClient


__all__ = ["classify_with_gpt"]

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT: str = (
    "Eres un analista de sentimiento político colombiano. "
    "Clasifica cada tweet en UNA de estas categorías:\n"
    "- POSITIVO: apoyo, entusiasmo, respaldo a un candidato o propuesta\n"
    "- NEGATIVO: rechazo, crítica, oposición a un candidato o propuesta\n"
    "- NEUTRO: informativo, pregunta, sin carga emocional clara\n"
    "- IRRELEVANTE: no relacionado con política colombiana o elecciones\n\n"
    'Responde SOLO con JSON: {"label": "CATEGORÍA", "explanation": "razón breve en español"}'
)

_USER_TEMPLATE: str = "Tweet: {tweet}"

_LABEL_MAP: dict[str, str] = {
    "POSITIVO": "positive",
    "NEGATIVO": "negative",
    "NEUTRO": "neutral",
    "IRRELEVANTE": "irrelevant",
}


def _call_openai(
    tweets: list[str],
    model: str,
    client: _OpenAIClient,  # type: ignore[valid-type]
) -> list[dict[str, str]]:
    """Call OpenAI API for batch sentiment classification.

    Args:
        tweets: List of tweet texts.
        model: OpenAI model name.
        client: OpenAI client instance.

    Returns:
        List of dicts with ``label`` and ``explanation`` keys.

    """
    results: list[dict[str, str]] = []
    for tweet in tweets:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": _USER_TEMPLATE.format(tweet=tweet)},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=100,
            )
            content = response.choices[0].message.content
            if content is None:
                content = "{}"
            parsed = json.loads(content)
            raw_label = parsed.get("label", "NEUTRO").upper()
            label = _LABEL_MAP.get(raw_label, "neutral")
            explanation = parsed.get("explanation", "")
        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenAI API call failed for tweet: %s", exc)
            label = "neutral"
            explanation = ""

        results.append({"label": label, "explanation": explanation})

    return results


def _mock_classifier(tweets: list[str]) -> list[dict[str, str]]:
    """Return mock results when no API key is available.

    Useful for testing and development without incurring API costs.
    """
    results: list[dict[str, str]] = []
    for tweet in tweets:
        tweet_lower = tweet.lower()
        if any(word in tweet_lower for word in ("malo", "odio", "terrible", "pésimo")):
            label = "negative"
            explanation = "Lenguaje negativo detectado"
        elif any(word in tweet_lower for word in ("bueno", "excelente", "apoyo", "gran")):
            label = "positive"
            explanation = "Lenguaje positivo detectado"
        else:
            label = "neutral"
            explanation = "No se detectó carga emocional clara"
        results.append({"label": label, "explanation": explanation})
    return results


def classify_with_gpt(
    tweets: Sequence[str] | pd.Series,
    model: str = "gpt-4o-mini",
) -> pd.DataFrame:
    """Classify tweets into sentiment categories using an LLM.

    Args:
        tweets: Iterable of tweet strings to classify.
        model: OpenAI model identifier.  Defaults to ``"gpt-4o-mini"``.

    Returns:
        DataFrame with columns:
        - ``label``: one of ``positive``, ``negative``, ``neutral``,
          ``irrelevant``.
        - ``explanation``: brief Spanish explanation of the classification.

    """
    input_list: list[str] = tweets.tolist() if isinstance(tweets, pd.Series) else list(tweets)

    if not input_list:
        return pd.DataFrame({"label": [], "explanation": []})

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI as _Client  # noqa: PLC0415

            client = _Client(api_key=api_key)
            results = _call_openai(input_list, model, client)
        except ImportError:
            logger.info("openai package not installed, using mock classifier")
            results = _mock_classifier(input_list)
        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenAI initialisation failed (%s), using mock classifier", exc)
            results = _mock_classifier(input_list)
    else:
        logger.info("OPENAI_API_KEY not set, using mock classifier")
        results = _mock_classifier(input_list)

    logger.info(
        "Classified %d tweets with %s",
        len(input_list),
        "GPT" if api_key else "mock",
    )
    return pd.DataFrame(results, index=pd.RangeIndex(len(input_list)))
