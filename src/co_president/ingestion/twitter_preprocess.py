"""SPEC-39: Twitter preprocessing — Spanish tweet normalization.

Normalises raw Spanish tweets using finite-state transducer patterns adapted
from Cerón-Guzmán 2016 (``sentiment_2014_elections.txt``) and the
``pysentimiento`` preprocessor.  Handles URLs, mentions, hashtags,
repeated characters, laughter patterns, and emoji transcription.

Returns a DataFrame with ``clean_text`` and ``original_text`` columns.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["normalize_spanish_tweets"]

logger = logging.getLogger(__name__)

# --- Regex constants (Cerón-Guzmán 2016 finite-state transducers) ---

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#\w+")
_REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")
_LAUGHTER_RE = re.compile(r"\b(j|a|h|e){3,}\b", re.IGNORECASE)

# Repeated punctuation (e.g. "!!!" -> "!")
_REPEAT_PUNCT_RE = re.compile(r"([!?])\1+")

# All-caps words of 3+ chars (e.g. "AMAZING" -> "amazing")
_ALL_CAPS_RE = re.compile(r"\b[A-ZÁÉÍÓÚÑ]{3,}\b")

# Digits
_DIGIT_RE = re.compile(r"\d+")

# Common emoticons (expanded to text tokens)
_EMOTICONS: dict[str, str] = {
    ":)": "sonrisa",
    ":-)": "sonrisa",
    ":(": "tristeza",
    ":-": "tristeza",
    ":d": "risa",
    ":-d": "risa",
    "xd": "risa",
    "xdd": "risa",
    "<3": "corazón",
    "</3": "corazón_roto",
}


def _replace_emoticons(text: str) -> str:
    """Replace ASCII emoticons with Spanish text tokens."""
    for emoticon, token in _EMOTICONS.items():
        text = text.replace(emoticon, f" {token} ")
    return text


def _remove_urls(text: str) -> str:
    """Replace URLs with a generic ``url`` token."""
    return _URL_RE.sub(" url ", text)


def _remove_mentions(text: str) -> str:
    """Replace mentions with a generic ``usuario`` token."""
    return _MENTION_RE.sub(" usuario ", text)


def _split_hashtags(text: str) -> str:
    """Split CamelCase hashtags into separate words."""
    return _HASHTAG_RE.sub(
        lambda m: " " + _camel_case_split(m.group(0)[1:]) + " ",
        text,
    )


def _camel_case_split(word: str) -> str:
    """Split a CamelCase word into space-separated tokens."""
    parts: list[str] = []
    current: list[str] = []
    for char in word:
        if char.isupper() and current:
            parts.append("".join(current).lower())
            current = [char.lower()]
        else:
            current.append(char.lower())
    if current:
        parts.append("".join(current).lower())
    return " ".join(parts)


def _shorten_repeated_chars(text: str) -> str:
    """Shorten runs of 3+ identical characters to 2 (e.g. "naaaa" -> "naa")."""
    return _REPEAT_CHAR_RE.sub(r"\1\1", text)


def _normalize_laughter(text: str) -> str:
    """Normalize laughter patterns to ``jaja``."""
    return _LAUGHTER_RE.sub("jaja", text)


def _normalize_punctuation(text: str) -> str:
    """Reduce repeated punctuation to a single mark."""
    return _REPEAT_PUNCT_RE.sub(r"\1", text)


def _normalize_all_caps(text: str) -> str:
    """Lowercase all-caps words of 3+ characters."""

    def _lower(match: re.Match[str]) -> str:
        return match.group(0).lower()

    return _ALL_CAPS_RE.sub(_lower, text)


def _remove_digits(text: str) -> str:
    """Replace digit sequences with ``numero`` token."""
    return _DIGIT_RE.sub(" numero ", text)


def _normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters to a single space."""
    return " ".join(text.split())


def _normalize_text(text: str) -> str:
    """Apply the full Cerón-Guzmán / pysentimiento normalization pipeline.

    Order matters: emoticons first, then structural tokens, then
    orthographic corrections, finally whitespace cleanup.
    """
    text = text.lower()
    text = _replace_emoticons(text)
    text = _remove_urls(text)
    text = _remove_mentions(text)
    text = _split_hashtags(text)
    text = _shorten_repeated_chars(text)
    text = _normalize_laughter(text)
    text = _normalize_punctuation(text)
    text = _normalize_all_caps(text)
    text = _remove_digits(text)
    text = _normalize_whitespace(text)
    return text.strip()


def normalize_spanish_tweets(
    tweets: Sequence[str] | pd.Series,
) -> pd.DataFrame:
    """Normalise a collection of Spanish tweets.

    Args:
        tweets: Iterable of raw tweet strings or a pandas ``Series``.

    Returns:
        DataFrame with two columns:
        - ``original_text`` — the raw tweet.
        - ``clean_text`` — the normalised tweet.

    """
    raw = tweets.tolist() if isinstance(tweets, pd.Series) else list(tweets)

    cleaned = [_normalize_text(t) for t in raw]

    logger.info("Normalised %d tweets", len(raw))
    return pd.DataFrame(
        {"original_text": raw, "clean_text": cleaned},
    )
