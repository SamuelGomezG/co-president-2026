"""SPEC-39: Tests for Twitter spam / bot detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from co_president.ingestion.twitter_denoise import detect_spammers

__all__: list[str] = []


# ═══════════════════════════════════════════════════════════════════
# Synthetic-data helpers
# ═══════════════════════════════════════════════════════════════════


def _make_tweets(  # noqa: PLR0913
    n: int,
    user_ids: list[str] | None = None,
    *,
    url_ratio: float = 0.0,
    retweet_ratio: float = 0.0,
    reply_ratio: float = 0.0,
    mention_ratio: float = 0.0,
    hashtag_ratio: float = 0.0,
    avg_length: float = 50.0,
) -> pd.DataFrame:
    """Build synthetic tweets DataFrame with controlled spam features."""
    rng = np.random.default_rng(seed=42)
    if user_ids is None:
        user_ids = [f"u{i}" for i in range(min(n, 5))]
    uids: list[str] = []
    texts: list[str] = []
    is_retweets: list[bool] = []
    is_replies: list[bool] = []
    created_ats: list[pd.Timestamp] = []
    base_date = pd.Timestamp("2026-01-01")

    for i in range(n):
        uid = user_ids[i % len(user_ids)]
        uids.append(uid)
        prefix = ""
        if rng.random() < url_ratio:
            prefix += "http://x.com "
        if rng.random() < mention_ratio:
            prefix += "@someone "
        if rng.random() < hashtag_ratio:
            prefix += "#tag "
        is_retweets.append(rng.random() < retweet_ratio)
        is_replies.append(rng.random() < reply_ratio)
        texts.append(prefix + "x" * max(1, int(avg_length - len(prefix))))
        created_ats.append(base_date + pd.Timedelta(days=i))
    return pd.DataFrame(
        {
            "user_id": uids,
            "text": texts,
            "is_retweet": is_retweets,
            "is_reply": is_replies,
            "created_at": created_ats,
        },
    )


def _make_users(user_ids: list[str]) -> pd.DataFrame:
    """Build synthetic users DataFrame with default benign features."""
    base_date = pd.Timestamp("2025-01-01")
    return pd.DataFrame(
        {
            "user_id": user_ids,
            "followers_count": [100] * len(user_ids),
            "friends_count": [50] * len(user_ids),
            "created_at": [base_date] * len(user_ids),
            "description": ["Soy un usuario real"] * len(user_ids),
            "profile_image_url": ["https://example.com/avatar.jpg"] * len(user_ids),
        },
    )


# ═══════════════════════════════════════════════════════════════════
# Detection tests
# ═══════════════════════════════════════════════════════════════════


class TestDetectSpammers:
    """detect_spammers flags accounts based on 16 threshold features."""

    def test_returns_boolean_series(self) -> None:
        """Return type is a boolean Series indexed by user_id."""
        tweets = _make_tweets(10)
        users = _make_users(["u0", "u1", "u2", "u3", "u4"])
        result = detect_spammers(tweets, users)
        assert isinstance(result, pd.Series)
        assert result.dtype == bool
        assert result.index.name == "user_id"

    def test_benign_users_not_flagged(self) -> None:
        """Normal users are not flagged as spam."""
        tweets = _make_tweets(10, avg_length=50.0)
        users = _make_users(["u0", "u1", "u2", "u3", "u4"])
        result = detect_spammers(tweets, users)
        assert not result.any()

    def test_high_tweet_count_flagged(self) -> None:
        """Users with >200 tweets are flagged."""
        uid = "u0"
        tweets = _make_tweets(250, user_ids=[uid])
        users = _make_users([uid])
        result = detect_spammers(tweets, users)
        assert result[uid]

    def test_high_duplicate_ratio_flagged(self) -> None:
        """Users with >80% duplicates are flagged."""
        uid = "u0"
        tweets = pd.DataFrame(
            {
                "user_id": [uid] * 10,
                "text": ["mismo texto"] * 9 + ["otro texto"],
                "is_retweet": [False] * 10,
                "is_reply": [False] * 10,
                "created_at": pd.Timestamp("2026-01-01"),
            },
        )
        users = _make_users([uid])
        result = detect_spammers(tweets, users)
        assert result[uid]

    def test_high_url_ratio_flagged(self) -> None:
        """Users with >90% URLs are flagged."""
        uid = "u0"
        tweets = _make_tweets(10, user_ids=[uid], url_ratio=1.0)
        users = _make_users([uid])
        result = detect_spammers(tweets, users)
        assert result[uid]

    def test_high_hashtag_ratio_flagged(self) -> None:
        """Users with >90% hashtags are flagged."""
        uid = "u0"
        tweets = _make_tweets(10, user_ids=[uid], hashtag_ratio=1.0)
        users = _make_users([uid])
        result = detect_spammers(tweets, users)
        assert result[uid]

    def test_high_retweet_ratio_flagged(self) -> None:
        """Users with >95% retweets are flagged."""
        uid = "u0"
        tweets = _make_tweets(10, user_ids=[uid], retweet_ratio=1.0)
        users = _make_users([uid])
        result = detect_spammers(tweets, users)
        assert result[uid]

    def test_high_reply_ratio_flagged(self) -> None:
        """Users with >95% replies are flagged."""
        uid = "u0"
        tweets = _make_tweets(10, user_ids=[uid], reply_ratio=1.0)
        users = _make_users([uid])
        result = detect_spammers(tweets, users)
        assert result[uid]

    def test_high_mention_ratio_flagged(self) -> None:
        """Users with >90% mentions are flagged."""
        uid = "u0"
        tweets = _make_tweets(10, user_ids=[uid], mention_ratio=1.0)
        users = _make_users([uid])
        result = detect_spammers(tweets, users)
        assert result[uid]

    def test_low_avg_length_flagged(self) -> None:
        """Users avg tweet length <15 chars flagged."""
        uid = "u0"
        tweets = _make_tweets(10, user_ids=[uid], avg_length=5.0)
        users = _make_users([uid])
        result = detect_spammers(tweets, users)
        assert result[uid]

    def test_young_account_flagged(self) -> None:
        """Accounts <7 days old are flagged."""
        uid = "u0"
        tweets = _make_tweets(5, user_ids=[uid])
        recent_date = pd.Timestamp("2026-06-10")
        users = pd.DataFrame(
            {
                "user_id": [uid],
                "followers_count": [100],
                "friends_count": [50],
                "created_at": [recent_date],
                "description": ["Soy nuevo"],
                "profile_image_url": ["https://x.com/avatar.jpg"],
            },
        )
        result = detect_spammers(tweets, users)
        assert result[uid]

    def test_no_description_flagged(self) -> None:
        """Users with empty description are flagged."""
        uid = "u0"
        tweets = _make_tweets(5, user_ids=[uid])
        users = pd.DataFrame(
            {
                "user_id": [uid],
                "followers_count": [100],
                "friends_count": [50],
                "created_at": [pd.Timestamp("2025-01-01")],
                "description": [""],
                "profile_image_url": ["https://x.com/avatar.jpg"],
            },
        )
        result = detect_spammers(tweets, users)
        assert result[uid]

    def test_default_avatar_flagged(self) -> None:
        """Users with default avatar are flagged."""
        uid = "u0"
        tweets = _make_tweets(5, user_ids=[uid])
        users = pd.DataFrame(
            {
                "user_id": [uid],
                "followers_count": [100],
                "friends_count": [50],
                "created_at": [pd.Timestamp("2025-01-01")],
                "description": ["Usuario"],
                "profile_image_url": ["https://x.com/avatars/default.png"],
            },
        )
        result = detect_spammers(tweets, users)
        assert result[uid]

    def test_low_followers_flagged(self) -> None:
        """Users with <5 followers are flagged."""
        uid = "u0"
        tweets = _make_tweets(5, user_ids=[uid])
        users = pd.DataFrame(
            {
                "user_id": [uid],
                "followers_count": [1],
                "friends_count": [50],
                "created_at": [pd.Timestamp("2025-01-01")],
                "description": ["Usuario"],
                "profile_image_url": ["https://x.com/avatar.jpg"],
            },
        )
        result = detect_spammers(tweets, users)
        assert result[uid]
