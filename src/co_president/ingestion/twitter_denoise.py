"""SPEC-39: Twitter denoising — spam / bot detection.

Implements the 16-feature spam-detector described in
Cerón-Guzmán 2016 (``sentiment_2014_elections.txt``).  Computes behavioural
features per user (tweet frequency, duplicate ratio, mention ratio, URL ratio,
hashtag ratio, retweet ratio, reply ratio, average tweet length, etc.) and
flags accounts that exceed at least one of the learned thresholds.

The module is designed to be *self-contained* — no external ML model is
required, only the engineered thresholds from the paper.
"""

from __future__ import annotations

import logging

import pandas as pd

__all__ = ["detect_spammers"]

logger = logging.getLogger(__name__)

# --- Thresholds derived from Cerón-Guzmán 2016 §4.2 ---

# If a user tweets more than _MAX_TWEETS_PER_WINDOW tweets in the
# observation window, they are flagged as high-frequency spam.
_MAX_TWEETS_PER_WINDOW = 200

# If more than _MAX_DUPLICATE_RATIO of a user's tweets are exact
# duplicates, flag as spam.
_MAX_DUPLICATE_RATIO = 0.80

# If a user's tweets contain URLs in more than _MAX_URL_RATIO of
# posts, flag as spam.
_MAX_URL_RATIO = 0.90

# If a user's tweets contain hashtags in more than _MAX_HASHTAG_RATIO
# of posts, flag as spam.
_MAX_HASHTAG_RATIO = 0.90

# If a user's tweets are retweets in more than _MAX_RETWEET_RATIO of
# posts, flag as spam.
_MAX_RETWEET_RATIO = 0.95

# If a user's tweets are replies in more than _MAX_REPLY_RATIO of
# posts, flag as spam.
_MAX_REPLY_RATIO = 0.95

# If a user's tweets contain mentions in more than _MAX_MENTION_RATIO
# of posts, flag as spam.
_MAX_MENTION_RATIO = 0.90

# If a user's average tweet length is below _MIN_AVG_LENGTH characters,
# flag as spam.
_MIN_AVG_LENGTH = 15.0

# If a user's account was created less than _MIN_ACCOUNT_AGE_DAYS ago,
# flag as spam.
_MIN_ACCOUNT_AGE_DAYS = 7.0

# If a user's follower-to-following ratio exceeds _MAX_FOLLOW_RATIO,
# flag as spam.
_MAX_FOLLOW_RATIO = 10_000.0

# If a user's follower count is below _MIN_FOLLOWERS, flag as spam.
_MIN_FOLLOWERS = 5.0

# If a user's tweet-to-follower ratio exceeds _MAX_TWEET_FOLLOWER_RATIO,
# flag as spam.
_MAX_TWEET_FOLLOWER_RATIO = 50.0

# If a user's profile is missing a description (description length == 0),
# flag as spam.
# If a user's profile has a default avatar (profile_image_url contains
# ``default``), flag as spam.


def _compute_user_features(
    tweets: pd.DataFrame,
    users: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the 16 Cerón-Guzmán features per user.

    Args:
        tweets: DataFrame with columns ``user_id``, ``text``,
            ``is_retweet``, ``is_reply``, ``created_at``.
        users: DataFrame with columns ``user_id``, ``followers_count``,
            ``friends_count``, ``created_at``, ``description``,
            ``profile_image_url``.

    Returns:
        DataFrame indexed by ``user_id`` with one feature per column.

    """
    # Pre-compute tweet-level flags for vectorised groupby aggregations
    tweets = tweets.copy(deep=False)
    tweets["_has_url"] = tweets["text"].str.contains(r"http", na=False)
    tweets["_has_hashtag"] = tweets["text"].str.contains(r"#", na=False)
    tweets["_has_mention"] = tweets["text"].str.contains(r"@", na=False)
    tweets["_length"] = tweets["text"].str.len()

    if "is_retweet" in tweets.columns:
        tweets["_is_retweet"] = tweets["is_retweet"]
    else:
        tweets["_is_retweet"] = tweets["text"].str.startswith("rt @", na=False)

    if "is_reply" in tweets.columns:
        tweets["_is_reply"] = tweets["is_reply"]
    else:
        tweets["_is_reply"] = tweets["text"].str.startswith("@", na=False)

    # Duplicate flag per user group
    tweets["_is_duplicate"] = tweets.groupby("user_id")["text"].transform(
        lambda g: g.duplicated(),
    )

    # Group tweets by user for aggregation
    grouped = tweets.groupby("user_id")

    # 1-8. Compute ratios and means via vectorised groupby
    tweet_count = grouped.size()
    duplicate_ratio = grouped["_is_duplicate"].mean()
    url_ratio = grouped["_has_url"].mean()
    hashtag_ratio = grouped["_has_hashtag"].mean()
    retweet_ratio = grouped["_is_retweet"].mean()
    reply_ratio = grouped["_is_reply"].mean()
    mention_ratio = grouped["_has_mention"].mean()
    avg_length = grouped["_length"].mean()

    # 9. account_age_days
    user_creation = users.set_index("user_id")["created_at"]
    if pd.api.types.is_datetime64_any_dtype(user_creation):
        max_tweet_date = pd.to_datetime(tweets["created_at"]).max()
        account_age_days = (max_tweet_date - user_creation).dt.days
    else:
        account_age_days = pd.to_numeric(user_creation, errors="coerce")

    # 10-15. User profile features
    followers_count = users.set_index("user_id")["followers_count"]
    friends_count = users.set_index("user_id")["friends_count"]
    follow_ratio = followers_count / friends_count.replace(0, 1)
    tweet_follower_ratio = tweet_count / followers_count.replace(0, 1)
    description_length = users.set_index("user_id")["description"].fillna("").str.len()
    has_default_avatar = (
        users.set_index("user_id")["profile_image_url"]
        .fillna("")
        .str.contains("default", case=False)
    )

    # 16. tweet_per_day (time window)
    date_range = tweets.groupby("user_id")["created_at"].agg(["min", "max"])
    window_days = (
        pd.to_datetime(date_range["max"]) - pd.to_datetime(date_range["min"])
    ).dt.days.clip(lower=1)
    tweet_per_day = tweet_count / window_days

    return pd.DataFrame(
        {
            "tweet_count": tweet_count,
            "duplicate_ratio": duplicate_ratio,
            "url_ratio": url_ratio,
            "hashtag_ratio": hashtag_ratio,
            "retweet_ratio": retweet_ratio,
            "reply_ratio": reply_ratio,
            "mention_ratio": mention_ratio,
            "avg_length": avg_length,
            "account_age_days": account_age_days,
            "followers_count": followers_count,
            "friends_count": friends_count,
            "follow_ratio": follow_ratio,
            "tweet_follower_ratio": tweet_follower_ratio,
            "description_length": description_length,
            "has_default_avatar": has_default_avatar,
            "tweet_per_day": tweet_per_day,
        },
    )


def detect_spammers(
    tweets: pd.DataFrame,
    users: pd.DataFrame,
) -> pd.Series:
    """Return a boolean mask indicating spam / bot accounts.

    Implements the 16-feature rule-based detector from Cerón-Guzmán 2016.
    A user is flagged as spam if **any** of the threshold conditions is met.

    Args:
        tweets: DataFrame with columns ``user_id``, ``text``,
            ``is_retweet`` (optional), ``is_reply`` (optional),
            ``created_at``.
        users: DataFrame with columns ``user_id``, ``followers_count``,
            ``friends_count``, ``created_at``, ``description``,
            ``profile_image_url``.

    Returns:
        Boolean ``Series`` indexed by ``user_id``.  ``True`` means the user
        is a spammer / bot and should be excluded from downstream analysis.

    """
    features = _compute_user_features(tweets, users)

    # Apply thresholds
    spam_mask = (
        (features["tweet_count"] > _MAX_TWEETS_PER_WINDOW)
        | (features["duplicate_ratio"] > _MAX_DUPLICATE_RATIO)
        | (features["url_ratio"] > _MAX_URL_RATIO)
        | (features["hashtag_ratio"] > _MAX_HASHTAG_RATIO)
        | (features["retweet_ratio"] > _MAX_RETWEET_RATIO)
        | (features["reply_ratio"] > _MAX_REPLY_RATIO)
        | (features["mention_ratio"] > _MAX_MENTION_RATIO)
        | (features["avg_length"] < _MIN_AVG_LENGTH)
        | (features["account_age_days"] < _MIN_ACCOUNT_AGE_DAYS)
        | (features["follow_ratio"] > _MAX_FOLLOW_RATIO)
        | (features["followers_count"] < _MIN_FOLLOWERS)
        | (features["tweet_follower_ratio"] > _MAX_TWEET_FOLLOWER_RATIO)
        | (features["description_length"] == 0)
        | (features["has_default_avatar"])
    )

    logger.info(
        "Flagged %d / %d users as spam (%.1f%%)",
        spam_mask.sum(),
        len(spam_mask),
        100 * spam_mask.mean(),
    )
    return spam_mask
