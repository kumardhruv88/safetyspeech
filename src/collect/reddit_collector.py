"""
reddit_collector.py
-------------------
Collects Reddit posts using PRAW (Python Reddit API Wrapper).
Targets subreddits with high signal for toxic/depressive content.

Requires a .env file with:
    REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

Usage:
    python -m src.collect.reddit_collector
    python -m src.collect.reddit_collector --post_limit 500 --output data/raw/reddit_raw.csv
"""

import argparse
import logging
import os
import time

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Subreddits and their associated label hints
SUBREDDIT_CONFIG = {
    "depression": "depressive",
    "SuicideWatch": "depressive",
    "offmychest": "depressive",
    "rant": "hate_speech|violent",
    "news": "normal",
    "AskReddit": "normal",
    "teenagers": "normal",
    "anger": "violent",
    "TrueOffMyChest": "depressive",
    "mentalhealth": "depressive",
}


def get_reddit_client():
    """Initialize and return a PRAW Reddit client."""
    try:
        import praw
    except ImportError:
        logger.error("praw not installed. Run: pip install praw")
        return None

    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "guardian_nlp_collector/1.0")

    if not client_id or not client_secret:
        logger.error(
            "Reddit API credentials not found. "
            "Create a .env file with REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET. "
            "Get credentials at: https://www.reddit.com/prefs/apps"
        )
        return None

    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )


def collect_reddit_posts(
    subreddits: dict,
    post_limit: int = 1000,
    reddit=None,
) -> pd.DataFrame:
    """
    Scrapes top/hot posts from each subreddit.
    Returns a DataFrame with text and label hints.
    """
    if reddit is None:
        reddit = get_reddit_client()
    if reddit is None:
        return pd.DataFrame()

    posts = []
    for sub, label_hint in subreddits.items():
        logger.info(f"Collecting from r/{sub} (label: {label_hint})...")
        try:
            subreddit = reddit.subreddit(sub)
            post_iter = list(subreddit.hot(limit=post_limit // 2)) + \
                        list(subreddit.new(limit=post_limit // 2))

            for post in tqdm(post_iter, desc=f"r/{sub}"):
                # Combine title + selftext for richer signal
                full_text = f"{post.title}. {post.selftext}".strip()
                if len(full_text.split()) < 5:
                    continue
                posts.append({
                    "id": post.id,
                    "text": full_text,
                    "subreddit": sub,
                    "platform": "reddit",
                    "label_hint": label_hint,
                    "score": post.score,
                    "num_comments": post.num_comments,
                })
                # PRAW rate limit: 60 req/min → sleep
                time.sleep(0.02)

        except Exception as e:
            logger.warning(f"Error collecting from r/{sub}: {e}")
            continue

    df = pd.DataFrame(posts)
    if not df.empty:
        df = df.drop_duplicates(subset=["id"])
        logger.info(f"Collected {len(df)} unique Reddit posts total.")
    return df


def save_posts(df: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} Reddit posts → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="GUARDIAN-NLP: Reddit Data Collector")
    parser.add_argument("--post_limit", type=int, default=1000,
                        help="Number of posts to collect per subreddit")
    parser.add_argument("--output", type=str, default="data/raw/reddit_raw.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    df = collect_reddit_posts(SUBREDDIT_CONFIG, post_limit=args.post_limit)
    if not df.empty:
        save_posts(df, args.output)
    else:
        logger.warning("No posts collected. Check Reddit API credentials in .env file.")


if __name__ == "__main__":
    main()
