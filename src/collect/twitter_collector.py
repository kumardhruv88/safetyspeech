"""
twitter_collector.py
--------------------
Collects tweets using snscrape (no API key required for historical data).
Searches for toxic / depressive content using predefined queries.

Usage:
    python -m src.collect.twitter_collector
    python -m src.collect.twitter_collector --max_per_query 2000 --output data/raw/twitter_raw.csv
"""

import argparse
import logging
import os

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEARCH_QUERIES = [
    "I want to kill myself",
    "I hate (group OR people OR them) -filter:retweets",
    "threatening you will regret this -filter:retweets",
    "feeling hopeless can't go on -filter:retweets",
    "can't take it anymore pointless -filter:retweets",
    "you deserve to die -filter:retweets",
    "white supremacy racist scum -filter:retweets",
    "I want to end it all -filter:retweets",
    "nobody cares I'm invisible depressed -filter:retweets",
    "incitement violence shoot them -filter:retweets",
]

LABEL_MAP = {
    "I want to kill myself": "violent|depressive",
    "I hate (group OR people OR them) -filter:retweets": "hate_speech",
    "threatening you will regret this -filter:retweets": "violent",
    "feeling hopeless can't go on -filter:retweets": "depressive",
    "can't take it anymore pointless -filter:retweets": "depressive",
    "you deserve to die -filter:retweets": "hate_speech|violent",
    "white supremacy racist scum -filter:retweets": "hate_speech",
    "I want to end it all -filter:retweets": "depressive",
    "nobody cares I'm invisible depressed -filter:retweets": "depressive",
    "incitement violence shoot them -filter:retweets": "violent",
}


def collect_tweets(queries: list, max_per_query: int = 5000) -> pd.DataFrame:
    """
    Collects tweets via snscrape. Falls back gracefully if snscrape is unavailable.
    NOTE: snscrape may break if Twitter changes its structure.
    """
    try:
        import snscrape.modules.twitter as sntwitter
    except ImportError:
        logger.error("snscrape not installed. Run: pip install snscrape")
        return pd.DataFrame()

    all_tweets = []
    for query in queries:
        logger.info(f"Collecting tweets for query: '{query[:60]}...'")
        label_hint = LABEL_MAP.get(query, "unknown")
        collected = 0
        try:
            scraper = sntwitter.TwitterSearchScraper(query)
            for tweet in tqdm(scraper.get_items(), desc=f"Query: {query[:40]}", total=max_per_query):
                if collected >= max_per_query:
                    break
                # PII protection: do NOT store usernames
                all_tweets.append({
                    "id": str(tweet.id),
                    "date": str(tweet.date),
                    "text": tweet.rawContent,
                    "platform": "twitter",
                    "label_hint": label_hint,
                    "like_count": tweet.likeCount,
                    "reply_count": tweet.replyCount,
                })
                collected += 1
        except Exception as e:
            logger.warning(f"Error collecting tweets for query '{query}': {e}")

    df = pd.DataFrame(all_tweets)
    if not df.empty:
        df = df.drop_duplicates(subset=["id"])
        logger.info(f"Collected {len(df)} unique tweets total.")
    return df


def save_tweets(df: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} tweets → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="GUARDIAN-NLP: Twitter Data Collector")
    parser.add_argument("--max_per_query", type=int, default=5000,
                        help="Maximum tweets to collect per search query")
    parser.add_argument("--output", type=str, default="data/raw/twitter_raw.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    df = collect_tweets(SEARCH_QUERIES, max_per_query=args.max_per_query)
    if not df.empty:
        save_tweets(df, args.output)
    else:
        logger.warning("No tweets collected. Check snscrape installation and network access.")


if __name__ == "__main__":
    main()
