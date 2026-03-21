"""
cleaner.py
----------
Text cleaning pipeline for GUARDIAN-NLP.
Handles HTML, URLs, mentions, emojis, hashtags, and short-text filtering.

Usage (standalone):
    python -m src.preprocess.cleaner --input data/raw/combined_raw.csv
                                     --output data/processed/cleaned.csv
"""

import argparse
import logging
import os
import re

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class TextCleaner:
    """
    Cleans raw social media text for NLP training.

    Operations (in order):
        1. Strip HTML tags (BeautifulSoup)
        2. Remove URLs
        3. Remove @mentions (preserves privacy)
        4. Keep hashtag text, remove '#' symbol
        5. Convert emojis to text descriptions
        6. Lowercase
        7. Collapse whitespace
        8. Filter texts shorter than 3 words
    """

    def __init__(self, min_words: int = 3):
        self.url_pattern = re.compile(r"http\S+|www\S+|https\S+", re.IGNORECASE)
        self.mention_pattern = re.compile(r"@\w+")
        self.hashtag_pattern = re.compile(r"#(\w+)")
        self.html_tag_pattern = re.compile(r"<[^>]+>")
        self.multi_space = re.compile(r"\s+")
        self.special_chars = re.compile(r"[^\w\s.,!?;:'\"-]")
        self.min_words = min_words

        # Try to import optional dependencies
        try:
            from bs4 import BeautifulSoup
            self._bs4_available = True
        except ImportError:
            logger.warning("beautifulsoup4 not installed. Falling back to regex HTML removal.")
            self._bs4_available = False

        try:
            import emoji as emoji_lib
            self._emoji_lib = emoji_lib
        except ImportError:
            logger.warning("emoji not installed. Emojis will be stripped.")
            self._emoji_lib = None

    def _remove_html(self, text: str) -> str:
        if self._bs4_available:
            from bs4 import BeautifulSoup
            return BeautifulSoup(text, "html.parser").get_text(separator=" ")
        return self.html_tag_pattern.sub(" ", text)

    def _convert_emojis(self, text: str) -> str:
        if self._emoji_lib:
            return self._emoji_lib.demojize(text, delimiters=(" ", " "))
        # Fallback: remove emojis via unicode range
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        return emoji_pattern.sub(" ", text)

    def clean(self, text: str) -> str | None:
        """
        Cleans a single text string.
        Returns None if text is invalid or too short after cleaning.
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return None

        # 1. Remove HTML
        text = self._remove_html(text)
        # 2. Remove URLs
        text = self.url_pattern.sub(" ", text)
        # 3. Remove @mentions
        text = self.mention_pattern.sub(" ", text)
        # 4. Keep hashtag text, remove #
        text = self.hashtag_pattern.sub(r"\1", text)
        # 5. Convert emojis
        text = self._convert_emojis(text)
        # 6. Lowercase
        text = text.lower()
        # 7. Collapse whitespace
        text = self.multi_space.sub(" ", text).strip()
        # 8. Filter short texts
        if len(text.split()) < self.min_words:
            return None

        return text

    def clean_dataframe(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        """Applies clean() to a DataFrame column, drops None rows."""
        logger.info(f"Cleaning {len(df)} texts...")
        tqdm.pandas(desc="Cleaning texts")
        df = df.copy()
        df[text_col] = df[text_col].progress_apply(self.clean)
        before = len(df)
        df = df.dropna(subset=[text_col])
        logger.info(f"Dropped {before - len(df)} invalid/short texts. Retained {len(df)} rows.")
        return df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="GUARDIAN-NLP: Text Cleaner")
    parser.add_argument("--input", type=str, default="data/raw/combined_raw.csv")
    parser.add_argument("--output", type=str, default="data/processed/cleaned.csv")
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--min_words", type=int, default=3)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        logger.error("Run: python -m src.collect.data_merger first.")
        return

    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} rows from {args.input}")

    cleaner = TextCleaner(min_words=args.min_words)
    df_clean = cleaner.clean_dataframe(df, text_col=args.text_col)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_clean.to_csv(args.output, index=False)
    logger.info(f"✅ Cleaned data saved → {args.output} ({len(df_clean)} rows)")


if __name__ == "__main__":
    main()
