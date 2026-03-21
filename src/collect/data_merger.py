"""
data_merger.py
--------------
Merges all raw data sources (Twitter raw, Reddit raw, and external datasets)
into a single unified CSV with standardized columns.

Label unification schema:
    0 → normal
    1 → depressive
    2 → hate_speech
    3 → violent

Output: data/raw/combined_raw.csv

Usage:
    python -m src.collect.data_merger
    python -m src.collect.data_merger --output data/raw/combined_raw.csv
"""

import argparse
import logging
import os

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LABEL2ID = {"normal": 0, "depressive": 1, "hate_speech": 2, "violent": 3}

# ─────────────────────────────────────────────
# Per-dataset loader functions
# ─────────────────────────────────────────────

def load_jigsaw(path: str) -> pd.DataFrame:
    """
    Loads the Jigsaw Toxic Comment Classification dataset.
    Maps: toxic/severe_toxic/threat → violent, insult/obscene/identity_hate → hate_speech
    normal if all labels 0.
    """
    if not os.path.exists(path):
        logger.warning(f"Jigsaw file not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    # Jigsaw columns: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate
    required_cols = ["comment_text", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    if not all(c in df.columns for c in required_cols):
        logger.warning(f"Unexpected Jigsaw columns: {df.columns.tolist()}")
        return pd.DataFrame()

    rows = []
    for _, row in df.iterrows():
        text = row["comment_text"]
        if row["threat"] == 1 or row["severe_toxic"] == 1:
            label = "violent"
        elif row["toxic"] == 1 or row["insult"] == 1 or row["obscene"] == 1 or row["identity_hate"] == 1:
            label = "hate_speech"
        else:
            label = "normal"
        rows.append({"text": text, "label": label, "platform": "wikipedia", "source_dataset": "jigsaw"})

    return pd.DataFrame(rows)


def load_davidson(path: str) -> pd.DataFrame:
    """
    Loads Davidson Hate Speech & Offensive Language dataset.
    Classes: 0=hate_speech, 1=offensive_language, 2=neither
    """
    if not os.path.exists(path):
        logger.warning(f"Davidson file not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    # Davidson columns: count, hate_speech, offensive_language, neither, class, tweet
    if "tweet" not in df.columns or "class" not in df.columns:
        logger.warning(f"Unexpected Davidson columns: {df.columns.tolist()}")
        return pd.DataFrame()

    label_map = {0: "hate_speech", 1: "hate_speech", 2: "normal"}
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "text": row["tweet"],
            "label": label_map.get(row["class"], "normal"),
            "platform": "twitter",
            "source_dataset": "davidson",
        })
    return pd.DataFrame(rows)

def load_ucsd(path: str) -> pd.DataFrame:
    """Loads UCSD Measuring Hate Speech dataset."""
    if not os.path.exists(path):
        logger.warning(f"UCSD file not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        logger.warning(f"Unexpected UCSD columns: {df.columns.tolist()}")
        return pd.DataFrame()
    df["platform"] = "reddit_twitter_gab"
    df["source_dataset"] = "ucsd_hate_speech"
    return df[["text", "label", "platform", "source_dataset"]]

def load_depression_reddit(path: str) -> pd.DataFrame:
    """
    Loads the Depression Reddit (CLPsych) dataset.
    Labels: 1=depression (depressive), 0=non-depression (normal)
    """
    if not os.path.exists(path):
        logger.warning(f"Depression Reddit file not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    # Expected columns: text, label
    if "text" not in df.columns:
        logger.warning(f"Unexpected Depression Reddit columns: {df.columns.tolist()}")
        return pd.DataFrame()

    label_col = "label" if "label" in df.columns else df.columns[-1]
    rows = []
    for _, row in df.iterrows():
        label = "depressive" if int(row[label_col]) == 1 else "normal"
        rows.append({
            "text": row["text"],
            "label": label,
            "platform": "reddit",
            "source_dataset": "depression_reddit",
        })
    return pd.DataFrame(rows)


def load_live_collected(path: str, platform: str) -> pd.DataFrame:
    """Loads live-collected Twitter or Reddit data."""
    if not os.path.exists(path):
        logger.warning(f"Live collected file not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    if "text" not in df.columns:
        logger.warning(f"No 'text' column in {path}")
        return pd.DataFrame()

    # Derive label from label_hint if available
    def hint_to_label(hint: str) -> str:
        if not isinstance(hint, str):
            return "normal"
        if "depressive" in hint:
            return "depressive"
        if "hate" in hint:
            return "hate_speech"
        if "violent" in hint:
            return "violent"
        return "normal"

    if "label_hint" in df.columns:
        df["label"] = df["label_hint"].apply(hint_to_label)
    else:
        df["label"] = "normal"

    df["platform"] = platform
    df["source_dataset"] = f"live_{platform}"
    return df[["text", "label", "platform", "source_dataset"]]


def merge_and_deduplicate(frames: list) -> pd.DataFrame:
    """Concatenates, deduplicates, and validates the merged dataset."""
    valid_frames = [f for f in frames if not f.empty]
    if not valid_frames:
        logger.error("No data loaded from any source!")
        return pd.DataFrame()

    merged = pd.concat(valid_frames, ignore_index=True)
    # Drop duplicates by text
    before = len(merged)
    merged = merged.drop_duplicates(subset=["text"])
    logger.info(f"Removed {before - len(merged)} duplicates ({len(merged)} records remaining).")

    # Drop rows with missing text
    merged = merged.dropna(subset=["text"])
    merged["text"] = merged["text"].astype(str).str.strip()
    merged = merged[merged["text"].str.len() > 10]

    # Validate labels
    valid_labels = set(LABEL2ID.keys())
    merged = merged[merged["label"].isin(valid_labels)]

    # Add numeric label ID
    merged["label_id"] = merged["label"].map(LABEL2ID)

    label_counts = merged["label"].value_counts()
    logger.info(f"Label distribution:\n{label_counts.to_string()}")
    logger.info(f"Source distribution:\n{merged['source_dataset'].value_counts().to_string()}")

    return merged[["text", "label", "label_id", "platform", "source_dataset"]]


def main():
    parser = argparse.ArgumentParser(description="GUARDIAN-NLP: Data Merger")
    parser.add_argument("--output", type=str, default="data/raw/combined_raw.csv")
    parser.add_argument("--jigsaw", type=str, default="data/external/jigsaw_toxic.csv")
    parser.add_argument("--davidson", type=str, default="data/external/hate_speech_davidson.csv")
    parser.add_argument("--ucsd", type=str, default="data/external/hate_speech_ucsd.csv")
    parser.add_argument("--depression", type=str, default="data/external/depression_reddit.csv")
    parser.add_argument("--twitter_raw", type=str, default="data/raw/twitter_raw.csv")
    parser.add_argument("--reddit_raw", type=str, default="data/raw/reddit_raw.csv")
    args = parser.parse_args()

    frames = [
        load_jigsaw(args.jigsaw),
        load_davidson(args.davidson),
        load_ucsd(args.ucsd),
        load_depression_reddit(args.depression),
        load_live_collected(args.twitter_raw, "twitter"),
        load_live_collected(args.reddit_raw, "reddit"),
    ]

    merged = merge_and_deduplicate(frames)
    if not merged.empty:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        merged.to_csv(args.output, index=False)
        logger.info(f"\n✅ Merged dataset saved → {args.output} ({len(merged)} rows)")
    else:
        logger.error("Merging failed — no data to save.")


if __name__ == "__main__":
    main()
