"""
label_encoder.py
----------------
Converts string labels in combined_raw.csv to multi-hot binary vectors
and saves an encoded CSV suitable for ToxicDataset.

Label schema:
    0 → normal
    1 → depressive
    2 → hate_speech
    3 → violent

Multi-hot format: a single text can have labels [1, 3] = depressive + violent.
For now, single-label rows get a 1 at the corresponding index; 0 elsewhere.

Usage:
    python -m src.preprocess.label_encoder
    python -m src.preprocess.label_encoder --input data/processed/cleaned.csv
                                           --output data/processed/labeled.csv
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LABEL_COLS = ["normal", "depressive", "hate_speech", "violent"]
LABEL2IDX = {label: idx for idx, label in enumerate(LABEL_COLS)}


def label_to_multihot(label_str: str) -> list:
    """
    Converts a label string (possibly pipe-separated) to a multi-hot vector.

    Examples:
        "depressive"         → [0, 1, 0, 0]
        "violent|depressive" → [0, 1, 0, 1]
        "normal"             → [1, 0, 0, 0]
    """
    vector = [0] * len(LABEL_COLS)
    if not isinstance(label_str, str):
        vector[0] = 1  # default to normal
        return vector

    for part in label_str.split("|"):
        part = part.strip().lower()
        if part in LABEL2IDX:
            vector[LABEL2IDX[part]] = 1

    # Safety fallback: if nothing set, mark as normal
    if sum(vector) == 0:
        vector[0] = 1

    return vector


def encode_labels(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """
    Adds multi-hot label columns to the DataFrame.
    Drops the original string label column afterward.
    """
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found in DataFrame. "
                         f"Available: {df.columns.tolist()}")

    logger.info(f"Encoding {len(df)} rows with multi-hot labels...")
    multihot = df[label_col].apply(label_to_multihot)
    label_df = pd.DataFrame(multihot.tolist(), columns=LABEL_COLS)
    df = pd.concat([df.reset_index(drop=True), label_df], axis=1)

    # Log distribution
    for col in LABEL_COLS:
        count = label_df[col].sum()
        pct = 100 * count / len(df)
        logger.info(f"  {col}: {int(count)} samples ({pct:.1f}%)")

    return df


def compute_pos_weights(df: pd.DataFrame) -> dict:
    """
    Computes BCEWithLogitsLoss pos_weight for each label.
    pos_weight[i] = num_negative_i / num_positive_i
    Used to handle class imbalance during training.
    """
    weights = {}
    for col in LABEL_COLS:
        n_pos = df[col].sum()
        n_neg = len(df) - n_pos
        weight = n_neg / max(n_pos, 1)
        weights[col] = round(weight, 4)
        logger.info(f"  pos_weight[{col}] = {weight:.4f}")
    return weights


def main():
    parser = argparse.ArgumentParser(description="GUARDIAN-NLP: Label Encoder")
    parser.add_argument("--input", type=str, default="data/processed/cleaned.csv")
    parser.add_argument("--output", type=str, default="data/processed/labeled.csv")
    parser.add_argument("--label_col", type=str, default="label",
                        help="Column name containing string labels")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input not found: {args.input}. Run cleaner first.")
        return

    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} rows from {args.input}")

    df_encoded = encode_labels(df, label_col=args.label_col)

    logger.info("Computing pos_weights for BCEWithLogitsLoss:")
    weights = compute_pos_weights(df_encoded)

    # Save pos_weights alongside the data
    weights_df = pd.DataFrame([weights])
    weights_path = os.path.join(os.path.dirname(args.output), "pos_weights.csv")
    weights_df.to_csv(weights_path, index=False)
    logger.info(f"Pos weights saved → {weights_path}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_encoded.to_csv(args.output, index=False)
    logger.info(f"✅ Labeled data saved → {args.output} ({len(df_encoded)} rows)")


if __name__ == "__main__":
    main()
