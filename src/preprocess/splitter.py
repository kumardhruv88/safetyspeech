"""
splitter.py
-----------
Splits labeled data into train/val/test sets (70%/15%/15%).
Uses stratified splitting by primary label to preserve class distribution.

Usage:
    python -m src.preprocess.splitter
    python -m src.preprocess.splitter --input data/processed/labeled.csv --stratify label
"""

import argparse
import logging
import os

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LABEL_COLS = ["normal", "depressive", "hate_speech", "violent"]


def get_primary_label(row: pd.Series) -> str:
    """
    For stratification: picks the highest-priority non-normal label,
    or 'normal' if the row is clean.
    Priority: violent > hate_speech > depressive > normal
    """
    if row.get("violent", 0) == 1:
        return "violent"
    if row.get("hate_speech", 0) == 1:
        return "hate_speech"
    if row.get("depressive", 0) == 1:
        return "depressive"
    return "normal"


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify_col: str = None,
    seed: int = 42,
) -> tuple:
    """
    Splits DataFrame into train/val/test.

    Returns:
        (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    stratify = df[stratify_col] if stratify_col and stratify_col in df.columns else None

    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
        stratify=stratify,
    )

    # Second split: val vs test from temp
    temp_stratify = temp_df[stratify_col] if stratify_col and stratify_col in temp_df.columns else None
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_ratio / (val_ratio + test_ratio),
        random_state=seed,
        stratify=temp_stratify,
    )

    logger.info(f"Split sizes → Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def log_label_distribution(df: pd.DataFrame, split_name: str) -> None:
    """Logs per-label positivity rates for a split."""
    logger.info(f"  [{split_name}] Label distribution:")
    for col in LABEL_COLS:
        if col in df.columns:
            pct = 100 * df[col].mean()
            logger.info(f"    {col}: {pct:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="GUARDIAN-NLP: Train/Val/Test Splitter")
    parser.add_argument("--input", type=str, default="data/processed/labeled.csv")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--stratify", type=str, default="primary_label",
                        help="Column to stratify on (use 'primary_label' for auto-derive)")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input not found: {args.input}. Run label_encoder first.")
        return

    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} rows from {args.input}")

    # Derive primary_label for stratification if needed
    if args.stratify == "primary_label":
        df["primary_label"] = df.apply(get_primary_label, axis=1)
        logger.info("Derived primary_label column for stratification")
        primary_counts = df["primary_label"].value_counts()
        logger.info(f"Primary label distribution:\n{primary_counts.to_string()}")

    stratify_col = args.stratify if args.stratify in df.columns else None

    train_df, val_df, test_df = split_dataset(
        df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        stratify_col=stratify_col,
        seed=args.seed,
    )

    log_label_distribution(train_df, "train")
    log_label_distribution(val_df, "val")
    log_label_distribution(test_df, "test")

    os.makedirs(args.output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(args.output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(args.output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(args.output_dir, "test.csv"), index=False)

    logger.info(f"✅ Splits saved → {args.output_dir}/train.csv, val.csv, test.csv")


if __name__ == "__main__":
    main()
