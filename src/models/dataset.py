"""
dataset.py
----------
PyTorch Dataset class for GUARDIAN-NLP.
Reads tokenized CSV data for training, validation, and testing.

Expected CSV columns:
    text, normal, depressive, hate_speech, violent (multi-hot binary)

Usage:
    dataset = ToxicDataset("data/processed/train.csv", tokenizer)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
"""

import logging
from typing import Dict, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LABEL_COLS = ["normal", "depressive", "hate_speech", "violent"]


class ToxicDataset(Dataset):
    """
    PyTorch Dataset for multi-label toxic content classification.

    Args:
        csv_path (str): Path to the preprocessed CSV file.
        tokenizer (BertTokenizer): HuggingFace BERT tokenizer.
        max_length (int): Maximum token sequence length (default 128).
        label_cols (list): Column names for multi-hot labels.
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer: BertTokenizer,
        max_length: int = 128,
        label_cols: Optional[list] = None,
    ):
        self.csv_path = csv_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_cols = label_cols or LABEL_COLS

        self.df = pd.read_csv(csv_path)
        self._validate_columns()
        logger.info(f"Loaded ToxicDataset: {len(self.df)} samples from '{csv_path}'")
        self._log_label_distribution()

    def _validate_columns(self) -> None:
        """Ensures required columns exist in the DataFrame."""
        missing = []
        if "text" not in self.df.columns:
            missing.append("text")
        for col in self.label_cols:
            if col not in self.df.columns:
                missing.append(col)
        if missing:
            raise ValueError(
                f"Missing columns in '{self.csv_path}': {missing}. "
                f"Available: {self.df.columns.tolist()}"
            )
        # Ensure text column is string
        self.df["text"] = self.df["text"].astype(str).fillna("")

    def _log_label_distribution(self) -> None:
        """Logs the positive rate for each label."""
        logger.info("Label distribution:")
        for col in self.label_cols:
            count = int(self.df[col].sum())
            pct = 100 * count / len(self.df)
            logger.info(f"  {col}: {count} ({pct:.1f}%)")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a single training sample as a dictionary:
            - input_ids:      [max_length]
            - attention_mask: [max_length]
            - labels:         [num_labels] float tensor with multi-hot values
        """
        row = self.df.iloc[idx]
        text = str(row["text"])

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Extract labels as float tensor
        labels = torch.tensor(
            row[self.label_cols].values.astype(float),
            dtype=torch.float32,
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),          # [max_length]
            "attention_mask": encoding["attention_mask"].squeeze(0), # [max_length]
            "labels": labels,                                         # [num_labels]
        }

    def get_pos_weights(self) -> torch.Tensor:
        """
        Computes pos_weight tensor for BCEWithLogitsLoss to handle class imbalance.
        pos_weight[i] = num_negative_i / num_positive_i
        """
        weights = []
        for col in self.label_cols:
            n_pos = self.df[col].sum()
            n_neg = len(self.df) - n_pos
            weight = n_neg / max(n_pos, 1)
            weights.append(weight)
        return torch.tensor(weights, dtype=torch.float32)
