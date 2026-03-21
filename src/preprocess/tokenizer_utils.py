"""
tokenizer_utils.py
------------------
BERT tokenization helpers for GUARDIAN-NLP.
Wraps HuggingFace BertTokenizer with convenient batch utilities.
"""

import logging
from typing import List, Dict

import torch
from transformers import BertTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "bert-base-uncased"
DEFAULT_MAX_LENGTH = 128


def get_tokenizer(model_name: str = DEFAULT_MODEL) -> BertTokenizer:
    """Load and return a BertTokenizer."""
    logger.info(f"Loading tokenizer: {model_name}")
    return BertTokenizer.from_pretrained(model_name)


def tokenize_batch(
    texts: List[str],
    tokenizer: BertTokenizer = None,
    max_length: int = DEFAULT_MAX_LENGTH,
    return_tensors: str = "pt",
) -> Dict[str, torch.Tensor]:
    """
    Tokenizes a batch of texts.

    Args:
        texts: List of raw text strings.
        tokenizer: BertTokenizer instance (loads default if None).
        max_length: Maximum token sequence length (default 128).
        return_tensors: 'pt' for PyTorch tensors.

    Returns:
        Dict with 'input_ids' and 'attention_mask' tensors.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()

    return tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors=return_tensors,
    )


def tokenize_single(
    text: str,
    tokenizer: BertTokenizer = None,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> Dict[str, torch.Tensor]:
    """Tokenizes a single text string."""
    return tokenize_batch([text], tokenizer=tokenizer, max_length=max_length)


def decode_tokens(input_ids: torch.Tensor, tokenizer: BertTokenizer = None) -> List[str]:
    """
    Decodes tokenized input_ids back to readable text (for debugging).
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    return [
        tokenizer.decode(ids, skip_special_tokens=True)
        for ids in input_ids
    ]


def save_tokenizer(tokenizer: BertTokenizer, save_path: str) -> None:
    """Saves tokenizer to disk for inference."""
    import os
    os.makedirs(save_path, exist_ok=True)
    tokenizer.save_pretrained(save_path)
    logger.info(f"Tokenizer saved → {save_path}")


def load_tokenizer(tokenizer_path: str) -> BertTokenizer:
    """Loads a saved tokenizer from disk."""
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    return BertTokenizer.from_pretrained(tokenizer_path)
