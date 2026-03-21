"""
predictor.py
------------
GUARDIAN-NLP Inference Engine.

Loads the trained GuardianBERT model and tokenizer, then predicts
toxicity labels and severity for given text inputs.

Severity thresholds:
  max non-normal score >= 0.85 → HIGH   🔴
  max non-normal score >= 0.60 → MEDIUM 🟠
  max non-normal score >= 0.50 → LOW    🟡
  else                         → SAFE   🟢

Usage:
    from src.inference.predictor import Predictor
    p = Predictor()
    result = p.predict("I can't take it anymore")
"""

import logging
import os
from typing import Dict, List, Optional

import torch
from transformers import BertTokenizer

from src.models.bert_classifier import GuardianBERT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LABELS = ["normal", "depressive", "hate_speech", "violent"]
LABEL_DISPLAY = {
    "normal": "Normal ✅",
    "depressive": "Depressive 😔",
    "hate_speech": "Hate Speech ⚠️",
    "violent": "Violent 🔴",
}
DEFAULT_MODEL_PATH = "models/checkpoints/best_model.pt"
DEFAULT_TOKENIZER_PATH = "models/checkpoints/tokenizer/"


class Predictor:
    """
    Inference wrapper for GuardianBERT.

    Args:
        model_path (str): Path to saved model state dict (.pt).
        tokenizer_path (str): Path to saved tokenizer directory.
        device (str): 'cuda' or 'cpu'.
        threshold (float): Sigmoid threshold for positive prediction.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
        device: str = "cpu",
        threshold: float = 0.5,
        num_labels: int = 4,
    ):
        self.threshold = threshold
        self.num_labels = num_labels

        # Device selection
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available — using CPU.")
            device = "cpu"
        self.device = torch.device(device)

        # Load tokenizer
        self._load_tokenizer(tokenizer_path)

        # Load model
        self._load_model(model_path)

    def _load_tokenizer(self, tokenizer_path: str) -> None:
        """Loads tokenizer from disk or downloads from HuggingFace."""
        if os.path.isdir(tokenizer_path) and os.listdir(tokenizer_path):
            logger.info(f"Loading tokenizer from: {tokenizer_path}")
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        else:
            logger.info("Local tokenizer not found. Attempting to load from auth HF Hub...")
            try:
                self.tokenizer = BertTokenizer.from_pretrained("aryan012234/safetyspeech-bert")
            except Exception:
                logger.warning("Tokenizer not fully uploaded to HF. Defaulting to bert-base-uncased base vocabulary...")
                self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def _load_model(self, model_path: str) -> None:
        """Loads GuardianBERT from saved weights."""
        self.model = GuardianBERT(num_labels=self.num_labels)
        if os.path.exists(model_path):
            logger.info(f"Loading local model weights from: {model_path}")
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            logger.info(f"Local model '{model_path}' not found. Downloading from HuggingFace Hub...")
            try:
                from huggingface_hub import hf_hub_download
                hf_path = hf_hub_download(repo_id="aryan012234/safetyspeech-bert", filename="best_model.pt")
                logger.info("Successfully fetched weights from HuggingFace. Loading model...")
                state_dict = torch.load(hf_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            except Exception as e:
                logger.error(f"Failed to download from HuggingFace Hub: {e}")
                logger.warning("Using UNTRAINED model.")
        self.model = self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str, threshold: Optional[float] = None) -> Dict:
        """
        Predicts toxicity labels and severity for a single text.

        Args:
            text: Raw input text string.
            threshold: Override the default decision threshold.

        Returns:
            Dict with keys: normal, depressive, hate_speech, violent,
                            severity, active_labels, top_label
        """
        if not isinstance(text, str) or not text.strip():
            return self._empty_result()

        thr = threshold or self.threshold

        encoding = self.tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            scores = self.model(
                encoding["input_ids"].to(self.device),
                encoding["attention_mask"].to(self.device),
            ).squeeze(0)  # [num_labels]

        result = {}
        for i, label in enumerate(LABELS):
            result[label] = round(float(scores[i]), 4)

        # Severity based on max non-normal score
        non_normal_scores = [result[l] for l in LABELS if l != "normal"]
        max_score = max(non_normal_scores) if non_normal_scores else 0.0

        if max_score >= 0.85:
            severity = "HIGH 🔴"
        elif max_score >= 0.60:
            severity = "MEDIUM 🟠"
        elif max_score >= thr:
            severity = "LOW 🟡"
        else:
            severity = "SAFE 🟢"

        result["severity"] = severity
        result["active_labels"] = [
            LABEL_DISPLAY[l] for l in LABELS if result[l] >= thr
        ]
        result["top_label"] = max(LABELS, key=lambda l: result[l])
        result["max_confidence"] = round(max(result[l] for l in LABELS), 4)

        return result

    def predict_batch(
        self, texts: List[str], threshold: Optional[float] = None
    ) -> List[Dict]:
        """Predicts for a list of texts."""
        return [self.predict(text, threshold=threshold) for text in texts]

    def predict_dataframe(self, df, text_col: str = "text", threshold: Optional[float] = None):
        """Runs prediction on a DataFrame and appends result columns."""
        import pandas as pd
        from tqdm import tqdm

        results = []
        for text in tqdm(df[text_col], desc="Running inference"):
            results.append(self.predict(text, threshold=threshold))

        results_df = pd.DataFrame(results)
        return pd.concat([df.reset_index(drop=True), results_df], axis=1)

    def _empty_result(self) -> Dict:
        result = {l: 0.0 for l in LABELS}
        result.update({"severity": "SAFE 🟢", "active_labels": [], "top_label": "normal", "max_confidence": 0.0})
        return result


# Convenience function for direct import in Gradio app
def predict_text(text: str, threshold: float = 0.5) -> Dict:
    """Module-level prediction function. Creates a fresh predictor each call (use Predictor class for production)."""
    _predictor = Predictor(threshold=threshold)
    return _predictor.predict(text)
