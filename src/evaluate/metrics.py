"""
metrics.py
----------
Evaluation metrics for GUARDIAN-NLP multi-label classification.

Metrics computed:
  - Micro F1 (aggregated across all labels)
  - Macro F1 (average per-label F1)
  - Per-label F1, Precision, Recall
  - ROC-AUC per label (and macro average)
  - Hamming Loss (fraction of wrong labels)
  - Full classification report

Usage:
    python -m src.evaluate.metrics --model models/checkpoints/best_model.pt
"""

import argparse
import json
import logging
import os
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    classification_report,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LABEL_COLS = ["normal", "depressive", "hate_speech", "violent"]


def evaluate(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    threshold: float = 0.5,
    label_names: Optional[list] = None,
) -> Dict[str, float]:
    """
    Computes multi-label classification metrics.

    Args:
        y_true: Ground truth binary labels [N, num_labels]
        y_pred_probs: Predicted sigmoid probabilities [N, num_labels]
        threshold: Decision threshold (default 0.5)
        label_names: List of label names for reporting

    Returns:
        Dict with micro_f1, macro_f1, hamming_loss, roc_auc, per-label metrics
    """
    label_names = label_names or LABEL_COLS
    y_pred = (y_pred_probs >= threshold).astype(int)

    metrics = {}

    # Aggregate metrics
    metrics["micro_f1"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["hamming_loss"] = float(hamming_loss(y_true, y_pred))

    # ROC-AUC (requires at least 2 classes present)
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_probs, average="macro"))
    except ValueError:
        metrics["roc_auc"] = 0.0
        logger.warning("Could not compute ROC-AUC (possibly missing positive samples).")

    # Per-label metrics
    for i, label in enumerate(label_names):
        if y_true.shape[1] > i:
            yt = y_true[:, i]
            yp = y_pred[:, i]
            ypp = y_pred_probs[:, i]
            metrics[f"{label}_f1"] = float(f1_score(yt, yp, zero_division=0))
            metrics[f"{label}_precision"] = float(precision_score(yt, yp, zero_division=0))
            metrics[f"{label}_recall"] = float(recall_score(yt, yp, zero_division=0))
            try:
                metrics[f"{label}_auc"] = float(roc_auc_score(yt, ypp))
            except ValueError:
                metrics[f"{label}_auc"] = 0.0

    return metrics


def print_report(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    threshold: float = 0.5,
    label_names: Optional[list] = None,
    save_path: Optional[str] = None,
) -> str:
    """
    Prints and optionally saves a full classification report.
    """
    label_names = label_names or LABEL_COLS
    y_pred = (y_pred_probs >= threshold).astype(int)

    report = classification_report(
        y_true, y_pred, target_names=label_names, zero_division=0
    )
    metrics = evaluate(y_true, y_pred_probs, threshold=threshold, label_names=label_names)

    full_report = (
        f"GUARDIAN-NLP Classification Report\n"
        f"{'='*60}\n"
        f"{report}\n"
        f"{'='*60}\n"
        f"Micro F1:      {metrics['micro_f1']:.4f}\n"
        f"Macro F1:      {metrics['macro_f1']:.4f}\n"
        f"Hamming Loss:  {metrics['hamming_loss']:.4f}\n"
        f"ROC-AUC:       {metrics['roc_auc']:.4f}\n"
    )

    logger.info(f"\n{full_report}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            f.write(full_report)
        # Save raw metrics as JSON too
        json_path = save_path.replace(".txt", ".json")
        with open(json_path, "w") as f:
            json.dump({k: round(v, 6) for k, v in metrics.items()}, f, indent=2)
        logger.info(f"Report saved → {save_path}")

    return full_report


def evaluate_from_model(
    model,
    test_loader,
    device,
    threshold: float = 0.5,
    report_dir: str = "outputs/reports/",
) -> Dict[str, float]:
    """
    Runs inference on a test DataLoader and computes full metrics.
    """
    import torch
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]
            probs = model(input_ids, attention_mask)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    y_true = np.vstack(all_labels)
    y_pred_probs = np.vstack(all_probs)

    report_path = os.path.join(report_dir, "classification_report.txt")
    print_report(y_true, y_pred_probs, threshold=threshold, save_path=report_path)

    return evaluate(y_true, y_pred_probs, threshold=threshold)


def main():
    parser = argparse.ArgumentParser(description="GUARDIAN-NLP: Evaluation Metrics")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to saved model checkpoint (.pt)")
    parser.add_argument("--test_data", type=str, default="data/processed/test.csv")
    parser.add_argument("--tokenizer", type=str, default="models/checkpoints/tokenizer/")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    import torch
    import yaml
    from torch.utils.data import DataLoader
    from transformers import BertTokenizer
    from src.models.bert_classifier import GuardianBERT
    from src.models.dataset import ToxicDataset

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer)

    model = GuardianBERT(
        num_labels=config["model"]["num_labels"],
        dropout=config["model"]["dropout"],
        model_name=config["model"]["name"],
    )
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device)
    model.eval()

    test_ds = ToxicDataset(args.test_data, tokenizer, max_length=config["model"]["max_length"])
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    metrics = evaluate_from_model(
        model, test_loader, device,
        threshold=args.threshold,
        report_dir=config["logging"]["report_dir"],
    )
    logger.info(f"\n✅ Evaluation complete: {metrics}")


if __name__ == "__main__":
    main()
