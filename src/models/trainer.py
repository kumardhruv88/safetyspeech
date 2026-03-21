"""
trainer.py
----------
GUARDIAN-NLP training loop.

Features:
  - AdamW optimizer with weight decay
  - Linear warmup + linear decay LR scheduler
  - BCEWithLogitsLoss with class-imbalance pos_weight
  - Per-epoch validation with Micro F1 tracking
  - Best-model checkpoint saving
  - Gradient accumulation support
  - tqdm progress bars + loguru logging
  - Seed everything for reproducibility

Usage (from train.py):
    from src.models.trainer import GuardianTrainer
    trainer = GuardianTrainer(config)
    trainer.train()
"""

import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, get_linear_schedule_with_warmup

from src.models.bert_classifier import GuardianBERT
from src.models.dataset import ToxicDataset
from src.evaluate.metrics import evaluate


LABEL_COLS = ["normal", "depressive", "hate_speech", "violent"]


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class GuardianTrainer:
    """
    Full training and validation pipeline for GuardianBERT.

    Args:
        config (dict): Configuration dictionary from config.yaml.
    """

    def __init__(self, config: dict):
        self.config = config
        self.model_cfg = config.get("model", {})
        self.train_cfg = config.get("training", {})
        self.data_cfg = config.get("data", {})
        self.infer_cfg = config.get("inference", {})
        self.log_cfg = config.get("logging", {})

        # Device
        device_name = self.train_cfg.get("device", "cuda")
        if device_name == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available — falling back to CPU.")
            device_name = "cpu"
        self.device = torch.device(device_name)
        logger.info(f"Using device: {self.device}")

        # Reproducibility
        seed = self.train_cfg.get("seed", 42)
        seed_everything(seed)

        # Directories
        self.checkpoint_dir = self.infer_cfg.get("model_path", "models/checkpoints/best_model.pt")
        self.checkpoint_dir = os.path.dirname(self.checkpoint_dir)
        self.tokenizer_path = self.infer_cfg.get("tokenizer_path", "models/checkpoints/tokenizer/")
        self.report_dir = self.log_cfg.get("report_dir", "outputs/reports/")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.tokenizer_path, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)

    def _build_model(self) -> GuardianBERT:
        model = GuardianBERT(
            num_labels=self.model_cfg.get("num_labels", 4),
            dropout=self.model_cfg.get("dropout", 0.3),
            model_name=self.model_cfg.get("name", "bert-base-uncased"),
        )
        model = model.to(self.device)
        logger.info(f"Model parameters: {model.count_parameters():,}")
        return model

    def _build_datasets(self, tokenizer: BertTokenizer):
        train_ds = ToxicDataset(
            self.data_cfg.get("train_path", "data/processed/train.csv"),
            tokenizer,
            max_length=self.model_cfg.get("max_length", 128),
        )
        val_ds = ToxicDataset(
            self.data_cfg.get("val_path", "data/processed/val.csv"),
            tokenizer,
            max_length=self.model_cfg.get("max_length", 128),
        )
        return train_ds, val_ds

    def _build_dataloaders(self, train_ds, val_ds):
        batch_size = self.train_cfg.get("batch_size", 16)
        num_workers = min(4, os.cpu_count() or 1)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=(self.device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size * 2, shuffle=False,
            num_workers=num_workers, pin_memory=(self.device.type == "cuda"),
        )
        return train_loader, val_loader

    def train(self) -> None:
        """Main training entry point."""
        start_time = time.time()

        # Load tokenizer
        model_name = self.model_cfg.get("name", "bert-base-uncased")
        tokenizer = BertTokenizer.from_pretrained(model_name)

        # Build datasets and dataloaders
        train_ds, val_ds = self._build_datasets(tokenizer)
        train_loader, val_loader = self._build_dataloaders(train_ds, val_ds)

        # Build model
        model = self._build_model()

        # Loss with class-imbalance handling
        pos_weight = train_ds.get_pos_weights().to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        logger.info(f"pos_weight: {pos_weight.cpu().tolist()}")

        # Optimizer
        lr = self.train_cfg.get("learning_rate", 2e-5)
        wd = self.train_cfg.get("weight_decay", 0.01)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)

        # LR Scheduler: linear warmup
        num_epochs = self.train_cfg.get("num_epochs", 5)
        accum_steps = self.train_cfg.get("gradient_accumulation_steps", 1)
        total_steps = (len(train_loader) // accum_steps) * num_epochs
        warmup_ratio = self.train_cfg.get("warmup_ratio", 0.1)
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        logger.info(f"Total steps: {total_steps} | Warmup steps: {warmup_steps}")

        # Training loop
        best_val_f1 = 0.0
        threshold = self.infer_cfg.get("threshold", 0.5)

        for epoch in range(1, num_epochs + 1):
            logger.info(f"\n{'='*60}\nEpoch {epoch}/{num_epochs}\n{'='*60}")
            train_loss = self._train_epoch(
                model, train_loader, criterion, optimizer, scheduler, accum_steps, epoch
            )
            val_metrics = self._validate_epoch(model, val_loader, criterion, threshold)

            val_f1 = val_metrics["micro_f1"]
            logger.info(
                f"Epoch {epoch} | Train Loss: {train_loss:.4f} | "
                f"Val Micro F1: {val_f1:.4f} | Val Macro F1: {val_metrics['macro_f1']:.4f} | "
                f"Val AUC-ROC: {val_metrics['roc_auc']:.4f} | "
                f"Val Hamming: {val_metrics['hamming_loss']:.4f}"
            )

            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self._save_checkpoint(model, tokenizer, val_metrics)
                logger.info(f"✅ New best model saved (Val Micro F1: {best_val_f1:.4f})")

        elapsed = (time.time() - start_time) / 60
        logger.info(f"\n🎉 Training complete! Time: {elapsed:.1f} min | Best Val Micro F1: {best_val_f1:.4f}")

    def _train_epoch(
        self, model, loader, criterion, optimizer, scheduler, accum_steps, epoch_num
    ) -> float:
        """Runs one training epoch. Returns average loss."""
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(loader, desc=f"Train Epoch {epoch_num}")
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward pass (use logits for BCEWithLogitsLoss)
            logits = model.get_logits(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss = loss / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accum_steps
            pbar.set_postfix({"loss": f"{total_loss / (step + 1):.4f}"})

        return total_loss / len(loader)

    @torch.no_grad()
    def _validate_epoch(self, model, loader, criterion, threshold: float) -> dict:
        """Runs validation and computes metrics."""
        model.eval()
        all_probs = []
        all_labels = []

        for batch in tqdm(loader, desc="Validating"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"]

            probs = model(input_ids, attention_mask)  # sigmoid probabilities
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

        y_true = np.vstack(all_labels)
        y_pred_probs = np.vstack(all_probs)

        return evaluate(y_true, y_pred_probs, threshold=threshold)

    def _save_checkpoint(
        self, model: GuardianBERT, tokenizer: BertTokenizer, metrics: dict
    ) -> None:
        """Saves model weights and tokenizer."""
        model_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        torch.save(model.state_dict(), model_path)
        tokenizer.save_pretrained(self.tokenizer_path)

        # Save metrics alongside
        import json
        metrics_path = os.path.join(self.report_dir, "best_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump({k: round(float(v), 6) for k, v in metrics.items()}, f, indent=2)
        logger.info(f"Checkpoint saved → {model_path}")
