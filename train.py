"""
train.py
--------
GUARDIAN-NLP: Main Training Entry Point.

Usage:
    python train.py --config config.yaml
    python train.py --config config.yaml --data_dir data/processed/ --output_dir models/checkpoints/
    python train.py --config config.yaml --epochs 3 --batch_size 8  (override config)
"""

import argparse
import os
import sys

import yaml
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="GUARDIAN-NLP: Train BERT multi-label toxic content classifier"
    )
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config.yaml file")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override data directory from config")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override checkpoint output directory")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cuda", "cpu"],
                        help="Override device (cuda or cpu)")
    parser.add_argument("--freeze_bert", action="store_true",
                        help="Freeze BERT backbone — train only classification head")
    parser.add_argument("--test", action="store_true",
                        help="Run on test set after training and save classification report")
    return parser.parse_args()


def load_and_patch_config(args) -> dict:
    """Loads config.yaml and applies any CLI overrides."""
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    if args.data_dir:
        data_dir = args.data_dir.rstrip("/")
        config["data"]["train_path"] = f"{data_dir}/train.csv"
        config["data"]["val_path"] = f"{data_dir}/val.csv"
        config["data"]["test_path"] = f"{data_dir}/test.csv"

    if args.output_dir:
        config["inference"]["model_path"] = os.path.join(args.output_dir, "best_model.pt")
        config["inference"]["tokenizer_path"] = os.path.join(args.output_dir, "tokenizer/")

    if args.epochs:
        config["training"]["num_epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.device:
        config["training"]["device"] = args.device

    return config


def check_data_exists(config: dict) -> bool:
    """Checks that processed data files exist before training."""
    train = config["data"]["train_path"]
    val = config["data"]["val_path"]
    missing = []
    for path in [train, val]:
        if not os.path.exists(path):
            missing.append(path)
    if missing:
        logger.error(f"Missing data files: {missing}")
        logger.error("Run the preprocessing pipeline first:")
        logger.error("  python -m src.collect.data_merger")
        logger.error("  python -m src.preprocess.cleaner")
        logger.error("  python -m src.preprocess.label_encoder")
        logger.error("  python -m src.preprocess.splitter")
        return False
    return True


def main():
    logger.add("runs/training_{time}.log", rotation="100 MB", level="INFO")

    args = parse_args()
    config = load_and_patch_config(args)

    logger.info("=" * 60)
    logger.info("🛡️  GUARDIAN-NLP Training")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")
    logger.info(f"Model: {config['model']['name']} | Epochs: {config['training']['num_epochs']} | "
                f"Batch: {config['training']['batch_size']} | LR: {config['training']['learning_rate']}")

    if not check_data_exists(config):
        sys.exit(1)

    # Import here to avoid slow startup if data checks fail
    from src.models.trainer import GuardianTrainer

    trainer = GuardianTrainer(config)

    if args.freeze_bert:
        logger.info("BERT backbone frozen — training classification head only.")

    trainer.train()

    # Optional: evaluate on test set
    if args.test:
        logger.info("\n🔬 Running evaluation on test set...")
        import torch
        from torch.utils.data import DataLoader
        from transformers import BertTokenizer
        from src.models.bert_classifier import GuardianBERT
        from src.models.dataset import ToxicDataset
        from src.evaluate.metrics import evaluate_from_model

        device = torch.device(config["training"]["device"])
        tokenizer = BertTokenizer.from_pretrained(config["inference"]["tokenizer_path"])
        model = GuardianBERT(
            num_labels=config["model"]["num_labels"],
            dropout=config["model"]["dropout"],
            model_name=config["model"]["name"],
        )
        model_path = config["inference"]["model_path"]
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)

        test_ds = ToxicDataset(
            config["data"]["test_path"], tokenizer,
            max_length=config["model"]["max_length"],
        )
        test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"] * 2, shuffle=False)
        metrics = evaluate_from_model(
            model, test_loader, device,
            threshold=config["inference"]["threshold"],
            report_dir=config["logging"]["report_dir"],
        )
        logger.info(f"Test metrics: {metrics}")

    logger.info("\n✅ Training pipeline complete!")
    logger.info(f"  Model saved: {config['inference']['model_path']}")
    logger.info(f"  Tokenizer saved: {config['inference']['tokenizer_path']}")
    logger.info("  Launch UI: python app.py")


if __name__ == "__main__":
    main()
