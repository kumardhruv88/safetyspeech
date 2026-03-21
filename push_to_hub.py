"""
push_to_hub.py
--------------
Pushes the trained GUARDIAN-NLP model and tokenizer to HuggingFace Hub.

Requirements:
  - huggingface_hub installed
  - HF_TOKEN in .env or set via `huggingface-cli login`
  - HF_REPO_ID in .env (format: your_username/guardian-nlp)

Usage:
    python push_to_hub.py
    python push_to_hub.py --repo your_username/guardian-nlp --model models/checkpoints/best_model.pt
"""

import argparse
import os

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(description="GUARDIAN-NLP: Push to HuggingFace Hub")
    parser.add_argument("--repo", type=str,
                        default=os.getenv("HF_REPO_ID", "your_username/guardian-nlp"),
                        help="HuggingFace repo ID (username/repo-name)")
    parser.add_argument("--model", type=str, default="models/checkpoints/best_model.pt")
    parser.add_argument("--tokenizer", type=str, default="models/checkpoints/tokenizer/")
    parser.add_argument("--private", action="store_true", help="Create private repository")
    return parser.parse_args()


def push_to_hub(args):
    try:
        from huggingface_hub import HfApi, upload_file, create_repo
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        return

    token = os.getenv("HF_TOKEN")
    if not token:
        logger.warning("HF_TOKEN not set in .env. Using cached credentials from huggingface-cli login.")

    api = HfApi()

    # Create or ensure the repo exists
    logger.info(f"Creating/verifying repo: {args.repo}")
    try:
        create_repo(
            repo_id=args.repo,
            repo_type="model",
            private=args.private,
            token=token,
            exist_ok=True,
        )
        logger.info(f"✅ Repo ready: https://huggingface.co/{args.repo}")
    except Exception as e:
        logger.error(f"Could not create repo: {e}")
        return

    # Upload model weights
    if os.path.exists(args.model):
        logger.info(f"Uploading model weights: {args.model}")
        api.upload_file(
            path_or_fileobj=args.model,
            path_in_repo="best_model.pt",
            repo_id=args.repo,
            token=token,
        )
        logger.info("✅ Model weights uploaded.")
    else:
        logger.error(f"Model file not found: {args.model}")

    # Upload tokenizer directory
    if os.path.isdir(args.tokenizer) and os.listdir(args.tokenizer):
        logger.info(f"Uploading tokenizer: {args.tokenizer}")
        api.upload_folder(
            folder_path=args.tokenizer,
            repo_id=args.repo,
            repo_type="model",
            path_in_repo="tokenizer",
            token=token,
        )
        logger.info("✅ Tokenizer uploaded.")
    else:
        logger.error(f"Tokenizer directory not found or empty: {args.tokenizer}")

    # Upload README and config
    for file in ["README.md", "config.yaml", "requirements.txt"]:
        if os.path.exists(file):
            api.upload_file(
                path_or_fileobj=file,
                path_in_repo=file,
                repo_id=args.repo,
                token=token,
            )

    logger.info(f"\n🎉 Successfully pushed to HuggingFace Hub!")
    logger.info(f"   Model URL: https://huggingface.co/{args.repo}")
    logger.info(f"   To load in inference: BertTokenizer.from_pretrained('{args.repo}/tokenizer')")


if __name__ == "__main__":
    args = parse_args()
    push_to_hub(args)
