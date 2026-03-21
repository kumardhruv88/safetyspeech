# SafetySpeech

**A Production-Grade Multi-Label Toxic Content Detection System**

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-Hugging_Face-yellow?style=for-the-badge&logo=huggingface)
![Gradio](https://img.shields.io/badge/Gradio-UI-ff69b4?style=for-the-badge)

[Launch Live Application](https://huggingface.co/spaces/aryan012234/safetyspeech-app)

---

## Overview

SafetySpeech is a high-accuracy AI system engineered to analyze social media text, forum posts, and user-generated content — automatically identifying harmful material and classifying it by category for downstream human review. The system is built on a fine-tuned BERT transformer and performs multi-label classification across four behavioral categories:

- **Normal** — Safe, non-toxic content
- **Depression** — Expressions of hopelessness, self-harm ideation, or acute grief
- **Hate Speech** — Targeted abuse based on race, gender, religion, or political affiliation
- **Violent** — Threats, incitement to violence, or language promoting physical harm

---

## Model & Technical Specifications

SafetySpeech is fine-tuned on top of a state-of-the-art NLP transformer backbone, optimized specifically for extreme behavioral context recognition.

| Parameter | Value |
|---|---|
| Base Architecture | `bert-base-uncased` |
| Total Parameters | ~110 Million |
| Vocabulary Size | 30,522 tokens |
| Max Sequence Length | 128 tokens |
| Optimizer | AdamW (LR: 2e-5) |
| Loss Function | `BCEWithLogitsLoss` with positive weight balancing |
| Training Hardware | NVIDIA T4 GPU ×2 |
| Training Duration | ~60 minutes over 5 epochs |

### Performance Metrics

All metrics are evaluated against a held-out test split unseen during training:

| Metric | Score |
|---|---|
| Overall Micro F1 | 0.8489 |
| Overall Macro F1 | 0.6299 |
| Hate Speech F1 | > 0.81 |
| Depressive Content F1 | > 0.79 |

### Training Data

The model was trained on a curated, multi-source corpus of approximately 330,000 human-annotated samples spanning a broad range of toxicity signals:

| Dataset | Volume | Coverage |
|---|---|---|
| Jigsaw Toxic Comments | ~160,000 | General toxicity, threats, insults |
| Davidson Hate Speech | ~25,000 | Hate speech vs. offensive language |
| Depression Reddit | ~7,700 | Clinical depression indicators |
| UCSD Measuring Hate Speech | ~135,000 | Diverse hate speech benchmarks |

---

## System Architecture

The end-to-end pipeline spans data ingestion, preprocessing, model fine-tuning, and real-time inference via a hosted Transformer backend.

```mermaid
graph TD
    A[Data Sources: Twitter, Reddit] -->|Collection Scripts| B(Raw Dataset)
    B -->|Preprocessing & Tokenization| C(Processed Dataset)
    C -->|BCEWithLogitsLoss| D[GuardianBERT Fine-Tuning]
    D -->|Export| E[(Model Weights & Vocab)]
    E -.->|hf_hub_download| F[SafetySpeech App Server]
    G[User Input Text] --> F
    F -->|Sigmoid Activation| H{Multi-Label Output}
    H -->|Scores < 0.5| I[Safe]
    H -->|Scores > 0.5| J[Flagged for Review]
```

---

## Inference Pipeline

```mermaid
sequenceDiagram
    participant User
    participant Web Interface
    participant BERT Model
    
    User->>Web Interface: Submits Social Media Text
    Web Interface->>BERT Model: Tokenizes & Encodes Input
    BERT Model-->>Web Interface: Returns 4-Dimensional Logits
    Web Interface->>Web Interface: Applies Sigmoid & Thresholding
    Web Interface-->>User: Displays Confidence Scores & Severity Labels
```

---

## Project Structure

```text
safetyspeech/
├── data/
│   ├── raw/                 # Raw downloaded CSV data
│   └── processed/           # Cleaned train/val/test splits
├── models/
│   └── checkpoints/         # Pre-trained .pt weights
│       └── tokenizer/       # BERT vocabulary parameters
└── src/
    ├── collect/             # Scrapers for Reddit/X
    ├── preprocess/          # Text cleaning and tokenization
    ├── models/              # GuardianBERT PyTorch class definitions
    ├── inference/           # Real-time predictor modules
    └── ui/                  # Gradio application server
```

---

## Local Development Setup

### 1. Environment Configuration

```bash
python -m venv guardian_env
source guardian_env/bin/activate  # Windows: guardian_env\Scripts\activate
pip install -r requirements.txt
python setup_structure.py
```

### 2. Model Training

```bash
python train.py --config config.yaml
```

### 3. Running the Application

```bash
python app.py
```

---

*SafetySpeech is intended strictly for research purposes and AI safety moderation assistance. It is not a substitute for human judgment.*
