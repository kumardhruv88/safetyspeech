# 🛡️ GUARDIAN-NLP
### United Nations NGO × Social Media Platform AI Safety Division
**GUARDIAN-NLP** is a production-grade, multi-label toxic content detection system powered by fine-tuned BERT, designed for UN analysts and NGO moderators to scan, flag, and report harmful content at scale.

---

## 🎯 What It Detects

| Label | Description |
|-------|-------------|
| 🟢 Normal | Safe, non-toxic content |
| 🟡 Depressive | Hopelessness, self-harm ideation, grief expressions |
| 🟠 Hate Speech | Racial, gender, religious, political targeting |
| 🔴 Violent | Threats, incitement, physical harm |

Each prediction includes:
- **Confidence scores** (0.0–1.0) per label
- **Severity level**: SAFE 🟢 / LOW 🟡 / MEDIUM 🟠 / HIGH 🔴
- **Source platform tag** (Twitter, Reddit, Instagram, etc.)

---

## 🏗️ Architecture

```
[Raw Text Input]
      ↓
[Data Collection Module] → data/raw/
      ↓
[Preprocessing Pipeline] → data/processed/
      ↓
[BERT Multi-Label Classifier] ← fine-tuned on toxic datasets
      ↓
[Confidence Score Generator] ← sigmoid outputs
      ↓
[Gradio Analyzer / Streamlit Dashboard]
      ↓
[UN/NGO Dashboard: flagged content, stats, CSV/PDF export]
```

**Model Architecture:**
- **Backbone**: `bert-base-uncased` (HuggingFace Transformers)
- **Head**: Multi-label classification (4 output nodes + sigmoid)
- **Loss**: `BCEWithLogitsLoss` with pos_weight for class imbalance
- **Optimizer**: AdamW + linear warmup scheduler

---

## 📂 Project Structure

```
guardian_nlp/
├── data/
│   ├── raw/              ← raw collected CSV files
│   ├── processed/        ← train.csv, val.csv, test.csv
│   └── external/         ← Jigsaw, Davidson, Depression Reddit datasets
├── src/
│   ├── collect/          ← Twitter + Reddit scrapers
│   ├── preprocess/       ← TextCleaner, tokenizer, label encoder
│   ├── models/           ← GuardianBERT, ToxicDataset, trainer
│   ├── evaluate/         ← metrics (F1, AUC, Hamming)
│   ├── inference/        ← Predictor class
│   └── ui/               ← Gradio app + Streamlit dashboard
├── models/checkpoints/   ← saved model weights + tokenizer
├── notebooks/            ← 4 Jupyter notebooks
├── outputs/              ← reports + visualizations
├── config.yaml           ← all hyperparameters
├── train.py              ← training entry point
├── app.py                ← app launcher
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
python -m venv guardian_env
# Windows:
guardian_env\Scripts\activate
# Linux/Mac:
source guardian_env/bin/activate

pip install -r requirements.txt
python setup_structure.py
```

### 2. Configure API Credentials
```bash
cp .env.example .env
# Edit .env with your Reddit API credentials
```

### 3. Download Datasets
```bash
# Jigsaw (requires Kaggle API):
kaggle competitions download -c jigsaw-toxic-comment-classification-challenge
# Davidson dataset:
git clone https://github.com/t-davidson/hate-speech-and-offensive-language data/external/davidson
# Depression Reddit (HuggingFace):
python -c "from datasets import load_dataset; ds = load_dataset('mrjunos/depression-reddit-cleaned'); ds['train'].to_csv('data/external/depression_reddit.csv')"
```

### 4. Collect Live Data
```bash
python -m src.collect.reddit_collector
python -m src.collect.twitter_collector
python -m src.collect.data_merger
```

### 5. Preprocess Data
```bash
python -m src.preprocess.cleaner --input data/raw/combined_raw.csv
python -m src.preprocess.label_encoder
python -m src.preprocess.splitter --stratify label
```

### 6. Train Model
```bash
python train.py --config config.yaml
# Expected: ~3-5 hours on GPU, ~12-18 hours on CPU
```

### 7. Launch UI
```bash
# Gradio single-text analyzer:
python src/ui/gradio_app.py
# → http://localhost:7860

# Streamlit NGO dashboard:
streamlit run src/ui/streamlit_dashboard.py
# → http://localhost:8501

# Or launch both:
python app.py
```

---

## 📊 Datasets Used

| Dataset | Size | Labels |
|---------|------|--------|
| [Jigsaw Toxic Comments](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) | ~160K | toxic, severe_toxic, obscene, threat, insult, identity_hate |
| [Davidson Hate Speech](https://github.com/t-davidson/hate-speech-and-offensive-language) | ~25K | hate_speech, offensive, neither |
| [Depression Reddit](https://huggingface.co/datasets/mrjunos/depression-reddit-cleaned) | ~7.7K | depression / non-depression |
| [UCSD Measuring Hate Speech](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech) | ~135K | hate_speech_score |

---

## 🎯 Benchmark Targets

| Metric | Target |
|--------|--------|
| Micro F1 | > 0.82 |
| AUC-ROC | > 0.88 per label |
| Hate Speech F1 | > 0.80 |
| Depressive F1 | > 0.78 |

---

## 🐳 Docker Deployment

```bash
docker build -t guardian-nlp .
docker run -p 8501:8501 guardian-nlp
```

## ☁️ HuggingFace Spaces Deployment
```bash
huggingface-cli login
python push_to_hub.py
```

---

## ⚠️ Ethical Notes
- User PII (usernames, account IDs) is hashed/removed before any training
- Content is analyzed for safety only — not stored permanently
- Add confidence threshold (>0.65) to reduce false positives
- Include "Report False Positive" feedback loop in production UI

---

## 📝 License
Research use only. Compliant with Kaggle CC0, MIT (Davidson), CC-BY-4.0 (UCSD) licenses.
