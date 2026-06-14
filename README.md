# Fraud & Account Risk Scoring — IEEE-CIS Fraud Detection

End-to-end ML pipeline for detecting fraudulent transactions using the
[IEEE-CIS Fraud Detection dataset](https://www.kaggle.com/c/ieee-fraud-detection)
(Kaggle). The dataset combines transaction-level data with device/identity
signals — a close real-world analogue to fraud, account-takeover, and
risk-scoring problems in fintech and digital-platform fraud prevention.

## What this project does

1. **Data loading & cleaning** — merges transaction and identity tables,
   handles missing values and high-cardinality categorical features.
2. **Feature engineering** — builds 40+ behavioral, device, and velocity
   features:
   - Transaction amount aggregates (mean/std/z-score by card, by email domain)
   - Time-based velocity features (transactions per card in last N hours)
   - Device/browser fingerprint signals (`DeviceType`, `DeviceInfo`, `id_30`–`id_38`)
   - Email domain risk encoding (`P_emaildomain`, `R_emaildomain`)
   - Categorical frequency encoding for high-cardinality fields (`card1`–`card6`, `addr1`, `addr2`)
3. **Supervised model** — gradient-boosted classifier (LightGBM/XGBoost) for
   fraud probability scoring, evaluated with precision, recall, F1, AUC, and
   calibration curves.
4. **Unsupervised anomaly detection** — Isolation Forest on the same feature
   set to flag bot-like / synthetic-identity behavior without relying on labels.
5. **Explainability** — SHAP values to identify top fraud drivers per
   prediction, framed as "risk reasons" an analyst could review.
6. **Model serving** — a minimal FastAPI service that loads the trained model
   and returns a fraud risk score + top SHAP reasons for a given transaction,
   containerized with Docker.
7. **Monitoring** — a script that computes feature drift (PSI) and tracks
   precision/recall over rolling windows, simulating production monitoring.

## Setup

```bash
pip install -r requirements.txt
```

Download the dataset from Kaggle (requires a free Kaggle account):

```bash
kaggle competitions download -c ieee-fraud-detection -p data/
unzip data/ieee-fraud-detection.zip -d data/
```

## Run the pipeline

```bash
# 1. Feature engineering
python src/features.py

# 2. Train models
python src/train.py

# 3. Evaluate
python src/evaluate.py

# 4. (Optional) Run SHAP explainability
python src/explain.py

# 5. Serve the model
docker build -t fraud-risk-api .
docker run -p 8000:8000 fraud-risk-api
```

## Project structure

```
fraud-risk-project/
├── data/                # raw + processed data (not committed)
├── notebooks/
│   └── eda_and_modeling.ipynb
├── src/
│   ├── features.py      # feature engineering pipeline
│   ├── train.py          # trains classifier + isolation forest
│   ├── evaluate.py        # precision/recall/AUC/calibration
│   ├── explain.py         # SHAP explainability
│   ├── monitor.py         # drift + performance monitoring
│   └── api.py             # FastAPI serving layer
├── models/               # saved model artifacts
├── Dockerfile
├── requirements.txt
└── README.md
```

## Results

After running the pipeline on the full dataset, fill in your actual numbers
here (these are what you'll cite on your resume):

- **AUC**: 0.9112
- **Precision @ chosen threshold**: 0.792
- **Recall @ chosen threshold**: 0.329
- **False positive rate**: 350
- **API p95 latency**: ___ ms (from `src/api.py` benchmark)

## Notes

This is a personal project built to explore production-style fraud detection
workflows: feature engineering on messy real-world data, model evaluation
trade-offs (precision vs. recall vs. business cost), explainability for
non-ML stakeholders, and basic MLOps (serving + monitoring).
# fraud-risk-project
# fraud-risk-project
