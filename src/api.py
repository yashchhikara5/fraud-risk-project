"""
api.py — FastAPI service that serves fraud risk scores for incoming
transactions, with SHAP-based "risk reasons" for analyst review.

Run locally:
    uvicorn src.api:app --reload --port 8000

Run via Docker:
    docker build -t fraud-risk-api .
    docker run -p 8000:8000 fraud-risk-api
"""

import time
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
from typing import Dict, Any, List

MODEL_DIR = "models"
TOP_N_REASONS = 5

app = FastAPI(title="Fraud Risk Scoring API", version="0.1.0")

# --- Load model artifacts at startup ---
clf = lgb.Booster(model_file=f"{MODEL_DIR}/fraud_classifier.txt")
prep = joblib.load(f"{MODEL_DIR}/preprocessing.joblib")
iso_forest = joblib.load(f"{MODEL_DIR}/isolation_forest.joblib")

FEATURE_COLS = prep["feature_cols"]
CAT_COLS = prep["cat_cols"]
NUM_COLS = prep["num_cols"]
ENCODERS = prep["encoders"]

explainer = shap.TreeExplainer(clf)


class TransactionFeatures(BaseModel):
    """Input schema — a dict of pre-computed feature values for one transaction.

    In production this would be populated by the feature pipeline
    (src/features.py) reading from the event stream / feature store.
    """
    model_config = ConfigDict(extra="allow")
    features: Dict[str, Any]


class RiskResponse(BaseModel):
    fraud_probability: float
    risk_flag: bool
    anomaly_flag: bool
    top_risk_reasons: List[str]
    latency_ms: float


def preprocess(features: Dict[str, Any]) -> pd.DataFrame:
    row = pd.DataFrame([features])

    # Ensure all expected columns are present
    for col in FEATURE_COLS:
        if col not in row.columns:
            row[col] = -999 if col in NUM_COLS else "missing"

    # Apply saved label encoders to categorical columns
    for col in CAT_COLS:
        le = ENCODERS[col]
        row[col] = row[col].astype(str)
        row[col] = row[col].map(lambda v: v if v in le.classes_ else "missing")
        if "missing" not in le.classes_:
            le.classes_ = np.append(le.classes_, "missing")
        row[col] = le.transform(row[col])

    row[NUM_COLS] = row[NUM_COLS].fillna(-999)
    return row[FEATURE_COLS]


@app.post("/score", response_model=RiskResponse)
def score_transaction(payload: TransactionFeatures):
    start = time.perf_counter()

    X = preprocess(payload.features)

    fraud_proba = float(clf.predict(X)[0])
    risk_flag = fraud_proba >= 0.5

    anomaly_pred = iso_forest.predict(X[NUM_COLS])[0]  # -1 = anomaly, 1 = normal
    anomaly_flag = anomaly_pred == -1

    shap_values = explainer.shap_values(X)
    contributions = pd.Series(shap_values[0], index=FEATURE_COLS)
    top_features = contributions.abs().sort_values(ascending=False).head(TOP_N_REASONS)

    reasons = []
    for feat in top_features.index:
        direction = "increases" if contributions[feat] > 0 else "decreases"
        reasons.append(f"{feat} {direction} risk (impact={contributions[feat]:+.4f})")

    latency_ms = (time.perf_counter() - start) * 1000

    return RiskResponse(
        fraud_probability=round(fraud_proba, 4),
        risk_flag=risk_flag,
        anomaly_flag=bool(anomaly_flag),
        top_risk_reasons=reasons,
        latency_ms=round(latency_ms, 2),
    )


@app.get("/health")
def health():
    return {"status": "ok"}
