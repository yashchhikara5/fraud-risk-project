"""
evaluate.py — Evaluates the trained fraud classifier on the validation set:
precision, recall, F1, AUC, and calibration curve. Also reports the
business-relevant trade-off between false positives and false negatives at
several decision thresholds.

Usage:
    python src/evaluate.py
"""

import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve

MODEL_DIR = "models"
TARGET = "isFraud"


def load_artifacts():
    clf = lgb.Booster(model_file=f"{MODEL_DIR}/fraud_classifier.txt")
    prep = joblib.load(f"{MODEL_DIR}/preprocessing.joblib")
    val_df = pd.read_parquet("data/val_split.parquet")
    return clf, prep, val_df


def evaluate_at_thresholds(y_true, y_proba, thresholds=(0.3, 0.5, 0.7, 0.9)):
    """Print precision/recall/FP/FN counts at several thresholds to illustrate
    the cost trade-off of stricter vs. looser fraud thresholds."""
    print("\nThreshold | Precision | Recall | F1 | False Positives | False Negatives")
    print("-" * 75)
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        print(f"{t:>9.2f} | {precision:>9.3f} | {recall:>6.3f} | {f1:>5.3f} | {fp:>16,} | {fn:>16,}")


def main():
    clf, prep, val_df = load_artifacts()
    feature_cols = prep["feature_cols"]

    X_val = val_df[feature_cols]
    y_val = val_df[TARGET]

    y_proba = clf.predict(X_val)

    auc = roc_auc_score(y_val, y_proba)
    print(f"Validation AUC: {auc:.4f}")

    evaluate_at_thresholds(y_val, y_proba)

    # Precision-recall curve
    precisions, recalls, _ = precision_recall_curve(y_val, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — Fraud Classifier")
    plt.grid(alpha=0.3)
    plt.savefig("models/precision_recall_curve.png", dpi=120, bbox_inches="tight")
    print("Saved precision_recall_curve.png")

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_val, y_proba, n_bins=10)
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed fraud rate")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("models/calibration_curve.png", dpi=120, bbox_inches="tight")
    print("Saved calibration_curve.png")


if __name__ == "__main__":
    main()
