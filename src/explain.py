"""
explain.py — Generates SHAP explanations for the fraud classifier,
producing global feature importance and per-prediction "risk reasons"
that an analyst could review.

Usage:
    python src/explain.py
"""

import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt

MODEL_DIR = "models"
TARGET = "isFraud"
TOP_N_REASONS = 5


def load_artifacts():
    clf = lgb.Booster(model_file=f"{MODEL_DIR}/fraud_classifier.txt")
    prep = joblib.load(f"{MODEL_DIR}/preprocessing.joblib")
    val_df = pd.read_parquet("data/val_split.parquet")
    return clf, prep, val_df


def global_importance(explainer, X_sample):
    shap_values = explainer.shap_values(X_sample)
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig("models/shap_summary.png", dpi=120, bbox_inches="tight")
    print("Saved shap_summary.png")
    return shap_values


def explain_single_prediction(explainer, X_row, feature_cols):
    """Return top-N features driving a single prediction, formatted as
    human-readable 'risk reasons'."""
    shap_values = explainer.shap_values(X_row)
    contributions = pd.Series(shap_values[0], index=feature_cols)
    top_features = contributions.abs().sort_values(ascending=False).head(TOP_N_REASONS)

    reasons = []
    for feat in top_features.index:
        direction = "increases" if contributions[feat] > 0 else "decreases"
        reasons.append(f"{feat} {direction} risk (value={X_row[feat].values[0]:.3f}, "
                        f"impact={contributions[feat]:+.4f})")
    return reasons


def main():
    clf, prep, val_df = load_artifacts()
    feature_cols = prep["feature_cols"]
    X_val = val_df[feature_cols]

    # Sample for global importance (SHAP on full dataset is expensive)
    sample = X_val.sample(min(5000, len(X_val)), random_state=42)

    explainer = shap.TreeExplainer(clf)
    global_importance(explainer, sample)

    # Example: explain the highest-risk prediction in the validation set
    y_proba = clf.predict(X_val)
    top_risk_idx = np.argsort(y_proba)[-1]
    top_risk_row = X_val.iloc[[top_risk_idx]]

    print("\nTop predicted-risk transaction — risk reasons:")
    for reason in explain_single_prediction(explainer, top_risk_row, feature_cols):
        print(f"  - {reason}")


if __name__ == "__main__":
    main()
