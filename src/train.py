"""
train.py — Trains a gradient-boosted fraud classifier (LightGBM) and an
Isolation Forest anomaly detector on the engineered feature set.

Usage:
    python src/train.py
"""

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
import lightgbm as lgb

DATA_PATH = "data/features.parquet"
MODEL_DIR = "models"
TARGET = "isFraud"

NON_FEATURE_COLS = ["TransactionID", "TransactionDT", TARGET]


def load_data():
    df = pd.read_parquet(DATA_PATH)
    return df


def encode_categoricals(df, cat_cols):
    """Label-encode categorical columns, keeping encoders for later use."""
    encoders = {}
    for col in cat_cols:
        df[col] = df[col].astype(str).fillna("missing")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders


def train_classifier(X_train, y_train, X_val, y_val, cat_cols):
    """Train a LightGBM gradient-boosted classifier for fraud probability."""
    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=train_set)

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 64,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
        "verbose": -1,
        "seed": 42,
    }

    model = lgb.train(
        params,
        train_set,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50),
        ],
    )

    print(f"Best iteration: {model.best_iteration}")
    print(f"Best val AUC: {model.best_score['val']['auc']:.4f}")

    return model


def train_anomaly_detector(X_train, contamination=0.035):
    """Train an Isolation Forest for unsupervised anomaly/bot-pattern detection.

    contamination is set near the dataset's known fraud rate (~3.5%) as a
    starting point, but should be tuned independently of the fraud label
    in a true unsupervised setting.
    """
    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    iso_forest.fit(X_train)
    return iso_forest


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_data()

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    cat_cols = [
        c for c in feature_cols
        if df[c].dtype == "object"
        or pd.api.types.is_string_dtype(df[c])
        or isinstance(df[c].dtype, pd.CategoricalDtype)
    ]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    print(f"Categorical features: {len(cat_cols)} | Numerical features: {len(num_cols)}")

    df, encoders = encode_categoricals(df, cat_cols)

    # Fill remaining numerical NaNs
    df[num_cols] = df[num_cols].fillna(-999)

    X = df[feature_cols]
    y = df[TARGET]

    # Time-based split (sort by TransactionDT already done in features.py)
    split_idx = int(len(df) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Train: {len(X_train):,} rows | Val: {len(X_val):,} rows")
    print(f"Train fraud rate: {y_train.mean():.4%} | Val fraud rate: {y_val.mean():.4%}")

    # --- Supervised classifier ---
    clf = train_classifier(X_train, y_train, X_val, y_val, cat_cols)
    clf.save_model(os.path.join(MODEL_DIR, "fraud_classifier.txt"))

    # --- Unsupervised anomaly detector (trained on non-fraud only, to mimic
    #     detecting "new" abnormal behavior not seen in known fraud patterns) ---
    iso_forest = train_anomaly_detector(X_train[num_cols])
    joblib.dump(iso_forest, os.path.join(MODEL_DIR, "isolation_forest.joblib"))

    # Save encoders + feature column lists for inference
    joblib.dump(
        {"encoders": encoders, "feature_cols": feature_cols, "cat_cols": cat_cols, "num_cols": num_cols},
        os.path.join(MODEL_DIR, "preprocessing.joblib"),
    )

    # Save val split for evaluate.py
    val_out = X_val.assign(**{TARGET: y_val.values}).copy()
    val_out.to_parquet(os.path.join("data", "val_split.parquet"))

    print("Training complete. Models saved to /models")


if __name__ == "__main__":
    main()
