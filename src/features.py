"""
features.py — Feature engineering pipeline for IEEE-CIS Fraud Detection.

Builds 40+ behavioral, device, and velocity features from raw transaction
and identity tables, designed to mirror real-world fraud/risk-scoring
feature sets (transaction velocity, device fingerprinting, email-domain
risk, categorical frequency encoding).

Usage:
    python src/features.py
"""

import pandas as pd
import re
import numpy as np
import os

DATA_DIR = "data"
OUTPUT_PATH = os.path.join(DATA_DIR, "features.parquet")


def load_raw_data():
    """Load and merge transaction + identity tables."""
    train_tx = pd.read_csv(os.path.join(DATA_DIR, "train_transaction.csv"))
    train_id = pd.read_csv(os.path.join(DATA_DIR, "train_identity.csv"))

    df = train_tx.merge(train_id, on="TransactionID", how="left")
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


def add_time_features(df):
    """Derive time-of-day / day-of-week features from TransactionDT (seconds offset)."""
    df["tx_hour"] = (df["TransactionDT"] // 3600) % 24
    df["tx_day"] = (df["TransactionDT"] // (3600 * 24)) % 7
    return df


def add_amount_features(df):
    """Transaction amount aggregates and z-scores grouped by card."""
    df["amt_log"] = np.log1p(df["TransactionAmt"])

    card_groups = df.groupby("card1")["TransactionAmt"]
    df["card1_amt_mean"] = card_groups.transform("mean")
    df["card1_amt_std"] = card_groups.transform("std").fillna(0)
    df["amt_zscore_card1"] = (
        (df["TransactionAmt"] - df["card1_amt_mean"]) / (df["card1_amt_std"] + 1e-6)
    )

    df["amt_decimal"] = (df["TransactionAmt"] - np.floor(df["TransactionAmt"])).round(2)
    return df


def add_velocity_features(df):
    """Transaction velocity per card over rolling time windows."""
    df = df.sort_values("TransactionDT")

    for window_hours, label in [(1, "1h"), (24, "24h")]:
        window_secs = window_hours * 3600
        df[f"card1_tx_count_{label}"] = (
            df.groupby("card1")["TransactionDT"]
            .transform(lambda x: x.diff().fillna(0).rolling(5, min_periods=1).apply(
                lambda w: (w < window_secs).sum()
            ))
        )

    df["card1_total_tx_count"] = df.groupby("card1")["TransactionID"].transform("count")
    return df


def add_email_domain_features(df):
    """Risk encoding for purchaser / recipient email domains."""
    for col in ["P_emaildomain", "R_emaildomain"]:
        if col in df.columns:
            freq = df[col].value_counts(normalize=True)
            df[f"{col}_freq"] = df[col].map(freq).fillna(0)
            df[f"{col}_is_free"] = df[col].isin(
                ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com"]
            ).astype(int)
    return df


def add_device_features(df):
    """Device/browser fingerprint signals."""
    if "DeviceType" in df.columns:
        df["device_type_freq"] = df["DeviceType"].map(
            df["DeviceType"].value_counts(normalize=True)
        ).fillna(0)

    if "DeviceInfo" in df.columns:
        df["device_info_freq"] = df["DeviceInfo"].map(
            df["DeviceInfo"].value_counts(normalize=True)
        ).fillna(0)
        # Flag rare/unusual device strings (possible emulator/spoofed device)
        device_counts = df["DeviceInfo"].value_counts()
        rare_devices = device_counts[device_counts <= 2].index
        df["device_is_rare"] = df["DeviceInfo"].isin(rare_devices).astype(int)

    # id_30 = OS, id_31 = browser, id_33 = screen resolution (per Kaggle docs)
    for col in ["id_30", "id_31", "id_33"]:
        if col in df.columns:
            df[f"{col}_freq"] = df[col].map(
                df[col].value_counts(normalize=True)
            ).fillna(0)

    return df


def add_categorical_frequency_encoding(df):
    """Frequency-encode high-cardinality categorical fields."""
    high_card_cols = ["card1", "card2", "card3", "card5", "addr1", "addr2"]
    for col in high_card_cols:
        if col in df.columns:
            freq = df[col].value_counts(normalize=True)
            df[f"{col}_freq_enc"] = df[col].map(freq).fillna(0)
    return df


def add_count_aggregations(df):
    """Generic count/nunique aggregations across identity columns (C1-C14, D1-D15)."""
    c_cols = [c for c in df.columns if re.fullmatch(r"C\d+", c)]
    d_cols = [c for c in df.columns if re.fullmatch(r"D\d+", c)]

    if c_cols:
        c_numeric = df[c_cols].apply(pd.to_numeric, errors="coerce")
        df["c_cols_sum"] = c_numeric.sum(axis=1)
        df["c_cols_nonzero"] = (c_numeric > 0).sum(axis=1)
    if d_cols:
        d_numeric = df[d_cols].apply(pd.to_numeric, errors="coerce")
        df["d_cols_mean"] = d_numeric.mean(axis=1)
        df["d_cols_null_count"] = d_numeric.isna().sum(axis=1)

    return df


def build_features():
    df = load_raw_data()
    df = add_time_features(df)
    df = add_amount_features(df)
    df = add_velocity_features(df)
    df = add_email_domain_features(df)
    df = add_device_features(df)
    df = add_categorical_frequency_encoding(df)
    df = add_count_aggregations(df)

    feature_cols = [c for c in df.columns if c not in ("TransactionID",)]
    print(f"Final feature set: {len(feature_cols)} columns")
    print(f"Fraud rate: {df['isFraud'].mean():.4%}")

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved features to {OUTPUT_PATH}")


if __name__ == "__main__":
    build_features()
