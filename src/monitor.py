"""
monitor.py — Simulates production monitoring: feature drift (PSI) between
a reference window and current window, plus rolling precision/recall on
labeled data as it becomes available.

Usage:
    python src/monitor.py
"""

import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.metrics import precision_score, recall_score

MODEL_DIR = "models"
TARGET = "isFraud"
N_BINS = 10


def population_stability_index(reference, current, bins=N_BINS):
    """Compute PSI for a single numeric feature between two distributions.

    PSI < 0.1  -> no significant shift
    0.1 - 0.25 -> moderate shift, monitor
    > 0.25     -> significant shift, investigate
    """
    reference = reference.dropna()
    current = current.dropna()

    if reference.empty or current.empty:
        return np.nan

    breakpoints = np.unique(
        np.percentile(reference, np.linspace(0, 100, bins + 1))
    )
    if len(breakpoints) < 2:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    ref_pct = np.where(ref_counts == 0, 1e-6, ref_counts / ref_counts.sum())
    cur_pct = np.where(cur_counts == 0, 1e-6, cur_counts / cur_counts.sum())

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return psi


def feature_drift_report(reference_df, current_df, feature_cols, top_n=10):
    psi_scores = {}
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(reference_df[col]):
            psi_scores[col] = population_stability_index(reference_df[col], current_df[col])

    psi_series = pd.Series(psi_scores).sort_values(ascending=False)
    print(f"\nTop {top_n} features by PSI (drift score):")
    print(psi_series.head(top_n).round(4))

    flagged = psi_series[psi_series > 0.25]
    if not flagged.empty:
        print(f"\n⚠ {len(flagged)} feature(s) show significant drift (PSI > 0.25):")
        print(flagged.round(4))
    else:
        print("\nNo features show significant drift (PSI > 0.25).")

    return psi_series


def rolling_performance(clf, df, feature_cols, window_size=5000):
    """Compute precision/recall over rolling windows to simulate
    production performance tracking over time."""
    X = df[feature_cols]
    y = df[TARGET]
    y_proba = clf.predict(X)
    y_pred = (y_proba >= 0.5).astype(int)

    results = []
    for start in range(0, len(df), window_size):
        end = min(start + window_size, len(df))
        if y.iloc[start:end].sum() == 0:
            continue
        p = precision_score(y.iloc[start:end], y_pred[start:end], zero_division=0)
        r = recall_score(y.iloc[start:end], y_pred[start:end], zero_division=0)
        results.append({"window_start": start, "precision": p, "recall": r})

    return pd.DataFrame(results)


def main():
    clf = lgb.Booster(model_file=f"{MODEL_DIR}/fraud_classifier.txt")
    prep = joblib.load(f"{MODEL_DIR}/preprocessing.joblib")
    feature_cols = prep["feature_cols"]
    num_cols = prep["num_cols"]

    val_df = pd.read_parquet("data/val_split.parquet")

    # Split validation set into "reference" (first half) and "current" (second half)
    # to simulate drift detection between two time periods.
    midpoint = len(val_df) // 2
    reference_df = val_df.iloc[:midpoint]
    current_df = val_df.iloc[midpoint:]

    feature_drift_report(reference_df, current_df, num_cols)

    perf_df = rolling_performance(clf, val_df, feature_cols)
    print("\nRolling precision/recall by window:")
    print(perf_df.round(4))


if __name__ == "__main__":
    main()
