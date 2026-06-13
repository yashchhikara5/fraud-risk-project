"""
benchmark_api.py — Measures p50/p95/p99 latency of the /score endpoint
under repeated requests, to produce the latency numbers cited in the
project README / resume.

Usage:
    python src/benchmark_api.py
"""

import time
import requests
import numpy as np
import pandas as pd

API_URL = "http://localhost:8000/score"
N_REQUESTS = 500


def sample_payload(val_df, feature_cols):
    row = val_df.sample(1, random_state=np.random.randint(0, 100000))
    return {"features": row[feature_cols].iloc[0].to_dict()}


def main():
    val_df = pd.read_parquet("data/val_split.parquet")
    feature_cols = [c for c in val_df.columns if c != "isFraud"]

    latencies = []
    for _ in range(N_REQUESTS):
        payload = sample_payload(val_df, feature_cols)
        start = time.perf_counter()
        resp = requests.post(API_URL, json=payload, timeout=5)
        end = time.perf_counter()
        if resp.status_code == 200:
            latencies.append((end - start) * 1000)

    latencies = np.array(latencies)
    print(f"Requests: {len(latencies)}")
    print(f"p50: {np.percentile(latencies, 50):.2f} ms")
    print(f"p95: {np.percentile(latencies, 95):.2f} ms")
    print(f"p99: {np.percentile(latencies, 99):.2f} ms")
    print(f"max: {latencies.max():.2f} ms")


if __name__ == "__main__":
    main()
