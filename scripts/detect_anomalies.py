"""
Module 5: Detect anomalies using Isolation Forest.
"""

import pandas as pd
from sklearn.ensemble import IsolationForest
from pathlib import Path

FEATURE_PATH = Path("data/processed/tx_features.parquet")
OUTPUT_PATH = Path("data/processed/tx_anomaly_scores.csv")

def main():
    df = pd.read_parquet(FEATURE_PATH)
    features = df.drop(columns=["wallet", "label"], errors="ignore")

    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    df["anomaly_score"] = model.fit_predict(features)

    df["anomaly_score"] = df["anomaly_score"].map({1: 0, -1: 1})  # 1 = anomaly
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved anomaly scores to: {OUTPUT_PATH}")

    flagged = df[df["anomaly_score"] == 1]
    print(f"Wallets flagged as anomalous: {len(flagged)}")

    # Optional: print examples
    print(flagged[["wallet", "anomaly_score"]].head())

if __name__ == "__main__":
    main()
