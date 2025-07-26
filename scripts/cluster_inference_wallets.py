"""
Module 6b: Clustering Unlabeled (Inference) Wallets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json

# --- Paths ---
INPUT_PATH = Path("data/processed/tx_features.parquet")
OUTPUT_CSV = Path("data/processed/inference_wallet_clusters.csv")
OUTPUT_JSON = Path("data/processed/inference_cluster_summary.json")

def main():
    df = pd.read_parquet(INPUT_PATH)
    df = df[df['label'] == -1]  # Select only inference wallets

    if df.empty:
        print("No inference wallets to cluster.")
        return

    wallets = df["wallet"]
    X = df.drop(columns=["wallet", "label"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # KMeans Clustering
    kmeans = KMeans(n_clusters=8, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    df["cluster_id"] = labels

    # Save assignments
    df_out = pd.DataFrame({"wallet": wallets, "cluster_id": labels})
    df_out.to_csv(OUTPUT_CSV, index=False)
    print("Cluster assignments saved:", OUTPUT_CSV)

    # Save summary
    summary = {
        "n_clusters": int(np.unique(labels).shape[0]),
        "silhouette_score": float(silhouette_score(X_scaled, labels)),
        "wallets_clustered": int(len(df))
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print("Cluster summary saved:", OUTPUT_JSON)

if __name__ == "__main__":
    main()
