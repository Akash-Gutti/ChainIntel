"""
Module 6: Wallet Clustering with KMeans and DBSCAN.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import json

# --- Paths ---
INPUT_PATH = Path("data/processed/tx_features.parquet")
OUTPUT_CSV = Path("data/processed/wallet_clusters.csv")
OUTPUT_JSON = Path("data/processed/wallet_cluster_summary.json")
HEATMAP_IMG = Path("docs/screenshots/cluster_heatmap.png")
DISTPLOT_IMG = Path("docs/screenshots/cluster_dist.png")

def load_and_prepare():
    df = pd.read_parquet(INPUT_PATH)
    if 'label' in df.columns:
        df = df[df['label'] >= 0]  # Optional: ignore unlabeled if present
    wallets = df['wallet']
    X = df.drop(columns=['wallet'] + (['label'] if 'label' in df.columns else []))
    return df, wallets, X

def cluster_wallets(X_scaled, wallets):
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)

    dbscan = DBSCAN(eps=1.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)

    cluster_df = pd.DataFrame({
        'wallet': wallets,
        'kmeans_cluster': kmeans_labels,
        'dbscan_cluster': dbscan_labels
    })
    return cluster_df, kmeans_labels, dbscan_labels

def save_summary(kmeans_labels, dbscan_labels, X_scaled):
    summary = {
        "kmeans": {
            "n_clusters": int(np.unique(kmeans_labels).shape[0]),
            "silhouette_score": float(silhouette_score(X_scaled, kmeans_labels))
        },
        "dbscan": {
            "n_clusters": int(len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)),
            "noise_points": int(np.sum(dbscan_labels == -1))
        }
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print("Cluster summary saved:", OUTPUT_JSON)

def generate_visuals(X_scaled, kmeans_labels, features):
    sns.set_theme(style="whitegrid")
    corr_matrix = pd.DataFrame(X_scaled, columns=features).corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(HEATMAP_IMG)
    print("Saved:", HEATMAP_IMG)

    plt.figure(figsize=(8, 5))
    sns.countplot(x=kmeans_labels)
    plt.title("KMeans Cluster Distribution")
    plt.xlabel("Cluster ID")
    plt.ylabel("Wallet Count")
    plt.tight_layout()
    plt.savefig(DISTPLOT_IMG)
    print("Saved:", DISTPLOT_IMG)

def main():
    df, wallets, X = load_and_prepare()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cluster_df, kmeans_labels, dbscan_labels = cluster_wallets(X_scaled, wallets)
    cluster_df.to_csv(OUTPUT_CSV, index=False)
    print("Cluster assignments saved:", OUTPUT_CSV)

    save_summary(kmeans_labels, dbscan_labels, X_scaled)
    generate_visuals(X_scaled, kmeans_labels, features=X.columns)

if __name__ == "__main__":
    main()
