"""
Module 8.2: Generate visualizations from final wallet risk report
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# === Paths ===
REPORT_PATH = Path("data/processed/wallet_risk_report.csv")
SCREENSHOT_DIR = Path("docs/screenshots")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

def plot_anomaly_distribution(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="anomaly_score", palette="Set2")
    plt.title("Anomaly Score Distribution")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Wallet Count")
    plt.tight_layout()
    plt.savefig(SCREENSHOT_DIR / "anomaly_score_dist.png")
    plt.close()

def plot_cluster_distribution(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="cluster_id", palette="Set1")
    plt.title("Cluster Distribution")
    plt.xlabel("Cluster ID")
    plt.ylabel("Wallet Count")
    plt.tight_layout()
    plt.savefig(SCREENSHOT_DIR / "cluster_distribution.png")
    plt.close()

def plot_top_features(df):
    # Flatten top_features text column into keyword list
    feature_counts = {}
    for val in df["top_features"].dropna():
        parts = val.split(", ")
        for part in parts:
            feat = part.split(":")[0].strip()
            feature_counts[feat] = feature_counts.get(feat, 0) + 1

    top_feats = pd.Series(feature_counts).sort_values(ascending=False).head(10)
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=top_feats.values, y=top_feats.index, palette="Blues_d")
    plt.title("Top Features in Risk Summaries")
    plt.xlabel("Frequency")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(SCREENSHOT_DIR / "top_features_barplot.png")
    plt.close()

def save_sample_table(df):
    sample = df[["wallet", "summary", "cluster_id", "anomaly_score"]].head(10)
    sample.to_csv(SCREENSHOT_DIR / "sample_summaries.csv", index=False)

def main():
    print("Loading risk report...")
    df = pd.read_csv(REPORT_PATH, low_memory=False)

    # Rename merged columns if needed
    df = df.rename(columns={
        "anomaly_score_y": "anomaly_score",
        "cluster_id_y": "cluster_id"
    })

    # Ensure correct types
    df["anomaly_score"] = pd.to_numeric(df["anomaly_score"], errors="coerce")
    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce")

    print("Generating visualizations...")
    plot_anomaly_distribution(df)
    plot_cluster_distribution(df)
    plot_top_features(df)
    save_sample_table(df)

    print(f"Visuals and sample table saved to: {SCREENSHOT_DIR}")

if __name__ == "__main__":
    main()
