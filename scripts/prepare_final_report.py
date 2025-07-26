'''
Script: prepare_final_report.py
Step: Module 8.1 - Aggregate Wallet Intelligence
Goal: Merge anomaly scores, clustering info, and LLM summaries into a single wallet-level risk report.
'''

import pandas as pd
import json
from pathlib import Path

# --- Paths ---
ANOMALY_PATH = Path("data/processed/tx_anomaly_scores.csv")
CLUSTERS_PATH = Path("data/processed/inference_wallet_clusters.csv")
SUMMARIES_PATH = Path("data/processed/wallet_summaries.json")
OUTPUT_PARQUET = Path("data/processed/wallet_risk_report.parquet")
OUTPUT_CSV = Path("data/processed/wallet_risk_report.csv")


def load_data():
    df_anomaly = pd.read_csv(ANOMALY_PATH)
    df_clusters = pd.read_csv(CLUSTERS_PATH)
    summaries = json.loads(SUMMARIES_PATH.read_text())
    return df_anomaly, df_clusters, summaries


def merge_and_export(df_anomaly, df_clusters, summaries):
    df = df_anomaly.merge(df_clusters, on="wallet", how="left")

    # Convert summary JSON to DataFrame
    summary_df = pd.DataFrame.from_dict(summaries, orient="index")
    summary_df.index.name = "wallet"
    summary_df.reset_index(inplace=True)

    # Merge with main
    df = df.merge(summary_df, on="wallet", how="left")

    # Export
    df.to_parquet(OUTPUT_PARQUET, index=False)
    df.to_csv(OUTPUT_CSV, index=False)
    print("Exported wallet risk report:")
    print("-", OUTPUT_PARQUET)
    print("-", OUTPUT_CSV)


def main():
    df_anomaly, df_clusters, summaries = load_data()
    merge_and_export(df_anomaly, df_clusters, summaries)


if __name__ == "__main__":
    main()
