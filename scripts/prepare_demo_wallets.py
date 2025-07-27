import pandas as pd
import json
from pathlib import Path

# Paths
RISK_CSV = Path("data/processed/wallet_risk_report.csv")
SUMMARY_JSON = Path("data/processed/wallet_summaries.json")
OUTPUT_CSV = Path("data/processed/demo_wallets.csv")

# Load full data
df = pd.read_csv(RISK_CSV, low_memory=False)
df = df.rename(columns={"cluster_id_y": "cluster_id", "anomaly_score_y": "anomaly_score"})

# Load summaries
with open(SUMMARY_JSON) as f:
    summaries = json.load(f)

# Extract 300 summary wallets
summary_wallets = list(summaries.keys())
df_summary = df[df.wallet.isin(summary_wallets)]

# Select top 500 anomaly-score wallets not already in summary list
df_top_anomaly = df[~df.wallet.isin(summary_wallets)].sort_values("anomaly_score", ascending=False).head(500)

# Combine and save
demo_df = pd.concat([df_summary, df_top_anomaly]).drop_duplicates(subset="wallet")
demo_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved demo_wallets.csv with {len(demo_df)} rows")
