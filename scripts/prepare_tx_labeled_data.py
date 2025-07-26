"""
Clean and integrate Ethereum transaction data with labeled wallet metadata.
"""

import pandas as pd
from pathlib import Path

# --- Paths
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# --- Load Transactions
tx_df = pd.read_csv(RAW_DIR / "transactions_6months.csv")
tx_df.columns = tx_df.columns.str.strip().str.lower()

# Normalize address casing
tx_df['from_address'] = tx_df['from_address'].str.lower()
tx_df['to_address'] = tx_df['to_address'].str.lower()

# Parse timestamp
tx_df['block_timestamp'] = pd.to_datetime(tx_df['block_timestamp'], utc=True)

# --- Load Labels
def load_labels(filename, label_value):
    df = pd.read_csv(RAW_DIR / filename, sep="\t")
    df.columns = df.columns.str.strip().str.lower()
    df = df[['address', 'label']].copy()
    df['address'] = df['address'].str.lower()
    return df

criminal_df = load_labels("real_cats_criminal_eth.tsv", "criminal")
benign_df = load_labels("real_cats_benign_eth.tsv", "benign")
label_df = pd.concat([criminal_df, benign_df], ignore_index=True)

# --- Label Merging
tx_df['from_label'] = tx_df['from_address'].map(dict(zip(label_df.address, label_df.label)))
tx_df['to_label'] = tx_df['to_address'].map(dict(zip(label_df.address, label_df.label)))

# --- Save Cleaned Output
output_path = PROCESSED_DIR / "tx_labeled_combined.parquet"
tx_df.to_parquet(output_path, index=False)
print(f"Saved: {output_path}")
print(tx_df[['from_address', 'to_address', 'eth_value', 'from_label', 'to_label']].sample(5))
