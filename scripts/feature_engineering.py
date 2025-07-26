"""
Module 2: Feature Engineering from labeled Ethereum transaction dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from scipy.stats import entropy

RAW_PATH = Path("data/processed/tx_labeled_combined.parquet")
OUT_PATH = Path("data/processed/tx_features.parquet")

def calculate_entropy(to_addresses):
    counts = Counter(to_addresses)
    probs = np.array(list(counts.values())) / sum(counts.values())
    return entropy(probs, base=2)

def main():
    df = pd.read_parquet(RAW_PATH)
    df['block_timestamp'] = pd.to_datetime(df['block_timestamp'], utc=True)
    grouped = df.groupby('from_address')

    # Compute features
    features = pd.DataFrame({
        'tx_count': grouped.size(),
        'unique_to_count': grouped['to_address'].nunique(),
        'eth_sent_total': grouped['eth_value'].sum(),
        'gas_price_avg': grouped['gas_price'].mean(),
        'gas_price_std': grouped['gas_price'].std(),
        'self_tx_count': grouped.apply(lambda g: (g['from_address'] == g['to_address']).sum()),
        'avg_eth_per_tx': grouped['eth_value'].mean(),
        'contract_interaction_rate': grouped['input'].apply(lambda x: (x.str.len() > 10).mean()),
        'active_days': grouped['block_timestamp'].agg(lambda x: (x.max() - x.min()).days + 1),
        'tx_velocity': grouped.size() / grouped['block_timestamp'].agg(lambda x: (x.max() - x.min()).days + 1),
        'tx_entropy': grouped['to_address'].agg(calculate_entropy),
    }).reset_index().rename(columns={'from_address': 'wallet'})

    # --- NaN Diagnostics
    null_cols = features.columns[features.isnull().any()]
    if not null_cols.empty:
        print("\n⚠️ Columns with NaNs:")
        print(features[null_cols].isnull().sum())

    # --- Safe fill
    features.fillna({
        'gas_price_std': 0,
        'contract_interaction_rate': 0,
        'tx_entropy': 0,
        'gas_price_avg': 0,
        'avg_eth_per_tx': 0,
        'active_days': 1,
        'tx_velocity': 0,
        'eth_sent_total': 0
    }, inplace=True)

    # Add target label
    df_labels = df[['from_address', 'from_label']].dropna().drop_duplicates()
    df_labels = df_labels.rename(columns={'from_address': 'wallet', 'from_label': 'label'})

    features = features.merge(df_labels, on='wallet', how='left')
    features['label'] = features['label'].map({'benign': 0, 'Other': 1, 'Hack Scam': 1, 'Metamorphic Contract': 1})
    features['label'] = features['label'].fillna(-1).astype(int)

    # Final checks
    assert not features.drop(columns=['label']).isnull().any().any(), "NaNs in features"
    assert features['wallet'].is_unique, "Duplicate wallets found"

    OUT_PATH.parent.mkdir(exist_ok=True, parents=True)
    features.to_parquet(OUT_PATH, index=False)
    print(f"Features saved to: {OUT_PATH}")
    print(features.sample(5))


if __name__ == "__main__":
    main()
