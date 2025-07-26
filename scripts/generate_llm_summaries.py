"""
Module 7: Generate LLM-based wallet summaries (GPT or fallback)
"""

import json
import pandas as pd
from pathlib import Path
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed
from dotenv import load_dotenv
import os

# === Load API key ===
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment or .env file")

client = OpenAI(api_key=api_key)

# === Paths ===
FEATURES_PATH = Path("data/processed/tx_features.parquet")
CLUSTERS_PATH = Path("data/processed/inference_wallet_clusters.csv")
ANOMALY_PATH = Path("data/processed/tx_anomaly_scores.csv")
SHAP_PATH = Path("explainability/shap_values/wallet_shap_values.json")
OUTPUT_JSON_PATH = Path("data/processed/wallet_summaries.json")

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def call_llm(prompt: str):
    """Call OpenAI LLM with retries"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a blockchain forensic analyst. Write concise summaries."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

def format_prompt(wallet_info):
    return f"""
Wallet address: {wallet_info['wallet']}
Cluster ID: {wallet_info['cluster_id']}
Anomaly Flag: {wallet_info['anomaly_flag']}
Top SHAP Features: {wallet_info['top_features']}
Feature Snapshot: {wallet_info['feature_snapshot']}

Write a short risk intelligence summary combining the above.
"""

def main():
    print("Loading data...")
    df = pd.read_parquet(FEATURES_PATH)
    clusters = pd.read_csv(CLUSTERS_PATH)
    anomalies = pd.read_csv(ANOMALY_PATH)
    shap_data = json.loads(Path(SHAP_PATH).read_text()) if SHAP_PATH.exists() else {}

    # Merge inference-only wallets
    df = df[df['label'] == -1].copy()
    df = df.merge(clusters.rename(columns={"kmeans_cluster": "cluster_id"}), on="wallet", how="left")
    df = df.merge(anomalies, on="wallet", how="left")

    # Prioritize high-risk wallets only
    df = df[(df["anomaly_score"] == 1) | (df["cluster_id"].isin([0, 1]))]

    # Budget-aware LLM cap
    df = df.head(300)
    print(f"Generating LLM summaries for {len(df)} wallets...")

    summaries = {}

    for _, row in df.iterrows():
        wallet = row["wallet"]
        shap_vals = shap_data.get(wallet, [])
        if shap_vals:
            top_feats = sorted(shap_vals, key=lambda x: abs(x["shap_value"]), reverse=True)[:3]
            top_summary = ", ".join(f"{x['feature']}: {x['shap_value']:.2f}" for x in top_feats)

        else:
            # Heuristic fallback
            fallback_features = ["tx_velocity", "tx_entropy", "gas_price_std"]
            top_summary = ", ".join(f"{f}: {row[f]:.2f}" for f in fallback_features if f in row)

        prompt = format_prompt({
            "wallet": wallet,
            "cluster_id": row.get("cluster_id", "N/A"),
            "anomaly_flag": row.get("anomaly_score", "N/A"),
            "top_features": top_summary,
            "feature_snapshot": row.drop(
                [col for col in ["wallet", "cluster_id", "anomaly_score"] if col in row]
            ).to_dict()
        })

        try:
            summary = call_llm(prompt)
        except Exception as e:
            summary = f"LLM Error: {str(e)}"

        summaries[wallet] = {
            "summary": summary,
            "cluster_id": row.get("cluster_id", None),
            "anomaly_score": row.get("anomaly_score", None),
            "top_features": top_summary
        }

    print(f"Writing summaries to: {OUTPUT_JSON_PATH}")
    OUTPUT_JSON_PATH.write_text(json.dumps(summaries, indent=2))

if __name__ == "__main__":
    main()
