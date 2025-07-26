"""
Module 4: Generate SHAP Explainability — global + per-wallet (HTML + JSON)
"""

import pandas as pd
import shap
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import json

# === Paths ===
FEATURES_PATH = Path("data/processed/tx_features.parquet")
MODEL_PATH = Path("models/xgboost.joblib")
OUTPUT_DIR = Path("explainability/shap_values")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === Load data & model ===
df = pd.read_parquet(FEATURES_PATH)
df = df[df["label"] >= 0]  # labeled only
X = df.drop(columns=["wallet", "label"])
wallets = df["wallet"].tolist()

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print("Trained model not found. Run Module 3 first.")
    exit()

# === Use SHAP TreeExplainer (no GPU/PyTorch dependency) ===
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# === Global Summary Plot ===
plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.savefig(OUTPUT_DIR / "shap_summary.png", bbox_inches="tight")
plt.close()
print("Saved: shap_summary.png")

# === Per-wallet Force Plots (HTML) ===
print("Saving per-wallet force plots (HTML)...")
force_plot_paths = []
force_plot_jsons = {}

for i in range(min(250, len(X))):
    wallet = wallets[i]
    force_html_path = OUTPUT_DIR / f"wallet_force_plot_{i:03}.html"
    
    force_plot = shap.plots.force(
        explainer.expected_value, shap_values[i], X.iloc[i], matplotlib=False
    )
    shap.save_html(str(force_html_path), force_plot)
    force_plot_paths.append(str(force_html_path))
    
    # Store as JSON for LLM use
    force_plot_jsons[wallet] = {
        "features": X.iloc[i].to_dict(),
        "shap_values": shap_values[i].tolist(),
    }

# === Save SHAP JSONs ===
with open(OUTPUT_DIR / "wallet_shap_values.json", "w") as f:
    json.dump(force_plot_jsons, f, indent=2)

print(f"Saved {len(force_plot_paths)} force plots (HTML)")
print(f"SHAP JSON exported: wallet_shap_values.json")
