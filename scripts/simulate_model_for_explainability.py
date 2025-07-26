import pandas as pd
from xgboost import XGBClassifier
import joblib
from pathlib import Path

# === Paths ===
INPUT_PATH = Path("data/processed/tx_features.parquet")
OUTPUT_MODEL = Path("models/xgboost.joblib")
OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)

# === Load Data ===
df = pd.read_parquet(INPUT_PATH)
df = df[df['label'] >= 0]

X = df.drop(columns=['wallet', 'label'])
y = df['label']

# === Simple Fit on All Labeled Data ===
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X, y)

joblib.dump(model, OUTPUT_MODEL)
print("Simulated model trained and saved to models/xgboost.joblib")
