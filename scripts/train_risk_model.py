"""
Module 3: Train risk classification models (XGBoost, LightGBM, LogisticRegression).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix, roc_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import matplotlib.pyplot as plt

INPUT_PATH = Path("data/processed/tx_features.parquet")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)

def load_data():
    df = pd.read_parquet(INPUT_PATH)
    df = df[df['label'] >= 0]  # Use only labeled data (0, 1)
    X = df.drop(columns=["wallet", "label"])
    y = df["label"]
    return X, y

def evaluate_model(model, X, y, name):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs, precisions, recalls = [], [], []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        probs = model.predict_proba(X.iloc[test_idx])[:, 1]

        auc = roc_auc_score(y.iloc[test_idx], probs)
        prec = precision_score(y.iloc[test_idx], preds, zero_division=0)
        rec = recall_score(y.iloc[test_idx], preds, zero_division=0)

        aucs.append(auc)
        precisions.append(prec)
        recalls.append(rec)

        print(f"Fold {fold}: AUC={auc:.3f} | Precision={prec:.3f} | Recall={rec:.3f}")

    print(f"\n {name} Avg AUC: {np.mean(aucs):.4f}")
    print(f"   Avg Precision: {np.mean(precisions):.4f}")
    print(f"   Avg Recall:    {np.mean(recalls):.4f}")

    joblib.dump(model, MODEL_DIR / f"{name.lower().replace(' ', '_')}.joblib")

    # Plot ROC curve for final fold
    fpr, tpr, _ = roc_curve(y.iloc[test_idx], probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"{name} ROC (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"ROC Curve - {name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(MODEL_DIR / f"{name.lower().replace(' ', '_')}_roc.png")
    print(f"ROC Curve saved: {name.lower().replace(' ', '_')}_roc.png")

def main():
    X, y = load_data()

    print(f"Data loaded: {X.shape[0]} wallets | {X.shape[1]} features")
    print("Class distribution:\n", y.value_counts(normalize=True))

    # ⚠️ Fallback safeguard
    if y.value_counts().min() < 5 or y.nunique() < 2:
        print("\n Not enough labeled data to train classifiers.")
        print("This is common in forensic blockchain datasets.")
        print("Proceeding to SHAP, anomaly, and clustering layers instead.")
        return

    models = [
        ("XGBoost", XGBClassifier(eval_metric="logloss", base_score=0.5)),
        ("LightGBM", LGBMClassifier()),
        ("Logistic Regression", LogisticRegression(max_iter=1000)),
    ]

    for name, model in models:
        print(f"\n Training: {name}")
        evaluate_model(model, X, y, name)

if __name__ == "__main__":
    main()
