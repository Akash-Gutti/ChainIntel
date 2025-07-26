"""
Module 8.3: Generate recruiter-facing notebook summary (static insights)
"""

from pathlib import Path
import nbformat as nbf

NOTEBOOK_PATH = Path("notebooks/08_wallet_risk_summary.ipynb")
SCREENSHOT_PATH = "docs/screenshots"

nb = nbf.v4.new_notebook()

cells = []

# 📌 Title
cells.append(nbf.v4.new_markdown_cell(
"# 🛡️ ChainIntel: Wallet Risk Intelligence Report\n"
"**Module 8.3 — Notebook Summary for Recruiters & Reviewers**\n\n"
"This notebook provides a visual and analytical summary of suspicious Ethereum wallet behavior detected by ChainIntel v2.5."
))

# 📌 Dataset Info
cells.append(nbf.v4.new_markdown_cell(
"## 📁 Dataset Overview\n"
"- Transactions: 100K+ over 6 months\n"
"- Wallets analyzed: Inference-only (label == -1)\n"
"- Clustering: KMeans (unsupervised)\n"
"- Anomaly Detection: Isolation Forest\n"
"- Summarization: GPT-4 forensic summaries\n"
))

# 📌 Key Metrics
cells.append(nbf.v4.new_code_cell(
"import pandas as pd\n"
"df = pd.read_csv('data/processed/wallet_risk_report.csv', low_memory=False)\n"
"df = df.rename(columns={\n"
"    'anomaly_score_y': 'anomaly_score',\n"
"    'cluster_id_y': 'cluster_id'\n"
"})\n"
"print('Total wallets:', len(df))\n"
"print('Anomalous wallets:', (df.anomaly_score == 1).sum())\n"
"print('Unique clusters:', df.cluster_id.nunique())"
))

# 📌 Show Sample Summaries
cells.append(nbf.v4.new_markdown_cell("## 🧠 Sample Wallet Risk Summaries"))
cells.append(nbf.v4.new_code_cell(
"df[['wallet', 'summary', 'cluster_id', 'anomaly_score']].dropna().sample(5, random_state=42)"
))

# 📌 Charts
cells.append(nbf.v4.new_markdown_cell("## 📊 Visual Insights"))

cells.append(nbf.v4.new_markdown_cell("### 🔹 Cluster Distribution"))
cells.append(nbf.v4.new_code_cell(
"from IPython.display import Image\n"
"Image(filename=f'{SCREENSHOT_PATH}/cluster_distribution.png')"
))

cells.append(nbf.v4.new_markdown_cell("### 🔹 Anomaly Score Distribution"))
cells.append(nbf.v4.new_code_cell(
"Image(filename=f'{SCREENSHOT_PATH}/anomaly_score_dist.png')"
))

cells.append(nbf.v4.new_markdown_cell("### 🔹 Top Risk Factors"))
cells.append(nbf.v4.new_code_cell(
"Image(filename=f'{SCREENSHOT_PATH}/top_features_barplot.png')"
))

# 📌 How to Extend
cells.append(nbf.v4.new_markdown_cell(
"##🔄 Future Extensions\n"
"- Deploy real-time wallet scoring APIs\n"
"- Integrate alerts with Telegram/Discord\n"
"- Add analyst feedback and audit trails\n"
"- Export reports to PDF / dashboard view\n"
))

# Save notebook
nb["cells"] = cells
NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook saved to: {NOTEBOOK_PATH}")
