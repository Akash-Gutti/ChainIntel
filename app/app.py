import gradio as gr
import pandas as pd
import json
import plotly.express as px
from pathlib import Path
from datetime import datetime
from fpdf import FPDF

# === Load Data ===
CSV_PATH = Path("data/processed/demo_wallets.csv")
SUMMARY_PATH = Path("data/processed/wallet_summaries.json")

# === Load DataFrames ===
df = pd.read_csv(CSV_PATH)
df = df.rename(columns={"cluster_id_y": "cluster_id", "anomaly_score_y": "anomaly_score"})
df["wallet"] = df["wallet"].astype(str)

with open(SUMMARY_PATH) as f:
    summaries = json.load(f)

wallet_list = df["wallet"].drop_duplicates().tolist()
wallet_dropdown = [f"{w[:8]}...{w[-6:]}" for w in wallet_list]
summaries_clean = {k: v for k, v in summaries.items() if "summary" in v}

# === Dashboard Intro ===
def landing_intro():
    return """
## 🛡️ ChainIntel Risk Intelligence Dashboard

Welcome to the ChainIntel dashboard — your AI-powered forensic platform for Ethereum wallets.

Navigate tabs to explore:
- 📊 KPI + Visuals
- 🧬 Cluster View
- 🔍 Wallet Inspector
- 🧠 GPT Risk Summary
- 📈 Timeline + 📤 Export Tools
"""

# === KPI Plots ===
def get_kpis():
    total = len(df)
    anomalies = (df["anomaly_score"] == 1).sum()
    clusters = df["cluster_id"].nunique()
    return total, anomalies, clusters

def plot_cluster_distribution():
    counts = df['cluster_id'].value_counts().reset_index()
    counts.columns = ['cluster_id', 'count']
    return px.bar(counts, x='cluster_id', y='count', labels={'cluster_id': 'Cluster ID', 'count': 'Wallets'}, title="Cluster Distribution")

def plot_anomaly_score_distribution():
    return px.histogram(df, x='anomaly_score', nbins=50, title="Anomaly Score Distribution")

# === Resolve Wallets ===
def resolve_wallet(short):
    return next((w for w in wallet_list if short.startswith(w[:8])), None)

def clean_json(data: dict):
    return {k: v for k, v in data.items() if not k.endswith("_x") and pd.notna(v) and v != "" and v != "nan"}

# === Wallet Viewer ===
def get_wallet_info(short_id):
    wallet = resolve_wallet(short_id)
    row = df[df.wallet == wallet].squeeze()
    risk_tag = (
        "<span style='color:white;background-color:red;padding:4px 8px;border-radius:4px;font-weight:bold;'>⚠️ HIGH RISK</span>"
        if row["anomaly_score"] == 1 else
        "<span style='color:white;background-color:green;padding:4px 8px;border-radius:4px;font-weight:bold;'>✅ LOW RISK</span>"
    )
    etherscan_link = f"https://etherscan.io/address/{wallet}"
    md = f"""
### 🧾 Wallet Report

**Wallet Address:** [{wallet}]({etherscan_link})  
**Risk Level:** {risk_tag}  
**Cluster ID:** `{row['cluster_id']}`  
**Anomaly Score:** `{row['anomaly_score']:.4f}`  
**ETH Sent:** `{row.get('eth_value_sum', 'N/A')}`  
**TX Count:** `{row.get('tx_count', 'N/A')}`
"""
    return md, json.dumps(clean_json(row.to_dict()), indent=2)

# === GPT Summary Viewer ===
def generate_tag(summary):
    tags = []
    s = summary.lower()
    if "mixer" in s or "tornado" in s: tags.append("🔀 Mixer Activity")
    if "flash loan" in s: tags.append("⚡ Flash Loan")
    if "smart contract" in s: tags.append("🤖 Contract Heavy")
    if "low activity" in s or "dormant" in s: tags.append("🧪 Dormant")
    if "high entropy" in s: tags.append("🌀 High Entropy")
    return ", ".join(tags) if tags else "No Tag"

def get_summary_card(short_id):
    wallet = resolve_wallet(short_id)
    data = summaries.get(wallet, {})
    summary_text = data.get("summary", "No GPT summary available.")
    risk = (
        "<span style='color:white;background-color:red;padding:3px 8px;border-radius:4px;font-weight:bold;'>⚠️ HIGH</span>"
        if data.get("anomaly_score", 0) == 1 else
        "<span style='color:white;background-color:green;padding:3px 8px;border-radius:4px;font-weight:bold;'>✅ LOW</span>"
    )
    tag = generate_tag(summary_text)
    card = f"""
### 🔍 GPT Forensic Summary

**Wallet:** `{wallet}`  
**Cluster ID:** `{data.get('cluster_id', 'N/A')}`  
**Anomaly Score:** `{data.get('anomaly_score', 'N/A')}`  
**Risk Level:** {risk}  
**Tag:** {tag}

#### 🧠 GPT Insight:
{summary_text}
"""
    return card, json.dumps(clean_json(data), indent=2)

# === Cluster Explorer ===
def get_cluster_wallets(cluster_id):
    subset = df[df["cluster_id"] == int(cluster_id)]
    out = []
    for _, row in subset.iterrows():
        risk = "⚠️ High" if row.anomaly_score == 1 else "✅ Low"
        out.append(f"{row.wallet} ({risk})")
    return "\n".join(out)

# === Export Tools ===
def export_cluster(cluster_id):
    sub = df[df.cluster_id == int(cluster_id)]
    path = f"cluster_{cluster_id}_wallets.csv"
    sub.to_csv(path, index=False)
    return path

def export_anomalies():
    sub = df[df.anomaly_score == 1]
    path = "high_risk_wallets.csv"
    sub.to_csv(path, index=False)
    return path

# === PDF Generator ===
class ReportPDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "ChainIntel Forensic Wallet Report", 0, 1, "C")

    def wallet_section(self, wallet, summary):
        self.set_font("Arial", size=10)
        self.multi_cell(0, 8, f"Wallet: {wallet}\n\nSummary:\n{summary}")

def generate_pdf(wallet):
    summary = summaries.get(wallet, {}).get("summary", "N/A")
    pdf = ReportPDF()
    pdf.add_page()
    pdf.wallet_section(wallet, summary)
    path = f"{wallet}_report.pdf"
    pdf.output(path)
    return path

# === Interface ===
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as app:
    gr.Markdown(landing_intro())

    with gr.Tab("📊 KPI Overview"):
        with gr.Row():
            total_kpi = gr.Markdown()
            anomaly_kpi = gr.Markdown()
            cluster_kpi = gr.Markdown()
        with gr.Row():
            cluster_chart = gr.Plot()
            anomaly_chart = gr.Plot()

        def render_kpis():
            t, a, c = get_kpis()
            return (
                f"### 🧮 Total Wallets\n**{t}**",
                f"### 🚨 Anomalous Wallets\n**{a}**",
                f"### 🧬 Unique Clusters\n**{c}**",
                plot_cluster_distribution(),
                plot_anomaly_score_distribution()
            )

        render_kpis_btn = gr.Button("🔄 Refresh KPIs")
        render_kpis_btn.click(render_kpis, outputs=[total_kpi, anomaly_kpi, cluster_kpi, cluster_chart, anomaly_chart])
        render_kpis()

    with gr.Tab("🧬 Cluster Explorer"):
        cluster_id = gr.Number(label="Enter Cluster ID")
        cluster_out = gr.Textbox(label="Wallets in Cluster", lines=10)
        cluster_id.change(get_cluster_wallets, inputs=cluster_id, outputs=cluster_out)

    with gr.Tab("🔍 Wallet Explorer"):
        dropdown = gr.Dropdown(choices=wallet_dropdown, label="Select Wallet")
        wallet_md = gr.Markdown()
        wallet_json = gr.Code(label="📄 JSON Wallet Report", language="json")
        dropdown.change(get_wallet_info, dropdown, outputs=[wallet_md, wallet_json])

    with gr.Tab("🧠 GPT Summary"):
        dropdown2 = gr.Dropdown(choices=list(summaries_clean.keys()), label="Select Wallet")
        summary_card = gr.Markdown()
        summary_json = gr.Code(label="🧠 GPT JSON", language="json")
        dropdown2.change(get_summary_card, dropdown2, outputs=[summary_card, summary_json])

    with gr.Tab("📤 Export & Reports"):
        clust_input = gr.Number(label="Export Cluster")
        clust_btn = gr.Button("Download CSV")
        clust_file = gr.File()
        clust_btn.click(export_cluster, inputs=clust_input, outputs=clust_file)

        anom_btn = gr.Button("Download High-Risk Wallets")
        anom_file = gr.File()
        anom_btn.click(export_anomalies, outputs=anom_file)

        pdf_input = gr.Textbox(label="Wallet Address for PDF")
        pdf_btn = gr.Button("Generate PDF")
        pdf_file = gr.File()
        pdf_btn.click(generate_pdf, inputs=pdf_input, outputs=pdf_file)

app.launch()
