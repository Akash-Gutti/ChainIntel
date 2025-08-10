# app/app.py
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr
import pandas as pd
from fpdf import FPDF
from fastapi import FastAPI
from fastapi.responses import RedirectResponse, JSONResponse
import uvicorn

# =========================
# Safe Data Loading
# =========================
CSV_PATH = Path("data/processed/demo_wallets.csv")
SUMMARY_PATH = Path("data/processed/wallet_summaries.json")

def _load_df() -> pd.DataFrame:
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        df = df.rename(columns={"cluster_id_y": "cluster_id", "anomaly_score_y": "anomaly_score"})
        if "wallet" in df:
            df["wallet"] = df["wallet"].astype(str)
        else:
            df["wallet"] = ""
        for col in ["cluster_id", "anomaly_score", "eth_value_sum", "tx_count"]:
            if col not in df:
                df[col] = pd.NA
        return df
    return pd.DataFrame(columns=["wallet", "cluster_id", "anomaly_score", "eth_value_sum", "tx_count"])

def _load_summaries() -> Dict[str, Any]:
    if SUMMARY_PATH.exists():
        with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

# Global state
df: pd.DataFrame = _load_df()
summaries: Dict[str, Any] = _load_summaries()
wallet_list: List[str] = df["wallet"].dropna().astype(str).drop_duplicates().tolist() if "wallet" in df else []
wallet_dropdown: List[str] = [f"{w[:8]}...{w[-6:]}" for w in wallet_list] if wallet_list else []

# =========================
# Helpers
# =========================
def landing_intro() -> str:
    return """
# ChainIntel — Risk Intelligence Dashboard

This dashboard provides a comprehensive overview of wallet activity, risk analysis, and clustering insights for Ethereum addresses.
It leverages advanced anomaly detection and clustering algorithms to identify high-risk wallets and their associated behaviors.

"""

def get_kpis() -> Tuple[int, int, int, float]:
    total = int(len(df)) if not df.empty else 0
    anomalies = int((df["anomaly_score"] == 1).sum()) if "anomaly_score" in df else 0
    clusters = int(df["cluster_id"].nunique()) if "cluster_id" in df and not df["cluster_id"].isna().all() else 0
    anomaly_rate = (anomalies / total * 100.0) if total else 0.0
    return total, anomalies, clusters, anomaly_rate

def anomaly_bar(rate: float) -> str:
    rate = max(0.0, min(rate, 100.0))
    return f"""
<div class="meter"><span style="width:{rate:.1f}%"></span></div>
<div class="meter-caption">{rate:.1f}% anomalous</div>
"""

def resolve_wallet(short_or_full: str) -> str | None:
    if not short_or_full:
        return None
    s = str(short_or_full).strip()
    if s in wallet_list:
        return s
    for w in wallet_list:
        if s.startswith(w[:8]):
            return w
    if s.startswith("0x") and len(s) >= 10:
        return s
    return None

def clean_json(data: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in data.items():
        if str(k).endswith("_x"): continue
        if pd.isna(v) or v == "" or str(v) == "nan": continue
        out[k] = v
    return out

def fmt4(v: Any) -> str:
    try:
        return f"{float(v):.4f}"
    except Exception:
        return str(v)

# =========================
# Insight Builders (lightweight)
# =========================
def insight_top_clusters(n: int = 10) -> str:
    if df.empty or "cluster_id" not in df:
        return "_No data._"
    s = df["cluster_id"].value_counts(dropna=True).head(n).reset_index()
    s.columns = ["Cluster ID", "Wallets"]
    rows = "\n".join(f"- **{int(r['Cluster ID'])}** — {int(r['Wallets'])} wallets" for _, r in s.iterrows())
    return f"### Top {n} Clusters by Wallet Count\n{rows}"

def _sorted_high_risk(key: str, n: int = 10) -> pd.DataFrame:
    if df.empty or "anomaly_score" not in df:
        return pd.DataFrame()
    subset = df[df["anomaly_score"] == 1].copy()
    if subset.empty:
        return pd.DataFrame()
    if key in subset and subset[key].notna().any():
        subset = subset.sort_values(by=[key], ascending=False, na_position="last")
    return subset.head(n)

def insight_high_risk_by_tx(n: int = 10) -> str:
    subset = _sorted_high_risk("tx_count", n)
    if subset.empty:
        return "_No high-risk wallets detected._"
    rows = [f"- `{r.get('wallet','N/A')}` — TX: **{r.get('tx_count','N/A')}**, ETH Sent: **{r.get('eth_value_sum','N/A')}**" for _, r in subset.iterrows()]
    return f"### High-Risk Wallets by Transactions (Top {n})\n" + "\n".join(rows)

def insight_high_risk_by_eth(n: int = 10) -> str:
    subset = _sorted_high_risk("eth_value_sum", n)
    if subset.empty:
        return "_No high-risk wallets detected._"
    rows = [f"- `{r.get('wallet','N/A')}` — ETH Sent: **{r.get('eth_value_sum','N/A')}**, TX: **{r.get('tx_count','N/A')}**" for _, r in subset.iterrows()]
    return f"### High-Risk Wallets by ETH Sent (Top {n})\n" + "\n".join(rows)

INSIGHT_OPTIONS = ["Top Clusters", "High-Risk by TX", "High-Risk by ETH"]

def get_insight(view: str) -> str:
    if view == "Top Clusters": return insight_top_clusters(10)
    if view == "High-Risk by TX": return insight_high_risk_by_tx(10)
    if view == "High-Risk by ETH": return insight_high_risk_by_eth(10)
    return "_Select an insight view._"

def set_insight(view: str):
    return get_insight(view), gr.update(value=view)

# =========================
# View Functions
# =========================
def get_wallet_info(short_id: str):
    wallet = resolve_wallet(short_id)
    if not wallet or df.empty:
        return "No data available for this wallet.", "{}"
    row = df[df["wallet"] == wallet]
    row = row.squeeze() if not row.empty else pd.Series(dtype="object")

    risk_tag = (
        "<span class='chip chip-danger'>High Risk</span>"
        if not row.empty and row.get("anomaly_score", 0) == 1 else
        "<span class='chip chip-success'>Low Risk</span>"
    )
    etherscan_link = f"https://etherscan.io/address/{wallet}"
    md = f"""
### Wallet Report

**Wallet Address:** [{wallet}]({etherscan_link})  
**Risk Level:** {risk_tag}  
**Cluster ID:** `{row.get('cluster_id', 'N/A') if not row.empty else 'N/A'}`  
**Anomaly Score:** `{fmt4(row.get('anomaly_score', 0)) if not row.empty else 'N/A'}`  
**ETH Sent:** `{row.get('eth_value_sum', 'N/A') if not row.empty else 'N/A'}`  
**TX Count:** `{row.get('tx_count', 'N/A') if not row.empty else 'N/A'}`
"""
    return md, json.dumps(clean_json(row.to_dict() if not row.empty else {}), indent=2)

def generate_tag(summary: str) -> str:
    tags = []
    s = (summary or "").lower()
    if "mixer" in s or "tornado" in s: tags.append("Mixer Activity")
    if "flash loan" in s: tags.append("Flash Loan")
    if "smart contract" in s: tags.append("Contract Heavy")
    if "low activity" in s or "dormant" in s: tags.append("Dormant")
    if "high entropy" in s: tags.append("High Entropy")
    return ", ".join(tags) if tags else "No Tag"

def get_summary_card(short_id: str):
    wallet = resolve_wallet(short_id)
    data = summaries.get(wallet, {}) if wallet else {}
    summary_text = data.get("summary", "No GPT summary available.")
    risk = (
        "<span class='chip chip-danger'>High</span>"
        if data.get("anomaly_score", 0) == 1 else
        "<span class='chip chip-success'>Low</span>"
    )
    tag = generate_tag(summary_text)
    card = f"""
### GPT Forensic Summary

**Wallet:** `{wallet or 'N/A'}`  
**Cluster ID:** `{data.get('cluster_id', 'N/A')}`  
**Anomaly Score:** `{data.get('anomaly_score', 'N/A')}`  
**Risk Level:** {risk}  
**Tag:** {tag}

**Insight:**  
{summary_text}
"""
    return card, json.dumps(clean_json(data), indent=2)

def get_cluster_wallets(cluster_id: Any) -> str:
    try:
        cid = int(float(cluster_id))
    except Exception:
        return "Enter a valid integer Cluster ID."
    if df.empty or "cluster_id" not in df:
        return "No cluster data available."
    subset = df[df["cluster_id"] == cid]
    if subset.empty:
        return "No wallets found for this cluster."
    out = []
    for _, row in subset.iterrows():
        risk = "High" if row.get("anomaly_score", 0) == 1 else "Low"
        out.append(f"{row['wallet']} ({risk})")
    return "\n".join(out)

def export_cluster(cluster_id: Any):
    try:
        cid = int(float(cluster_id))
    except Exception:
        return None
    sub = df[df["cluster_id"] == cid] if "cluster_id" in df else pd.DataFrame()
    path = f"cluster_{cid}_wallets.csv"
    sub.to_csv(path, index=False)
    return path

def export_anomalies():
    sub = df[df["anomaly_score"] == 1] if "anomaly_score" in df else pd.DataFrame()
    path = "high_risk_wallets.csv"
    sub.to_csv(path, index=False)
    return path

class ReportPDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "ChainIntel Forensic Wallet Report", 0, 1, "C")
    def wallet_section(self, wallet, summary):
        self.set_font("Arial", size=10)
        self.multi_cell(0, 8, f"Wallet: {wallet}\n\nSummary:\n{summary}")

def generate_pdf(wallet: str):
    if not wallet:
        return None
    summary = summaries.get(wallet, {}).get("summary", "N/A")
    pdf = ReportPDF()
    pdf.add_page()
    pdf.wallet_section(wallet, summary)
    path = f"{wallet}_report.pdf"
    pdf.output(path)
    return path

# =========================
# Data Reload
# =========================
def reload_data():
    global df, summaries, wallet_list
    df = _load_df()
    summaries = _load_summaries()
    wallet_list = df["wallet"].dropna().astype(str).drop_duplicates().tolist() if "wallet" in df else []
    dd = [f"{w[:8]}...{w[-6:]}" for w in wallet_list] if wallet_list else ["No wallets found"]

    t, a, c, rate = get_kpis()
    # return updated button labels + progress + dropdown choices
    return (
        gr.update(value=f"Total: {t}"),
        gr.update(value=f"Anomalous: {a}"),
        gr.update(value=f"Clusters: {c}"),
        anomaly_bar(rate),
        gr.update(choices=dd),
        gr.update(choices=dd),
    )

# =========================
# Full-width, neutral CSS (no dark boxes, no center squeeze)
# =========================
CUSTOM_CSS = """
/* Full width immediately (no jump), light padding */
body, .gradio-container { max-width: 100% !important; }
.gradio-container { padding: 0 24px !important; }

/* Remove colored boxes; keep things clean */
.markdown { background: transparent; padding: 0; border: none; }

/* KPI buttons styled as compact cards */
.kpi-btn button { 
  border: 1px solid #e5e7eb; border-radius: 10px; padding: 8px 12px; 
  background: #ffffff; color: #111827; font-weight: 800; font-size: 14px; 
  display:flex; align-items:center; justify-content: space-between;
}

/* Chips for risk: subtle */
.chip { color:#111827; padding:3px 10px; border-radius: 999px; font-weight:600; border:1px solid #e5e7eb; background:#f9fafb; }
.chip-danger { border-color:#ef4444; color:#991b1b; background:#fef2f2; }
.chip-success { border-color:#10b981; color:#065f46; background:#ecfdf5; }

/* Progress bar */
.meter { width:100%; background:#f3f4f6; border-radius: 8px; overflow:hidden; height: 10px; border:1px solid #e5e7eb; }
.meter > span { display:block; height:100%; background: linear-gradient(90deg, #60a5fa, #34d399); }
.meter-caption { font-size: 12px; color: #6b7280; margin-top: 6px; }

/* Fixed-height result area to prevent jumping */
.scrollbox { height: 420px; overflow: auto; border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px; background: #fff; }

/* Export tab: tidy spacing */
.compact-row .gr-box, .compact-row .gr-form { margin: 0 !important; }
.section-card { border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px; background: #ffffff; }
"""

# =========================
# UI (Blocks)
# =========================
with gr.Blocks(theme=gr.themes.Soft(), css=CUSTOM_CSS, analytics_enabled=False) as ui:
    gr.Markdown(landing_intro())

    # ---------- Overview (compact, full-width, interactive) ----------
    with gr.Tab("📊 Overview"):
        t0, a0, c0, rate0 = get_kpis()
        # Compact KPI row (3 small clickable buttons styled as cards)
        with gr.Row():
            with gr.Column(scale=1):
                kpi_total = gr.Button(f"Total: {t0}", elem_classes=["kpi-btn"])
            with gr.Column(scale=1):
                kpi_anoms = gr.Button(f"Anomalous: {a0}", elem_classes=["kpi-btn"])
            with gr.Column(scale=1):
                kpi_clusters = gr.Button(f"Clusters: {c0}", elem_classes=["kpi-btn"])

        # Progress bar (anomaly rate)
        anomaly_progress = gr.HTML(value=anomaly_bar(rate0))

        # Insight picker (interactive)
        with gr.Row():
            insight_picker = gr.Radio(choices=INSIGHT_OPTIONS, value="Top Clusters", label="Insight View", interactive=True)
        insight_panel = gr.Markdown(value=get_insight("Top Clusters"))

        # KPI button clicks -> switch insight view
        def kpi_click(view):
            return get_insight(view), gr.update(value=view)

        kpi_total.click(lambda: kpi_click("Top Clusters"), outputs=[insight_panel, insight_picker])
        kpi_anoms.click(lambda: kpi_click("High-Risk by TX"), outputs=[insight_panel, insight_picker])
        kpi_clusters.click(lambda: kpi_click("High-Risk by ETH"), outputs=[insight_panel, insight_picker])

        # Data reload (updates KPI labels, progress, dropdowns)
        reload_btn = gr.Button("Reload Data", variant="secondary")

        # Wire interactivity
        insight_picker.change(lambda v: get_insight(v), inputs=insight_picker, outputs=insight_panel)

    # ---------- Cluster Explorer (no jumping; fixed result box) ----------
    with gr.Tab("🧬 Cluster Explorer"):
        with gr.Row():
            cluster_id_in = gr.Number(label="Cluster ID", precision=0)
            cluster_out_html = gr.HTML(label=None, value="<div class='scrollbox'></div>")
        def _cluster_html(cid):
            text = get_cluster_wallets(cid)
            safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            return f"<div class='scrollbox'><pre style='margin:0;font:13px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace;'>{safe}</pre></div>"
        cluster_id_in.change(_cluster_html, inputs=cluster_id_in, outputs=cluster_out_html)

    # ---------- Wallet Inspector ----------
    with gr.Tab("🔍 Wallet Inspector"):
        with gr.Row():
            dropdown_wallet = gr.Dropdown(choices=wallet_dropdown or ["No wallets found"], label="Select Wallet")
            manual_wallet = gr.Textbox(label="Or paste wallet address", placeholder="0xabc... or 0x1234...abcd")
        wallet_md = gr.Markdown()
        wallet_json = gr.Code(label="JSON Wallet Report", language="json")
        dropdown_wallet.change(get_wallet_info, inputs=dropdown_wallet, outputs=[wallet_md, wallet_json])
        manual_wallet.submit(get_wallet_info, inputs=manual_wallet, outputs=[wallet_md, wallet_json])

    # ---------- GPT Summary ----------
    with gr.Tab("🧠 GPT Summary"):
        dropdown_summary = gr.Dropdown(choices=wallet_dropdown or ["No wallets found"], label="Select Wallet")
        summary_card = gr.Markdown()
        summary_json = gr.Code(label="GPT JSON", language="json")
        dropdown_summary.change(get_summary_card, inputs=dropdown_summary, outputs=[summary_card, summary_json])

    # ---------- Export & Reports (clean, two compact sections) ----------
    with gr.Tab("📤 Export & Reports"):
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group(elem_classes=["section-card"]):
                    gr.Markdown("**CSV Exports**")
                    with gr.Row(elem_classes=["compact-row"]):
                        clust_input = gr.Number(label="Cluster ID", precision=0)
                        clust_btn = gr.Button("Download Cluster CSV")
                    clust_file = gr.File(label=None)
                    with gr.Row(elem_classes=["compact-row"]):
                        anom_btn = gr.Button("Download High-Risk CSV")
                    anom_file = gr.File(label=None)
            with gr.Column(scale=1):
                with gr.Group(elem_classes=["section-card"]):
                    gr.Markdown("**PDF Report**")
                    pdf_input = gr.Textbox(label="Wallet Address")
                    pdf_btn = gr.Button("Generate PDF")
                    pdf_file = gr.File(label=None)

        # Wiring for exports
        clust_btn.click(export_cluster, inputs=clust_input, outputs=clust_file)
        anom_btn.click(export_anomalies, outputs=anom_file)
        pdf_btn.click(generate_pdf, inputs=pdf_input, outputs=pdf_file)

    # Wire reload after components exist (also updates dropdowns)
    reload_btn.click(
        reload_data,
        outputs=[kpi_total, kpi_anoms, kpi_clusters, anomaly_progress, dropdown_wallet, dropdown_summary]
    )

# =========================
# FastAPI wrapper + health + redirects
# =========================
fastapi_app = FastAPI()

@fastapi_app.get("/health")
def health():
    return {"status": "ok"}

@fastapi_app.get("/manifest.json")
def manifest():
    return JSONResponse({
        "name": "ChainIntel",
        "short_name": "ChainIntel",
        "start_url": "/app/",
        "display": "standalone",
        "icons": []
    })

@fastapi_app.get("/")
def root():
    return RedirectResponse(url="/app/")

# Mount Gradio at a subpath (not "/")
app = gr.mount_gradio_app(fastapi_app, ui, path="/app")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
