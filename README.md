# 🔐 ChainIntel: AI-Powered Web3 Wallet Risk Intelligence

ChainIntel is a forensic-grade AI + Web3 platform that analyzes Ethereum wallets using anomaly detection, clustering, and GPT-style behavioral summaries (precomputed). It flags suspicious behavior with explainable signals and presents a clean, recruiter-friendly dashboard with CSV/PDF exports.

**Live App →** [https://chainintel-rc2b.onrender.com/app/](https://chainintel-rc2b.onrender.com/app/) • **Health:** `/health`
No login required. Gradio mounted at `/app` behind FastAPI.

---

## Table of contents

* Why
* Features
* How it works
* Tech stack
* Data inputs
* Quickstart (local)
* Deploy to Render
* App tour
* Project structure
* Exports & Git hygiene
* Roadmap
* Contributing
* Notes & disclaimers
* License

---

## 🔍 Why

Major Web3 incidents (Euler \~\$197M, Ronin \~\$620M, Nomad \~\$190M) showed the industry is often reactive.
ChainIntel flips that by proactively surfacing high-risk wallets and summarizing behaviors in plain English for analysts, investigators, and compliance teams.

---

## ✨ Features

* **800+ wallets analyzed** (sample dataset)
* **Risk signals:** anomaly score, cluster membership, activity stats
* **Interactive overview (no heavy charts):** KPI tiles (clickable), insight switcher, progress bar
* **Wallet inspector:** one-click report + JSON
* **GPT summaries:** precomputed narrative insights with tags
* **Cluster explorer:** fixed-height, scrollable results (smooth UX)
* **Exports:** per-cluster CSV, high-risk CSV, per-wallet PDF
* **Production routing:** FastAPI `/health`, Gradio UI at `/app`, root `/` → redirect
* **Live on Render** (Free tier friendly; 1 worker)

---

## 🧪 How it works

```
📦 Raw Txns → 📊 Feature Engineering → 🤖 Risk Signals (anomaly, clusters)
                       ↓
               🧠 Precomputed GPT-style summaries
                       ↓
            🖥️ Dashboard + CSV/PDF Exports (FastAPI + Gradio)
```

---

## 🧬 Tech stack

* **Python** 3.10+
* **Runtime / API:** FastAPI + Uvicorn/Gunicorn
* **UI:** Gradio (Blocks) with lightweight HTML/CSS micro-visuals
* **Data:** Pandas
* **PDF:** fpdf2
* **Hosting:** Render Web Service

> Note: Plotly/Matplotlib were intentionally removed to keep the app fast and stable on free hosting.

---

## 📥 Data inputs

**Expected files** (relative to repo root):

```
data/processed/demo_wallets.csv
```

**Columns (min):**
`wallet (str), cluster_id (int), anomaly_score (0/1),`
*optional:* `eth_value_sum (float), tx_count (int)`

```
data/processed/wallet_summaries.json
```

**Mapping:**
`{ "<wallet>": { "summary": str, "cluster_id": int, "anomaly_score": 0/1, ... } }`

### Minimal sample

```csv
# data/processed/demo_wallets.csv
wallet,cluster_id,anomaly_score,eth_value_sum,tx_count
0xaaaabbbbccccddddeeeeffff0000111122223333,1,0,12.5,42
0xbbbbccccddddeeeeffff00001111222233334444,1,1,3.1,8
0xccccddddeeeeffff000011112222333344445555,2,1,7.0,19
```

```js
data/processed/wallet_summaries.json
{
  "0xaaaabbbbccccddddeeeeffff0000111122223333": {
    "summary": "Low activity wallet; regular transfers; no mixers detected.",
    "cluster_id": 1,
    "anomaly_score": 0
  },
  "0xbbbbccccddddeeeeffff00001111222233334444": {
    "summary": "High-risk pattern: frequent interactions with known mixers.",
    "cluster_id": 1,
    "anomaly_score": 1
  }
}
```
---

## 🚀 Quickstart (local)

```bash
git clone <your-repo>
cd <your-repo>

python -m venv .venv
# mac/linux: source .venv/bin/activate
# windows:   .venv\Scripts\Activate.ps1

pip install -r requirements.txt

# ensure data/processed/ contains CSV + JSON (see samples above)

python -m uvicorn app.app:app --host 127.0.0.1 --port 8000
# UI:     http://127.0.0.1:8000/app/
# Health: http://127.0.0.1:8000/health
```

**Requirements tip**

```txt
gradio>=4.15,<5
fastapi>=0.111
uvicorn[standard]>=0.30
gunicorn>=21.2
pandas>=2.0
fpdf2>=2.7
```

If you ever see a PyFPDF/fpdf2 conflict: uninstall `pypdf`, `fpdf`, `PyFPDF` and keep **fpdf2** only.

---

## ☁️ Deploy to Render

Create a **Web Service** (no login required for users).

**Build Command**

```bash
pip install --upgrade pip && pip install -r requirements.txt
```

**Start Command (Free tier stable)**

```
gunicorn -k uvicorn.workers.UvicornWorker -w 1 -t 120 --graceful-timeout 30 --keep-alive 5 -b 0.0.0.0:$PORT app.app:app
```

**Health Check Path:** `/health`
**UI path:** `/app/` (root `/` redirects there)

*Cold starts:* Free services sleep after \~15 min idle. First hit wakes the service (10–30s). For always-on demos, upgrade to a paid plan.

---

## 🖥️ App tour

### Overview

* **KPI tiles:** Total / Anomalous / Clusters — clickable to switch the insight view
* **Anomaly progress bar:** quick risk signal
* **Insight picker (radio):** Top Clusters / High-Risk by TX / High-Risk by ETH
* **Reload Data:** re-reads input files and updates UI

### Cluster Explorer

* Enter **Cluster ID** → stable, scrollable results with risk indicators

### Wallet Inspector

* Pick from dropdown or paste an address → formatted report + JSON

### GPT Summary

* Precomputed narrative summary + tags (from `wallet_summaries.json`)

### Export & Reports

* **CSV:** per-cluster, high-risk
* **PDF:** per-wallet forensic sheet (fpdf2)

---

## 🗂️ Project structure

```bash
.
├─ app/
│  └─ app.py                 # FastAPI + Gradio UI (mounted at /app)
├─ data/
│  └─ processed/
│     ├─ demo_wallets.csv    # input CSV (see schema)
│     └─ wallet_summaries.json
├─ requirements.txt
├─ Procfile                  # (optional for Render; Start Command in dashboard also works)
└─ render.yaml               # (optional) blueprint; not required for single web service
```

---

## 🗺️ Roadmap

* Optional always-on deployment (paid Render plan) for zero cold starts
* API scoring endpoint for programmatic risk checks
* Per-wallet SHAP/explainers for a subset (if models integrated)
* IPFS publishing for reports
* Discord/Slack alerts for spikes
* Data refresh job (cron/worker) if you add live pipelines

---

## 🤝 Contributing

Pull requests welcome! Ideas:

* Extra heuristics/features
* Better summaries/tags (e.g., mixer heuristics, exchange hop-detection)
* More robust dataset loaders + validation
* Additional export formats

Open an issue with context + sample data if possible.

---

## 📝 Notes & disclaimers

* Current app does **not** call OpenAI directly; it loads precomputed summaries from JSON.
* Hosting on Render Free may sleep when idle (expected behavior).
* This is an analytics tool; **not** financial advice or AML/KYC certification.

---

## ⚖️ License

MIT License © 2025 Akash Gutti