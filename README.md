## 🔐 ChainIntel: AI-Powered Web3 Wallet Risk Intelligence

ChainIntel is a forensic-grade AI + Web3 platform that analyzes 800+ Ethereum wallets using a hybrid of anomaly detection, clustering, and GPT-based behavioral summaries. The platform flags suspicious behavior using explainable ML and presents a modern risk dashboard with downloadable reports and intelligent tagging.

---

## 🔍 Motivation

Major Web3 attacks like Euler (\$197M), Ronin (\$620M), and Nomad (\$190M) prove one thing:

> Crypto wallets are reactive, not proactive.

**ChainIntel flips that.** It proactively flags high-risk Ethereum wallets using ML + anomaly detection, and explains its decisions using GPT-4 summaries, clusters, and forensic logic. Ideal for crypto analysts, compliance leads, and Web3 investigators.

---

## 🛠️ Features

* Risk analysis of 800+ Ethereum wallets
* Upload wallet tx data → get risk scores + clusters
* XGBoost model with 10+ wallet behavior features
* Anomaly detection via Isolation Forest
* Clustering via KMeans
* GPT-4 summaries (top 300 wallets)
* Visual dashboard (Gradio + Plotly)
* CSV export + PDF report download
* **Live deployment on Render**

---

## 🧪 How It Works

```
📦 Raw Txns → 📊 Feature Engineering → 🤖 XGBoost Risk Model
 → 🔍 Anomaly & Clustering → 🧠 GPT Summaries → 📥 Final Report
```

---

## 🧬 Tech Stack

* **Language**: Python 3.10+
* **Libraries**: Pandas, NumPy, XGBoost, scikit-learn
* **LLMs**: OpenAI GPT-4 API
* **Explainability**: SHAP (partial), Anomaly/Cluster heuristics
* **Visualization**: Gradio, Plotly, Matplotlib
* **Export Tools**: FPDF (PDF), CSV
* **Deployment**: Render.com (Free Tier)

---

## 📄 Key Artifacts

* `data/processed/wallet_risk_report.csv`: All wallets with risk scores + summaries
* `data/processed/wallet_summaries.json`: GPT summaries with cluster/anomaly info
* `docs/screenshots/`: Visual plots for anomaly & cluster distribution
* `app/app.py`: Gradio dashboard

---

## 🌐 Live App

**Gradio Frontend**: [https://chainintel-stik.onrender.com](https://chainintel-stik.onrender.com)

---

## 🔍 Project Scope

* 800+ wallets processed
* 300 GPT summaries
* 6-month Ethereum activity window
* Full clustering, anomaly scores, tags
* Deployed, mobile-ready dashboard

---

## ⚙️ Next Extensions

* IPFS integration for report publishing
* Per-wallet SHAP visual (subset)
* FastAPI wrapper for API scoring (optional)
* Discord alert system for risk spikes

---

## 🤝 Contributing

Pull requests welcome! Focus: new features, LLM improvement, compliance module, more datasets.

---

## ⚖️ License

MIT License © 2025 Akash Gutti

