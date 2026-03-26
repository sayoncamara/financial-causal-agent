# 🔬 Financial Causal Intelligence Agent

**An AI agent that answers financial "why" questions using real causal inference — not just LLM correlations. Now available as an MCP server for any AI client.**

[![Live Demo](https://img.shields.io/badge/%F0%9F%9A%80_Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://financial-causal-agent-kxda65qww6bbx4yxnzbcf9.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-v0.2.60-1C3C3C?style=for-the-badge)](https://github.com/langchain-ai/langgraph)
[![DoWhy](https://img.shields.io/badge/DoWhy-Causal_Inference-2E86C1?style=for-the-badge)](https://github.com/py-why/dowhy)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![MCP](https://img.shields.io/badge/MCP-Server-6366F1?style=for-the-badge)](https://modelcontextprotocol.io/)

> *"Why did fund outflows increase when ECB raised interest rates?"*
> 
> Instead of guessing, the agent runs **DoWhy** causal analysis with robustness checks, discovers causal structure with the **PC algorithm**, and retrieves policy context via **RAG** — then synthesizes everything into a clear explanation.

---

[![Agent Response & Causal Analysis](docs/screenshots/hero_1.png)](docs/screenshots/hero_1.png)

[![Causal DAG & Effect Estimates](docs/screenshots/hero_2.png)](docs/screenshots/hero_2.png)

[![RAG Policy Sources & Full DAG](docs/screenshots/hero_3.png)](docs/screenshots/hero_3.png)

---

## 🎯 The Problem

LLMs can say *"interest rates affect bond prices"* because they've seen that statement in training data — but they **can't compute the actual effect size**, **can't validate with refutation tests**, and **can't distinguish correlation from causation** in new data.

This agent separates the two concerns:

- **🧮 DoWhy computes** — real statistical causal inference with robustness checks
- **🗣️ LLM explains** — natural language synthesis of validated results

This design decision is backed by [Han et al. (2024)](https://arxiv.org/abs/2305.00050) on the limitations of LLMs for causal reasoning.

---

## 🏗️ Architecture

```mermaid
graph TB
    A["🧑 User Question"] --> B["🤖 LangGraph Agent"]
    B --> C["📊 Data Loader"]
    B --> D["🔬 DoWhy Engine"]
    B --> E["🕸️ PC Algorithm"]
    B --> F["📄 FAISS RAG"]
    B --> G["⚡ FastAPI"]
    
    D --> H["Effect Estimation\n+ Refutation Tests"]
    E --> I["Causal Discovery\n+ Edge Detection"]
    F --> J["ECB Policy Docs\n+ Macro Context"]
    
    H --> K["📋 Results Dashboard"]
    I --> K
    J --> K
    
    style A fill:#1e3a5f,color:#fff
    style B fill:#2563eb,color:#fff
    style K fill:#16a34a,color:#fff
```

The agent **autonomously decides** which tools to use based on the question. Ask about a causal effect → it runs DoWhy. Ask about structure → it runs the PC algorithm. Ask about policy → it queries RAG. Often it combines all three.

---

## 🔌 MCP Server — AI Client Interoperability

The agent's causal analysis capabilities are exposed via the **Model Context Protocol (MCP)**, allowing any MCP-compatible client (Claude Desktop, Cursor, or custom applications) to use them as tools.

### Available Tools

| Tool | What It Does |
|------|-------------|
| `analyze_causal_effect` | Run DoWhy's 4-step pipeline on any variable pair — returns effect size, p-value, CI, and refutation results |
| `discover_causal_structure` | Run the PC algorithm to discover causal edges from data |
| `query_financial_policies` | RAG search over ECB monetary policy, fund risk management, and macro indicator documents |
| `load_financial_data` | Access 120 months of European fund market data with optional filtering |

### Why MCP?

Without MCP, Claude answers causal questions from training data — plausible but unverified. With MCP, Claude calls DoWhy to compute the actual effect (e.g., ECB rate → fund outflows: **+8.73 units, CI [4.40, 13.59]**) with robustness checks. The system is **model-agnostic** — any LLM client that supports MCP can connect.

### Setup

```bash
pip install "mcp[cli]>=1.0.0"
python mcp_server.py
```

Then configure your MCP client to connect to the server. See the [MCP documentation](https://modelcontextprotocol.io/) for client-specific setup.

---

## 🧠 Causal DAG (Domain Knowledge)

```mermaid
graph LR
    INF["📈 Inflation Rate"] --> ECB["🏦 ECB Rate"]
    ECB --> BOND["📉 Bond Price Index"]
    ECB --> EQ["📊 Equity Returns"]
    ECB --> CS["💳 Credit Spread"]
    BOND --> OUT["🔴 Fund Outflows"]
    EQ --> IN["🟢 Fund Inflows"]
    CS --> RISK["⚠️ Client Risk Score"]
    UE["👷 Unemployment Rate"] --> RISK
    RISK --> OUT
    
    style ECB fill:#1e3a5f,color:#fff
    style RISK fill:#dc2626,color:#fff
```

## 📊 Variables & Data Overview

[![Sidebar with Variables & Data Overview](docs/screenshots/sidebar.png)](docs/screenshots/sidebar.png)

9 variables, 9 directed edges, representing the monetary policy transmission mechanism in the Eurozone financial system. Credit spread serves as a **mediator variable** — supported by monetary policy transmission literature.

---

## ⚡ Key Features

| Feature | Description |
|---------|------------|
| 🔬 **Real Causal Inference** | DoWhy's 4-step pipeline: model → identify → estimate → refute. Not correlation. |
| ✅ **Refutation Badges** | Placebo treatment + random common cause tests visible to user — green ✓ or red ✗ |
| 🕸️ **Causal Discovery** | PC algorithm discovers structure from data, complementing the domain DAG |
| 📄 **RAG Policy Context** | FAISS vector search over ECB monetary policy, fund risk management, and macro indicator documents |
| 🤖 **Autonomous Agent** | LangGraph orchestrates 4 tools — the agent decides which to use based on the question |
| ⚡ **API Backend** | FastAPI with POST /analyze for programmatic access + Swagger docs |
| 🔌 **MCP Server** | Expose all tools via Model Context Protocol for AI client interoperability |

---

## 🔍 Key Design Decisions

| Decision | Why |
|----------|-----|
| **DoWhy computes, LLM explains** | LLMs hallucinate causal claims ([Han et al., 2024](https://arxiv.org/abs/2305.00050)). We separate computation from explanation. |
| **Refutation tests visible** | Placebo treatment + random common cause tests shown to user — this is what separates rigorous analysis from toy demos. |
| **PC algorithm for discovery** | Cross-sectional causal discovery complements the domain DAG. Transfer entropy explored but insufficient data (120 months) — documented honestly. |
| **Credit spread as mediator** | Monetary policy transmission mechanism literature supports this causal pathway. |
| **GPT-4o-mini over GPT-4** | Cost/speed optimization — the LLM only explains, it doesn't compute. A smaller model suffices. |
| **MCP for interoperability** | Any AI client can use the causal tools — not locked to one LLM provider. Model-agnostic by design. |

---

## 📊 Example Results

**Question:** *"Why did fund outflows increase when ECB raised interest rates?"*

[![Causal Effect Estimates with Refutation Badge](docs/screenshots/effects.png)](docs/screenshots/effects.png)

| Relationship | Effect | P-value | Refutation |
|-------------|--------|---------|------------|
| ECB Rate → Bond Price Index | **-5.30** | 0.001 | ✅ Passed |
| Bond Price Index → Fund Outflows | **-0.42** | 0.003 | ✅ Passed |
| ECB Rate → Equity Returns | **-2.15** | 0.012 | ✅ Passed |

The agent correctly identifies the **causal chain**: ECB rate ↑ → bond prices ↓ → fund outflows ↑, with all refutation tests passing.

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| 🤖 Agent | LangGraph v0.2.60 | Autonomous tool orchestration |
| 🔬 Causal Inference | DoWhy | Model, identify, estimate, refute |
| 🕸️ Discovery | causal-learn (PC algorithm) | Data-driven structure learning |
| 🗣️ LLM | OpenAI GPT-4o-mini | Natural language explanation |
| 📄 RAG | FAISS + LangChain v0.3.25 | Policy document retrieval |
| 🖥️ Frontend | Streamlit | Interactive dashboard |
| ⚡ API | FastAPI | Programmatic access |
| 🔌 Interoperability | MCP Server | AI client tool integration |
| 📦 Data | Synthetic (120 months) | Known ground-truth causal effects |

---

## 🚀 Quickstart

### Prerequisites

- Python 3.10+
- OpenAI API key

### Setup

```bash
git clone https://github.com/sayoncamara/financial-causal-agent.git
cd financial-causal-agent
pip install -r requirements.txt
python data/generate_data.py
```

Create a `.env` file:

```
OPENAI_API_KEY=sk-your-key-here
```

### Run

```bash
# 🖥️ Streamlit UI
python -m streamlit run app.py

# ⚡ FastAPI backend
python -m uvicorn api:app --reload --port 8000

# 🔌 MCP Server (for Claude Desktop, Cursor, etc.)
pip install "mcp[cli]>=1.0.0"
python mcp_server.py

# 🧪 CLI test
python agent.py
```

---

## ⚡ API

**POST /analyze**

```json
{
  "question": "Why did fund outflows increase when ECB raised interest rates?"
}
```

Returns structured JSON:

```json
{
  "answer": "Based on causal inference analysis...",
  "causal_effects": [
    {
      "treatment": "ecb_rate",
      "outcome": "bond_price_index",
      "estimate": -5.30,
      "p_value": 0.001,
      "refutation_passed": true
    }
  ],
  "discovered_edges": [...],
  "policy_sources": [...],
  "processing_time": 4.7
}
```

**GET /health** → `{"status": "ok"}`

📖 Interactive API docs at `http://localhost:8000/docs`

---

## 📁 Project Structure

```
financial-causal-agent/
├── 🤖 agent.py                # LangGraph agent (4 tools)
├── 🖥️ app.py                  # Streamlit dashboard
├── ⚡ api.py                   # FastAPI backend
├── 🔌 mcp_server.py           # MCP server for AI client interoperability
├── 📋 requirements.txt
├── data/
│   ├── generate_data.py       # Synthetic data with known causal effects
│   └── financial_data.csv
├── tools/
│   ├── causal_engine.py       # DoWhy 4-step pipeline
│   ├── causal_discovery.py    # PC algorithm + transfer entropy
│   └── rag_engine.py          # FAISS vector store
└── policies/
    ├── ecb_monetary_policy.txt
    ├── fund_risk_management.txt
    └── european_macro_indicators.txt
```

---

## 🐛 Notable Bug Fix

**Python 3.14 + NumPy compatibility:** Streamlit Cloud deployed on Python 3.14 where DoWhy's `test_stat_significance()` returns numpy arrays instead of floats. Fixed with `np.asarray().item()` for p-values, confidence intervals, and refutation results. [See commit →](https://github.com/sayoncamara/financial-causal-agent)

---

## 🗺️ Roadmap

- [x] **MCP Server** — expose causal analysis tools via Model Context Protocol
- [ ] **Docker + CI/CD** — containerization and automated testing pipeline
- [ ] **Additional causal methods** — instrumental variables, difference-in-differences
- [ ] **Real-time data feeds** — connect to live ECB/macro data sources
- [ ] **Multi-market support** — extend beyond Eurozone to Fed, BoE, BoJ

---

## 👤 Author

| |
|---|
| **Sayon Camara** <br> MSc Business Administration (Finance & Banking) — KU Leuven <br> Specialization in causal inference, machine learning & GenAI <br> [LinkedIn](https://www.linkedin.com/in/sayon-camara-aa2baa1a1/) · [GitHub](https://github.com/sayoncamara) |
