# Financial Causal Intelligence Agent

An AI agent that answers financial **"why"** questions using **real causal inference** — not just LLM correlations.

> *"Why did fund outflows increase when ECB raised interest rates?"*

Instead of guessing, the agent runs [DoWhy](https://github.com/py-why/dowhy) causal analysis with robustness checks, discovers causal structure with the PC algorithm, and retrieves policy context via RAG — then synthesizes everything into a clear explanation.

**[Live Demo](https://financial-causal-agent-kxda65qww6bbx4yxnzbcf9.streamlit.app/)**

---

## Architecture

```
                         Streamlit UI (app.py)
              Question input | Results | Causal DAG viz
                              |
                    LangGraph Agent (agent.py)
            Autonomously selects tools based on question
                              |
         ┌────────┬───────────┼───────────┬────────────┐
         v        v           v           v            v
      Data     DoWhy       PC Algo     FAISS RAG   FastAPI
      Load     4-step      causal      policy      backend
               pipeline    discovery   docs        (api.py)
```

### Causal DAG (Domain Knowledge)

```
inflation_rate --> ecb_rate --> bond_price_index --> fund_outflows
                      |                                   ^
                      |--> equity_returns --> fund_inflows |
                      |                                   |
                      '--> credit_spread --> client_risk_score
                                                ^
                                                |
                              unemployment_rate -'
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **DoWhy computes, LLM explains** | Research shows LLMs hallucinate causal relationships (Han et al., 2024). We separate computation from explanation. |
| **Refutation tests visible** | Placebo treatment + random common cause tests shown to user — separates this from toy demos. |
| **PC algorithm for discovery** | Cross-sectional causal discovery complements the domain DAG. Transfer entropy explored but insufficient data (120 months) — documented honestly. |
| **Credit spread as mediator** | Supported by monetary policy transmission mechanism literature. |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Agent orchestration | LangGraph v0.2.60 |
| Causal inference | DoWhy (model, identify, estimate, refute) |
| Causal discovery | causal-learn (PC algorithm) |
| LLM | OpenAI GPT-4o-mini |
| RAG | FAISS + LangChain v0.3.25 |
| Frontend | Streamlit |
| API | FastAPI |

---

## Quickstart

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
# Streamlit UI
python -m streamlit run app.py

# FastAPI backend
python -m uvicorn api:app --reload --port 8000

# CLI test
python agent.py
```

---

## API

**POST /analyze**
```json
{
  "question": "Why did fund outflows increase when ECB raised interest rates?"
}
```

Returns structured JSON with: answer, causal_effects (with estimates, p-values, refutation results), discovered_edges, policy_sources, and processing time.

**GET /health** — returns `{"status": "ok"}`

API docs at `http://localhost:8000/docs`

---

## Project Structure

```
financial-causal-agent/
├── agent.py                  # LangGraph agent (4 tools)
├── app.py                    # Streamlit dashboard
├── api.py                    # FastAPI backend
├── requirements.txt
├── data/
│   ├── generate_data.py      # Synthetic data with known causal effects
│   └── financial_data.csv
├── tools/
│   ├── causal_engine.py      # DoWhy 4-step pipeline
│   ├── causal_discovery.py   # PC algorithm + transfer entropy
│   └── rag_engine.py         # FAISS vector store
└── policies/
    ├── ecb_monetary_policy.txt
    ├── fund_risk_management.txt
    └── european_macro_indicators.txt
```

---

## Example Results

**Question:** *"How does unemployment affect client risk scores?"*

| Metric | Value |
|--------|-------|
| Causal effect | -0.40 (true: -0.4) |
| P-value | 0.853 |
| Placebo treatment | Passed |
| Random common cause | Passed |

The agent correctly estimates the effect and honestly reports it is not statistically significant.

---

## Author

**Sayon Camara** — MSc Business Administration (Finance & Banking), KU Leuven
