"""
Financial Causal Intelligence Agent — Streamlit UI
====================================================
Run:  streamlit run app.py
"""

import os
import re
import json
import time
import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Financial Causal Intelligence Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="st-"] {
        font-family: 'DM Sans', sans-serif;
    }
    code, pre, .stCode {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 {
        margin: 0 0 0.3rem 0;
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 0;
        opacity: 0.8;
        font-size: 0.95rem;
    }

    .result-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .result-card h3 {
        margin: 0 0 0.75rem 0;
        font-size: 1rem;
        font-weight: 600;
        color: #0f172a;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .metric-row {
        display: flex;
        gap: 0.75rem;
        flex-wrap: wrap;
        margin-bottom: 1rem;
    }
    .metric-chip {
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-size: 0.85rem;
    }
    .metric-chip .label {
        color: #64748b;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        display: block;
    }
    .metric-chip .value {
        font-weight: 600;
        color: #0f172a;
        font-size: 1.1rem;
    }

    section[data-testid="stSidebar"] {
        background: #f8fafc;
    }

    .source-tag {
        display: inline-block;
        background: #eff6ff;
        color: #1e40af;
        border: 1px solid #bfdbfe;
        border-radius: 6px;
        padding: 0.25rem 0.6rem;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.2rem 0.2rem 0.2rem 0;
    }

    .status-pass {
        color: #16a34a; background: #f0fdf4; border: 1px solid #bbf7d0;
        border-radius: 6px; padding: 0.15rem 0.5rem; font-size: 0.8rem; font-weight: 500;
    }
    .status-fail {
        color: #dc2626; background: #fef2f2; border: 1px solid #fecaca;
        border-radius: 6px; padding: 0.15rem 0.5rem; font-size: 0.8rem; font-weight: 500;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CAUSAL_EDGES = [
    ("inflation_rate", "ecb_rate"),
    ("ecb_rate", "bond_price_index"),
    ("ecb_rate", "equity_returns"),
    ("ecb_rate", "credit_spread"),
    ("bond_price_index", "fund_outflows"),
    ("equity_returns", "fund_inflows"),
    ("credit_spread", "client_risk_score"),
    ("unemployment_rate", "client_risk_score"),
    ("client_risk_score", "fund_outflows"),
]

VARIABLE_INFO = {
    "inflation_rate":     ("Inflation Rate",     "HICP year-on-year % change (Eurozone)"),
    "ecb_rate":           ("ECB Rate",           "Main refinancing operations rate"),
    "bond_price_index":   ("Bond Price Index",   "Euro-area sovereign bond index"),
    "equity_returns":     ("Equity Returns",     "STOXX Europe 600 monthly return"),
    "credit_spread":      ("Credit Spread",      "BBB–Bund spread in basis points"),
    "fund_inflows":       ("Fund Inflows",       "Monthly net inflows (€ millions)"),
    "fund_outflows":      ("Fund Outflows",      "Monthly net outflows (€ millions)"),
    "client_risk_score":  ("Client Risk Score",  "Composite risk metric (0–100)"),
    "unemployment_rate":  ("Unemployment Rate",  "Eurozone harmonised rate"),
}

EXAMPLE_QUESTIONS = [
    "Why did fund outflows increase when ECB raised interest rates?",
    "What is the causal effect of inflation on bond prices?",
    "How does unemployment affect client risk scores?",
    "What drives credit spread changes in the Eurozone?",
    "Why do equity returns drop when ECB tightens monetary policy?",
]


# ---------------------------------------------------------------------------
# Agent loader
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_agent():
    """
    Import agent.py — this loads data, builds the vectorstore, and
    compiles the LangGraph at import time. Returns (compiled_graph, is_mock).
    """
    try:
        import agent as agent_module
        return agent_module.agent, False
    except Exception as e:
        st.warning(
            f"⚠️ Agent not available (`{e}`). Running in **demo mode** with sample data."
        )
        return None, True


# ---------------------------------------------------------------------------
# Text parsers (same logic as api.py — tools return plain text, not JSON)
# ---------------------------------------------------------------------------

def _parse_causal_effect(text: str) -> dict | None:
    header = re.search(r"CAUSAL ANALYSIS:\s*(\S+)\s*→\s*(\S+)", text)
    if not header:
        return None
    treatment, outcome = header.group(1), header.group(2)
    effect_m = re.search(r"Estimated causal effect:\s*([-\d.]+)", text)
    estimate = float(effect_m.group(1)) if effect_m else None
    p_m = re.search(r"P-value:\s*([-\d.eE+]+)", text)
    p_value = float(p_m.group(1)) if p_m else None
    passes = text.count("✓ PASSED")
    failures = text.count("✗ FAILED")
    refutation_passed = (failures == 0) if (passes + failures) > 0 else None
    return {
        "treatment": treatment, "outcome": outcome,
        "estimate": estimate, "p_value": p_value,
        "refutation_passed": refutation_passed,
    }


def _parse_discovery_edges(text: str) -> list[dict]:
    edges = []
    for m in re.finditer(r"^\s+(\S+)\s*→\s*(\S+)", text, re.MULTILINE):
        edges.append({"source": m.group(1), "target": m.group(2), "edge_type": "directed"})
    for m in re.finditer(r"^\s+(\S+)\s*—\s*(\S+)", text, re.MULTILINE):
        edges.append({"source": m.group(1), "target": m.group(2), "edge_type": "undirected"})
    return edges


def _parse_policy_sources(text: str) -> list[dict]:
    sources = []
    blocks = re.split(r"\[Source:\s*([^\]]+)\]", text)
    i = 1
    while i < len(blocks) - 1:
        source_file = blocks[i].strip()
        excerpt = blocks[i + 1].strip().split("\n---")[0].strip()[:300]
        sources.append({"source_file": source_file, "excerpt": excerpt})
        i += 2
    return sources


# ---------------------------------------------------------------------------
# Run agent
# ---------------------------------------------------------------------------
def run_question(question: str, graph, is_mock: bool) -> dict:
    if is_mock:
        return _mock_response(question)

    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

    t0 = time.time()
    try:
        raw = graph.invoke({"messages": [HumanMessage(content=question)]})
    except Exception as e:
        return {"answer": f"❌ Agent error: {e}", "causal_effects": [],
                "discovered_edges": [], "policy_sources": [], "processing_time": 0}

    result = {"answer": "", "causal_effects": [], "discovered_edges": [],
              "policy_sources": [], "processing_time": round(time.time() - t0, 2)}

    for msg in raw.get("messages", []):
        # Final answer: last AIMessage with content and no tool_calls
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            result["answer"] = msg.content

        if isinstance(msg, ToolMessage):
            name = getattr(msg, "name", "")
            content = msg.content if isinstance(msg.content, str) else str(msg.content)

            if name == "analyze_causal_effect":
                parsed = _parse_causal_effect(content)
                if parsed:
                    result["causal_effects"].append(parsed)
            elif name == "discover_causal_structure":
                result["discovered_edges"].extend(_parse_discovery_edges(content))
            elif name == "search_financial_policies":
                result["policy_sources"].extend(_parse_policy_sources(content))

    return result


def _mock_response(question: str) -> dict:
    time.sleep(1.5)
    return {
        "answer": (
            f"**Analysis of: {question}**\n\n"
            "Based on causal inference analysis using DoWhy, there is a statistically "
            "significant causal pathway from ECB rate changes to fund outflows, mediated "
            "through the bond price index.\n\n"
            "**Key findings:**\n"
            "- ECB rate → Bond Price Index: estimated effect of **−5.30** (true: −5.0), "
            "both refutation tests passed ✓\n"
            "- Bond Price Index → Fund Outflows: estimated effect of **−0.42** (true: −0.4), "
            "refutation tests passed ✓\n\n"
            "**Policy context (RAG):** ECB monetary policy documents confirm that rate "
            "increases mechanically depress bond valuations, which triggers portfolio "
            "rebalancing and outflows from fixed-income funds."
        ),
        "causal_effects": [
            {"treatment": "ecb_rate", "outcome": "bond_price_index",
             "estimate": -5.30, "p_value": 0.001, "refutation_passed": True},
            {"treatment": "bond_price_index", "outcome": "fund_outflows",
             "estimate": -0.42, "p_value": 0.003, "refutation_passed": True},
            {"treatment": "ecb_rate", "outcome": "equity_returns",
             "estimate": -2.15, "p_value": 0.012, "refutation_passed": True},
        ],
        "discovered_edges": [
            {"source": "inflation_rate", "target": "ecb_rate", "edge_type": "directed"},
            {"source": "ecb_rate", "target": "bond_price_index", "edge_type": "directed"},
            {"source": "unemployment_rate", "target": "client_risk_score", "edge_type": "undirected"},
        ],
        "policy_sources": [
            {"source_file": "policies/ecb_monetary_policy.txt",
             "excerpt": "The transmission mechanism of monetary policy operates through several channels, including the interest rate channel affecting bond valuations…"},
            {"source_file": "policies/fund_risk_management.txt",
             "excerpt": "Portfolio rebalancing triggers are activated when key indices deviate beyond 2σ from rolling 12-month averages…"},
        ],
        "processing_time": 3.2,
    }


# ---------------------------------------------------------------------------
# Causal graph visualization
# ---------------------------------------------------------------------------
def draw_causal_graph(highlight_edges: list[dict] | None = None):
    """Render the causal DAG. Optionally highlight discovered edges."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        st.info("Install `networkx` and `matplotlib` to see the causal graph.")
        return

    G = nx.DiGraph()
    G.add_edges_from(CAUSAL_EDGES)

    pos = {
        "inflation_rate":     (-2.0,  2.0),
        "ecb_rate":           ( 0.0,  2.0),
        "bond_price_index":   ( 2.0,  3.0),
        "equity_returns":     ( 2.0,  2.0),
        "credit_spread":      ( 2.0,  1.0),
        "fund_inflows":       ( 4.0,  2.5),
        "fund_outflows":      ( 4.0,  3.5),
        "client_risk_score":  ( 4.0,  1.0),
        "unemployment_rate":  ( 2.0,  0.0),
    }

    short = {
        "inflation_rate": "Inflation", "ecb_rate": "ECB Rate",
        "bond_price_index": "Bond Index", "equity_returns": "Equity Ret.",
        "credit_spread": "Credit Spread", "fund_inflows": "Fund Inflows",
        "fund_outflows": "Fund Outflows", "client_risk_score": "Risk Score",
        "unemployment_rate": "Unemployment",
    }

    # Determine which edges to highlight (from discovered edges)
    highlighted = set()
    if highlight_edges:
        for e in highlight_edges:
            highlighted.add((e["source"], e["target"]))

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")

    # Draw non-highlighted edges first
    normal_edges = [e for e in G.edges() if e not in highlighted]
    highlight_edge_list = [e for e in G.edges() if e in highlighted]

    nx.draw_networkx_edges(
        G, pos, edgelist=normal_edges, ax=ax,
        edge_color="#cbd5e1", width=1.5, arrows=True, arrowsize=16,
        arrowstyle="-|>", connectionstyle="arc3,rad=0.1",
        min_source_margin=22, min_target_margin=22,
    )

    if highlight_edge_list:
        nx.draw_networkx_edges(
            G, pos, edgelist=highlight_edge_list, ax=ax,
            edge_color="#2563eb", width=2.5, arrows=True, arrowsize=18,
            arrowstyle="-|>", connectionstyle="arc3,rad=0.1",
            min_source_margin=22, min_target_margin=22,
        )

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#1e3a5f", node_size=1600, alpha=0.9)
    nx.draw_networkx_labels(G, pos, labels=short, ax=ax,
                            font_size=8, font_weight="bold", font_color="white")

    if highlight_edge_list:
        ax.text(0.01, 0.01, "Blue edges = discovered / analyzed in this query",
                transform=ax.transAxes, fontsize=8, color="#2563eb", alpha=0.8)

    ax.margins(0.15)
    ax.axis("off")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Load data for sidebar
# ---------------------------------------------------------------------------
@st.cache_data
def load_data_summary():
    for p in ["data/financial_data.csv", "../data/financial_data.csv"]:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None


# ===================================================================
#  LAYOUT
# ===================================================================

# ----- Sidebar -----
with st.sidebar:
    st.markdown("### 🔬 About This Agent")
    st.markdown(
        "Answers financial **\"why\"** questions using "
        "**real causal inference** (DoWhy), not LLM correlations.\n\n"
        "**Pipeline:** LangGraph orchestrates DoWhy (causal estimation), "
        "PC algorithm (causal discovery), and FAISS RAG (policy context)."
    )

    st.markdown("---")
    st.markdown("### 📊 Variables")
    for var_key, (name, desc) in VARIABLE_INFO.items():
        st.markdown(
            f"**{name}**  \n"
            f"<span style='color:#64748b;font-size:0.85rem'>{desc}</span>",
            unsafe_allow_html=True,
        )

    df_summary = load_data_summary()
    if df_summary is not None:
        st.markdown("---")
        st.markdown(f"### 📈 Data Overview ({len(df_summary)} months)")
        st.dataframe(
            df_summary.describe().round(2).T[["mean", "std", "min", "max"]],
            use_container_width=True, height=340,
        )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.8rem;color:#94a3b8;text-align:center'>"
        "Built by Sayon Camara · KU Leuven<br>"
        "DoWhy · LangGraph · FAISS · FastAPI"
        "</div>",
        unsafe_allow_html=True,
    )

# ----- Header -----
st.markdown("""
<div class="main-header">
    <h1>🔬 Financial Causal Intelligence Agent</h1>
    <p>Ask a financial "why" question — get causal inference, not just correlations</p>
</div>
""", unsafe_allow_html=True)

# ----- Load agent -----
graph, is_mock = load_agent()

# ----- Question input -----
# Use session state to handle example question clicks
if "question_text" not in st.session_state:
    st.session_state.question_text = ""

col_input, col_examples = st.columns([3, 2])

with col_input:
    question = st.text_area(
        "Your question",
        value=st.session_state.question_text,
        placeholder="e.g. Why did fund outflows increase when ECB raised interest rates?",
        height=80,
        label_visibility="collapsed",
    )

with col_examples:
    st.markdown("**💡 Try an example:**")
    for eq in EXAMPLE_QUESTIONS:
        if st.button(eq, key=f"ex_{eq}", use_container_width=True):
            st.session_state.question_text = eq
            st.rerun()

analyze_btn = st.button("🔍  Analyze", type="primary", use_container_width=True)

# ----- Results -----
if analyze_btn and question.strip():
    with st.spinner("Running causal analysis… (agent selects tools autonomously)"):
        result = run_question(question.strip(), graph, is_mock)

    # Store result so it survives interactions
    st.session_state.last_result = result
    st.session_state.last_question = question.strip()

# Display results if we have them
if "last_result" in st.session_state:
    result = st.session_state.last_result

    n_effects = len(result.get("causal_effects", []))
    n_edges = len(result.get("discovered_edges", []))
    n_sources = len(result.get("policy_sources", []))
    proc_time = result.get("processing_time", 0)

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-chip"><span class="label">Processing time</span><span class="value">{proc_time}s</span></div>
        <div class="metric-chip"><span class="label">Causal effects</span><span class="value">{n_effects}</span></div>
        <div class="metric-chip"><span class="label">Discovered edges</span><span class="value">{n_edges}</span></div>
        <div class="metric-chip"><span class="label">Policy sources</span><span class="value">{n_sources}</span></div>
    </div>
    """, unsafe_allow_html=True)

    # ----- Agent answer -----
    st.markdown('<div class="result-card"><h3>💬 Agent Response</h3></div>', unsafe_allow_html=True)
    st.markdown(result.get("answer", "No response generated."))

    # ----- Two columns: graph + effects -----
    col_graph, col_effects = st.columns([1.2, 1])

    with col_graph:
        st.markdown('<div class="result-card"><h3>🕸️ Causal DAG</h3></div>', unsafe_allow_html=True)
        discovered = result.get("discovered_edges", []) + [
            {"source": e["treatment"], "target": e["outcome"]}
            for e in result.get("causal_effects", [])
            if e.get("treatment") and e.get("outcome")
        ]
        draw_causal_graph(highlight_edges=discovered if discovered else None)

    with col_effects:
        st.markdown('<div class="result-card"><h3>📐 Causal Effect Estimates</h3></div>', unsafe_allow_html=True)
        effects = result.get("causal_effects", [])
        if effects:
            for eff in effects:
                treatment = eff.get("treatment", "?")
                outcome = eff.get("outcome", "?")
                estimate = eff.get("estimate", "N/A")
                p_val = eff.get("p_value")
                passed = eff.get("refutation_passed")

                est_str = f"{estimate}" if isinstance(estimate, str) else f"{estimate:.4f}"
                p_str = f"p = {p_val:.4f}" if p_val is not None else ""
                badge = ""
                if passed is True:
                    badge = '<span class="status-pass">✓ Refutation passed</span>'
                elif passed is False:
                    badge = '<span class="status-fail">✗ Refutation failed</span>'

                st.markdown(f"""
                <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
                            padding:0.75rem 1rem;margin-bottom:0.5rem;">
                    <div style="font-weight:600;font-size:0.9rem;color:#0f172a;">
                        {treatment} → {outcome}
                    </div>
                    <div style="display:flex;align-items:center;gap:0.75rem;margin-top:0.3rem;">
                        <span style="font-size:1.2rem;font-weight:700;color:#1e3a5f;">{est_str}</span>
                        <span style="font-size:0.8rem;color:#64748b;">{p_str}</span>
                        {badge}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No causal effects estimated for this query.")

    # ----- Policy sources -----
    sources = result.get("policy_sources", [])
    if sources:
        st.markdown('<div class="result-card"><h3>📄 Retrieved Policy Sources (RAG)</h3></div>', unsafe_allow_html=True)
        for src in sources:
            doc_name = src.get("source_file", src.get("document", "Unknown"))
            excerpt = src.get("excerpt", "")
            st.markdown(f"""
            <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;
                        padding:0.75rem 1rem;margin-bottom:0.5rem;">
                <span class="source-tag">{doc_name}</span>
                <p style="margin:0.4rem 0 0 0;font-size:0.9rem;color:#334155;">{excerpt}</p>
            </div>
            """, unsafe_allow_html=True)

elif analyze_btn:
    st.warning("Please enter a question to analyze.")

# ----- Always-visible DAG expander -----
with st.expander("📖 View Full Causal DAG & Edge Definitions", expanded=False):
    draw_causal_graph()
    edge_data = [{"Source": s, "Target": t} for s, t in CAUSAL_EDGES]
    st.dataframe(pd.DataFrame(edge_data), use_container_width=True, hide_index=True)