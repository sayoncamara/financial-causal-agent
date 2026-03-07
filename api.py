"""
Financial Causal Intelligence Agent — FastAPI Backend
=====================================================
POST /analyze  → accepts a question, returns structured JSON
GET  /health   → health check for Azure App Service

Run locally:  uvicorn api:app --reload --port 8000
Deploy:       Azure App Service (free tier)
"""

from __future__ import annotations

import os
import re
import time
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("causal-agent-api")

# ---------------------------------------------------------------------------
# Lazy-load the heavy agent module (imports DoWhy, FAISS, LangGraph, etc.)
# ---------------------------------------------------------------------------
_agent_graph = None
_ask_fn = None


def _load_agent():
    """Import agent.py once — it builds the graph and loads data at import time."""
    global _agent_graph, _ask_fn
    if _ask_fn is None:
        logger.info("Loading agent module (data + vectorstore + LangGraph)…")
        import agent as agent_module
        _agent_graph = agent_module.agent       # compiled LangGraph
        _ask_fn = agent_module.ask_agent        # convenience wrapper
        logger.info("Agent ready.")
    return _agent_graph, _ask_fn


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up on startup so the first request isn't slow."""
    try:
        _load_agent()
    except Exception as exc:
        logger.warning("Agent pre-load failed (will retry on first request): %s", exc)
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Financial Causal Intelligence Agent",
    version="0.1.0",
    description=(
        "Answers financial 'why' questions using real causal inference "
        "(DoWhy), causal discovery (PC algorithm), and RAG over policy documents."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        examples=["Why did fund outflows increase when ECB raised interest rates?"],
    )


class CausalEffect(BaseModel):
    treatment: str
    outcome: str
    estimate: float | None = None
    p_value: float | None = None
    refutation_passed: bool | None = None


class DiscoveredEdge(BaseModel):
    source: str
    target: str
    edge_type: str = "directed"


class PolicySource(BaseModel):
    source_file: str
    excerpt: str


class AnalyzeResponse(BaseModel):
    question: str
    answer: str
    causal_effects: list[CausalEffect] = []
    discovered_edges: list[DiscoveredEdge] = []
    policy_sources: list[PolicySource] = []
    processing_time_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Parsers — extract structured data from plain-text tool outputs
# ---------------------------------------------------------------------------

def _parse_causal_tool_output(text: str) -> CausalEffect | None:
    """
    Parse the text returned by analyze_causal_effect, e.g.:

        CAUSAL ANALYSIS: ecb_rate → bond_price_index
        ==================================================
        Estimated causal effect: -5.2987
        ...
        ✓ PASSED — Placebo Treatment: ...
        ✓ PASSED — Random Common Cause: ...
    """
    header = re.search(r"CAUSAL ANALYSIS:\s*(\S+)\s*→\s*(\S+)", text)
    if not header:
        return None
    treatment, outcome = header.group(1), header.group(2)

    effect_match = re.search(r"Estimated causal effect:\s*([-\d.]+)", text)
    estimate = float(effect_match.group(1)) if effect_match else None

    p_match = re.search(r"P-value:\s*([-\d.eE+]+)", text)
    p_value = float(p_match.group(1)) if p_match else None

    passes = text.count("✓ PASSED")
    failures = text.count("✗ FAILED")
    if passes + failures > 0:
        refutation_passed = failures == 0
    else:
        refutation_passed = None

    return CausalEffect(
        treatment=treatment,
        outcome=outcome,
        estimate=estimate,
        p_value=p_value,
        refutation_passed=refutation_passed,
    )


def _parse_discovery_tool_output(text: str) -> list[DiscoveredEdge]:
    """
    Parse output of discover_causal_structure, e.g.:

        Directed edges (cause → effect):
          inflation_rate → ecb_rate
        Undirected edges (direction unknown):
          equity_returns — fund_inflows
    """
    edges: list[DiscoveredEdge] = []

    for m in re.finditer(r"^\s+(\S+)\s*→\s*(\S+)", text, re.MULTILINE):
        edges.append(DiscoveredEdge(source=m.group(1), target=m.group(2), edge_type="directed"))

    for m in re.finditer(r"^\s+(\S+)\s*—\s*(\S+)", text, re.MULTILINE):
        edges.append(DiscoveredEdge(source=m.group(1), target=m.group(2), edge_type="undirected"))

    return edges


def _parse_rag_tool_output(text: str) -> list[PolicySource]:
    """
    Parse output of search_financial_policies, e.g.:

        [Source: policies/ecb_monetary_policy.txt]
        The ECB sets the main refinancing rate …

        ---

        [Source: policies/fund_risk_management.txt]
        Portfolio rebalancing triggers …
    """
    sources: list[PolicySource] = []
    blocks = re.split(r"\[Source:\s*([^\]]+)\]", text)
    # blocks alternates: preamble, source1, content1, source2, content2, …
    i = 1
    while i < len(blocks) - 1:
        source_file = blocks[i].strip()
        excerpt = blocks[i + 1].strip().split("\n---")[0].strip()[:300]
        sources.append(PolicySource(source_file=source_file, excerpt=excerpt))
        i += 2
    return sources


# ---------------------------------------------------------------------------
# Full response parser — walks LangGraph message history
# ---------------------------------------------------------------------------

def _parse_full_response(raw: dict[str, Any]) -> dict[str, Any]:
    """Walk the LangGraph message list and extract structured components."""
    from langchain_core.messages import AIMessage, ToolMessage

    result: dict[str, Any] = {
        "answer": "",
        "causal_effects": [],
        "discovered_edges": [],
        "policy_sources": [],
    }

    messages = raw.get("messages", [])

    for msg in messages:
        # Final AI answer = last AIMessage with content and no tool_calls
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            result["answer"] = msg.content

        # Tool outputs carry structured text
        if isinstance(msg, ToolMessage):
            name = getattr(msg, "name", "")
            content = msg.content if isinstance(msg.content, str) else str(msg.content)

            if name == "analyze_causal_effect":
                parsed = _parse_causal_tool_output(content)
                if parsed:
                    result["causal_effects"].append(parsed)

            elif name == "discover_causal_structure":
                result["discovered_edges"].extend(_parse_discovery_tool_output(content))

            elif name == "search_financial_policies":
                result["policy_sources"].extend(_parse_rag_tool_output(content))

    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """Run the causal intelligence agent on a financial question."""
    t0 = time.time()

    try:
        graph, _ = _load_agent()
    except Exception as exc:
        logger.error("Failed to load agent: %s", exc)
        raise HTTPException(status_code=503, detail="Agent unavailable. Check server logs.")

    from langchain_core.messages import HumanMessage
    try:
        raw = graph.invoke({"messages": [HumanMessage(content=req.question)]})
    except Exception as exc:
        logger.error("Agent invocation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")

    parsed = _parse_full_response(raw)
    elapsed = round(time.time() - t0, 2)

    return AnalyzeResponse(
        question=req.question,
        answer=parsed["answer"],
        causal_effects=parsed["causal_effects"],
        discovered_edges=parsed["discovered_edges"],
        policy_sources=parsed["policy_sources"],
        processing_time_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# Standalone
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
