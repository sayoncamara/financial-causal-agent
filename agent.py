"""
Financial Causal Intelligence Agent

A LangGraph agent that answers financial "why" questions by
autonomously choosing between:
- Causal analysis (DoWhy) for effect estimation
- Causal discovery (PC algorithm) for structure learning
- RAG retrieval for policy context
- Data loading for raw financial data

The agent decides which tools to use based on the question,
executes them, and synthesizes the results into a clear explanation.
"""

import os
import json
import pandas as pd
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Load environment variables
load_dotenv()

# Import our custom tools
from tools.causal_engine import run_causal_analysis, get_all_direct_effects
from tools.causal_discovery import run_pc_algorithm, compare_discovered_vs_true
from tools.rag_engine import build_vector_store, search_policies
from data.generate_data import get_true_causal_graph

# --- LOAD DATA AND BUILD VECTOR STORE ---
DATA_PATH = "data/financial_data.csv"
df = pd.read_csv(DATA_PATH)
vectorstore = build_vector_store("policies")


# --- DEFINE TOOLS ---

@tool
def load_financial_data(
    start_date: str = "",
    end_date: str = "",
    variables: str = "",
) -> str:
    """
    Load financial data from the database. Optionally filter by date range
    and select specific variables.
    
    Args:
        start_date: Start date filter (e.g., "2022-01-01"). Empty for no filter.
        end_date: End date filter (e.g., "2023-12-31"). Empty for no filter.
        variables: Comma-separated variable names (e.g., "ecb_rate,fund_outflows"). Empty for all.
    
    Returns:
        Summary statistics of the requested data.
    """
    data = df.copy()
    
    # Apply date filters
    if start_date:
        data = data[data["date"] >= start_date]
    if end_date:
        data = data[data["date"] <= end_date]
    
    # Select specific variables
    if variables:
        var_list = [v.strip() for v in variables.split(",")]
        var_list = [v for v in var_list if v in data.columns]
        if var_list:
            cols = ["date"] + var_list
            data = data[[c for c in cols if c in data.columns]]
    
    if len(data) == 0:
        return "No data found for the specified filters."
    
    # Return summary
    summary = f"Data: {len(data)} months"
    if start_date or end_date:
        summary += f" ({data['date'].min()} to {data['date'].max()})"
    summary += f"\n\nStatistics:\n{data.describe().round(2).to_string()}"
    
    # Add recent trends
    numeric_cols = data.select_dtypes(include="number").columns
    if len(data) > 6:
        last_6 = data.tail(6)[numeric_cols].mean()
        first_6 = data.head(6)[numeric_cols].mean()
        changes = last_6 - first_6
        summary += f"\n\nChange (first 6 months vs last 6 months):\n"
        for col in numeric_cols:
            direction = "↑" if changes[col] > 0 else "↓"
            summary += f"  {col}: {direction} {changes[col]:+.2f}\n"
    
    return summary


@tool
def analyze_causal_effect(treatment: str, outcome: str) -> str:
    """
    Estimate the causal effect of one variable on another using DoWhy.
    This runs the full 4-step causal inference pipeline:
    model → identify → estimate → refute.
    
    Args:
        treatment: The cause variable (e.g., "ecb_rate", "inflation_rate", "credit_spread")
        outcome: The effect variable (e.g., "fund_outflows", "bond_price_index", "fund_inflows")
    
    Returns:
        Causal effect estimate with robustness checks.
    
    Available variables: inflation_rate, unemployment_rate, ecb_rate,
    bond_price_index, equity_returns, credit_spread, client_risk_score,
    fund_inflows, fund_outflows
    """
    try:
        result = run_causal_analysis(df, treatment, outcome)
        
        output = f"CAUSAL ANALYSIS: {treatment} → {outcome}\n"
        output += f"{'=' * 50}\n"
        output += f"Estimated causal effect: {result['causal_effect']}\n"
        output += f"(A 1-unit increase in {treatment} causes a "
        output += f"{abs(result['causal_effect']):.2f} unit "
        output += f"{'increase' if result['causal_effect'] > 0 else 'decrease'} "
        output += f"in {outcome})\n\n"
        
        if result["p_value"] is not None:
            output += f"P-value: {result['p_value']}\n"
            output += f"Statistically significant: {'Yes' if result['p_value'] < 0.05 else 'No'}\n"
        
        ci = result["confidence_interval"]
        if ci[0] is not None:
            output += f"95% Confidence interval: [{ci[0]}, {ci[1]}]\n"
        
        output += f"\nRobustness checks:\n"
        for r in result["refutation_results"]:
            status = "✓ PASSED" if r.get("passed") else "✗ FAILED"
            output += f"  {status} — {r['test']}: {r['description']}\n"
        
        output += f"\nInterpretation: {result['interpretation']}"
        
        return output
    
    except Exception as e:
        return f"Error in causal analysis: {str(e)}"


@tool
def discover_causal_structure(
    variables: str = "",
    significance_level: float = 0.05,
) -> str:
    """
    Automatically discover causal relationships from the data using 
    the PC algorithm. This finds which variables cause which, without
    assuming the causal graph in advance.
    
    Args:
        variables: Comma-separated variable names to analyze. Empty for all.
        significance_level: Threshold for edge detection (default 0.05, lower = fewer edges).
    
    Returns:
        Discovered causal graph with comparison to domain knowledge.
    """
    try:
        var_list = None
        if variables:
            var_list = [v.strip() for v in variables.split(",")]
        
        result = run_pc_algorithm(df, variables=var_list, alpha=significance_level)
        
        output = f"CAUSAL DISCOVERY RESULTS\n"
        output += f"{'=' * 50}\n"
        output += result["summary"]
        
        # Compare with true graph if using all variables
        if not var_list:
            true_graph = get_true_causal_graph()
            comparison = compare_discovered_vs_true(result["edges"], true_graph)
            output += f"\n\nComparison with domain knowledge:\n{comparison['summary']}"
        
        return output
    
    except Exception as e:
        return f"Error in causal discovery: {str(e)}"


@tool
def search_financial_policies(query: str) -> str:
    """
    Search financial policy documents for relevant information about
    ECB monetary policy, fund risk management, and European macro indicators.
    Use this when you need context about WHY certain financial relationships exist.
    
    Args:
        query: Natural language search query (e.g., "ECB rate hike impact on bond funds")
    
    Returns:
        Relevant policy excerpts with sources.
    """
    result = search_policies(vectorstore, query, k=3)
    
    output = f"POLICY SEARCH RESULTS ({result['num_results']} documents)\n"
    output += f"{'=' * 50}\n\n"
    output += result["summary"]
    
    return output


# --- DEFINE AGENT ---

# Agent state
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# All tools
tools = [
    load_financial_data,
    analyze_causal_effect,
    discover_causal_structure,
    search_financial_policies,
]

# LLM with tool binding
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.environ.get("OPENAI_API_KEY", ""),
).bind_tools(tools)

# System prompt
SYSTEM_PROMPT = """You are a Financial Causal Intelligence Agent specialized in 
analyzing European fund markets. You help users understand WHY financial events 
happen by using causal inference — not just correlations.

You have access to:
1. Financial data (120 months of European fund market data)
2. Causal analysis tools (DoWhy) that estimate TRUE causal effects with robustness checks
3. Causal discovery tools (PC algorithm) that find causal structure from data
4. Financial policy documents (ECB policy, risk management, macro indicators)

Your approach:
- When asked WHY something happened, use causal analysis tools (not just data lookup)
- Always ground explanations in both causal estimates AND policy context
- Clearly distinguish between correlation and causation
- Report robustness check results to show reliability
- Use plain business language, not technical jargon

Important: You use DoWhy for causal computation because research shows LLMs 
hallucinate causal relationships. You provide the explanation, DoWhy provides 
the causal evidence.

Available variables: inflation_rate, unemployment_rate, ecb_rate, 
bond_price_index, equity_returns, credit_spread, client_risk_score, 
fund_inflows, fund_outflows"""


def call_agent(state: AgentState):
    """Call the LLM with tools."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState):
    """Check if the agent should call tools or finish."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# Build the graph
graph_builder = StateGraph(AgentState)

# Add nodes
graph_builder.add_node("agent", call_agent)
graph_builder.add_node("tools", ToolNode(tools))

# Add edges
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "agent")

# Compile
agent = graph_builder.compile()


def ask_agent(question: str) -> str:
    """
    Send a question to the agent and get a response.
    The agent will autonomously decide which tools to use.
    
    Returns the final text response.
    """
    result = agent.invoke({
        "messages": [HumanMessage(content=question)]
    })
    
    # Get the last AI message (the final response)
    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage) and message.content and not message.tool_calls:
            return message.content
    
    return "The agent could not generate a response."


# --- TEST ---
if __name__ == "__main__":
    print("=" * 60)
    print("FINANCIAL CAUSAL INTELLIGENCE AGENT — TEST")
    print("=" * 60)
    
    # Test question
    question = "Why did fund outflows increase when ECB raised interest rates?"
    
    print(f"\nQuestion: {question}")
    print("-" * 60)
    
    response = ask_agent(question)
    print(f"\nAgent response:\n{response}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)