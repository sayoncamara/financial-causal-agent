"""
Financial Causal Intelligence Agent — MCP Server

Exposes the agent's causal analysis capabilities via the Model Context Protocol,
allowing any MCP-compatible client (Claude Desktop, Cursor, etc.) to use:

1. analyze_causal_effect — DoWhy causal inference with robustness checks
2. discover_causal_structure — PC algorithm for causal discovery
3. query_financial_policies — RAG search over ECB/macro policy documents
4. load_financial_data — Access to 120 months of European fund market data
"""

import os
import json
import sys
import pandas as pd
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# Load environment variables
load_dotenv()

# Add project root to path so we can import existing tools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.causal_engine import run_causal_analysis, get_all_direct_effects
from tools.causal_discovery import run_pc_algorithm, compare_discovered_vs_true
from tools.rag_engine import build_vector_store, search_policies
from data.generate_data import get_true_causal_graph

# --- Initialize ---
DATA_PATH = "data/financial_data.csv"
df = pd.read_csv(DATA_PATH)
vectorstore = build_vector_store("policies")

# Create MCP server
mcp = FastMCP("Financial Causal Intelligence Agent")

AVAILABLE_VARIABLES = [
    "inflation_rate", "unemployment_rate", "ecb_rate",
    "bond_price_index", "equity_returns", "credit_spread",
    "client_risk_score", "fund_inflows", "fund_outflows",
]


@mcp.tool()
def analyze_causal_effect(treatment: str, outcome: str) -> str:
    """
    Estimate the causal effect of one financial variable on another using DoWhy.
    Runs the full 4-step causal inference pipeline: model → identify → estimate → refute.

    Returns the effect size, p-value, confidence interval, and robustness check results
    (placebo treatment + random common cause tests).

    Available variables: inflation_rate, unemployment_rate, ecb_rate,
    bond_price_index, equity_returns, credit_spread, client_risk_score,
    fund_inflows, fund_outflows

    Example: treatment="ecb_rate", outcome="bond_price_index"
    """
    if treatment not in AVAILABLE_VARIABLES:
        return f"Unknown treatment variable: {treatment}. Available: {', '.join(AVAILABLE_VARIABLES)}"
    if outcome not in AVAILABLE_VARIABLES:
        return f"Unknown outcome variable: {outcome}. Available: {', '.join(AVAILABLE_VARIABLES)}"
    if treatment == outcome:
        return "Treatment and outcome must be different variables."

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
            status = "PASSED" if r.get("passed") else "FAILED"
            output += f"  {status} — {r['test']}: {r['description']}\n"

        output += f"\nInterpretation: {result['interpretation']}"
        return output

    except Exception as e:
        return f"Error in causal analysis: {str(e)}"


@mcp.tool()
def discover_causal_structure(
    variables: str = "",
    significance_level: float = 0.05,
) -> str:
    """
    Discover causal relationships from data using the PC algorithm.
    Finds which variables cause which without assuming the causal graph in advance.

    Args:
        variables: Comma-separated variable names to analyze. Empty string for all variables.
        significance_level: Threshold for edge detection (default 0.05, lower = fewer edges).

    Returns discovered edges with comparison to domain knowledge DAG.

    Available variables: inflation_rate, unemployment_rate, ecb_rate,
    bond_price_index, equity_returns, credit_spread, client_risk_score,
    fund_inflows, fund_outflows
    """
    try:
        var_list = None
        if variables:
            var_list = [v.strip() for v in variables.split(",")]

        result = run_pc_algorithm(df, variables=var_list, alpha=significance_level)

        output = f"CAUSAL DISCOVERY RESULTS\n"
        output += f"{'=' * 50}\n"
        output += result["summary"]

        if not var_list:
            true_graph = get_true_causal_graph()
            comparison = compare_discovered_vs_true(result["edges"], true_graph)
            output += f"\n\nComparison with domain knowledge:\n{comparison['summary']}"

        return output

    except Exception as e:
        return f"Error in causal discovery: {str(e)}"


@mcp.tool()
def query_financial_policies(query: str) -> str:
    """
    Search ECB monetary policy, fund risk management, and European macro indicator
    documents using RAG (FAISS vector similarity search).

    Use this to understand WHY certain financial relationships exist — the policy
    and regulatory context behind the causal mechanisms.

    Args:
        query: Natural language search query (e.g., "ECB rate hike impact on bond funds")
    """
    try:
        result = search_policies(vectorstore, query, k=3)

        output = f"POLICY SEARCH RESULTS ({result['num_results']} documents)\n"
        output += f"{'=' * 50}\n\n"
        output += result["summary"]
        return output

    except Exception as e:
        return f"Error in policy search: {str(e)}"


@mcp.tool()
def load_financial_data(
    start_date: str = "",
    end_date: str = "",
    variables: str = "",
) -> str:
    """
    Load European fund market data (120 months). Returns summary statistics
    and recent trends. Optionally filter by date range and variables.

    Args:
        start_date: Start date filter (e.g., "2022-01-01"). Empty for no filter.
        end_date: End date filter (e.g., "2023-12-31"). Empty for no filter.
        variables: Comma-separated variable names. Empty for all.

    Available variables: inflation_rate, unemployment_rate, ecb_rate,
    bond_price_index, equity_returns, credit_spread, client_risk_score,
    fund_inflows, fund_outflows
    """
    data = df.copy()

    if start_date:
        data = data[data["date"] >= start_date]
    if end_date:
        data = data[data["date"] <= end_date]
    if variables:
        var_list = [v.strip() for v in variables.split(",")]
        var_list = [v for v in var_list if v in data.columns]
        if var_list:
            cols = ["date"] + var_list
            data = data[[c for c in cols if c in data.columns]]

    if len(data) == 0:
        return "No data found for the specified filters."

    summary = f"Data: {len(data)} months"
    if start_date or end_date:
        summary += f" ({data['date'].min()} to {data['date'].max()})"
    summary += f"\n\nStatistics:\n{data.describe().round(2).to_string()}"

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


# --- Resources ---

@mcp.resource("causal://dag")
def get_causal_dag() -> str:
    """The domain knowledge causal DAG for the European fund market model."""
    return json.dumps({
        "description": "Causal DAG for European fund market — monetary policy transmission mechanism",
        "variables": AVAILABLE_VARIABLES,
        "edges": [
            {"from": "inflation_rate", "to": "ecb_rate", "mechanism": "ECB responds to inflation"},
            {"from": "ecb_rate", "to": "bond_price_index", "mechanism": "Rate hikes reduce bond prices"},
            {"from": "ecb_rate", "to": "equity_returns", "mechanism": "Higher rates compress equity valuations"},
            {"from": "ecb_rate", "to": "credit_spread", "mechanism": "Rate changes affect credit risk pricing"},
            {"from": "bond_price_index", "to": "fund_outflows", "mechanism": "Falling bond prices trigger redemptions"},
            {"from": "equity_returns", "to": "fund_inflows", "mechanism": "Rising equities attract capital"},
            {"from": "credit_spread", "to": "client_risk_score", "mechanism": "Wider spreads increase perceived risk"},
            {"from": "unemployment_rate", "to": "client_risk_score", "mechanism": "Higher unemployment raises default risk"},
            {"from": "client_risk_score", "to": "fund_outflows", "mechanism": "Higher risk scores trigger defensive flows"},
        ],
    }, indent=2)


@mcp.resource("causal://variables")
def get_variable_descriptions() -> str:
    """Descriptions of all available financial variables."""
    return json.dumps({
        "inflation_rate": "Eurozone HICP inflation rate (%)",
        "unemployment_rate": "Eurozone unemployment rate (%)",
        "ecb_rate": "ECB main refinancing operations rate (%)",
        "bond_price_index": "European government bond price index",
        "equity_returns": "Euro Stoxx 50 monthly returns (%)",
        "credit_spread": "Investment-grade credit spread (bps)",
        "client_risk_score": "Composite client risk score (0-100)",
        "fund_inflows": "Monthly fund inflows (EUR millions)",
        "fund_outflows": "Monthly fund outflows (EUR millions)",
    }, indent=2)


if __name__ == "__main__":
    mcp.run()
