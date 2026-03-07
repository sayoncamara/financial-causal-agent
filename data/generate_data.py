"""
Generate synthetic financial data based on a causal DAG
that models the monetary policy transmission mechanism
in the European fund industry.

Causal DAG:
    inflation_rate ──→ ecb_rate ──→ bond_price_index ──→ fund_outflows
                          │                                     ▲
                          ├──→ equity_returns ──→ fund_inflows  │
                          │                                     │
                          └──→ credit_spread ──→ client_risk_score ─┘
                                                    ▲
                                                    │
                                  unemployment_rate ─┘

Each variable is generated from its causal parents + noise,
so the data has KNOWN causal structure we can validate against.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_financial_data(n_months: int = 120, seed: int = 42) -> pd.DataFrame:
    """
    Generate n_months of synthetic monthly financial data.
    Default is 120 months (10 years: Jan 2015 - Dec 2024).
    
    The causal effects are calibrated to be realistic:
    - A 1% ECB rate hike causes ~5 point bond price drop
    - A 1% ECB rate hike causes ~2% equity return decline
    - A 1% ECB rate hike widens credit spreads by ~0.3%
    - Bond price drops cause fund outflows
    - Equity returns drive fund inflows
    - Credit spread widening reduces client risk appetite
    - Lower risk appetite amplifies fund outflows
    """
    np.random.seed(seed)
    
    # --- EXOGENOUS VARIABLES (no causal parents) ---
    
    # Inflation rate: mean ~2%, with a spike period (2022-2023 style)
    base_inflation = np.random.normal(2.0, 0.3, n_months)
    # Add a realistic inflation spike in months 84-96 (year 8-9, like 2022-2023)
    inflation_spike = np.zeros(n_months)
    if n_months > 96:
        inflation_spike[84:96] = np.linspace(0, 6, 12)  # gradual rise to ~8%
        inflation_spike[96:min(108, n_months)] = np.linspace(6, 1, min(12, n_months - 96))  # gradual fall
    inflation_rate = np.clip(base_inflation + inflation_spike, 0.5, 10.0)
    
    # Unemployment rate: mean ~7%, slow-moving
    unemployment_base = np.random.normal(7.0, 0.5, n_months)
    # Smooth it to make it realistic (unemployment doesn't jump month to month)
    unemployment_rate = np.zeros(n_months)
    unemployment_rate[0] = unemployment_base[0]
    for i in range(1, n_months):
        unemployment_rate[i] = 0.9 * unemployment_rate[i-1] + 0.1 * unemployment_base[i]
    unemployment_rate = np.clip(unemployment_rate, 4.0, 12.0)
    
    # --- ENDOGENOUS VARIABLES (caused by parents) ---
    
    # ECB rate: caused by inflation (ECB raises rates when inflation is high)
    # Causal effect: +0.4% rate per 1% inflation above target (2%)
    ecb_rate = np.zeros(n_months)
    ecb_rate[0] = 0.5  # starting rate
    for i in range(1, n_months):
        # ECB adjusts gradually toward target rate based on inflation
        target_rate = 0.0 + 0.4 * (inflation_rate[i] - 2.0)
        target_rate = np.clip(target_rate, -0.5, 4.5)
        # Gradual adjustment (ECB doesn't jump, it moves in steps)
        ecb_rate[i] = ecb_rate[i-1] + 0.15 * (target_rate - ecb_rate[i-1])
        ecb_rate[i] += np.random.normal(0, 0.05)  # small noise
    ecb_rate = np.clip(ecb_rate, -0.5, 5.0)
    
    # Bond price index: caused by ECB rate (inverse relationship)
    # Higher rates = lower bond prices (mechanical relationship)
    # Causal effect: -5 points per 1% rate increase
    bond_price_index = 100 - 5.0 * ecb_rate + np.random.normal(0, 1.5, n_months)
    bond_price_index = np.clip(bond_price_index, 70, 110)
    
    # Equity returns: caused by ECB rate (negative, but noisy)
    # Causal effect: -2% return per 1% rate increase
    equity_returns = 8.0 - 2.0 * ecb_rate + np.random.normal(0, 4.0, n_months)
    
    # Credit spread: caused by ECB rate (positive — higher rates widen spreads)
    # Causal effect: +0.3% spread per 1% rate increase
    credit_spread = 1.0 + 0.3 * ecb_rate + np.random.normal(0, 0.15, n_months)
    credit_spread = np.clip(credit_spread, 0.3, 4.0)
    
    # Client risk score: caused by credit_spread (negative) and unemployment (negative)
    # Scale 0-100, where 100 = maximum risk appetite
    # Causal effects: -10 points per 1% credit spread, -3 points per 1% unemployment
    client_risk_score = (
        80
        - 10.0 * (credit_spread - 1.0)  # spreads above 1% reduce appetite
        - 3.0 * (unemployment_rate - 6.0)  # unemployment above 6% reduces appetite
        + np.random.normal(0, 5.0, n_months)
    )
    client_risk_score = np.clip(client_risk_score, 10, 100)
    
    # Fund inflows (€M): caused by equity returns (positive)
    # Causal effect: +5€M per 1% equity return
    fund_inflows = 50 + 5.0 * equity_returns + np.random.normal(0, 10, n_months)
    fund_inflows = np.clip(fund_inflows, 5, 200)
    
    # Fund outflows (€M): caused by bond_price_index (negative) and client_risk_score (negative)
    # When bond prices drop, outflows increase; when risk appetite drops, outflows increase
    # Causal effects: -2€M per bond price point, -0.5€M per risk score point
    fund_outflows = (
        150
        - 2.0 * (bond_price_index - 90)  # bond prices below 90 increase outflows
        - 0.5 * (client_risk_score - 60)  # risk scores below 60 increase outflows
        + np.random.normal(0, 8, n_months)
    )
    fund_outflows = np.clip(fund_outflows, 10, 300)
    
    # --- BUILD DATAFRAME ---
    
    # Generate date range
    start_date = datetime(2015, 1, 1)
    dates = [start_date + timedelta(days=30 * i) for i in range(n_months)]
    
    df = pd.DataFrame({
        "date": dates,
        "inflation_rate": np.round(inflation_rate, 2),
        "unemployment_rate": np.round(unemployment_rate, 2),
        "ecb_rate": np.round(ecb_rate, 2),
        "bond_price_index": np.round(bond_price_index, 2),
        "equity_returns": np.round(equity_returns, 2),
        "credit_spread": np.round(credit_spread, 2),
        "client_risk_score": np.round(client_risk_score, 2),
        "fund_inflows": np.round(fund_inflows, 2),
        "fund_outflows": np.round(fund_outflows, 2),
    })
    
    return df


def get_true_causal_graph() -> dict:
    """
    Return the ground-truth causal graph as an adjacency list.
    This is the DAG we used to generate the data.
    Useful for validating causal discovery results.
    """
    return {
        "inflation_rate": ["ecb_rate"],
        "unemployment_rate": ["client_risk_score"],
        "ecb_rate": ["bond_price_index", "equity_returns", "credit_spread"],
        "bond_price_index": ["fund_outflows"],
        "equity_returns": ["fund_inflows"],
        "credit_spread": ["client_risk_score"],
        "client_risk_score": ["fund_outflows"],
        "fund_inflows": [],
        "fund_outflows": [],
    }


def get_true_causal_effects() -> dict:
    """
    Return the ground-truth causal effect sizes used in data generation.
    Useful for validating causal estimation results.
    """
    return {
        ("inflation_rate", "ecb_rate"): 0.4,
        ("ecb_rate", "bond_price_index"): -5.0,
        ("ecb_rate", "equity_returns"): -2.0,
        ("ecb_rate", "credit_spread"): 0.3,
        ("credit_spread", "client_risk_score"): -10.0,
        ("unemployment_rate", "client_risk_score"): -3.0,
        ("bond_price_index", "fund_outflows"): -2.0,
        ("client_risk_score", "fund_outflows"): -0.5,
        ("equity_returns", "fund_inflows"): 5.0,
    }


if __name__ == "__main__":
    # Generate the data
    df = generate_financial_data(n_months=120)
    
    # Save to CSV
    df.to_csv("data/financial_data.csv", index=False)
    
    # Print summary
    print(f"Generated {len(df)} months of financial data")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nVariable summary:")
    print(df.describe().round(2).to_string())
    
    # Print the true causal graph
    print(f"\nTrue causal graph (adjacency list):")
    for parent, children in get_true_causal_graph().items():
        if children:
            for child in children:
                print(f"  {parent} ──→ {child}")
    
    # Print true causal effects
    print(f"\nTrue causal effects:")
    for (cause, effect), size in get_true_causal_effects().items():
        print(f"  {cause} → {effect}: {size:+.1f}")
    
    print(f"\nData saved to data/financial_data.csv")