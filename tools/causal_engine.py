"""
Causal inference engine using DoWhy.

This module provides the core causal analysis capabilities:
1. Define a causal model from a DAG + data
2. Identify the causal estimand
3. Estimate the causal effect
4. Refute the estimate with robustness checks

The key insight: we use DoWhy for the actual causal computation,
NOT the LLM. Research shows LLMs hallucinate causal relationships,
so we separate computation (DoWhy) from explanation (LLM).
"""

import pandas as pd
import numpy as np
import warnings
from dowhy import CausalModel

# Suppress verbose DoWhy output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def get_causal_graph_gml() -> str:
    """
    Return the causal DAG in GML format (DoWhy's preferred input format).
    This encodes our domain knowledge about the monetary policy
    transmission mechanism in European fund markets.
    """
    return """
    graph [
        directed 1
        node [ id "inflation_rate" label "inflation_rate" ]
        node [ id "unemployment_rate" label "unemployment_rate" ]
        node [ id "ecb_rate" label "ecb_rate" ]
        node [ id "bond_price_index" label "bond_price_index" ]
        node [ id "equity_returns" label "equity_returns" ]
        node [ id "credit_spread" label "credit_spread" ]
        node [ id "client_risk_score" label "client_risk_score" ]
        node [ id "fund_inflows" label "fund_inflows" ]
        node [ id "fund_outflows" label "fund_outflows" ]
        
        edge [ source "inflation_rate" target "ecb_rate" ]
        edge [ source "ecb_rate" target "bond_price_index" ]
        edge [ source "ecb_rate" target "equity_returns" ]
        edge [ source "ecb_rate" target "credit_spread" ]
        edge [ source "credit_spread" target "client_risk_score" ]
        edge [ source "unemployment_rate" target "client_risk_score" ]
        edge [ source "bond_price_index" target "fund_outflows" ]
        edge [ source "client_risk_score" target "fund_outflows" ]
        edge [ source "equity_returns" target "fund_inflows" ]
    ]
    """


def run_causal_analysis(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
) -> dict:
    """
    Run the full DoWhy 4-step causal inference pipeline:
    1. Model: create CausalModel with our DAG
    2. Identify: find the causal estimand (what to compute)
    3. Estimate: compute the causal effect using linear regression
    4. Refute: run robustness checks (placebo treatment + random common cause)
    
    Parameters:
        df: DataFrame with financial data
        treatment: the cause variable (e.g., "ecb_rate")
        outcome: the effect variable (e.g., "fund_outflows")
    
    Returns:
        dict with keys:
            - treatment, outcome: the variable names
            - estimand: string description of what we're estimating
            - causal_effect: the estimated causal effect (float)
            - p_value: statistical significance
            - confidence_interval: (low, high) tuple
            - refutation_results: list of refutation test results
            - interpretation: human-readable interpretation
    """
    # Drop the date column for analysis
    data = df.drop(columns=["date"], errors="ignore")
    
    # Step 1: MODEL — create the causal model
    model = CausalModel(
        data=data,
        treatment=treatment,
        outcome=outcome,
        graph=get_causal_graph_gml(),
    )
    
    # Step 2: IDENTIFY — find the causal estimand
    identified_estimand = model.identify_effect(
        proceed_when_unidentifiable=True
    )
    
    # Step 3: ESTIMATE — compute the causal effect
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
        confidence_intervals=True,
        test_significance=True,
    )
    
    # Extract results
    causal_effect = float(np.asarray(estimate.value).item())
    
    # Get p-value and confidence interval safely
    # Note: on Python 3.14+ / newer numpy, DoWhy may return numpy arrays
    # instead of plain floats, so we use np.asarray().item() throughout.
    p_value = None
    if hasattr(estimate, "test_stat_significance") and estimate.test_stat_significance():
        sig = estimate.test_stat_significance()
        if isinstance(sig, dict) and "p_value" in sig:
            p_value = float(np.asarray(sig["p_value"]).item())
        elif isinstance(sig, (list, tuple)) and len(sig) > 0:
            p_value = float(np.asarray(sig[0]).item())
    
    ci_low, ci_high = None, None
    if hasattr(estimate, "get_confidence_intervals"):
        try:
            ci = estimate.get_confidence_intervals()
            if ci is not None:
                ci_arr = np.asarray(ci)
                if ci_arr.ndim == 1 and len(ci_arr) >= 2:
                    ci_low, ci_high = float(ci_arr[0].item()), float(ci_arr[1].item())
                elif ci_arr.ndim == 2 and ci_arr.shape[0] >= 1:
                    ci_low, ci_high = float(ci_arr[0][0].item()), float(ci_arr[0][1].item())
        except Exception:
            pass
    
    # Step 4: REFUTE — run robustness checks
    refutation_results = []
    
    # Refutation 1: Placebo treatment (replace treatment with random noise)
    # If result is near zero, our original estimate is likely real
    try:
        placebo = model.refute_estimate(
            identified_estimand,
            estimate,
            method_name="placebo_treatment_refuter",
            placebo_type="permute",
            num_simulations=50,
        )
        placebo_effect = float(np.asarray(placebo.new_effect).item())
        placebo_passed = abs(placebo_effect) < abs(causal_effect) * 0.3
        refutation_results.append({
            "test": "Placebo Treatment",
            "description": "Replace treatment with random noise — effect should vanish",
            "placebo_effect": round(placebo_effect, 4),
            "original_effect": round(causal_effect, 4),
            "passed": placebo_passed,
        })
    except Exception as e:
        refutation_results.append({
            "test": "Placebo Treatment",
            "description": "Replace treatment with random noise",
            "error": str(e),
            "passed": None,
        })
    
    # Refutation 2: Random common cause (add a random confounder)
    # If estimate barely changes, it's robust to unobserved confounders
    try:
        random_cause = model.refute_estimate(
            identified_estimand,
            estimate,
            method_name="random_common_cause",
            num_simulations=50,
        )
        new_effect = float(np.asarray(random_cause.new_effect).item())
        effect_change = abs(new_effect - causal_effect)
        random_passed = effect_change < abs(causal_effect) * 0.15
        refutation_results.append({
            "test": "Random Common Cause",
            "description": "Add a random confounder — effect should barely change",
            "new_effect": round(new_effect, 4),
            "original_effect": round(causal_effect, 4),
            "effect_change": round(effect_change, 4),
            "passed": random_passed,
        })
    except Exception as e:
        refutation_results.append({
            "test": "Random Common Cause",
            "description": "Add a random confounder",
            "error": str(e),
            "passed": None,
        })
    
    # Build interpretation
    direction = "increases" if causal_effect > 0 else "decreases"
    abs_effect = abs(causal_effect)
    tests_passed = sum(1 for r in refutation_results if r.get("passed") is True)
    tests_total = sum(1 for r in refutation_results if r.get("passed") is not None)
    
    interpretation = (
        f"A 1-unit increase in {treatment} {direction} {outcome} "
        f"by {abs_effect:.2f} units. "
        f"Robustness: {tests_passed}/{tests_total} refutation tests passed."
    )
    
    if p_value is not None and p_value < 0.05:
        interpretation += " This effect is statistically significant (p < 0.05)."
    elif p_value is not None:
        interpretation += f" This effect is NOT statistically significant (p = {p_value:.3f})."
    
    return {
        "treatment": treatment,
        "outcome": outcome,
        "estimand": str(identified_estimand),
        "causal_effect": round(causal_effect, 4),
        "p_value": round(p_value, 4) if p_value is not None else None,
        "confidence_interval": (
            round(ci_low, 4) if ci_low is not None else None,
            round(ci_high, 4) if ci_high is not None else None,
        ),
        "refutation_results": refutation_results,
        "interpretation": interpretation,
    }


def get_all_direct_effects(df: pd.DataFrame) -> list[dict]:
    """
    Estimate all direct causal effects in the DAG.
    Returns a list of results, one per edge in the graph.
    """
    edges = [
        ("inflation_rate", "ecb_rate"),
        ("ecb_rate", "bond_price_index"),
        ("ecb_rate", "equity_returns"),
        ("ecb_rate", "credit_spread"),
        ("credit_spread", "client_risk_score"),
        ("unemployment_rate", "client_risk_score"),
        ("bond_price_index", "fund_outflows"),
        ("client_risk_score", "fund_outflows"),
        ("equity_returns", "fund_inflows"),
    ]
    
    results = []
    for treatment, outcome in edges:
        try:
            result = run_causal_analysis(df, treatment, outcome)
            results.append(result)
        except Exception as e:
            results.append({
                "treatment": treatment,
                "outcome": outcome,
                "error": str(e),
            })
    
    return results


# --- TEST ---
if __name__ == "__main__":
    # Load the generated data
    df = pd.read_csv("data/financial_data.csv")
    
    print("=" * 60)
    print("CAUSAL INFERENCE ENGINE — TEST RUN")
    print("=" * 60)
    
    # Test a single causal analysis: ECB rate → fund outflows
    print("\n--- Testing: ecb_rate → fund_outflows ---")
    result = run_causal_analysis(df, "ecb_rate", "fund_outflows")
    
    print(f"Causal effect: {result['causal_effect']}")
    print(f"P-value: {result['p_value']}")
    print(f"Confidence interval: {result['confidence_interval']}")
    print(f"Interpretation: {result['interpretation']}")
    print(f"\nRefutation results:")
    for r in result["refutation_results"]:
        status = "✓ PASSED" if r.get("passed") else "✗ FAILED" if r.get("passed") is False else "? ERROR"
        print(f"  {status} — {r['test']}: {r.get('description', '')}")
    
    # Test a direct effect: ECB rate → bond price
    print("\n--- Testing: ecb_rate → bond_price_index ---")
    result2 = run_causal_analysis(df, "ecb_rate", "bond_price_index")
    print(f"Causal effect: {result2['causal_effect']}")
    print(f"True effect was: -5.0")
    print(f"Interpretation: {result2['interpretation']}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

