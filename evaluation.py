"""
evaluate.py - Golden-set evaluation harness for the Financial Causal Intelligence Agent.
 
Validates the system against KNOWN ground truth: the DAG and the effect sizes
used to GENERATE the synthetic data. Two parts:
 
  PART A - Estimation accuracy:
      For every true causal edge, run the DoWhy pipeline and compare the estimated
      effect to the true structural coefficient. Reports absolute/relative error,
      whether it is within tolerance, and whether it passed its refutation tests.
 
  PART B - Discovery accuracy:
      Run the PC algorithm and score the recovered graph against the true DAG
      (precision / recall / F1).
 
Run:  python evaluate.py
Prints a scorecard and saves eval_results/scorecard.json for regression tracking.
This file uses only DoWhy + causal-learn (no OpenAI key needed).
"""
 
import os
import json
from datetime import datetime
 
import pandas as pd
 
from data.generate_data import (
    generate_financial_data,
    get_true_causal_graph,
    get_true_causal_effects,
)
from tools.causal_engine import run_causal_analysis
from tools.causal_discovery import run_pc_algorithm, compare_discovered_vs_true
 
# How close an estimate must be to the truth (relative error) to count as "accurate".
TOLERANCE = 0.30  # 30%
 
DATA_PATH = "data/financial_data.csv"
 
 
def _is_trustworthy(refutation_results: list) -> bool:
    """An estimate is trustworthy only if every refutation that actually ran passed.
    Works whether or not the deterministic-gate patch is applied."""
    ran = [r for r in refutation_results if r.get("passed") is not None]
    return len(ran) > 0 and all(r["passed"] for r in ran)
 
 
def evaluate_estimation(df: pd.DataFrame) -> dict:
    """PART A: compare each estimated direct effect to the known true effect."""
    true_effects = get_true_causal_effects()
    rows = []
 
    for (treatment, outcome), true_effect in true_effects.items():
        try:
            result = run_causal_analysis(df, treatment, outcome)
            est = result["causal_effect"]
            abs_err = abs(est - true_effect)
            rel_err = abs_err / abs(true_effect) if true_effect != 0 else float("inf")
            rows.append({
                "treatment": treatment,
                "outcome": outcome,
                "true_effect": true_effect,
                "estimated_effect": est,
                "abs_error": round(abs_err, 3),
                "rel_error": round(rel_err, 3),
                "within_tolerance": rel_err <= TOLERANCE,
                "refutation_trustworthy": _is_trustworthy(result["refutation_results"]),
            })
        except Exception as e:
            rows.append({
                "treatment": treatment,
                "outcome": outcome,
                "true_effect": true_effect,
                "error": str(e),
            })
 
    valid = [r for r in rows if "estimated_effect" in r]
    n = len(valid)
    summary = {
        "edges_evaluated": n,
        "mean_abs_error": round(sum(r["abs_error"] for r in valid) / n, 3) if n else None,
        "pct_within_tolerance": round(100 * sum(r["within_tolerance"] for r in valid) / n, 1) if n else None,
        "pct_trustworthy": round(100 * sum(r["refutation_trustworthy"] for r in valid) / n, 1) if n else None,
        "tolerance": TOLERANCE,
    }
    return {"rows": rows, "summary": summary}
 
 
def evaluate_discovery(df: pd.DataFrame) -> dict:
    """PART B: score PC-discovered structure against the true DAG."""
    pc_result = run_pc_algorithm(df, alpha=0.05)
    comparison = compare_discovered_vs_true(pc_result["edges"], get_true_causal_graph())
    return {
        "discovered_edges": pc_result["edges"],
        "undirected_edges": pc_result.get("undirected_edges", []),
        "precision": comparison["precision"],
        "recall": comparison["recall"],
        "f1_score": comparison["f1_score"],
        "summary": comparison["summary"],
    }
 
 
def print_scorecard(estimation: dict, discovery: dict) -> None:
    print("=" * 72)
    print("FINANCIAL CAUSAL AGENT - EVALUATION SCORECARD")
    print("=" * 72)
 
    print("\nPART A - ESTIMATION ACCURACY (estimate vs. known true effect)")
    print("-" * 72)
    print(f"{'treatment -> outcome':<44}{'true':>7}{'est':>8}{'ok':>4}{'trust':>7}")
    for r in estimation["rows"]:
        edge = f"{r['treatment']} -> {r['outcome']}"
        if "estimated_effect" in r:
            ok = "Y" if r["within_tolerance"] else "n"
            tr = "Y" if r["refutation_trustworthy"] else "n"
            print(f"{edge:<44}{r['true_effect']:>7.1f}{r['estimated_effect']:>8.2f}{ok:>4}{tr:>7}")
        else:
            print(f"{edge:<44}{'ERROR':>7}  ({r.get('error', '')[:30]})")
    s = estimation["summary"]
    print("-" * 72)
    print(f"Edges evaluated:        {s['edges_evaluated']}")
    print(f"Mean absolute error:    {s['mean_abs_error']}")
    print(f"Within {int(s['tolerance']*100)}% tolerance:    {s['pct_within_tolerance']}%")
    print(f"Passed refutation:      {s['pct_trustworthy']}%")
 
    print("\nPART B - DISCOVERY ACCURACY (PC algorithm vs. true DAG)")
    print("-" * 72)
    print(discovery["summary"])
    print(f"Precision: {discovery['precision']}   Recall: {discovery['recall']}   F1: {discovery['f1_score']}")
    print("Note: PC returns a CPDAG; undirected edges count as misses, so recall is conservative.")
    print("=" * 72)
 
 
def main():
    if not os.path.exists(DATA_PATH):
        print("Data file not found - generating it...")
        os.makedirs("data", exist_ok=True)
        df = generate_financial_data(n_months=120)
        df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH)
 
    estimation = evaluate_estimation(df)
    discovery = evaluate_discovery(df)
    print_scorecard(estimation, discovery)
 
    os.makedirs("eval_results", exist_ok=True)
    scorecard = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "estimation": estimation,
        "discovery": {k: v for k, v in discovery.items() if k != "summary"},
    }
    with open("eval_results/scorecard.json", "w") as f:
        json.dump(scorecard, f, indent=2)
    print("\nSaved -> eval_results/scorecard.json")
 
 
if __name__ == "__main__":
    main()