"""
Causal discovery: automatically discover causal structure from data.

Two complementary methods:
1. PC Algorithm (causal-learn) — discovers causal graph from 
   conditional independence tests. Best for cross-sectional relationships.
2. Transfer Entropy — measures directed information flow between 
   time series. Best for temporal causality (does X predict Y?).

The agent chooses the right method based on the question:
- "What causes what in general?" → PC algorithm
- "Does ECB rate information flow into fund outflows over time?" → Transfer entropy
"""

import pandas as pd
import numpy as np
import warnings
from typing import Optional

warnings.filterwarnings("ignore")


def run_pc_algorithm(
    df: pd.DataFrame,
    variables: Optional[list[str]] = None,
    alpha: float = 0.05,
) -> dict:
    """
    Run the PC algorithm to discover causal structure from data.
    
    The PC algorithm works by:
    1. Start with a fully connected graph
    2. Remove edges where variables are conditionally independent
    3. Orient remaining edges using v-structures and rules
    
    Parameters:
        df: DataFrame with financial data
        variables: list of column names to include (None = all numeric)
        alpha: significance level for independence tests (lower = fewer edges)
    
    Returns:
        dict with:
            - edges: list of (cause, effect) tuples discovered
            - adjacency_matrix: numpy array
            - variables: list of variable names
            - summary: human-readable summary
    """
    from causallearn.search.ConstraintBased.PC import pc
    
    # Prepare data
    data = df.drop(columns=["date"], errors="ignore")
    if variables:
        data = data[variables]
    
    variable_names = list(data.columns)
    data_array = data.values.astype(float)
    
    # Run PC algorithm
    cg = pc(
        data_array,
        alpha=alpha,
        indep_test="fisherz",
        stable=True,
        uc_rule=0,
        uc_priority=2,
        show_progress=False,
    )
    
    # Extract adjacency matrix
    # cg.G.graph encodes: 
    #   -1 means tail (cause), 1 means arrowhead (effect), 0 means no edge
    adj_matrix = cg.G.graph
    
    # Extract directed edges
    edges = []
    undirected_edges = []
    n = len(variable_names)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # Directed edge i → j: adj[i,j] == -1 and adj[j,i] == 1
                if adj_matrix[i, j] == -1 and adj_matrix[j, i] == 1:
                    edges.append((variable_names[i], variable_names[j]))
                # Undirected edge i — j: adj[i,j] == -1 and adj[j,i] == -1
                elif adj_matrix[i, j] == -1 and adj_matrix[j, i] == -1 and i < j:
                    undirected_edges.append((variable_names[i], variable_names[j]))
    
    # Build summary
    summary_lines = [f"PC Algorithm discovered {len(edges)} directed edges and {len(undirected_edges)} undirected edges:"]
    summary_lines.append("\nDirected edges (cause → effect):")
    for cause, effect in edges:
        summary_lines.append(f"  {cause} → {effect}")
    if undirected_edges:
        summary_lines.append("\nUndirected edges (direction unknown):")
        for v1, v2 in undirected_edges:
            summary_lines.append(f"  {v1} — {v2}")
    
    return {
        "method": "PC Algorithm",
        "edges": edges,
        "undirected_edges": undirected_edges,
        "adjacency_matrix": adj_matrix.tolist(),
        "variables": variable_names,
        "alpha": alpha,
        "summary": "\n".join(summary_lines),
    }


def compute_transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    lag: int = 1,
    bins: int = 10,
) -> float:
    """
    Compute transfer entropy from source → target.
    
    Transfer entropy measures how much knowing the past of the source
    reduces uncertainty about the future of the target, beyond what
    the target's own past already tells us.
    
    TE(source → target) > 0 means source has causal influence on target.
    
    Uses a binned histogram estimator for simplicity and speed.
    """
    n = len(source) - lag
    if n <= 0:
        return 0.0
    
    # Create lagged variables
    target_future = target[lag:]        # target(t)
    target_past = target[:n]            # target(t-lag)
    source_past = source[:n]            # source(t-lag)
    
    # Bin the continuous variables
    target_future_binned = np.digitize(target_future, np.linspace(target_future.min(), target_future.max(), bins))
    target_past_binned = np.digitize(target_past, np.linspace(target_past.min(), target_past.max(), bins))
    source_past_binned = np.digitize(source_past, np.linspace(source_past.min(), source_past.max(), bins))
    
    # Compute joint and marginal probabilities using histograms
    # TE = H(target_future | target_past) - H(target_future | target_past, source_past)
    # Which simplifies to: 
    # TE = H(target_future, target_past) + H(target_past, source_past) 
    #    - H(target_future, target_past, source_past) - H(target_past)
    
    def entropy(variables):
        """Compute joint entropy of one or more discrete variables."""
        if isinstance(variables, np.ndarray) and variables.ndim == 1:
            _, counts = np.unique(variables, return_counts=True)
        else:
            # Stack variables and compute joint
            combined = np.column_stack(variables)
            _, counts = np.unique(combined, axis=0, return_counts=True)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))
    
    h_tf_tp = entropy([target_future_binned, target_past_binned])
    h_tp_sp = entropy([target_past_binned, source_past_binned])
    h_tf_tp_sp = entropy([target_future_binned, target_past_binned, source_past_binned])
    h_tp = entropy(target_past_binned)
    
    te = h_tf_tp + h_tp_sp - h_tf_tp_sp - h_tp
    
    return max(te, 0.0)  # TE should be non-negative


def run_transfer_entropy_analysis(
    df: pd.DataFrame,
    variables: Optional[list[str]] = None,
    lag: int = 1,
    n_permutations: int = 100,
    significance_level: float = 0.05,
) -> dict:
    """
    Compute transfer entropy between all pairs of variables
    and test for statistical significance using permutation testing.
    
    Parameters:
        df: DataFrame with financial data
        variables: list of columns to analyze (None = all numeric)
        lag: number of time steps for lagged influence
        n_permutations: number of permutations for significance testing
        significance_level: p-value threshold for significance
    
    Returns:
        dict with:
            - significant_flows: list of (source, target, te, p_value) for significant flows
            - all_flows: full matrix of transfer entropy values
            - summary: human-readable summary
    """
    data = df.drop(columns=["date"], errors="ignore")
    if variables:
        data = data[variables]
    
    variable_names = list(data.columns)
    n_vars = len(variable_names)
    
    # Compute TE for all pairs
    te_matrix = np.zeros((n_vars, n_vars))
    p_value_matrix = np.ones((n_vars, n_vars))
    
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                continue
            
            source = data.iloc[:, i].values
            target = data.iloc[:, j].values
            
            # Compute actual TE
            te_actual = compute_transfer_entropy(source, target, lag=lag)
            te_matrix[i, j] = te_actual
            
            # Permutation test for significance
            te_null = np.zeros(n_permutations)
            for k in range(n_permutations):
                source_shuffled = np.random.permutation(source)
                te_null[k] = compute_transfer_entropy(source_shuffled, target, lag=lag)
            
            # P-value: fraction of null TE values >= actual TE
            p_value_matrix[i, j] = np.mean(te_null >= te_actual)
    
    # Extract significant flows
    significant_flows = []
    for i in range(n_vars):
        for j in range(n_vars):
            if i != j and p_value_matrix[i, j] < significance_level:
                significant_flows.append({
                    "source": variable_names[i],
                    "target": variable_names[j],
                    "transfer_entropy": round(float(te_matrix[i, j]), 4),
                    "p_value": round(float(p_value_matrix[i, j]), 4),
                })
    
    # Sort by transfer entropy (strongest flows first)
    significant_flows.sort(key=lambda x: x["transfer_entropy"], reverse=True)
    
    # Build summary
    summary_lines = [
        f"Transfer Entropy Analysis (lag={lag}, significance={significance_level}):",
        f"Found {len(significant_flows)} significant information flows:",
    ]
    for flow in significant_flows:
        summary_lines.append(
            f"  {flow['source']} → {flow['target']}: "
            f"TE = {flow['transfer_entropy']:.4f} (p = {flow['p_value']:.4f})"
        )
    
    return {
        "method": "Transfer Entropy",
        "lag": lag,
        "significant_flows": significant_flows,
        "te_matrix": te_matrix.tolist(),
        "p_value_matrix": p_value_matrix.tolist(),
        "variables": variable_names,
        "summary": "\n".join(summary_lines),
    }


def compare_discovered_vs_true(discovered_edges: list[tuple], true_graph: dict) -> dict:
    """
    Compare discovered edges against the true causal graph.
    
    Returns precision, recall, and F1 score.
    """
    # Build true edge set
    true_edges = set()
    for parent, children in true_graph.items():
        for child in children:
            true_edges.add((parent, child))
    
    discovered_set = set(discovered_edges)
    
    true_positives = discovered_set & true_edges
    false_positives = discovered_set - true_edges
    false_negatives = true_edges - discovered_set
    
    precision = len(true_positives) / len(discovered_set) if discovered_set else 0
    recall = len(true_positives) / len(true_edges) if true_edges else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "true_positives": [list(e) for e in true_positives],
        "false_positives": [list(e) for e in false_positives],
        "false_negatives": [list(e) for e in false_negatives],
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3),
        "summary": (
            f"Precision: {precision:.1%} | Recall: {recall:.1%} | F1: {f1:.1%}\n"
            f"Correctly found: {len(true_positives)}/{len(true_edges)} true edges\n"
            f"False discoveries: {len(false_positives)}"
        ),
    }


# --- TEST ---
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.generate_data import get_true_causal_graph
    
    df = pd.read_csv("data/financial_data.csv")
    
    print("=" * 60)
    print("CAUSAL DISCOVERY — TEST RUN")
    print("=" * 60)
    
    # Test 1: PC Algorithm
    print("\n--- PC Algorithm ---")
    pc_result = run_pc_algorithm(df, alpha=0.05)
    print(pc_result["summary"])
    
    # Compare with ground truth
    print("\n--- Comparison with true graph ---")
    true_graph = get_true_causal_graph()
    comparison = compare_discovered_vs_true(pc_result["edges"], true_graph)
    print(comparison["summary"])
    
    # Test 2: Transfer Entropy (on a subset for speed)
    print("\n--- Transfer Entropy (subset of variables) ---")
    te_result = run_transfer_entropy_analysis(
        df,
        variables=["ecb_rate", "bond_price_index", "fund_outflows", "equity_returns"],
        lag=1,
        n_permutations=50,
    )
    print(te_result["summary"])
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
