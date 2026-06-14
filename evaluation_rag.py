"""
evaluate_rag.py - Evaluation harness for the RAG layer of the Financial Causal Agent.
 
Evaluates the policy-document retrieval + grounded generation, in three parts:
 
  PART 1 - RETRIEVAL QUALITY:
      A golden set of questions, each mapped to the policy document that should
      answer it. Measures Hit@k (did the right doc make the top-k) and MRR.
 
  PART 2 - FAITHFULNESS (RAGAS-style, LLM-as-judge):
      For each question, retrieve context, generate an answer grounded ONLY in
      that context, then a judge LLM decomposes the answer into claims and scores
      the fraction supported by the context. This is the hallucination metric.
 
  PART 3 - ABSTENTION (hallucination guard):
      Out-of-scope questions that the policy docs cannot answer. A grounded system
      should REFUSE ("I don't have enough information") rather than fabricate.
 
Run:  python evaluate_rag.py
Needs OPENAI_API_KEY (uses OpenAIEmbeddings for retrieval + ChatOpenAI for
generation and judging). Prints a scorecard and saves eval_results/rag_scorecard.json.
"""
 
import os
import json
import re
from datetime import datetime
 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
 
from tools.rag_engine import build_vector_store, search_policies
 
load_dotenv()
 
K = 3                      # top-k for retrieval (matches the agent's k=3)
JUDGE_MODEL = "gpt-4o-mini"
ABSTAIN_PHRASE = "I don't have enough information"
 
# --- GOLDEN SET: question -> the policy file that should answer it ---
# (built from the actual content of your three policy docs)
GOLDEN_SET = [
    {"q": "How much do eurozone government bonds fall when the ECB raises rates by 100 basis points?",
     "source": "ecb_monetary_policy.txt"},
    {"q": "How large were European UCITS bond fund outflows during the 2022-2023 tightening cycle?",
     "source": "ecb_monetary_policy.txt"},
    {"q": "By how much do investment-grade credit spreads widen per 100bp ECB rate hike?",
     "source": "ecb_monetary_policy.txt"},
    {"q": "What was the peak eurozone HICP inflation during the 2022 shock?",
     "source": "european_macro_indicators.txt"},
    {"q": "How does a 1 percentage point rise in unemployment affect new fund subscriptions?",
     "source": "european_macro_indicators.txt"},
    {"q": "What credit spread level signals a low-risk environment?",
     "source": "european_macro_indicators.txt"},
    {"q": "What liquidity buffer must fund managers maintain to handle redemptions?",
     "source": "fund_risk_management.txt"},
    {"q": "When are fund redemption gates activated?",
     "source": "fund_risk_management.txt"},
    {"q": "What client risk score triggers automatic portfolio de-risking?",
     "source": "fund_risk_management.txt"},
    {"q": "How is client risk appetite measured in the risk framework?",
     "source": "fund_risk_management.txt"},
]
 
# Out-of-scope questions the policy docs cannot answer -> the system should abstain.
OUT_OF_SCOPE = [
    "What is the current price of Bitcoin?",
    "Who is the CEO of Deutsche Bank?",
    "What is the capital of Australia?",
]
 
 
def _basename(path: str) -> str:
    return os.path.basename(path or "")
 
 
def _grounded_answer(llm, question: str, context: str) -> str:
    """Generate an answer constrained to the retrieved context (or abstain)."""
    prompt = (
        "Answer the question using ONLY the context below. "
        f"If the context does not contain the answer, reply exactly: '{ABSTAIN_PHRASE}'.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    return llm.invoke(prompt).content.strip()
 
 
def _faithfulness(llm, answer: str, context: str) -> dict:
    """RAGAS-style: decompose the answer into claims, score fraction supported by context."""
    if ABSTAIN_PHRASE.lower() in answer.lower():
        return {"faithfulness": None, "note": "abstained (excluded from faithfulness)"}
    judge_prompt = (
        "You are a strict evaluator. Break the ANSWER into individual factual claims. "
        "For each claim, decide if it is directly supported by the CONTEXT. "
        "Return ONLY JSON: {\"num_claims\": int, \"num_supported\": int}. No prose.\n\n"
        f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"
    )
    raw = llm.invoke(judge_prompt).content.strip()
    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        data = json.loads(raw)
        n, m = int(data["num_claims"]), int(data["num_supported"])
        score = m / n if n else None
        return {"faithfulness": round(score, 3) if score is not None else None,
                "num_claims": n, "num_supported": m}
    except Exception as e:
        return {"faithfulness": None, "note": f"judge parse error: {e}"}
 
 
def evaluate_retrieval(vs) -> dict:
    hits, reciprocal_ranks, rows = 0, [], []
    for item in GOLDEN_SET:
        res = search_policies(vs, item["q"], k=K)
        sources = [_basename(r["source"]) for r in res["results"]]
        rank = next((i + 1 for i, s in enumerate(sources) if s == item["source"]), None)
        hit = rank is not None
        hits += hit
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        rows.append({"q": item["q"], "expected": item["source"],
                     "retrieved": sources, "rank": rank, "hit": hit})
    n = len(GOLDEN_SET)
    return {"rows": rows,
            "hit_at_k": round(hits / n, 3),
            "mrr": round(sum(reciprocal_ranks) / n, 3),
            "k": K}
 
 
def evaluate_faithfulness(vs, llm) -> dict:
    rows, scores = [], []
    for item in GOLDEN_SET:
        res = search_policies(vs, item["q"], k=K)
        context = res["summary"]
        answer = _grounded_answer(llm, item["q"], context)
        f = _faithfulness(llm, answer, context)
        if f.get("faithfulness") is not None:
            scores.append(f["faithfulness"])
        rows.append({"q": item["q"], "answer": answer[:160], **f})
    return {"rows": rows,
            "mean_faithfulness": round(sum(scores) / len(scores), 3) if scores else None,
            "scored": len(scores)}
 
 
def evaluate_abstention(vs, llm) -> dict:
    rows, abstained = [], 0
    for q in OUT_OF_SCOPE:
        res = search_policies(vs, q, k=K)
        answer = _grounded_answer(llm, q, res["summary"])
        did = ABSTAIN_PHRASE.lower() in answer.lower()
        abstained += did
        rows.append({"q": q, "abstained": did, "answer": answer[:160]})
    return {"rows": rows,
            "abstention_rate": round(abstained / len(OUT_OF_SCOPE), 3)}
 
 
def print_scorecard(retr, faith, abst) -> None:
    print("=" * 72)
    print("RAG LAYER - EVALUATION SCORECARD")
    print("=" * 72)
 
    print("\nPART 1 - RETRIEVAL QUALITY")
    print("-" * 72)
    for r in retr["rows"]:
        mark = "Y" if r["hit"] else "n"
        print(f"[{mark}] rank={str(r['rank']):<4} expected={r['expected']:<28} {r['q'][:38]}")
    print("-" * 72)
    print(f"Hit@{retr['k']}: {retr['hit_at_k']}    MRR: {retr['mrr']}")
 
    print("\nPART 2 - FAITHFULNESS (fraction of answer claims supported by context)")
    print("-" * 72)
    for r in faith["rows"]:
        print(f"  faithfulness={str(r.get('faithfulness')):<6} {r['q'][:50]}")
    print("-" * 72)
    print(f"Mean faithfulness: {faith['mean_faithfulness']}  (over {faith['scored']} answered questions)")
 
    print("\nPART 3 - ABSTENTION ON OUT-OF-SCOPE (hallucination guard)")
    print("-" * 72)
    for r in abst["rows"]:
        mark = "Y abstained" if r["abstained"] else "n ANSWERED (hallucination risk)"
        print(f"  [{mark}] {r['q']}")
    print("-" * 72)
    print(f"Abstention rate: {abst['abstention_rate']}  (higher is better here)")
    print("=" * 72)
 
 
def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set - this harness needs it for embeddings + judging.")
 
    print("Building vector store from policies/ ...")
    vs = build_vector_store("policies")
    llm = ChatOpenAI(model=JUDGE_MODEL, temperature=0)
 
    retr = evaluate_retrieval(vs)
    faith = evaluate_faithfulness(vs, llm)
    abst = evaluate_abstention(vs, llm)
    print_scorecard(retr, faith, abst)
 
    os.makedirs("eval_results", exist_ok=True)
    with open("eval_results/rag_scorecard.json", "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(timespec="seconds"),
                   "retrieval": retr, "faithfulness": faith, "abstention": abst},
                  f, indent=2)
    print("\nSaved -> eval_results/rag_scorecard.json")
 
 
if __name__ == "__main__":
    main()