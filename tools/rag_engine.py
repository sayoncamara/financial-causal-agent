"""
RAG engine for retrieving relevant financial policy documents.

This provides context to the LLM so it can ground its causal
explanations in actual policy language. When the agent discovers
that ECB rate hikes caused fund outflows, the RAG retrieves
relevant ECB monetary policy text to enrich the explanation.
"""

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def build_vector_store(policy_dir: str = "policies") -> FAISS:
    """
    Load all policy documents from the policies/ folder,
    split them into chunks, embed them, and store in FAISS.
    
    Returns a FAISS vector store ready for retrieval.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    
    # Load all .txt files from the policies directory
    documents = []
    for filename in os.listdir(policy_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(policy_dir, filename)
            loader = TextLoader(filepath, encoding="utf-8")
            documents.extend(loader.load())
    
    if not documents:
        raise FileNotFoundError(f"No .txt files found in {policy_dir}/")
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(documents)
    
    # Embed and store
    embeddings = OpenAIEmbeddings(api_key=api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    return vectorstore


def search_policies(
    vectorstore: FAISS,
    query: str,
    k: int = 3,
) -> dict:
    """
    Search the policy vector store for relevant documents.
    
    Parameters:
        vectorstore: FAISS vector store built from policy documents
        query: the search query (e.g., "ECB rate hike impact on funds")
        k: number of results to return
    
    Returns:
        dict with:
            - results: list of {content, source, relevance_score}
            - summary: concatenated text for LLM context
    """
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    formatted_results = []
    for doc, score in results:
        formatted_results.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "relevance_score": round(float(score), 4),
        })
    
    # Build a summary string for the LLM
    summary = "\n\n---\n\n".join(
        f"[Source: {r['source']}]\n{r['content']}"
        for r in formatted_results
    )
    
    return {
        "results": formatted_results,
        "summary": summary,
        "num_results": len(formatted_results),
    }


# --- TEST ---
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("Building vector store from policies/...")
    vs = build_vector_store()
    print(f"Vector store built successfully")
    
    # Test queries
    test_queries = [
        "ECB interest rate impact on bond funds",
        "debt-to-income ratio risk assessment",
        "fund redemption risk during rate hikes",
    ]
    
    for query in test_queries:
        print(f"\n--- Query: '{query}' ---")
        result = search_policies(vs, query, k=2)
        for r in result["results"]:
            print(f"  [{r['source']}] (score: {r['relevance_score']})")
            print(f"  {r['content'][:100]}...")
