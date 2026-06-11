"""
retriever.py

Hybrid retrieval combining FAISS (semantic) and BM25 (keyword)
using Reciprocal Rank Fusion (RRF).

Why hybrid?
- FAISS finds meaning-based matches
- BM25 finds exact keyword matches
- RRF combines rankings without needing to normalize scores
- A chunk ranking well in BOTH systems rises to the top
"""

from src.rag.faiss_index import search_faiss
from src.rag.bm25_index import search_bm25
from src.rag.embedder import embed_text


# ── RRF constant ──────────────────────────────────────
RRF_K = 60   # standard value from research paper
TOP_K  = 3   # final number of results to return


# ── Reciprocal Rank Fusion ────────────────────────────
def reciprocal_rank_fusion(faiss_results, bm25_results, top_k=TOP_K):
    """
    Combines FAISS and BM25 results using RRF.

    Formula: score = 1 / (rank + 60)
    Higher combined score = more relevant chunk.
    """
    scores = {}  # chunk_id -> combined RRF score
    chunks = {}  # chunk_id -> chunk data

    # Score FAISS results
    for result in faiss_results:
        chunk_id = result["chunk_id"]
        rank     = result["rank"]
        rrf_score = 1 / (rank + RRF_K)

        scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score
        chunks[chunk_id] = result

    # Score BM25 results and ADD to existing scores
    for result in bm25_results:
        chunk_id = result["chunk_id"]
        rank     = result["rank"]
        rrf_score = 1 / (rank + RRF_K)

        scores[chunk_id] = scores.get(chunk_id, 0) + rrf_score
        chunks[chunk_id] = result

    # Sort by combined RRF score (highest first)
    sorted_ids = sorted(
        scores.keys(),
        key=lambda cid: scores[cid],
        reverse=True
    )[:top_k]

    # Build final results
    results = []
    for rank, chunk_id in enumerate(sorted_ids):
        chunk = chunks[chunk_id]
        results.append({
            "chunk_id"  : chunk_id,
            "source"    : chunk["source"],
            "text"      : chunk["text"],
            "rrf_score" : round(scores[chunk_id], 6),
            "rank"      : rank + 1,
        })

    return results


# ── Main retrieval function ───────────────────────────
def retrieve(query, faiss_index, bm25_index, embedded_chunks, top_k=TOP_K):
    """
    Full hybrid retrieval pipeline.

    1. Embed the query
    2. Search FAISS (semantic)
    3. Search BM25 (keyword)
    4. Combine with RRF
    5. Return top_k chunks
    """
    # Step 1 — embed the query
    query_embedding = embed_text(query)

    # Step 2 — semantic search
    faiss_results = search_faiss(
        faiss_index,
        embedded_chunks,
        query_embedding,
        top_k=top_k
    )

    # Step 3 — keyword search
    bm25_results = search_bm25(
        bm25_index,
        embedded_chunks,
        query,
        top_k=top_k
    )

    # Step 4 — combine with RRF
    final_results = reciprocal_rank_fusion(
        faiss_results,
        bm25_results,
        top_k=top_k
    )

    return final_results


# ── Main ──────────────────────────────────────────────
if __name__ == "__main__":
    print("This module is imported by other files.")
    print("Test it in Colab after building the full pipeline.")