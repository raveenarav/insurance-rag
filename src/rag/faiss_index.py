"""
faiss_index.py

Stores chunk embeddings in a FAISS index for semantic search.

Why FAISS?
- Fast similarity search across thousands of vectors
- Finds meaning-based matches even with different words
- Built by Meta, industry standard for vector search
"""

import numpy as np
import faiss
from pathlib import Path

# ── Settings ──────────────────────────────────────────
EMBEDDING_DIMENSIONS = 384   # must match embedder output
TOP_K = 3                    # how many results to return


# ── Build the index ───────────────────────────────────
def build_faiss_index(embedded_chunks):
    """
    Takes embedded chunks and builds a FAISS index.

    Why IndexFlatL2?
    - L2 = Euclidean distance (straight line between vectors)
    - Flat = checks every vector (exact, not approximate)
    - Good for small datasets like ours
    - Production alternative: IndexIVFFlat (faster for millions of vectors)
    """
    # Extract just the embeddings as a numpy array
    embeddings = np.array(
        [chunk["embedding"] for chunk in embedded_chunks],
        dtype=np.float32   # FAISS requires float32
    )

    # Create the index
    index = faiss.IndexFlatL2(EMBEDDING_DIMENSIONS)

    # Add all embeddings to the index
    index.add(embeddings)

    print(f"FAISS index built with {index.ntotal} vectors")
    return index


# ── Search the index ──────────────────────────────────
def search_faiss(index, embedded_chunks, query_embedding, top_k=TOP_K):
    """
    Search FAISS for the most similar chunks to a query.

    Returns the top_k most similar chunks.
    """
    # Convert query to numpy array
    query_vector = np.array(
        [query_embedding],
        dtype=np.float32
    )

    # Search — returns distances and indexes of closest vectors
    distances, indices = index.search(query_vector, top_k)

    # Build results
    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:   # FAISS returns -1 if not enough results
            continue
        chunk = embedded_chunks[idx]
        results.append({
            "chunk_id" : chunk["chunk_id"],
            "source"   : chunk["source"],
            "text"     : chunk["text"],
            "score"    : float(distances[0][i]),  # lower = more similar
            "rank"     : i + 1,
        })

    return results


# ── Main ──────────────────────────────────────────────
if __name__ == "__main__":
    print("This module is imported by other files.")
    print("Test it in Colab after building the full pipeline.")