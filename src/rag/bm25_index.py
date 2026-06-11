"""
bm25_index.py

Keyword search using BM25 algorithm.

Why BM25?
- Finds exact keyword matches
- Great for policy numbers, jargon, section references
- Complements FAISS which finds meaning-based matches
- Industry standard since 1990s (used by Elasticsearch)
"""

from rank_bm25 import BM25Okapi


# ── Build the index ───────────────────────────────────
def build_bm25_index(chunks):
    """
    Takes chunks and builds a BM25 index.

    BM25 works on words so we tokenize each chunk
    by splitting on spaces.

    Why BM25Okapi?
    - Okapi is the most common BM25 variant
    - Used by Elasticsearch under the hood
    - Better than basic BM25 for longer documents
    """
    # Tokenize — split each chunk into list of words
    tokenized = [chunk["text"].lower().split() for chunk in chunks]

    # Build index
    index = BM25Okapi(tokenized)

    print(f"BM25 index built with {len(chunks)} chunks")
    return index


# ── Search the index ──────────────────────────────────
def search_bm25(index, chunks, query, top_k=3):
    """
    Search BM25 for chunks matching the query keywords.

    Higher score = better keyword match.
    Opposite of FAISS where lower score = better match.
    """
    # Tokenize query the same way we tokenized chunks
    tokenized_query = query.lower().split()

    # Get scores for all chunks
    scores = index.get_scores(tokenized_query)

    # Get top_k chunk indexes sorted by score (highest first)
    top_indexes = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    # Build results
    results = []
    for rank, idx in enumerate(top_indexes):
        results.append({
            "chunk_id" : chunks[idx]["chunk_id"],
            "source"   : chunks[idx]["source"],
            "text"     : chunks[idx]["text"],
            "score"    : float(scores[idx]),  # higher = more relevant
            "rank"     : rank + 1,
        })

    return results


# ── Main ──────────────────────────────────────────────
if __name__ == "__main__":
    print("This module is imported by other files.")
    print("Test it in Colab after building the full pipeline.")