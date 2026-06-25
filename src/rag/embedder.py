"""
embedder.py

Converts text chunks into embeddings using sentence-transformers.

Why sentence-transformers?
- Free, no API key needed
- Runs locally, no data leaves environment
- Production alternative: OpenAI text-embedding-3-small
  or Azure OpenAI embeddings (just swap the embed_text function)
"""

from sentence_transformers import SentenceTransformer

# Load model once at module level
# Why? Loading is slow - we only want to do it once
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# 384 dimensions
# OpenAI text-embedding-3-small = 1536 dimensions (more precise)
EMBEDDING_DIMENSIONS = 384


def embed_text(text):
    """
    Convert one piece of text into a vector of numbers.

    Example:
    "what is a deductible?"
    → [-0.037, 0.037, 0.056, ...] (384 numbers)
    """
    vector = model.encode(text)
    return vector.tolist()


def embed_chunks(chunks):
    """
    Takes a list of chunks and adds an embedding to each one.

    Input:
    [{"chunk_id": "auto_policy_1_chunk_0", "text": "...", ...}]

    Output:
    [{"chunk_id": "auto_policy_1_chunk_0", "text": "...",
      "embedding": [...384 numbers...]}]
    """
    embedded = []

    for i, chunk in enumerate(chunks):
        print(f"  Embedding {i+1}/{len(chunks)}: {chunk['chunk_id']}")
        embedded.append({
            **chunk,
            "embedding": embed_text(chunk["text"])
        })

    print(f"\nEmbedded {len(embedded)} chunks")
    return embedded


if __name__ == "__main__":
    # Quick test
    vector = embed_text("what is a deductible?")
    print(f"Dimensions: {len(vector)}")
    print(f"First 5 numbers: {vector[:5]}")