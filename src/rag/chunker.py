"""
chunker.py

Splits documents into smaller overlapping chunks.

Why chunking?
- LLMs have token limits - can't read whole documents
- Smaller chunks = more precise retrieval
- Overlap = no answer gets cut at a boundary
"""

from pathlib import Path

# ── Settings ──────────────────────────────────────────
CHUNK_SIZE = 50     # words per chunk (small for our synthetic docs)
OVERLAP    = 10     # words shared between chunks
RAW_DIR    = Path(__file__).resolve().parents[2] / "data" / "raw"
OUT_DIR    = Path(__file__).resolve().parents[2] / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Step 1: Read a document ───────────────────────────
def read_document(filepath):
    """Read a text file and return its contents as a string."""
    return Path(filepath).read_text(encoding="utf-8")


# ── Step 2: Split into words ──────────────────────────
def split_into_words(text):
    """Split text into individual words."""
    return text.split()


# ── Step 3: Create chunks ─────────────────────────────
def create_chunks(words, chunk_size, overlap):
    """
    Slide a window over the word list to create chunks.

    Example with chunk_size=5, overlap=2:
    words = [A, B, C, D, E, F, G, H]

    Chunk 1: [A, B, C, D, E]
    Chunk 2: [D, E, F, G, H]  <- D,E repeated from chunk 1
    """
    chunks = []
    start  = 0

    while start < len(words):
        end   = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start = start + chunk_size - overlap

    return chunks


# ── Step 4: Chunk one file ────────────────────────────
def chunk_file(filepath, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Read a file and return its chunks with metadata."""
    text     = read_document(filepath)
    words    = split_into_words(text)
    chunks   = create_chunks(words, chunk_size, overlap)

    result = []
    for i, chunk_text in enumerate(chunks):
        result.append({
            "chunk_id"   : f"{Path(filepath).stem}_chunk_{i}",
            "source"     : Path(filepath).name,
            "chunk_index": i,
            "text"       : chunk_text,
            "word_count" : len(chunk_text.split()),
        })

    return result


# ── Step 5: Chunk ALL files ───────────────────────────
def chunk_all_documents():
    """Chunk every .txt file in the raw folder."""
    all_chunks = []

    txt_files = list(RAW_DIR.glob("*.txt"))

    if not txt_files:
        print("No documents found. Run generate_docs.py first.")
        return []

    for filepath in txt_files:
        chunks = chunk_file(filepath)
        all_chunks.extend(chunks)
        print(f"  {filepath.name} -> {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")
    return all_chunks


# ── Main ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Chunking documents...\n")
    chunks = chunk_all_documents()

    if chunks:
        print("\n── Sample chunk ──")
        sample = chunks[5]
        print(f"ID     : {sample['chunk_id']}")
        print(f"Source : {sample['source']}")
        print(f"Words  : {sample['word_count']}")
        print(f"Text   : {sample['text'][:200]}...")