"""
api.py

FastAPI web service wrapping the RAG pipeline.

Why FastAPI?
- Fast, modern Python web framework
- Auto-generates interactive docs at /docs
- Type hints = automatic request validation
- Industry standard for ML APIs

Endpoints:
  GET  /health  → check if service is running
  POST /ask     → ask an insurance question
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import re
import json
import random
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from openai import AzureOpenAI


# ── Request/Response models ───────────────────────────
class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer  : str
    sources : list
    blocked : bool = False
    reason  : str  = ""


# ── Global state ──────────────────────────────────────
state = {
    "embedding_model": None,
    "faiss_index"    : None,
    "bm25_index"     : None,
    "all_chunks"     : [],
    "all_embeddings" : [],
}


# ── Insurance keywords & patterns ─────────────────────
INSURANCE_KEYWORDS = [
    "insurance", "policy", "claim", "deductible", "premium",
    "coverage", "liability", "flood", "collision", "theft",
    "accident", "damage", "fire", "cancel", "payment",
    "auto", "home", "life", "health", "exclusion"
]

PII_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",
    r"\b\d{16}\b",
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
]

INJECTION_PATTERNS = [
    "ignore all instructions",
    "ignore previous instructions",
    "you are now",
    "forget everything",
    "act as",
    "jailbreak",
]


# ── Document generation ───────────────────────────────
def generate_synthetic_docs(raw_dir):
    """Generate synthetic insurance docs if folder is empty."""
    random.seed(42)

    AUTO = """POLICY DOCUMENT - AUTO INSURANCE
Policy Number: {policy_no}
Policyholder: {holder}
Effective Date: 2024-01-15

SECTION 1 - COVERAGE
Comprehensive auto insurance including liability, collision, theft, fire.

SECTION 2 - PREMIUMS AND DEDUCTIBLES
Annual Premium: ${premium}
Collision Deductible: ${collision_ded}
A deductible is the amount you pay before insurance covers the rest.

SECTION 3 - CLAIMS PROCEDURE
To file a claim:
1. Report within 72 hours by calling 1-800-CLAIMS.
2. Provide policy number and incident description.
3. Submit police report for theft or injury.
Claims processed within 14 business days.

SECTION 4 - EXCLUSIONS
Does NOT cover: DUI, commercial use, wear and tear."""

    HOME = """POLICY DOCUMENT - HOME INSURANCE
Policy Number: {policy_no}
Policyholder: {holder}
Effective Date: 2024-03-01

SECTION 1 - DWELLING COVERAGE
Covers loss up to ${dwelling_limit}.
Perils: fire, lightning, windstorm, hail, vandalism.
Flood and earthquake require separate policies.

SECTION 2 - PERSONAL PROPERTY
Covered up to ${personal_limit}. Deductible: ${deductible}

SECTION 3 - LIABILITY
Personal liability coverage of ${liability_limit}.

SECTION 4 - CLAIMS
Claims must be reported within 30 days of discovery."""

    FAQS = [
        ("How long does it take to process an auto claim?",
         "Auto claims are processed within 14 business days after all documents are received."),
        ("What is a deductible?",
         "A deductible is the amount you pay out of pocket before insurance covers the rest."),
        ("Is flood damage covered under homeowners policy?",
         "No. Standard homeowners insurance excludes flood damage. You need a separate policy."),
        ("Can I cancel my policy at any time?",
         "Yes. Most policies allow cancellation with 30 days written notice."),
        ("What happens if I miss a premium payment?",
         "Most policies provide a 30 day grace period before the policy lapses."),
    ]

    NAMES     = ["Alex Patel", "Jordan Garcia", "Sam Smith", "Taylor Nguyen"]
    ADDRESSES = ["123 Maple St", "456 Oak Ave", "789 Pine Rd"]

    def rand_policy_no(prefix):
        return f"{prefix}-{random.randint(100000, 999999)}"

    for i in range(3):
        # Auto policy
        content = AUTO.format(
            policy_no   = rand_policy_no("AUTO"),
            holder      = random.choice(NAMES),
            premium     = random.choice([850, 1200, 1450]),
            collision_ded = random.choice([250, 500, 1000]),
        )
        (raw_dir / f"auto_policy_{i+1}.txt").write_text(content)

        # Home policy
        content = HOME.format(
            policy_no     = rand_policy_no("HOME"),
            holder        = random.choice(NAMES),
            dwelling_limit = random.choice([250000, 400000]),
            personal_limit = random.choice([50000, 100000]),
            deductible    = random.choice([500, 1000]),
            liability_limit = random.choice([100000, 300000]),
        )
        (raw_dir / f"home_policy_{i+1}.txt").write_text(content)

    # FAQ
    faq_content = "FREQUENTLY ASKED QUESTIONS\n\n"
    for q, a in FAQS:
        faq_content += f"Q: {q}\nA: {a}\n\n"
    (raw_dir / "faqs.txt").write_text(faq_content)

    print(f"Generated {6 + 1} synthetic documents")


# ── Lifespan ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on startup and shutdown.
    Loads models and builds indexes once.
    """
    print("Starting up...")

    # Step 1 — generate docs if needed
    RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not list(RAW_DIR.glob("*.txt")):
        generate_synthetic_docs(RAW_DIR)

    # Step 2 — load embedding model
    print("Loading embedding model...")
    state["embedding_model"] = SentenceTransformer('all-MiniLM-L6-v2')

    # Step 3 — chunk documents
    print("Chunking documents...")
    CHUNK_SIZE = 50
    OVERLAP    = 10
    all_chunks = []

    for filepath in RAW_DIR.glob("*.txt"):
        text  = filepath.read_text(encoding="utf-8")
        words = text.split()
        start = 0
        i     = 0
        while start < len(words):
            end        = start + CHUNK_SIZE
            chunk_text = " ".join(words[start:end])
            all_chunks.append({
                "chunk_id": f"{filepath.stem}_chunk_{i}",
                "source"  : filepath.name,
                "text"    : chunk_text,
            })
            start = start + CHUNK_SIZE - OVERLAP
            i    += 1

    # Step 4 — embed chunks
    print("Embedding chunks...")
    all_embeddings = []
    for chunk in all_chunks:
        vector = state["embedding_model"].encode(chunk["text"])
        all_embeddings.append({**chunk, "embedding": vector.tolist()})

    # Step 5 — build FAISS
    embeddings_matrix = np.array(
        [c["embedding"] for c in all_embeddings],
        dtype=np.float32
    )
    faiss_idx = faiss.IndexFlatL2(384)
    faiss_idx.add(embeddings_matrix)

    # Step 6 — build BM25
    tokenized = [c["text"].lower().split() for c in all_chunks]
    bm25_idx  = BM25Okapi(tokenized)

    # Store everything
    state["faiss_index"]    = faiss_idx
    state["bm25_index"]     = bm25_idx
    state["all_chunks"]     = all_chunks
    state["all_embeddings"] = all_embeddings

    print(f"Ready. {len(all_chunks)} chunks indexed.")

    yield

    print("Shutting down.")


# ── App ───────────────────────────────────────────────
app = FastAPI(
    title       = "Insurance RAG API",
    description = "Answers insurance questions using hybrid retrieval and GPT-4.1",
    version     = "1.0.0",
    lifespan    = lifespan
)


# ── Helper functions ──────────────────────────────────
def check_input(query):
    query_lower = query.lower()
    for pattern in PII_PATTERNS:
        if re.search(pattern, query):
            return False, "Query contains personal data."
    for phrase in INJECTION_PATTERNS:
        if phrase in query_lower:
            return False, "Query contains invalid instructions."
    if not any(kw in query_lower for kw in INSURANCE_KEYWORDS):
        return False, "Only insurance questions are supported."
    return True, None


def hybrid_retrieve(query, top_k=3):
    embedding_model = state["embedding_model"]
    faiss_index     = state["faiss_index"]
    bm25_index      = state["bm25_index"]
    all_chunks      = state["all_chunks"]
    all_embeddings  = state["all_embeddings"]

    # FAISS search
    query_embedding    = embedding_model.encode(query).tolist()
    query_vector       = np.array([query_embedding], dtype=np.float32)
    distances, indices = faiss_index.search(query_vector, top_k)

    faiss_results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        faiss_results.append({
            "chunk_id": all_embeddings[idx]["chunk_id"],
            "source"  : all_embeddings[idx]["source"],
            "text"    : all_embeddings[idx]["text"],
            "rank"    : i + 1,
        })

    # BM25 search
    tokenized_query = query.lower().split()
    scores          = bm25_index.get_scores(tokenized_query)
    top_indexes     = sorted(range(len(scores)),
                             key=lambda i: scores[i],
                             reverse=True)[:top_k]
    bm25_results = []
    for rank, idx in enumerate(top_indexes):
        bm25_results.append({
            "chunk_id": all_chunks[idx]["chunk_id"],
            "source"  : all_chunks[idx]["source"],
            "text"    : all_chunks[idx]["text"],
            "rank"    : rank + 1,
        })

    # RRF fusion
    scores_rrf = {}
    chunks_map = {}
    for result in faiss_results + bm25_results:
        cid = result["chunk_id"]
        scores_rrf[cid] = scores_rrf.get(cid, 0) + 1 / (result["rank"] + 60)
        chunks_map[cid] = result

    sorted_ids = sorted(scores_rrf.keys(),
                        key=lambda cid: scores_rrf[cid],
                        reverse=True)[:top_k]
    return [chunks_map[cid] for cid in sorted_ids]


def generate_answer(query, chunks):
    client = AzureOpenAI(
        azure_endpoint = "https://gdas-openai.openai.azure.com/",
        api_key        = os.environ.get("AZURE_OPENAI_KEY"),
        api_version    = "2024-12-01-preview"
    )
    context = ""
    for i, chunk in enumerate(chunks):
        context += f"\n[Source {i+1}: {chunk['source']}]\n{chunk['text']}\n"

    response = client.chat.completions.create(
        model    = "gpt-4.1",
        messages = [
            {"role": "system", "content": """You are a helpful insurance assistant.
Try to answer using the provided context first.
If the context doesn't contain enough information,
you may use your general insurance knowledge BUT
clearly state: 'Based on general insurance knowledge
(not your specific policy):' before answering.
Always mention which source your answer comes from when using context.
Keep your answer clear and concise."""},
            {"role": "user", "content": f"Context:\n{context}\nQuestion: {query}\nAnswer:"}
        ],
        temperature = 0,
        max_tokens  = 500,
    )
    return response.choices[0].message.content


# ── Endpoints ─────────────────────────────────────────
@app.get("/health")
def health():
    """Health check — used by AWS and Docker."""
    return {
        "status"        : "ok",
        "chunks_indexed": len(state["all_chunks"])
    }


@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    """
    Main endpoint — takes a question, returns an answer.

    Flow:
    1. Input guardrail
    2. Hybrid retrieval
    3. GPT generation
    4. Return answer with sources
    """
    query = request.question.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Step 1 — input guardrail
    safe, reason = check_input(query)
    if not safe:
        return AnswerResponse(
            answer  = "",
            sources = [],
            blocked = True,
            reason  = reason
        )

    # Step 2 — retrieve
    chunks = hybrid_retrieve(query)

    # Step 3 — generate
    answer  = generate_answer(query, chunks)
    sources = list(set(c["source"] for c in chunks))

    return AnswerResponse(
        answer  = answer,
        sources = sources,
        blocked = False,
    )