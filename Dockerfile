# ── Base image ─────────────────────────────────────────
# Why python:3.11-slim?
# - slim = smaller image (no unnecessary OS packages)
# - 3.11 = stable, fast, widely supported
# - Alternative: python:3.11-alpine (even smaller but
#   compilation issues with faiss and numpy)
FROM python:3.11-slim

# ── Set working directory ──────────────────────────────
# All commands run from here inside the container
WORKDIR /app

# ── Install system dependencies ────────────────────────
# faiss-cpu needs these C libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Copy requirements first ────────────────────────────
# Why copy requirements before code?
# Docker caches each step (layer).
# If code changes but requirements don't,
# Docker skips reinstalling libraries.
# This makes rebuilds much faster.
COPY requirements.txt .

# ── Install Python dependencies ────────────────────────
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source code only ──────────────────────────────
COPY src/ ./src/

# ── Create empty data directories ─────────────────────
# Documents are generated at startup, not baked into image
# Why? Baking docs into image means rebuilding every time
# docs change. Runtime generation is more flexible.
RUN mkdir -p data/raw data/processed

# ── Environment variables ──────────────────────────────
# Don't hardcode secrets — pass them at runtime
ENV AZURE_OPENAI_KEY=""

# ── Expose port ────────────────────────────────────────
# FastAPI runs on 8000 by default
EXPOSE 8000

# ── Start command ──────────────────────────────────────
# uvicorn = the server that runs FastAPI
# --host 0.0.0.0 = accept connections from outside container
# --port 8000 = listen on port 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]