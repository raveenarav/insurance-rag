"""
generator.py

Generates grounded answers using Azure OpenAI GPT-4.1.

Why grounded generation?
- LLMs hallucinate without context
- We inject retrieved chunks into the prompt
- GPT answers from context first, general knowledge as fallback
- Fallback is clearly labelled so user knows the source
"""

import os
from openai import AzureOpenAI


# ── Azure OpenAI client ───────────────────────────────
def get_client():
    return AzureOpenAI(
        azure_endpoint = "https://gdas-openai.openai.azure.com/",
        api_key        = os.environ.get("AZURE_OPENAI_KEY"),
        api_version    = "2024-12-01-preview"
    )


DEPLOYMENT = "gpt-4.1"


# ── Build the prompt ──────────────────────────────────
def build_prompt(query, retrieved_chunks):
    """
    Builds a grounded prompt from query + retrieved chunks.

    Structure:
    - System message: rules for GPT
    - User message: context + question
    """
    # Build context from retrieved chunks
    context = ""
    for i, chunk in enumerate(retrieved_chunks):
        context += f"\n[Source {i+1}: {chunk['source']}]\n"
        context += chunk["text"]
        context += "\n"

    system_message = """You are a helpful insurance assistant.
Try to answer using the provided context first.
If the context doesn't contain enough information,
you may use your general insurance knowledge BUT
clearly state: 'Based on general insurance knowledge
(not your specific policy):' before answering.
Always prefer context over general knowledge.
Always mention which source your answer comes from when using context.
Keep your answer clear and concise."""

    user_message = f"""Context:
{context}

Question: {query}

Answer:"""

    return system_message, user_message


# ── Generate answer ───────────────────────────────────
def generate_answer(query, retrieved_chunks):
    """
    Generates a grounded answer using Azure OpenAI GPT-4.1.

    Steps:
    1. Build prompt with retrieved chunks as context
    2. Call GPT-4.1
    3. Return the answer
    """
    client = get_client()
    system_message, user_message = build_prompt(query, retrieved_chunks)

    response = client.chat.completions.create(
        model    = DEPLOYMENT,
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user",   "content": user_message},
        ],
        temperature = 0,      # 0 = deterministic, no creativity
        max_tokens  = 500,    # limit response length
    )

    return response.choices[0].message.content


# ── Main ──────────────────────────────────────────────
if __name__ == "__main__":
    print("This module is imported by other files.")
    print("Test it through the full pipeline.")