from huggingface_hub import InferenceClient
from .retrieval import search

"""This integrates your Hugging Face InferenceClient chat completions pattern from bot.py bot, but adds retrieved context."""

SYSTEM_PROMPT = """You are a helpful assistant.
Answer using the provided context when relevant.
If the context does not contain the answer, say you don't know and ask a clarifying question.
Keep answers concise and in the user's language.
"""

verbose = False

def build_context_block(chunks, verbose: bool = False) -> str:
    # Keep it compact; you can add citations like [doc_id:chunk_id]
    parts = []
    for c in chunks:
        string = f"[{c.doc_id}:{c.chunk_id}] {c.content}"
        parts.append(string)
        if verbose:
            print(f"Context chunk (distance {c.distance:.4f}): [{c.doc_id}:{c.chunk_id}] {c.content[:100]}\n")

    return "\n\n".join(parts)

def rag_answer(
    hf_client: InferenceClient,
    user_message: str,
    chat_history: list[dict],
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
    top_k: int = 5,
    verbose: bool = False,
) -> str:
    verbose = verbose
    retrieved = search(user_message, top_k=top_k)
    if verbose:
        print(f"Retrieved {len(retrieved)} chunks for query: '{user_message}'")
    context = build_context_block(retrieved, verbose=verbose)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # short history (avoid unbounded growth)
    messages.extend(chat_history[-10:])

    messages.append({
        "role": "user",
        "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{user_message}"
    })

    completion = hf_client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=500,
    )
    return completion.choices[0].message["content"]
