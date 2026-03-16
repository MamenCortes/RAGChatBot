from huggingface_hub import InferenceClient
from dataclasses import dataclass
from .retrieval import search, hybrid_search, language_aware_hybrid_search

"""This integrates your Hugging Face InferenceClient chat completions pattern from bot.py bot, but adds retrieved context."""

# @TODO Evaluate using the following refined prompts instead of the current
# ones, keeping both modes as similar as possible except for the allowed source
# of information.

# RAG prompt proposal:
SYSTEM_PROMPT = """You are a supportive assistant for people affected by breast cancer. Use only the information provided in the retrieved context to answer the user. Do not rely on outside knowledge. 
If the retrieved context does not contain enough information to answer safely, say so clearly.

Rules:
- Be clear, calm, and compassionate.
- Answer in the user's language.
- Keep answers concise, unless the user asks for more detail.
- Do not invent facts or make unsupported claims.
- Do not present yourself as a doctor or replace a healthcare professional.
- Do not give definitive diagnoses or personalized treatment decisions.
- Do not tell the user to start, stop, or change medication or cancer treatment.
- If the user describes alarming symptoms, severe emotional distress, or a possible emergency, advise them to contact their oncology team or emergency services immediately.
- When the available information is limited, acknowledge the uncertainty plainly.
- Use simple language and avoid unnecessary jargon. """

# No-retrieval prompt proposal:
NO_RETRIEVAL_SYSTEM_PROMPT = """
    Use only your general knowledge to answer the user. Do not consult or refer to internet documents or external sources. If you do not know enough to answer safely, say so clearly.
    
    Rules:
    - Be clear, calm, and compassionate.
    - Answer in the user's language.
    - Keep answers concise, unless the user asks for more detail.
    - Do not invent facts or make unsupported claims.
    - Do not present yourself as a doctor or replace a healthcare professional.
    - Do not give definitive diagnoses or personalized treatment decisions.
    - Do not tell the user to start, stop, or change medication or cancer treatment.
    - If the user describes alarming symptoms, severe emotional distress, or a possible emergency, advise them to contact their oncology team or emergency services immediately.
    - When the available information is limited, acknowledge the uncertainty plainly.
    - Use simple language and avoid unnecessary jargon.
# """

#Old system prompt
#SYSTEM_PROMPT = """You are a helpful assistant.
#Answer using the provided context when relevant.
#If the context does not contain the answer, say you don't know and ask a clarifying question.
#Keep answers concise and in the user's language."""

#NO_RETRIEVAL_SYSTEM_PROMPT = """You are a helpful assistant.
#Answer the question using only your own knowledge, without any external context.
#Keep answers concise and in the user's language."""

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

def _call_llm(
    hf_client: InferenceClient,
    messages: list[dict],
    model: str,
    max_tokens: int = 500,
) -> str:
    completion = hf_client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
    )
    """
    Example completion object:
    completion = {
        "choices": [
            {
            "message": {
                "role": "assistant",
                "content": "Chemotherapy is a treatment that..."
            }
            }
        ]
    }"""

    return completion.choices[0].message["content"]

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
    # add short history (avoid unbounded growth)
    messages.extend(chat_history[-10:])

    # Revised: the user message is added after retrieving the LLM answer. 
    # @TODO Possible duplicate send: the current user turn appears to have been
    # added first in app/telegram_bot.py:107 (normal mode), and similarly in
    # app/telegram_bot.py:125 for eval mode, before being appended again here.
    
    # Add the user message with the retrieved context injected.
    messages.append({
        "role": "user",
        "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{user_message}"
    })

    """
    Messages format example to send to the LLM:
    messages = [
    {"role":"system", "content": SYSTEM_PROMPT},

    # History (trimmed to last 10 turns to avoid token overload):
    {"role":"user","content":"hola"},
    {"role":"assistant","content":"hola!"},

    {"role":"user","content":
    "CONTEXT:\n[retrieved docs]\n\nQUESTION:\nWhat is chemotherapy?"
    }]"""

    return _call_llm(hf_client, messages, model)

@dataclass
class TripleAnswer:
    system_prompt_only: str
    no_retrieval: str
    rag_retrieval: str

def rag_answer_3_modes(hf_client: InferenceClient,
    user_message: str,
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
    top_k: int = 5,
    verbose: bool = False,) -> TripleAnswer:
    """
    Generate 3 answers for the same user query:

    1. System promt only  — pure LLM but using a curated system promt.
    3. No retrieval       — pure LLM, no context injected.
    3. Hybrid RAG         — combines keword search + semantic via RRF (hybrid_search()).

    Returns a TripleAnswer dataclass with .system_prompt_only, .no_retrieval, .rag_retrieval fields.
    The chat history is not included in this function since it is intended for eval mode where the user evaluates individual answers for specific queries"""

    # ── 1. No retrieval ───────────────────────────────────────────────────────
    no_retrieval_messages = [{"role": "system", "content": NO_RETRIEVAL_SYSTEM_PROMPT}]
    no_retrieval_messages.append({"role": "user", "content": user_message})
    no_retrieval_answer = _call_llm(hf_client, no_retrieval_messages, model)

    # ── 2. No retrieval, system promt only ───────────────────────────────────────────────────────
    sp_only_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    sp_only_messages.append({"role": "user", "content": user_message})
    sp_only_answer = _call_llm(hf_client, sp_only_messages, model)

    # ── 3. Hybrid RAG ────────────────────────────────────────────────────────
    hybrid_chunks = hybrid_search(user_message, top_k=top_k)
    if verbose:
        print(f"[Hybrid] Retrieved {len(hybrid_chunks)} chunks")
    hybrid_context = build_context_block(hybrid_chunks, verbose=verbose)

    hybrid_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    hybrid_messages.append({
        "role": "user",
        "content": f"CONTEXT:\n{hybrid_context}\n\nQUESTION:\n{user_message}",
    })
    hybrid_answer = _call_llm(hf_client, hybrid_messages, model)    

    return TripleAnswer(
        no_retrieval=no_retrieval_answer,
        system_prompt_only=sp_only_answer,
        rag_retrieval=hybrid_answer,
    )


@dataclass
class MultipleAnswer:
    semantic: str
    hybrid: str
    language_aware_hybrid: str
    no_retrieval: str

def rag_answer_all_modes(
    hf_client: InferenceClient,
    user_message: str,
    chat_history: list[dict],
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
    top_k: int = 5,
    verbose: bool = False,
) -> MultipleAnswer:
    """
    Generate multiple answers for the same user query:

    1. Semantic-only RAG  — uses cosine vector search (search()).
    2. Hybrid RAG         — combines keword search + semantic via RRF (hybrid_search()).
    3. Language aware Hybrid RAG - combines semantic search and keyword search on same-language query-chunks via via RRF (language_aware_hybrid_search())
    3. No retrieval       — pure LLM, no context injected.

    Returns a MultipleAnswer dataclass with .semantic, .hybrid, .language_aware_hybrid, .no_retrieval fields.
    """
    trimmed_history = chat_history[-10:]

    # ── 1. Semantic RAG ──────────────────────────────────────────────────────
    semantic_chunks = search(user_message, top_k=top_k)
    if verbose:
        print(f"[Semantic] Retrieved {len(semantic_chunks)} chunks")
    semantic_context = build_context_block(semantic_chunks, verbose=verbose)

    semantic_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    semantic_messages.extend(trimmed_history)
    semantic_messages.append({
        "role": "user",
        "content": f"CONTEXT:\n{semantic_context}\n\nQUESTION:\n{user_message}",
    })
    semantic_answer = _call_llm(hf_client, semantic_messages, model)

    # ── 2. Hybrid RAG ────────────────────────────────────────────────────────
    hybrid_chunks = hybrid_search(user_message, top_k=top_k)
    if verbose:
        print(f"[Hybrid] Retrieved {len(hybrid_chunks)} chunks")
    hybrid_context = build_context_block(hybrid_chunks, verbose=verbose)

    hybrid_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    hybrid_messages.extend(trimmed_history)
    hybrid_messages.append({
        "role": "user",
        "content": f"CONTEXT:\n{hybrid_context}\n\nQUESTION:\n{user_message}",
    })
    hybrid_answer = _call_llm(hf_client, hybrid_messages, model)

    # ── 2. Language-aware Hybrid RAG ────────────────────────────────────────────────────────
    la_hybrid_chunks = language_aware_hybrid_search(user_message, top_k=top_k)
    if verbose:
        print(f"[Hybrid] Retrieved {len(la_hybrid_chunks)} chunks")
    la_hybrid_context = build_context_block(la_hybrid_chunks, verbose=verbose)

    la_hybrid_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    la_hybrid_messages.extend(trimmed_history)
    la_hybrid_messages.append({
        "role": "user",
        "content": f"CONTEXT:\n{la_hybrid_context}\n\nQUESTION:\n{user_message}",
    })
    la_hybrid_answer = _call_llm(hf_client, la_hybrid_messages, model)

    # ── 3. No retrieval ───────────────────────────────────────────────────────
    no_retrieval_messages = [{"role": "system", "content": NO_RETRIEVAL_SYSTEM_PROMPT}]
    no_retrieval_messages.extend(trimmed_history)
    no_retrieval_messages.append({"role": "user", "content": user_message})
    no_retrieval_answer = _call_llm(hf_client, no_retrieval_messages, model)

    return MultipleAnswer(
        semantic=semantic_answer,
        hybrid=hybrid_answer,
        language_aware_hybrid=la_hybrid_answer,
        no_retrieval=no_retrieval_answer,
    )
