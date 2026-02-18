import re
from typing import List
# Data / vector math
import numpy as np
# Embeddings (for semantic chunking + search)
from sentence_transformers import SentenceTransformer

def split_into_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    print(f"Split into {len(parts)} paragraphs.")
    return parts

def chunk_paragraphs(paragraphs: list[str], chunk_chars: int = 1600, overlap: int = 150) -> list[str]:
    """
    Chunk paragraphs with optional overlap.
    - chunk_chars: target max chars per chunk (for Spanish, ~1600 chars is often ~300–500 tokens)
    - overlap: how many chars to overlap between chunks (to preserve context across boundaries) 
    """
    chunks = []
    buf = ""

    for p in paragraphs:
        if len(buf) + len(p) + 2 <= chunk_chars:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            # start new buffer with overlap from previous
            if chunks and overlap > 0:
                tail = chunks[-1][-overlap:]
                buf = (tail + "\n\n" + p).strip()
            else:
                buf = p.strip()

    if buf:
        chunks.append(buf)

    return chunks

def fixed_size_chunks(
    text: str,
    chunk_chars: int = 1600,
    min_chars: int = 200
) -> List[str]:
    """
    Split text into fixed-size chunks by character length.

    For Spanish, ~1600 chars is often ~300–500 tokens (roughly).
    """
    text = text.strip()
    if not text:
        print("[ERR] empty text")
        return []

    out = []
    for start in range(0, len(text), chunk_chars):
        chunk = text[start:start + chunk_chars].strip()
        if len(chunk) >= min_chars:
            out.append(chunk)
    return out

def fixed_size_chunks_with_overlap(text: str,chunk_chars: int = 1600, overlap_chars: int = 200,min_chars: int = 200) -> List[str]:
    """
    Sliding window chunking: fixed size + overlap.

    Important: overlap_chars must be < chunk_chars.
    We also ensure the step makes progress (avoids infinite loops).
    """
    text = text.strip()
    if not text:
        print("[ERR] empty text")
        return []
    if overlap_chars >= chunk_chars:
        raise ValueError("overlap_chars must be < chunk_chars")

    out = []
    start = 0
    step = chunk_chars - overlap_chars

    while start < len(text):
        end = min(start + chunk_chars, len(text))
        chunk = text[start:end].strip()
        if len(chunk) >= min_chars:
            out.append(chunk)
        if end == len(text):
            break
        start += step

    return out

#Split after ., ? or ! and keeps the punctuation attached to the sentence
def split_into_sentences_simple(text: str) -> List[str]:
    """
    Simple sentence splitter (heuristic).
    Not perfect, but works decently for Spanish in many cases.
    """
    sents = re.split(r"(?<=[\.\?\!])\s+", text.strip())
    return [s.strip() for s in sents if s.strip()]

def recursive_character_split(
    text: str,
    chunk_chars: int = 1600,
    min_chars: int = 200
) -> List[str]:
    """
    Variable-size chunker:
    - Try to build chunks by paragraphs first
    - If paragraphs are too big, fall back to sentences
    - If still too big, fall back to fixed char splitting

    Why this approach?
    - Maintains semantic coherence where possible (paragraph/sentence)
    - Still guarantees chunks won't exceed the desired size

    Goal:
    Produce chunks that:
    - are not longer than chunk_chars
    - are not too small (min_chars)
    - preserve meaning (paragraphs and sentences stay together when possible)
    """
    text = text.strip()
    if not text:
        print("[ERR] empty text")
        return []

    paragraphs = split_into_paragraphs(text)
    # If we have no real paragraphs, fallback to sentences
    if len(paragraphs) <= 1:
        paragraphs = split_into_sentences_simple(text)

    chunks = [] #final output
    cur = "" #the current chunk being built

    #Takes whatever text is currently accumulated in cur, cleans it and if it's big enough, stores it. 
    # Then resets cur to start a nuew chunk
    def flush():
        nonlocal cur #without nonlocal, flush() would think cur is a local variable inside it
        cur = cur.strip()
        if len(cur) >= min_chars:
            chunks.append(cur)
        cur = ""


    for unit in paragraphs: #Iterate over paragraphs or sentences
        unit = unit.strip()
        if not unit: #skip empty ones
            continue

        #CASE 1: Unit is too large on its own
        #Fixed size chunk
        # If a single unit is bigger than the chunk size, split it further
        if len(unit) > chunk_chars:
            # flush current chunk first
            if cur:
                flush()
            # split oversized unit using fixed-size chunking
            for sub in fixed_size_chunks(unit, chunk_chars=chunk_chars, min_chars=min_chars):
                chunks.append(sub)
            continue

        #CASE 2: Unit fits into current chunk
        # Add unit to current chunk if it fits, otherwise flush and start new
        if len(cur) + len(unit) + 2 <= chunk_chars:
            cur = (cur + "\n\n" + unit) if cur else unit
        #CASE 3: Unit doesn't fit, then flush and start new chunk with current unit
        else:
            flush()
            cur = unit

    if cur:
        flush()

    return chunks

def semantic_chunking(
    text: str,
    model: SentenceTransformer,
    max_chunk_chars: int = 1600,
    min_chars: int = 200,
    sim_threshold: float = 0.55
) -> List[str]:
    """
    Semantic chunking via sentence embeddings:
    - Split into sentences
    - If there is only one sentence, apply fixed-size chunking
    - Embed each sentence
    - Break chunk when similarity to next sentence drops under threshold
    - Also enforce max_chunk_chars

    Why do this?
    - Helps keep each chunk about "one idea" which often improves retrieval quality
    """
    sentences = split_into_sentences_simple(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return fixed_size_chunks(sentences[0], chunk_chars=max_chunk_chars, min_chars=min_chars)

    sent_vecs = model.encode(sentences, normalize_embeddings=True)
    chunks = []
    cur_sents = [sentences[0]]
    cur_len = len(sentences[0])

    for i in range(len(sentences) - 1):
        # cosine similarity because normalized embeddings
        sim = float(np.dot(sent_vecs[i], sent_vecs[i+1]))

        nxt = sentences[i+1]
        # boundary if meaning shifts OR chunk too big
        if sim < sim_threshold or (cur_len + 1 + len(nxt) > max_chunk_chars):
            chunk = " ".join(cur_sents).strip()
            if len(chunk) >= min_chars:
                chunks.append(chunk)
            cur_sents = [nxt]
            cur_len = len(nxt)
        else:
            cur_sents.append(nxt)
            cur_len += 1 + len(nxt)

    # flush
    chunk = " ".join(cur_sents).strip()
    if len(chunk) >= min_chars:
        chunks.append(chunk)

    return chunks
