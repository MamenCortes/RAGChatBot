import re
from typing import List
# Data / vector math
import numpy as np
# Embeddings (for semantic chunking + search)
from sentence_transformers import SentenceTransformer

"""
def split_into_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\n\\s*\n", text) if p.strip()]
    print(f"Split into {len(parts)} paragraphs.")
    return parts"""

def split_into_paragraphs(text: str, min_len: int = 30) -> list[str]:
    """
    Split PyMuPDF-extracted text into paragraphs.

    PyMuPDF separates lines with a single '\\n', never '\\n\\n', so a naive
    double-newline split returns the whole page as one paragraph.  This
    function detects paragraph boundaries from typographic and linguistic
    cues instead.

    A new paragraph starts when ANY of the following is true:

    1. **Blank line** – one or more lines containing only whitespace
       (handles PDFs that do produce blank lines occasionally).

    2. **Sentence end → capital start** – the previous non-empty line ends
       with sentence-closing punctuation (. ! ? : … ») AND the current line
       starts with an uppercase letter that is not a continuation abbreviation.

    3. **Indented line** – the current line starts with 2+ spaces or a tab
       (common in body-text PDFs where paragraphs are indented).

    4. **Short "orphan" line** – the previous line is significantly shorter
       than the median line length (≤ 60 % of median) and doesn't end with a
       hyphen, suggesting it was the last line of a paragraph.

    5. **Bullet / numbered list item** – line begins with a bullet character
       or a pattern like "1.", "a)", "(2)", etc.

    Args:
        text:    Cleaned page text from normalize_pdf_text().
        min_len: Paragraphs (after joining) shorter than this are merged
                 with the next one rather than emitted stand-alone.

    Returns:
        List of paragraph strings with internal newlines collapsed to spaces.
    """
    lines = text.splitlines()
    if not lines:
        return []

    # ------------------------------------------------------------------ #
    # Pre-compute median line length to detect short "orphan" lines       #
    # ------------------------------------------------------------------ #
    lengths = [len(ln) for ln in lines if ln.strip()]
    if lengths:
        sorted_len = sorted(lengths)
        median_len = sorted_len[len(sorted_len) // 2]
    else:
        median_len = 80  # safe fallback

    ORPHAN_RATIO   = 0.60   # line is "short" if len ≤ ratio * median
    INDENT_CHARS   = 2      # spaces at line start → new paragraph

    # Sentence-ending punctuation (handles Spanish / English)
    _SENT_END = re.compile(r'[.!?…:»\"]$')
    # Uppercase start (but not a single capital followed by '.' = abbreviation)
    _UPPER_START = re.compile(r'^[A-ZÁÉÍÓÚÜÑ][^.]')
    # Bullet or numbered list
    _LIST_ITEM = re.compile(
        r'^(\s*[\•\-\–\*\·▪▸►✓✔]\s+'          # bullet symbols
        r'|\s*\(?[0-9]{1,2}[\.\)]\s+'           # 1. 2) (3)
        r'|\s*\(?[a-záéíóú][\.\)]\s+)'          # a. b) (c)
        , re.IGNORECASE
    )

    def _is_boundary(prev_line: str, curr_line: str) -> bool:
        prev_s = prev_line.rstrip()
        curr_s = curr_line

        # Rule 1: blank line (already handled in the loop below, but kept
        #         here for completeness when called directly)
        if not prev_s.strip():
            return True

        # Rule 2: sentence end → capital start
        if _SENT_END.search(prev_s) and _UPPER_START.match(curr_s.lstrip()):
            return True

        # Rule 3: indented line (new paragraph)
        if curr_line.startswith(" " * INDENT_CHARS) or curr_line.startswith("\t"):
            return True

        # Rule 4: previous line is a short orphan (not hyphenated)
        if (len(prev_s.strip()) <= median_len * ORPHAN_RATIO
                and not prev_s.endswith("-")
                and prev_s.strip()):
            return True

        # Rule 5: current line is a list item
        if _LIST_ITEM.match(curr_s):
            return True

        return False

    # ------------------------------------------------------------------ #
    # Main grouping loop                                                   #
    # ------------------------------------------------------------------ #
    groups: list[list[str]] = [[]]

    i = 0
    while i < len(lines):
        line = lines[i]

        # Blank line → always a boundary; consume consecutive blank lines
        if not line.strip():
            if groups[-1]:          # avoid creating an empty leading group
                groups.append([])
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            continue

        # Non-blank: check if it starts a new paragraph relative to previous
        if groups[-1] and _is_boundary(groups[-1][-1], line):
            groups.append([])

        groups[-1].append(line.strip())
        i += 1

    # ------------------------------------------------------------------ #
    # Convert groups → paragraph strings; merge very short ones           #
    # ------------------------------------------------------------------ #
    paragraphs: list[str] = []
    buffer = ""

    for group in groups:
        if not group:
            continue
        text_block = " ".join(group)

        if len(text_block) < min_len:
            # Too short: accumulate into buffer
            buffer = (buffer + " " + text_block).strip() if buffer else text_block
        else:
            if buffer:
                # Flush buffer as its own paragraph before the next one
                paragraphs.append(buffer)
                buffer = ""
            paragraphs.append(text_block)

    if buffer:
        if paragraphs:
            # Attach leftover buffer to the last paragraph
            paragraphs[-1] = (paragraphs[-1] + " " + buffer).strip()
        else:
            paragraphs.append(buffer)

    print(f"Split into {len(paragraphs)} paragraphs.")
    return paragraphs

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
    # Revised to include tail chunks that may be shorter than min_chars.
    # @TODO This currently drops any chunk shorter than min_chars, including a
    # potentially valid final tail chunk at the end of the text. That means
    # some content can be silently omitted during ingestion, although for now
    # the impact does not appear too critical with the current chunk sizes.
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
        #Save if the chunk is big enough, or if it's the last chunk (even if small)
        if len(chunk) >= min_chars or end == len(text):
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
