# scripts/ingest_docs.py
#python -m scripts.ingest_docs
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator
import pymupdf
from collections import Counter

from app.chunking import split_into_paragraphs, chunk_paragraphs, fixed_size_chunks_with_overlap
from app.ingest import ChunkRecord, insert_or_replace_chunks_for_doc
from app.db import get_conn


# ------------------ Registry table (rag_documents) ------------------

def ensure_doc_registry_table() -> None:
    sql = """
    CREATE TABLE IF NOT EXISTS rag_documents (
      doc_id TEXT PRIMARY KEY,
      source_path TEXT,
      num_pages INT,
      fingerprint TEXT NOT NULL,
      updated_at TIMESTAMPTZ DEFAULT now()
    );
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
    finally:
        conn.close()

def get_existing_fingerprint(doc_id: str) -> str | None:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT fingerprint FROM rag_documents WHERE doc_id = %s;", (doc_id,))
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        conn.close()

def upsert_document_registry(doc_id: str, source_path: str, fingerprint: str, num_pages: int | None) -> None:
    sql = """
    INSERT INTO rag_documents (doc_id, source_path, fingerprint, num_pages)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (doc_id)
    DO UPDATE SET
      source_path = EXCLUDED.source_path,
      fingerprint = EXCLUDED.fingerprint,
      num_pages = EXCLUDED.num_pages,
      updated_at = now();
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (doc_id, source_path, fingerprint, num_pages))
        conn.commit()
    finally:
        conn.close()


# ------------------ File discovery ------------------

SUPPORTED_EXTS = {".txt", ".md", ".pdf"}

def iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for p in path.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            yield p


# ------------------ Fingerprints ------------------

def fingerprint_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ------------------ PDF text cleaning ------------------

def extract_header_footer_candidates(
    text: str, *, top_n: int = 3, bottom_n: int = 3
) -> list[str]:
    """
    Return the first `top_n` and last `bottom_n` non-empty lines of a page
    as header/footer candidates.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    return lines[:top_n] + lines[-bottom_n:]


def detect_repeated_headers_footers(
    page_texts: list[str],
    *,
    min_freq_ratio: float = 0.45,
    top_n: int = 3,
    bottom_n: int = 3,
) -> set[str]:
    """
    Identify lines that appear on many pages and are likely headers/footers.

    For every page the first `top_n` and last `bottom_n` non-empty lines are
    collected as candidates. Any candidate that appears on at least
    `min_freq_ratio` of pages is returned as a repeated element to be removed.
    Short purely-numeric strings (page numbers) are always included.

    Args:
        page_texts:     One raw-text string per page (from pymupdf get_text).
        min_freq_ratio: Fraction of pages (0-1) a line must appear on to be
                        flagged. Default 0.45 → 45 % of pages.
        top_n:          Lines examined from the top of each page.
        bottom_n:       Lines examined from the bottom of each page.

    Returns:
        Set of strings that should be stripped from every page.
    """
    n_pages = len(page_texts)
    if n_pages == 0:
        return set()

    counter: Counter[str] = Counter()
    for txt in page_texts:
        # Deduplicate within a single page so one repeated line doesn't
        # artificially inflate the cross-page count.
        candidates = set(extract_header_footer_candidates(txt, top_n=top_n, bottom_n=bottom_n))
        for c in candidates:
            counter[c] += 1

    def _is_page_number(line: str) -> bool:
        """True for short strings that look like bare page numbers."""
        return bool(re.fullmatch(r"[\-\u2013]?\s*\d+\s*[\-\u2013]?", line.strip()))

    return {
        line
        for line, count in counter.items()
        if count / n_pages >= min_freq_ratio or _is_page_number(line)
    }


def normalize_pdf_text(text: str, headers_footers: set[str] | None = None) -> str:
    """
    Clean raw text extracted from a PDF page (or a multi-page concatenation).

    Operations performed in order:
      1. Remove lines whose stripped content is in `headers_footers`.
      2. Replace soft hyphens (U+00AD) and common PDF ligature characters.
      3. Rejoin words hyphenated across a line break (``word-\\nword`` → ``wordword``).
      4. Strip trailing whitespace from every line.
      5. Collapse consecutive blank lines to a single blank line.
      6. Collapse runs of multiple spaces within a line to one space.

    Args:
        text:            Raw string from ``page.get_text("text")``.
        headers_footers: Optional set of exact stripped line strings to remove
                         before any other processing. Typically the output of
                         :func:`detect_repeated_headers_footers`.

    Returns:
        Cleaned string ready for paragraph splitting.
    """
    # Step 1 – remove known header/footer lines
    # We only elimintae the header/footer lines (including page numbers) if they are exact matches after stripping, to avoid accidentally removing valid content.
    if headers_footers:
        text = "\n".join(
            ln for ln in text.splitlines()
            if ln.strip() not in headers_footers
        )

    # Step 2 – soft hyphens and ligatures
    text = text.replace("\u00ad", "")  # soft hyphen

    ligatures: dict[str, str] = {
        "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
        "\ufb03": "ffi", "\ufb04": "ffl",
        "\u2019": "'",  "\u2018": "'",
        "\u201c": '"',  "\u201d": '"',
        "\u2013": "-",  "\u2014": "--",
    }
    for char, replacement in ligatures.items():
        text = text.replace(char, replacement)

    # Step 3 – rejoin hyphenated line breaks
    text = re.sub(r"-\n(\S)", r"\1", text)

    # Steps 4 & 5 – strip trailing whitespace + collapse blank lines
    lines = [ln.rstrip() for ln in text.splitlines()]
    cleaned: list[str] = []
    prev_blank = False
    for ln in lines:
        is_blank = not ln.strip()
        if is_blank and prev_blank:
            continue
        cleaned.append(ln)
        prev_blank = is_blank

    # Step 6 – collapse internal whitespace
    cleaned = [re.sub(r" {2,}", " ", ln) for ln in cleaned]

    return "\n".join(cleaned).strip()

@dataclass(frozen=True)
class DocMeta:
    doc_id: str
    topic: str | None
    lang: str | None
    source: str | None
    slug: str | None
    version: str | None

def _normalize_stem(stem: str) -> str:
    # mimic the notebook's “safe stem” idea
    return stem.strip()

def parse_filename_meta(pdf_path: Path) -> DocMeta:
    """
    Expected filename (before .pdf):
      <topic>_<lang>_<source>_<slug>_<version>

    Example:
      mama_es_novartis_guia-pacientes-CM_2025.pdf
    """
    stem = _normalize_stem(pdf_path.stem)

    m = re.match(
        r"^(?P<topic>[a-z0-9-]+)_(?P<lang>[a-z]{2})_(?P<source>[a-z0-9-]+)_(?P<slug>.+)_(?P<version>[a-z0-9.-]+)$",
        stem,
        flags=re.IGNORECASE,
    )
    if not m:
        # fallback: still ingest, but with minimal metadata
        return DocMeta(
            doc_id=str(pdf_path.resolve()),
            topic=None, lang=None, source=None, slug=None, version=None
        )

    topic = m.group("topic").lower()
    lang = m.group("lang").lower()
    source = m.group("source").lower()
    slug = m.group("slug")
    version = m.group("version")

    # Use a stable doc_id; notebook often uses a composed id, but this works well:
    doc_id = f"{topic}_{lang}_{source}_{slug}_{version}"

    return DocMeta(
        doc_id=doc_id,
        topic=topic,
        lang=lang,
        source=source,
        slug=slug,
        version=version,
    )

# ------------------ PDF chunk extraction ------------------

def extract_pdf_page_paragraph_chunks(
    pdf_path: Path, meta: DocMeta, chunk_chars: int = 1600
) -> list[ChunkRecord]:
    """
    Open `pdf_path`, remove repeated headers/footers, normalise each page,
    split into paragraphs, and group them into chunks ready for ingestion.
    """
    overlap = int(chunk_chars * 0.15)  # 15 % overlap
    records: list[ChunkRecord] = []

    with pymupdf.open(pdf_path) as doc:
        # Load all raw page texts up-front so we can do cross-page analysis
        raw_texts = [doc.load_page(p).get_text("text") for p in range(doc.page_count)]

        # Detect repeated headers/footers across all pages at once
        headers_footers = detect_repeated_headers_footers(raw_texts)
        print(f"Detected {len(headers_footers)} repeated header/footer lines: {headers_footers}")

        for i, raw in enumerate(raw_texts):
            # normalize_pdf_text removes header/footer lines and cleans the text
            text = normalize_pdf_text(raw, headers_footers=headers_footers)
            if len(text) < 20:
                continue

            page_num = i + 1
            """paragraphs = split_into_paragraphs(text)
            print(f"Page {page_num}: {len(paragraphs)} paragraphs")
            print(f"paragraphs: {[p[:50] for p in paragraphs]}")

            subchunks = chunk_paragraphs(paragraphs, chunk_chars=chunk_chars, overlap=overlap)"""
            chunks = fixed_size_chunks_with_overlap(
            text,
            chunk_chars=1600,
            overlap_chars=200,
            min_chars=200
            )

            #print(f"Page {page_num}: {len(chunks)} chunks")

            for j, ch in enumerate(chunks):
                records.append(ChunkRecord(
                    doc_id=meta.doc_id,
                    chunk_id=f"p{page_num:03d}_c{j:02d}",
                    content=ch,
                    topic=meta.topic,
                    source=str(pdf_path),
                    lang=meta.lang,
                    page_num=page_num,
                ))

    return records

# ------------------ Text extraction for txt/md ------------------

def read_text_file(p: Path) -> tuple[str, bytes]:
    b = p.read_bytes()
    try:
        text = b.decode("utf-8")
    except UnicodeDecodeError:
        text = b.decode("latin-1", errors="ignore")
    return text, b


# ------------------ Ingestion per file ------------------

def ingest_one_file(p: Path, fallback_topic: str | None = None, fallback_lang: str | None = None) -> None:
    raw_bytes = p.read_bytes()
    fp = fingerprint_bytes(raw_bytes)

    num_pages = None
    if p.suffix.lower() == ".pdf":
        # .pdf: extract text, split into paragraphs, then chunk with overlap to preserve context across boundaries.
        # paragraph-aware chunking
        meta = parse_filename_meta(p)

        with pymupdf.open(p) as doc:
            num_pages = doc.page_count
        
        chunk_records = extract_pdf_page_paragraph_chunks(p, meta, chunk_chars=1600)
        doc_id = meta.doc_id
        topic = meta.topic or fallback_topic
        lang = meta.lang or fallback_lang

        # apply fallbacks to each record if meta missing
        if topic or lang:
            fixed = []
            for r in chunk_records:
                fixed.append(ChunkRecord(
                    doc_id=r.doc_id,
                    chunk_id=r.chunk_id,
                    content=r.content,
                    topic=topic,
                    source=r.source,
                    lang=lang,
                    page_num=r.page_num
                ))
            chunk_records = fixed

        #print(f"INGESTING: {p}  (chunks={len(chunk_records)}) [doc_id={doc_id},topic={topic}, lang={lang}, num_pages={num_pages}]")
    else:
        # txt/md: Pure fixed-size sliding window chunking
        text, _ = read_text_file(p)
        doc_id = str(p.resolve())
        topic = fallback_topic
        lang = fallback_lang

        chunks = fixed_size_chunks_with_overlap(
            text,
            chunk_chars=1600,
            overlap_chars=200,
            min_chars=200
        )

        chunk_records = [
            ChunkRecord(
                doc_id=doc_id,
                chunk_id=str(i),
                content=ch,
                topic=topic,
                source=str(p),
                lang=lang,
            )
            for i, ch in enumerate(chunks)
        ]


        #print(f"INGESTING: {p}  (chunks={len(chunk_records)}) [doc_id={doc_id},topic={topic}, lang={lang}]")

    existing = get_existing_fingerprint(doc_id)
    if existing == fp:
        print(f"SKIP (already ingested, unchanged): {p}")
        return

    #print(f"  fingerprint: {fp} (existing: {existing})")
    # With FK: doc must exist first
    upsert_document_registry(doc_id=doc_id, source_path=str(p), fingerprint=fp, num_pages=num_pages if p.suffix.lower() == ".pdf" else None)
    insert_or_replace_chunks_for_doc(chunk_records)

    print(f"INGESTED: {p}  (chunks={len(chunk_records)}), num_pages={num_pages}")


def main() -> None:
    ensure_doc_registry_table()

    raw_path = input("Enter a file path or folder path to ingest: ").strip().strip('"\'')
    if not raw_path:
        print("No path provided. Exiting.")
        return

    path = Path(raw_path).expanduser()
    if not path.exists():
        print(f"Path does not exist: {path}")
        return

    topic = input("Optional topic label (press Enter to skip): ").strip() or None
    lang  = input("Optional language code (e.g. 'es', press Enter to skip): ").strip() or None

    files = list(iter_files(path))
    if not files:
        print(f"No supported files found under {path}. Supported: {sorted(SUPPORTED_EXTS)}")
        return

    print(f"Found {len(files)} file(s) to consider.")
    for p in files:
        try:
            ingest_one_file(p, fallback_topic=topic, fallback_lang=lang)
        except Exception as exc:
            print(f"[ERR] {p}: {exc}")

    print("Done.")


if __name__ == "__main__":
    main()
