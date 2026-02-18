# scripts/ingest_docs.py
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator
import pymupdf
from collections import Counter

from app.chunking import split_into_paragraphs, chunk_paragraphs, fixed_size_chunks_with_overlap
from app.ingest import ChunkRecord, upsert_chunks
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


# ------------------ Notebook-style PDF extraction ------------------
def extract_header_footer_candidates(text: str, *, top_n: int = 3, bottom_n: int = 3) -> list[str]:
    """
    Take the first and last N non-empty lines of a page as header/footer candidates.
    """
    lines = [l.strip() for l in text.splitlines() if l.strip()]
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
    Identify lines that appear on many pages → likely headers/footers.
    """
    counter = Counter()
    n_pages = len(page_texts)

    for txt in page_texts:
        candidates = extract_header_footer_candidates(txt, top_n=top_n, bottom_n=bottom_n)
        for c in candidates:
            counter[c] += 1

    # Keep lines that appear in at least X% of pages
    repeated = {
        line
        for line, count in counter.items()
        if count / n_pages >= min_freq_ratio
    }

    return repeated


def normalize_pdf_text(
    text: str,
    *,
    headers_footers: set[str] | None = None,
) -> str:
    if not text:
        return ""

    # Normalize newlines early
    text = text.replace("\r\n", "\n")
    # Split into lines
    lines = [l.rstrip() for l in text.splitlines()]
    # Remove exact-match header/footer lines
    if headers_footers:
        lines = [l for l in lines if l.strip() and l.strip() not in headers_footers]
    #Rebuild text from filtered lines (this was missing)
    text = "\n".join(lines)
    # Join hyphenation across line breaks
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Paragraph heuristics (keep)
    text = re.sub(
        r"(?<=[:\.\?\!])\n(?=\s*(?:[A-ZÁÉÍÓÚÜÑ0-9•\-–—]))",
        "\n\n",
        text
    )
    text = re.sub(r"\n(?=\s*[A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\s]{3,}\n)", "\n\n", text)
    # Flatten single newlines (keeping blank lines)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Cleanup spaces/newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

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

def extract_pdf_page_paragraph_chunks(pdf_path, meta, chunk_chars: int = 1600) -> list[ChunkRecord]:
    overlap = int(chunk_chars * 0.15)  # 15%
    records: list[ChunkRecord] = []

    with pymupdf.open(pdf_path) as doc:
        raw_texts = [doc.load_page(p).get_text("text") for p in range(doc.page_count)]
        headers_footers = detect_repeated_headers_footers(raw_texts)
        print(f"Detected {len(headers_footers)} repeated header/footer lines: {headers_footers}")

        for i in range(doc.page_count):
            page = doc.load_page(i)
            raw = page.get_text("text")
            text = normalize_pdf_text(raw, headers_footers=headers_footers)
            if len(text) < 20:
                continue

            # page label (PyMuPDF supports labels if document defines them)
            # If not defined, you can just store None.
            page_num = i + 1

            #DEBUG
            #print("len:", len(text), "n\\n:", text.count("\n"), "n\\n\\n:", text.count("\n\n"))
            #print(text[:500])

            paragraphs = split_into_paragraphs(text)
            print(f"Page {page_num}: {len(paragraphs)} paragraphs")
            print(f"paragraphs: {[p[:50] for p in paragraphs]}")
            subchunks = chunk_paragraphs(paragraphs, chunk_chars=chunk_chars, overlap=overlap)

            for j, ch in enumerate(subchunks):
                chunk_id = f"p{page_num:03d}_c{j:02d}"
                records.append(ChunkRecord(
                    doc_id=meta.doc_id,
                    chunk_id=chunk_id,
                    content=ch,
                    topic=meta.topic,
                    source=str(pdf_path),
                    lang=meta.lang,
                    page_num=page_num
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

        print(f"INGESTING: {p}  (chunks={len(chunk_records)}) [doc_id={doc_id},topic={topic}, lang={lang}, num_pages={num_pages}]")
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


        print(f"INGESTING: {p}  (chunks={len(chunk_records)}) [doc_id={doc_id},topic={topic}, lang={lang}]")

    existing = get_existing_fingerprint(doc_id)
    if existing == fp:
        print(f"SKIP (already ingested, unchanged): {p}")
        return

    print(f"  fingerprint: {fp} (existing: {existing})")
    # With FK: doc must exist first
    #upsert_document_registry(doc_id=doc_id, source_path=str(p), fingerprint=fp, num_pages=num_pages if p.suffix.lower() == ".pdf" else None)
    #upsert_chunks(chunk_records)

    print(f"INGESTED: {p}  (chunks={len(chunk_records)})")


def main():
    ensure_doc_registry_table()

    raw_path = input("Enter a file path or folder path to ingest: ").strip().strip('"').strip("'")
    if not raw_path:
        print("No path provided. Exiting.")
        return

    path = Path(raw_path).expanduser()
    if not path.exists():
        print(f"Path does not exist: {path}")
        return

    topic = input("Optional topic label (press Enter to skip): ").strip() or None
    lang = input("Optional language code (e.g., 'es', press Enter to skip): ").strip() or None

    files = list(iter_files(path))
    if not files:
        print(f"No supported files found under {path}. Supported: {sorted(SUPPORTED_EXTS)}")
        return

    print(f"Found {len(files)} files to consider.")
    for p in files:
        try:
            ingest_one_file(p, fallback_topic=topic, fallback_lang=lang)
        except Exception as e:
            print(f"[ERR] {p}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
