from dataclasses import dataclass
from psycopg2.extras import execute_values
from .db import get_conn
from .embeddings import embed_texts
from .config import settings

"""
Module for handling chunk records and database operations related to them.
This assumes you already have text extracted (from PDF/HTML/etc.)."""

@dataclass
class ChunkRecord:
    doc_id: str
    chunk_id: str
    content: str
    topic: str | None = None
    source: str | None = None
    lang: str | None = None
    page_num: int | None = None

def upsert_chunks(chunks: list[ChunkRecord]) -> None:
    # @TODO This currently performs a merge-style upsert by (doc_id, chunk_id),
    # not a full replace of all chunks for a document. If a document is
    # re-ingested with fewer or different chunk_ids, stale rows for that doc_id
    # can remain in rag_chunks unless they are deleted first in the same transaction.
    if not chunks:
        return

    texts = [c.content for c in chunks]
    vectors = embed_texts(settings.embed_model_name, texts)

    rows = []
    for c, v in zip(chunks, vectors):
        rows.append((
            c.doc_id,
            c.chunk_id,
            c.topic,
            c.source,
            c.lang,
            c.page_num,
            c.content,
            v
        ))

    sql = """
    INSERT INTO rag_chunks (doc_id, chunk_id, topic, source, lang, page_num, content, embedding)
    VALUES %s
    ON CONFLICT (doc_id, chunk_id)
    DO UPDATE SET
      topic = EXCLUDED.topic,
      source = EXCLUDED.source,
      lang = EXCLUDED.lang,
      page_num = EXCLUDED.page_num,
      content = EXCLUDED.content,
      embedding = EXCLUDED.embedding;
    """

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows, page_size=500)
        conn.commit()
    finally:
        conn.close()

    # @TODO This delete helper appears to be the intended cleanup step for
    # document re-ingestion, but it is currently nested inside upsert_chunks()
    def delete_chunks_for_doc(doc_id: str) -> None:
        conn = get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM rag_chunks WHERE doc_id = %s;", (doc_id,))
            conn.commit()
        finally:
            conn.close()
