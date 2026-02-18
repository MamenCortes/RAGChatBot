from dataclasses import dataclass
from .db import get_conn
from .embeddings import embed_query
from .config import settings

"""Module for retrieving relevant chunks from the database based on a query.
It uses the same embedding model as ingest.py to embed the query and performs a vector similarity search in PostgreSQL.
The retrieved chunks are returned as a list of RetrievedChunk dataclass instances, which include the content and metadata of each chunk along with its distance from the query embedding.
"""

@dataclass
class RetrievedChunk:
    doc_id: str
    chunk_id: str
    content: str
    distance: float
    topic: str | None = None
    source: str | None = None
    lang: str | None = None

def search(query: str, top_k: int | None = None, topic: str | None = None) -> list[RetrievedChunk]:
    k = top_k or settings.top_k
    q_emb = embed_query(settings.embed_model_name, query)

    base_sql = """
      SELECT doc_id, chunk_id, content, topic, source, lang,
             (embedding <=> %s::vector) AS distance
      FROM rag_chunks
    """
    where = ""
    params = [q_emb]

    if topic:
        where = " WHERE topic = %s "
        params.append(topic) # type: ignore

    sql = base_sql + where + """
      ORDER BY embedding <=> %s::vector
      LIMIT %s;
    """

    # ORDER BY needs q_emb again
    params.append(q_emb)
    params.append(k) # type: ignore

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        out = []
        for (doc_id, chunk_id, content, topic, source, lang, distance) in rows:
            out.append(RetrievedChunk(
                doc_id=doc_id,
                chunk_id=chunk_id,
                content=content,
                topic=topic,
                source=source,
                lang=lang,
                distance=float(distance),
            ))
        return out
    finally:
        conn.close()
