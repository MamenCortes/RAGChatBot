from dataclasses import dataclass
from .db import get_conn
from .embeddings import embed_query
from .config import settings
from langdetect import detect
"""Module for retrieving relevant chunks from the database based on a query.
It uses the same embedding model as ingest.py to embed the query and performs a vector similarity (cosine distance) search in PostgreSQL.
The retrieved chunks are returned as a list of RetrievedChunk dataclass instances, which include the content and metadata of each chunk along with its distance from the query embedding.´
Retrieves the top-k nearest neighbors by cosine distance.
"""

from langdetect import detect, LangDetectException

SUPPORTED_LANGS = {
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "de": "german",
}

@dataclass
class RetrievedChunk:
    doc_id: str
    chunk_id: str
    content: str
    distance: float | None = None
    topic: str | None = None
    source: str | None = None
    lang: str | None = None
    score: float | None = None  # For hybrid search, this can store the RRF score instead of distance

    @property
    def debug_metric(self) -> tuple[str, float] | None:
        if self.distance is not None and self.score is not None:
            raise ValueError("Chunk has both distance and score")

        if self.distance is not None:
            return ("distance", self.distance)

        if self.score is not None:
            return ("score", self.score)

        return None
    


def search(query: str, top_k: int | None = None, topic: str | None = None) -> list[RetrievedChunk]:
    """
    Retrieves the top-k nearest neighbors by cosine distance.
    If `topic` is provided, it filters chunks by that topic before ranking.
    """
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


def hybrid_search(query: str, top_k: int | None = None, topic: str | None = None, rrf_k: int = 60) -> list[RetrievedChunk]:
    """
    Hybrid search combining semantic (pgvector cosine) and keyword/full-text (ts_rank_cd)
    via Reciprocal Rank Fusion entirely in Postgres. No Python-side ranking needed.
    - Semantic search finds chunks with similar meaning to the query embedding.
    - Keyword/Full-text search finds chunks with similar keywords to the query.

    Reciprocal Rank Fusion (RRF) combines the ranks from both methods, giving a boost to chunks that rank well in either method. 
    The `rrf_k` parameter controls how much the rank influences the final score (lower means more influence).

    Args: 
    - `query`: The user's query string.
    - `top_k`: The number of top results to return.
    - `topic`: Optional topic filter to narrow down the search and apply metadata filtering. 
    - `rrf_k`: The k parameter for Reciprocal Rank Fusion, controlling the influence of rank on the final score.
        - It the code, this looks like: score = 1/(k + semantic_rank) + 1/(k + lexical_rank)

    Returns:
    A list of RetrievedChunk instances, each containing the document ID, chunk ID, content, topic, source, language, and a combined relevance score (distance) based on both semantic and keyword relevance.

    The `score` field of the RetrievedChunk stores the RRF score (higher = more relevant).

    Problem: keyword/full-text search usually works poorly or not at all across languages. RRF will favor same-language retrieval. 
    """
    k = top_k or settings.top_k
    q_emb = embed_query(settings.embed_model_name, query)

    #Build where conditions for both semantic and full-text search based on the presence of a topic filter.
    semantic_conditions = []
    fulltext_conditions = ["to_tsvector('simple', content) @@ query"]

    if topic:
        semantic_conditions.append("topic = %(topic)s")
        fulltext_conditions.append("topic = %(topic)s")

    semantic_where = (
        "WHERE " + " AND ".join(semantic_conditions)
        if semantic_conditions else ""
    )

    fulltext_where = "WHERE " + " AND ".join(fulltext_conditions)

    #The query uses three Common Table Expressions (CTEs):
    #1. `semantic`: Retrieves chunks ranked by cosine similarity to the query embedding using pgvector. It retrieves more than `k` results (e.g., `k*4`) to allow for a good mix in the final fusion step.
    #2. `fulltext`: Retrieves chunks ranked by full-text relevance to the query. It also retrieves more than `k` results for the same reason.
    #   - ts_rank_cd(): computes a relevance score for how well the content matches the tsquery. It considers term coverage and proximity better than plain substring matching
    #3. `rrf`: Combines the ranks from both methods using Reciprocal Rank Fusion to compute a final relevance score.
    sql = f"""
      WITH semantic AS (
        SELECT doc_id, chunk_id,
               ROW_NUMBER() OVER (ORDER BY embedding <=> %(emb)s::vector) AS rank
        FROM rag_chunks
        {semantic_where}
        ORDER BY embedding <=> %(emb)s::vector
        LIMIT %(k)s * 4
      ),
      fulltext AS (
        SELECT doc_id, chunk_id,
               ROW_NUMBER() OVER (ORDER BY ts_rank_cd(to_tsvector('simple', content), query) DESC) AS rank
        FROM rag_chunks, plainto_tsquery('simple', %(query)s) query
        {fulltext_where}
        ORDER BY ts_rank_cd(to_tsvector('simple', content), query) DESC
        LIMIT %(k)s * 4
      ),
      rrf AS (
        SELECT COALESCE(s.doc_id, f.doc_id) AS doc_id,
               COALESCE(s.chunk_id, f.chunk_id) AS chunk_id,
               COALESCE(1.0 / (%(rrf_k)s + s.rank), 0) +
               COALESCE(1.0 / (%(rrf_k)s + f.rank), 0) AS score
        FROM semantic s
        FULL OUTER JOIN fulltext f USING (doc_id, chunk_id)
      )
      SELECT r.doc_id, r.chunk_id, c.content, c.topic, c.source, c.lang, r.score
      FROM rrf r
      JOIN rag_chunks c USING (doc_id, chunk_id)
      ORDER BY r.score DESC
      LIMIT %(k)s;
    """

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, {"emb": q_emb, "query": query, "k": k, "rrf_k": rrf_k, "topic": topic})
            rows = cur.fetchall()
        return [
            RetrievedChunk(doc_id=doc_id, chunk_id=chunk_id, content=content,
                           topic=topic, source=source, lang=lang, score=float(score))
            for (doc_id, chunk_id, content, topic, source, lang, score) in rows
        ]
    finally:
        conn.close()


def _safe_detect_lang(query: str) -> str | None:
    query = query.strip()

    # Very short queries are unreliable: "ok", "si", "no", etc.
    if len(query) < 4:
        return None

    try:
        lang = detect(query)
    except LangDetectException:
        return None

    return lang if lang in SUPPORTED_LANGS else None

def language_aware_hybrid_search(query: str, top_k: int | None = None, topic: str | None = None, rrf_k: int = 60) -> list[RetrievedChunk]:
    """
    A wrapper around `hybrid_search` that adds a language filter to prioritize chunks in the user's language.
    If `user_lang` is provided, it filters results to that language first before applying the hybrid search logic. 
    This helps ensure that the retrieved context is in the same language as the user's query, which can improve relevance and answer quality.
    """
    """
    Hybrid search with optional language filtering.

    If language detection succeeds, both semantic and full-text retrieval
    are restricted to chunks in the detected language.

    If language detection fails, the function falls back to normal hybrid
    retrieval behavior without a language filter.
    """
    k = top_k or settings.top_k
    q_emb = embed_query(settings.embed_model_name, query)

    query_lang = _safe_detect_lang(query)
    ts_config = SUPPORTED_LANGS[query_lang] if query_lang is not None else "simple"

    semantic_conditions = []
    fulltext_conditions = [
        f"to_tsvector('{ts_config}', content) @@ query"
    ]

    if query_lang is not None:
        semantic_conditions.append("lang = %(query_lang)s")
        fulltext_conditions.append("lang = %(query_lang)s")

    if topic:
        semantic_conditions.append("topic = %(topic)s")
        fulltext_conditions.append("topic = %(topic)s")

    semantic_where = (
        "WHERE " + " AND ".join(semantic_conditions)
        if semantic_conditions
        else ""
    )

    fulltext_where = "WHERE " + " AND ".join(fulltext_conditions)

    #The query uses three Common Table Expressions (CTEs):
    #1. `semantic`: Retrieves chunks ranked by cosine similarity to the query embedding using pgvector. It retrieves more than `k` results (e.g., `k*4`) to allow for a good mix in the final fusion step.
    #2. `fulltext`: Retrieves chunks ranked by full-text relevance to the query. It also retrieves more than `k` results for the same reason.
    #   - ts_rank_cd(): computes a relevance score for how well the content matches the tsquery. It considers term coverage and proximity better than plain substring matching
    #3. `rrf`: Combines the ranks from both methods using Reciprocal Rank Fusion to compute a final relevance score.
    sql = f"""
      WITH semantic AS (
        SELECT doc_id, chunk_id,
               ROW_NUMBER() OVER (ORDER BY embedding <=> %(emb)s::vector) AS rank
        FROM rag_chunks
        {semantic_where}
        ORDER BY embedding <=> %(emb)s::vector
        LIMIT %(k)s * 4
      ),
      fulltext AS (
        SELECT doc_id, chunk_id,
               ROW_NUMBER() OVER (ORDER BY ts_rank_cd(to_tsvector('{ts_config}', content), query) DESC) AS rank
        FROM rag_chunks, plainto_tsquery('{ts_config}', %(query)s) query
        {fulltext_where}
        ORDER BY ts_rank_cd(to_tsvector('{ts_config}', content), query) DESC
        LIMIT %(k)s * 4
      ),
      rrf AS (
        SELECT COALESCE(s.doc_id, f.doc_id) AS doc_id,
               COALESCE(s.chunk_id, f.chunk_id) AS chunk_id,
               COALESCE(1.0 / (%(rrf_k)s + s.rank), 0) +
               COALESCE(1.0 / (%(rrf_k)s + f.rank), 0) AS score
        FROM semantic s
        FULL OUTER JOIN fulltext f USING (doc_id, chunk_id)
      )
      SELECT r.doc_id, r.chunk_id, c.content, c.topic, c.source, c.lang, r.score
      FROM rrf r
      JOIN rag_chunks c USING (doc_id, chunk_id)
      ORDER BY r.score DESC
      LIMIT %(k)s;
    """

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, {"emb": q_emb, "query": query, "k": k, "rrf_k": rrf_k, "topic": topic, "query_lang": query_lang})
            rows = cur.fetchall()
        return [
            RetrievedChunk(doc_id=doc_id, chunk_id=chunk_id, content=content,
                           topic=topic, source=source, lang=lang, score=float(score))
            for (doc_id, chunk_id, content, topic, source, lang, score) in rows
        ]
    finally:
        conn.close()
