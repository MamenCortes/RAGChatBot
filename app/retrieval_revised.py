"""Retrieval semántico, léxico, híbrido y sensible al idioma.

Esta es una copia revisada de ``app/retrieval.py``. Mantiene los nombres públicos
``search``, ``hybrid_search`` y ``language_aware_hybrid_search`` para que el
resto de la aplicación pueda seguir llamándolos sin cambios.

Mejoras principales:
* ``candidate_k`` es independiente de ``top_k``: primero recuperamos un pool
  amplio y solo después elegimos los resultados finales.
* La búsqueda léxica es una función pública y observable.
* RRF conserva los rangos semántico y léxico para poder diagnosticar la fusión.
* El idioma se usa como preferencia con fallback, no como filtro duro que puede
  dejar la respuesta sin contexto.
* Se devuelve ``page_num`` para trazar cada resultado hasta el PDF original.
* Los desempates son estables, lo que hace reproducibles las evaluaciones.

Requisito de base de datos: ``rag_chunks`` debe contener las columnas usadas en
el proyecto original: doc_id, chunk_id, content, topic, source, lang, page_num y
embedding. Para rendimiento en producción conviene crear índices pgvector y
full-text mediante una migración separada; este módulo no modifica el esquema.
"""

from dataclasses import dataclass
from typing import Literal

from langdetect import DetectorFactory, LangDetectException, detect

from .config import settings
from .db import get_conn
from .embeddings import embed_query


# langdetect puede variar entre ejecuciones en textos ambiguos. Fijar la semilla
# hace que una misma consulta produzca siempre la misma decisión en evaluación.
DetectorFactory.seed = 0

SUPPORTED_LANGS: dict[str, str] = {
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "de": "german",
}

LanguagePolicy = Literal["prefer", "filter", "ignore"]


@dataclass
class RetrievedChunk:
    doc_id: str
    chunk_id: str
    content: str
    distance: float | None = None
    topic: str | None = None
    source: str | None = None
    lang: str | None = None
    page_num: int | None = None
    score: float | None = None

    # Campos de diagnóstico nuevos. Permiten saber por qué un resultado quedó
    # arriba sin volver a ejecutar las ramas individuales del retriever.
    semantic_rank: int | None = None
    lexical_rank: int | None = None
    semantic_distance: float | None = None
    lexical_score: float | None = None
    query_lang: str | None = None
    ts_config: str | None = None
    language_match: bool | None = None

    @property
    def debug_metric(self) -> tuple[str, float] | None:
        if self.distance is not None and self.score is not None:
            raise ValueError("Chunk has both distance and score")
        if self.distance is not None:
            return ("distance", self.distance)
        if self.score is not None:
            return ("score", self.score)
        return None


def _positive_k(value: int | None, default: int, name: str) -> int:
    """Valida límites para evitar consultas vacías o negativas por accidente."""
    resolved = default if value is None else value
    if resolved < 1:
        raise ValueError(f"{name} must be >= 1")
    return resolved


def _candidate_depth(top_k: int, candidate_k: int | None) -> int:
    """El pool nunca puede ser menor que la lista final solicitada."""
    depth = max(top_k * 10, 50) if candidate_k is None else candidate_k
    if depth < top_k:
        raise ValueError("candidate_k must be >= top_k")
    return depth


def _safe_detect_lang(query: str) -> str | None:
    """Detecta solo idiomas soportados y evita confiar en textos muy cortos."""
    clean_query = query.strip()
    if len(clean_query) < 8 or len(clean_query.split()) < 2:
        return None
    try:
        detected = detect(clean_query)
    except LangDetectException:
        return None
    return detected if detected in SUPPORTED_LANGS else None


def search(
    query: str,
    top_k: int | None = None,
    topic: str | None = None,
    lang: str | None = None,
) -> list[RetrievedChunk]:
    """Semantic retrieval mediante distancia coseno de pgvector.

    Ventaja: encuentra paráfrasis y conceptos parecidos. Limitación: puede no
    priorizar códigos, fechas o nombres exactos. ``lang`` es opcional para que
    la búsqueda semántica normal siga siendo multilingüe por defecto.
    """
    k = _positive_k(top_k, settings.top_k, "top_k")
    q_emb = embed_query(settings.embed_model_name, query)

    conditions: list[str] = []
    params: dict[str, object] = {"emb": q_emb, "k": k}
    if topic is not None:
        conditions.append("topic = %(topic)s")
        params["topic"] = topic
    if lang is not None:
        conditions.append("lang = %(lang)s")
        params["lang"] = lang
    where = "WHERE " + " AND ".join(conditions) if conditions else ""

    sql = f"""
        SELECT doc_id, chunk_id, content, topic, source, lang, page_num,
               embedding <=> %(emb)s::vector AS distance
        FROM rag_chunks
        {where}
        ORDER BY distance ASC, doc_id ASC, chunk_id ASC
        LIMIT %(k)s;
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [
            RetrievedChunk(
                doc_id=doc_id, chunk_id=chunk_id, content=content, topic=row_topic,
                source=source, lang=row_lang, page_num=page_num,
                distance=float(distance), semantic_distance=float(distance),
            )
            for doc_id, chunk_id, content, row_topic, source, row_lang, page_num, distance in rows
        ]
    finally:
        conn.close()


def lexical_search(
    query: str,
    top_k: int | None = None,
    topic: str | None = None,
    query_lang: str | None = None,
) -> list[RetrievedChunk]:
    """Búsqueda literal con PostgreSQL Full-Text Search.

    Es especialmente útil para números, códigos, nombres y expresiones exactas.
    La configuración lingüística elimina palabras vacías y reduce variantes de
    una misma palabra. Si el idioma es desconocido se usa ``simple``.
    """
    k = _positive_k(top_k, settings.top_k, "top_k")
    resolved_lang = query_lang or _safe_detect_lang(query)
    ts_config = SUPPORTED_LANGS.get(resolved_lang, "simple")

    conditions = ["document @@ websearch_to_tsquery(%(ts_config)s::regconfig, %(query)s)"]
    params: dict[str, object] = {
        "query": query, "k": k, "topic": topic, "ts_config": ts_config,
    }
    if topic is not None:
        conditions.append("topic = %(topic)s")

    # CROSS JOIN calcula la representación de la consulta y del documento una
    # sola vez por fila lógica y evita repetir expresiones largas en el ranking.
    sql = f"""
        SELECT doc_id, chunk_id, content, topic, source, lang, page_num,
               ts_rank_cd(document, tsquery) AS lexical_score
        FROM rag_chunks
        CROSS JOIN LATERAL (
            SELECT to_tsvector(%(ts_config)s::regconfig, content) AS document,
                   websearch_to_tsquery(%(ts_config)s::regconfig, %(query)s) AS tsquery
        ) fts
        WHERE {' AND '.join(conditions)}
        ORDER BY lexical_score DESC, doc_id ASC, chunk_id ASC
        LIMIT %(k)s;
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [
            RetrievedChunk(
                doc_id=doc_id, chunk_id=chunk_id, content=content, topic=row_topic,
                source=source, lang=row_lang, page_num=page_num,
                score=float(score), lexical_score=float(score),
                query_lang=resolved_lang, ts_config=ts_config,
            )
            for doc_id, chunk_id, content, row_topic, source, row_lang, page_num, score in rows
        ]
    finally:
        conn.close()


def _hybrid_query(
    *, query: str, q_emb: list[float], top_k: int, candidate_k: int,
    topic: str | None, rrf_k: int, query_lang: str | None,
    language_policy: LanguagePolicy,
) -> list[RetrievedChunk]:
    """Ejecuta las dos ramas y las fusiona con RRF en una única consulta."""
    ts_config = SUPPORTED_LANGS.get(query_lang, "simple")
    semantic_conditions: list[str] = []
    lexical_conditions = ["document @@ tsquery"]
    if topic is not None:
        semantic_conditions.append("topic = %(topic)s")
        lexical_conditions.append("topic = %(topic)s")
    if language_policy == "filter" and query_lang is not None:
        semantic_conditions.append("lang = %(query_lang)s")
        lexical_conditions.append("lang = %(query_lang)s")

    semantic_where = "WHERE " + " AND ".join(semantic_conditions) if semantic_conditions else ""
    lexical_where = "WHERE " + " AND ".join(lexical_conditions)

    # En política "prefer", lang_boost solo rompe empates/prioriza ligeramente;
    # no puede borrar evidencias válidas en otro idioma. Con "filter" reproduce
    # el comportamiento restrictivo para realizar una ablación controlada.
    language_boost = (
        "CASE WHEN c.lang = %(query_lang)s THEN %(lang_boost)s ELSE 0 END"
        if language_policy == "prefer" and query_lang is not None else "0"
    )

    sql = f"""
        WITH semantic_candidates AS (
            SELECT doc_id, chunk_id,
                   embedding <=> %(emb)s::vector AS semantic_distance
            FROM rag_chunks
            {semantic_where}
            ORDER BY semantic_distance ASC, doc_id ASC, chunk_id ASC
            LIMIT %(candidate_k)s
        ),
        semantic AS (
            SELECT *, ROW_NUMBER() OVER (
                ORDER BY semantic_distance ASC, doc_id ASC, chunk_id ASC
            ) AS semantic_rank
            FROM semantic_candidates
        ),
        lexical_candidates AS (
            SELECT doc_id, chunk_id, ts_rank_cd(document, tsquery) AS lexical_score
            FROM rag_chunks
            CROSS JOIN LATERAL (
                SELECT to_tsvector(%(ts_config)s::regconfig, content) AS document,
                       websearch_to_tsquery(%(ts_config)s::regconfig, %(query)s) AS tsquery
            ) fts
            {lexical_where}
            ORDER BY lexical_score DESC, doc_id ASC, chunk_id ASC
            LIMIT %(candidate_k)s
        ),
        lexical AS (
            SELECT *, ROW_NUMBER() OVER (
                ORDER BY lexical_score DESC, doc_id ASC, chunk_id ASC
            ) AS lexical_rank
            FROM lexical_candidates
        ),
        fused AS (
            SELECT COALESCE(s.doc_id, l.doc_id) AS doc_id,
                   COALESCE(s.chunk_id, l.chunk_id) AS chunk_id,
                   s.semantic_rank, l.lexical_rank,
                   s.semantic_distance, l.lexical_score,
                   COALESCE(%(semantic_weight)s / (%(rrf_k)s + s.semantic_rank), 0) +
                   COALESCE(%(lexical_weight)s / (%(rrf_k)s + l.lexical_rank), 0) AS rrf_score
            FROM semantic s
            FULL OUTER JOIN lexical l USING (doc_id, chunk_id)
        )
        SELECT f.doc_id, f.chunk_id, c.content, c.topic, c.source, c.lang,
               c.page_num, f.rrf_score + {language_boost} AS final_score,
               f.semantic_rank, f.lexical_rank,
               f.semantic_distance, f.lexical_score
        FROM fused f
        JOIN rag_chunks c USING (doc_id, chunk_id)
        ORDER BY final_score DESC, f.semantic_rank ASC NULLS LAST,
                 f.lexical_rank ASC NULLS LAST, f.doc_id ASC, f.chunk_id ASC
        LIMIT %(top_k)s;
    """
    params = {
        "emb": q_emb, "query": query, "top_k": top_k,
        "candidate_k": candidate_k, "topic": topic, "rrf_k": rrf_k,
        "query_lang": query_lang, "ts_config": ts_config,
        "semantic_weight": 1.0, "lexical_weight": 1.0,
        # Muy pequeño frente al score RRF: expresa preferencia sin dominar la
        # relevancia. Debe ajustarse únicamente en el conjunto de desarrollo.
        "lang_boost": 0.001,
    }
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [
            RetrievedChunk(
                doc_id=doc_id, chunk_id=chunk_id, content=content, topic=row_topic,
                source=source, lang=row_lang, page_num=page_num, score=float(score),
                semantic_rank=semantic_rank, lexical_rank=lexical_rank,
                semantic_distance=float(semantic_distance) if semantic_distance is not None else None,
                lexical_score=float(lexical_score) if lexical_score is not None else None,
                query_lang=query_lang, ts_config=ts_config,
                language_match=(row_lang == query_lang) if query_lang is not None else None,
            )
            for (doc_id, chunk_id, content, row_topic, source, row_lang, page_num,
                 score, semantic_rank, lexical_rank, semantic_distance, lexical_score) in rows
        ]
    finally:
        conn.close()


def hybrid_search(
    query: str,
    top_k: int | None = None,
    topic: str | None = None,
    rrf_k: int = 60,
    candidate_k: int | None = None,
) -> list[RetrievedChunk]:
    """Fusiona semantic y lexical retrieval mediante Reciprocal Rank Fusion."""
    k = _positive_k(top_k, settings.top_k, "top_k")
    depth = _candidate_depth(k, candidate_k)
    if rrf_k < 1:
        raise ValueError("rrf_k must be >= 1")
    return _hybrid_query(
        query=query, q_emb=embed_query(settings.embed_model_name, query),
        top_k=k, candidate_k=depth, topic=topic, rrf_k=rrf_k,
        query_lang=None, language_policy="ignore",
    )


def language_aware_hybrid_search(
    query: str,
    top_k: int | None = None,
    topic: str | None = None,
    rrf_k: int = 60,
    candidate_k: int | None = None,
    language_policy: LanguagePolicy = "prefer",
) -> list[RetrievedChunk]:
    """Hybrid retrieval que conoce el idioma sin perder cobertura por defecto.

    ``prefer`` (recomendado): favorece el idioma detectado y permite fallback.
    ``filter``: limita estrictamente al idioma; útil para experimentos, pero
    puede devolver menos de ``top_k`` si faltan chunks de ese idioma.
    ``ignore``: detecta el idioma para FTS, pero no altera el ranking por idioma.
    """
    if language_policy not in {"prefer", "filter", "ignore"}:
        raise ValueError("language_policy must be 'prefer', 'filter', or 'ignore'")
    k = _positive_k(top_k, settings.top_k, "top_k")
    depth = _candidate_depth(k, candidate_k)
    if rrf_k < 1:
        raise ValueError("rrf_k must be >= 1")
    query_lang = _safe_detect_lang(query)
    return _hybrid_query(
        query=query, q_emb=embed_query(settings.embed_model_name, query),
        top_k=k, candidate_k=depth, topic=topic, rrf_k=rrf_k,
        query_lang=query_lang, language_policy=language_policy,
    )
