"""Retrieval semantico, lexico e hibrido para una comparacion controlada.

Este modulo es una copia independiente de ``app/retrieval_revised.py``. No
modifica el modulo original ni el esquema de PostgreSQL. Todas las consultas
son SELECT sobre ``rag_chunks``.

La comparacion esta disenada como una ablacion:

* ``search`` usa solo pgvector.
* ``lexical_search`` usa solo PostgreSQL Full-Text Search.
* ``hybrid_search`` fusiona exactamente esas dos ramas mediante RRF.

La rama lexica independiente y la rama lexica hibrida comparten ``ts_config``
y la misma expresion FTS. De este modo, cualquier diferencia del hibrido se
debe a la fusion, no a deteccion de idioma o a consultas lexicas distintas.
"""

from dataclasses import dataclass
from langdetect import DetectorFactory, LangDetectException, detect

from .config import settings
from .db import get_conn
from .embeddings import embed_query

# langdetect puede producir resultados distintos en textos breves o ambiguos.
# Fijar la semilla hace reproducible la detección durante la evaluación.
DetectorFactory.seed = 0

# langdetect puede producir resultados distintos en textos breves o ambiguos.
# Fijar la semilla hace reproducible la detección durante la evaluación.
DetectorFactory.seed = 0

# Idiomas soportados por la evaluacion. Se usan para filtrar los resultados
# de langdetect y para seleccionar la configuracion FTS correspondiente.
SUPPORTED_LANGS = {
    "en": "english",
    "es": "spanish",
}


# Intersección entre los códigos producidos por langdetect y las
# configuraciones disponibles en el PostgreSQL de este proyecto.
LANGDETECT_TO_POSTGRES: dict[str, str] = {
    "ar": "arabic",
    "ca": "catalan",
    "da": "danish",
    "de": "german",
    "el": "greek",
    "en": "english",
    "es": "spanish",
    "et": "estonian",
    "fi": "finnish",
    "fr": "french",
    "hi": "hindi",
    "hu": "hungarian",
    "id": "indonesian",
    "it": "italian",
    "lt": "lithuanian",
    "ne": "nepali",
    "nl": "dutch",
    "no": "norwegian",
    "pt": "portuguese",
    "ro": "romanian",
    "ru": "russian",
    "sv": "swedish",
    "ta": "tamil",
    "tr": "turkish",
}

@dataclass(frozen=True)
class LexicalLanguageResolution:
    """Resultado auditable de la selección de configuración FTS."""

    detected_language: str | None
    ts_config: str
    language_supported: bool
    fallback_to_simple: bool
    note: str


@dataclass
class RetrievedChunk:
    """Resultado comun con campos observables para auditar el ranking."""

    doc_id: str
    chunk_id: str
    content: str
    distance: float | None = None
    topic: str | None = None
    source: str | None = None
    lang: str | None = None
    page_num: int | None = None
    score: float | None = None
    semantic_rank: int | None = None
    lexical_rank: int | None = None
    semantic_distance: float | None = None
    lexical_score: float | None = None
    ts_config: str | None = None
    detected_language: str | None = None
    language_supported: bool | None = None
    fallback_to_simple: bool | None = None
    language_note: str | None = None

    @property
    def debug_metric(self) -> tuple[str, float] | None:
        if self.distance is not None and self.score is not None:
            raise ValueError("Chunk has both distance and score")
        if self.distance is not None:
            return ("distance", self.distance)
        if self.score is not None:
            return ("score", self.score)
        return None

def resolve_lexical_language(query: str) -> LexicalLanguageResolution:
    """
    Detecta el idioma de la consulta, comprueba si está soportado por el RAG y selecciona la configuración FTS.

    - Si langdetect devuelve un idioma soportado, usa su configuración.
    - Si no está soportado o la evaluación no lo soporta (SUPPORTED_LANGUAGES), utiliza "simple" y registra una nota.
    - Si no se puede detectar, también utiliza "simple".
    """
    clean_query = query.strip()

    if not clean_query:
        return LexicalLanguageResolution(
            detected_language=None,
            ts_config="simple",
            language_supported=False,
            fallback_to_simple=True,
            note=(
                "Consulta vacía: no se pudo detectar el idioma; "
                "la búsqueda léxica usa 'simple'."
            ),
        )

    try:
        detected_language = detect(clean_query)
    except LangDetectException:
        return LexicalLanguageResolution(
            detected_language=None,
            ts_config="simple",
            language_supported=False,
            fallback_to_simple=True,
            note=(
                "langdetect no pudo identificar el idioma; "
                "la búsqueda léxica usa 'simple'."
            ),
        )

    if detected_language not in SUPPORTED_LANGS:
        return LexicalLanguageResolution(
            detected_language=detected_language,
            ts_config="simple",
            language_supported=False,
            fallback_to_simple=True,
            note=(
                f"El idioma '{detected_language}' detectado por langdetect "
                "no está soportado por la evaluación; "
                "la búsqueda léxica usa 'simple'."
            ),
        )

    postgres_config = LANGDETECT_TO_POSTGRES.get(detected_language)

    if postgres_config is None:
        return LexicalLanguageResolution(
            detected_language=detected_language,
            ts_config="simple",
            language_supported=False,
            fallback_to_simple=True,
            note=(
                f"El idioma '{detected_language}' detectado por langdetect "
                "no tiene una configuración PostgreSQL soportada; "
                "la búsqueda léxica usa 'simple'."
            ),
        )

    return LexicalLanguageResolution(
        detected_language=detected_language,
        ts_config=postgres_config,
        language_supported=True,
        fallback_to_simple=False,
        note="",
    )


def _positive_k(value: int | None, default: int, name: str) -> int:
    resolved = default if value is None else value
    if resolved < 1:
        raise ValueError(f"{name} must be >= 1")
    return resolved


def _candidate_depth(top_k: int, candidate_k: int | None) -> int:
    """El pool de cada rama nunca puede ser menor que la salida final."""
    depth = max(top_k, 100) if candidate_k is None else candidate_k
    if depth < top_k:
        raise ValueError("candidate_k must be >= top_k")
    return depth


def _validate_ts_config(ts_config: str) -> str:
    """Evita configuraciones vacias; PostgreSQL valida el regconfig real."""
    resolved = ts_config.strip()
    if not resolved:
        raise ValueError("ts_config must not be empty")
    return resolved


def search(
    query: str,
    top_k: int | None = None,
    topic: str | None = None,
) -> list[RetrievedChunk]:
    """Recupera vecinos semanticos ordenados por distancia coseno."""
    k = _positive_k(top_k, settings.top_k, "top_k")
    q_emb = embed_query(settings.embed_model_name, query)

    conditions: list[str] = []
    params: dict[str, object] = {"emb": q_emb, "k": k}
    if topic is not None:
        conditions.append("topic = %(topic)s")
        params["topic"] = topic
    where = "WHERE " + " AND ".join(conditions) if conditions else ""

    sql = f"""
        SELECT doc_id, chunk_id, content, topic, source, lang, page_num,
               embedding <=> %(emb)s::vector AS semantic_distance
        FROM rag_chunks
        {where}
        ORDER BY semantic_distance ASC, doc_id ASC, chunk_id ASC
        LIMIT %(k)s;
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [
            RetrievedChunk(
                doc_id=doc_id,
                chunk_id=chunk_id,
                content=content,
                topic=row_topic,
                source=source,
                lang=row_lang,
                page_num=page_num,
                distance=float(distance),
                semantic_distance=float(distance),
                semantic_rank=rank,
            )
            for rank, (
                doc_id, chunk_id, content, row_topic, source, row_lang,
                page_num, distance,
            ) in enumerate(rows, start=1)
        ]
    finally:
        conn.close()


def lexical_search(
    query: str,
    top_k: int | None = None,
    topic: str | None = None,
    language_resolution: LexicalLanguageResolution | None = None,
) -> list[RetrievedChunk]:
    """Recupera chunks exclusivamente mediante PostgreSQL Full-Text Search."""
    k = _positive_k(top_k, settings.top_k, "top_k")
    resolution = (
        language_resolution
        if language_resolution is not None
        else resolve_lexical_language(query)
    )

    config = _validate_ts_config(resolution.ts_config)
    conditions = ["document @@ tsquery"]
    params: dict[str, object] = {
        "query": query,
        "k": k,
        "ts_config": config,
    }
    if topic is not None:
        conditions.append("topic = %(topic)s")
        params["topic"] = topic

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
                doc_id=doc_id,
                chunk_id=chunk_id,
                content=content,
                topic=row_topic,
                source=source,
                lang=row_lang,
                page_num=page_num,
                score=float(score),
                lexical_score=float(score),
                lexical_rank=rank,
                ts_config=config,
                detected_language=resolution.detected_language,
                language_supported=resolution.language_supported,
                fallback_to_simple=resolution.fallback_to_simple,
                language_note=resolution.note,
            )
            for rank, (
                doc_id, chunk_id, content, row_topic, source, row_lang,
                page_num, score,
            ) in enumerate(rows, start=1)
        ]
    finally:
        conn.close()


def hybrid_search(
    query: str,
    top_k: int | None = None,
    topic: str | None = None,
    rrf_k: int = 60,
    candidate_k: int | None = None,
    language_resolution: LexicalLanguageResolution | None = None,
    semantic_weight: float = 1.0,
    lexical_weight: float = 1.0,
) -> list[RetrievedChunk]:
    """Fusiona las ramas semantic y lexical mediante RRF ponderado.

    Cada rama recupera ``candidate_k`` candidatos antes de la fusion. La salida
    contiene como maximo ``top_k`` elementos y conserva ambos rangos y scores.
    """
    k = _positive_k(top_k, settings.top_k, "top_k")
    depth = _candidate_depth(k, candidate_k)
    resolution = (
        language_resolution
        if language_resolution is not None
        else resolve_lexical_language(query)
    )

    config = _validate_ts_config(resolution.ts_config)
    if rrf_k < 1:
        raise ValueError("rrf_k must be >= 1")
    if semantic_weight < 0 or lexical_weight < 0:
        raise ValueError("RRF weights must be >= 0")
    if semantic_weight == 0 and lexical_weight == 0:
        raise ValueError("At least one RRF weight must be > 0")

    semantic_conditions: list[str] = []
    lexical_conditions = ["document @@ tsquery"]
    params: dict[str, object] = {
        "emb": embed_query(settings.embed_model_name, query),
        "query": query,
        "top_k": k,
        "candidate_k": depth,
        "rrf_k": rrf_k,
        "ts_config": config,
        "semantic_weight": semantic_weight,
        "lexical_weight": lexical_weight,
    }
    if topic is not None:
        semantic_conditions.append("topic = %(topic)s")
        lexical_conditions.append("topic = %(topic)s")
        params["topic"] = topic

    semantic_where = (
        "WHERE " + " AND ".join(semantic_conditions)
        if semantic_conditions else ""
    )
    lexical_where = "WHERE " + " AND ".join(lexical_conditions)

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
            SELECT doc_id, chunk_id,
                   ts_rank_cd(document, tsquery) AS lexical_score
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
                   s.semantic_rank,
                   l.lexical_rank,
                   s.semantic_distance,
                   l.lexical_score,
                   COALESCE(%(semantic_weight)s / (%(rrf_k)s + s.semantic_rank), 0) +
                   COALESCE(%(lexical_weight)s / (%(rrf_k)s + l.lexical_rank), 0) AS rrf_score
            FROM semantic s
            FULL OUTER JOIN lexical l USING (doc_id, chunk_id)
        )
        SELECT f.doc_id, f.chunk_id, c.content, c.topic, c.source, c.lang,
               c.page_num, f.rrf_score, f.semantic_rank, f.lexical_rank,
               f.semantic_distance, f.lexical_score
        FROM fused f
        JOIN rag_chunks c USING (doc_id, chunk_id)
        ORDER BY f.rrf_score DESC,
                 f.semantic_rank ASC NULLS LAST,
                 f.lexical_rank ASC NULLS LAST,
                 f.doc_id ASC,
                 f.chunk_id ASC
        LIMIT %(top_k)s;
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        return [
            RetrievedChunk(
                doc_id=doc_id,
                chunk_id=chunk_id,
                content=content,
                topic=row_topic,
                source=source,
                lang=row_lang,
                page_num=page_num,
                score=float(score),
                semantic_rank=semantic_rank,
                lexical_rank=lexical_rank,
                semantic_distance=(
                    float(semantic_distance)
                    if semantic_distance is not None else None
                ),
                lexical_score=(
                    float(lexical_score) if lexical_score is not None else None
                ),
                ts_config=config,
                detected_language=resolution.detected_language,
                language_supported=resolution.language_supported,
                fallback_to_simple=resolution.fallback_to_simple,
                language_note=resolution.note,
            )
            for (
                doc_id, chunk_id, content, row_topic, source, row_lang,
                page_num, score, semantic_rank, lexical_rank,
                semantic_distance, lexical_score,
            ) in rows
        ]
    finally:
        conn.close()
