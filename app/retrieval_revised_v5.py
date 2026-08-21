"""Retrieval semantico, lexico e hibrido para una comparacion controlada.

Este modulo es una copia independiente de ``app/retrieval_revised.py``. No
modifica el modulo original ni el esquema de PostgreSQL. Todas las consultas
son SELECT sobre ``rag_chunks``.

La comparacion esta disenada como una ablacion:

* ``search`` usa solo pgvector.
* ``lexical_search`` usa solo PostgreSQL Full-Text Search.
* ``hybrid_search`` fusiona exactamente esas dos ramas mediante RRF.

La rama lexica independiente y la rama lexica hibrida comparten ``ts_config``
y la misma recuperacion en dos etapas: primero una consulta AND estricta y,
si no completa el pool solicitado, un fallback OR. De este modo, la comparacion
sigue siendo una ablacion controlada y cada candidato conserva su procedencia.
"""

from dataclasses import dataclass
from lingua import Language, LanguageDetectorBuilder

from .config import settings
from .db import get_conn
from .embeddings import embed_query

# Detector local y sin llamadas a servicios externos.
# Se limita el conjunto de lenguas para mejorar la clasificación
# de preguntas cortas y permitir identificar las confusiones observadas.
LANGUAGE_DETECTOR = (
    LanguageDetectorBuilder
    .from_languages(
        Language.SPANISH,
        Language.ENGLISH,
        Language.CATALAN,
        Language.FRENCH,
        Language.PORTUGUESE,
    )
    .build()
)

# Únicamente estas lenguas tienen una configuración PostgreSQL
# admitida actualmente por el retrieval.
SUPPORTED_LANGS = {
    "es": "spanish",
    "en": "english",
}

@dataclass(frozen=True)
class LexicalLanguageResolution:
    """Resultado auditable de la selección de configuración FTS."""

    detected_language: str | None
    postgres_config: str
    supported: bool
    fallback_to_simple: bool
    note: str
    detector_name: str
    confidence: float | None
    confidence_margin: float | None


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

    # Etapa que produjo el candidato lexico. Los valores observables son
    # "and" para la busqueda estricta y "or_fallback" para la relajada.
    lexical_query_mode: str | None = None

    # Metadatos de diagnostico repetidos en cada resultado. Permiten reconstruir
    # cuantos candidatos aporto cada etapa despues de exportar a CSV.
    lexical_and_candidates: int | None = None
    lexical_or_candidates_added: int | None = None
    lexical_or_fallback_used: bool | None = None

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
    """Detecta localmente el idioma y selecciona la configuración FTS."""

    clean_query = (query or "").strip()

    if not clean_query:
        return LexicalLanguageResolution(
            detected_language=None,
            postgres_config="simple",
            supported=False,
            fallback_to_simple=True,
            note="Empty query; lexical search defaults to PostgreSQL simple.",
            detector_name="lingua",
            confidence=None,
            confidence_margin=None,
        )

    detected = LANGUAGE_DETECTOR.detect_language_of(clean_query)
    confidence_values = (
        LANGUAGE_DETECTOR.compute_language_confidence_values(clean_query)
    )

    if detected is None:
        return LexicalLanguageResolution(
            detected_language=None,
            postgres_config="simple",
            supported=False,
            fallback_to_simple=True,
            note="Lingua could not determine the language; defaulting to simple.",
            detector_name="lingua",
            confidence=None,
            confidence_margin=None,
        )

    detected_code = detected.iso_code_639_1.name.lower()

    confidence = next(
        (
            item.value
            for item in confidence_values
            if item.language == detected
        ),
        None,
    )

    confidence_margin = (
        confidence_values[0].value - confidence_values[1].value
        if len(confidence_values) >= 2
        else None
    )

    postgres_config = SUPPORTED_LANGS.get(detected_code)

    if postgres_config is None:
        return LexicalLanguageResolution(
            detected_language=detected_code,
            postgres_config="simple",
            supported=False,
            fallback_to_simple=True,
            note=(
                f"Language '{detected_code}' is not supported by the "
                "retrieval configuration; defaulting to simple."
            ),
            detector_name="lingua",
            confidence=confidence,
            confidence_margin=confidence_margin,
        )

    return LexicalLanguageResolution(
        detected_language=detected_code,
        postgres_config=postgres_config,
        supported=True,
        fallback_to_simple=False,
        note=f"Using PostgreSQL '{postgres_config}' configuration.",
        detector_name="lingua",
        confidence=confidence,
        confidence_margin=confidence_margin,
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


def _execute_lexical_stage(
    conn,
    *,
    query: str,
    limit: int,
    topic: str | None,
    ts_config: str,
    query_mode: str,
) -> list[tuple]:
    """Ejecuta una etapa de PostgreSQL Full-Text Search de solo lectura.

    query_mode="and" usa websearch_to_tsquery y exige que el documento
    satisfaga la consulta estricta. query_mode="or_fallback" normaliza la
    pregunta con PostgreSQL, extrae sus lexemas y los combina con OR.

    La consulta OR no se usa como sustituta de la AND: solo sirve para completar
    el pool cuando la etapa estricta devuelve menos candidatos de los pedidos.
    """

    if query_mode == "and":
        query_ctes = """
            parsed_query AS (
                SELECT websearch_to_tsquery(
                    %(ts_config)s::regconfig,
                    %(query)s
                ) AS tsquery
            )
        """
    elif query_mode == "or_fallback":
        query_ctes = """
            normalized_terms AS (
                SELECT tsvector_to_array(
                    to_tsvector(
                        %(ts_config)s::regconfig,
                        %(query)s
                    )
                ) AS lexemes
            ),
            parsed_query AS (
                SELECT CASE
                    WHEN cardinality(lexemes) > 0
                    THEN to_tsquery(
                        %(ts_config)s::regconfig,
                        array_to_string(lexemes, ' | ')
                    )
                    ELSE NULL
                END AS tsquery
                FROM normalized_terms
            )
        """
    else:
        raise ValueError("query_mode must be 'and' or 'or_fallback'")

    topic_condition = ""
    params: dict[str, object] = {
        "query": query,
        "limit": limit,
        "ts_config": ts_config,
    }
    if topic is not None:
        topic_condition = "AND c.topic = %(topic)s"
        params["topic"] = topic

    # La configuracion FTS se pasa como parametro y se convierte a regconfig.
    # El marcador del filtro de topic se sustituye por una clausula controlada,
    # nunca por texto procedente del usuario.
    sql = """
        WITH __QUERY_CTES__
        SELECT
            c.doc_id,
            c.chunk_id,
            c.content,
            c.topic,
            c.source,
            c.lang,
            c.page_num,
            ts_rank_cd(document.vector, q.tsquery) AS lexical_score
        FROM rag_chunks AS c
        CROSS JOIN parsed_query AS q
        CROSS JOIN LATERAL (
            SELECT to_tsvector(
                %(ts_config)s::regconfig,
                c.content
            ) AS vector
        ) AS document
        WHERE q.tsquery IS NOT NULL
          AND document.vector @@ q.tsquery
          __TOPIC_CONDITION__
        ORDER BY
            lexical_score DESC,
            c.doc_id ASC,
            c.chunk_id ASC
        LIMIT %(limit)s;
    """.replace("__QUERY_CTES__", query_ctes).replace(
        "__TOPIC_CONDITION__",
        topic_condition,
    )

    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def lexical_search(
    query: str,
    top_k: int | None = None,
    topic: str | None = None,
    language_resolution: LexicalLanguageResolution | None = None,
    or_fallback_k: int = 100,
) -> list[RetrievedChunk]:
    """Recupera candidatos lexicos mediante AND y un fallback OR.

    Procedimiento:

    1. Ejecuta la consulta AND estricta con websearch_to_tsquery.
    2. Si AND devuelve menos de top_k, ejecuta la consulta OR.
    3. Conserva todos los candidatos AND antes que cualquier candidato OR.
    4. Elimina del fallback los chunks ya recuperados por AND.
    5. Detiene el resultado combinado al alcanzar top_k.

    Los scores de AND y OR no se comparan directamente porque proceden de
    tsquery diferentes. La prioridad entre etapas se define explicitamente
    mediante el orden AND -> OR y queda registrada en lexical_query_mode.
    """

    k = _positive_k(top_k, settings.top_k, "top_k")

    if or_fallback_k < 0:
        raise ValueError("or_fallback_k must be >= 0")
    
    resolution = (
        language_resolution
        if language_resolution is not None
        else resolve_lexical_language(query)
    )
    config = _validate_ts_config(resolution.postgres_config)

    conn = get_conn()
    try:
        # Primera etapa: coincidencia estricta. Si completa el pool, no se
        # ejecuta la etapa relajada.
        strict_rows = _execute_lexical_stage(
            conn,
            query=query,
            limit=k,
            topic=topic,
            ts_config=config,
            query_mode="and",
        )

        strict_keys = {
            (str(row[0]), str(row[1]))
            for row in strict_rows
        }

        relaxed_rows: list[tuple] = []

        # Número de posiciones que quedan libres después de AND.
        remaining_slots = max(k - len(strict_rows), 0)

        # OR nunca podrá añadir más candidatos que el límite configurado.
        # Por ejemplo, con top_k=100 y or_fallback_k=30:
        # - si AND devuelve 0, OR añade como máximo 30;
        # - si AND devuelve 8, OR añade como máximo 30;
        # - si AND devuelve 90, OR añade como máximo 10.
        or_candidates_requested = min(
            remaining_slots,
            or_fallback_k,
        )

        fallback_used = or_candidates_requested > 0

        if fallback_used:
            # Los primeros resultados OR pueden ser también resultados AND.
            # Se solicita margen para eliminarlos después sin perder capacidad.
            relaxed_limit = (
                or_candidates_requested + len(strict_rows)
            )

            relaxed_rows_raw = _execute_lexical_stage(
                conn,
                query=query,
                limit=relaxed_limit,
                topic=topic,
                ts_config=config,
                query_mode="or_fallback",
            )

            for row in relaxed_rows_raw:
                key = (str(row[0]), str(row[1]))

                # No se repiten chunks que ya fueron recuperados por AND.
                if key in strict_keys:
                    continue

                relaxed_rows.append(row)

                if len(relaxed_rows) >= or_candidates_requested:
                    break

        strict_count = len(strict_rows)
        relaxed_count = len(relaxed_rows)
        combined_rows = [
            (row, "and") for row in strict_rows
        ] + [
            (row, "or_fallback") for row in relaxed_rows
        ]

        results: list[RetrievedChunk] = []
        for rank, (row, query_mode) in enumerate(combined_rows, start=1):
            (
                doc_id,
                chunk_id,
                content,
                row_topic,
                source,
                row_lang,
                page_num,
                lexical_score,
            ) = row

            results.append(
                RetrievedChunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    content=content,
                    topic=row_topic,
                    source=source,
                    lang=row_lang,
                    page_num=page_num,
                    score=float(lexical_score),
                    lexical_score=float(lexical_score),
                    lexical_rank=rank,
                    ts_config=config,
                    detected_language=resolution.detected_language,
                    language_supported=resolution.supported,
                    fallback_to_simple=resolution.fallback_to_simple,
                    language_note=resolution.note,
                    lexical_query_mode=query_mode,
                    lexical_and_candidates=strict_count,
                    lexical_or_candidates_added=relaxed_count,
                    lexical_or_fallback_used=fallback_used,
                )
            )

        return results
    finally:
        # No se hace commit: las dos etapas contienen exclusivamente SELECT.
        conn.close()


def hybrid_search(
    query: str,
    top_k: int | None = None,
    topic: str | None = None,
    rrf_k: int = 60,
    candidate_k: int | None = None,
    language_resolution: LexicalLanguageResolution | None = None,
    or_fallback_k: int = 100,
    semantic_weight: float = 1.0,
    lexical_and_weight: float = 1.0,
    lexical_or_weight: float = 1.0,
) -> list[RetrievedChunk]:
    """Fusiona con RRF semantic, lexical AND y lexical OR.

    Cada rama recupera candidate_k candidatos. La funcion llama a los
    mismos retrievers publicos que se evaluan por separado, de modo que la rama
    lexical independiente y la que participa en hybrid son exactamente iguales.

    Las coincidencias AND y OR pueden recibir pesos distintos. Los tres pesos
    valen 1.0 por defecto para reproducir el comportamiento previo. El peso
    lexical aplicable a cada candidato se selecciona mediante
    ``lexical_query_mode``.
    """

    k = _positive_k(top_k, settings.top_k, "top_k")
    depth = _candidate_depth(k, candidate_k)
    if rrf_k < 1:
        raise ValueError("rrf_k must be >= 1")
    if (
        semantic_weight < 0
        or lexical_and_weight < 0
        or lexical_or_weight < 0
    ):
        raise ValueError("RRF weights must be >= 0")
    if (
        semantic_weight == 0
        and lexical_and_weight == 0
        and lexical_or_weight == 0
    ):
        raise ValueError("At least one RRF weight must be > 0")

    resolution = (
        language_resolution
        if language_resolution is not None
        else resolve_lexical_language(query)
    )

    # Estas son exactamente las mismas ramas que se comparan en el notebook.
    semantic_results = search(
        query,
        top_k=depth,
        topic=topic,
    )
    lexical_results = lexical_search(
        query,
        top_k=depth,
        topic=topic,
        language_resolution=resolution,
        or_fallback_k=or_fallback_k,
    )

    candidates: dict[
        tuple[str, str],
        dict[str, RetrievedChunk | None],
    ] = {}

    for chunk in semantic_results:
        key = (str(chunk.doc_id), str(chunk.chunk_id))
        candidates.setdefault(
            key,
            {"semantic": None, "lexical": None},
        )
        candidates[key]["semantic"] = chunk

    for chunk in lexical_results:
        key = (str(chunk.doc_id), str(chunk.chunk_id))
        candidates.setdefault(
            key,
            {"semantic": None, "lexical": None},
        )
        candidates[key]["lexical"] = chunk

    if lexical_results:
        strict_count = lexical_results[0].lexical_and_candidates
        relaxed_count = lexical_results[0].lexical_or_candidates_added
        fallback_used = lexical_results[0].lexical_or_fallback_used
    else:
        # Con top_k positivo, una lista vacia implica que AND no completo el
        # pool y que el fallback OR tampoco encontro candidatos.
        strict_count = 0
        relaxed_count = 0
        fallback_used = True

    fused_results: list[RetrievedChunk] = []
    for pair in candidates.values():
        semantic_chunk = pair["semantic"]
        lexical_chunk = pair["lexical"]
        base_chunk = semantic_chunk or lexical_chunk

        semantic_rank = (
            semantic_chunk.semantic_rank
            if semantic_chunk is not None
            else None
        )
        lexical_rank = (
            lexical_chunk.lexical_rank
            if lexical_chunk is not None
            else None
        )

        rrf_score = 0.0
        if semantic_rank is not None:
            rrf_score += semantic_weight / (rrf_k + semantic_rank)
        if lexical_rank is not None:
            # Una coincidencia AND es mas estricta que una OR. Mantener pesos
            # separados permite reducir la influencia del fallback sin perder
            # la señal de las coincidencias lexicales exactas.
            lexical_mode = lexical_chunk.lexical_query_mode
            if lexical_mode == "and":
                candidate_lexical_weight = lexical_and_weight
            elif lexical_mode == "or_fallback":
                candidate_lexical_weight = lexical_or_weight
            else:
                raise ValueError(
                    "Lexical candidates must identify 'and' or "
                    "'or_fallback' in lexical_query_mode"
                )

            rrf_score += candidate_lexical_weight / (
                rrf_k + lexical_rank
            )

        fused_results.append(
            RetrievedChunk(
                doc_id=base_chunk.doc_id,
                chunk_id=base_chunk.chunk_id,
                content=base_chunk.content,
                topic=base_chunk.topic,
                source=base_chunk.source,
                lang=base_chunk.lang,
                page_num=base_chunk.page_num,
                score=rrf_score,
                semantic_rank=semantic_rank,
                lexical_rank=lexical_rank,
                semantic_distance=(
                    semantic_chunk.semantic_distance
                    if semantic_chunk is not None
                    else None
                ),
                lexical_score=(
                    lexical_chunk.lexical_score
                    if lexical_chunk is not None
                    else None
                ),
                ts_config=resolution.postgres_config,
                detected_language=resolution.detected_language,
                language_supported=resolution.supported,
                fallback_to_simple=resolution.fallback_to_simple,
                language_note=resolution.note,
                lexical_query_mode=(
                    lexical_chunk.lexical_query_mode
                    if lexical_chunk is not None
                    else None
                ),
                lexical_and_candidates=strict_count,
                lexical_or_candidates_added=relaxed_count,
                lexical_or_fallback_used=fallback_used,
            )
        )

    # Desempates deterministas: score RRF, rango semantic, rango lexical e IDs.
    fused_results.sort(
        key=lambda chunk: (
            -float(chunk.score),
            (
                chunk.semantic_rank
                if chunk.semantic_rank is not None
                else float("inf")
            ),
            (
                chunk.lexical_rank
                if chunk.lexical_rank is not None
                else float("inf")
            ),
            str(chunk.doc_id),
            str(chunk.chunk_id),
        )
    )
    return fused_results[:k]

