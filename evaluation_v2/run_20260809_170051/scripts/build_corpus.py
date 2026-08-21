from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

from pypdf import PdfReader


RUN_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = RUN_DIR.parent.parent
LOG_DIR = RUN_DIR / "logs"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def norm(text: str) -> str:
    value = unicodedata.normalize("NFKD", text.casefold())
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", value).strip()


STOPWORDS = set(
    "a al algo algunas algunos ante antes como con contra cual cuales cuando de del desde donde dos el ella ellas "
    "ellos en entre era es esa esas ese esos esta estas este estos fue ha hay la las lo los mas me mi muy no o para "
    "pero por porque que qué se sin sobre son su sus tras un una unas unos y ya the a an of to in on for with is are "
    "be by from or and as at this that these those what which how when where patient patients treatment cancer".split()
)


def tokens(text: str) -> list[str]:
    return [
        token for token in re.findall(r"[a-z0-9]+", norm(text))
        if len(token) >= 2 and token not in STOPWORDS
    ]


def number_tokens(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:[.,]\d+)?\b", norm(text)))


def coverage_score(query: str, evidence: str) -> tuple[float, int, bool]:
    query_tokens = set(tokens(query))
    if not query_tokens:
        return 0.0, 0, not number_tokens(query)
    evidence_tokens = set(tokens(evidence))
    common = len(query_tokens & evidence_tokens)
    numbers = number_tokens(query)
    return common / len(query_tokens), common, numbers.issubset(number_tokens(evidence))


def sentence_windows(text: str) -> list[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\s*[•●▪]\s*", clean) if part.strip()]
    windows = list(sentences)
    for width in (2, 3):
        windows.extend(" ".join(sentences[i:i + width]) for i in range(max(0, len(sentences) - width + 1)))
    return [window[:900] for window in windows if window]


def best_snippet(text: str, query: str, max_chars: int = 700) -> tuple[str, float]:
    candidates = sentence_windows(text)
    if not candidates:
        return "", 0.0
    scored: list[tuple[float, float, int, str]] = []
    query_norm = norm(query)
    for candidate in candidates:
        cov, common, numbers_ok = coverage_score(query, candidate)
        sequence = SequenceMatcher(None, query_norm[:1000], norm(candidate)[:1000]).ratio()
        exact_bonus = 0.5 if query_norm and query_norm in norm(candidate) else 0.0
        score = cov + min(common, 8) * 0.015 + (0.08 if numbers_ok else -0.08) + sequence * 0.15 + exact_bonus
        scored.append((score, cov, -len(candidate), candidate))
    _, cov, _, winner = max(scored)
    if len(winner) <= max_chars:
        return winner, cov
    # Center a bounded excerpt on the first informative query term.
    lower = norm(winner)
    positions = [lower.find(token) for token in tokens(query) if lower.find(token) >= 0]
    start = max(0, (min(positions) if positions else 0) - max_chars // 3)
    return winner[start:start + max_chars].strip(), cov


def parse_literal(value: str) -> object:
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []


questions = read_csv(RUN_DIR / "questions.csv")
claims = read_csv(RUN_DIR / "claims.csv")
gold_rows = read_csv(PROJECT_ROOT / "eval" / "gold_references.csv")
db_chunks = read_csv(LOG_DIR / "db_chunks.csv")
db_schema = read_csv(LOG_DIR / "db_schema.csv")
db_profile_raw = read_csv(LOG_DIR / "db_profile_before.csv")[0]
lexical_rows = read_csv(LOG_DIR / "lexical_pool.csv")

question_by_id = {row["question_id"]: row for row in questions}
claims_by_question: dict[str, list[dict[str, str]]] = defaultdict(list)
for row in claims:
    claims_by_question[row["question_id"]].append(row)
gold_by_question: dict[str, list[dict[str, str]]] = defaultdict(list)
for row in gold_rows:
    gold_by_question[row["question_id"]].append(row)

chunk_by_key = {(row["doc_id"], row["chunk_id"]): row for row in db_chunks}
chunks_by_doc_page: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
doc_source: dict[str, str] = {}
stem_to_doc: dict[str, str] = {}
for row in db_chunks:
    try:
        page = int(row["page_num"])
    except (TypeError, ValueError):
        page = -1
    chunks_by_doc_page[(row["doc_id"], page)].append(row)
    doc_source[row["doc_id"]] = row["source"]
    if row["source"]:
        stem_to_doc[Path(row["source"]).stem.casefold()] = row["doc_id"]
        stem_to_doc[Path(row["source"]).stem.casefold().replace("_", "-")] = row["doc_id"]


def resolve_doc_id(value: str) -> str:
    stem = Path(value).stem.casefold()
    if stem in stem_to_doc:
        return stem_to_doc[stem]
    for doc_id in doc_source:
        if doc_id.casefold() == stem or doc_id.casefold().replace("_", "-") == stem.replace("_", "-"):
            return doc_id
    return Path(value).stem


# Extract every PDF page directly; extraction failures remain explicit.
pdf_pages: list[dict[str, object]] = []
pdf_inventory: list[dict[str, object]] = []
pdf_errors: list[dict[str, str]] = []
for pdf_path in sorted((PROJECT_ROOT / "docs").rglob("*.pdf")):
    try:
        reader = PdfReader(str(pdf_path))
        page_count = len(reader.pages)
        extracted_nonempty = 0
        doc_id = resolve_doc_id(pdf_path.name)
        for index, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception as exc:  # a page failure must not erase the rest of the document
                text = ""
                pdf_errors.append({"source": str(pdf_path), "page_num": str(index), "error": f"{type(exc).__name__}: {exc}"})
            if text.strip():
                extracted_nonempty += 1
            pdf_pages.append({
                "doc_id": doc_id,
                "source": str(pdf_path.resolve()),
                "page_num": index,
                "text": text,
                "norm": norm(text),
                "tokens": set(tokens(text)),
            })
        pdf_inventory.append({
            "source": str(pdf_path.resolve()), "doc_id": doc_id, "page_count": page_count,
            "nonempty_extracted_pages": extracted_nonempty,
        })
    except Exception as exc:
        pdf_errors.append({"source": str(pdf_path), "page_num": "", "error": f"{type(exc).__name__}: {exc}"})

pages_by_doc_page = {(str(row["doc_id"]), int(row["page_num"])): row for row in pdf_pages}

raw_pool: list[dict[str, object]] = []


def add_candidate(
    *, question_id: str, doc_id: str = "", chunk_id: str = "", page_num: object = "",
    source: str = "", lang: str = "", content: str = "", retrieval_method: str,
    retrieval_rank: object = "", retrieval_score: object = "", score_type: str = "",
    query_language: str = "", detected_language: str = "", ts_config: str = "",
    rrf_k: object = "", found_in_docs: bool = False, document_only: bool = False,
    original_gold_candidate: bool = False, claim_id: str = "", query_type: str = "question",
    ingestion_gap: bool = False, provenance_detail: str = "",
) -> None:
    raw_pool.append({
        "question_id": question_id, "candidate_id": "", "doc_id": doc_id, "chunk_id": chunk_id,
        "page_num": page_num, "source": source, "lang": lang, "content": content,
        "retrieval_method": retrieval_method, "retrieval_rank": retrieval_rank,
        "retrieval_score": retrieval_score, "score_type": score_type,
        "query_language": query_language, "detected_language": detected_language,
        "ts_config": ts_config, "rrf_k": rrf_k, "found_in_docs": str(found_in_docs).lower(),
        "document_only": str(document_only).lower(),
        "original_gold_candidate": str(original_gold_candidate).lower(),
        "claim_id": claim_id, "query_type": query_type,
        "ingestion_gap": str(ingestion_gap).lower(), "provenance_detail": provenance_detail,
    })


# Live read-only PostgreSQL lexical results.
for row in lexical_rows:
    add_candidate(
        question_id=row["question_id"], doc_id=row["doc_id"], chunk_id=row["chunk_id"],
        page_num=row["page_num"], source=row["source"], lang=row["lang"], content=row["content"],
        retrieval_method=row["retrieval_method"], retrieval_rank=row["retrieval_rank"],
        retrieval_score=row["retrieval_score"], score_type=row["score_type"],
        ts_config=row["ts_config"], claim_id=row["claim_id"], query_type=row["query_type"],
        provenance_detail="PostgreSQL FTS in forced read-only transaction",
    )


# Persisted semantic/hybrid/language-aware results. Semantic stored exact chunks;
# hybrid variants stored only doc/page pairs, so those remain unresolved references.
persisted_files = {
    "semantic_search_persisted": PROJECT_ROOT / "eval" / "retrieval_semantic_search_evaluation_results.csv",
    "hybrid_search_persisted": PROJECT_ROOT / "eval" / "retrieval_hybrid_search_evaluation_results.csv",
    "language_aware_hybrid_search_persisted": PROJECT_ROOT / "eval" / "retrieval_language_aware_hybrid_search_evaluation_results.csv",
}
for method, path in persisted_files.items():
    for row in read_csv(path):
        qid = row["question_id"]
        if method == "semantic_search_persisted" and row.get("retrieved_context"):
            contexts = parse_literal(row["retrieved_context"])
            if isinstance(contexts, list):
                for fallback_rank, item in enumerate(contexts, start=1):
                    if not isinstance(item, dict):
                        continue
                    add_candidate(
                        question_id=qid, doc_id=str(item.get("doc_id", "")), chunk_id=str(item.get("chunk_id", "")),
                        page_num=item.get("page_num", ""), source=str(item.get("source", "")),
                        lang=str(item.get("lang", "")), content=str(item.get("content", "")),
                        retrieval_method=method, retrieval_rank=item.get("rank", fallback_rank),
                        retrieval_score=item.get("distance", ""), score_type="cosine_distance",
                        provenance_detail="Existing immutable evaluation output; top-5 only",
                    )
        else:
            pairs = parse_literal(row.get("retrieved_doc_pages_ranked", ""))
            if isinstance(pairs, list):
                for rank, pair in enumerate(pairs, start=1):
                    if not isinstance(pair, (tuple, list)) or len(pair) < 2:
                        continue
                    doc_id, page = str(pair[0]), pair[1]
                    add_candidate(
                        question_id=qid, doc_id=doc_id, page_num=page,
                        source=doc_source.get(doc_id, ""), retrieval_method=method, retrieval_rank=rank,
                        score_type="rank_only", rrf_k=60,
                        provenance_detail="Existing output omitted chunk_id/content/raw branch scores",
                    )


# Gold references are candidates, not automatic evidence. Add corresponding DB chunks.
for row in gold_rows:
    qid = row["question_id"]
    doc_id = resolve_doc_id(row["expected_doc"])
    try:
        page_num = int(float(row["expected_page"]))
    except (TypeError, ValueError):
        page_num = -1
    matches = chunks_by_doc_page.get((doc_id, page_num), [])
    if matches:
        for rank, chunk in enumerate(matches, start=1):
            add_candidate(
                question_id=qid, doc_id=doc_id, chunk_id=chunk["chunk_id"], page_num=page_num,
                source=chunk["source"], lang=chunk["lang"], content=chunk["content"],
                retrieval_method="existing_gold_db_chunk", retrieval_rank=rank,
                score_type="gold_reference_candidate", original_gold_candidate=True,
                provenance_detail=f"gold_evidence_id={row['evidence_id']}; gold status not treated as judgment",
            )
    else:
        add_candidate(
            question_id=qid, doc_id=doc_id, page_num=page_num,
            source=doc_source.get(doc_id, ""), retrieval_method="existing_gold_unresolved",
            score_type="gold_reference_candidate", original_gold_candidate=True,
            provenance_detail=f"gold_evidence_id={row['evidence_id']}; no exact DB doc/page chunk",
        )


db_token_sets = {(row["doc_id"], row["chunk_id"]): set(tokens(row["content"])) for row in db_chunks}


def top_db_matches(query: str, limit: int, minimum: float) -> list[tuple[float, dict[str, str]]]:
    query_tokens = set(tokens(query))
    if not query_tokens:
        return []
    scored: list[tuple[float, int, str, str, dict[str, str]]] = []
    for chunk in db_chunks:
        common = len(query_tokens & db_token_sets[(chunk["doc_id"], chunk["chunk_id"])])
        score = common / len(query_tokens)
        if score >= minimum and common >= 2:
            scored.append((score, common, chunk["doc_id"], chunk["chunk_id"], chunk))
    scored.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
    return [(score, chunk) for score, _, _, _, chunk in scored[:limit]]


# Offline lexical pass over the immutable DB snapshot: questions and every claim.
for question in questions:
    qid = question["question_id"]
    for rank, (score, chunk) in enumerate(top_db_matches(question["question"], 15, 0.25), start=1):
        add_candidate(
            question_id=qid, doc_id=chunk["doc_id"], chunk_id=chunk["chunk_id"], page_num=chunk["page_num"],
            source=chunk["source"], lang=chunk["lang"], content=chunk["content"],
            retrieval_method="lexical_question_snapshot", retrieval_rank=rank, retrieval_score=score,
            score_type="unique_query_token_coverage", provenance_detail="Offline scan of read-only rag_chunks snapshot",
        )
    for claim in claims_by_question[qid]:
        for rank, (score, chunk) in enumerate(top_db_matches(claim["claim_text"], 8, 0.35), start=1):
            add_candidate(
                question_id=qid, doc_id=chunk["doc_id"], chunk_id=chunk["chunk_id"], page_num=chunk["page_num"],
                source=chunk["source"], lang=chunk["lang"], content=chunk["content"],
                retrieval_method="lexical_claim_snapshot", retrieval_rank=rank, retrieval_score=score,
                score_type="unique_claim_token_coverage", claim_id=claim["claim_id"], query_type="claim",
                provenance_detail="Second-pass offline scan of read-only rag_chunks snapshot",
            )


def page_score(query: str, page: dict[str, object]) -> tuple[float, int]:
    query_tokens = set(tokens(query))
    if not query_tokens:
        return 0.0, 0
    page_tokens = page["tokens"] if isinstance(page["tokens"], set) else set()
    common = len(query_tokens & page_tokens)
    return common / len(query_tokens), common


def map_document_snippet(doc_id: str, page_num: int, snippet: str) -> dict[str, str] | None:
    snippet_norm = norm(snippet)
    snippet_tokens = set(tokens(snippet))
    best: tuple[float, dict[str, str]] | None = None
    for chunk in chunks_by_doc_page.get((doc_id, page_num), []):
        chunk_norm = norm(chunk["content"])
        exact = bool(snippet_norm and (snippet_norm in chunk_norm or chunk_norm in snippet_norm))
        overlap = len(snippet_tokens & set(tokens(chunk["content"]))) / max(1, len(snippet_tokens))
        score = overlap + (1.0 if exact else 0.0)
        if best is None or score > best[0]:
            best = (score, chunk)
    if best and best[0] >= 0.85:
        return best[1]
    return None


def add_document_candidate(
    *, qid: str, page: dict[str, object], query: str, method: str, rank: int,
    score: float, claim_id: str = "", gold: bool = False, detail: str = "",
) -> None:
    snippet, snippet_cov = best_snippet(str(page["text"]), query)
    if not snippet:
        return
    doc_id, page_num = str(page["doc_id"]), int(page["page_num"])
    mapped = map_document_snippet(doc_id, page_num, snippet)
    if mapped:
        add_candidate(
            question_id=qid, doc_id=doc_id, chunk_id=mapped["chunk_id"], page_num=page_num,
            source=mapped["source"], lang=mapped["lang"], content=mapped["content"],
            retrieval_method=method, retrieval_rank=rank, retrieval_score=score,
            score_type="document_page_token_coverage", found_in_docs=True,
            original_gold_candidate=gold, claim_id=claim_id, query_type="claim" if claim_id else "question",
            provenance_detail=f"{detail}; verified snippet in PDF page; snippet_coverage={snippet_cov:.4f}",
        )
    else:
        source = str(page["source"])
        lang = Path(source).stem.split("_")[1] if len(Path(source).stem.split("_")) > 1 else ""
        add_candidate(
            question_id=qid, doc_id=doc_id, page_num=page_num, source=source, lang=lang, content=snippet,
            retrieval_method=method, retrieval_rank=rank, retrieval_score=score,
            score_type="document_page_token_coverage", found_in_docs=True, document_only=True,
            original_gold_candidate=gold, claim_id=claim_id, query_type="claim" if claim_id else "question",
            ingestion_gap=True, provenance_detail=f"{detail}; PDF text verified but no equivalent chunk confidently mapped",
        )


# Direct search across every extracted page for questions and claims.
for question in questions:
    qid = question["question_id"]
    scored_pages = []
    for page in pdf_pages:
        score, common = page_score(question["question"], page)
        if score >= 0.25 and common >= 2:
            scored_pages.append((score, common, str(page["source"]), int(page["page_num"]), page))
    scored_pages.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
    for rank, (score, _, _, _, page) in enumerate(scored_pages[:10], start=1):
        add_document_candidate(qid=qid, page=page, query=question["question"], method="document_question_search", rank=rank, score=score)
    for claim in claims_by_question[qid]:
        scored_claim_pages = []
        claim_norm = norm(claim["claim_text"])
        for page in pdf_pages:
            score, common = page_score(claim["claim_text"], page)
            exact = bool(claim_norm and claim_norm in str(page["norm"]))
            if exact or (score >= 0.35 and common >= 2):
                scored_claim_pages.append((score + (1.0 if exact else 0.0), common, str(page["source"]), int(page["page_num"]), page, exact))
        scored_claim_pages.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
        for rank, (score, _, _, _, page, exact) in enumerate(scored_claim_pages[:6], start=1):
            add_document_candidate(
                qid=qid, page=page, query=claim["claim_text"],
                method="document_exact_claim" if exact else "document_claim_search",
                rank=rank, score=score, claim_id=claim["claim_id"],
                detail="second-pass exhaustive PDF page scan",
            )


# Verify gold text against indicated and adjacent PDF pages without accepting it automatically.
for row in gold_rows:
    qid = row["question_id"]
    doc_id = resolve_doc_id(row["expected_doc"])
    try:
        page_num = int(float(row["expected_page"]))
    except (TypeError, ValueError):
        continue
    candidates = []
    for offset in (-1, 0, 1):
        page = pages_by_doc_page.get((doc_id, page_num + offset))
        if page:
            score, common = page_score(row["expected_text"], page)
            exact = norm(row["expected_text"]) in str(page["norm"])
            candidates.append((score + (1.0 if exact else 0.0), common, abs(offset), page, exact))
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    for rank, (score, _, _, page, exact) in enumerate(candidates[:2], start=1):
        add_document_candidate(
            qid=qid, page=page, query=row["expected_text"], method="gold_document_verification",
            rank=rank, score=score, gold=True,
            detail=f"gold_evidence_id={row['evidence_id']}; exact_text={str(exact).lower()}; adjacent_page_checked",
        )


# Stable raw candidate identifiers.
raw_pool.sort(key=lambda row: (
    str(row["question_id"]), str(row["retrieval_method"]),
    int(row["retrieval_rank"]) if str(row["retrieval_rank"]).isdigit() else 10**9,
    str(row["doc_id"]), str(row["chunk_id"]), str(row["page_num"]), text_sha256(str(row["content"])),
))
per_question_pool_counter: Counter[str] = Counter()
for row in raw_pool:
    per_question_pool_counter[str(row["question_id"])] += 1
    row["candidate_id"] = f"{row['question_id']}-P{per_question_pool_counter[str(row['question_id'])]:05d}"

# Explicit per-query method coverage, including zero-result and short-list cases.
method_depths = {
    "semantic_search_persisted": 5,
    "hybrid_search_persisted": 5,
    "language_aware_hybrid_search_persisted": 5,
    "lexical_simple": 100,
    "lexical_spanish": 100,
    "lexical_english": 100,
}
method_counts_by_question = Counter(
    (str(row["question_id"]), str(row["retrieval_method"])) for row in raw_pool
)
method_coverage_rows = []
for question in questions:
    qid = question["question_id"]
    for method, requested in method_depths.items():
        returned = method_counts_by_question[(qid, method)]
        method_coverage_rows.append({
            "question_id": qid, "retrieval_method": method, "requested_depth": requested,
            "returned_candidates": returned, "zero_results": str(returned == 0).lower(),
            "fewer_than_requested": str(returned < requested).lower(),
            "detected_language": "", "notes": "Detected language was not persisted by the original evaluation." if "language_aware" in method else "",
        })
write_csv(
    LOG_DIR / "retrieval_method_coverage.csv", method_coverage_rows,
    ["question_id", "retrieval_method", "requested_depth", "returned_candidates", "zero_results", "fewer_than_requested", "detected_language", "notes"],
)

pool_fields = [
    "question_id", "candidate_id", "doc_id", "chunk_id", "page_num", "source", "lang", "content",
    "retrieval_method", "retrieval_rank", "retrieval_score", "score_type", "query_language",
    "detected_language", "ts_config", "rrf_k", "found_in_docs", "document_only",
    "original_gold_candidate", "claim_id", "query_type", "ingestion_gap", "provenance_detail",
]
write_csv(RUN_DIR / "retrieval_pool.csv", raw_pool, pool_fields)


# Deduplicate while preserving every retrieval provenance row in retrieval_pool.csv.
groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
for row in raw_pool:
    content = str(row["content"])
    if row["chunk_id"]:
        key = f"chunk|{row['doc_id']}|{row['chunk_id']}"
    else:
        key = f"docpage|{row['doc_id']}|{row['page_num']}|{text_sha256(norm(content))}"
    groups[(str(row["question_id"]), key)].append(row)

evidence_candidates: list[dict[str, object]] = []
per_question_evidence_counter: Counter[str] = Counter()
method_rank_columns = {
    "semantic": "best_rank_semantic", "lexical": "best_rank_lexical",
    "hybrid": "best_rank_hybrid", "language": "best_rank_language_aware",
}
for (qid, key), rows in sorted(groups.items()):
    per_question_evidence_counter[qid] += 1
    evidence_id = f"{qid}-E{per_question_evidence_counter[qid]:04d}"
    representative = max(rows, key=lambda row: (bool(str(row["content"])), len(str(row["content"]))))
    methods = sorted({str(row["retrieval_method"]) for row in rows})
    rank_values = {column: "" for column in method_rank_columns.values()}
    for label, column in method_rank_columns.items():
        ranks = [
            int(row["retrieval_rank"]) for row in rows
            if label in str(row["retrieval_method"]) and str(row["retrieval_rank"]).isdigit()
        ]
        rank_values[column] = min(ranks) if ranks else ""
    evidence_candidates.append({
        "question_id": qid, "candidate_id": ";".join(str(row["candidate_id"]) for row in rows),
        "evidence_id": evidence_id, "doc_id": representative["doc_id"], "chunk_id": representative["chunk_id"],
        "page_num": representative["page_num"], "source": representative["source"], "lang": representative["lang"],
        "content": representative["content"], "normalized_text_sha256": text_sha256(norm(str(representative["content"]))),
        "retrieved_by": ";".join(methods), **rank_values,
        "found_manually": str(any(str(row["found_in_docs"]) == "true" for row in rows)).lower(),
        "document_only": str(any(str(row["document_only"]) == "true" for row in rows) and not representative["chunk_id"]).lower(),
        "ingestion_gap": str(any(str(row["ingestion_gap"]) == "true" for row in rows) and not representative["chunk_id"]).lower(),
        "original_gold_candidate": str(any(str(row["original_gold_candidate"]) == "true" for row in rows)).lower(),
    })

candidate_fields = [
    "question_id", "candidate_id", "evidence_id", "doc_id", "chunk_id", "page_num", "source", "lang",
    "content", "normalized_text_sha256", "retrieved_by", "best_rank_semantic", "best_rank_lexical",
    "best_rank_hybrid", "best_rank_language_aware", "found_manually", "document_only", "ingestion_gap",
    "original_gold_candidate",
]
write_csv(RUN_DIR / "evidence_candidates.csv", evidence_candidates, candidate_fields)


def score_claim(claim_text: str, evidence_text: str) -> tuple[int, float, str]:
    if not evidence_text.strip():
        return 0, 0.0, "empty candidate"
    claim_norm = norm(claim_text)
    evidence_norm = norm(evidence_text)
    cov, common, numbers_ok = coverage_score(claim_text, evidence_text)
    exact = bool(claim_norm and len(claim_norm) >= 12 and claim_norm in evidence_norm)
    claim_token_count = len(set(tokens(claim_text)))
    if exact:
        return 3, 1.0, "claim text appears directly in evidence"
    if cov >= 0.88 and common >= min(4, max(2, claim_token_count)) and numbers_ok:
        return 3, cov, "nearly complete distinctive-token coverage with numeric consistency"
    if cov >= 0.68 and common >= 3 and numbers_ok:
        return 2, cov, "substantial but incomplete direct token coverage"
    if cov >= 0.30 and common >= 2:
        return 1, cov, "topic-related overlap insufficient to assert claim"
    return 0, cov, "no useful claim-level support detected"


judgments: list[dict[str, object]] = []
evidence_by_id = {row["evidence_id"]: row for row in evidence_candidates}
for evidence in evidence_candidates:
    qid = str(evidence["question_id"])
    results = []
    for claim in claims_by_question[qid]:
        grade, cov, reason = score_claim(claim["claim_text"], str(evidence["content"]))
        results.append((grade, cov, claim["claim_id"], reason, claim["claim_text"]))
    results.sort(key=lambda item: (-item[0], -item[1], item[2]))
    best_grade = results[0][0] if results else 0
    covered = [item[2] for item in results if item[0] >= 2]
    best_cov = results[0][1] if results else 0.0
    source_exists = bool(evidence["source"] and Path(str(evidence["source"])).exists())
    document_only = evidence["document_only"] == "true"
    if best_grade == 3:
        confidence = "high" if best_cov >= 0.95 else "medium"
        decision = "accept"
        coverage_type = "full_claim"
    elif best_grade == 2:
        confidence = "medium"
        decision = "uncertain"
        coverage_type = "partial_claim"
    elif best_grade == 1:
        confidence = "medium"
        decision = "reject"
        coverage_type = "related_only"
    else:
        confidence = "high" if str(evidence["content"]).strip() else "medium"
        decision = "reject"
        coverage_type = "none"
    review_reasons = []
    if best_grade == 2:
        review_reasons.append("grade_2_vs_3_ambiguous")
    if document_only and best_grade >= 2:
        review_reasons.append("document_only_evidence")
    if not source_exists and best_grade >= 1:
        review_reasons.append("source_path_missing")
    if confidence == "low":
        review_reasons.append("low_confidence")
    rationale = results[0][3] if results else "no claims"
    rationale += f"; best_token_coverage={best_cov:.4f}; text_verified_in={'PDF' if document_only else 'DB_snapshot'}"
    judgments.append({
        "question_id": qid, "evidence_id": evidence["evidence_id"], "relevance_grade": best_grade,
        "covered_claim_ids": ";".join(covered), "coverage_type": coverage_type,
        "direct_or_inferred": "direct" if best_grade >= 2 else "",
        "confidence": confidence, "decision": decision, "rationale": rationale,
        "contradiction": "false", "outdated": "false", "wrong_scope": "false",
        "needs_human_review": str(bool(review_reasons)).lower(), "review_reason": ";".join(review_reasons),
        "annotator_type": "machine_proposed", "annotator_version": "rag-evidence-builder-v2.0-rulebased",
    })

judgment_fields = [
    "question_id", "evidence_id", "relevance_grade", "covered_claim_ids", "coverage_type",
    "direct_or_inferred", "confidence", "decision", "rationale", "contradiction", "outdated",
    "wrong_scope", "needs_human_review", "review_reason", "annotator_type", "annotator_version",
]
write_csv(RUN_DIR / "evidence_judgments_machine.csv", judgments, judgment_fields)


judgment_by_evidence = {row["evidence_id"]: row for row in judgments}
provisional: list[dict[str, object]] = []
for evidence in evidence_candidates:
    judgment = judgment_by_evidence[evidence["evidence_id"]]
    grade = int(judgment["relevance_grade"])
    if grade < 2:
        continue
    claim_texts = [
        claim["claim_text"] for claim in claims_by_question[str(evidence["question_id"])]
        if claim["claim_id"] in str(judgment["covered_claim_ids"]).split(";")
    ]
    query = claim_texts[0] if claim_texts else question_by_id[str(evidence["question_id"])]["question"]
    excerpt, _ = best_snippet(str(evidence["content"]), query, max_chars=600)
    provisional.append({
        "question_id": evidence["question_id"], "evidence_id": evidence["evidence_id"],
        "chunk_id": evidence["chunk_id"], "doc_id": evidence["doc_id"], "page_num": evidence["page_num"],
        "source": evidence["source"], "lang": evidence["lang"], "evidence_text": excerpt,
        "relevance_grade": grade, "covered_claim_ids": judgment["covered_claim_ids"],
        "confidence": judgment["confidence"],
        "evidence_origin": "document_only" if evidence["document_only"] == "true" else "database_chunk",
        "retrieved_by": evidence["retrieved_by"], "content_sha256": text_sha256(str(evidence["content"])),
        "status": "provisional_machine_judgment", "notes": judgment["review_reason"],
    })

provisional_fields = [
    "question_id", "evidence_id", "chunk_id", "doc_id", "page_num", "source", "lang", "evidence_text",
    "relevance_grade", "covered_claim_ids", "confidence", "evidence_origin", "retrieved_by",
    "content_sha256", "status", "notes",
]
write_csv(RUN_DIR / "evidence_set_provisional.csv", provisional, provisional_fields)


support_by_claim: dict[str, list[dict[str, object]]] = defaultdict(list)
for row in provisional:
    for claim_id in str(row["covered_claim_ids"]).split(";"):
        if claim_id:
            support_by_claim[claim_id].append(row)

claim_coverage: list[dict[str, object]] = []
for claim in claims:
    supporting = support_by_claim.get(claim["claim_id"], [])
    grades = [int(row["relevance_grade"]) for row in supporting]
    best_grade = max(grades, default=0)
    if best_grade >= 3:
        status = "fully_supported"
    elif best_grade == 2:
        status = "partially_supported"
    else:
        status = "unsupported"
    needs_review = status != "fully_supported" or any(row["confidence"] != "high" for row in supporting)
    claim_coverage.append({
        "question_id": claim["question_id"], "claim_id": claim["claim_id"], "claim_text": claim["claim_text"],
        "evidence_ids": ";".join(row["evidence_id"] for row in supporting),
        "num_supporting_evidences": len(supporting), "best_relevance_grade": best_grade,
        "covered": str(best_grade >= 3).lower(), "coverage_status": status,
        "contradictory_evidence_ids": "", "needs_human_review": str(needs_review).lower(),
        "notes": "Machine coverage only; second-pass DB snapshot and all extracted PDF pages searched.",
    })

coverage_fields = [
    "question_id", "claim_id", "claim_text", "evidence_ids", "num_supporting_evidences",
    "best_relevance_grade", "covered", "coverage_status", "contradictory_evidence_ids",
    "needs_human_review", "notes",
]
write_csv(RUN_DIR / "claim_coverage.csv", claim_coverage, coverage_fields)


unresolved = []
for row in claim_coverage:
    if row["coverage_status"] != "fully_supported":
        unresolved.append({
            "question_id": row["question_id"], "claim_id": row["claim_id"], "claim_text": row["claim_text"],
            "resolution_status": "not_found" if row["coverage_status"] == "unsupported" else "requires_human_investigation",
            "coverage_status": row["coverage_status"],
            "searches_performed": "PostgreSQL FTS(simple,spanish,english); DB snapshot token search; all PDF pages; gold page and adjacent pages",
            "possible_cause": "possible_extraction_failure_or_ground_truth_questionable" if pdf_errors else "requires_human_investigation",
            "notes": "Absence was not interpreted as proof that the claim is not present in the corpus.",
        })
unresolved_fields = [
    "question_id", "claim_id", "claim_text", "resolution_status", "coverage_status",
    "searches_performed", "possible_cause", "notes",
]
write_csv(RUN_DIR / "unresolved_questions.csv", unresolved, unresolved_fields)


review_rows: list[dict[str, object]] = []
review_keys: set[tuple[str, str, str, str]] = set()


def enqueue_review(qid: str, claim_id: str = "", evidence_id: str = "", issue_type: str = "", rationale: str = "") -> None:
    key = (qid, claim_id, evidence_id, issue_type)
    if key in review_keys:
        return
    review_keys.add(key)
    question = question_by_id[qid]
    claim = next((row for row in claims_by_question[qid] if row["claim_id"] == claim_id), None)
    evidence = evidence_by_id.get(evidence_id)
    judgment = judgment_by_evidence.get(evidence_id)
    review_rows.append({
        "review_id": "", "question_id": qid, "claim_id": claim_id, "evidence_id": evidence_id,
        "question": question["question"], "claim_text": claim["claim_text"] if claim else "",
        "evidence_text": (best_snippet(str(evidence["content"]), claim["claim_text"] if claim else question["question"])[0] if evidence else ""),
        "doc_id": evidence["doc_id"] if evidence else "", "page_num": evidence["page_num"] if evidence else "",
        "machine_grade": judgment["relevance_grade"] if judgment else "",
        "machine_confidence": judgment["confidence"] if judgment else "",
        "issue_type": issue_type, "review_question": "Confirm claim segmentation, evidence grade, scope and direct support.",
        "machine_rationale": rationale or (judgment["rationale"] if judgment else ""),
        "human_grade": "", "human_covered_claims": "", "human_decision": "", "human_notes": "",
        "adjudicator": "", "adjudicated_grade": "", "adjudication_notes": "",
    })


for claim in claims:
    notes = claim["notes"]
    if any(flag in notes for flag in ["long_claim", "compound_claim", "many_claims", "very_short", "unsplittable"]):
        enqueue_review(claim["question_id"], claim["claim_id"], issue_type="claim_segmentation_review", rationale=notes)
for judgment in judgments:
    if judgment["needs_human_review"] == "true":
        covered_ids = str(judgment["covered_claim_ids"]).split(";")
        enqueue_review(
            judgment["question_id"], covered_ids[0] if covered_ids and covered_ids[0] else "",
            judgment["evidence_id"], "machine_evidence_judgment_review", str(judgment["rationale"]),
        )
for row in claim_coverage:
    if row["coverage_status"] != "fully_supported":
        enqueue_review(row["question_id"], row["claim_id"], issue_type="claim_not_fully_supported", rationale=str(row["notes"]))
for item in pdf_errors:
    # Page extraction errors are document-level; attach them to all questions only if the doc was a gold source.
    source_name = Path(item["source"]).name
    affected = sorted({row["question_id"] for row in gold_rows if Path(row["expected_doc"]).name.casefold() == source_name.casefold()})
    for qid in affected:
        enqueue_review(qid, issue_type="pdf_extraction_failure", rationale=json.dumps(item, ensure_ascii=False))

review_rows.sort(key=lambda row: (str(row["question_id"]), str(row["claim_id"]), str(row["evidence_id"]), str(row["issue_type"])))
for index, row in enumerate(review_rows, start=1):
    row["review_id"] = f"R{index:05d}"
review_fields = [
    "review_id", "question_id", "claim_id", "evidence_id", "question", "claim_text", "evidence_text",
    "doc_id", "page_num", "machine_grade", "machine_confidence", "issue_type", "review_question",
    "machine_rationale", "human_grade", "human_covered_claims", "human_decision", "human_notes",
    "adjudicator", "adjudicated_grade", "adjudication_notes",
]
write_csv(RUN_DIR / "human_review_queue.csv", review_rows, review_fields)


# Source inventory with hashes; .env is deliberately excluded.
source_paths = [
    PROJECT_ROOT / "eval" / "preguntas_val.xlsx", PROJECT_ROOT / "eval" / "unique_questions.csv",
    PROJECT_ROOT / "eval" / "gold_references.csv",
    PROJECT_ROOT / "notebooks" / "RAG-evaluation.ipynb",
    PROJECT_ROOT / "eval" / "final_answers_with_retrieval_eval.csv",
    PROJECT_ROOT / "eval" / "content_evaluation_full.csv",
    PROJECT_ROOT / "eval" / "content_evaluation_full_updated.csv",
    PROJECT_ROOT / "eval" / "content_evaluation_manual.csv",
    *persisted_files.values(),
    PROJECT_ROOT / "app" / "db.py", PROJECT_ROOT / "app" / "retrieval.py",
    PROJECT_ROOT / "app" / "config.py", PROJECT_ROOT / "app" / "embeddings.py",
    PROJECT_ROOT / "app" / "chunking.py", PROJECT_ROOT / "app" / "ingest.py",
    PROJECT_ROOT / "scripts" / "ingest_docs.py", PROJECT_ROOT / "scripts" / "create_tables.sql",
    PROJECT_ROOT / "requirements.txt", *sorted((PROJECT_ROOT / "docs").rglob("*.pdf")),
]
source_inventory = []
pdf_meta_by_source = {row["source"]: row for row in pdf_inventory}
for path in source_paths:
    role = "document_corpus" if path.suffix.casefold() == ".pdf" else "code_or_evaluation_source"
    pdf_meta = pdf_meta_by_source.get(str(path.resolve()), {})
    source_inventory.append({
        "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"), "role": role,
        "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
        "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        "page_count": pdf_meta.get("page_count", ""),
        "nonempty_extracted_pages": pdf_meta.get("nonempty_extracted_pages", ""),
        "used": "true",
    })
inventory_fields = ["path", "role", "sha256", "size_bytes", "modified_utc", "page_count", "nonempty_extracted_pages", "used"]
write_csv(RUN_DIR / "source_inventory.csv", source_inventory, inventory_fields)


per_doc_counts = Counter(row["doc_id"] for row in db_chunks)
language_counts = Counter(row["lang"] or "<NULL>" for row in db_chunks)
source_correspondence = []
for doc_id, source in sorted(doc_source.items()):
    source_path = Path(source)
    source_correspondence.append({
        "doc_id": doc_id, "source": source, "exists_in_docs": source_path.exists(),
        "chunk_count": per_doc_counts[doc_id],
    })
database_profile = {
    "profile_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "connection_method": "psql using project PostgreSQL variables loaded indirectly; values not logged",
    "transaction_read_only": True,
    "table": "public.rag_chunks",
    "schema": db_schema,
    "num_documents": int(db_profile_raw["num_documents"]),
    "num_chunks": int(db_profile_raw["num_chunks"]),
    "null_counts": {key: int(value) for key, value in db_profile_raw.items() if key.startswith("null_")},
    "languages": dict(sorted(language_counts.items())),
    "chunks_per_document": dict(sorted(per_doc_counts.items())),
    "source_correspondence": source_correspondence,
    "missing_source_paths": [row["source"] for row in source_correspondence if not row["exists_in_docs"]],
    "embeddings_exported": False,
    "database_write_operations": [],
}
(RUN_DIR / "database_profile.json").write_text(json.dumps(database_profile, ensure_ascii=False, indent=2), encoding="utf-8")


(LOG_DIR / "direct_document_search.json").write_text(json.dumps({
    "documents": pdf_inventory, "extraction_errors": pdf_errors,
    "pages_scanned": len(pdf_pages), "nonempty_pages": sum(bool(str(row["text"]).strip()) for row in pdf_pages),
    "methods": ["document_question_search", "document_claim_search", "document_exact_claim", "gold_document_verification"],
    "adjacent_gold_pages_checked": True,
}, ensure_ascii=False, indent=2), encoding="utf-8")


guidelines = """# Annotation guidelines

## Status and scope

This corpus contains machine-proposed judgments. Nothing is human-adjudicated. Human fields must remain empty until independent review.

## Claim

A claim is one independently verifiable statement preserved from the existing expected-answer text. Do not add clinical knowledge, repair the expected answer silently, or merge distinct requirements.

## Evidence

Evidence is text verifiably present in `rag_chunks.content` or on the specified page of a source document. Document or gold provenance alone is not evidence.

## Relevance scale

- **3 - direct and sufficient:** explicitly supports a complete claim.
- **2 - substantial partial support:** supports a material part but needs another fragment or interpretation.
- **1 - related only:** discusses the topic but cannot establish a claim.
- **0 - no useful evidence:** irrelevant, wrong entity/scope, or empty/unverifiable.

## Rules

- Inference requiring external knowledge is not accepted.
- Grade 2 is always queued when grade 2 versus 3 is ambiguous.
- Contradictions, outdated guidance and wrong scope must be recorded, not reconciled automatically.
- A correct document with a missing/incorrect chunk is `document_only` and an `ingestion_gap` candidate.
- Several chunks may jointly support a claim; each fragment keeps its own grade.
- Adjacent pages may be read for context, but the saved excerpt must be from the recorded source/page.
- Tables or visually ambiguous pages require human review and, when necessary, page rendering.
- Human reviewers fill only the `human_*` columns. A separate adjudicator fills `adjudicated_*` after disagreement review.

## Real, anonymized examples from this run

- A definition claim such as “Astenia es el término médico para denominar al cansancio” receives grade 3 only when that definition appears in the candidate text.
- A fragment that merely mentions tamoxifen without the requested recommendation is grade 1.
- A PDF excerpt with no confidently corresponding database chunk remains `document_only`, even when its text is relevant.
"""
(RUN_DIR / "annotation_guidelines.md").write_text(guidelines, encoding="utf-8")


coverage_counter = Counter(row["coverage_status"] for row in claim_coverage)
summary_rows = []
for question in questions:
    qid = question["question_id"]
    q_claims = [row for row in claim_coverage if row["question_id"] == qid]
    q_evidence = [row for row in provisional if row["question_id"] == qid]
    summary_rows.append({
        "question_id": qid, "num_claims": len(q_claims), "num_candidates": per_question_evidence_counter[qid],
        "num_relevant_evidences": len(q_evidence),
        "covered_claims": sum(row["coverage_status"] == "fully_supported" for row in q_claims),
        "uncovered_claims": sum(row["coverage_status"] != "fully_supported" for row in q_claims),
        "review_required": "yes" if any(row["question_id"] == qid for row in review_rows) else "no",
    })

summary_table = "\n".join(
    f"| {row['question_id']} | {row['num_claims']} | {row['num_candidates']} | {row['num_relevant_evidences']} | "
    f"{row['covered_claims']} | {row['uncovered_claims']} | {row['review_required']} |"
    for row in summary_rows
)
readme = f"""# Evaluation evidence corpus v2

**Status:** corpus de evidencia provisional generado y verificado automáticamente, pendiente de revisión y adjudicación humana.

## What was done

The 31 selected questions were frozen from `eval/preguntas_val.xlsx`. Expected answers were reconstructed exactly as the original evaluation notebook does, from ordered `expected_text` rows in `eval/gold_references.csv`. They were split conservatively into {len(claims)} machine-generated claims.

The pool combines persisted semantic/hybrid/language-aware top-5 outputs, new read-only PostgreSQL FTS (`simple`, `spanish`, `english`, requested depth 100), offline lexical search over the complete read-only chunk snapshot, existing gold candidates, and direct search across every extractable page in `docs/`. Gold provenance was never treated as automatic relevance.

Candidates were deduplicated by `doc_id + chunk_id`, or by document/page/normalized-text hash when no chunk existed. Grades 0-3 are conservative machine proposals. Grade 2/3 excerpts were checked against the exported DB content or extracted PDF page text.

## Reproduction

1. Run `scripts/prepare_questions_claims.py` with Python 3.12+.
2. Run `scripts/export_db_readonly.ps1`; it forces PostgreSQL read-only mode and exports no embeddings.
3. Run `scripts/build_corpus.py` with the bundled `pypdf` runtime.
4. Run `scripts/validate_final.py` after human-independent checks if available.

No ingestion function, schema change, dependency installation, model download or paid API is used.

## Limitations

- Live semantic/hybrid retrieval to depth 100 was not rerun because the installed runtime lacks the project embedding/model dependencies. Persisted top-5 outputs are retained instead.
- Hybrid persisted outputs omit exact chunk IDs and raw semantic/lexical/RRF scores.
- Language-aware outputs do not persist detected language.
- One database source path does not exist in current `docs/`, indicating stale DB/document correspondence.
- Machine claim splitting and judgments require independent human review.
- Full-text extraction cannot prove that visually complex tables or scanned pages were interpreted correctly.
- Failure to find evidence is not proof that evidence is absent; absolute exhaustiveness is not claimed.

## Per-question summary

| question_id | num_claims | num_candidates | num_relevant_evidences | covered_claims | uncovered_claims | review_required |
|---|---:|---:|---:|---:|---:|---|
{summary_table}
"""
(RUN_DIR / "README.md").write_text(readme, encoding="utf-8")


# Automatic validation. Failed checks remain visible.
question_ids = set(question_by_id)
claim_ids = {row["claim_id"] for row in claims}
evidence_ids = {(row["question_id"], row["evidence_id"]) for row in evidence_candidates}
checks: list[tuple[str, bool, str]] = []
checks.append(("all_question_ids_known", all(row["question_id"] in question_ids for row in raw_pool + evidence_candidates + judgments), "references must resolve to questions.csv"))
checks.append(("all_claim_ids_known", all(not cid or cid in claim_ids for row in judgments for cid in str(row["covered_claim_ids"]).split(";")), "covered claim references"))
checks.append(("evidence_ids_unique_per_question", len(evidence_ids) == len(evidence_candidates), "deduplicated evidence IDs"))
checks.append(("database_chunks_verify", all(not row["chunk_id"] or (row["doc_id"], row["chunk_id"]) in chunk_by_key for row in evidence_candidates), "verified against read-only DB snapshot"))
checks.append(("all_source_paths_exist", all(not row["source"] or Path(str(row["source"])).exists() for row in evidence_candidates), "known stale DB source causes an explicit failure"))
pdf_page_counts = {str(row["source"]): int(row["page_count"]) for row in pdf_inventory}
checks.append(("pages_within_pdf_range", all(
    not row["source"] or str(row["source"]) not in pdf_page_counts or not str(row["page_num"]).isdigit()
    or 1 <= int(row["page_num"]) <= pdf_page_counts[str(row["source"])] for row in evidence_candidates
), "document-backed page bounds"))
checks.append(("accepted_evidence_has_text", all(row["evidence_text"].strip() for row in provisional), "grade 2/3 evidence excerpts"))
checks.append(("grade_2_or_3_has_claims", all(str(row["covered_claim_ids"]).strip() for row in provisional), "claim-level relevance"))
checks.append(("covered_claim_has_evidence", all(row["coverage_status"] != "fully_supported" or int(row["num_supporting_evidences"]) > 0 for row in claim_coverage), "coverage integrity"))
checks.append(("no_embeddings_exported", all("embedding" not in row for row in raw_pool), "pool schema/content excludes embedding column"))
checks.append(("human_fields_empty", all(not row[field] for row in review_rows for field in ["human_grade", "human_covered_claims", "human_decision", "human_notes", "adjudicator", "adjudicated_grade", "adjudication_notes"]), "no fabricated human decisions"))

validation_lines = [
    "# Validation report", "", f"Generated UTC: {datetime.now(timezone.utc).isoformat()}", "",
    "This report does not hide failed checks.", "", "| Check | Result | Detail |", "|---|---|---|",
]
for name, passed, detail in checks:
    validation_lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} | {detail} |")
validation_lines.extend([
    "", "## Known limitations", "",
    f"- PDF extraction errors: {len(pdf_errors)}.",
    f"- Missing DB source paths: {len(database_profile['missing_source_paths'])}.",
    "- Database before/after counts are finalized by `validate_final.py`.",
    "- Git confinement is finalized after all result files exist.",
    "- A PASS validates internal consistency, not human correctness of relevance grades.",
])
(RUN_DIR / "validation_report.md").write_text("\n".join(validation_lines) + "\n", encoding="utf-8")


try:
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
except Exception:
    git_commit = "UNAVAILABLE"
initial_git_status = (LOG_DIR / "git_status_initial.txt").read_text(encoding="utf-8").splitlines()

result_paths = sorted(
    path for path in RUN_DIR.rglob("*")
    if path.is_file() and path.name != "manifest.json"
)
result_hashes = {
    str(path.relative_to(RUN_DIR)).replace("\\", "/"): sha256_file(path)
    for path in result_paths
}
method_counts = Counter(str(row["retrieval_method"]) for row in raw_pool)
manifest = {
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "git_commit": git_commit,
    "initial_git_status_excluding_new_evaluation_v2": initial_git_status,
    "canonical_questions_path": "eval/preguntas_val.xlsx",
    "ground_truth_source": "eval/gold_references.csv",
    "original_sources": source_inventory,
    "question_count": len(questions), "claim_count": len(claims),
    "candidate_rows_before_deduplication": len(raw_pool),
    "unique_candidates_after_deduplication": len(evidence_candidates),
    "candidates_by_method": dict(sorted(method_counts.items())),
    "retrieval_depth": {
        "postgres_fts_requested": 100, "persisted_semantic_hybrid_language_aware": 5,
        "offline_db_question": 15, "offline_db_claim": 8,
        "direct_document_question": 10, "direct_document_claim": 6,
    },
    "retrieval_parameters": {
        "postgres_fts_configs": ["simple", "spanish", "english"],
        "postgres_fts_function": "plainto_tsquery + ts_rank_cd",
        "rrf_k_from_project_code": 60,
        "language_detection_persisted": False,
    },
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2 (configuration default; effective secret-bearing settings were not loaded or displayed)",
    "rag_chunks_schema": db_schema,
    "provisional_evidence_count": len(provisional),
    "fully_supported_claims": coverage_counter["fully_supported"],
    "partially_supported_claims": coverage_counter["partially_supported"],
    "unsupported_claims": coverage_counter["unsupported"],
    "human_review_case_count": len(review_rows),
    "scripts_created": sorted(str(path.relative_to(RUN_DIR)).replace("\\", "/") for path in (RUN_DIR / "scripts").glob("*")),
    "result_hashes_excluding_self_referential_manifest": result_hashes,
    "database_profile_path": "database_profile.json",
    "limitations": [
        "No live semantic/hybrid depth-100 rerun: dependencies/model unavailable and installation/download forbidden.",
        "Persisted hybrid outputs omit chunk IDs and raw branch/RRF scores.",
        "Detected language is absent from persisted language-aware outputs.",
        "Machine claim splitting and relevance judgments are provisional.",
        "One DB source path is missing from current docs.",
        "No claim absence is asserted as exhaustive proof.",
        "manifest.json cannot contain its own stable SHA-256 without self-reference.",
    ],
}
(RUN_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

print(json.dumps({
    "questions": len(questions), "claims": len(claims), "pool_rows": len(raw_pool),
    "unique_candidates": len(evidence_candidates), "provisional_evidence": len(provisional),
    "coverage": dict(coverage_counter), "human_review": len(review_rows),
    "pdf_pages_scanned": len(pdf_pages), "pdf_errors": len(pdf_errors),
}, ensure_ascii=False))
