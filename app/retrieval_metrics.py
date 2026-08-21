"""Utilidades reutilizables para evaluar rankings de recuperación RAG.

El módulo calcula métricas *offline*: recibe rankings ya persistidos y qrels
(juicios de relevancia), por lo que nunca ejecuta retrieval ni accede a la base
de datos. La unidad de evidencia se identifica mediante ``evidence_id``.

Métricas principales recomendadas:

* nDCG@10: premia colocar primero evidencias con mayor grado de relevancia.
* Evidence Recall@5: proporción de evidencias relevantes conocidas recuperadas
  entre las cinco primeras posiciones.

No confundir Evidence Recall@k con Hit@k: Hit solo pregunta si apareció al
menos una evidencia; Recall mide cuántas de todas las evidencias relevantes
conocidas fueron recuperadas.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class RankedItem:
    """Una evidencia recuperada para una pregunta.

    ``rank`` empieza en 1. ``score`` es opcional porque las métricas usan el
    orden persistido, no necesitan volver a calcular ni comparar scores.
    """

    question_id: str
    evidence_id: str
    rank: int
    score: float | None = None


Qrels = Mapping[str, Mapping[str, float]]
Rankings = Mapping[str, Sequence[RankedItem]]


def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError("k debe ser >= 1")


def validate_qrels(qrels: Qrels) -> None:
    """Valida que IDs y grados sean utilizables y no negativos."""
    for question_id, judgments in qrels.items():
        if not question_id:
            raise ValueError("question_id vacío en qrels")
        for evidence_id, grade in judgments.items():
            if not evidence_id:
                raise ValueError(f"evidence_id vacío para {question_id}")
            if not math.isfinite(float(grade)) or float(grade) < 0:
                raise ValueError(f"Grado inválido: {question_id}/{evidence_id}={grade}")


def validate_ranking(items: Sequence[RankedItem], question_id: str | None = None) -> None:
    """Exige ranks 1..n, sin duplicados, para evitar métricas ambiguas."""
    ordered = sorted(items, key=lambda item: item.rank)
    expected = list(range(1, len(ordered) + 1))
    observed = [item.rank for item in ordered]
    if observed != expected:
        raise ValueError(f"Ranks no contiguos para {question_id or '?'}: {observed}")
    ids = [item.evidence_id for item in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Evidencias duplicadas para {question_id or '?'}")
    if question_id is not None and any(item.question_id != question_id for item in ordered):
        raise ValueError(f"Ranking mezcla question_id bajo {question_id}")


def dcg_at_k(grades: Sequence[float], k: int, *, exponential_gain: bool = True) -> float:
    """Discounted Cumulative Gain hasta ``k``.

    Por defecto usa ganancia ``2**grado - 1``, apropiada para qrels graduados
    0--3. El descuento es ``log2(rank + 1)``.
    """
    _validate_k(k)
    total = 0.0
    for rank, raw_grade in enumerate(grades[:k], start=1):
        grade = float(raw_grade)
        if grade < 0 or not math.isfinite(grade):
            raise ValueError(f"Grado inválido: {raw_grade}")
        gain = (2.0**grade - 1.0) if exponential_gain else grade
        total += gain / math.log2(rank + 1)
    return total


def ndcg_at_k(
    ranked_evidence_ids: Sequence[str],
    judgments: Mapping[str, float],
    k: int = 10,
    *,
    exponential_gain: bool = True,
) -> float:
    """Calcula nDCG@k para una pregunta; devuelve 0 si no hay ganancia ideal."""
    _validate_k(k)
    observed = [float(judgments.get(evidence_id, 0.0)) for evidence_id in ranked_evidence_ids[:k]]
    ideal = sorted((float(value) for value in judgments.values()), reverse=True)[:k]
    denominator = dcg_at_k(ideal, k, exponential_gain=exponential_gain)
    if denominator == 0.0:
        return 0.0
    return dcg_at_k(observed, k, exponential_gain=exponential_gain) / denominator


def evidence_recall_at_k(
    ranked_evidence_ids: Sequence[str],
    judgments: Mapping[str, float],
    k: int = 5,
    *,
    relevance_threshold: float = 2.0,
) -> float:
    """Fracción de evidencias relevantes conocidas recuperadas en top-k.

    Por defecto los grados 2 y 3 son relevantes. Si una pregunta no posee
    ninguna evidencia relevante adjudicada, devuelve ``nan``: esa pregunta no
    debe contribuir silenciosamente al promedio de recall.
    """
    _validate_k(k)
    relevant = {eid for eid, grade in judgments.items() if float(grade) >= relevance_threshold}
    if not relevant:
        return math.nan
    retrieved = set(ranked_evidence_ids[:k])
    return len(relevant & retrieved) / len(relevant)


def hit_at_k(
    ranked_evidence_ids: Sequence[str],
    judgments: Mapping[str, float],
    k: int = 5,
    *,
    relevance_threshold: float = 2.0,
) -> float:
    """Indica si hay al menos una evidencia relevante en top-k."""
    _validate_k(k)
    return float(any(float(judgments.get(eid, 0.0)) >= relevance_threshold for eid in ranked_evidence_ids[:k]))


def reciprocal_rank_at_k(
    ranked_evidence_ids: Sequence[str],
    judgments: Mapping[str, float],
    k: int = 10,
    *,
    relevance_threshold: float = 2.0,
) -> float:
    """Recíproco del rank de la primera evidencia relevante, o 0."""
    _validate_k(k)
    for rank, evidence_id in enumerate(ranked_evidence_ids[:k], start=1):
        if float(judgments.get(evidence_id, 0.0)) >= relevance_threshold:
            return 1.0 / rank
    return 0.0


def evaluate_question(
    ranked_evidence_ids: Sequence[str],
    judgments: Mapping[str, float],
    *,
    ndcg_k: int = 10,
    recall_k: int = 5,
    relevance_threshold: float = 2.0,
) -> dict[str, float]:
    """Calcula el conjunto estándar de métricas para una pregunta."""
    return {
        f"ndcg@{ndcg_k}": ndcg_at_k(ranked_evidence_ids, judgments, ndcg_k),
        f"evidence_recall@{recall_k}": evidence_recall_at_k(
            ranked_evidence_ids, judgments, recall_k,
            relevance_threshold=relevance_threshold,
        ),
        f"hit@{recall_k}": hit_at_k(
            ranked_evidence_ids, judgments, recall_k,
            relevance_threshold=relevance_threshold,
        ),
        f"reciprocal_rank@{ndcg_k}": reciprocal_rank_at_k(
            ranked_evidence_ids, judgments, ndcg_k,
            relevance_threshold=relevance_threshold,
        ),
    }


def evaluate_run(
    rankings: Rankings,
    qrels: Qrels,
    *,
    ndcg_k: int = 10,
    recall_k: int = 5,
    relevance_threshold: float = 2.0,
) -> tuple[list[dict[str, str | float | int]], dict[str, float | int]]:
    """Evalúa un run completo sin ejecutar retrieval.

    Se evalúan todas las preguntas de ``qrels``. Una pregunta ausente del run
    recibe un ranking vacío. Los ``nan`` (recall indefinido por ausencia de
    evidencias relevantes) se excluyen del macro-promedio y se contabilizan.
    """
    validate_qrels(qrels)
    rows: list[dict[str, str | float | int]] = []
    for question_id in sorted(qrels):
        items = list(rankings.get(question_id, ()))
        validate_ranking(items, question_id)
        evidence_ids = [item.evidence_id for item in sorted(items, key=lambda item: item.rank)]
        metrics = evaluate_question(
            evidence_ids, qrels[question_id], ndcg_k=ndcg_k,
            recall_k=recall_k, relevance_threshold=relevance_threshold,
        )
        rows.append({"question_id": question_id, "retrieved_count": len(items), **metrics})

    metric_names = [f"ndcg@{ndcg_k}", f"evidence_recall@{recall_k}", f"hit@{recall_k}", f"reciprocal_rank@{ndcg_k}"]
    summary: dict[str, float | int] = {"num_questions": len(rows)}
    for name in metric_names:
        values = [float(row[name]) for row in rows if not math.isnan(float(row[name]))]
        summary[f"macro_{name}"] = fmean(values) if values else math.nan
        summary[f"valid_questions_{name}"] = len(values)
    return rows, summary


def load_qrels_csv(
    path: str | Path,
    *,
    question_col: str = "question_id",
    evidence_col: str = "evidence_id",
    grade_col: str = "relevance_grade",
) -> dict[str, dict[str, float]]:
    """Carga qrels CSV con una fila por pregunta-evidencia.

    Los duplicados idénticos se toleran; grados conflictivos producen error.
    """
    qrels: dict[str, dict[str, float]] = defaultdict(dict)
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            question_id, evidence_id = row[question_col].strip(), row[evidence_col].strip()
            grade = float(row[grade_col])
            previous = qrels[question_id].get(evidence_id)
            if previous is not None and previous != grade:
                raise ValueError(f"Qrel conflictivo en fila {row_number}: {question_id}/{evidence_id}")
            qrels[question_id][evidence_id] = grade
    result = {qid: dict(values) for qid, values in qrels.items()}
    validate_qrels(result)
    return result


def load_rankings_csv(
    path: str | Path,
    *,
    question_col: str = "question_id",
    evidence_col: str = "evidence_id",
    rank_col: str = "rank",
    score_col: str | None = "score",
) -> dict[str, list[RankedItem]]:
    """Carga un run CSV persistido y valida orden, ranks y duplicados."""
    rankings: dict[str, list[RankedItem]] = defaultdict(list)
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            raw_score = row.get(score_col, "") if score_col else ""
            item = RankedItem(
                question_id=row[question_col].strip(),
                evidence_id=row[evidence_col].strip(),
                rank=int(row[rank_col]),
                score=float(raw_score) if raw_score not in (None, "") else None,
            )
            rankings[item.question_id].append(item)
    result = {qid: sorted(items, key=lambda item: item.rank) for qid, items in rankings.items()}
    for qid, items in result.items():
        validate_ranking(items, qid)
    return result


def write_evaluation_csv(rows: Iterable[Mapping[str, object]], path: str | Path) -> None:
    """Guarda resultados por pregunta en CSV."""
    materialized = list(rows)
    if not materialized:
        raise ValueError("No hay resultados que guardar")
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: calcula métricas desde ``rankings.csv`` y ``qrels.csv``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rankings_csv")
    parser.add_argument("qrels_csv")
    parser.add_argument("--output", default="retrieval_metrics_by_question.csv")
    parser.add_argument("--ndcg-k", type=int, default=10)
    parser.add_argument("--recall-k", type=int, default=5)
    parser.add_argument("--relevance-threshold", type=float, default=2.0)
    args = parser.parse_args(argv)

    rows, summary = evaluate_run(
        load_rankings_csv(args.rankings_csv), load_qrels_csv(args.qrels_csv),
        ndcg_k=args.ndcg_k, recall_k=args.recall_k,
        relevance_threshold=args.relevance_threshold,
    )
    write_evaluation_csv(rows, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
