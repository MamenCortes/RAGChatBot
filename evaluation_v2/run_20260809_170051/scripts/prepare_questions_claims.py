from __future__ import annotations

import csv
import json
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = RUN_DIR.parent.parent
CANONICAL_QUESTIONS = PROJECT_ROOT / "eval" / "preguntas_val.xlsx"
QUESTION_CROSSCHECK = PROJECT_ROOT / "eval" / "unique_questions.csv"
GOLD_REFERENCES = PROJECT_ROOT / "eval" / "gold_references.csv"


def read_xlsx_rows(path: Path) -> list[list[str]]:
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for si in root.findall("m:si", ns):
                shared.append("".join(t.text or "" for t in si.iterfind(".//m:t", ns)))
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows: list[list[str]] = []
    for row in sheet.findall(".//m:sheetData/m:row", ns):
        values: list[str] = []
        for cell in row.findall("m:c", ns):
            ref = cell.attrib.get("r", "A1")
            col_letters = re.match(r"[A-Z]+", ref)
            if not col_letters:
                continue
            col = 0
            for char in col_letters.group(0):
                col = col * 26 + ord(char) - ord("A") + 1
            while len(values) < col:
                values.append("")
            kind = cell.attrib.get("t")
            value_node = cell.find("m:v", ns)
            if kind == "inlineStr":
                text = "".join(t.text or "" for t in cell.iterfind(".//m:t", ns))
            elif value_node is None:
                text = ""
            elif kind == "s":
                text = shared[int(value_node.text or "0")]
            else:
                text = value_node.text or ""
            values[col - 1] = text
        rows.append(values)
    return rows


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_atomic_claims(text: str) -> tuple[list[str], list[str]]:
    notes: list[str] = []
    cleaned = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    # Split only at explicit textual boundaries. Do not paraphrase or add content.
    pieces = re.split(r"(?:\n\s*[•●▪*-]\s+|\n+|(?<=[.!?])\s+|\s*;\s*)", cleaned)
    claims: list[str] = []
    seen: set[str] = set()
    for piece in pieces:
        claim = normalize_whitespace(piece.strip(" •●▪*\t"))
        if not claim:
            continue
        if re.fullmatch(r"\d+[.)]?", claim):
            notes.append("standalone_enumeration_marker_removed")
            continue
        key = claim.casefold()
        if key in seen:
            continue
        seen.add(key)
        claims.append(claim)
        if len(claim) > 420:
            notes.append("long_claim_requires_human_segmentation")
        if claim.count(",") >= 5 or len(re.findall(r"\b(?:y|o)\b", claim.casefold())) >= 4:
            notes.append("compound_claim_possible")
    if not claims and cleaned:
        claims = [normalize_whitespace(cleaned)]
        notes.append("unsplittable_ground_truth")
    if len(claims) > 12:
        notes.append("many_claims_from_single_question")
    return claims, sorted(set(notes))


def classify_claim(text: str) -> str:
    t = text.casefold()
    if any(token in t for token in ["se define", " es el ", " es la ", "se denomina"]):
        return "definition"
    if any(token in t for token in ["debe", "recomienda", "hay que", "se aconseja"]):
        return "recommendation"
    if any(token in t for token in ["semana", "mes", "hora", "día", "plazo"]):
        return "temporal"
    if any(token in t for token in ["efecto adverso", "efecto secundario", "síntoma"]):
        return "effect_or_symptom"
    if any(char.isdigit() for char in text):
        return "quantitative"
    return "factual"


def main() -> None:
    xlsx_rows = read_xlsx_rows(CANONICAL_QUESTIONS)
    if not xlsx_rows or xlsx_rows[0][:2] != ["question_id", "question"]:
        raise ValueError("Unexpected canonical XLSX structure")
    selected = [
        {"question_id": row[0], "question": row[1], "original_row": index}
        for index, row in enumerate(xlsx_rows[1:], start=2)
        if len(row) >= 2 and (row[0] or row[1])
    ]

    with QUESTION_CROSSCHECK.open(encoding="utf-8-sig", newline="") as handle:
        crosscheck = list(csv.DictReader(handle))
    selected_pairs = [(row["question_id"], row["question"]) for row in selected]
    crosscheck_pairs = [(row["question_id"], row["question"]) for row in crosscheck]
    normalized_selected_pairs = [(qid, normalize_whitespace(question)) for qid, question in selected_pairs]
    normalized_crosscheck_pairs = [(qid, normalize_whitespace(question)) for qid, question in crosscheck_pairs]
    if normalized_selected_pairs != normalized_crosscheck_pairs:
        raise ValueError("preguntas_val.xlsx differs from unique_questions.csv; refusing to combine sources")
    whitespace_only_differences = [
        qid for (qid, canonical), (_, checked) in zip(selected_pairs, crosscheck_pairs)
        if canonical != checked and normalize_whitespace(canonical) == normalize_whitespace(checked)
    ]

    with GOLD_REFERENCES.open(encoding="utf-8-sig", newline="") as handle:
        gold_rows = list(csv.DictReader(handle))
    gold_by_question: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in gold_rows:
        gold_by_question[row["question_id"]].append(row)

    selected_ids = [row["question_id"] for row in selected]
    duplicates = sorted({qid for qid in selected_ids if selected_ids.count(qid) > 1})
    empty_questions = [row["question_id"] for row in selected if not row["question"].strip()]
    missing_gold = [qid for qid in selected_ids if not gold_by_question.get(qid)]
    extra_gold = sorted(set(gold_by_question) - set(selected_ids))
    if duplicates or empty_questions or extra_gold:
        raise ValueError(
            f"Question integrity failure: duplicates={duplicates}, empty={empty_questions}, extra_gold={extra_gold}"
        )

    question_fields = [
        "question_id", "question", "ground_truth", "query_language", "topic",
        "original_source", "original_row", "ground_truth_source",
    ]
    question_rows: list[dict[str, object]] = []
    claim_rows: list[dict[str, object]] = []
    review_flags: list[dict[str, object]] = []
    for selected_row in selected:
        qid = selected_row["question_id"]
        source_rows = gold_by_question.get(qid, [])
        expected_texts = [row["expected_text"] for row in source_rows if row["expected_text"].strip()]
        ground_truth = "\n\n".join(expected_texts)
        question_rows.append({
            "question_id": qid,
            "question": selected_row["question"],
            "ground_truth": ground_truth,
            "query_language": "",
            "topic": "",
            "original_source": str(CANONICAL_QUESTIONS.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "original_row": selected_row["original_row"],
            "ground_truth_source": str(GOLD_REFERENCES.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        })
        claims, notes = split_atomic_claims(ground_truth)
        evidence_ids = ";".join(row["evidence_id"] for row in source_rows)
        for index, claim in enumerate(claims, start=1):
            claim_id = f"{qid}-C{index:02d}"
            claim_notes = list(notes)
            if len(claim) < 20:
                claim_notes.append("very_short_claim")
            claim_rows.append({
                "question_id": qid,
                "claim_id": claim_id,
                "claim_text": claim,
                "claim_type": classify_claim(claim),
                "required_for_complete_answer": "true",
                "source_ground_truth": ground_truth,
                "machine_generated": "true",
                "notes": ";".join(sorted(set(claim_notes + [f"source_evidence_ids={evidence_ids}"]))),
            })
            if notes or len(claim) < 20:
                review_flags.append({
                    "question_id": qid,
                    "claim_id": claim_id,
                    "issue_type": "claim_segmentation_review",
                    "notes": ";".join(sorted(set(claim_notes))),
                })

    with (RUN_DIR / "questions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=question_fields)
        writer.writeheader()
        writer.writerows(question_rows)
    claim_fields = [
        "question_id", "claim_id", "claim_text", "claim_type",
        "required_for_complete_answer", "source_ground_truth", "machine_generated", "notes",
    ]
    with (RUN_DIR / "claims.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=claim_fields)
        writer.writeheader()
        writer.writerows(claim_rows)

    log = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_questions": str(CANONICAL_QUESTIONS.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "question_crosscheck": str(QUESTION_CROSSCHECK.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "ground_truth_source": str(GOLD_REFERENCES.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "question_count": len(question_rows),
        "claim_count": len(claim_rows),
        "duplicate_question_ids": duplicates,
        "empty_questions": empty_questions,
        "missing_ground_truth": missing_gold,
        "whitespace_only_cross_source_question_differences": whitespace_only_differences,
        "claim_review_flags": review_flags,
        "limitations": [
            "Claims are machine-generated by conservative boundary splitting and are not human-adjudicated.",
            "query_language and topic are blank because they are absent from the canonical question source.",
        ],
    }
    (RUN_DIR / "logs" / "questions_claims_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"questions": len(question_rows), "claims": len(claim_rows), "review_flags": len(review_flags)}))


if __name__ == "__main__":
    main()
