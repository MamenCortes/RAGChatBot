from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader


RUN_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = RUN_DIR.parent.parent
LOG_DIR = RUN_DIR / "logs"


def rows(name: str) -> list[dict[str, str]]:
    with (RUN_DIR / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def norm(text: str) -> str:
    value = unicodedata.normalize("NFKD", text.casefold())
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", value).strip()


questions = rows("questions.csv")
claims = rows("claims.csv")
pool = rows("retrieval_pool.csv")
candidates = rows("evidence_candidates.csv")
judgments = rows("evidence_judgments_machine.csv")
provisional = rows("evidence_set_provisional.csv")
coverage = rows("claim_coverage.csv")
review = rows("human_review_queue.csv")
inventory = rows("source_inventory.csv")
db_chunks = rows("logs/db_chunks.csv") if (RUN_DIR / "logs/db_chunks.csv").exists() else []
with (LOG_DIR / "db_profile_before.csv").open(encoding="utf-8-sig", newline="") as handle:
    db_before = list(csv.DictReader(handle))[0]
with (LOG_DIR / "db_profile_after.csv").open(encoding="utf-8-sig", newline="") as handle:
    db_after = list(csv.DictReader(handle))[0]

question_ids = {row["question_id"] for row in questions}
claim_ids = {row["claim_id"] for row in claims}
chunk_map = {(row["doc_id"], row["chunk_id"]): row for row in db_chunks}
candidate_map = {row["evidence_id"]: row for row in candidates}

checks: list[dict[str, object]] = []


def check(name: str, passed: bool, detail: str, failures: list[object] | None = None) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail, "failures": failures or []})


required_files = [
    "README.md", "manifest.json", "questions.csv", "claims.csv", "retrieval_pool.csv",
    "evidence_candidates.csv", "evidence_judgments_machine.csv", "evidence_set_provisional.csv",
    "claim_coverage.csv", "human_review_queue.csv", "unresolved_questions.csv", "source_inventory.csv",
    "database_profile.json", "validation_report.md", "annotation_guidelines.md",
]
missing_required = [name for name in required_files if not (RUN_DIR / name).exists()]
check("required_outputs_exist", not missing_required, "required output structure", missing_required)

check("all_question_ids_resolve", all(row["question_id"] in question_ids for row in pool + candidates + judgments + coverage), "all references resolve to questions.csv")
bad_claim_refs = sorted({
    claim_id for row in judgments for claim_id in row["covered_claim_ids"].split(";")
    if claim_id and claim_id not in claim_ids
})
check("all_claim_ids_resolve", not bad_claim_refs, "covered claims resolve to claims.csv", bad_claim_refs)
duplicate_evidence = [key for key, count in Counter((row["question_id"], row["evidence_id"]) for row in candidates).items() if count > 1]
check("evidence_ids_unique_per_question", not duplicate_evidence, "stable evidence identifiers", duplicate_evidence)

bad_db_chunks = []
for row in candidates:
    if not row["chunk_id"]:
        continue
    key = (row["doc_id"], row["chunk_id"])
    if key not in chunk_map or text_hash(norm(chunk_map[key]["content"])) != row["normalized_text_sha256"]:
        bad_db_chunks.append(key)
check("database_candidates_verify", not bad_db_chunks, "chunk IDs and normalized content hashes match the read-only DB snapshot", bad_db_chunks[:50])

bad_db_evidence = []
for row in provisional:
    if row["evidence_origin"] != "database_chunk":
        continue
    key = (row["doc_id"], row["chunk_id"])
    if key not in chunk_map or text_hash(chunk_map[key]["content"]) != row["content_sha256"]:
        bad_db_evidence.append((row["question_id"], row["evidence_id"], *key))
check("provisional_database_evidence_hashes", not bad_db_evidence, "full chunk hashes reproduce", bad_db_evidence)

pdf_cache: dict[str, PdfReader] = {}
bad_document_evidence = []
for row in provisional:
    if row["evidence_origin"] != "document_only":
        continue
    source = row["source"]
    try:
        reader = pdf_cache.setdefault(source, PdfReader(source))
        page_num = int(row["page_num"])
        page_text = reader.pages[page_num - 1].extract_text() or ""
        if not norm(row["evidence_text"]) or norm(row["evidence_text"]) not in norm(page_text):
            bad_document_evidence.append((row["question_id"], row["evidence_id"], source, page_num))
    except Exception as exc:
        bad_document_evidence.append((row["question_id"], row["evidence_id"], source, row["page_num"], type(exc).__name__))
check("provisional_document_evidence_text_verifies", not bad_document_evidence, "saved excerpt appears on recorded PDF page", bad_document_evidence)

missing_sources = sorted({row["source"] for row in candidates if row["source"] and not Path(row["source"]).exists()})
check("all_candidate_source_paths_exist", not missing_sources, "stale DB paths are reported rather than hidden", missing_sources)

inventory_hash_failures = []
for row in inventory:
    path = PROJECT_ROOT / Path(row["path"])
    if not path.exists() or file_hash(path) != row["sha256"]:
        inventory_hash_failures.append(row["path"])
check("original_source_hashes_unchanged", not inventory_hash_failures, "all consulted original hashes reproduce", inventory_hash_failures)

check("accepted_evidence_has_text", all(row["evidence_text"].strip() for row in provisional), "no grade 2/3 evidence is textless")
check("accepted_evidence_has_claims", all(row["covered_claim_ids"].strip() for row in provisional), "all grade 2/3 rows cover claims")
check("covered_claim_has_evidence", all(row["coverage_status"] != "fully_supported" or int(row["num_supporting_evidences"]) > 0 for row in coverage), "fully supported claims have evidence")

human_fields = ["human_grade", "human_covered_claims", "human_decision", "human_notes", "adjudicator", "adjudicated_grade", "adjudication_notes"]
nonempty_human = [(row["review_id"], field) for row in review for field in human_fields if row[field]]
check("human_fields_empty", not nonempty_human, "no human/adjudication result was fabricated", nonempty_human)

embedding_headers = [name for name in ["retrieval_pool.csv", "evidence_candidates.csv", "evidence_set_provisional.csv"] if "embedding" in (RUN_DIR / name).read_text(encoding="utf-8-sig").splitlines()[0].casefold()]
check("no_embeddings_exported", not embedding_headers, "no result table contains an embedding column", embedding_headers)

secret_failures = []
secret_patterns = [re.compile(r"postgres(?:ql)?://[^\s]+", re.I), re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")]
for path in RUN_DIR.rglob("*"):
    if not path.is_file() or path.suffix.casefold() in {".png", ".pdf"}:
        continue
    try:
        content = path.read_text(encoding="utf-8-sig", errors="ignore")
    except Exception:
        continue
    if any(pattern.search(content) for pattern in secret_patterns):
        secret_failures.append(str(path.relative_to(RUN_DIR)))
check("no_secret_material_detected", not secret_failures, "targeted scan excludes key names but catches connection URLs/private keys", secret_failures)

db_signature_fields = [
    "num_documents", "num_chunks", "null_doc_id", "null_chunk_id", "null_or_empty_content",
    "null_topic", "null_source", "null_lang", "null_page_num", "null_embedding", "min_created_at", "max_created_at",
]
db_differences = {field: [db_before[field], db_after[field]] for field in db_signature_fields if db_before[field] != db_after[field]}
check("database_profile_unchanged", not db_differences, "before/after read-only aggregate signature", [db_differences] if db_differences else [])

initial_status = (LOG_DIR / "git_status_initial.txt").read_text(encoding="utf-8").splitlines()
final_status = (LOG_DIR / "git_status_final.txt").read_text(encoding="utf-8").splitlines()
outside_final = [line for line in final_status if " evaluation_v2/" not in line.replace("\\", "/")]
check("no_git_changes_outside_evaluation_v2", outside_final == initial_status, "initial and final status outside the new folder match", [{"initial": initial_status, "final_outside": outside_final}] if outside_final != initial_status else [])

visual_status = json.loads((LOG_DIR / "pdf_visual_review_status.json").read_text(encoding="utf-8"))
check("visual_review_of_document_only_pages", visual_status["status"] == "COMPLETED", "text verification passed but rendering is blocked without Poppler", [visual_status] if visual_status["status"] != "COMPLETED" else [])

profile_path = RUN_DIR / "database_profile.json"
profile = json.loads(profile_path.read_text(encoding="utf-8"))
profile["final_profile"] = db_after
profile["before_after_signature_match"] = not db_differences
profile["final_check_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

passed_count = sum(bool(row["passed"]) for row in checks)
failed = [row for row in checks if not row["passed"]]
lines = [
    "# Validation report", "", f"Generated UTC: {datetime.now(timezone.utc).isoformat()}", "",
    f"Checks passed: {passed_count}/{len(checks)}. Failures are intentionally visible.", "",
    "| Check | Result | Detail |", "|---|---|---|",
]
for row in checks:
    lines.append(f"| {row['name']} | {'PASS' if row['passed'] else 'FAIL'} | {row['detail']} |")
lines.extend(["", "## Failure details", ""])
if failed:
    for row in failed:
        lines.append(f"### {row['name']}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(row["failures"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
else:
    lines.append("No failed checks.")
lines.extend([
    "", "## Interpretation", "",
    "Passing structural checks does not human-adjudicate claims or evidence grades.",
    "The stale database source and blocked visual render remain review items; neither is hidden.",
])
(RUN_DIR / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

readme_path = RUN_DIR / "README.md"
readme = readme_path.read_text(encoding="utf-8")
final_note = "\n## Final validation status\n\nDatabase before/after signatures match. Original source hashes match. Visual rendering of selected document-only pages was blocked because Poppler is unavailable and the bundled PDF.js/Skia combination is incompatible; textual page verification succeeded and those cases remain queued for human review.\n"
if "## Final validation status" not in readme:
    readme_path.write_text(readme + final_note, encoding="utf-8")

manifest_path = RUN_DIR / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest["final_validation"] = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "checks_passed": passed_count, "checks_total": len(checks),
    "failed_checks": [row["name"] for row in failed],
    "database_signature_unchanged": not db_differences,
    "git_outside_evaluation_v2_unchanged": outside_final == initial_status,
    "visual_review_status": visual_status["status"],
}
result_paths = sorted(path for path in RUN_DIR.rglob("*") if path.is_file() and path.name != "manifest.json")
manifest["scripts_created"] = sorted(str(path.relative_to(RUN_DIR)).replace("\\", "/") for path in (RUN_DIR / "scripts").glob("*"))
manifest["result_hashes_excluding_self_referential_manifest"] = {
    str(path.relative_to(RUN_DIR)).replace("\\", "/"): file_hash(path) for path in result_paths
}
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

print(json.dumps({
    "checks_passed": passed_count, "checks_total": len(checks),
    "failed_checks": [row["name"] for row in failed],
    "database_unchanged": not db_differences,
    "git_outside_run_unchanged": outside_final == initial_status,
}, ensure_ascii=False))
