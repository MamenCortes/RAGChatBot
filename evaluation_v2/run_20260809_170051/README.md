# Evaluation evidence corpus v2

**Status:** corpus de evidencia provisional generado y verificado automáticamente, pendiente de revisión y adjudicación humana.

## What was done

The 31 selected questions were frozen from `eval/preguntas_val.xlsx`. Expected answers were reconstructed exactly as the original evaluation notebook does, from ordered `expected_text` rows in `eval/gold_references.csv`. They were split conservatively into 155 machine-generated claims.

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
| Q001 | 1 | 42 | 1 | 1 | 0 | no |
| Q002 | 5 | 56 | 7 | 5 | 0 | yes |
| Q003 | 1 | 30 | 1 | 1 | 0 | no |
| Q004 | 1 | 11 | 1 | 1 | 0 | no |
| Q005 | 1 | 32 | 3 | 1 | 0 | yes |
| Q006 | 1 | 63 | 2 | 1 | 0 | yes |
| Q007 | 2 | 38 | 2 | 2 | 0 | no |
| Q008 | 3 | 50 | 4 | 3 | 0 | yes |
| Q009 | 2 | 44 | 2 | 2 | 0 | yes |
| Q010 | 5 | 43 | 6 | 5 | 0 | no |
| Q011 | 2 | 35 | 2 | 2 | 0 | yes |
| Q012 | 1 | 32 | 3 | 1 | 0 | yes |
| Q013 | 11 | 75 | 15 | 11 | 0 | yes |
| Q014 | 2 | 42 | 3 | 2 | 0 | yes |
| Q015 | 10 | 89 | 30 | 10 | 0 | yes |
| Q016 | 6 | 53 | 6 | 6 | 0 | yes |
| Q017 | 1 | 15 | 1 | 1 | 0 | no |
| Q018 | 6 | 59 | 16 | 6 | 0 | yes |
| Q019 | 5 | 71 | 13 | 5 | 0 | yes |
| Q020 | 15 | 131 | 25 | 15 | 0 | yes |
| Q021 | 7 | 64 | 6 | 6 | 1 | yes |
| Q022 | 2 | 49 | 1 | 2 | 0 | no |
| Q023 | 3 | 47 | 3 | 3 | 0 | yes |
| Q024 | 1 | 25 | 1 | 1 | 0 | no |
| Q025 | 8 | 85 | 8 | 8 | 0 | yes |
| Q026 | 32 | 135 | 33 | 30 | 2 | yes |
| Q027 | 4 | 37 | 1 | 4 | 0 | no |
| Q028 | 11 | 84 | 19 | 11 | 0 | yes |
| Q029 | 1 | 22 | 1 | 1 | 0 | no |
| Q030 | 3 | 53 | 6 | 3 | 0 | yes |
| Q031 | 2 | 31 | 2 | 2 | 0 | yes |

## Final validation status

Database before/after signatures match. Original source hashes match. Visual rendering of selected document-only pages was blocked because Poppler is unavailable and the bundled PDF.js/Skia combination is incompatible; textual page verification succeeded and those cases remain queued for human review.
