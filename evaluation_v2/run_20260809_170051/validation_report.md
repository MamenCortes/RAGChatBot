# Validation report

Generated UTC: 2026-08-09T15:24:37.629583+00:00

Checks passed: 16/18. Failures are intentionally visible.

| Check | Result | Detail |
|---|---|---|
| required_outputs_exist | PASS | required output structure |
| all_question_ids_resolve | PASS | all references resolve to questions.csv |
| all_claim_ids_resolve | PASS | covered claims resolve to claims.csv |
| evidence_ids_unique_per_question | PASS | stable evidence identifiers |
| database_candidates_verify | PASS | chunk IDs and normalized content hashes match the read-only DB snapshot |
| provisional_database_evidence_hashes | PASS | full chunk hashes reproduce |
| provisional_document_evidence_text_verifies | PASS | saved excerpt appears on recorded PDF page |
| all_candidate_source_paths_exist | FAIL | stale DB paths are reported rather than hidden |
| original_source_hashes_unchanged | PASS | all consulted original hashes reproduce |
| accepted_evidence_has_text | PASS | no grade 2/3 evidence is textless |
| accepted_evidence_has_claims | PASS | all grade 2/3 rows cover claims |
| covered_claim_has_evidence | PASS | fully supported claims have evidence |
| human_fields_empty | PASS | no human/adjudication result was fabricated |
| no_embeddings_exported | PASS | no result table contains an embedding column |
| no_secret_material_detected | PASS | targeted scan excludes key names but catches connection URLs/private keys |
| database_profile_unchanged | PASS | before/after read-only aggregate signature |
| no_git_changes_outside_evaluation_v2 | PASS | initial and final status outside the new folder match |
| visual_review_of_document_only_pages | FAIL | text verification passed but rendering is blocked without Poppler |

## Failure details

### all_candidate_source_paths_exist

```json
[
  "C:\\Users\\mamen\\Documents\\Python\\RAGChatBot\\docs\\mama\\mama_en_SEOM-GEICAM-SOLTI_clinical-guidelines_2022.pdf"
]
```

### visual_review_of_document_only_pages

```json
[
  {
    "status": "NOT_COMPLETED",
    "textual_pdf_verification": "COMPLETED_WITH_PYPDF",
    "visual_rendering": "BLOCKED",
    "reason": "Poppler pdfinfo/pdftoppm are not installed. The available pdfjs-dist and skia-canvas versions have incompatible no-argument Canvas fill/stroke overloads.",
    "installation_attempted": false,
    "source_pdfs_modified": false,
    "targets_planned": 6,
    "targets_rendered": 0,
    "required_follow_up": "Human reviewer should visually inspect document_only evidence pages, especially tables and lists."
  }
]
```


## Interpretation

Passing structural checks does not human-adjudicate claims or evidence grades.
The stale database source and blocked visual render remain review items; neither is hidden.
