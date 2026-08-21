# Annotation guidelines

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
