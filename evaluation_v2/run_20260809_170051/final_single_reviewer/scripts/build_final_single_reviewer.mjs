import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const outputDir = path.resolve(scriptDir, "..");
const runDir = path.resolve(outputDir, "..");
const repoDir = path.resolve(runDir, "..", "..");
const docsDir = path.join(repoDir, "docs");
const gitExe = "C:\\Users\\mamen\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\native\\git\\cmd\\git.exe";

const inputFiles = {
  questions: "questions.csv",
  claims: "claims.csv",
  reviewed: "human_review_queue_reviewed.csv",
  provisional: "evidence_set_provisional.csv",
  candidates: "evidence_candidates.csv",
};

const outputFiles = {
  revisionMap: "claim_revision_map.csv",
  claimsFinal: "claims_final_single_reviewer.csv",
  evidenceFinal: "evidence_set_final_single_reviewer.csv",
  evidenceLinks: "evidence_claim_links_final_single_reviewer.csv",
  coverageFinal: "claim_coverage_final_single_reviewer.csv",
  unresolved: "unresolved_claims_final_single_reviewer.csv",
  validation: "validation_report.md",
  readme: "README.md",
  manifest: "manifest_final_single_reviewer.json",
  workbook: "evaluation_final_single_reviewer.xlsx",
};

function clean(value) {
  return value === null || value === undefined ? "" : String(value).trim();
}

function boolText(value) {
  return /^(true|1|yes|sí|si)$/i.test(clean(value)) ? "true" : "false";
}

function sha256Text(value) {
  return crypto.createHash("sha256").update(String(value), "utf8").digest("hex");
}

async function sha256File(filePath) {
  return crypto.createHash("sha256").update(await fs.readFile(filePath)).digest("hex");
}

async function readCsvObjects(filePath) {
  const csvText = await fs.readFile(filePath, "utf8");
  const workbook = await Workbook.fromCSV(csvText, { sheetName: "Source" });
  const sheet = workbook.worksheets.getItemAt(0);
  const used = sheet.getUsedRange(true);
  const values = used?.values ?? [];
  if (!values.length) return [];
  const headers = values[0].map((v, i) => clean(v).replace(/^\uFEFF/, "") || `column_${i + 1}`);
  return values.slice(1)
    .filter((row) => row.some((value) => clean(value) !== ""))
    .map((row) => Object.fromEntries(headers.map((header, i) => [header, row[i] ?? ""])));
}

function csvEscape(value) {
  const text = value === null || value === undefined ? "" : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

async function writeCsv(filePath, rows, columns) {
  const lines = [columns.map(csvEscape).join(",")];
  for (const row of rows) lines.push(columns.map((column) => csvEscape(row[column] ?? "")).join(","));
  await fs.writeFile(filePath, `\uFEFF${lines.join("\r\n")}\r\n`, "utf8");
}

function parseClaimIds(value) {
  return [...new Set(clean(value).split(/[;,]/).map(clean).filter(Boolean))];
}

function parseClaimAction(note) {
  const text = clean(note);
  const rewrite = text.match(/^rewrite\s+to\s*:\s*(.+)$/i);
  if (rewrite) return { action: "rewrite", replacement: clean(rewrite[1]) };
  if (/^reject(?:\b|\.)/i.test(text)) return { action: "reject", replacement: "" };
  if (/^keep$/i.test(text)) return { action: "keep", replacement: "" };
  return { action: "unparsed", replacement: "" };
}

function numericGrade(value) {
  const number = Number(clean(value));
  return Number.isFinite(number) ? number : null;
}

function key(questionId, evidenceId) {
  return `${clean(questionId)}\u0000${clean(evidenceId)}`;
}

function columnName(index) {
  let name = "";
  let value = index + 1;
  while (value > 0) {
    const remainder = (value - 1) % 26;
    name = String.fromCharCode(65 + remainder) + name;
    value = Math.floor((value - 1) / 26);
  }
  return name;
}

function gitStatus() {
  try {
    return execFileSync(gitExe, ["status", "--porcelain=v1"], { cwd: repoDir, encoding: "utf8" })
      .split(/\r?\n/).filter(Boolean);
  } catch (error) {
    return [`NOT AVAILABLE: ${error.message}`];
  }
}

async function listDocBasenames(root) {
  const names = new Set();
  async function walk(directory) {
    for (const entry of await fs.readdir(directory, { withFileTypes: true })) {
      const fullPath = path.join(directory, entry.name);
      if (entry.isDirectory()) await walk(fullPath);
      else if (entry.isFile()) names.add(entry.name.toLowerCase());
    }
  }
  try { await walk(root); } catch { /* docs unavailable is reported by source checks */ }
  return names;
}

const sourcePaths = Object.fromEntries(
  Object.entries(inputFiles).map(([name, file]) => [name, path.join(runDir, file)]),
);
const inputHashesBefore = Object.fromEntries(
  await Promise.all(Object.entries(sourcePaths).map(async ([name, file]) => [name, await sha256File(file)])),
);
const gitStatusBefore = gitStatus();

const [questions, claims, reviewed, provisional, candidates] = await Promise.all([
  readCsvObjects(sourcePaths.questions),
  readCsvObjects(sourcePaths.claims),
  readCsvObjects(sourcePaths.reviewed),
  readCsvObjects(sourcePaths.provisional),
  readCsvObjects(sourcePaths.candidates),
]);

const questionIds = new Set(questions.map((row) => clean(row.question_id)));
const claimById = new Map(claims.map((row) => [clean(row.claim_id), row]));
const segmentationRows = reviewed.filter((row) => clean(row.issue_type) === "claim_segmentation_review");
const segmentationByClaim = new Map(segmentationRows.map((row) => [clean(row.claim_id), row]));

const revisionMap = claims.map((claim) => {
  const claimId = clean(claim.claim_id);
  const review = segmentationByClaim.get(claimId);
  const parsed = review ? parseClaimAction(review.human_notes) : { action: "not_reviewed", replacement: "" };
  const active = parsed.action !== "reject";
  const finalText = parsed.action === "rewrite" ? parsed.replacement : clean(claim.claim_text);
  return {
    question_id: clean(claim.question_id),
    original_claim_id: claimId,
    original_claim_text: clean(claim.claim_text),
    review_action: parsed.action,
    final_claim_id: active ? claimId : "",
    final_claim_text: active ? finalText : "",
    required_for_complete_answer: active ? boolText(claim.required_for_complete_answer) : "false",
    reason: review ? clean(review.human_notes) : "No estaba marcado para revisión de segmentación.",
    evidence_revalidation_required: parsed.action === "rewrite" ? "true" : "false",
    status: active ? "active" : "retired",
    reviewer_type: review ? "single_human_reviewer" : "not_human_reviewed",
  };
});

const revisionByOriginalId = new Map(revisionMap.map((row) => [row.original_claim_id, row]));
const activeClaimIds = new Set(revisionMap.filter((row) => row.status === "active").map((row) => row.final_claim_id));
const rewrittenClaimIds = new Set(revisionMap.filter((row) => row.review_action === "rewrite").map((row) => row.final_claim_id));
const retiredClaimIds = new Set(revisionMap.filter((row) => row.status === "retired").map((row) => row.original_claim_id));

const claimsFinal = revisionMap.filter((row) => row.status === "active").map((revision) => {
  const original = claimById.get(revision.original_claim_id);
  return {
    question_id: revision.question_id,
    claim_id: revision.final_claim_id,
    claim_text: revision.final_claim_text,
    claim_type: clean(original.claim_type),
    required_for_complete_answer: revision.required_for_complete_answer,
    source_ground_truth: clean(original.source_ground_truth),
    original_claim_id: revision.original_claim_id,
    review_action: revision.review_action,
    review_notes: revision.reason,
    status: "active",
    reviewer_type: revision.reviewer_type,
  };
});
const finalClaimById = new Map(claimsFinal.map((row) => [row.claim_id, row]));

const evidenceReviewRows = reviewed.filter((row) => clean(row.issue_type) === "machine_evidence_judgment_review");
const evidenceReviewByKey = new Map(evidenceReviewRows.map((row) => [key(row.question_id, row.evidence_id), row]));
const candidateByKey = new Map(candidates.map((row) => [key(row.question_id, row.evidence_id), row]));
const docsBasenames = await listDocBasenames(docsDir);

const evidenceFinal = [];
const excludedEvidence = [];
for (const source of provisional) {
  const evidenceKey = key(source.question_id, source.evidence_id);
  const human = evidenceReviewByKey.get(evidenceKey);
  const candidate = candidateByKey.get(evidenceKey);
  let grade;
  let coveredClaimIds;
  let reviewStatus;
  let decision;
  let confidence;
  let status;
  let reviewerType;
  let reviewNotes;

  if (human) {
    grade = numericGrade(human.human_grade);
    const humanDecision = clean(human.human_decision).toLowerCase();
    if (humanDecision !== "accept" || grade === null || grade < 2) {
      excludedEvidence.push({
        question_id: clean(source.question_id), evidence_id: clean(source.evidence_id),
        exclusion_reason: `human_${humanDecision || "missing_decision"}_grade_${grade ?? "missing"}`,
      });
      continue;
    }
    coveredClaimIds = parseClaimIds(human.human_covered_claims);
    reviewStatus = "human_reviewed";
    decision = "accept";
    confidence = "human_confidence_not_recorded";
    status = "human_reviewed_single_reviewer";
    reviewerType = "single_human_reviewer";
    reviewNotes = clean(human.human_notes);
  } else {
    grade = numericGrade(source.relevance_grade);
    coveredClaimIds = parseClaimIds(source.covered_claim_ids);
    reviewStatus = "machine_only_unreviewed";
    decision = "accept_provisional";
    confidence = clean(source.confidence);
    status = "provisional_machine_judgment_not_human_reviewed";
    reviewerType = "machine_proposed";
    reviewNotes = clean(source.notes);
  }

  const originalClaims = [...coveredClaimIds];
  coveredClaimIds = coveredClaimIds.filter((claimId) => activeClaimIds.has(claimId));
  if (grade === null || grade < 2 || coveredClaimIds.length === 0) {
    excludedEvidence.push({
      question_id: clean(source.question_id), evidence_id: clean(source.evidence_id),
      exclusion_reason: coveredClaimIds.length === 0 ? "no_active_covered_claim" : "grade_below_2",
    });
    continue;
  }

  const rewrittenLinks = coveredClaimIds.filter((claimId) => rewrittenClaimIds.has(claimId));
  const sourceName = path.basename(clean(source.source)).toLowerCase();
  const sourceExists = sourceName ? docsBasenames.has(sourceName) : false;
  evidenceFinal.push({
    question_id: clean(source.question_id),
    evidence_id: clean(source.evidence_id),
    chunk_id: clean(source.chunk_id),
    doc_id: clean(source.doc_id),
    page_num: clean(source.page_num),
    source: clean(source.source),
    lang: clean(source.lang),
    evidence_text: clean(source.evidence_text),
    relevance_grade: grade,
    covered_claim_ids: coveredClaimIds.join(";"),
    original_covered_claim_ids: originalClaims.join(";"),
    review_status: reviewStatus,
    decision,
    confidence,
    evidence_origin: clean(source.evidence_origin),
    retrieved_by: clean(source.retrieved_by),
    source_content_sha256: clean(source.content_sha256),
    evidence_text_sha256: sha256Text(clean(source.evidence_text)),
    source_exists_in_docs: sourceExists ? "true" : "false",
    ingestion_gap: boolText(candidate?.ingestion_gap ?? (clean(source.evidence_origin) === "document_only")),
    claim_text_changed_since_machine_judgment: rewrittenLinks.length ? "true" : "false",
    needs_human_revalidation: reviewStatus === "machine_only_unreviewed" ? "true" : "false",
    status,
    reviewer_type: reviewerType,
    adjudication_status: "not_adjudicated",
    review_notes: reviewNotes,
  });
}

evidenceFinal.sort((a, b) => a.question_id.localeCompare(b.question_id) || a.evidence_id.localeCompare(b.evidence_id));

const evidenceLinks = [];
for (const evidence of evidenceFinal) {
  for (const claimId of parseClaimIds(evidence.covered_claim_ids)) {
    evidenceLinks.push({
      question_id: evidence.question_id,
      evidence_id: evidence.evidence_id,
      claim_id: claimId,
      relevance_grade: evidence.relevance_grade,
      support_type: Number(evidence.relevance_grade) === 3 ? "direct_and_sufficient" : "substantial_partial",
      review_status: evidence.review_status,
      reviewer_type: evidence.reviewer_type,
      status: evidence.status,
      adjudication_status: "not_adjudicated",
    });
  }
}

const linksByClaim = new Map();
for (const link of evidenceLinks) {
  if (!linksByClaim.has(link.claim_id)) linksByClaim.set(link.claim_id, []);
  linksByClaim.get(link.claim_id).push(link);
}

const coverageFinal = claimsFinal.map((claim) => {
  const links = linksByClaim.get(claim.claim_id) ?? [];
  const human = links.filter((row) => row.review_status === "human_reviewed");
  const machine = links.filter((row) => row.review_status === "machine_only_unreviewed");
  const maxHuman = human.length ? Math.max(...human.map((row) => Number(row.relevance_grade))) : 0;
  const maxMachine = machine.length ? Math.max(...machine.map((row) => Number(row.relevance_grade))) : 0;
  const best = Math.max(maxHuman, maxMachine);
  let coverageStatus;
  if (maxHuman >= 3) coverageStatus = "fully_supported_human_reviewed";
  else if (maxHuman === 2 && maxMachine >= 3) coverageStatus = "fully_supported_mixed_human_partial";
  else if (maxMachine >= 3) coverageStatus = "fully_supported_machine_only";
  else if (maxHuman === 2) coverageStatus = "partially_supported_human_reviewed";
  else if (maxMachine === 2) coverageStatus = "partially_supported_machine_only";
  else coverageStatus = "unsupported";
  const needsHumanReview = maxHuman < 3;
  return {
    question_id: claim.question_id,
    claim_id: claim.claim_id,
    claim_text: claim.claim_text,
    evidence_ids: [...new Set(links.map((row) => row.evidence_id))].join(";"),
    num_supporting_evidences: new Set(links.map((row) => row.evidence_id)).size,
    num_human_reviewed_evidences: new Set(human.map((row) => row.evidence_id)).size,
    num_machine_only_evidences: new Set(machine.map((row) => row.evidence_id)).size,
    best_relevance_grade: best || "",
    covered: best >= 3 ? "true" : "false",
    coverage_status: coverageStatus,
    needs_human_review: needsHumanReview ? "true" : "false",
    needs_additional_evidence: best < 3 ? "true" : "false",
    claim_was_rewritten: rewrittenClaimIds.has(claim.claim_id) ? "true" : "false",
    notes: needsHumanReview && machine.length ? "La mejor cobertura disponible incluye juicio automático no revisado." : "",
  };
});

const unresolved = coverageFinal.filter((row) => row.needs_human_review === "true" || row.needs_additional_evidence === "true").map((row) => ({
  question_id: row.question_id,
  claim_id: row.claim_id,
  claim_text: row.claim_text,
  coverage_status: row.coverage_status,
  evidence_ids: row.evidence_ids,
  best_relevance_grade: row.best_relevance_grade,
  pending_reason: row.coverage_status === "unsupported"
    ? "no_supporting_evidence_in_final_set"
    : row.coverage_status.includes("machine") || row.coverage_status.includes("mixed")
      ? "machine_evidence_requires_human_review"
      : "partial_evidence_requires_additional_support",
  recommended_action: row.coverage_status === "unsupported"
    ? "Buscar evidencia adicional o revisar el ground truth."
    : row.coverage_status.includes("machine") || row.coverage_status.includes("mixed")
      ? "Revisar manualmente las evidencias machine_only_unreviewed enlazadas."
      : "Buscar evidencia complementaria directa.",
}));

const revisionColumns = ["question_id", "original_claim_id", "original_claim_text", "review_action", "final_claim_id", "final_claim_text", "required_for_complete_answer", "reason", "evidence_revalidation_required", "status", "reviewer_type"];
const claimsColumns = ["question_id", "claim_id", "claim_text", "claim_type", "required_for_complete_answer", "source_ground_truth", "original_claim_id", "review_action", "review_notes", "status", "reviewer_type"];
const evidenceColumns = ["question_id", "evidence_id", "chunk_id", "doc_id", "page_num", "source", "lang", "evidence_text", "relevance_grade", "covered_claim_ids", "original_covered_claim_ids", "review_status", "decision", "confidence", "evidence_origin", "retrieved_by", "source_content_sha256", "evidence_text_sha256", "source_exists_in_docs", "ingestion_gap", "claim_text_changed_since_machine_judgment", "needs_human_revalidation", "status", "reviewer_type", "adjudication_status", "review_notes"];
const linkColumns = ["question_id", "evidence_id", "claim_id", "relevance_grade", "support_type", "review_status", "reviewer_type", "status", "adjudication_status"];
const coverageColumns = ["question_id", "claim_id", "claim_text", "evidence_ids", "num_supporting_evidences", "num_human_reviewed_evidences", "num_machine_only_evidences", "best_relevance_grade", "covered", "coverage_status", "needs_human_review", "needs_additional_evidence", "claim_was_rewritten", "notes"];
const unresolvedColumns = ["question_id", "claim_id", "claim_text", "coverage_status", "evidence_ids", "best_relevance_grade", "pending_reason", "recommended_action"];

await writeCsv(path.join(outputDir, outputFiles.revisionMap), revisionMap, revisionColumns);
await writeCsv(path.join(outputDir, outputFiles.claimsFinal), claimsFinal, claimsColumns);
await writeCsv(path.join(outputDir, outputFiles.evidenceFinal), evidenceFinal, evidenceColumns);
await writeCsv(path.join(outputDir, outputFiles.evidenceLinks), evidenceLinks, linkColumns);
await writeCsv(path.join(outputDir, outputFiles.coverageFinal), coverageFinal, coverageColumns);
await writeCsv(path.join(outputDir, outputFiles.unresolved), unresolved, unresolvedColumns);

const workbook = Workbook.create();
const sheetSpecs = [
  ["Claims Final", claimsFinal, claimsColumns],
  ["Evidence Final", evidenceFinal, evidenceColumns],
  ["Evidence Claim Links", evidenceLinks, linkColumns],
  ["Claim Coverage", coverageFinal, coverageColumns],
  ["Claim Revisions", revisionMap, revisionColumns],
  ["Pending Review", unresolved, unresolvedColumns],
];

for (const [sheetName, rows, columns] of sheetSpecs) {
  const sheet = workbook.worksheets.add(sheetName);
  const matrix = [columns, ...rows.map((row) => columns.map((column) => row[column] ?? ""))];
  const range = sheet.getRangeByIndexes(0, 0, matrix.length, columns.length);
  range.values = matrix;
  sheet.getRangeByIndexes(0, 0, 1, columns.length).format = {
    fill: "#1F4E78",
    font: { bold: true, color: "#FFFFFF" },
    wrapText: true,
  };
  sheet.freezePanes.freezeRows(1);
  sheet.showGridLines = false;
  range.format.autofitColumns();
  for (let column = 0; column < columns.length; column += 1) {
    const columnRange = sheet.getRangeByIndexes(0, column, matrix.length, 1);
    const header = columns[column];
    if (/text|notes|reason|source_ground_truth|recommended_action/.test(header)) {
      columnRange.format.columnWidth = 42;
      columnRange.format.wrapText = true;
    } else if (columnRange.format.columnWidth > 28) {
      columnRange.format.columnWidth = 28;
    }
  }
}

const workbookOutput = await SpreadsheetFile.exportXlsx(workbook);
await workbookOutput.save(path.join(outputDir, outputFiles.workbook));

const previewDir = path.join(scriptDir, "previews");
await fs.mkdir(previewDir, { recursive: true });
const previews = [];
for (const [sheetName, rows, columns] of sheetSpecs) {
  const lastColumn = columnName(Math.min(columns.length, 8) - 1);
  const lastRow = Math.min(rows.length + 1, 8);
  const blob = await workbook.render({ sheetName, range: `A1:${lastColumn}${lastRow}`, scale: 1, format: "png" });
  const previewPath = path.join(previewDir, `${sheetName.replaceAll(" ", "_").toLowerCase()}.png`);
  await fs.writeFile(previewPath, new Uint8Array(await blob.arrayBuffer()));
  previews.push(path.relative(outputDir, previewPath).replaceAll("\\", "/"));
}

const evidenceKeySet = new Set(evidenceFinal.map((row) => key(row.question_id, row.evidence_id)));
const linkEvidenceKeySet = new Set(evidenceLinks.map((row) => key(row.question_id, row.evidence_id)));
const validations = [
  ["Todos los question_id de claims existen", claimsFinal.every((row) => questionIds.has(row.question_id))],
  ["Todos los question_id de evidencias existen", evidenceFinal.every((row) => questionIds.has(row.question_id))],
  ["Todos los claim_id enlazados están activos", evidenceLinks.every((row) => activeClaimIds.has(row.claim_id))],
  ["Toda evidencia final tiene texto", evidenceFinal.every((row) => clean(row.evidence_text) !== "")],
  ["Toda evidencia final tiene grado 2 o 3", evidenceFinal.every((row) => [2, 3].includes(Number(row.relevance_grade)))],
  ["Toda evidencia final cubre al menos un claim activo", evidenceFinal.every((row) => parseClaimIds(row.covered_claim_ids).length > 0)],
  ["No hay evidencias duplicadas por pregunta", evidenceKeySet.size === evidenceFinal.length],
  ["Toda evidencia tiene al menos un enlace", evidenceFinal.every((row) => linkEvidenceKeySet.has(key(row.question_id, row.evidence_id)))],
  ["No se incluyeron decisiones humanas reject/uncertain", evidenceFinal.filter((row) => row.review_status === "human_reviewed").every((row) => row.decision === "accept")],
  ["No se incluyeron claims retirados", evidenceLinks.every((row) => !retiredClaimIds.has(row.claim_id))],
  ["No hay columnas de embeddings o secretos", !evidenceColumns.some((column) => /embedding|password|token|secret|api_key/i.test(column))],
  ["Los claims reescritos tienen texto final", revisionMap.filter((row) => row.review_action === "rewrite").every((row) => clean(row.final_claim_text) !== "")],
  ["No quedan notas de segmentación sin interpretar", revisionMap.every((row) => row.review_action !== "unparsed")],
];

const humanEvidenceCount = evidenceFinal.filter((row) => row.review_status === "human_reviewed").length;
const machineEvidenceCount = evidenceFinal.filter((row) => row.review_status === "machine_only_unreviewed").length;
const fullySupportedCount = coverageFinal.filter((row) => row.covered === "true").length;
const humanFullySupportedCount = coverageFinal.filter((row) => row.coverage_status === "fully_supported_human_reviewed").length;
const missingSourceCount = evidenceFinal.filter((row) => row.source_exists_in_docs === "false").length;

const validationText = `# Validación de los documentos finales\n\n` +
  `Fecha UTC: ${new Date().toISOString()}\n\n` +
  `Este informe valida el corpus final de un único revisor. No equivale a adjudicación externa.\n\n` +
  validations.map(([name, passed]) => `- ${passed ? "PASS" : "FAIL"}: ${name}`).join("\n") +
  `\n\n## Recuentos de control\n\n` +
  `- Claims originales: ${claims.length}\n` +
  `- Claims activos: ${claimsFinal.length}\n` +
  `- Claims retirados: ${retiredClaimIds.size}\n` +
  `- Claims reescritos: ${rewrittenClaimIds.size}\n` +
  `- Evidencias finales revisadas por una persona: ${humanEvidenceCount}\n` +
  `- Evidencias finales machine-only no revisadas: ${machineEvidenceCount}\n` +
  `- Evidencias provisionales excluidas por revisión humana o falta de claim activo: ${excludedEvidence.length}\n` +
  `- Evidencias con fuente no localizada por nombre en docs/: ${missingSourceCount}\n` +
  `- Claims totalmente cubiertos por cualquier evidencia final: ${fullySupportedCount}/${claimsFinal.length}\n` +
  `- Claims totalmente cubiertos por evidencia revisada por una persona: ${humanFullySupportedCount}/${claimsFinal.length}\n\n` +
  `## Limitaciones\n\n` +
  `- Las evidencias machine_only_unreviewed conservan la anotación automática provisional y requieren revisión humana futura.\n` +
  `- No se realizó adjudicación independiente.\n` +
  `- Una fuente no localizada en docs/ puede seguir siendo un chunk verificable de base de datos; este proceso no volvió a consultar la base de datos.\n` +
  `- Los claims reescritos enlazados solo con evidencia automática deben volver a revisarse contra el nuevo texto.\n`;
await fs.writeFile(path.join(outputDir, outputFiles.validation), validationText, "utf8");

const summaryByQuestion = questions.map((question) => {
  const qid = clean(question.question_id);
  const qClaims = claimsFinal.filter((row) => row.question_id === qid);
  const qEvidence = evidenceFinal.filter((row) => row.question_id === qid);
  const qCoverage = coverageFinal.filter((row) => row.question_id === qid);
  return {
    question_id: qid,
    num_claims: qClaims.length,
    num_evidences: qEvidence.length,
    human_reviewed_evidences: qEvidence.filter((row) => row.review_status === "human_reviewed").length,
    machine_only_evidences: qEvidence.filter((row) => row.review_status === "machine_only_unreviewed").length,
    fully_supported_claims: qCoverage.filter((row) => row.covered === "true").length,
    pending_human_review: qCoverage.filter((row) => row.needs_human_review === "true").length,
  };
});

const readmeText = `# Evidence set final — revisión de una sola persona\n\n` +
  `## Qué contiene\n\n` +
  `Este directorio contiene una versión final operativa construida sin modificar los archivos anteriores. Los claims marcados con \`rewrite to:\` se sustituyeron conservando su ID; los marcados con \`reject\` se retiraron y permanecen trazables en \`${outputFiles.revisionMap}\`.\n\n` +
  `El evidence set combina dos niveles de revisión:\n\n` +
  `- \`human_reviewed\`: evidencia aceptada manualmente con grado 2 o 3.\n` +
  `- \`machine_only_unreviewed\`: evidencia provisional automática que no apareció en la cola revisada.\n\n` +
  `Las decisiones humanas \`reject\` o \`uncertain\` no forman parte del evidence set. Ninguna fila se etiqueta como adjudicada.\n\n` +
  `## Archivo que debe usarse como evidence set\n\n` +
  `Utiliza \`${outputFiles.evidenceFinal}\`. Para evaluación por claims, acompáñalo de \`${outputFiles.claimsFinal}\` y \`${outputFiles.evidenceLinks}\`.\n\n` +
  `## Recuentos\n\n` +
  `- Preguntas: ${questions.length}\n` +
  `- Claims activos: ${claimsFinal.length}\n` +
  `- Claims reescritos: ${rewrittenClaimIds.size}\n` +
  `- Claims retirados: ${retiredClaimIds.size}\n` +
  `- Evidencias finales: ${evidenceFinal.length}\n` +
  `- Revisadas y aceptadas por una persona: ${humanEvidenceCount}\n` +
  `- Machine-only no revisadas: ${machineEvidenceCount}\n` +
  `- Claims totalmente cubiertos: ${fullySupportedCount}/${claimsFinal.length}\n` +
  `- Casos pendientes de revisión o evidencia adicional: ${unresolved.length}\n\n` +
  `## Resumen por pregunta\n\n` +
  `| question_id | claims | evidencias | revisadas | machine-only | claims cubiertos | pendientes |\n` +
  `|---|---:|---:|---:|---:|---:|---:|\n` +
  summaryByQuestion.map((row) => `| ${row.question_id} | ${row.num_claims} | ${row.num_evidences} | ${row.human_reviewed_evidences} | ${row.machine_only_evidences} | ${row.fully_supported_claims} | ${row.pending_human_review} |`).join("\n") +
  `\n\n## Limitación principal\n\n` +
  `Este resultado es un corpus de evidencia provisional generado y verificado automáticamente, complementado con revisión de una sola persona y pendiente de adjudicación humana independiente. Las filas \`machine_only_unreviewed\` no deben presentarse como revisadas manualmente.\n`;
await fs.writeFile(path.join(outputDir, outputFiles.readme), readmeText, "utf8");

const inputHashesAfter = Object.fromEntries(
  await Promise.all(Object.entries(sourcePaths).map(async ([name, file]) => [name, await sha256File(file)])),
);
const originalsUnchanged = Object.keys(inputHashesBefore).every((name) => inputHashesBefore[name] === inputHashesAfter[name]);
const gitStatusAfter = gitStatus();

const outputHashNames = [
  outputFiles.revisionMap, outputFiles.claimsFinal, outputFiles.evidenceFinal,
  outputFiles.evidenceLinks, outputFiles.coverageFinal, outputFiles.unresolved,
  outputFiles.validation, outputFiles.readme, outputFiles.workbook,
];
const outputHashes = Object.fromEntries(
  await Promise.all(outputHashNames.map(async (name) => [name, await sha256File(path.join(outputDir, name))])),
);

const manifest = {
  generated_at_utc: new Date().toISOString(),
  corpus_status: "single_human_reviewer_plus_machine_only_unreviewed_not_adjudicated",
  source_run: runDir,
  output_directory: outputDir,
  inputs: Object.fromEntries(Object.entries(sourcePaths).map(([name, file]) => [name, { path: file, sha256: inputHashesBefore[name] }])),
  originals_unchanged_after_generation: originalsUnchanged,
  git_status_before: gitStatusBefore,
  git_status_after: gitStatusAfter,
  counts: {
    questions: questions.length,
    original_claims: claims.length,
    active_claims: claimsFinal.length,
    rewritten_claims: rewrittenClaimIds.size,
    retired_claims: retiredClaimIds.size,
    provisional_evidences_input: provisional.length,
    final_evidences: evidenceFinal.length,
    human_reviewed_accepted_evidences: humanEvidenceCount,
    machine_only_unreviewed_evidences: machineEvidenceCount,
    excluded_evidences: excludedEvidence.length,
    evidence_claim_links: evidenceLinks.length,
    fully_supported_claims_any_source: fullySupportedCount,
    fully_supported_claims_human_reviewed: humanFullySupportedCount,
    pending_review_or_evidence: unresolved.length,
    missing_sources_in_docs_by_basename: missingSourceCount,
  },
  inclusion_rules: {
    human_reviewed: "human_decision=accept and human_grade>=2 and at least one active covered claim",
    machine_only_unreviewed: "provisional evidence with no machine_evidence_judgment_review row and at least one active covered claim",
    excluded: "human reject/uncertain, grade below 2, or no active covered claim",
  },
  output_hashes_sha256: outputHashes,
  preview_files: previews,
  limitations: [
    "No independent adjudicator was used.",
    "Machine-only evidence remains provisional and requires human review.",
    "The database was not queried during finalization.",
    "Source existence was checked by filename against docs/; database-only chunks can still be valid.",
  ],
};
await fs.writeFile(path.join(outputDir, outputFiles.manifest), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");

console.log(JSON.stringify({
  outputDir,
  counts: manifest.counts,
  originalsUnchanged,
  validations: Object.fromEntries(validations),
  outputFiles,
  previewFiles: previews,
}, null, 2));
