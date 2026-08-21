import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import * as pdfjsLib from "file:///C:/Users/mamen/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/pdfjs-dist/legacy/build/pdf.mjs";
import { Canvas } from "file:///C:/Users/mamen/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/node_modules/skia-canvas/lib/index.mjs";

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const runDir = path.resolve(scriptDir, "..");
const projectRoot = path.resolve(runDir, "..", "..");
const outputDir = path.join(runDir, "logs", "pdf_renders");

const targets = [
  ["docs/general/general_es_pfizer_manual-pacientes_2007.pdf", 34, "Q002_fever_recommendation"],
  ["docs/general/general_es_pfizer_manual-pacientes_2007.pdf", 98, "Q016_diet_list"],
  ["docs/general/general_es_pfizer_manual-pacientes_2007.pdf", 169, "Q026_risk_factor_list"],
  ["docs/mama/mama_es_HUReinaSofia_protocolo-cancer-mama_2021.pdf", 38, "Q018_PAM50"],
  ["docs/mama/mama_es_HUReinaSofia_protocolo-cancer-mama_2021.pdf", 88, "Q020_tamoxifen_recommendation"],
  ["docs/mama/mama_en_SEOM-GEICAM-SOLTI_clinical-guidelines-hereditary-breast-ovarian-cancer_2019.pdf", 3, "Q028_genetic_table"],
];

await fs.mkdir(outputDir, { recursive: true });
const results = [];
for (const [relativePdf, pageNumber, label] of targets) {
  const inputPath = path.join(projectRoot, relativePdf);
  const bytes = new Uint8Array(await fs.readFile(inputPath));
  const pdf = await pdfjsLib.getDocument({ data: bytes, disableWorker: true }).promise;
  if (pageNumber < 1 || pageNumber > pdf.numPages) {
    throw new Error(`Page ${pageNumber} outside 1..${pdf.numPages} for ${relativePdf}`);
  }
  const page = await pdf.getPage(pageNumber);
  const viewport = page.getViewport({ scale: 1.5 });
  const canvas = new Canvas(Math.ceil(viewport.width), Math.ceil(viewport.height));
  const context = canvas.getContext("2d");
  const originalFill = context.fill.bind(context);
  context.fill = (fillRule) => originalFill(fillRule === "evenodd" ? "evenodd" : "nonzero");
  const originalClip = context.clip.bind(context);
  context.clip = (fillRule) => originalClip(fillRule === "evenodd" ? "evenodd" : "nonzero");
  await page.render({ canvasContext: context, viewport }).promise;
  const outputPath = path.join(outputDir, `${label}_page_${String(pageNumber).padStart(3, "0")}.png`);
  await fs.writeFile(outputPath, await canvas.png);
  results.push({ relativePdf, pageNumber, output: path.relative(runDir, outputPath).replaceAll("\\", "/") });
  await pdf.destroy();
}

await fs.writeFile(
  path.join(outputDir, "render_manifest.json"),
  JSON.stringify({ renderer: "pdfjs-dist + skia-canvas", scale: 1.5, targets: results }, null, 2),
  "utf8",
);
console.log(JSON.stringify({ rendered: results.length, outputDir }));
