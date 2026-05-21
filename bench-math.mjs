#!/usr/bin/env node
/**
 * Math reasoning probe — two cells:
 *   gsm8k   — 50 sampled grade-school word problems (data/gsm8k.jsonl)
 *   math500 — 30 sampled MATH-500 competition problems (data/math500.jsonl)
 *
 * Both are deterministic-graded: extract the model's final answer, compare to
 * gold. No judge model.
 *
 * Usage:
 *   node bench-math.mjs [--model gemma4:26b] [--host http://ollama:11434]
 *                       [--out ./baseline.json] [--save|--compare]
 *                       [--cell gsm8k|math500|all] [-v|--verbose]
 *
 * Per-call timeout: 240s (MATH problems benefit from chain-of-thought, which
 * runs long on a slow model). Override: OLLAMA_BENCH_TIMEOUT_MS.
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { getModelSection, writeModelSection } from "./bench-baseline.mjs";
import { startSampler, stopSampler, fmtGpuSummary } from "./bench-gpu.mjs";

const ROOT = dirname(fileURLToPath(import.meta.url));

const args = process.argv.slice(2);
const arg = (n, fb) => { const i = args.lastIndexOf(n); return i >= 0 ? args[i + 1] : fb; };
const MODEL   = arg("--model", "gemma4:26b");
const HOST    = arg("--host",  "http://ollama:11434");
const OUT     = arg("--out",   "./baseline.json");
const CELL    = arg("--cell",  "all"); // gsm8k | math500 | all
const LIMIT   = (() => { const v = arg("--limit", null); return v ? parseInt(v, 10) : null; })();
const VERBOSE = args.includes("-v") || args.includes("--verbose");
const MODE    = args.includes("--save")    ? "save"
              : args.includes("--compare") ? "compare"
              : "smart";

const REG_PP = 5;

function genTimeoutMs() {
  const override = parseInt(process.env.OLLAMA_BENCH_TIMEOUT_MS ?? "", 10);
  if (Number.isFinite(override) && override > 0) return override;
  return 240_000;
}
function withTimeout(ms) {
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), ms);
  return { signal: ac.signal, cancel: () => clearTimeout(timer) };
}

function loadJsonl(path) {
  return readFileSync(path, "utf-8").trim().split("\n").map(l => JSON.parse(l));
}

// ── Cells ────────────────────────────────────────────────────────────────────

// GSM8K — every answer terminator is the literal "#### <number>", so prompt
// the model to use the same format. This is the canonical lm-eval convention.
const GSM8K_INSTRUCTION =
  "Solve the problem step by step. End your response with '#### <answer>' " +
  "where <answer> is just the final numeric value.";

// MATH — canonical Hendrycks convention is \\boxed{<answer>}. The boxed
// extractor handles nested braces in answers like \\frac{1}{2}.
const MATH_INSTRUCTION =
  "Solve the problem step by step. Put your final answer in \\boxed{}.";

function extractGsm8kAnswer(text) {
  // Prefer the last "#### N" occurrence (model may emit it mid-CoT).
  const matches = [...text.matchAll(/####\s*(-?[\d,]+(?:\.\d+)?)/g)];
  if (matches.length) return normalizeNumber(matches[matches.length - 1][1]);
  // Fallback: last number anywhere in the response. Catches models that
  // forget the #### marker but still produce a final numeric answer.
  const nums = [...text.matchAll(/(-?\d+(?:,\d{3})*(?:\.\d+)?)/g)];
  if (nums.length) return normalizeNumber(nums[nums.length - 1][1]);
  return null;
}

function normalizeNumber(s) {
  if (s == null) return null;
  const cleaned = String(s).replace(/,/g, "").replace(/^\.+|\.+$/g, "");
  const n = Number(cleaned);
  return Number.isFinite(n) ? n : null;
}

function gradeGsm8k(response, gold) {
  const pred = extractGsm8kAnswer(response);
  const want = normalizeNumber(gold);
  if (pred == null || want == null) return { pass: false, pred, reason: `no parseable answer (gold=${want})` };
  // Tolerance for floats; integer answers compare exactly.
  const close = Math.abs(pred - want) < 1e-6;
  return { pass: close, pred, reason: close ? "ok" : `got ${pred}, want ${want}` };
}

// Extract the last \boxed{...} content. Handles nested braces by walking
// brace depth — a regex /\\boxed\{([^}]*)\}/ would truncate \boxed{\frac{1}{2}}.
function extractBoxed(text) {
  const i = text.lastIndexOf("\\boxed{");
  if (i < 0) return null;
  let depth = 1;
  let j = i + "\\boxed{".length;
  let start = j;
  while (j < text.length && depth > 0) {
    if (text[j] === "{") depth++;
    else if (text[j] === "}") depth--;
    j++;
  }
  if (depth !== 0) return null;
  return text.slice(start, j - 1);
}

// Light normalization of LaTeX answers to allow common-shape equivalents to
// match. This is intentionally not the full Hendrycks normalize_answer — for
// 30 cases the extra precision isn't worth the porting effort, and we'll
// under-credit some near-misses (acceptable for relative ranking).
function normalizeMath(s) {
  if (s == null) return null;
  return String(s)
    .trim()
    .replace(/\\!|\\,|\\;|\\:|\\ /g, "")
    .replace(/\\left|\\right/g, "")
    .replace(/\\dfrac|\\tfrac/g, "\\frac")
    .replace(/\s+/g, "")
    .replace(/\\\\/g, "\\")
    .replace(/^\{|\}$/g, "")
    .replace(/\$/g, "")
    .toLowerCase();
}

function gradeMath(response, gold) {
  const boxed = extractBoxed(response);
  if (boxed == null) return { pass: false, pred: null, reason: "no \\boxed{} in response" };
  const a = normalizeMath(boxed);
  const b = normalizeMath(gold);
  // Try numeric comparison too — `0.5` vs `\frac{1}{2}` is a common miss
  // that text-normalization alone won't catch. If both sides parse as
  // numbers, compare with tolerance.
  const na = Number(a), nb = Number(b);
  if (Number.isFinite(na) && Number.isFinite(nb)) {
    if (Math.abs(na - nb) < 1e-6) return { pass: true, pred: boxed, reason: "ok (numeric)" };
  }
  const close = a === b;
  return { pass: close, pred: boxed, reason: close ? "ok" : `got ${boxed}, want ${gold}` };
}

// ── Generator ────────────────────────────────────────────────────────────────
async function generate(prompt) {
  const timeoutMs = genTimeoutMs();
  const t = withTimeout(timeoutMs);
  let res;
  try {
    res = await fetch(`${HOST}/api/generate`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: MODEL,
        prompt,
        stream: false,
        options: { temperature: 0, num_predict: 2048 },
      }),
      signal: t.signal,
    });
  } catch (e) {
    if (e.name === "AbortError") throw new Error(`generate timed out after ${timeoutMs}ms`);
    throw e;
  } finally {
    t.cancel();
  }
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);
  const j = await res.json();
  return j.response ?? "";
}

// ── Runner ───────────────────────────────────────────────────────────────────
async function runCell(cellName, cases, gradeFn, buildPrompt) {
  const failed = [];
  const t0 = performance.now();
  let pass = 0;
  for (const c of cases) {
    let scored;
    try {
      const response = await generate(buildPrompt(c));
      scored = gradeFn(response, c.gold ?? c.answer);
    } catch (e) {
      scored = { pass: false, pred: null, reason: `threw: ${e.message}` };
    }
    if (scored.pass) pass++;
    else failed.push(c.id ?? c.question?.slice(0, 60) ?? "?");
    if (VERBOSE) console.log(`${scored.pass ? "✔" : "✘"} [${cellName}] ${(c.id ?? c.question ?? "").slice(0,60)} — ${scored.reason}`);
  }
  return {
    total: cases.length,
    pass,
    pct: 100 * pass / cases.length,
    failed,
    durationSec: (performance.now() - t0) / 1000,
  };
}

async function runAll() {
  const out = { savedAt: new Date().toISOString(), model: MODEL };
  const wantGsm = CELL === "all" || CELL === "gsm8k";
  const wantMath = CELL === "all" || CELL === "math500";
  const gpuHandle = startSampler();

  if (wantGsm) {
    let cases = loadJsonl(join(ROOT, "data", "gsm8k.jsonl"));
    if (LIMIT) cases = cases.slice(0, LIMIT);
    out.gsm8k = await runCell("gsm8k", cases, gradeGsm8k,
      c => `${GSM8K_INSTRUCTION}\n\nProblem: ${c.question}`);
  }
  if (wantMath) {
    let cases = loadJsonl(join(ROOT, "data", "math500.jsonl"));
    if (LIMIT) cases = cases.slice(0, LIMIT);
    out.math500 = await runCell("math500", cases, gradeMath,
      c => `${MATH_INSTRUCTION}\n\nProblem: ${c.problem}`);
  }

  // Overall = case-weighted mean across cells run (so re-running with --cell
  // produces a comparable composite to the saved baseline if both cells exist).
  const cells = [out.gsm8k, out.math500].filter(Boolean);
  const totalCases = cells.reduce((a, c) => a + c.total, 0);
  const totalPass  = cells.reduce((a, c) => a + c.pass,  0);
  out.totalCases = totalCases;
  out.totalPass  = totalPass;
  out.mathPct    = totalCases ? 100 * totalPass / totalCases : null;
  out.gpu        = await stopSampler(gpuHandle);
  return out;
}

// ── Reporting ────────────────────────────────────────────────────────────────
function fmtPct(n)   { return n == null ? "—" : `${n.toFixed(0)}%`; }
function fmtDelta(d) {
  if (d == null || !Number.isFinite(d)) return "—";
  const sign = d >= 0 ? "+" : "";
  const tag  = d < -REG_PP ? " ⚠" : "";
  return `${sign}${d.toFixed(0)}pp${tag}`;
}

function printReport(current, base) {
  const cols = base
    ? ["cell", "total", "pass", "pass%", "Δ pass%", "wall(s)"]
    : ["cell", "total", "pass", "pass%", "wall(s)"];
  const widths = [10, 5, 4, 5, 8, 7].slice(0, cols.length);
  const pad = (s, w, right = true) => right ? String(s).padStart(w) : String(s).padEnd(w);

  console.log("");
  console.log(cols.map((c, i) => i === 0 ? pad(c, widths[i], false) : pad(c, widths[i])).join(" | "));
  console.log("-".repeat(widths.reduce((a, w) => a + w + 3, 0)));

  const cellRow = (name, r, br) => {
    if (!r) return null;
    const dP = br ? r.pct - br.pct : null;
    const cells = [pad(name, widths[0], false), pad(r.total, widths[1]), pad(r.pass, widths[2]), pad(fmtPct(r.pct), widths[3])];
    if (base) return [...cells, pad(fmtDelta(dP), widths[4]), pad(r.durationSec.toFixed(1), widths[5])].join(" | ");
    return [...cells, pad(r.durationSec.toFixed(1), widths[4])].join(" | ");
  };

  for (const [name, key] of [["gsm8k", "gsm8k"], ["math500", "math500"]]) {
    const row = cellRow(name, current[key], base?.[key]);
    if (row) console.log(row);
  }
  console.log("-".repeat(widths.reduce((a, w) => a + w + 3, 0)));
  const overall = { total: current.totalCases, pass: current.totalPass, pct: current.mathPct, durationSec:
    (current.gsm8k?.durationSec ?? 0) + (current.math500?.durationSec ?? 0) };
  const baseOverall = base ? { total: base.totalCases, pass: base.totalPass, pct: base.mathPct } : null;
  console.log(cellRow("OVERALL", overall, baseOverall));
  console.log(`\n${fmtGpuSummary(current.gpu)}`);
}

// ── Main ─────────────────────────────────────────────────────────────────────
async function main() {
  console.log(`\nbench-math: model=${MODEL} host=${HOST} cell=${CELL}\n`);
  const existing = getModelSection(OUT, MODEL, "math");
  let mode = MODE;
  if (mode === "smart") mode = existing ? "compare" : "save";
  else if (mode === "compare" && !existing) {
    console.error(`no math entry for ${MODEL} at ${OUT} — run with --save first`);
    process.exit(1);
  }

  const current = await runAll();
  printReport(current, mode === "compare" ? existing : null);

  if (mode === "save") {
    writeModelSection(OUT, MODEL, "math", current);
    console.log(`\nmath entry saved for ${MODEL} → ${OUT}`);
  }
}

main().catch(e => { console.error(e); process.exit(1); });
