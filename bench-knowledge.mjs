#!/usr/bin/env node
/**
 * Knowledge probe — MMLU-Pro, 50 cases stratified across categories.
 *
 * 10-option MCQ (vs original MMLU's 4). Direct prompt, no CoT — keeps the
 * runtime tight and the score reflects retrieval/recognition rather than
 * reasoning chain length.
 *
 * Deterministic grading: extract the model's chosen letter A–J, compare to
 * gold. No judge.
 *
 * Usage:
 *   node bench-knowledge.mjs [--model gemma4:26b] [--host http://ollama:11434]
 *                            [--out ./baseline.json] [--save|--compare]
 *                            [--limit N] [-v|--verbose]
 *
 * Per-call timeout: 120s (MCQ responses are short).
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
const LIMIT   = (() => { const v = arg("--limit", null); return v ? parseInt(v, 10) : null; })();
const VERBOSE = args.includes("-v") || args.includes("--verbose");
const MODE    = args.includes("--save")    ? "save"
              : args.includes("--compare") ? "compare"
              : "smart";

const REG_PP = 5;
const LETTERS = "ABCDEFGHIJ";

function genTimeoutMs() {
  const override = parseInt(process.env.OLLAMA_BENCH_TIMEOUT_MS ?? "", 10);
  if (Number.isFinite(override) && override > 0) return override;
  return 120_000;
}
function withTimeout(ms) {
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), ms);
  return { signal: ac.signal, cancel: () => clearTimeout(timer) };
}

function loadJsonl(path) {
  return readFileSync(path, "utf-8").trim().split("\n").map(l => JSON.parse(l));
}

// ── Prompt + grading ─────────────────────────────────────────────────────────
function buildPrompt(c) {
  const lines = [`Question: ${c.question}`, ""];
  c.options.forEach((opt, i) => lines.push(`${LETTERS[i]}) ${opt}`));
  lines.push("", "Respond with only the single letter of the correct option.");
  return lines.join("\n");
}

// Letter extraction priorities (most reliable first):
//   1. "Answer: X" / "answer is X" / "the answer is X"   → X
//   2. "**X**" or "[X]" or "(X)"                          → last one
//   3. Lone letter on its own line                        → last one
//   4. Fallback: first A-J letter in the full response
function extractLetter(text, numOptions) {
  const valid = new Set(LETTERS.slice(0, numOptions).split(""));
  const upper = text.toUpperCase();

  // Pattern 1: explicit "answer" prefix
  const ansMatch = upper.match(/(?:THE\s+)?ANSWER\s*(?:IS|:)\s*\*?\*?\(?\[?([A-J])\b/);
  if (ansMatch && valid.has(ansMatch[1])) return ansMatch[1];

  // Pattern 2: bracketed / starred letter — take the last occurrence so CoT
  // chains that mention multiple options early settle on the final pick.
  const bracketed = [...upper.matchAll(/\*\*([A-J])\*\*|\[([A-J])\]|\(([A-J])\)/g)];
  if (bracketed.length) {
    const last = bracketed[bracketed.length - 1];
    const ch = last[1] || last[2] || last[3];
    if (valid.has(ch)) return ch;
  }

  // Pattern 3: lone letter on its own line
  const loneLines = upper.split("\n").map(l => l.trim()).filter(l => /^[A-J]\.?$/.test(l));
  if (loneLines.length) {
    const ch = loneLines[loneLines.length - 1][0];
    if (valid.has(ch)) return ch;
  }

  // Pattern 4: first valid letter in the text (use \b to skip letters inside
  // words like "Answer is something")
  const anyMatch = upper.match(/\b([A-J])\b/);
  if (anyMatch && valid.has(anyMatch[1])) return anyMatch[1];

  return null;
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
        // 128 tokens is enough for "Answer: X" plus a brief justification.
        // Some models always emit CoT even when asked not to; this caps it.
        options: { temperature: 0, num_predict: 256 },
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
async function runCases() {
  let cases = loadJsonl(join(ROOT, "data", "mmlupro.jsonl"));
  if (LIMIT) cases = cases.slice(0, LIMIT);

  const byCat = new Map();
  const failed = [];
  const gpuHandle = startSampler();
  const t0 = performance.now();
  for (const c of cases) {
    let pred = null, pass = false, reason = "";
    try {
      const response = await generate(buildPrompt(c));
      pred = extractLetter(response, c.options.length);
      if (pred == null) reason = "no letter extracted";
      else if (pred === c.answer) { pass = true; reason = "ok"; }
      else reason = `picked ${pred}, want ${c.answer}`;
    } catch (e) {
      reason = `threw: ${e.message}`;
    }
    const row = byCat.get(c.category) ?? { total: 0, pass: 0 };
    row.total++;
    if (pass) row.pass++;
    byCat.set(c.category, row);
    if (!pass) failed.push(`${c.id}::${c.category}`);
    if (VERBOSE) console.log(`${pass ? "✔" : "✘"} [${c.category}] ${c.question.slice(0, 60)} — ${reason}`);
  }
  const durationSec = (performance.now() - t0) / 1000;
  const gpu = await stopSampler(gpuHandle);
  const total = cases.length;
  const pass = [...byCat.values()].reduce((a, r) => a + r.pass, 0);
  return {
    savedAt: new Date().toISOString(),
    model: MODEL,
    total, pass,
    mmluPct: 100 * pass / total,
    byCategory: Object.fromEntries(byCat),
    failed,
    durationSec,
    gpu,
  };
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
    ? ["category", "total", "pass", "pass%", "Δ pass%"]
    : ["category", "total", "pass", "pass%"];
  const widths = [14, 5, 4, 5, 8].slice(0, cols.length);
  const pad = (s, w, right = true) => right ? String(s).padStart(w) : String(s).padEnd(w);

  console.log("");
  console.log(cols.map((c, i) => i === 0 ? pad(c, widths[i], false) : pad(c, widths[i])).join(" | "));
  console.log("-".repeat(widths.reduce((a, w) => a + w + 3, 0)));

  const cats = Object.keys(current.byCategory).sort();
  for (const cat of cats) {
    const r = current.byCategory[cat];
    const br = base?.byCategory?.[cat];
    const pct = 100 * r.pass / r.total;
    const bp = br ? 100 * br.pass / br.total : null;
    const row = [pad(cat, widths[0], false), pad(r.total, widths[1]), pad(r.pass, widths[2]), pad(fmtPct(pct), widths[3])];
    if (base) row.push(pad(fmtDelta(bp != null ? pct - bp : null), widths[4]));
    console.log(row.join(" | "));
  }
  console.log("-".repeat(widths.reduce((a, w) => a + w + 3, 0)));

  const overall = [pad("OVERALL", widths[0], false), pad(current.total, widths[1]), pad(current.pass, widths[2]), pad(fmtPct(current.mmluPct), widths[3])];
  if (base) overall.push(pad(fmtDelta(current.mmluPct - base.mmluPct), widths[4]));
  console.log(overall.join(" | "));
  console.log(`\nwall: ${current.durationSec.toFixed(1)}s   ${fmtGpuSummary(current.gpu)}`);
}

// ── Main ─────────────────────────────────────────────────────────────────────
async function main() {
  console.log(`\nbench-knowledge: model=${MODEL} host=${HOST} (MMLU-Pro${LIMIT ? `, first ${LIMIT}` : ""})\n`);
  const existing = getModelSection(OUT, MODEL, "mmlu");
  let mode = MODE;
  if (mode === "smart") mode = existing ? "compare" : "save";
  else if (mode === "compare" && !existing) {
    console.error(`no mmlu entry for ${MODEL} at ${OUT} — run with --save first`);
    process.exit(1);
  }

  const current = await runCases();
  printReport(current, mode === "compare" ? existing : null);

  if (mode === "save") {
    writeModelSection(OUT, MODEL, "mmlu", current);
    console.log(`\nmmlu entry saved for ${MODEL} → ${OUT}`);
  }
}

main().catch(e => { console.error(e); process.exit(1); });
