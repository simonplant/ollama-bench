#!/usr/bin/env node
/**
 * Instruction-following probe — IFEval (Google), 50 sampled cases.
 *
 * Each case has one or more verifiable constraints (instruction_id_list).
 * Constraints are checked programmatically: keyword frequency, length,
 * markdown shape, case, punctuation, JSON validity, etc. Pure deterministic.
 *
 * Pass rule per case: ALL constraints satisfied → pass. Per-instruction-type
 * stats reported alongside overall pass% (so you can see whether a model
 * struggles with, say, length constraints specifically).
 *
 * Verifier semantics mirror the upstream IFEval evaluator
 * (github.com/google-research/google-research/tree/master/instruction_following_eval).
 * Strict-mode only — we don't run the "loose" pass that strips markdown
 * before re-checking.
 *
 * Usage:
 *   node bench-ifeval.mjs [--model gemma4:26b] [--host http://ollama:11434]
 *                         [--out ./baseline.json] [--save|--compare]
 *                         [--limit N] [-v|--verbose]
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { getModelSection, writeModelSection } from "./bench-baseline.mjs";
import { startSampler, stopSampler, fmtGpuSummary } from "./bench-gpu.mjs";
import { startSysSampler, stopSysSampler, fmtSysSummary } from "./bench-sys.mjs";
import { thinkingParams } from "./bench-thinking.mjs";

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

function genTimeoutMs(numPredict) {
  const override = parseInt(process.env.OLLAMA_BENCH_TIMEOUT_MS ?? "", 10);
  if (Number.isFinite(override) && override > 0) return override;
  return 240_000 + (numPredict || 1024) * 100;
}
function withTimeout(ms) {
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), ms);
  return { signal: ac.signal, cancel: () => clearTimeout(timer) };
}

function loadJsonl(path) {
  return readFileSync(path, "utf-8").trim().split("\n").map(l => JSON.parse(l));
}

// ── Constraint verifiers ─────────────────────────────────────────────────────
// Each returns { ok: bool, reason: string }.
// Semantics match google-research IFEval instructions.py wherever practical.

function checkRelation(value, target, relation) {
  if (relation === "at least") return value >= target;
  if (relation === "less than") return value < target;
  // IFEval also has "at most" and "exactly" in places — accept generously.
  if (relation === "at most")   return value <= target;
  if (relation === "exactly")   return value === target;
  return false;
}

const VERIFIERS = {
  "keywords:existence": (r, k) => {
    const missing = k.keywords.filter(kw => !new RegExp(`\\b${escapeRe(kw)}\\b`, "i").test(r));
    return { ok: missing.length === 0, reason: missing.length ? `missing: ${missing.join(",")}` : "ok" };
  },
  "keywords:frequency": (r, k) => {
    const count = (r.match(new RegExp(`\\b${escapeRe(k.keyword)}\\b`, "gi")) ?? []).length;
    const ok = checkRelation(count, k.frequency, k.relation);
    return { ok, reason: ok ? "ok" : `'${k.keyword}' x${count}, need ${k.relation} ${k.frequency}` };
  },
  "keywords:forbidden_words": (r, k) => {
    const found = k.forbidden_words.filter(w => new RegExp(`\\b${escapeRe(w)}\\b`, "i").test(r));
    return { ok: found.length === 0, reason: found.length ? `forbidden present: ${found.join(",")}` : "ok" };
  },
  "keywords:letter_frequency": (r, k) => {
    const count = (r.match(new RegExp(escapeRe(k.letter), "gi")) ?? []).length;
    const ok = checkRelation(count, k.let_frequency, k.let_relation);
    return { ok, reason: ok ? "ok" : `'${k.letter}' x${count}, need ${k.let_relation} ${k.let_frequency}` };
  },

  "length_constraints:number_sentences": (r, k) => {
    const count = splitSentences(r).length;
    const ok = checkRelation(count, k.num_sentences, k.relation);
    return { ok, reason: ok ? "ok" : `${count} sentences, need ${k.relation} ${k.num_sentences}` };
  },
  "length_constraints:number_paragraphs": (r, k) => {
    // IFEval: split on `\s?\*\*\*\s?`, count must equal num_paragraphs.
    const paragraphs = r.split(/\s?\*\*\*\s?/).filter(p => p.trim().length > 0);
    const ok = paragraphs.length === k.num_paragraphs;
    return { ok, reason: ok ? "ok" : `${paragraphs.length} paragraphs (***-separated), need ${k.num_paragraphs}` };
  },
  "length_constraints:number_words": (r, k) => {
    const count = countWords(r);
    const ok = checkRelation(count, k.num_words, k.relation);
    return { ok, reason: ok ? "ok" : `${count} words, need ${k.relation} ${k.num_words}` };
  },
  "length_constraints:nth_paragraph_first_word": (r, k) => {
    const paragraphs = r.split(/\s?\*\*\*\s?/).filter(p => p.trim().length > 0);
    if (paragraphs.length !== k.num_paragraphs) {
      return { ok: false, reason: `${paragraphs.length} paragraphs, need ${k.num_paragraphs}` };
    }
    const nth = paragraphs[k.nth_paragraph - 1] ?? "";
    const first = (nth.trim().split(/\s+/)[0] ?? "").toLowerCase().replace(/[^a-z0-9]/g, "");
    const want  = String(k.first_word).toLowerCase().replace(/[^a-z0-9]/g, "");
    const ok = first === want;
    return { ok, reason: ok ? "ok" : `paragraph ${k.nth_paragraph} starts with '${first}', need '${want}'` };
  },

  "detectable_format:number_bullet_lists": (r, k) => {
    const stars = (r.match(/^\s*\*[^\*].*$/mg) ?? []).length;
    const dashes = (r.match(/^\s*-.*$/mg) ?? []).length;
    const count = stars + dashes;
    const ok = count === k.num_bullets;
    return { ok, reason: ok ? "ok" : `${count} bullets, need exactly ${k.num_bullets}` };
  },
  "detectable_format:number_highlighted_sections": (r, k) => {
    // Single-asterisk *highlights* AND **bold highlights** both count.
    const single = (r.match(/\*[^\n*]+\*/g) ?? []).length;
    const dbl    = (r.match(/\*\*[^\n*]+\*\*/g) ?? []).length;
    const count = single + dbl;
    const ok = count >= k.num_highlights;
    return { ok, reason: ok ? "ok" : `${count} highlights, need at least ${k.num_highlights}` };
  },
  "detectable_format:multiple_sections": (r, k) => {
    // Split on `\s?<spliter>\s?\d+` — count of splits minus 1 = num_sections.
    const re = new RegExp(`\\s?${escapeRe(k.section_spliter)}\\s?\\d+`, "g");
    const parts = r.split(re);
    const count = parts.length - 1;
    const ok = count === k.num_sections;
    return { ok, reason: ok ? "ok" : `${count} sections by '${k.section_spliter} N', need ${k.num_sections}` };
  },
  "detectable_format:json_format": (r) => {
    // Strip code fences if present — many models wrap JSON in ```json...```.
    let stripped = r.trim();
    const fence = stripped.match(/^```(?:json)?\s*\n?([\s\S]*?)\n?```\s*$/);
    if (fence) stripped = fence[1].trim();
    try { JSON.parse(stripped); return { ok: true, reason: "ok" }; }
    catch (e) { return { ok: false, reason: `invalid JSON: ${e.message.slice(0, 60)}` }; }
  },
  "detectable_format:title": (r) => {
    const ok = /<<[^<>\n]+>>/.test(r);
    return { ok, reason: ok ? "ok" : "no <<title>> found" };
  },
  "detectable_format:constrained_response": (r) => {
    // IFEval hardcodes three permitted responses.
    const allowed = ["My answer is yes.", "My answer is no.", "My answer is maybe."];
    const trimmed = r.trim();
    const ok = allowed.includes(trimmed);
    return { ok, reason: ok ? "ok" : `not one of: ${allowed.join(" | ")}` };
  },
  "detectable_format:number_placeholders": (r, k) => {
    const count = (r.match(/\[[^\[\]\n]+\]/g) ?? []).length;
    const ok = count >= k.num_placeholders;
    return { ok, reason: ok ? "ok" : `${count} placeholders, need at least ${k.num_placeholders}` };
  },

  "detectable_content:postscript": (r, k) => {
    // Must contain marker, followed by content. Case-insensitive on the marker.
    const idx = r.toLowerCase().lastIndexOf(String(k.postscript_marker).toLowerCase());
    if (idx < 0) return { ok: false, reason: `no '${k.postscript_marker}' marker` };
    const after = r.slice(idx + k.postscript_marker.length).trim();
    return { ok: after.length > 0, reason: after.length > 0 ? "ok" : "marker present but empty postscript" };
  },
  "detectable_content:number_placeholders": (r, k) => {
    // Same as detectable_format:number_placeholders — IFEval ships both names.
    const count = (r.match(/\[[^\[\]\n]+\]/g) ?? []).length;
    const ok = count >= k.num_placeholders;
    return { ok, reason: ok ? "ok" : `${count} placeholders, need at least ${k.num_placeholders}` };
  },

  "punctuation:no_comma": (r) => {
    const ok = !r.includes(",");
    return { ok, reason: ok ? "ok" : `${(r.match(/,/g) ?? []).length} commas present` };
  },

  "change_case:capital_word_frequency": (r, k) => {
    const count = (r.match(/\b[A-Z]{2,}\b/g) ?? []).length;
    const ok = checkRelation(count, k.capital_frequency, k.capital_relation);
    return { ok, reason: ok ? "ok" : `${count} ALL_CAPS words, need ${k.capital_relation} ${k.capital_frequency}` };
  },
  "change_case:english_capital": (r) => {
    const ok = r === r.toUpperCase() && /[A-Z]/.test(r);
    return { ok, reason: ok ? "ok" : "contains lowercase characters" };
  },
  "change_case:english_lowercase": (r) => {
    const ok = r === r.toLowerCase() && /[a-z]/.test(r);
    return { ok, reason: ok ? "ok" : "contains uppercase characters" };
  },

  "startend:quotation": (r) => {
    const t = r.trim();
    const ok = t.length >= 2 && t.startsWith('"') && t.endsWith('"');
    return { ok, reason: ok ? "ok" : "response not wrapped in double quotes" };
  },
  "startend:end_checker": (r, k) => {
    const t = r.trim();
    const ok = t.toLowerCase().endsWith(String(k.end_phrase).toLowerCase());
    return { ok, reason: ok ? "ok" : `does not end with '${k.end_phrase}'` };
  },

  "combination:two_responses": (r) => {
    const parts = r.split(/\*{6,}/);
    const nonEmpty = parts.filter(p => p.trim().length > 0);
    const ok = nonEmpty.length === 2;
    return { ok, reason: ok ? "ok" : `${nonEmpty.length} responses split by ****** (need 2)` };
  },
  "combination:repeat_prompt": (r, k) => {
    const t = r.trim();
    const want = String(k.prompt_to_repeat).trim();
    const ok = t.startsWith(want);
    return { ok, reason: ok ? "ok" : "response does not begin with verbatim prompt" };
  },
};

function escapeRe(s) {
  return String(s).replace(/[-\/\\^$*+?.()|[\]{}]/g, "\\$&");
}

// Sentence splitter — naive but matches IFEval's tokenize behavior closely
// enough for length checks. Counts strings between ./!/? followed by space.
function splitSentences(text) {
  return text.split(/(?<=[.!?])\s+/).map(s => s.trim()).filter(Boolean);
}

function countWords(text) {
  return text.trim().split(/\s+/).filter(Boolean).length;
}

// ── Generation ───────────────────────────────────────────────────────────────
async function generate(prompt, think, numPredict) {
  const timeoutMs = genTimeoutMs(numPredict);
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
        think,
        // 1024 tokens — IFEval prompts often ask for >300 words. Thinking
        // models need a larger budget so reasoning + answer both fit.
        options: { temperature: 0, num_predict: numPredict },
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
  let cases = loadJsonl(join(ROOT, "data", "ifeval.jsonl"));
  if (LIMIT) cases = cases.slice(0, LIMIT);

  const { think, numPredict, supports } = await thinkingParams(HOST, MODEL, 1024, 8192);
  if (supports) console.log(`(thinking model: think=${think}, num_predict=${numPredict})\n`);

  const byInstr = new Map();        // instruction id → { total, pass }
  const failed = [];
  const gpuHandle = startSampler();
  const sysHandle = startSysSampler();
  const t0 = performance.now();
  let pass = 0;
  for (const c of cases) {
    let response = "", caseOk = true, reasons = [];
    try {
      response = await generate(c.prompt, think, numPredict);
    } catch (e) {
      caseOk = false;
      reasons.push(`generate threw: ${e.message}`);
    }
    for (const inst of c.instructions) {
      const v = VERIFIERS[inst.id];
      let res;
      if (!v) {
        // Should not happen — fetch-data.mjs filters to SUPPORTED set —
        // but if we add a new instruction id we want to know loudly.
        res = { ok: false, reason: `no verifier for ${inst.id}` };
      } else if (!response) {
        res = { ok: false, reason: "empty response" };
      } else {
        try { res = v(response, inst.kwargs); }
        catch (e) { res = { ok: false, reason: `verifier threw: ${e.message}` }; }
      }
      const row = byInstr.get(inst.id) ?? { total: 0, pass: 0 };
      row.total++;
      if (res.ok) row.pass++;
      byInstr.set(inst.id, row);
      if (!res.ok) {
        caseOk = false;
        reasons.push(`${inst.id}: ${res.reason}`);
      }
    }
    if (caseOk) pass++;
    else failed.push(`${c.key}::${c.instructions.map(i => i.id).join("+")}`);
    if (VERBOSE) console.log(`${caseOk ? "✔" : "✘"} key=${c.key} [${c.instructions.map(i => i.id).join(", ")}] ${caseOk ? "" : "— " + reasons.join("; ")}`);
  }
  const durationSec = (performance.now() - t0) / 1000;
  const gpu = await stopSampler(gpuHandle);
  const sys = stopSysSampler(sysHandle);
  return {
    savedAt: new Date().toISOString(),
    model: MODEL,
    total: cases.length,
    pass,
    ifevalPct: 100 * pass / cases.length,
    byInstruction: Object.fromEntries(byInstr),
    failed,
    durationSec,
    gpu,
    sys,
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
  console.log(`\nOVERALL: ${current.pass}/${current.total} cases = ${fmtPct(current.ifevalPct)}` +
              (base ? `  (Δ ${fmtDelta(current.ifevalPct - base.ifevalPct)})` : "") +
              `   wall: ${current.durationSec.toFixed(1)}s`);
  console.log(fmtGpuSummary(current.gpu));
  console.log(fmtSysSummary(current.sys) + "\n");

  // Per-instruction pass rate. Long names; truncate the column heading.
  const cols = ["instruction", "total", "pass", "pass%"];
  const widths = [42, 5, 4, 5];
  const pad = (s, w, right = true) => right ? String(s).padStart(w) : String(s).padEnd(w);
  console.log(cols.map((c, i) => i === 0 ? pad(c, widths[i], false) : pad(c, widths[i])).join(" | "));
  console.log("-".repeat(widths.reduce((a, w) => a + w + 3, 0)));

  const sorted = Object.entries(current.byInstruction).sort((a, b) => (a[1].pass / a[1].total) - (b[1].pass / b[1].total));
  for (const [id, r] of sorted) {
    console.log([pad(id, widths[0], false), pad(r.total, widths[1]), pad(r.pass, widths[2]), pad(fmtPct(100 * r.pass / r.total), widths[3])].join(" | "));
  }
}

// ── Main ─────────────────────────────────────────────────────────────────────
async function main() {
  console.log(`\nbench-ifeval: model=${MODEL} host=${HOST} (IFEval${LIMIT ? `, first ${LIMIT}` : ""})\n`);
  const existing = getModelSection(OUT, MODEL, "ifeval");
  let mode = MODE;
  if (mode === "smart") mode = existing ? "compare" : "save";
  else if (mode === "compare" && !existing) {
    console.error(`no ifeval entry for ${MODEL} at ${OUT} — run with --save first`);
    process.exit(1);
  }

  const current = await runCases();
  printReport(current, mode === "compare" ? existing : null);

  if (mode === "save") {
    writeModelSection(OUT, MODEL, "ifeval", current);
    console.log(`\nifeval entry saved for ${MODEL} → ${OUT}`);
  }
}

main().catch(e => { console.error(e); process.exit(1); });
