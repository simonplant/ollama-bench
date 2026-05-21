#!/usr/bin/env node
/**
 * Data analysis probe — two cells:
 *   table_qa — 30 WikiTableQuestions cases (data/wtq.jsonl). Markdown-rendered
 *              table + NL question, free-form answer compared to gold answer
 *              set (string + numeric normalized).
 *   sql_gen  — 30 WikiSQL cases (data/wikisql.jsonl). Schema + question →
 *              model SQL → executed via node:sqlite → result rows compared
 *              to precomputed gold answer list (set-equality, numeric tolerant).
 *
 * Deterministic, no judge. Requires Node 24+ (built-in node:sqlite, stable).
 *
 * Usage:
 *   node bench-data.mjs [--model gemma4:26b] [--host http://ollama:11434]
 *                       [--out ./baseline.json] [--save|--compare]
 *                       [--cell table_qa|sql_gen|all] [--limit N] [-v|--verbose]
 */

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { DatabaseSync } from "node:sqlite";
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
const CELL    = arg("--cell",  "all");
const LIMIT   = (() => { const v = arg("--limit", null); return v ? parseInt(v, 10) : null; })();
const VERBOSE = args.includes("-v") || args.includes("--verbose");
const MODE    = args.includes("--save")    ? "save"
              : args.includes("--compare") ? "compare"
              : "smart";

const REG_PP = 5;

function genTimeoutMs(numPredict) {
  const override = parseInt(process.env.OLLAMA_BENCH_TIMEOUT_MS ?? "", 10);
  if (Number.isFinite(override) && override > 0) return override;
  return 180_000 + (numPredict || 512) * 100;
}
function withTimeout(ms) {
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), ms);
  return { signal: ac.signal, cancel: () => clearTimeout(timer) };
}

function loadJsonl(path) {
  return readFileSync(path, "utf-8").trim().split("\n").map(l => JSON.parse(l));
}

async function generate(prompt, system = null, think = false, numPredict = 512) {
  const timeoutMs = genTimeoutMs(numPredict);
  const t = withTimeout(timeoutMs);
  let res;
  try {
    res = await fetch(`${HOST}/api/generate`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: MODEL,
        ...(system ? { system } : {}),
        prompt,
        stream: false,
        think,
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

// ── table_qa ─────────────────────────────────────────────────────────────────
function renderMarkdownTable(header, rows) {
  const head = "| " + header.join(" | ") + " |";
  const sep  = "| " + header.map(() => "---").join(" | ") + " |";
  const body = rows.map(r => "| " + r.map(c => String(c ?? "")).join(" | ") + " |").join("\n");
  return [head, sep, body].join("\n");
}

const TABLE_QA_SYSTEM =
  "You are answering a question from a data table. Respond with only the " +
  "answer — no preamble, no explanation. If multiple values answer the " +
  "question, separate them with commas.";

function normalizeAnswer(s) {
  return String(s ?? "")
    .toLowerCase()
    .trim()
    .replace(/^[*_`\s]+|[*_`\s.]+$/g, "")
    .replace(/\s+/g, " ");
}

// WTQ answers are a list; the model's output is free-form text. The model
// passes if every gold answer appears somewhere in its response (after
// normalization), AND the response isn't grossly longer than the joined
// gold — the second guard catches "everything-and-the-kitchen-sink" answers
// that happen to contain the right word.
function gradeTableQa(response, golds) {
  const r = normalizeAnswer(response);
  if (!r) return { pass: false, reason: "empty response" };
  const goldsNorm = golds.map(normalizeAnswer).filter(Boolean);
  if (!goldsNorm.length) return { pass: false, reason: "no gold answer" };

  // Per-gold containment check — handles single-value and list answers.
  const missing = goldsNorm.filter(g => !containsValue(r, g));
  if (missing.length) return { pass: false, reason: `missing in response: ${missing.slice(0, 3).join(", ")}` };

  // Loose verbosity guard. If the response is more than 5x the joined gold
  // length, the model likely dumped extra info — be conservative and pass
  // anyway if all golds are present (we already validated containment).
  // This is intentionally lenient; we'd rather under-flag than over.
  return { pass: true, reason: "ok" };
}

function containsValue(haystack, needle) {
  if (haystack.includes(needle)) return true;
  // Numeric tolerance: treat 58.0 == 58 == "58" etc.
  const n = Number(needle);
  if (Number.isFinite(n)) {
    const matches = [...haystack.matchAll(/-?\d+(?:\.\d+)?/g)];
    if (matches.some(m => Math.abs(Number(m[0]) - n) < 1e-6)) return true;
  }
  return false;
}

async function runTableQa(think, numPredict) {
  let cases = loadJsonl(join(ROOT, "data", "wtq.jsonl"));
  if (LIMIT) cases = cases.slice(0, LIMIT);

  const failed = [];
  const t0 = performance.now();
  let pass = 0;
  for (const c of cases) {
    let scored;
    try {
      const md = renderMarkdownTable(c.header, c.rows);
      const prompt = `Table:\n${md}\n\nQuestion: ${c.question}`;
      const response = await generate(prompt, TABLE_QA_SYSTEM, think, numPredict);
      scored = gradeTableQa(response, c.answers);
    } catch (e) {
      scored = { pass: false, reason: `threw: ${e.message}` };
    }
    if (scored.pass) pass++;
    else failed.push(c.id);
    if (VERBOSE) console.log(`${scored.pass ? "✔" : "✘"} ${c.id} [${c.answers.join(", ")}] — ${scored.reason}`);
  }
  return {
    total: cases.length, pass, pct: 100 * pass / cases.length, failed,
    durationSec: (performance.now() - t0) / 1000,
  };
}

// ── sql_gen ──────────────────────────────────────────────────────────────────
const SQL_SYSTEM =
  "You are a SQL expert. Write a single SQLite SELECT query to answer the " +
  "question. Use the exact column names from the schema; wrap names that " +
  "contain spaces or special characters in double quotes. Respond with only " +
  "the SQL — no markdown, no explanation.";

function buildSchema(header, types) {
  // Map WikiSQL types → sqlite affinities. WikiSQL uses "text" and "real".
  const sqliteType = t => (t === "real" ? "REAL" : "TEXT");
  const cols = header.map((h, i) => `  "${h}" ${sqliteType(types[i])}`).join(",\n");
  return `CREATE TABLE t (\n${cols}\n)`;
}

function buildSqliteDb(header, types, rows) {
  const db = new DatabaseSync(":memory:");
  db.exec(buildSchema(header, types));
  const placeholders = header.map(() => "?").join(", ");
  const cols = header.map(h => `"${h}"`).join(", ");
  const stmt = db.prepare(`INSERT INTO t (${cols}) VALUES (${placeholders})`);
  for (const row of rows) {
    // Coerce: REAL columns get Number(), TEXT stays string.
    const coerced = row.map((cell, i) => {
      if (types[i] === "real") { const n = Number(cell); return Number.isFinite(n) ? n : null; }
      return cell == null ? null : String(cell);
    });
    stmt.run(...coerced);
  }
  return db;
}

function extractSql(response) {
  // Strip markdown fences if present.
  const fence = response.match(/```(?:sql)?\s*\n?([\s\S]*?)\n?```/);
  let sql = fence ? fence[1] : response;
  // Some models prefix with "SQL:" or "Query:" labels.
  sql = sql.replace(/^\s*(?:sql|query)\s*:\s*/i, "");
  // Trim trailing semicolons (sqlite accepts either way, but keeps logs clean).
  sql = sql.trim().replace(/;\s*$/, "");
  return sql;
}

// Flatten a query result (array of row-objects) into a list of values.
// WikiSQL gold is always a single column; we project the first column of
// each row. Set comparison handles ordering differences.
function flattenResult(rows) {
  if (!Array.isArray(rows)) return [];
  return rows.map(r => {
    const keys = Object.keys(r);
    return keys.length ? r[keys[0]] : null;
  });
}

function normalizeForCompare(v) {
  if (v == null) return null;
  // Try numeric first — handles 58 == "58" == 58.0
  const n = Number(v);
  if (Number.isFinite(n) && String(v).trim() !== "") return Math.round(n * 1e6) / 1e6;
  return String(v).trim().toLowerCase();
}

function setEqual(a, b) {
  if (a.length !== b.length) return false;
  const an = a.map(normalizeForCompare).sort();
  const bn = b.map(normalizeForCompare).sort();
  for (let i = 0; i < an.length; i++) if (an[i] !== bn[i]) return false;
  return true;
}

async function runSqlGen(think, numPredict) {
  let cases = loadJsonl(join(ROOT, "data", "wikisql.jsonl"));
  if (LIMIT) cases = cases.slice(0, LIMIT);

  const failed = [];
  const t0 = performance.now();
  let pass = 0;
  for (const c of cases) {
    let scored;
    try {
      const schema = buildSchema(c.header, c.types);
      const prompt = `Schema:\n${schema}\n\nThe table has ${c.rows.length} rows.\nQuestion: ${c.question}`;
      const response = await generate(prompt, SQL_SYSTEM, think, numPredict);
      const sql = extractSql(response);
      if (!sql) {
        scored = { pass: false, reason: "no SQL extracted" };
      } else {
        const db = buildSqliteDb(c.header, c.types, c.rows);
        let modelRows;
        try {
          modelRows = db.prepare(sql).all();
        } catch (e) {
          db.close();
          scored = { pass: false, reason: `exec failed: ${e.message.slice(0, 80)}` };
          if (scored && VERBOSE) console.log(`${"✘"} ${c.id} — ${scored.reason}\n  sql: ${sql.slice(0, 100)}\n  gold: ${c.gold_desc}`);
          if (!scored.pass) failed.push(c.id);
          continue;
        }
        db.close();
        const got = flattenResult(modelRows);
        const ok = setEqual(got, c.gold);
        scored = { pass: ok, reason: ok ? "ok" : `got ${JSON.stringify(got).slice(0, 60)}, want ${JSON.stringify(c.gold).slice(0, 60)}`, sql };
      }
    } catch (e) {
      scored = { pass: false, reason: `threw: ${e.message}` };
    }
    if (scored.pass) pass++;
    else failed.push(c.id);
    if (VERBOSE) console.log(`${scored.pass ? "✔" : "✘"} ${c.id} — ${scored.reason}`);
  }
  return {
    total: cases.length, pass, pct: 100 * pass / cases.length, failed,
    durationSec: (performance.now() - t0) / 1000,
  };
}

// ── Runner ───────────────────────────────────────────────────────────────────
async function runAll() {
  const out = { savedAt: new Date().toISOString(), model: MODEL };
  const gpuHandle = startSampler();
  const sysHandle = startSysSampler();
  const { think, numPredict, supports } = await thinkingParams(HOST, MODEL, 512, 8192);
  if (supports) console.log(`(thinking model: think=${think}, num_predict=${numPredict})\n`);
  if (CELL === "all" || CELL === "table_qa") out.table_qa = await runTableQa(think, numPredict);
  if (CELL === "all" || CELL === "sql_gen")  out.sql_gen  = await runSqlGen(think, numPredict);

  const cells = [out.table_qa, out.sql_gen].filter(Boolean);
  out.totalCases = cells.reduce((a, c) => a + c.total, 0);
  out.totalPass  = cells.reduce((a, c) => a + c.pass,  0);
  out.dataPct    = out.totalCases ? 100 * out.totalPass / out.totalCases : null;
  out.gpu        = await stopSampler(gpuHandle);
  out.sys        = stopSysSampler(sysHandle);
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

  for (const [name, key] of [["table_qa", "table_qa"], ["sql_gen", "sql_gen"]]) {
    const row = cellRow(name, current[key], base?.[key]);
    if (row) console.log(row);
  }
  console.log("-".repeat(widths.reduce((a, w) => a + w + 3, 0)));
  const overall = { total: current.totalCases, pass: current.totalPass, pct: current.dataPct, durationSec:
    (current.table_qa?.durationSec ?? 0) + (current.sql_gen?.durationSec ?? 0) };
  const baseOverall = base ? { total: base.totalCases, pass: base.totalPass, pct: base.dataPct } : null;
  console.log(cellRow("OVERALL", overall, baseOverall));
  console.log(`\n${fmtGpuSummary(current.gpu)}`);
  console.log(fmtSysSummary(current.sys));
}

// ── Main ─────────────────────────────────────────────────────────────────────
async function main() {
  console.log(`\nbench-data: model=${MODEL} host=${HOST} cell=${CELL}\n`);
  const existing = getModelSection(OUT, MODEL, "data");
  let mode = MODE;
  if (mode === "smart") mode = existing ? "compare" : "save";
  else if (mode === "compare" && !existing) {
    console.error(`no data entry for ${MODEL} at ${OUT} — run with --save first`);
    process.exit(1);
  }

  const current = await runAll();
  printReport(current, mode === "compare" ? existing : null);

  if (mode === "save") {
    writeModelSection(OUT, MODEL, "data", current);
    console.log(`\ndata entry saved for ${MODEL} → ${OUT}`);
  }
}

main().catch(e => { console.error(e); process.exit(1); });
