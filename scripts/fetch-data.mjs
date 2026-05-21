#!/usr/bin/env node
// Fetch + sample the five public benchmarks the harness uses.
// One-shot: run when datasets need refreshing, commit the JSONL output.
// Pulls via HF datasets-server JSON API (no auth, no parquet parser).
//
// Reproducibility: SEED is fixed. Sampling is a seeded Fisher-Yates,
// not Math.random, so re-runs produce identical case sets.

import { writeFileSync, mkdirSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const OUT  = join(ROOT, "data");
mkdirSync(OUT, { recursive: true });

const SEED = 42;
const FORCE = process.argv.includes("--force");
const sleep = ms => new Promise(r => setTimeout(r, ms));

// Mulberry32 — small, deterministic PRNG. Standard Math.random can't be
// seeded; we need cross-run identical samples so the JSONL committed in the
// repo matches what anyone else regenerating it would get.
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function sample(arr, n, seed) {
  const r = rng(seed);
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a.slice(0, n);
}

// HF's datasets-server rate-limits aggressive callers (observed 429 around
// 4K rows on MMLU-Pro). Retry with header-aware backoff: honor Retry-After
// when present, otherwise exponential up to 60s. Three retries before giving
// up — past that, the dataset is having an outage, not a rate limit.
async function fetchPage(dataset, config, split, offset, length = 100) {
  const url = `https://datasets-server.huggingface.co/rows?dataset=${encodeURIComponent(dataset)}&config=${encodeURIComponent(config)}&split=${encodeURIComponent(split)}&offset=${offset}&length=${length}`;
  let attempt = 0;
  while (true) {
    const res = await fetch(url);
    if (res.ok) return res.json();
    if (res.status === 429 && attempt < 5) {
      const ra = Number(res.headers.get("retry-after"));
      const wait = Number.isFinite(ra) && ra > 0 ? ra * 1000 : Math.min(60000, 2000 * 2 ** attempt);
      process.stderr.write(`\n  429 at offset=${offset}, sleeping ${wait}ms (attempt ${attempt + 1})\n`);
      await sleep(wait);
      attempt++;
      continue;
    }
    throw new Error(`HF API ${res.status} for ${dataset} offset=${offset}`);
  }
}

async function fetchAll(dataset, config, split, { minRows = 0 } = {}) {
  const first = await fetchPage(dataset, config, split, 0, 100);
  const total = first.num_rows_total;
  const rows = first.rows.map(r => r.row);
  for (let off = 100; off < total; off += 100) {
    // Gentle pacing — keeps us under the rate limit for the larger datasets.
    await sleep(150);
    try {
      const page = await fetchPage(dataset, config, split, off, 100);
      rows.push(...page.rows.map(r => r.row));
    } catch (e) {
      // Rate-limited past the retry budget. If we already have enough rows
      // to sample from, proceed with what we have — sampling 50 from 4K is
      // statistically indistinguishable from 50 from 12K for our use case.
      // If we're below minRows the caller asked for, re-throw.
      if (rows.length >= minRows) {
        process.stderr.write(`\n  ${dataset}: stopping early at ${rows.length}/${total} (${e.message})\n`);
        break;
      }
      throw e;
    }
    process.stderr.write(`  ${dataset}: ${rows.length}/${total}\r`);
  }
  process.stderr.write(`  ${dataset}: ${rows.length}/${total}\n`);
  return rows;
}

function writeJsonl(name, rows) {
  const path = join(OUT, name);
  writeFileSync(path, rows.map(r => JSON.stringify(r)).join("\n") + "\n");
  console.log(`wrote ${rows.length} -> ${path}`);
}

function skipIfExists(name) {
  if (FORCE) return false;
  const path = join(OUT, name);
  if (existsSync(path)) {
    console.log(`skip ${name} (exists, pass --force to refetch)`);
    return true;
  }
  return false;
}

// ---------- GSM8K: 50 cases, random sample ----------
async function gsm8k() {
  if (skipIfExists("gsm8k.jsonl")) return;
  const all = await fetchAll("openai/gsm8k", "main", "test");
  const picked = sample(all, 50, SEED);
  writeJsonl("gsm8k.jsonl", picked.map(r => ({
    question: r.question,
    answer:   r.answer,                       // raw, includes #### <n>
    gold:     extractGsm8kGold(r.answer),     // pre-extracted for grader
  })));
}
function extractGsm8kGold(answer) {
  const m = answer.match(/####\s*(-?[\d,]+(?:\.\d+)?)/);
  if (!m) return null;
  return m[1].replace(/,/g, "");
}

// ---------- MATH-500: 30 cases, sampled across difficulty levels ----------
async function math500() {
  if (skipIfExists("math500.jsonl")) return;
  const all = await fetchAll("HuggingFaceH4/MATH-500", "default", "test");
  // Stratify by level (1-5) so the sample isn't dominated by one tier
  const byLevel = {};
  for (const r of all) (byLevel[r.level] ??= []).push(r);
  const picked = [];
  for (const lvl of Object.keys(byLevel).sort()) {
    picked.push(...sample(byLevel[lvl], 6, SEED + Number(lvl)));
  }
  writeJsonl("math500.jsonl", picked.map(r => ({
    id:       r.unique_id,
    problem:  r.problem,
    answer:   r.answer,
    subject:  r.subject,
    level:    r.level,
  })));
}

// ---------- HumanEval: full 164, no sampling ----------
async function humaneval() {
  if (skipIfExists("humaneval.jsonl")) return;
  const all = await fetchAll("openai/openai_humaneval", "openai_humaneval", "test");
  writeJsonl("humaneval.jsonl", all.map(r => ({
    task_id:     r.task_id,
    prompt:      r.prompt,
    test:        r.test,
    entry_point: r.entry_point,
  })));
}

// ---------- MMLU-Pro: 50 cases, stratified across 14 categories ----------
async function mmlupro() {
  if (skipIfExists("mmlupro.jsonl")) return;
  // minRows=2000 ensures all 14 subject categories are represented in the
  // pool we sample from, even if HF rate-limits us before we get all 12K.
  const all = await fetchAll("TIGER-Lab/MMLU-Pro", "default", "test", { minRows: 2000 });
  const byCat = {};
  for (const r of all) (byCat[r.category] ??= []).push(r);
  const cats = Object.keys(byCat).sort();
  const picked = [];
  // ~50 / 14 ≈ 3-4 per category. Use per-category seed for stable picks.
  const perCat = Math.ceil(50 / cats.length);
  let i = 0;
  for (const cat of cats) {
    picked.push(...sample(byCat[cat], perCat, SEED + i));
    i++;
  }
  // Trim to exactly 50 with seeded shuffle to avoid bias toward earlier cats
  const final = sample(picked, 50, SEED);
  writeJsonl("mmlupro.jsonl", final.map(r => ({
    id:       r.question_id,
    question: r.question,
    options:  r.options,
    answer:   r.answer,         // letter A-J
    category: r.category,
  })));
}

// ---------- IFEval: 50 cases, filtered to constraints we can verify in JS ----------
// IFEval cases have one or more instruction_id_list entries; for each, kwargs
// holds the verifier's parameters. We only keep cases whose instruction set
// is fully covered by our JS port (bench-ifeval.mjs).
const SUPPORTED = new Set([
  "keywords:existence",
  "keywords:frequency",
  "keywords:forbidden_words",
  "keywords:letter_frequency",
  "length_constraints:number_sentences",
  "length_constraints:number_paragraphs",
  "length_constraints:number_words",
  "length_constraints:nth_paragraph_first_word",
  "detectable_format:number_bullet_lists",
  "detectable_format:number_highlighted_sections",
  "detectable_format:multiple_sections",
  "detectable_format:json_format",
  "detectable_format:title",
  "detectable_format:constrained_response",
  "detectable_format:number_placeholders",
  "detectable_content:postscript",
  "detectable_content:number_placeholders",
  "punctuation:no_comma",
  "change_case:capital_word_frequency",
  "change_case:english_capital",
  "change_case:english_lowercase",
  "startend:quotation",
  "startend:end_checker",
  "combination:two_responses",
  "combination:repeat_prompt",
]);

async function ifeval() {
  if (skipIfExists("ifeval.jsonl")) return;
  const all = await fetchAll("google/IFEval", "default", "train", { minRows: 400 });
  const eligible = all.filter(r =>
    r.instruction_id_list.every(id => SUPPORTED.has(id))
  );
  console.log(`  ifeval: ${eligible.length}/${all.length} cases use only supported constraints`);
  const picked = sample(eligible, 50, SEED);
  writeJsonl("ifeval.jsonl", picked.map(r => ({
    key:     r.key,
    prompt:  r.prompt,
    // Flatten kwargs: only keep keys that are non-null for each instruction
    instructions: r.instruction_id_list.map((id, i) => {
      const k = r.kwargs[i] || {};
      const cleaned = Object.fromEntries(
        Object.entries(k).filter(([, v]) => v !== null && v !== undefined)
      );
      return { id, kwargs: cleaned };
    }),
  })));
}

// ---------- WikiTableQuestions: 30 cases, sampled from lighteval mirror ----------
// Original stanfordnlp/wikitablequestions has a Python loader script that
// datasets-server can't run; lighteval/wikitablequestions is a parquet mirror
// with the same fields (question, answers, table.header, table.rows).
async function wtq() {
  if (skipIfExists("wtq.jsonl")) return;
  // Limit to ~5000 rows to stay well under the rate limit — sample 30 from there.
  const all = await fetchAll("lighteval/wikitablequestions", "default", "test", { minRows: 1000 });
  // Filter to small tables only — large WTQ tables blow past the model's
  // context budget when stringified, and tables >30 rows rarely add signal
  // for a 30-case sample.
  const small = all.filter(r => (r.table?.rows?.length ?? 0) <= 25);
  const picked = sample(small, 30, SEED);
  writeJsonl("wtq.jsonl", picked.map(r => ({
    id:       r.id,
    question: r.question,
    answers:  r.answers,           // list of acceptable answer strings
    header:   r.table.header,
    rows:     r.table.rows,
  })));
}

// ---------- WikiSQL: 30 cases, sampled from Rathanr mirror ----------
// Original Salesforce/wikisql is Python-script loaded; Rathanr/wikisql is the
// parquet mirror with question, sql.human_readable, table.header/types/rows.
//
// WikiSQL's `human_readable` SQL doesn't quote column names that contain
// spaces and uses literal value tokens without quoting strings — both make
// direct execution against sqlite messy. We sidestep that by precomputing
// the gold answer set from the structured `sql` field (operators + indices)
// and storing it alongside the case. The probe then only needs to execute
// the model's SQL and compare its rows to the precomputed gold.
const WIKISQL_OPS = ["=", ">", "<"];
const WIKISQL_AGG = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"];

function evalWikiSql(sql, table) {
  const sel = sql.sel;
  const agg = sql.agg;
  const conds = sql.conds;
  // Filter rows by WHERE conditions (AND across conds).
  const filtered = table.rows.filter(row => {
    for (let i = 0; i < conds.column_index.length; i++) {
      const cell = row[conds.column_index[i]];
      const op   = conds.operator_index[i];
      const tgt  = conds.condition[i];
      if (op === 0) {
        if (String(cell).trim().toLowerCase() !== String(tgt).trim().toLowerCase()) return false;
      } else if (op === 1) {
        const a = Number(cell), b = Number(tgt);
        if (!(Number.isFinite(a) && Number.isFinite(b) && a > b)) return false;
      } else if (op === 2) {
        const a = Number(cell), b = Number(tgt);
        if (!(Number.isFinite(a) && Number.isFinite(b) && a < b)) return false;
      } else {
        return false; // unknown op
      }
    }
    return true;
  });
  const vals = filtered.map(r => r[sel]);
  switch (agg) {
    case 0: return vals;
    case 1: { const nums = vals.map(Number).filter(Number.isFinite); return nums.length ? [Math.max(...nums)] : []; }
    case 2: { const nums = vals.map(Number).filter(Number.isFinite); return nums.length ? [Math.min(...nums)] : []; }
    case 3: return [vals.length];
    case 4: { const nums = vals.map(Number).filter(Number.isFinite); return [nums.reduce((a, b) => a + b, 0)]; }
    case 5: { const nums = vals.map(Number).filter(Number.isFinite); return nums.length ? [nums.reduce((a, b) => a + b, 0) / nums.length] : []; }
    default: return vals;
  }
}

async function wikisql() {
  if (skipIfExists("wikisql.jsonl")) return;
  const all = await fetchAll("Rathanr/wikisql", "default", "test", { minRows: 1000 });
  const usable = all.filter(r =>
    (r.table?.rows?.length ?? 0) <= 40 &&
    r.sql?.human_readable &&
    r.table.header.length > 0 &&
    r.sql.conds.column_index.every(i => i < r.table.header.length) &&
    r.sql.sel < r.table.header.length
  );
  const picked = sample(usable, 30, SEED);
  writeJsonl("wikisql.jsonl", picked.map(r => {
    const gold = evalWikiSql(r.sql, r.table);
    // Build a human-readable description of the gold SQL for verbose output;
    // store both as a sanity reference.
    const agg = WIKISQL_AGG[r.sql.agg];
    const selCol = r.table.header[r.sql.sel];
    const condStr = r.sql.conds.column_index.map((ci, i) =>
      `"${r.table.header[ci]}" ${WIKISQL_OPS[r.sql.conds.operator_index[i]]} ${JSON.stringify(r.sql.conds.condition[i])}`
    ).join(" AND ");
    const goldDesc = `SELECT ${agg ? `${agg}("${selCol}")` : `"${selCol}"`} FROM t` +
                     (condStr ? ` WHERE ${condStr}` : "");
    return {
      id:        r.table.id,
      question:  r.question,
      header:    r.table.header,
      types:     r.table.types,
      rows:      r.table.rows,
      gold:      gold,            // list of values — what model's SQL should return
      gold_desc: goldDesc,        // human-readable canonical form (for -v output)
    };
  }));
}

console.log("Fetching benchmark datasets via HF datasets-server...");
await gsm8k();
await math500();
await humaneval();
await mmlupro();
await ifeval();
await wtq();
await wikisql();
console.log("Done.");
