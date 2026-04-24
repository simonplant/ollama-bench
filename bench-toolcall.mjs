#!/usr/bin/env node
/**
 * Single-turn tool-call accuracy probe via Ollama's OpenAI-compatible endpoint.
 * Uses the LifeOps tool catalogue (email, calendar, tasks, quote, web search)
 * shared with bench-multiturn.mjs.
 *
 * Usage:
 *   node bench-toolcall.mjs [--model gemma4:26b] [--host http://ollama:11434]
 *                           [--out ./baseline.json] [--save|--compare] [-v|--verbose]
 *
 * Default mode is smart: saves if --model has no toolcall entry in the
 * baseline, compares otherwise. --save forces overwrite; --compare errors
 * if the entry is missing.
 *
 * Scoring per case:
 *   - tool_call_expected + got_call + accepted_name + args_superset(expected) → PASS
 *     (accepted_name = expect.name OR any expect.altNames)
 *   - tool_call_expected=false + no_call_produced                              → PASS
 *   - otherwise FAIL
 *
 * schema% scores arguments against whichever tool the model actually called,
 * independently of pick correctness — a wrong-tool pick with well-formed
 * args still registers valid schema fidelity. Only malformed JSON or missing
 * required keys fail schema. Pass% deltas flag drops past ±REG_PP (5pp) to
 * filter run-to-run noise.
 *
 * Per-call request timeout: 180s, override via OLLAMA_BENCH_TIMEOUT_MS.
 */

import { TOOLS } from "./bench-tools.mjs";
import { getModelSection, writeModelSection } from "./bench-baseline.mjs";

const args = process.argv.slice(2);
// lastIndexOf so a later forwarded flag (e.g. from the ./bench wrapper) wins.
const arg = (n, fb) => { const i = args.lastIndexOf(n); return i >= 0 ? args[i + 1] : fb; };
const MODEL   = arg("--model", "gemma4:26b");
const HOST    = arg("--host",  "http://ollama:11434");
const OUT     = arg("--out",   "./baseline.json");
const VERBOSE = args.includes("-v") || args.includes("--verbose");
const MODE    = args.includes("--save")    ? "save"
              : args.includes("--compare") ? "compare"
              : "smart";

// Per-call timeout (ms). Env override lets weaker hardware bump it.
// Flat ceiling: tool-call responses are short (single message w/ JSON args),
// but model load + prompt-eval with a tools block can still take ~60s on
// first request. 180s covers cold-load + generation with comfortable margin.
function chatTimeoutMs() {
  const override = parseInt(process.env.OLLAMA_BENCH_TIMEOUT_MS ?? "", 10);
  if (Number.isFinite(override) && override > 0) return override;
  return 180_000;
}
function withTimeout(ms) {
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), ms);
  return { signal: ac.signal, cancel: () => clearTimeout(timer) };
}

// Noise-floor threshold for pass%/schema% deltas. Each category has 5–10
// cases, so one case flipping is ~10–20pp. Flagging anything negative
// (the old behavior) turned run-to-run noise into false regressions.
const REG_PP = 5;

// ── Cases ────────────────────────────────────────────────────────────────────
// each case: { cat, prompt, expect: { call: bool, name?, args?: {required keys w/ expected value semantics} } }
const CASES = [
  // simple — one obvious tool
  { cat: "simple",   prompt: "Show me my inbox.",                                    expect: { call: true, name: "email_inbox" } },
  { cat: "simple",   prompt: "Any unread emails?",                                   expect: { call: true, name: "email_inbox" } },
  { cat: "simple",   prompt: "What's my calendar look like from 2026-04-20 to 2026-04-25?", expect: { call: true, name: "calendar_events", args: { start: "2026-04-20", end: "2026-04-25" } } },
  { cat: "simple",   prompt: "Get me the AAPL quote.",                               expect: { call: true, name: "quote", args: { symbol: "AAPL" } } },
  { cat: "simple",   prompt: "Price of NVDA?",                                       expect: { call: true, name: "quote", args: { symbol: "NVDA" } } },
  { cat: "simple",   prompt: "Send an email to alice@example.com with subject 'Lunch?' and body 'How about tomorrow at noon?'", expect: { call: true, name: "email_send", args: { to: "alice@example.com", subject: "Lunch?" } } },
  { cat: "simple",   prompt: "Add a task: File taxes. Due April 15. High priority.", expect: { call: true, name: "task_create", args: { title: "File taxes" } } },
  { cat: "simple",   prompt: "Create a todo item titled 'Order dog food'.",          expect: { call: true, name: "task_create", args: { title: "Order dog food" } } },
  { cat: "simple",   prompt: "Search the web for latest OpenAI announcements.",      expect: { call: true, name: "web_search" } },
  { cat: "simple",   prompt: "Read email 271.",                                      expect: { call: true, name: "email_read", args: { id: "271" } } },
  // multiple — must disambiguate across tools
  { cat: "multiple", prompt: "Price of TSLA and create a task to review it tomorrow.", expect: { call: true, name: "quote", altNames: ["task_create"], args: { symbol: "TSLA" } } },
  { cat: "multiple", prompt: "Check my inbox, I'm expecting something from the bank.", expect: { call: true, name: "email_inbox" } },
  { cat: "multiple", prompt: "What do I have on my calendar today?",                  expect: { call: true, name: "calendar_events" } },
  { cat: "multiple", prompt: "Find news about AI regulation.",                        expect: { call: true, name: "web_search" } },
  { cat: "multiple", prompt: "Remind me to call mom tomorrow.",                       expect: { call: true, name: "task_create", args: { title: "Call mom" } } },
  // relevance — no tool should be called
  { cat: "relevance", prompt: "Hi, how are you?",                                     expect: { call: false } },
  { cat: "relevance", prompt: "What is 17 times 43?",                                 expect: { call: false } },
  { cat: "relevance", prompt: "Explain dollar-cost averaging in two sentences.",      expect: { call: false } },
  { cat: "relevance", prompt: "Tell me a short joke.",                                expect: { call: false } },
  { cat: "relevance", prompt: "Who wrote the book 'Dune'?",                           expect: { call: false } },
  { cat: "relevance", prompt: "What does the word 'sovereign' mean?",                 expect: { call: false } },
  { cat: "relevance", prompt: "Say hello.",                                           expect: { call: false } },
];

// ── Runner ───────────────────────────────────────────────────────────────────
async function runOne(c) {
  const timeoutMs = chatTimeoutMs();
  const t = withTimeout(timeoutMs);
  let res;
  try {
    res = await fetch(`${HOST}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: MODEL,
        messages: [{ role: "user", content: c.prompt }],
        tools: TOOLS,
        temperature: 0,
      }),
      signal: t.signal,
    });
  } catch (e) {
    if (e.name === "AbortError") throw new Error(`tool-call request timed out after ${timeoutMs}ms — check ${HOST} or set OLLAMA_BENCH_TIMEOUT_MS`);
    throw e;
  } finally {
    t.cancel();
  }
  if (!res.ok) {
    const body = (await res.text()).slice(0, 200);
    throw new Error(`HTTP ${res.status}: ${body}`);
  }
  const j = await res.json();
  const msg = j.choices?.[0]?.message ?? {};
  const calls = msg.tool_calls ?? [];
  return { calls, content: msg.content ?? "", raw: msg };
}

function scoreCase(c, out) {
  const { calls } = out;
  const got = calls.length > 0;
  if (!c.expect.call) {
    return { pass: !got, schema: true, reason: got ? `called ${calls[0]?.function?.name} unnecessarily` : "no tool, correct" };
  }
  if (!got) return { pass: false, schema: false, reason: "expected tool call, got none" };

  // Schema fidelity is scored against whatever tool the model actually
  // called, independently of whether it was the right tool. That way the
  // schema% column measures "does the model produce well-formed args",
  // not "does it pick the right tool AND produce well-formed args".
  const first = calls[0];
  const calledName = first.function?.name;
  let parsed;
  let schemaOk = false;
  let schemaReason = "";
  try { parsed = JSON.parse(first.function?.arguments ?? "{}"); }
  catch { schemaReason = `arguments not valid JSON: ${first.function?.arguments?.slice(0, 80)}`; }
  if (parsed) {
    const calledDef = TOOLS.find(t => t.function.name === calledName)?.function;
    if (!calledDef) {
      schemaReason = `unknown tool: ${calledName}`;
    } else {
      const required = calledDef.parameters?.required ?? [];
      const missing = required.filter(k => !(k in parsed) || parsed[k] === "");
      schemaOk = missing.length === 0;
      if (!schemaOk) schemaReason = `missing required keys: ${missing.join(",")}`;
    }
  }

  // `altNames` lets a case accept multiple tools as equally correct first
  // picks — used for prompts that compound ("price of X and add a task…")
  // where either tool is a defensible choice.
  const acceptedNames = [c.expect.name, ...(c.expect.altNames ?? [])];
  if (!acceptedNames.includes(calledName)) {
    const expectStr = acceptedNames.length === 1 ? c.expect.name : `${c.expect.name} or ${c.expect.altNames.join("/")}`;
    return { pass: false, schema: schemaOk, reason: `expected ${expectStr}, got ${calledName}` };
  }
  // arg-level check only applies when the model picked the primary tool —
  // alts are accepted without arg validation (each would need its own spec).
  const argsMatter = calledName === c.expect.name && c.expect.args;
  if (!schemaOk) return { pass: false, schema: false, reason: schemaReason };
  // arg-level check (loose — expected is a subset the model MUST reach)
  if (argsMatter) {
    for (const [k, v] of Object.entries(c.expect.args)) {
      if (parsed[k] !== v) return { pass: false, schema: true, reason: `arg ${k}: got ${JSON.stringify(parsed[k])}, expected ${JSON.stringify(v)}` };
    }
  }
  return { pass: true, schema: true, reason: "ok" };
}

// Case identity for baseline failure diff. Prompts are unique across CASES so
// `cat::prompt` is stable across runs and human-readable when we print diffs.
const caseId = c => `${c.cat}::${c.prompt}`;

async function runCases() {
  const byCat = new Map();
  const failedPrompts = [];
  const t0 = performance.now();
  for (const c of CASES) {
    let out, scored;
    try {
      out = await runOne(c);
      scored = scoreCase(c, out);
    } catch (e) {
      scored = { pass: false, schema: false, reason: `threw: ${e.message}` };
    }
    const row = byCat.get(c.cat) ?? { total: 0, pass: 0, schema: 0 };
    row.total++;
    if (scored.pass) row.pass++;
    if (scored.schema) row.schema++;
    byCat.set(c.cat, row);
    if (!scored.pass) failedPrompts.push(caseId(c));
    if (VERBOSE) console.log(`${scored.pass ? "✔" : "✘"} [${c.cat}] ${c.prompt.slice(0,60)}  — ${scored.reason}`);
  }
  const dur = (performance.now() - t0) / 1000;
  const total = CASES.length;
  const pass   = [...byCat.values()].reduce((a, r) => a + r.pass, 0);
  const schema = [...byCat.values()].reduce((a, r) => a + r.schema, 0);
  return {
    savedAt: new Date().toISOString(),
    model: MODEL,
    total, pass, schema,
    byCat: Object.fromEntries(byCat),
    failedPrompts,
    durationSec: dur,
  };
}

// ── Reporting ────────────────────────────────────────────────────────────────
function fmtPct(n)   { return `${n.toFixed(0)}%`; }
function fmtDelta(d) {
  if (d === null || d === undefined || !Number.isFinite(d)) return "—";
  const sign = d >= 0 ? "+" : "";
  // Flag only drops past the noise-floor threshold — a 1-case flip in a
  // 10-case category is ~10pp and represents real signal; sub-5pp drops
  // are typically reorderings of near-tie picks.
  const tag  = d < -REG_PP ? " ⚠" : "";
  return `${sign}${d.toFixed(0)}pp${tag}`;
}

function printReport(current, base) {
  const byCatBase = base?.byCat ?? null;
  const hasBase = !!base;
  const cols = hasBase
    ? ["category", "total", "pass", "pass%", "Δ pass%", "schema%", "Δ schema%"]
    : ["category", "total", "pass", "pass%", "schema%"];
  const widths = [12, 5, 4, 5, 8, 7, 9].slice(0, cols.length);
  const pad = (s, w, right = true) => right ? String(s).padStart(w) : String(s).padEnd(w);

  console.log("");
  console.log(cols.map((c, i) => i === 0 ? pad(c, widths[i], false) : pad(c, widths[i])).join(" | "));
  console.log("-".repeat(widths.reduce((a, w) => a + w + 3, 0)));

  const rowFor = (label, r, bRow) => {
    const passPct = 100 * r.pass / r.total;
    const schPct  = 100 * r.schema / r.total;
    const bP = bRow ? 100 * bRow.pass / bRow.total : null;
    const bS = bRow ? 100 * bRow.schema / bRow.total : null;
    const dP = bP !== null ? passPct - bP : null;
    const dS = bS !== null ? schPct - bS : null;
    const base = [pad(label, widths[0], false), pad(r.total, widths[1]), pad(r.pass, widths[2]), pad(fmtPct(passPct), widths[3])];
    if (hasBase) return [...base, pad(fmtDelta(dP), widths[4]), pad(fmtPct(schPct), widths[5]), pad(fmtDelta(dS), widths[6])].join(" | ");
    return [...base, pad(fmtPct(schPct), widths[4])].join(" | ");
  };

  for (const [cat, r] of Object.entries(current.byCat)) {
    console.log(rowFor(cat, r, byCatBase?.[cat]));
  }
  console.log("-".repeat(widths.reduce((a, w) => a + w + 3, 0)));
  const overallBase = base ? { total: base.total, pass: base.pass, schema: base.schema } : null;
  console.log(rowFor("OVERALL", { total: current.total, pass: current.pass, schema: current.schema }, overallBase));
  console.log(`\nwall: ${current.durationSec.toFixed(1)}s`);

  if (hasBase) {
    const baseSet = new Set(base.failedPrompts ?? []);
    const curSet  = new Set(current.failedPrompts);
    const newFails  = [...curSet].filter(f => !baseSet.has(f));
    const newPasses = [...baseSet].filter(f => !curSet.has(f));
    if (newFails.length) {
      console.log("\nnewly failed vs baseline:");
      for (const f of newFails) {
        const [cat, prompt] = f.split("::");
        console.log(`  [${cat}] ${prompt}`);
      }
    }
    if (newPasses.length) {
      console.log("\nnewly passed vs baseline:");
      for (const f of newPasses) {
        const [cat, prompt] = f.split("::");
        console.log(`  [${cat}] ${prompt}`);
      }
    }
    const overallDelta = (100 * current.pass / current.total) - (100 * base.pass / base.total);
    if (overallDelta < -REG_PP)      console.log(`\n⚠ overall pass% regressed ${overallDelta.toFixed(0)}pp vs baseline (saved ${base.savedAt})`);
    else if (overallDelta > REG_PP)  console.log(`\n✓ overall pass% improved +${overallDelta.toFixed(0)}pp vs baseline (saved ${base.savedAt})`);
    else                             console.log(`\n✓ overall pass% within ±${REG_PP}pp of baseline (saved ${base.savedAt})`);
  } else if (current.failedPrompts.length) {
    console.log("\nfailures:");
    for (const f of current.failedPrompts) {
      const [cat, prompt] = f.split("::");
      console.log(`  [${cat}] ${prompt}`);
    }
  }
}

// ── Main ─────────────────────────────────────────────────────────────────────
async function main() {
  console.log(`\nbench-toolcall: model=${MODEL} host=${HOST} cases=${CASES.length}\n`);

  // v2 baseline: toolcall is keyed under models[<MODEL>].toolcall, so the
  // model-mismatch case from v1 (one slot for one model) doesn't apply —
  // each model has its own slice. If MODEL has never been benched here,
  // smart mode saves; otherwise compares.
  const existingToolcall = getModelSection(OUT, MODEL, "toolcall");
  let mode = MODE;
  if (mode === "smart") {
    if (!existingToolcall) {
      mode = "save";
      console.log(`(no toolcall entry for ${MODEL} at ${OUT} — saving one now)`);
    } else {
      mode = "compare";
    }
  } else if (mode === "compare" && !existingToolcall) {
    console.error(`no toolcall entry for ${MODEL} at ${OUT} — run with --save first`);
    process.exit(1);
  }

  const current = await runCases();
  printReport(current, mode === "compare" ? existingToolcall : null);

  if (mode === "save") {
    writeModelSection(OUT, MODEL, "toolcall", current);
    console.log(`\ntoolcall entry saved for ${MODEL} → ${OUT}`);
  }
}

main().catch(e => { console.error(e); process.exit(1); });
