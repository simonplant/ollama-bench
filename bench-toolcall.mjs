#!/usr/bin/env node
/**
 * Lightweight tool-calling benchmark for local models via the OpenAI-compat
 * Ollama endpoint. Designed around LifeOps tool shapes (email, calendar,
 * tasks, quote, web search).
 *
 * Usage:
 *   node bench-toolcall.mjs [--model gemma4:26b] [--host http://localhost:11434]
 *
 * Scoring per case:
 *   - tool_call_expected + got_call + right_name + args_superset(expected) → PASS
 *   - tool_call_expected=false + no_call_produced                           → PASS
 *   - otherwise FAIL
 * Schema score: arguments parse as JSON and every declared-required key is
 * present with a non-empty value.
 */

import { TOOLS } from "./bench-tools.mjs";

const args = process.argv.slice(2);
const arg = (n, fb) => { const i = args.indexOf(n); return i >= 0 ? args[i + 1] : fb; };
const MODEL = arg("--model", "gemma4:26b");
const HOST  = arg("--host",  "http://ollama:11434");
const VERBOSE = args.includes("-v");

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
  { cat: "multiple", prompt: "Price of TSLA and create a task to review it tomorrow.", expect: { call: true, name: "quote", args: { symbol: "TSLA" } }, note: "either quote or task_create is acceptable as the first call" },
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
  const res = await fetch(`${HOST}/v1/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      model: MODEL,
      messages: [{ role: "user", content: c.prompt }],
      tools: TOOLS,
      temperature: 0,
    }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
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

  if (calledName !== c.expect.name) {
    return { pass: false, schema: schemaOk, reason: `expected ${c.expect.name}, got ${calledName}` };
  }
  if (!schemaOk) return { pass: false, schema: false, reason: schemaReason };
  // arg-level check (loose — expected is a subset the model MUST reach)
  if (c.expect.args) {
    for (const [k, v] of Object.entries(c.expect.args)) {
      if (parsed[k] !== v) return { pass: false, schema: true, reason: `arg ${k}: got ${JSON.stringify(parsed[k])}, expected ${JSON.stringify(v)}` };
    }
  }
  return { pass: true, schema: true, reason: "ok" };
}

async function main() {
  console.log(`\nbench: model=${MODEL} host=${HOST} cases=${CASES.length}\n`);
  const byCat = new Map();
  const fails = [];
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
    if (!scored.pass) fails.push({ c, scored, out });
    if (VERBOSE) console.log(`${scored.pass ? "✔" : "✘"} [${c.cat}] ${c.prompt.slice(0,60)}  — ${scored.reason}`);
  }
  const dur = ((performance.now() - t0) / 1000).toFixed(1);

  console.log("");
  console.log("category    | total | pass | pass% | schema%");
  console.log("-".repeat(50));
  for (const [cat, r] of byCat) {
    console.log(`${cat.padEnd(12)}| ${String(r.total).padStart(5)} | ${String(r.pass).padStart(4)} | ${(100 * r.pass / r.total).toFixed(0).padStart(4)}% | ${(100 * r.schema / r.total).toFixed(0).padStart(6)}%`);
  }
  const total = CASES.length;
  const pass = [...byCat.values()].reduce((a, r) => a + r.pass, 0);
  const schemaPass = [...byCat.values()].reduce((a, r) => a + r.schema, 0);
  console.log("-".repeat(50));
  console.log(`${"OVERALL".padEnd(12)}| ${String(total).padStart(5)} | ${String(pass).padStart(4)} | ${(100 * pass / total).toFixed(0).padStart(4)}% | ${(100 * schemaPass / total).toFixed(0).padStart(6)}%`);
  console.log(`\nwall: ${dur}s`);

  if (fails.length) {
    console.log("\nfailures:");
    for (const f of fails) {
      console.log(`  [${f.c.cat}] ${f.c.prompt}\n    → ${f.scored.reason}`);
    }
  }
}

main().catch(e => { console.error(e); process.exit(1); });
