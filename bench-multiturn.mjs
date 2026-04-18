#!/usr/bin/env node
/**
 * Multi-turn tool-call probe — the companion to bench-toolcall.mjs.
 *
 * bench-toolcall checks whether the model picks the right tool on the
 * first turn. This probe checks what happens AFTER a tool returns —
 * the failure mode that "loop-prone" local models hit:
 *
 *   - model calls a tool,
 *   - we inject a plausible tool result,
 *   - does the model synthesize a final answer, or does it call the same
 *     tool again / call unrelated tools / emit nothing?
 *
 * Each case specifies:
 *   - initial user prompt
 *   - expected first tool call (name)
 *   - what the tool "returns" (we fabricate a result)
 *   - pass rule: final message (most cases) or a specific chained call
 *
 * Pass rules:
 *   FINAL       — second turn must be a text message, no tool_calls
 *   CHAIN:<nm>  — second turn must call tool <nm> (legitimate chain)
 *   EMPTY_OK    — any response is acceptable; we only check non-loop behavior
 *
 * Fail signatures caught:
 *   - same tool called again with same args (the canonical loop)
 *   - same tool called with different args (arg-hallucination loop)
 *   - unrelated tool called when no tool is needed
 *   - empty/whitespace response when text was expected
 *   - more than 1 tool_call when only 1 was expected
 *
 * Usage:
 *   node bench-multiturn.mjs [--model gemma4:26b] [--host http://localhost:11434] [-v]
 */

const args = process.argv.slice(2);
const arg = (n, fb) => { const i = args.indexOf(n); return i >= 0 ? args[i + 1] : fb; };
const MODEL = arg("--model", "gemma4:26b");
const HOST  = arg("--host",  "http://ollama:11434");
const VERBOSE = args.includes("-v");

// ── Tool catalogue (shared shape with bench-toolcall) ────────────────────────
const TOOLS = [
  { type: "function", function: { name: "email_inbox", description: "List unread emails. Returns id, subject, from, date.",
    parameters: { type: "object", properties: { limit: { type: "integer" } }, required: [] } } },
  { type: "function", function: { name: "email_read", description: "Read email body by id.",
    parameters: { type: "object", properties: { id: { type: "string" } }, required: ["id"] } } },
  { type: "function", function: { name: "calendar_events", description: "List calendar events in a date range.",
    parameters: { type: "object", properties: { start: { type: "string" }, end: { type: "string" } }, required: ["start", "end"] } } },
  { type: "function", function: { name: "task_create", description: "Create a to-do item.",
    parameters: { type: "object", properties: { title: { type: "string" }, due: { type: "string" } }, required: ["title"] } } },
  { type: "function", function: { name: "quote", description: "Get a stock quote.",
    parameters: { type: "object", properties: { symbol: { type: "string" } }, required: ["symbol"] } } },
  { type: "function", function: { name: "web_search", description: "Search the web.",
    parameters: { type: "object", properties: { query: { type: "string" } }, required: ["query"] } } },
];

// ── Cases ────────────────────────────────────────────────────────────────────
// rule: "FINAL" | "CHAIN:<tool>" | "EMPTY_OK"
const CASES = [
  // Synthesis — tool returned useful data, should summarize
  { cat: "synthesis", prompt: "Show me my inbox.", firstTool: "email_inbox",
    toolResult: JSON.stringify([
      { id: "271", subject: "Board deck review — Friday", from: "alice@example.com", date: "2026-04-18" },
      { id: "272", subject: "Your order has shipped", from: "shop@example.net", date: "2026-04-18" },
      { id: "273", subject: "1Password security alert", from: "hello@1password.com", date: "2026-04-17" },
    ]),
    rule: "FINAL" },
  { cat: "synthesis", prompt: "Get the AAPL quote.", firstTool: "quote",
    toolResult: JSON.stringify({ symbol: "AAPL", price: 189.42, change: -1.25, changePct: -0.66, asOf: "2026-04-18T18:00Z" }),
    rule: "FINAL" },
  { cat: "synthesis", prompt: "What's on my calendar this week?", firstTool: "calendar_events",
    toolResult: JSON.stringify([
      { id: "evt-1", title: "Standup", start: "2026-04-21T09:00", end: "2026-04-21T09:30" },
      { id: "evt-2", title: "Lunch with Alex", start: "2026-04-23T12:30", end: "2026-04-23T13:30" },
      { id: "evt-3", title: "Board review", start: "2026-04-24T15:00", end: "2026-04-24T16:30" },
    ]),
    rule: "FINAL" },
  { cat: "synthesis", prompt: "Search the web for OpenAI's latest announcement.", firstTool: "web_search",
    toolResult: JSON.stringify([
      { title: "OpenAI launches GPT-5 Turbo", url: "https://example.com/a", snippet: "New model with improved reasoning..." },
      { title: "OpenAI Q1 2026 roadmap", url: "https://example.com/b", snippet: "Focus on enterprise..." },
    ]),
    rule: "FINAL" },
  { cat: "synthesis", prompt: "Read email 271.", firstTool: "email_read",
    toolResult: JSON.stringify({ id: "271", subject: "Board deck review", body: "Hi, the deck is in the shared drive. Let me know by EOD Thursday if the Q3 section needs changes. — Alice" }),
    rule: "FINAL" },

  // Empty result — tool returned nothing, model must not re-call
  { cat: "empty", prompt: "Any unread emails?", firstTool: "email_inbox",
    toolResult: "[]", rule: "FINAL" },
  { cat: "empty", prompt: "What's on my calendar today?", firstTool: "calendar_events",
    toolResult: "[]", rule: "FINAL" },
  { cat: "empty", prompt: "Search for news about magenta alpacas.", firstTool: "web_search",
    toolResult: "[]", rule: "FINAL" },

  // Error result — tool failed, model should surface or handle
  { cat: "error", prompt: "Get the XYZZY quote.", firstTool: "quote",
    toolResult: JSON.stringify({ error: "Unknown symbol: XYZZY" }),
    rule: "FINAL" },
  { cat: "error", prompt: "Read email 99999.", firstTool: "email_read",
    toolResult: JSON.stringify({ error: "Email not found" }),
    rule: "FINAL" },
  { cat: "error", prompt: "Show me my inbox.", firstTool: "email_inbox",
    toolResult: JSON.stringify({ error: "IMAP connection timeout" }),
    rule: "FINAL" },

  // Legitimate chain — model should call the next tool, not synthesize
  { cat: "chain", prompt: "Read the first email in my inbox and summarize it.", firstTool: "email_inbox",
    toolResult: JSON.stringify([{ id: "271", subject: "Board deck review", from: "alice@example.com", date: "2026-04-18" }]),
    rule: "CHAIN:email_read" },
  { cat: "chain", prompt: "Find my next meeting and add a prep task for it.", firstTool: "calendar_events",
    toolResult: JSON.stringify([{ id: "evt-1", title: "Board review", start: "2026-04-24T15:00" }]),
    rule: "CHAIN:task_create" },

  // Long result (~2K chars) — synthesis should still work, not loop
  { cat: "synthesis", prompt: "Show me my inbox.", firstTool: "email_inbox",
    toolResult: JSON.stringify(Array.from({ length: 30 }, (_, i) => ({
      id: String(300 + i),
      subject: `Update ${i + 1} — progress notes from the team meeting this week`,
      from: `person${i}@example.org`,
      date: "2026-04-18",
    }))),
    rule: "FINAL" },
];

// ── Runner ───────────────────────────────────────────────────────────────────
async function chat(messages) {
  const res = await fetch(`${HOST}/v1/chat/completions`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ model: MODEL, messages, tools: TOOLS, temperature: 0 }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const j = await res.json();
  return j.choices?.[0]?.message ?? {};
}

function argsEqual(a, b) {
  try { return JSON.stringify(JSON.parse(a)) === JSON.stringify(JSON.parse(b)); }
  catch { return a === b; }
}

async function runOne(c) {
  // Turn 1 — user asks, model should call firstTool
  const msgs = [{ role: "user", content: c.prompt }];
  const m1 = await chat(msgs);
  const firstCalls = m1.tool_calls ?? [];
  if (firstCalls.length === 0) {
    return { pass: false, stage: "turn1", reason: "expected tool call on turn 1, model produced text" };
  }
  const first = firstCalls[0];
  if (first.function?.name !== c.firstTool) {
    return { pass: false, stage: "turn1", reason: `turn 1: expected ${c.firstTool}, got ${first.function?.name}` };
  }

  // Turn 2 — inject the tool result, model should synthesize (or chain)
  msgs.push({ role: "assistant", content: m1.content ?? "", tool_calls: firstCalls });
  msgs.push({ role: "tool", tool_call_id: first.id, name: c.firstTool, content: c.toolResult });
  const m2 = await chat(msgs);
  const secondCalls = m2.tool_calls ?? [];

  if (c.rule === "FINAL") {
    if (secondCalls.length > 0) {
      const same = secondCalls[0].function?.name === c.firstTool;
      const sameArgs = same && argsEqual(secondCalls[0].function.arguments, first.function.arguments);
      if (sameArgs) return { pass: false, stage: "turn2", reason: `LOOP: re-called ${c.firstTool} with identical args` };
      if (same) return { pass: false, stage: "turn2", reason: `LOOP: re-called ${c.firstTool} with different args` };
      return { pass: false, stage: "turn2", reason: `unexpected tool call: ${secondCalls[0].function?.name}` };
    }
    const content = (m2.content ?? "").trim();
    if (content.length < 10) return { pass: false, stage: "turn2", reason: `expected synthesis, got ${content.length}-char content` };
    return { pass: true, stage: "turn2", reason: "synthesized" };
  }

  if (c.rule.startsWith("CHAIN:")) {
    const want = c.rule.slice(6);
    if (secondCalls.length === 0) return { pass: false, stage: "turn2", reason: `expected chain call to ${want}, got text` };
    if (secondCalls[0].function?.name !== want) return { pass: false, stage: "turn2", reason: `expected chain to ${want}, got ${secondCalls[0].function?.name}` };
    return { pass: true, stage: "turn2", reason: `chained → ${want}` };
  }

  return { pass: true, stage: "turn2", reason: "accepted (empty_ok)" };
}

// ── Main ─────────────────────────────────────────────────────────────────────
async function main() {
  console.log(`\nbench-multiturn: model=${MODEL} host=${HOST} cases=${CASES.length}\n`);
  const byCat = new Map();
  const fails = [];
  const t0 = performance.now();
  for (const c of CASES) {
    let r;
    try { r = await runOne(c); }
    catch (e) { r = { pass: false, stage: "threw", reason: e.message }; }
    const row = byCat.get(c.cat) ?? { total: 0, pass: 0 };
    row.total++;
    if (r.pass) row.pass++;
    byCat.set(c.cat, row);
    if (!r.pass) fails.push({ c, r });
    if (VERBOSE) console.log(`${r.pass ? "✔" : "✘"} [${c.cat}] ${c.prompt.slice(0, 60)}  — ${r.reason}`);
  }
  const dur = ((performance.now() - t0) / 1000).toFixed(1);

  console.log("");
  console.log("category    | total | pass | pass%");
  console.log("-".repeat(40));
  for (const [cat, r] of byCat) {
    console.log(`${cat.padEnd(12)}| ${String(r.total).padStart(5)} | ${String(r.pass).padStart(4)} | ${(100 * r.pass / r.total).toFixed(0).padStart(4)}%`);
  }
  const total = CASES.length;
  const pass = [...byCat.values()].reduce((a, r) => a + r.pass, 0);
  console.log("-".repeat(40));
  console.log(`${"OVERALL".padEnd(12)}| ${String(total).padStart(5)} | ${String(pass).padStart(4)} | ${(100 * pass / total).toFixed(0).padStart(4)}%`);
  console.log(`\nwall: ${dur}s`);

  if (fails.length) {
    console.log("\nfailures:");
    for (const f of fails) {
      console.log(`  [${f.c.cat}] ${f.c.prompt}\n    → ${f.r.reason}`);
    }
  }
}

main().catch(e => { console.error(e); process.exit(1); });
