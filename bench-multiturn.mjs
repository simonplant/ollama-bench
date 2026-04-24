#!/usr/bin/env node
/**
 * Multi-turn tool-call probe — scores what the model does AFTER a tool returns.
 * Each case: initial user prompt → expected first tool call → fabricated tool
 * result injected → second turn scored against a pass rule.
 *
 * Pass rules:
 *   FINAL       — second turn must be a text message, no tool_calls
 *   CHAIN:<nm>  — second turn must call tool <nm> (legitimate chain)
 *   EMPTY_OK    — any response acceptable; only non-loop behavior is checked
 *
 * Failure signatures:
 *   - same tool called again with identical args        (canonical loop)
 *   - same tool called again with different args        (arg-hallucination loop)
 *   - unrelated tool called when text was expected
 *   - empty/whitespace content where synthesis expected
 *
 * Usage:
 *   node bench-multiturn.mjs [--model gemma4:26b] [--host http://ollama:11434]
 *                            [--out ./baseline.json] [--save|--compare] [-v|--verbose]
 *
 * Default mode is smart: saves if --model has no multiturn entry in the
 * baseline, compares otherwise. Pass% deltas flag drops past ±REG_PP (5pp).
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

// Per-call timeout (ms). Each case does two chat turns, so we apply the
// timeout per turn and let OLLAMA_BENCH_TIMEOUT_MS override if weaker
// hardware needs more headroom.
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

// Noise floor for pass% delta (pp). One case flip = 100/14 ≈ 7pp; drops
// below this are typically reorderings, not regressions.
const REG_PP = 5;

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

  // Empty result — tool returned nothing, model must not re-call.
  // minContent: 1 lets a terse "none" / "no results" pass as a valid reply
  // without requiring the model to pad it out.
  { cat: "empty", prompt: "Any unread emails?", firstTool: "email_inbox",
    toolResult: "[]", rule: "FINAL", minContent: 1 },
  { cat: "empty", prompt: "What's on my calendar today?", firstTool: "calendar_events",
    toolResult: "[]", rule: "FINAL", minContent: 1 },
  { cat: "empty", prompt: "Search for news about magenta alpacas.", firstTool: "web_search",
    toolResult: "[]", rule: "FINAL", minContent: 1 },

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
  { cat: "synthesis", prompt: "Show me my inbox (recent).", firstTool: "email_inbox",
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
  const timeoutMs = chatTimeoutMs();
  const t = withTimeout(timeoutMs);
  let res;
  try {
    res = await fetch(`${HOST}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: MODEL, messages, tools: TOOLS, temperature: 0 }),
      signal: t.signal,
    });
  } catch (e) {
    if (e.name === "AbortError") throw new Error(`multiturn chat timed out after ${timeoutMs}ms — check ${HOST} or set OLLAMA_BENCH_TIMEOUT_MS`);
    throw e;
  } finally {
    t.cancel();
  }
  if (!res.ok) {
    const body = (await res.text()).slice(0, 200);
    throw new Error(`HTTP ${res.status}: ${body}`);
  }
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
    // `empty` cases get a terse pass bar — "None." is a valid reply to
    // "any unread emails?" with result []. Synthesis cases keep the higher
    // floor so a single "ok." doesn't register as a real summary.
    const minLen = c.minContent ?? 10;
    if (content.length < minLen) return { pass: false, stage: "turn2", reason: `expected synthesis, got ${content.length}-char content` };
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

// Case identity for failure diff. Prompts are unique → `cat::prompt` is stable.
const caseId = c => `${c.cat}::${c.prompt}`;

async function runCases() {
  const byCat = new Map();
  const failedPrompts = [];
  const t0 = performance.now();
  for (const c of CASES) {
    let r;
    try { r = await runOne(c); }
    catch (e) { r = { pass: false, stage: "threw", reason: e.message }; }
    const row = byCat.get(c.cat) ?? { total: 0, pass: 0 };
    row.total++;
    if (r.pass) row.pass++;
    byCat.set(c.cat, row);
    if (!r.pass) failedPrompts.push(caseId(c));
    if (VERBOSE) console.log(`${r.pass ? "✔" : "✘"} [${c.cat}] ${c.prompt.slice(0, 60)}  — ${r.reason}`);
  }
  const dur = (performance.now() - t0) / 1000;
  const total = CASES.length;
  const pass = [...byCat.values()].reduce((a, r) => a + r.pass, 0);
  return {
    savedAt: new Date().toISOString(),
    model: MODEL,
    total, pass,
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
  // One case flip in 14 ≈ 7pp; sub-REG_PP drops aren't meaningful signal.
  const tag  = d < -REG_PP ? " ⚠" : "";
  return `${sign}${d.toFixed(0)}pp${tag}`;
}

function printReport(current, base) {
  const hasBase = !!base;
  const cols = hasBase
    ? ["category", "total", "pass", "pass%", "Δ pass%"]
    : ["category", "total", "pass", "pass%"];
  const widths = [12, 5, 4, 5, 8].slice(0, cols.length);
  const pad = (s, w, right = true) => right ? String(s).padStart(w) : String(s).padEnd(w);

  console.log("");
  console.log(cols.map((c, i) => i === 0 ? pad(c, widths[i], false) : pad(c, widths[i])).join(" | "));
  console.log("-".repeat(widths.reduce((a, w) => a + w + 3, 0)));

  const rowFor = (label, r, bRow) => {
    const passPct = 100 * r.pass / r.total;
    const bP = bRow ? 100 * bRow.pass / bRow.total : null;
    const dP = bP !== null ? passPct - bP : null;
    const base = [pad(label, widths[0], false), pad(r.total, widths[1]), pad(r.pass, widths[2]), pad(fmtPct(passPct), widths[3])];
    if (hasBase) return [...base, pad(fmtDelta(dP), widths[4])].join(" | ");
    return base.join(" | ");
  };

  for (const [cat, r] of Object.entries(current.byCat)) {
    console.log(rowFor(cat, r, base?.byCat?.[cat]));
  }
  console.log("-".repeat(widths.reduce((a, w) => a + w + 3, 0)));
  const overallBase = base ? { total: base.total, pass: base.pass } : null;
  console.log(rowFor("OVERALL", { total: current.total, pass: current.pass }, overallBase));
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
  console.log(`\nbench-multiturn: model=${MODEL} host=${HOST} cases=${CASES.length}\n`);

  // v2 baseline: multiturn is keyed under models[<MODEL>].multiturn. Each
  // model has its own slice; no model-mismatch case.
  const existingMultiturn = getModelSection(OUT, MODEL, "multiturn");
  let mode = MODE;
  if (mode === "smart") {
    if (!existingMultiturn) {
      mode = "save";
      console.log(`(no multiturn entry for ${MODEL} at ${OUT} — saving one now)`);
    } else {
      mode = "compare";
    }
  } else if (mode === "compare" && !existingMultiturn) {
    console.error(`no multiturn entry for ${MODEL} at ${OUT} — run with --save first`);
    process.exit(1);
  }

  const current = await runCases();
  printReport(current, mode === "compare" ? existingMultiturn : null);

  if (mode === "save") {
    writeModelSection(OUT, MODEL, "multiturn", current);
    console.log(`\nmultiturn entry saved for ${MODEL} → ${OUT}`);
  }
}

main().catch(e => { console.error(e); process.exit(1); });
