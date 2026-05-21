#!/usr/bin/env node
/**
 * Agent probe — multi-turn ReAct loop with tool execution against mutable
 * world state. Tests whether the model can drive a real workflow end to end:
 * search → read → reason → act → verify.
 *
 * Four categories:
 *   workflow      — multi-step goal completion (email → calendar → task)
 *   recovery      — reason past tool errors/empties without looping or giving up
 *   triage        — sift signal from noisy inbox/calendar
 *   data_analysis — query a sqlite db, interpret results, answer the question
 *
 * Each case runs with a fresh copy of the synthetic world state (inbox,
 * calendar, tasks, db). Tools mutate that state. Grading inspects final
 * state ("did the right email get sent? was the right task created?") and
 * the final text answer.
 *
 * Loop bounds: up to 8 turns per case. A LOOP signature (same tool+args
 * called twice in a row) fails the case immediately.
 *
 * Usage:
 *   node bench-agent.mjs [--model gemma4:26b] [--host http://ollama:11434]
 *                        [--out ./baseline.json] [--save|--compare]
 *                        [--cat workflow|recovery|triage|data_analysis|all]
 *                        [--real-web] [-v|--verbose]
 *
 * --real-web: web_search hits DuckDuckGo's HTML endpoint instead of the
 * fixture. Default is fixture for repeatability.
 */

import { DatabaseSync } from "node:sqlite";
import { getModelSection, writeModelSection } from "./bench-baseline.mjs";
import { startSampler, stopSampler, fmtGpuSummary } from "./bench-gpu.mjs";

const args = process.argv.slice(2);
const arg = (n, fb) => { const i = args.lastIndexOf(n); return i >= 0 ? args[i + 1] : fb; };
const MODEL    = arg("--model", "gemma4:26b");
const HOST     = arg("--host",  "http://ollama:11434");
const OUT      = arg("--out",   "./baseline.json");
const CAT      = arg("--cat",   "all");
const REAL_WEB = args.includes("--real-web");
const VERBOSE  = args.includes("-v") || args.includes("--verbose");
const MODE     = args.includes("--save")    ? "save"
               : args.includes("--compare") ? "compare"
               : "smart";

const REG_PP = 5;
const MAX_TURNS = 8;

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

// ── World fixtures ───────────────────────────────────────────────────────────
// Coherent synthetic state. All dates are anchored to a fixed "today" so cases
// can reference "tomorrow"/"next week" unambiguously regardless of when run.
const TODAY = "2026-05-20";
const TOMORROW = "2026-05-21";

const DEFAULT_INBOX = [
  { id: "e1",  from: "alice@acme.com",       to: "me", subject: "Q3 budget review — please confirm",       date: "2026-05-19", body: "Hi, can you confirm you'll attend the Q3 budget review tomorrow at 10am? — Alice" },
  { id: "e2",  from: "bob@acme.com",         to: "me", subject: "Re: Acme contract draft",                  date: "2026-05-19", body: "Legal flagged section 7.2 of the Acme contract — they want the indemnity cap raised to $2M before we sign. Can we discuss?" },
  { id: "e3",  from: "newsletter@stratechery.com", to: "me", subject: "Stratechery Daily",                  date: "2026-05-19", body: "Today's analysis: Apple's M5 announcement and what it means for AI workloads." },
  { id: "e4",  from: "calendar-bot@acme.com",to: "me", subject: "Meeting tomorrow: Q3 budget review (10am)",date: "2026-05-19", body: "Reminder: Q3 budget review tomorrow May 21 at 10:00 with Alice Chen (organizer)." },
  { id: "e5",  from: "noreply@github.com",   to: "me", subject: "[repo/main] CI passed",                    date: "2026-05-18", body: "Your push to main passed CI." },
  { id: "e6",  from: "carol@acme.com",       to: "me", subject: "Re: Acme contract draft",                  date: "2026-05-18", body: "Legal here. Confirmed — section 7.2 indemnity cap should be $2M. That's our final position." },
  { id: "e7",  from: "amazon@notify.com",    to: "me", subject: "Your order has shipped",                   date: "2026-05-18", body: "Your order #112-3344 has shipped and will arrive May 22." },
  { id: "e8",  from: "alice@acme.com",       to: "me", subject: "Re: lunch?",                                date: "2026-05-17", body: "Sure, Friday works. The usual place?" },
  { id: "e9",  from: "spam@offer.io",        to: "me", subject: "Final notice: upgrade your plan",          date: "2026-05-17", body: "Limited time offer for our premium tier!" },
  { id: "e10", from: "noreply@linkedin.com", to: "me", subject: "You have 4 new connection requests",       date: "2026-05-17", body: "See your new requests on LinkedIn." },
  { id: "e11", from: "dave@vendor.com",      to: "me", subject: "Invoice INV-2026-0421 attached",           date: "2026-05-16", body: "Attached is invoice INV-2026-0421 for May services. Net 30." },
  { id: "e12", from: "team@figma.com",       to: "me", subject: "New comment on design draft",              date: "2026-05-16", body: "Erin commented: 'Can we increase the contrast on the warning state?'" },
  { id: "e13", from: "bob@acme.com",         to: "me", subject: "Q2 retro deck",                            date: "2026-05-15", body: "Final Q2 retro deck attached. Let me know if anything needs to change before Monday." },
  { id: "e14", from: "alice@acme.com",       to: "me", subject: "Welcome to the team!",                     date: "2026-05-01", body: "Welcome aboard. Let me know if you need anything in your first week." },
  { id: "e15", from: "noreply@stripe.com",   to: "me", subject: "Payment received: $1,200.00",              date: "2026-05-15", body: "Payment of $1,200.00 from Acme Corp has been received." },
  { id: "e16", from: "carol@acme.com",       to: "me", subject: "Re: Acme contract draft",                  date: "2026-05-15", body: "I'll send the redline by EOD Friday." },
  { id: "e17", from: "support@aws.com",      to: "me", subject: "Your bill is ready",                       date: "2026-05-14", body: "Your AWS bill for April is $342.10." },
  { id: "e18", from: "events@oreilly.com",   to: "me", subject: "Conference early-bird ends Friday",        date: "2026-05-14", body: "Save 30% on registration before Friday." },
  { id: "e19", from: "dave@vendor.com",      to: "me", subject: "Quick question about Q3 SOW",              date: "2026-05-13", body: "Wanted to check if you've reviewed the Q3 SOW draft I sent last week." },
  { id: "e20", from: "calendar-bot@acme.com",to: "me", subject: "Cancelled: 1:1 with Alice on May 22",      date: "2026-05-13", body: "The 1:1 with Alice scheduled for May 22 has been cancelled." },
];

const DEFAULT_CALENDAR = [
  { id: "c1", title: "Q3 budget review",  start: `${TOMORROW}T10:00`, end: `${TOMORROW}T11:00`, organizer: "alice@acme.com", attendees: ["alice@acme.com", "me"] },
  { id: "c2", title: "1:1 with Bob",      start: `${TOMORROW}T14:00`, end: `${TOMORROW}T14:30`, organizer: "bob@acme.com",   attendees: ["bob@acme.com", "me"] },
  { id: "c3", title: "Team standup",      start: `2026-05-22T09:30`,  end: `2026-05-22T10:00`,  organizer: "alice@acme.com", attendees: ["alice@acme.com", "bob@acme.com", "me"] },
  { id: "c4", title: "Lunch with Alice",  start: `2026-05-23T12:00`,  end: `2026-05-23T13:00`,  organizer: "me",             attendees: ["alice@acme.com", "me"] },
  { id: "c5", title: "Vendor demo",       start: `2026-05-24T15:00`,  end: `2026-05-24T16:00`,  organizer: "dave@vendor.com",attendees: ["dave@vendor.com", "me"] },
  { id: "c6", title: "Q2 retro",          start: `2026-05-26T11:00`,  end: `2026-05-26T12:00`,  organizer: "bob@acme.com",   attendees: ["alice@acme.com", "bob@acme.com", "carol@acme.com", "me"] },
];

const DEFAULT_TASKS = [
  { id: "t1", title: "Review Acme contract redline",   due: "2026-05-22", priority: "high",   done: false },
  { id: "t2", title: "File Q1 expense report",         due: "2026-05-25", priority: "medium", done: false },
];

// Synthetic e-commerce dataset for data_analysis cases.
function buildDataDb() {
  const db = new DatabaseSync(":memory:");
  db.exec(`
    CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, category TEXT, price REAL);
    CREATE TABLE orders   (id INTEGER PRIMARY KEY, product_id INTEGER, quantity INTEGER, total REAL, order_date TEXT, status TEXT);
    CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, signup_date TEXT, country TEXT);
  `);
  const products = [
    [1, "Widget Pro",     "hardware",    49.99],
    [2, "Gadget Max",     "hardware",    89.99],
    [3, "API Subscription","software",   29.00],
    [4, "Pro Support",    "services",   199.00],
    [5, "Sticker Pack",   "merch",       12.00],
    [6, "Widget Mini",    "hardware",    24.99],
    [7, "Cloud Storage 1TB","software",  14.99],
    [8, "Onboarding Session","services",499.00],
  ];
  for (const p of products) db.prepare("INSERT INTO products VALUES (?,?,?,?)").run(...p);
  // Orders spanning the last quarter — Q2 2026 = Apr/May/Jun 2026
  const orders = [
    [101, 1, 10, 499.90, "2026-04-03", "completed"],
    [102, 2,  5, 449.95, "2026-04-15", "completed"],
    [103, 3, 20, 580.00, "2026-04-20", "completed"],
    [104, 4,  1, 199.00, "2026-04-22", "completed"],
    [105, 1,  3, 149.97, "2026-04-28", "returned"],
    [106, 5, 50, 600.00, "2026-05-02", "completed"],
    [107, 2,  2, 179.98, "2026-05-04", "completed"],
    [108, 6, 12, 299.88, "2026-05-08", "completed"],
    [109, 7, 30, 449.70, "2026-05-11", "completed"],
    [110, 8,  1, 499.00, "2026-05-12", "completed"],
    [111, 3, 10, 290.00, "2026-05-14", "returned"],
    [112, 1,  8, 399.92, "2026-05-15", "completed"],
    [113, 4,  1, 199.00, "2026-05-16", "completed"],
    [114, 2,  3, 269.97, "2026-05-18", "completed"],
  ];
  for (const o of orders) db.prepare("INSERT INTO orders VALUES (?,?,?,?,?,?)").run(...o);
  return db;
}

// Web search fixture — keyed by lowercase query substring → results.
const WEB_FIXTURES = [
  { q: "anthropic claude",  results: [
      { title: "Claude (Anthropic) — Wikipedia", snippet: "Claude is a family of large language models developed by Anthropic.", url: "https://en.wikipedia.org/wiki/Claude_(language_model)" },
      { title: "Anthropic homepage",             snippet: "Anthropic builds AI systems people can rely on.",                     url: "https://www.anthropic.com" },
    ]},
  { q: "current price of nvda",  results: [
      { title: "NVIDIA Corp (NVDA) Stock Quote",  snippet: "Live quote not available in test fixtures. Use the quote tool.",     url: "https://finance.example/NVDA" },
    ]},
  { q: "rfc 9110",  results: [
      { title: "RFC 9110: HTTP Semantics",        snippet: "The Hypertext Transfer Protocol (HTTP) is a stateless application-level protocol...", url: "https://www.rfc-editor.org/rfc/rfc9110" },
    ]},
];

// ── World-state instance per case ────────────────────────────────────────────
function newWorld(overrides = {}) {
  return {
    inbox:        structuredClone(overrides.inbox    ?? DEFAULT_INBOX),
    calendar:     structuredClone(overrides.calendar ?? DEFAULT_CALENDAR),
    tasks:        structuredClone(overrides.tasks    ?? DEFAULT_TASKS),
    sentEmails:   [],
    createdTasks: [],
    createdEvents:[],
    today: TODAY,
    db: null, // lazy — only data_analysis cases build it
    // Fail-injection counters — incremented per tool call, used by failInject
    callCounts: {},
  };
}

// ── Tool catalogue ───────────────────────────────────────────────────────────
const TOOLS = [
  { type: "function", function: {
      name: "email_inbox",
      description: "List recent emails in the inbox, newest first.",
      parameters: { type: "object", properties: {
        limit: { type: "integer", description: "Max number of emails to return (default 10)." },
      }, required: [] },
  }},
  { type: "function", function: {
      name: "email_search",
      description: "Search emails by sender, subject, or body text.",
      parameters: { type: "object", properties: {
        query: { type: "string", description: "Free-text query. Matches sender, subject, or body (case-insensitive)." },
      }, required: ["query"] },
  }},
  { type: "function", function: {
      name: "email_read",
      description: "Read the full body of one email by id.",
      parameters: { type: "object", properties: {
        id: { type: "string", description: "Email id (e.g. 'e3')." },
      }, required: ["id"] },
  }},
  { type: "function", function: {
      name: "email_send",
      description: "Send an email.",
      parameters: { type: "object", properties: {
        to:      { type: "string" },
        subject: { type: "string" },
        body:    { type: "string" },
      }, required: ["to", "subject", "body"] },
  }},
  { type: "function", function: {
      name: "calendar_events",
      description: "List calendar events in a date range (ISO date strings YYYY-MM-DD).",
      parameters: { type: "object", properties: {
        start: { type: "string", description: "Start date inclusive. Defaults to today." },
        end:   { type: "string", description: "End date inclusive. Defaults to start+7d." },
      }, required: [] },
  }},
  { type: "function", function: {
      name: "calendar_create",
      description: "Create a calendar event.",
      parameters: { type: "object", properties: {
        title:     { type: "string" },
        start:     { type: "string", description: "ISO datetime YYYY-MM-DDTHH:MM" },
        end:       { type: "string", description: "ISO datetime YYYY-MM-DDTHH:MM" },
        attendees: { type: "array", items: { type: "string" } },
      }, required: ["title", "start", "end"] },
  }},
  { type: "function", function: {
      name: "task_list",
      description: "List current tasks.",
      parameters: { type: "object", properties: {}, required: [] },
  }},
  { type: "function", function: {
      name: "task_create",
      description: "Create a new task.",
      parameters: { type: "object", properties: {
        title:    { type: "string" },
        due:      { type: "string", description: "ISO date YYYY-MM-DD" },
        priority: { type: "string", description: "low | medium | high" },
      }, required: ["title"] },
  }},
  { type: "function", function: {
      name: "web_search",
      description: "Search the web. Returns a list of {title, snippet, url}.",
      parameters: { type: "object", properties: {
        query: { type: "string" },
      }, required: ["query"] },
  }},
  { type: "function", function: {
      name: "list_tables",
      description: "List tables in the analytics database.",
      parameters: { type: "object", properties: {}, required: [] },
  }},
  { type: "function", function: {
      name: "describe_table",
      description: "Get the schema of a table.",
      parameters: { type: "object", properties: {
        name: { type: "string" },
      }, required: ["name"] },
  }},
  { type: "function", function: {
      name: "query_data",
      description: "Execute a SQLite SELECT against the analytics database. Result rows capped at 50.",
      parameters: { type: "object", properties: {
        sql: { type: "string" },
      }, required: ["sql"] },
  }},
];

// ── Tool execution ───────────────────────────────────────────────────────────
async function execTool(name, rawArgs, world, failInject) {
  let argsObj;
  try { argsObj = JSON.parse(rawArgs ?? "{}"); }
  catch { return { error: `invalid JSON arguments: ${rawArgs}` }; }

  // Fail injection — used by recovery cases to test resilience.
  world.callCounts[name] = (world.callCounts[name] ?? 0) + 1;
  const fi = failInject?.[name];
  if (fi && world.callCounts[name] <= (fi.failTimes ?? 1)) {
    if (fi.shape === "error") return { error: fi.message ?? "service unavailable" };
    if (fi.shape === "empty") return Array.isArray(fi.emptyAs) ? [] : { [fi.emptyKey ?? "results"]: [] };
  }

  switch (name) {
    case "email_inbox": {
      const limit = argsObj.limit ?? 10;
      return world.inbox.slice(0, limit).map(e => ({ id: e.id, from: e.from, subject: e.subject, date: e.date }));
    }
    case "email_search": {
      const q = String(argsObj.query ?? "").toLowerCase();
      if (!q) return { error: "query is required" };
      return world.inbox
        .filter(e => `${e.from} ${e.subject} ${e.body}`.toLowerCase().includes(q))
        .map(e => ({ id: e.id, from: e.from, subject: e.subject, date: e.date }));
    }
    case "email_read": {
      const e = world.inbox.find(x => x.id === argsObj.id);
      if (!e) return { error: `no email with id ${argsObj.id}` };
      return { id: e.id, from: e.from, to: e.to, subject: e.subject, date: e.date, body: e.body };
    }
    case "email_send": {
      const sent = { to: argsObj.to, subject: argsObj.subject, body: argsObj.body, id: `sent_${world.sentEmails.length + 1}` };
      world.sentEmails.push(sent);
      return { sent: true, id: sent.id };
    }
    case "calendar_events": {
      const start = argsObj.start ?? TODAY;
      const end   = argsObj.end   ?? TODAY.replace(/-\d{2}$/, m => `-${String(Number(m.slice(1)) + 7).padStart(2, "0")}`);
      return world.calendar.filter(e => {
        const d = e.start.slice(0, 10);
        return d >= start && d <= end;
      });
    }
    case "calendar_create": {
      const ev = { id: `c_new_${world.createdEvents.length + 1}`, title: argsObj.title, start: argsObj.start, end: argsObj.end, attendees: argsObj.attendees ?? [] };
      world.createdEvents.push(ev);
      world.calendar.push(ev);
      return { created: true, id: ev.id };
    }
    case "task_list": return world.tasks.filter(t => !t.done);
    case "task_create": {
      const tk = { id: `t_new_${world.createdTasks.length + 1}`, title: argsObj.title, due: argsObj.due ?? null, priority: argsObj.priority ?? "medium", done: false };
      world.createdTasks.push(tk);
      world.tasks.push(tk);
      return { created: true, id: tk.id };
    }
    case "web_search": {
      const q = String(argsObj.query ?? "").toLowerCase();
      if (REAL_WEB) return await realWebSearch(q);
      const hit = WEB_FIXTURES.find(f => q.includes(f.q));
      return hit ? hit.results : [];
    }
    case "list_tables": {
      if (!world.db) world.db = buildDataDb();
      return world.db.prepare("SELECT name FROM sqlite_master WHERE type='table'").all().map(r => r.name);
    }
    case "describe_table": {
      if (!world.db) world.db = buildDataDb();
      try {
        const rows = world.db.prepare(`PRAGMA table_info("${String(argsObj.name).replace(/"/g, '""')}")`).all();
        if (!rows.length) return { error: `no table named ${argsObj.name}` };
        return rows.map(r => ({ name: r.name, type: r.type }));
      } catch (e) { return { error: e.message }; }
    }
    case "query_data": {
      if (!world.db) world.db = buildDataDb();
      try {
        const rows = world.db.prepare(argsObj.sql).all();
        return { rows: rows.slice(0, 50), truncated: rows.length > 50 };
      } catch (e) { return { error: e.message }; }
    }
    default:
      return { error: `unknown tool ${name}` };
  }
}

async function realWebSearch(query) {
  // DuckDuckGo HTML endpoint — no API key, returns scrape-friendly HTML.
  // Best-effort parse — keep three results.
  try {
    const url = `https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`;
    const res = await fetch(url, { headers: { "User-Agent": "Mozilla/5.0 ollama-bench" } });
    const html = await res.text();
    const out = [];
    const re = /<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>([\s\S]*?)<\/a>[\s\S]*?<a[^>]+class="result__snippet"[^>]*>([\s\S]*?)<\/a>/g;
    let m;
    while ((m = re.exec(html)) && out.length < 5) {
      out.push({ url: m[1], title: stripTags(m[2]), snippet: stripTags(m[3]) });
    }
    return out;
  } catch (e) {
    return { error: `web fetch failed: ${e.message}` };
  }
}
function stripTags(s) { return s.replace(/<[^>]+>/g, "").replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">").trim(); }

// ── Cases ────────────────────────────────────────────────────────────────────
// Each case: { id, cat, prompt, [failInject], grade(world, answerText) → {pass, reason} }
const CASES = [
  // ── workflow ──
  { id: "workflow::confirm_meeting", cat: "workflow",
    prompt: `Find tomorrow's first meeting on my calendar and send a reply to the organizer confirming I'll attend.`,
    grade: (w) => {
      const e = w.sentEmails.find(m => /alice@acme\.com/i.test(m.to));
      if (!e) return { pass: false, reason: "no email sent to alice@acme.com" };
      if (!/budget|q3|confirm|attend|10/i.test(`${e.subject} ${e.body}`)) return { pass: false, reason: "email doesn't reference the meeting" };
      return { pass: true, reason: "ok" };
    },
  },
  { id: "workflow::contract_followup", cat: "workflow",
    prompt: `Read the latest email from legal (Carol) about the Acme contract, then create a task to action it.`,
    grade: (w) => {
      const t = w.createdTasks.find(t => /acme|contract|indemnity|7\.2|2m|redline/i.test(t.title));
      if (!t) return { pass: false, reason: "no task created referencing the Acme contract" };
      return { pass: true, reason: "ok" };
    },
  },
  { id: "workflow::schedule_followup", cat: "workflow",
    prompt: `Find the most recent email about an invoice. Create a task to review and pay it, due in 30 days.`,
    grade: (w) => {
      const t = w.createdTasks.find(t => /invoice|inv-2026|pay/i.test(t.title));
      if (!t) return { pass: false, reason: "no task created for the invoice" };
      // Roughly 30 days from today (2026-05-20) = around 2026-06-19. Accept ±5 days.
      if (t.due) {
        const due = new Date(t.due);
        const target = new Date("2026-06-19");
        const diffDays = Math.abs((due - target) / 86_400_000);
        if (diffDays > 5) return { pass: false, reason: `due ${t.due} not within 5 days of 2026-06-19` };
      }
      return { pass: true, reason: "ok" };
    },
  },
  { id: "workflow::block_focus_time", cat: "workflow",
    prompt: `Create a 2-hour calendar event on 2026-05-23 from 09:00 to 11:00 titled "Deep work — review Q2 retro deck".`,
    grade: (w) => {
      const e = w.createdEvents.find(ev => /deep work|q2 retro/i.test(ev.title));
      if (!e) return { pass: false, reason: "no calendar event created" };
      if (!e.start?.includes("2026-05-23") || !e.start?.includes("09:00")) return { pass: false, reason: `event start ${e.start} wrong` };
      return { pass: true, reason: "ok" };
    },
  },

  // ── recovery ──
  { id: "recovery::search_fails_then_inbox", cat: "recovery",
    prompt: `Find all emails from Carol about the Acme contract.`,
    failInject: { email_search: { failTimes: 1, shape: "error", message: "search index temporarily unavailable" } },
    grade: (w, answer) => {
      // Recovery success: model fell back to email_inbox (or retried search
      // with different args after the first failure) and surfaced Carol's
      // emails in the final answer.
      const usedInboxOrRetry = (w.callCounts.email_inbox ?? 0) >= 1 || (w.callCounts.email_search ?? 0) >= 2;
      if (!usedInboxOrRetry) return { pass: false, reason: "model didn't retry or fall back after search error" };
      const a = (answer ?? "").toLowerCase();
      const mentionsCarol = /carol/.test(a);
      const mentionsContract = /acme|contract|indemnity|7\.2|2m|redline/.test(a);
      if (!mentionsCarol || !mentionsContract) return { pass: false, reason: "answer doesn't surface Carol's contract emails" };
      return { pass: true, reason: "ok" };
    },
  },
  { id: "recovery::empty_result", cat: "recovery",
    prompt: `Find emails from sender 'frank@example.com' about onboarding. Tell me what you find.`,
    grade: (w, answer) => {
      // Frank doesn't exist in the inbox — model should report "no results"
      // rather than fabricate. Failure modes: hallucinated emails, infinite
      // re-querying (caught by MAX_TURNS), or empty final answer.
      const a = (answer ?? "").toLowerCase();
      if (a.length === 0) return { pass: false, reason: "empty answer" };
      const acknowledgesNone = /\bno\b|none|nothing|didn'?t find|couldn'?t find|did not find|no results|no emails|no matches|not\s+find/i.test(answer);
      if (!acknowledgesNone) return { pass: false, reason: "didn't acknowledge no results / possibly fabricated" };
      // Loose check that model didn't fabricate frank's email
      if (/frank.*(?:sent|wrote|said|emailed|attached)/i.test(answer)) return { pass: false, reason: "appears to fabricate content from frank" };
      return { pass: true, reason: "ok" };
    },
  },
  { id: "recovery::malformed_arg_repair", cat: "recovery",
    prompt: `Show me my calendar for next Tuesday only (the week starting Monday 2026-05-25).`,
    grade: (w) => {
      // Success criterion: model called calendar_events with a sensible date
      // range — narrow window around 2026-05-26. We don't require an exact
      // match (the model may pass start=end=05-26 or a 1-2 day window).
      const calls = w.callCounts.calendar_events ?? 0;
      return calls >= 1
        ? { pass: true, reason: "ok" }
        : { pass: false, reason: "didn't call calendar_events" };
    },
  },

  // ── triage ──
  { id: "triage::find_contract_thread", cat: "triage",
    prompt: `Summarize what legal (Carol) and Bob have said about the Acme contract. Cite the key facts.`,
    grade: (w, answer) => {
      const a = (answer ?? "").toLowerCase();
      const hasCap   = /2m|\$2|2 million|indemnity/.test(a);
      const hasSec7  = /7\.2|section 7|section seven/.test(a);
      const hasNames = /carol/.test(a) && /bob/.test(a);
      if (!(hasCap && hasSec7 && hasNames)) return { pass: false, reason: `missing facts (cap:${hasCap} sec7:${hasSec7} names:${hasNames})` };
      return { pass: true, reason: "ok" };
    },
  },
  { id: "triage::actionable_only", cat: "triage",
    prompt: `Look through my recent inbox and list only the emails I need to personally respond to (not notifications, newsletters, or automated messages).`,
    grade: (w, answer) => {
      const a = (answer ?? "").toLowerCase();
      // Should mention Alice's budget review confirm, Bob/Carol on contract,
      // Dave's Q3 SOW question. Should NOT list GitHub CI, Amazon shipping,
      // LinkedIn, Stripe payment, AWS bill notifications.
      const mentions = (s) => a.includes(s);
      const has = ["alice", "carol", "bob"].filter(mentions).length;
      const falsePos = ["github", "amazon", "linkedin", "stripe", "aws bill"].filter(mentions).length;
      if (has < 2) return { pass: false, reason: `only ${has} of 3 actionable senders surfaced` };
      if (falsePos > 1) return { pass: false, reason: `${falsePos} notification/automated mentions leaked into list` };
      return { pass: true, reason: "ok" };
    },
  },

  // ── data_analysis ──
  { id: "data::top_categories", cat: "data_analysis",
    prompt: `Using the analytics database, find the top 3 product categories by total revenue in May 2026, excluding returned orders. Give the category name and total.`,
    grade: (w, answer) => {
      // Gold: May completed orders only
      //   software: 449.70 (cloud) + 0 (api returned) = 449.70
      //   hardware: 299.88 (mini) + 179.98 (gadget) + 399.92 (widget) + 269.97 = 1149.75
      //   services: 499 + 199 = 698
      //   merch: 600
      // Top 3 by revenue: hardware, services, merch
      const a = (answer ?? "").toLowerCase();
      const hits = ["hardware", "services", "merch"].filter(c => a.includes(c)).length;
      if (hits < 3) return { pass: false, reason: `top-3 categories missing (found ${hits}/3)` };
      return { pass: true, reason: "ok" };
    },
  },
  { id: "data::best_seller", cat: "data_analysis",
    prompt: `Which single product had the most units sold across all completed orders? Give the product name.`,
    grade: (w, answer) => {
      // Gold: Sticker Pack (50 units, completed)
      const a = (answer ?? "").toLowerCase();
      return /sticker/.test(a)
        ? { pass: true, reason: "ok" }
        : { pass: false, reason: "didn't identify Sticker Pack" };
    },
  },
  { id: "data::return_rate", cat: "data_analysis",
    prompt: `What percentage of all orders (regardless of date) were returned? Give a numeric percentage.`,
    grade: (w, answer) => {
      // Gold: 2 returned of 14 total = ~14.3%
      const nums = [...(answer ?? "").matchAll(/(\d+(?:\.\d+)?)\s*%?/g)].map(m => Number(m[1]));
      const close = nums.some(n => Math.abs(n - 14.3) < 1.5);
      return close ? { pass: true, reason: "ok" } : { pass: false, reason: `no ~14% figure in answer (got: ${nums.slice(0, 5).join(",")})` };
    },
  },
];

// ── Generator + agent loop ───────────────────────────────────────────────────
async function chat(messages) {
  const timeoutMs = chatTimeoutMs();
  const t = withTimeout(timeoutMs);
  let res;
  try {
    res = await fetch(`${HOST}/v1/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: MODEL,
        messages,
        tools: TOOLS,
        temperature: 0,
      }),
      signal: t.signal,
    });
  } catch (e) {
    if (e.name === "AbortError") throw new Error(`chat timed out after ${timeoutMs}ms`);
    throw e;
  } finally {
    t.cancel();
  }
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);
  return res.json();
}

async function runAgent(c) {
  const world = newWorld();
  const messages = [
    { role: "system", content: `Today's date is ${TODAY}. You have access to tools for email, calendar, tasks, web search, and a sqlite analytics database. Use them to complete the user's request. When you have a final answer, respond with text (no further tool calls).` },
    { role: "user",   content: c.prompt },
  ];
  let lastSig = null;     // tool+args signature, for loop detection
  let lastAnswer = "";
  for (let turn = 0; turn < MAX_TURNS; turn++) {
    let j;
    try { j = await chat(messages); }
    catch (e) { return { pass: false, reason: `chat threw on turn ${turn + 1}: ${e.message}`, world, turns: turn, answer: lastAnswer }; }
    const msg  = j.choices?.[0]?.message ?? {};
    const calls = msg.tool_calls ?? [];
    if (calls.length === 0) {
      // No tool call → final text answer.
      lastAnswer = msg.content ?? "";
      return { ...c.grade(world, lastAnswer), world, turns: turn + 1, answer: lastAnswer };
    }
    // Loop detection: a tool call with same name+args as the immediately
    // previous call signals the model isn't making progress.
    const call = calls[0];
    const sig = `${call.function?.name}::${call.function?.arguments}`;
    if (sig === lastSig) {
      return { pass: false, reason: `LOOP: re-called ${call.function?.name} with identical args`, world, turns: turn + 1, answer: lastAnswer };
    }
    lastSig = sig;

    // Append the assistant turn (preserving all tool calls) and execute them.
    messages.push(msg);
    for (const tc of calls) {
      const result = await execTool(tc.function?.name, tc.function?.arguments, world, c.failInject);
      messages.push({
        role: "tool",
        tool_call_id: tc.id,
        content: JSON.stringify(result),
      });
    }
  }
  return { pass: false, reason: `did not converge in ${MAX_TURNS} turns`, world, turns: MAX_TURNS, answer: lastAnswer };
}

// ── Runner ───────────────────────────────────────────────────────────────────
async function runCases() {
  const cases = CAT === "all" ? CASES : CASES.filter(c => c.cat === CAT);
  if (!cases.length) {
    console.error(`no cases for cat=${CAT}`);
    process.exit(1);
  }
  const byCat = new Map();
  const failedCases = [];
  const gpuHandle = startSampler();
  const t0 = performance.now();
  for (const c of cases) {
    let scored;
    try { scored = await runAgent(c); }
    catch (e) { scored = { pass: false, reason: `threw: ${e.message}`, turns: 0 }; }
    const row = byCat.get(c.cat) ?? { total: 0, pass: 0, turns: 0 };
    row.total++;
    if (scored.pass) row.pass++;
    row.turns += scored.turns ?? 0;
    byCat.set(c.cat, row);
    if (!scored.pass) failedCases.push(`${c.id}::${scored.reason}`);
    if (VERBOSE) console.log(`${scored.pass ? "✔" : "✘"} ${c.id}  [${scored.turns}t]  — ${scored.reason}`);
  }
  const durationSec = (performance.now() - t0) / 1000;
  const gpu = await stopSampler(gpuHandle);
  const total = cases.length;
  const pass  = [...byCat.values()].reduce((a, r) => a + r.pass, 0);
  return {
    savedAt: new Date().toISOString(),
    model: MODEL,
    realWeb: REAL_WEB,
    total, pass,
    agentPct: 100 * pass / total,
    byCategory: Object.fromEntries([...byCat].map(([k, v]) => [k, { total: v.total, pass: v.pass, avgTurns: v.turns / v.total }])),
    failed: failedCases,
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
    ? ["category", "total", "pass", "pass%", "Δ pass%", "avg turns"]
    : ["category", "total", "pass", "pass%", "avg turns"];
  const widths = [16, 5, 4, 5, 8, 9].slice(0, cols.length);
  const pad = (s, w, right = true) => right ? String(s).padStart(w) : String(s).padEnd(w);

  console.log("");
  console.log(cols.map((c, i) => i === 0 ? pad(c, widths[i], false) : pad(c, widths[i])).join(" | "));
  console.log("-".repeat(widths.reduce((a, w) => a + w + 3, 0)));

  for (const [cat, r] of Object.entries(current.byCategory)) {
    const br = base?.byCategory?.[cat];
    const pct = 100 * r.pass / r.total;
    const bp  = br ? 100 * br.pass / br.total : null;
    const cells = [pad(cat, widths[0], false), pad(r.total, widths[1]), pad(r.pass, widths[2]), pad(fmtPct(pct), widths[3])];
    if (base) cells.push(pad(fmtDelta(bp != null ? pct - bp : null), widths[4]), pad(r.avgTurns.toFixed(1), widths[5]));
    else cells.push(pad(r.avgTurns.toFixed(1), widths[4]));
    console.log(cells.join(" | "));
  }
  console.log("-".repeat(widths.reduce((a, w) => a + w + 3, 0)));

  const totalTurns = Object.values(current.byCategory).reduce((a, r) => a + r.avgTurns * r.total, 0);
  const overall = [pad("OVERALL", widths[0], false), pad(current.total, widths[1]), pad(current.pass, widths[2]), pad(fmtPct(current.agentPct), widths[3])];
  if (base) overall.push(pad(fmtDelta(current.agentPct - base.agentPct), widths[4]), pad((totalTurns / current.total).toFixed(1), widths[5]));
  else overall.push(pad((totalTurns / current.total).toFixed(1), widths[4]));
  console.log(overall.join(" | "));
  console.log(`\nwall: ${current.durationSec.toFixed(1)}s` + (REAL_WEB ? "  (--real-web)" : "") + `   ${fmtGpuSummary(current.gpu)}`);

  if (current.failed.length) {
    console.log("\nfailures:");
    for (const f of current.failed.slice(0, 15)) console.log(`  ${f}`);
  }
}

// ── Main ─────────────────────────────────────────────────────────────────────
async function main() {
  console.log(`\nbench-agent: model=${MODEL} host=${HOST} cat=${CAT}${REAL_WEB ? " (--real-web)" : ""}\n`);
  const existing = getModelSection(OUT, MODEL, "agent");
  let mode = MODE;
  if (mode === "smart") mode = existing ? "compare" : "save";
  else if (mode === "compare" && !existing) {
    console.error(`no agent entry for ${MODEL} at ${OUT} — run with --save first`);
    process.exit(1);
  }

  const current = await runCases();
  printReport(current, mode === "compare" ? existing : null);

  if (mode === "save") {
    writeModelSection(OUT, MODEL, "agent", current);
    console.log(`\nagent entry saved for ${MODEL} → ${OUT}`);
  }
}

main().catch(e => { console.error(e); process.exit(1); });
