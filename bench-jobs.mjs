#!/usr/bin/env node
/**
 * Job-shaped quality probe. Scores a model on the actual task roles a router
 * needs to differentiate: structured synthesis, adversarial short-form,
 * long-form distillation, ambiguous tool-use, and multi-step reasoning.
 *
 * Hybrid scoring per case:
 *   - deterministic: JSON parse, required keys, gold-label match, numeric tolerance
 *   - judge:         a separate model rates the response on a 0–3 rubric
 *   - caseScore = 0.5 * deterministic + 0.5 * (judge / 3), all in [0,1]
 *
 * Judge model defaults to gemma4:31b. Auto-swaps to gpt-oss:20b when the
 * candidate IS the default judge (so the model isn't asked to grade itself).
 * Override with --judge or OLLAMA_BENCH_JUDGE. The judge tag must already
 * appear in `ollama list` — the probe refuses to let Ollama auto-pull a
 * missing judge, which on this box has triggered host-killer model loads.
 *
 * Two-pass execution (under OLLAMA_MAX_LOADED_MODELS=1):
 *   pass 1 — candidate generates responses for all cases (one warm load)
 *   pass 2 — judge loads once, scores all cached responses
 *
 * Usage:
 *   node bench-jobs.mjs [--model gemma4:26b] [--host http://ollama:11434]
 *                       [--judge gemma4:31b] [--out ./baseline.json]
 *                       [--save|--compare] [-v|--verbose]
 *
 * Per-call request timeout: 240s, override via OLLAMA_BENCH_TIMEOUT_MS.
 */

import { TOOLS } from "./bench-tools.mjs";
import { getModelSection, writeModelSection } from "./bench-baseline.mjs";

const args = process.argv.slice(2);
const arg = (n, fb) => { const i = args.lastIndexOf(n); return i >= 0 ? args[i + 1] : fb; };
const MODEL   = arg("--model", "gemma4:26b");
const HOST    = arg("--host",  "http://ollama:11434");
const OUT     = arg("--out",   "./baseline.json");
const VERBOSE = args.includes("-v") || args.includes("--verbose");
const MODE    = args.includes("--save")    ? "save"
              : args.includes("--compare") ? "compare"
              : "smart";

const JUDGE_DEFAULT  = "gemma4:31b";
const JUDGE_FALLBACK = "gpt-oss:20b";
const JUDGE_CLI      = arg("--judge", null);

// When the candidate model is also the desired judge, swap to the fallback so
// self-eval bias doesn't confound the score.
function pickJudge(target) {
  const desired = JUDGE_CLI || process.env.OLLAMA_BENCH_JUDGE || JUDGE_DEFAULT;
  return desired === target ? JUDGE_FALLBACK : desired;
}
const JUDGE = pickJudge(MODEL);

// Refuse to run with a judge that isn't already pulled. Ollama auto-pulls on
// first reference, which on this box has triggered a host-killer model just
// by running `bench rank`. Treats `tag` and `tag-<quant>` as the same model
// since Ollama exposes both forms in `/api/tags` for a single pull.
async function assertJudgeInstalled() {
  const url = `${HOST}/api/tags`;
  const t = withTimeout(10_000);
  let res;
  try {
    res = await fetch(url, { signal: t.signal });
  } catch (e) {
    if (e.name === "AbortError") throw new Error(`bench-jobs: ${url} timed out after 10s — is Ollama up?`);
    throw new Error(`bench-jobs: could not reach ${url} to verify judge: ${e.message}`);
  } finally {
    t.cancel();
  }
  if (!res.ok) throw new Error(`bench-jobs: ${url} returned ${res.status} — cannot verify judge`);
  const body = await res.json().catch(() => ({}));
  const tags = (body.models ?? []).map(m => m.name);
  const matches = t => t === JUDGE || t.startsWith(`${JUDGE}-`);
  if (!tags.some(matches)) {
    const list = tags.length ? tags.join(", ") : "(none)";
    throw new Error(
      `bench-jobs: judge model '${JUDGE}' is not pulled on ${HOST}.\n` +
      `  installed: ${list}\n` +
      `  fix: 'ollama pull ${JUDGE}' (only if you've verified it's safe on this box),\n` +
      `       or pass --judge <installed-model> / set OLLAMA_BENCH_JUDGE=<installed-model>.`
    );
  }
}

function chatTimeoutMs() {
  const override = parseInt(process.env.OLLAMA_BENCH_TIMEOUT_MS ?? "", 10);
  if (Number.isFinite(override) && override > 0) return override;
  return 240_000;
}
function withTimeout(ms) {
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), ms);
  return { signal: ac.signal, cancel: () => clearTimeout(timer) };
}

// ── Contexts ─────────────────────────────────────────────────────────────────
// Synthetic but realistic. Numbers are specific so deterministic checks can
// catch hallucinated values; structure is varied so format-matchers don't pass.

const CTX_TRADING_TECH_DAY = `
Market session notes — 2026-04-22 close.
Tape was choppy into the bell. SPX +0.12% to 5841.20, NDX +0.08%, RUT -0.34%.
NVDA reported AMC: revenue $46.1B vs $43.8B est, data-center +28% YoY, gross
margin 75.1%. Guide $48–49B vs $46.2B consensus. Stock +6.4% afterhours to
$152.30 from $143.10 close. Key technical: $145 cleared the prior April high,
next resistance the round $160; support cluster $138.10 (50dma) then $132.
AAPL closed $221.80 -0.4% on cautious services commentary from a sell-side note.
Levels: support $218 (the post-earnings gap), resistance $230. Watching the
$220 strike for max-pain dynamics into May expiry.
TSLA pinned at $258 ahead of next week's deliveries print. Whisper number is
462k units vs 451k consensus. A miss takes us back to $238 fast.
MSFT quietly notched a new ATH at $478.40 on enterprise AI commentary;
nothing actionable yet but the breakout from $462 looks clean.
Macro: 10Y at 4.18%, dollar flat, oil $78.20. Fed minutes Wednesday — no
expectation of a cut but watch the dot-plot revisions.
Risk flags: semis are extended after a 3-week run; reversal in NVDA tomorrow
would drag the whole complex. Watch SOXX. Earnings density next week is the
peak (META, GOOGL, AMZN all Wed-Thu) — position sizing should account for
back-to-back gap risk. Crypto unwound: BTC -2.1% to $61,400, no spillover
to risk-on equity, decoupling continues.
`;

const CTX_TRADING_MACRO = `
Macro recap — 2026-05-09. CPI print came in hot: headline +0.4% MoM, +3.2%
YoY vs 2.9% expected. Core +0.3% MoM, +3.6% YoY vs 3.3% expected.
Bonds sold off hard: 10Y up 14bp to 4.41%, 2Y up 19bp to 4.78%, curve
inverted further. SPX -1.8% intraday low, closed -1.1% at 5712. NDX -1.6%.
DXY +0.7% to 105.4, broad-based USD strength.
Rate-sensitive areas hit hardest: XLU -2.4%, XLRE -3.1%, regional banks
(KRE) -2.8%. Mega-cap tech relatively resilient: AAPL -0.6%, MSFT -0.4%.
NVDA -2.1% on profit-taking, not flow-driven.
Fed-funds futures now price 1.5 cuts in 2026 vs 2.8 yesterday. Powell speaks
Thursday — risk is hawkish guidance reinforcing terminal-rate higher-for-longer.
TLT broke 90.50 support, next stop 87. Gold +1.1% to $2412, safe-haven bid.
Watch tomorrow: PPI 8:30, U Mich inflation expectations Friday. A second
hot inflation read takes 10Y to 4.55–4.60 quickly. Risk-off setup favors
defensive overweights — XLP, XLV — and dollar longs vs. AUD/NZD.
Risk flags: positioning is still long-tilted into this; CTAs may flip
short if SPX takes out 5680. Vol regime change watch — VIX 18.4, term
structure flattening.
`;

const CTX_TRADING_SINGLE = `
NVDA single-name update — 2026-05-11.
Stock $148.20, down from $155.80 last week. Selloff catalysts: (1) Reuters
report that one large hyperscaler is delaying Blackwell rack deployments by a
quarter, citing power-delivery issues at two data center sites; (2) China
export-license commentary from Commerce hinting at tightened H20 controls;
(3) sector-wide derisking ahead of META/GOOGL prints.
Technical picture: $148 is the 50dma. Lose $145 and the next support is
$138 (post-earnings gap fill from April) then $132 (the 100dma).
Resistance overhead: $152, $158, then $165.
Options: 30d IV at 47, up from 38 a week ago. Skew steepened — downside
puts bid. Largest open interest cluster is the $145 strike for May 17 expiry.
Catalysts: GTC keynote 2026-05-20 (Jensen, expect Rubin teaser), then earnings
2026-05-29 AMC. Whisper Q1 revenue $47.5B vs $47.0B consensus.
Risk flags: the hyperscaler delay is a real demand-side data point, not just
noise — watch SMCI/CRDO/AVGO for confirmation. China headline risk is binary;
size accordingly. The $145 strike pin into May expiry could create a mechanical
floor short-term but doesn't solve the demand question.
`;

const CTX_DOC_RESEARCH = `
RESEARCH NOTE — Energy Storage Build-out, Q2 2026

EXECUTIVE OVERVIEW
The grid-scale battery storage market accelerated meaningfully in Q1 2026.
Global deployments reached 18.4 GWh, +47% YoY and +12% QoQ. The US led at
6.2 GWh (+38% YoY), driven by Texas (1.9 GWh) and California (1.4 GWh).
China deployed 7.8 GWh (+62% YoY) primarily co-located with new solar PV.
Europe lagged at 1.9 GWh, hampered by interconnection delays in Germany
and the UK.

KEY DRIVERS
1. LFP cell prices fell to $58/kWh at the system level, down from $79 a
year ago. CATL guided to $52 by year-end. The price floor most analysts
forecasted for 2027 has been pulled forward 18 months.
2. Tax credit clarity from the IRS Q4 2025 guidance unlocked $4.8B in
deferred US project financing. Stranded projects from 2024 are clearing.
3. PJM and ERCOT interconnection reform began bearing fruit; ERCOT's
queue cleared 14 GW of storage in Q1, up from 4 GW in Q1 2025.
4. Battery cycle counts for new LFP chemistries are validating at 8,000+
cycles in third-party testing (Sandia, NREL), de-risking 20-year contracts.

COMPETITIVE DYNAMICS
Tesla Megapack pricing remains 18–22% above CATL-based competitors
(Fluence, Wartsila/CATL). Tesla's market share by GWh fell from 18% to
13% YoY. Sungrow and BYD took share, particularly in EMEA.
The integrator gross-margin compression is real — Fluence guided to
7–9% GM for 2026 vs 12% in 2024. Software-and-services attach rates are
the only differentiator left.

RISKS
- Tariff regime: Section 301 review on Chinese cells could lift landed
costs 15–25%, distorting US deployments. Decision expected Q3 2026.
- Cell oversupply: LFP global capacity reached 1.8 TWh nameplate vs 850
GWh demand. Utilization sub-50%. Pricing risk to the downside continues.
- Fire safety: two large-scale fires in Q1 (one Australia, one Arizona)
have insurers re-pricing project policies +30–50%.

OUTLOOK
Base case: full-year 2026 global deployments 92 GWh (+38% YoY). Bull
case 108 GWh if tariff outcome is benign. Bear case 78 GWh if Section
301 raises duties and a third major fire event prompts permitting delay.

INVESTMENT IMPLICATIONS
Prefer integrators with software attach (Fluence FY26 SaaS rev guide
$420M, +85% YoY) over hardware-only commodity exposure. Cell suppliers
remain a no-touch on margin compression. Utilities deploying behind-the-
meter (PCG, EIX) gain from arbitrage spreads that widened to $32/MWh in
Q1 vs $24 in 2024.
`;

const CTX_DOC_PRODUCT = `
PRODUCT POSTMORTEM — Project Lattice (internal)
Status: Closed, archived 2026-04-30

CONTEXT
Project Lattice was the cross-team initiative to migrate our legacy
session-state service from the v3 redis-backed implementation to a v4
sharded postgres backend. Kicked off 2025-09-15, targeted GA 2026-02-01,
actually shipped to GA 2026-04-12 — 70 days late.

WHAT WE SET OUT TO DO
v3 was hitting two ceilings:
1. Redis cluster at 88 nodes was near the management complexity limit;
ops on-call paged ~14x/week in summer 2025.
2. Session TTL semantics couldn't express the new auth model (sliding
windows with per-tenant policy), and the proxy hacks were brittle.
The v4 design used Citus on Postgres 16, sharded by tenant_id, with a
write-through cache layer (KeyDB) for the hot path.

WHAT WENT WELL
- KeyDB stand-in for the hot path: p99 latency on reads stayed flat at
2.1ms through the cutover. Customers noticed nothing.
- Migration tooling: the per-tenant cutover gate let us roll forward and
back independently. We used the rollback exactly once (tenant cluster-7,
a schema-mismatch bug) and lost 0 sessions.
- Observability: prometheus rules + per-shard dashboards meant the
on-call engineer caught the cluster-7 incident in 8 minutes.

WHAT WENT WRONG
- We underestimated the cost of dual-writes during the migration. The
cross-region replication lag spiked from <100ms to 1.4s for ~3 weeks
in November, breaching our SLO. Caused six P2 customer incidents.
- The schema for session-extensions had three rounds of revision after
GA-blocker bugs from the auth team's downstream consumers. Each cost
about 2 weeks. Better cross-team review at the design stage would have
caught all three.
- Hiring: two of the four planned engineers didn't backfill until
January. We tried to ship at half-velocity rather than rescope. We
should have rescoped — we shipped Q4 features as Q1.

NUMBERS
- Effort: 7 engineer-quarters consumed vs. 5 planned (+40%).
- Outage minutes attributable to Lattice: 47 minutes across 4 customer-
visible incidents. SLO budget consumed: 38% of annual.
- Cost: $1.18M actual vs $760K planned ($420K overage).
- Customer NPS impact: -3 points during the dual-write window, recovered
by GA+30d.

LESSONS / ACTION ITEMS
1. Design-review checklist must include downstream consumers — owned by
engineering excellence, due 2026-06-01.
2. Hiring contingency: any migration project with <4 ICs locked in by
project start must include a rescope decision point at -50% staffing.
3. Dual-write windows over 5 days require an explicit SLO budget
allocation in the project plan.
4. Postmortem the postmortem: this writeup is 90 days after GA; we
should aim for 30 days while context is fresh.
`;

// ── Cases ────────────────────────────────────────────────────────────────────
//
// Each case:
//   { id, job, prompt, [context], deterministic: (response) => 0..1,
//     judgePrompt: (response) => string, weight?: { det, judge } }
//
// Default weight is 0.5/0.5. deterministic returns the fraction of checks
// that pass. judgePrompt builds the rubric string the judge model evaluates.

function tryJsonParse(s) {
  if (!s) return null;
  // Strip common fenced-code wrappers a model might add.
  const fenced = s.match(/```(?:json)?\s*([\s\S]*?)```/);
  const body = (fenced ? fenced[1] : s).trim();
  try { return JSON.parse(body); }
  catch {
    // last-ditch: largest {...} block
    const m = body.match(/\{[\s\S]*\}/);
    if (!m) return null;
    try { return JSON.parse(m[0]); } catch { return null; }
  }
}

function hasKeys(obj, keys) {
  if (!obj || typeof obj !== "object") return 0;
  let hit = 0;
  for (const k of keys) if (k in obj) hit++;
  return hit / keys.length;
}

function num(v) {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "string") {
    const m = v.match(/-?\d+(\.\d+)?/);
    return m ? parseFloat(m[0]) : NaN;
  }
  return NaN;
}

function near(actual, expected, tolPct = 10) {
  const a = num(actual);
  if (!Number.isFinite(a)) return 0;
  if (expected === 0) return a === 0 ? 1 : 0;
  const pct = Math.abs(a - expected) / Math.abs(expected) * 100;
  return pct <= tolPct ? 1 : 0;
}

// ── Trading brief ────────────────────────────────────────────────────────────
const TRADING_SCHEMA_KEYS = ["date", "watchlist", "market_summary", "top_risks"];

function tradingDet(response, expectedTickers) {
  const j = tryJsonParse(response);
  if (!j) return 0;
  let score = 0; let checks = 0;
  // schema presence
  checks++; score += hasKeys(j, TRADING_SCHEMA_KEYS);
  // watchlist is array with the right tickers
  checks++;
  if (Array.isArray(j.watchlist) && j.watchlist.length > 0) {
    const got = new Set(j.watchlist.map(w => (w?.symbol ?? "").toUpperCase()));
    const wantHit = expectedTickers.filter(t => got.has(t)).length;
    score += wantHit / expectedTickers.length;
  }
  // top_risks is non-empty array
  checks++;
  score += Array.isArray(j.top_risks) && j.top_risks.length >= 1 ? 1 : 0;
  // watchlist entries have key_levels with numeric values
  checks++;
  if (Array.isArray(j.watchlist) && j.watchlist.length > 0) {
    const numericLevels = j.watchlist.filter(w =>
      w?.key_levels && (Number.isFinite(num(w.key_levels.support)) || Number.isFinite(num(w.key_levels.resistance)))
    ).length;
    score += numericLevels / j.watchlist.length;
  }
  return score / checks;
}

// ── X analysis ───────────────────────────────────────────────────────────────
const STANCES = ["bullish", "bearish", "neutral", "sarcastic"];

function xDet(response, gold) {
  const j = tryJsonParse(response);
  if (!j) return 0;
  let score = 0; let checks = 0;
  checks++; score += hasKeys(j, ["stance", "inferred_target"]);
  checks++; score += STANCES.includes(String(j.stance).toLowerCase()) ? 1 : 0;
  checks++; score += String(j.stance).toLowerCase() === gold.stance ? 1 : 0;
  checks++; score += String(j.inferred_target ?? "").toUpperCase().includes(gold.target) ? 1 : 0;
  return score / checks;
}

// ── Document prep ───────────────────────────────────────────────────────────
const DOC_SCHEMA_KEYS = ["tl_dr", "key_findings", "data_points", "action_items"];

function docDet(response, expectedMetrics) {
  const j = tryJsonParse(response);
  if (!j) return 0;
  let score = 0; let checks = 0;
  checks++; score += hasKeys(j, DOC_SCHEMA_KEYS);
  checks++;
  // tl_dr length within ~60 words (generous on the ≤40 ask)
  const tl = String(j.tl_dr ?? "");
  const words = tl.split(/\s+/).filter(Boolean).length;
  score += words > 0 && words <= 60 ? 1 : 0;
  checks++;
  score += Array.isArray(j.key_findings) && j.key_findings.length >= 3 && j.key_findings.length <= 7 ? 1 : 0;
  checks++;
  // data_points should mention at least 2/3 of the expected metric tokens
  if (Array.isArray(j.data_points) && j.data_points.length > 0) {
    const blob = JSON.stringify(j.data_points).toLowerCase();
    const hits = expectedMetrics.filter(m => blob.includes(m.toLowerCase())).length;
    score += hits / expectedMetrics.length;
  }
  return score / checks;
}

// ── Hard toolcall ───────────────────────────────────────────────────────────
function hardToolDet(toolCalls, content, expect) {
  // expect: { call: bool, name?: string, altNames?: string[], requireArgs?: string[], allowClarify?: bool }
  const got = toolCalls.length > 0;
  if (!expect.call) {
    if (!got) return 1; // correct refusal
    if (expect.allowClarify && content && content.trim().length >= 20) return 0.5; // partial — asked, but also called
    return 0;
  }
  if (!got) {
    return expect.allowClarify ? 0.5 : 0;
  }
  const first = toolCalls[0];
  const calledName = first.function?.name;
  const accepted = [expect.name, ...(expect.altNames ?? [])];
  if (!accepted.includes(calledName)) return 0;
  let score = 0.6; // right tool picked
  if (expect.requireArgs && expect.requireArgs.length) {
    let parsed = {};
    try { parsed = JSON.parse(first.function?.arguments ?? "{}"); } catch {}
    const hit = expect.requireArgs.filter(k => k in parsed && parsed[k] !== "" && parsed[k] != null).length;
    score += 0.4 * (hit / expect.requireArgs.length);
  } else {
    score += 0.4;
  }
  return score;
}

// ── Reasoning ───────────────────────────────────────────────────────────────
function reasoningDet(response, expected) {
  const j = tryJsonParse(response);
  if (!j) return 0;
  let score = 0; let checks = 0;
  checks++; score += hasKeys(j, Object.keys(expected));
  for (const [k, v] of Object.entries(expected)) {
    checks++;
    score += near(j[k], v, 10);
  }
  return score / checks;
}

// ── Cases ────────────────────────────────────────────────────────────────────
const CASES = [
  // ── trading_brief ──────────────────────────────────────────────────────────
  {
    id: "trading_brief::tech_day", job: "trading_brief",
    context: CTX_TRADING_TECH_DAY,
    prompt:
`You are preparing a structured trading brief for tomorrow's session. Using ONLY the market notes below, output a brief as JSON with this exact shape:
{"date": "<YYYY-MM-DD>", "watchlist": [{"symbol": "...", "thesis": "...", "key_levels": {"support": <num>, "resistance": <num>}, "risk_flags": ["..."]}], "market_summary": "...", "top_risks": ["..."]}

Include 3–5 watchlist names that are explicitly discussed. Use real numeric levels from the notes — do not invent. Reply with JSON only.

NOTES:
${CTX_TRADING_TECH_DAY}`,
    deterministic: r => tradingDet(r, ["NVDA", "AAPL", "TSLA", "MSFT"]),
    judgePrompt: r => buildJudgePrompt({
      job: "trading_brief",
      task: "Produce a faithful structured trading brief from the supplied market notes.",
      contextExcerpt: CTX_TRADING_TECH_DAY,
      response: r,
      rubric:
`0 — invalid JSON, refusal, or wholly invented content.
1 — JSON-shaped but contains tickers/levels not present in the notes.
2 — faithful tickers and levels, captures the main narrative.
3 — faithful + identifies the real cross-asset / cross-name signal (e.g. semis-extension risk, earnings density next week).`,
    }),
  },
  {
    id: "trading_brief::macro_shock", job: "trading_brief",
    context: CTX_TRADING_MACRO,
    prompt:
`Produce a structured trading brief from the macro session notes below as JSON:
{"date": "<YYYY-MM-DD>", "watchlist": [{"symbol": "...", "thesis": "...", "key_levels": {"support": <num>, "resistance": <num>}, "risk_flags": ["..."]}], "market_summary": "...", "top_risks": ["..."]}

Watchlist should be 3–5 names or sector ETFs explicitly mentioned. Use real numeric levels. JSON only.

NOTES:
${CTX_TRADING_MACRO}`,
    deterministic: r => tradingDet(r, ["TLT", "XLU", "XLP"]),
    judgePrompt: r => buildJudgePrompt({
      job: "trading_brief",
      task: "Produce a faithful macro-driven trading brief.",
      contextExcerpt: CTX_TRADING_MACRO,
      response: r,
      rubric:
`0 — invalid JSON or invented content (e.g. wrong CPI prints, fabricated tickers).
1 — structurally fine but misses the macro setup (rates, USD, defensive rotation).
2 — captures rates move + sector rotation + risk-off setup faithfully.
3 — also captures positioning / CTA flip risk / Powell event risk.`,
    }),
  },
  {
    id: "trading_brief::single_name", job: "trading_brief",
    context: CTX_TRADING_SINGLE,
    prompt:
`Produce a single-name trading brief from the notes below as JSON:
{"date": "<YYYY-MM-DD>", "watchlist": [{"symbol": "NVDA", "thesis": "...", "key_levels": {"support": <num>, "resistance": <num>}, "risk_flags": ["..."]}], "market_summary": "...", "top_risks": ["..."]}

Use real levels and catalysts from the notes. JSON only.

NOTES:
${CTX_TRADING_SINGLE}`,
    deterministic: r => tradingDet(r, ["NVDA"]),
    judgePrompt: r => buildJudgePrompt({
      job: "trading_brief",
      task: "Produce a faithful single-name trading brief.",
      contextExcerpt: CTX_TRADING_SINGLE,
      response: r,
      rubric:
`0 — fabricated levels or catalysts.
1 — right ticker, vague thesis.
2 — captures levels + main catalysts (hyperscaler delay, China headline, GTC/earnings).
3 — also captures the demand-side read (SMCI/CRDO/AVGO confirmation tells, options pinning).`,
    }),
  },

  // ── x_analysis ─────────────────────────────────────────────────────────────
  {
    id: "x_analysis::sarcasm_nvda", job: "x_analysis",
    prompt:
`Classify this X post. Return JSON:
{"stance": "bullish"|"bearish"|"neutral"|"sarcastic", "confidence": 0-1, "claims": ["..."], "inferred_target": "<TICKER or topic>"}

POST: "Oh sure, NVDA totally needs another upgrade from the sell-side after a 4-week run. That's the catalyst we've been waiting for. 🙄"`,
    gold: { stance: "sarcastic", target: "NVDA" },
    deterministic: r => xDet(r, { stance: "sarcastic", target: "NVDA" }),
    judgePrompt: r => buildJudgePrompt({
      job: "x_analysis",
      task: "Detect sarcasm; identify NVDA as target.",
      response: r,
      rubric:
`0 — flagged as bullish/bearish at face value (missed irony).
1 — got target right but stance wrong.
2 — both right.
3 — both right and extracted the underlying claim (sell-side upgrade isn't a real catalyst after an extended run).`,
    }),
  },
  {
    id: "x_analysis::quote_inversion", job: "x_analysis",
    prompt:
`Classify this X post. Return JSON:
{"stance": "bullish"|"bearish"|"neutral"|"sarcastic", "confidence": 0-1, "claims": ["..."], "inferred_target": "<TICKER or topic>"}

POST: "Quote-tweeting 'TSLA delivery numbers crushed it!!!' to say — they crushed estimates that were revised down twice in the last month. Read the room."`,
    gold: { stance: "bearish", target: "TSLA" },
    deterministic: r => xDet(r, { stance: "bearish", target: "TSLA" }),
    judgePrompt: r => buildJudgePrompt({
      job: "x_analysis",
      task: "Identify TSLA as target; recognize bearish-via-quote-inversion stance.",
      response: r,
      rubric:
`0 — read the surface quote as bullish.
1 — neutral or unclear.
2 — correctly bearish on TSLA.
3 — also captures the 'estimates revised down' mechanism as the claim.`,
    }),
  },
  {
    id: "x_analysis::straight_bullish", job: "x_analysis",
    prompt:
`Classify this X post. Return JSON:
{"stance": "bullish"|"bearish"|"neutral"|"sarcastic", "confidence": 0-1, "claims": ["..."], "inferred_target": "<TICKER or topic>"}

POST: "AVGO setup looks clean here — $1820 holds, three-day base, AI capex narrative intact. Targets $1920 / $1980."`,
    gold: { stance: "bullish", target: "AVGO" },
    deterministic: r => xDet(r, { stance: "bullish", target: "AVGO" }),
    judgePrompt: r => buildJudgePrompt({
      job: "x_analysis",
      task: "Direct bullish call on AVGO.",
      response: r,
      rubric:
`0 — misread.
1 — partial.
2 — correct.
3 — correct + extracts the levels as claims.`,
    }),
  },
  {
    id: "x_analysis::neutral_observation", job: "x_analysis",
    prompt:
`Classify this X post. Return JSON:
{"stance": "bullish"|"bearish"|"neutral"|"sarcastic", "confidence": 0-1, "claims": ["..."], "inferred_target": "<TICKER or topic>"}

POST: "FOMC minutes drop Wednesday at 2pm. Last set was meaningfully more divided than the statement suggested."`,
    gold: { stance: "neutral", target: "FOMC" },
    deterministic: r => xDet(r, { stance: "neutral", target: "FOMC" }),
    judgePrompt: r => buildJudgePrompt({
      job: "x_analysis",
      task: "Neutral / informational post about FOMC minutes.",
      response: r,
      rubric:
`0 — forced a directional read.
1 — neutral but missed target.
2 — neutral + FOMC target.
3 — also flags the implied claim (more dissent than the statement showed).`,
    }),
  },
  {
    id: "x_analysis::backhanded_compliment", job: "x_analysis",
    prompt:
`Classify this X post. Return JSON:
{"stance": "bullish"|"bearish"|"neutral"|"sarcastic", "confidence": 0-1, "claims": ["..."], "inferred_target": "<TICKER or topic>"}

POST: "INTC up 4% on… a CEO succession leak. Truly the bull case in 2026."`,
    gold: { stance: "sarcastic", target: "INTC" },
    deterministic: r => xDet(r, { stance: "sarcastic", target: "INTC" }),
    judgePrompt: r => buildJudgePrompt({
      job: "x_analysis",
      task: "Sarcastic dismissal of INTC rally.",
      response: r,
      rubric:
`0 — read as bullish at face value.
1 — got target.
2 — sarcastic + INTC.
3 — also extracts the claim (CEO succession leak is a weak catalyst).`,
    }),
  },

  // ── document_prep ──────────────────────────────────────────────────────────
  {
    id: "document_prep::energy_research", job: "document_prep",
    context: CTX_DOC_RESEARCH,
    prompt:
`Distill the research note below into a structured executive summary as JSON:
{"tl_dr": "<≤40 words>", "key_findings": ["..."], "data_points": [{"metric": "...", "value": "...", "source_section": "..."}], "action_items": ["..."]}

Use real numbers from the note. JSON only.

NOTE:
${CTX_DOC_RESEARCH}`,
    deterministic: r => docDet(r, ["18.4", "47%", "58", "92 GWh"]),
    judgePrompt: r => buildJudgePrompt({
      job: "document_prep",
      task: "Distill the energy-storage research note.",
      contextExcerpt: CTX_DOC_RESEARCH,
      response: r,
      rubric:
`0 — hallucinated content.
1 — surface bullets without numbers.
2 — captures the main drivers + scenario range + a real risk.
3 — also synthesizes the integrator vs. cell-supplier preference and the tariff binary.`,
    }),
  },
  {
    id: "document_prep::product_postmortem", job: "document_prep",
    context: CTX_DOC_PRODUCT,
    prompt:
`Distill the postmortem below into a structured executive summary as JSON:
{"tl_dr": "<≤40 words>", "key_findings": ["..."], "data_points": [{"metric": "...", "value": "...", "source_section": "..."}], "action_items": ["..."]}

Use real numbers from the note. JSON only.

NOTE:
${CTX_DOC_PRODUCT}`,
    deterministic: r => docDet(r, ["70 days", "47 minutes", "1.18M", "+40%"]),
    judgePrompt: r => buildJudgePrompt({
      job: "document_prep",
      task: "Distill the Project Lattice postmortem.",
      contextExcerpt: CTX_DOC_PRODUCT,
      response: r,
      rubric:
`0 — hallucinated or generic.
1 — wins and losses but no numbers.
2 — captures the four numeric facts (effort overrun, outage minutes, cost, NPS).
3 — also surfaces the underlying lesson (hiring contingency + design review with downstream consumers).`,
    }),
  },

  // ── hard_toolcall ──────────────────────────────────────────────────────────
  {
    id: "hard_toolcall::implicit_sender", job: "hard_toolcall",
    prompt: "Did Bob ever send me that deck I asked him for yesterday?",
    expect: { call: true, name: "email_inbox", altNames: ["web_search"] },
    deterministic: (response, calls, content) => hardToolDet(calls, content, { call: true, name: "email_inbox", altNames: ["web_search"] }),
    judgePrompt: r => buildJudgePrompt({
      job: "hard_toolcall",
      task: "Resolve 'Did Bob send the deck' — implicit sender filter, no perfect tool.",
      response: r,
      rubric:
`0 — wrong tool or no call without acknowledging the limitation.
1 — picked email_inbox but no sender-handling note.
2 — picked email_inbox and acknowledged the sender filter limitation in content.
3 — picked email_inbox, acknowledged limitation, asked for clarification on Bob's address.`,
    }),
  },
  {
    id: "hard_toolcall::missing_tool", job: "hard_toolcall",
    prompt: "Push the 3pm standup tomorrow to Friday at the same time.",
    expect: { call: false, allowClarify: true },
    deterministic: (response, calls, content) => hardToolDet(calls, content, { call: false, allowClarify: true }),
    judgePrompt: r => buildJudgePrompt({
      job: "hard_toolcall",
      task: "User asks to reschedule a calendar event — no calendar_update tool exists.",
      response: r,
      rubric:
`0 — fabricated a tool call or silently failed.
1 — declined without explanation.
2 — declined and explained no rescheduling tool is available.
3 — declined, explained, and offered an alternative (create a task to do it manually, or asked to confirm cancel-and-recreate).`,
    }),
  },
  {
    id: "hard_toolcall::compound", job: "hard_toolcall",
    prompt: "Get me the AAPL quote, and if it's down more than 2% today, add a task to review the position.",
    expect: { call: true, name: "quote", requireArgs: ["symbol"] },
    deterministic: (response, calls, content) => hardToolDet(calls, content, { call: true, name: "quote", requireArgs: ["symbol"] }),
    judgePrompt: r => buildJudgePrompt({
      job: "hard_toolcall",
      task: "Conditional tool chain — fetch quote first, defer the conditional task.",
      response: r,
      rubric:
`0 — wrong tool or skipped the quote.
1 — called quote.
2 — called quote and acknowledged the conditional step is deferred until the result is back.
3 — also articulated the threshold (>2% drop) in content.`,
    }),
  },
  {
    id: "hard_toolcall::ambiguous_search", job: "hard_toolcall",
    prompt: "Find that thing about the EU regulation I was reading last week.",
    expect: { call: true, name: "web_search", altNames: ["email_inbox"], requireArgs: ["query"] },
    deterministic: (response, calls, content) => hardToolDet(calls, content, { call: true, name: "web_search", altNames: ["email_inbox"], requireArgs: ["query"] }),
    judgePrompt: r => buildJudgePrompt({
      job: "hard_toolcall",
      task: "Vague 'that thing about EU regulation' — needs a search, but the query is underspecified.",
      response: r,
      rubric:
`0 — refused or wrong tool.
1 — called web_search with a generic query.
2 — called web_search with a sensible inferred query (EU regulation).
3 — also asked for narrowing (which sector? AI Act? GDPR?).`,
    }),
  },
  {
    id: "hard_toolcall::price_or_news", job: "hard_toolcall",
    prompt: "Why is NVDA down so much today?",
    expect: { call: true, name: "web_search", altNames: ["quote"], requireArgs: ["query"] },
    deterministic: (response, calls, content) => hardToolDet(calls, content, { call: true, name: "web_search", altNames: ["quote"], requireArgs: ["query"] }),
    judgePrompt: r => buildJudgePrompt({
      job: "hard_toolcall",
      task: "User wants the reason — web_search is the right tool, quote alone doesn't explain.",
      response: r,
      rubric:
`0 — called nothing or fabricated a reason.
1 — only called quote.
2 — called web_search with a reasonable query.
3 — chained quote + web_search OR explained that web_search is needed for the 'why'.`,
    }),
  },

  // ── reasoning ──────────────────────────────────────────────────────────────
  {
    id: "reasoning::position_size", job: "reasoning",
    prompt:
`Position sizing problem. Account size $50,000. Risk per trade: 2%. NVDA at $145.30. Stop at $138.10.
Compute: shares to buy, dollar risk, stop distance per share.
Show your reasoning, then output the final answer as JSON:
{"shares": <int>, "dollar_risk": <float>, "stop_distance": <float>}
The shares must round DOWN to a whole number.`,
    expected: { shares: 138, dollar_risk: 1000, stop_distance: 7.20 },
    deterministic: r => reasoningDet(r, { shares: 138, dollar_risk: 1000, stop_distance: 7.20 }),
    judgePrompt: r => buildJudgePrompt({
      job: "reasoning",
      task: "Position sizing: 2% of $50k = $1000 risk; stop distance $7.20; shares = floor(1000/7.20) = 138.",
      response: r,
      rubric:
`0 — wrong arithmetic or wrong rounding direction.
1 — right numbers but no reasoning shown.
2 — right numbers + clear reasoning.
3 — also flags the rounding convention or fractional-shares note.`,
    }),
  },
  {
    id: "reasoning::expectancy", job: "reasoning",
    prompt:
`A trading strategy has: 45% win rate, average winner +$320, average loser -$180. Over 200 trades, compute:
- expected total P&L
- expectancy per trade
Output JSON: {"expected_pnl": <float>, "expectancy_per_trade": <float>}`,
    expected: { expected_pnl: 9000, expectancy_per_trade: 45 },
    deterministic: r => reasoningDet(r, { expected_pnl: 9000, expectancy_per_trade: 45 }),
    judgePrompt: r => buildJudgePrompt({
      job: "reasoning",
      task: "Expectancy = 0.45*320 + 0.55*(-180) = 144 - 99 = 45. Total over 200 = $9000.",
      response: r,
      rubric:
`0 — wrong.
1 — one of two right.
2 — both right.
3 — both right + clear formulation.`,
    }),
  },
  {
    id: "reasoning::dca", job: "reasoning",
    prompt:
`Dollar-cost averaging: I invested $500/month for 6 months at these prices: 10, 12, 15, 11, 9, 13. Compute:
- total shares purchased (allow fractional)
- average cost per share
- final value at current price $14
Output JSON: {"total_shares": <float>, "avg_cost": <float>, "final_value": <float>}`,
    // shares = 50 + 41.667 + 33.333 + 45.455 + 55.556 + 38.462 = 264.473
    // avg_cost = 3000 / 264.473 = 11.343
    // final_value = 264.473 * 14 = 3702.62
    expected: { total_shares: 264.47, avg_cost: 11.34, final_value: 3702.62 },
    deterministic: r => reasoningDet(r, { total_shares: 264.47, avg_cost: 11.34, final_value: 3702.62 }),
    judgePrompt: r => buildJudgePrompt({
      job: "reasoning",
      task: "DCA: sum of (500/price) across months; avg_cost = 3000/total_shares; final_value = total_shares × 14.",
      response: r,
      rubric:
`0 — wrong setup.
1 — partial.
2 — all three within tolerance.
3 — also flags that DCA avg_cost < arithmetic mean of prices (harmonic effect).`,
    }),
  },
  {
    id: "reasoning::breakeven", job: "reasoning",
    prompt:
`Options breakeven: I bought 5 calls on TSLA strike $260 for $4.80 premium each (per-share). Contract multiplier is 100.
- total cost
- breakeven price
- max loss
Output JSON: {"total_cost": <float>, "breakeven": <float>, "max_loss": <float>}`,
    expected: { total_cost: 2400, breakeven: 264.80, max_loss: 2400 },
    deterministic: r => reasoningDet(r, { total_cost: 2400, breakeven: 264.80, max_loss: 2400 }),
    judgePrompt: r => buildJudgePrompt({
      job: "reasoning",
      task: "Calls: total_cost = 5 × $4.80 × 100 = $2400; breakeven = strike + premium = $264.80; max_loss = total_cost.",
      response: r,
      rubric:
`0 — wrong.
1 — partial.
2 — all three correct.
3 — correct + notes max_loss = total_cost because long calls.`,
    }),
  },
  {
    id: "reasoning::r_multiple", job: "reasoning",
    prompt:
`R-multiple: I entered AAPL at $221.80, stop at $218.40, target $232.00.
- R-distance (per share)
- R-multiple of the target
- if I size for 1R = $250 of risk, how many shares?
Output JSON: {"r_distance": <float>, "target_r": <float>, "shares": <int>}`,
    // r_distance = 3.40
    // target_r = (232 - 221.80) / 3.40 = 10.20 / 3.40 = 3.0
    // shares = floor(250 / 3.40) = 73
    expected: { r_distance: 3.40, target_r: 3.0, shares: 73 },
    deterministic: r => reasoningDet(r, { r_distance: 3.40, target_r: 3.0, shares: 73 }),
    judgePrompt: r => buildJudgePrompt({
      job: "reasoning",
      task: "R = 3.40; target = 3R; shares = floor(250/3.40) = 73.",
      response: r,
      rubric:
`0 — wrong arithmetic.
1 — partial.
2 — all three correct.
3 — correct + notes rounding direction.`,
    }),
  },
];

// ── Judge prompt builder ─────────────────────────────────────────────────────
function buildJudgePrompt({ job, task, contextExcerpt, response, rubric }) {
  // Trim context to keep judge input under control. The judge needs enough to
  // verify faithfulness but not the full 8k blob.
  const ctx = contextExcerpt ? `\nREFERENCE CONTEXT (excerpt):\n${contextExcerpt.slice(0, 2500)}\n` : "";
  return `You are scoring a model's response to a "${job}" task on a 0–3 rubric.

TASK: ${task}
${ctx}
MODEL RESPONSE:
${String(response).slice(0, 4000)}

RUBRIC:
${rubric}

Reply with ONLY a JSON object — no prose, no fenced code. Schema:
{"score": <int 0-3>, "reason": "<≤25 words explaining the score>"}`;
}

// ── HTTP runner ─────────────────────────────────────────────────────────────
async function generate(model, prompt, { tools, timeoutMs } = {}) {
  const ms = timeoutMs ?? chatTimeoutMs();
  const t = withTimeout(ms);
  const body = { model, messages: [{ role: "user", content: prompt }], temperature: 0 };
  if (tools) body.tools = tools;
  const endpoint = "/v1/chat/completions";
  let res;
  try {
    res = await fetch(`${HOST}${endpoint}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      signal: t.signal,
    });
  } catch (e) {
    if (e.name === "AbortError") throw new Error(`generate timed out after ${ms}ms — check ${HOST} or set OLLAMA_BENCH_TIMEOUT_MS`);
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
  // OpenAI-compat fields available for token counts on Ollama: usage.completion_tokens / prompt_tokens
  return {
    content: msg.content ?? "",
    tool_calls: msg.tool_calls ?? [],
    promptTokens: j.usage?.prompt_tokens ?? 0,
    completionTokens: j.usage?.completion_tokens ?? 0,
  };
}

async function runCandidate(c) {
  const wantsTools = c.job === "hard_toolcall";
  return await generate(MODEL, c.prompt, { tools: wantsTools ? TOOLS : null });
}

// For tool-using cases the model often returns empty content + a tool_calls
// array. The judge needs to see the rendered tool call (name + args) to score
// "did it pick the right tool / handle ambiguity"; raw empty content reads
// as a refusal. For non-tool cases we just pass content.
function renderForJudge(out) {
  const content = out.content ?? "";
  const calls = out.tool_calls ?? [];
  if (calls.length === 0) return content || "(model returned empty response)";
  const rendered = calls.map((c, i) => {
    let args = c.function?.arguments ?? "{}";
    try { args = JSON.stringify(JSON.parse(args)); } catch {}
    return `tool_call[${i}]: ${c.function?.name}(${args})`;
  }).join("\n");
  return content ? `${rendered}\n\nassistant content: ${content}` : rendered;
}

async function runJudge(c, candidateOut) {
  const rendered = renderForJudge(candidateOut);
  const prompt = c.judgePrompt(rendered);
  const out = await generate(JUDGE, prompt);
  const parsed = tryJsonParse(out.content);
  if (!parsed || typeof parsed.score !== "number") {
    return { score: 0, reason: "judge: unparseable response", raw: out.content.slice(0, 120) };
  }
  // Clamp to 0..3
  const s = Math.max(0, Math.min(3, Math.round(parsed.score)));
  return { score: s, reason: String(parsed.reason ?? "").slice(0, 200) };
}

// ── Aggregation ─────────────────────────────────────────────────────────────
function scoreCase(c, candidateOut, judgeOut) {
  // Deterministic score depends on case type
  let det;
  if (c.job === "hard_toolcall") {
    det = c.deterministic(candidateOut.content, candidateOut.tool_calls, candidateOut.content);
  } else {
    det = c.deterministic(candidateOut.content);
  }
  const judgeNorm = judgeOut.score / 3;
  const caseScore = 0.5 * det + 0.5 * judgeNorm;
  return { detScore: det, judgeScore: judgeNorm, caseScore };
}

async function runCases() {
  const t0 = performance.now();
  // ── Pass 1: candidate generations ────────────────────────────────────────
  const cached = [];
  console.log(`pass 1/2: candidate=${MODEL} generating ${CASES.length} responses…`);
  for (let i = 0; i < CASES.length; i++) {
    const c = CASES[i];
    const tCase = performance.now();
    let out;
    try {
      out = await runCandidate(c);
    } catch (e) {
      out = { content: "", tool_calls: [], error: e.message };
    }
    cached.push({ case: c, out, candidateMs: performance.now() - tCase });
    if (VERBOSE) console.log(`  [${i + 1}/${CASES.length}] ${c.id} — ${(performance.now() - tCase).toFixed(0)}ms ${out.error ? "ERR: " + out.error : `(${out.completionTokens || 0} tok)`}`);
  }
  const pass1Ms = performance.now() - t0;

  // ── Pass 2: judge scoring ────────────────────────────────────────────────
  console.log(`pass 2/2: judge=${JUDGE} scoring ${cached.length} responses…`);
  const tJudge0 = performance.now();
  for (const row of cached) {
    const tCase = performance.now();
    try {
      row.judge = await runJudge(row.case, row.out);
    } catch (e) {
      row.judge = { score: 0, reason: `judge error: ${e.message}` };
    }
    row.judgeMs = performance.now() - tCase;
    if (VERBOSE) console.log(`  [${row.case.id}] judge=${row.judge.score}/3 — ${row.judge.reason}`);
  }
  const pass2Ms = performance.now() - tJudge0;

  // ── Aggregate ────────────────────────────────────────────────────────────
  const byJob = {};
  const failedCases = [];
  for (const row of cached) {
    const s = scoreCase(row.case, row.out, row.judge);
    row.scored = s;
    const job = row.case.job;
    if (!byJob[job]) byJob[job] = { total: 0, detSum: 0, judgeSum: 0, caseSum: 0, durationMs: 0, completionTokens: 0, cases: [] };
    const r = byJob[job];
    r.total++;
    r.detSum += s.detScore;
    r.judgeSum += s.judgeScore;
    r.caseSum += s.caseScore;
    r.durationMs += row.candidateMs;
    r.completionTokens += row.out.completionTokens || 0;
    r.cases.push({
      id: row.case.id,
      detScore: +s.detScore.toFixed(3),
      judgeScore: +s.judgeScore.toFixed(3),
      caseScore: +s.caseScore.toFixed(3),
      judgeReason: row.judge.reason,
      candidateMs: Math.round(row.candidateMs),
      completionTokens: row.out.completionTokens || 0,
      error: row.out.error ?? null,
    });
    if (s.caseScore < 0.5) failedCases.push(row.case.id);
  }
  const byJobOut = {};
  for (const [job, r] of Object.entries(byJob)) {
    byJobOut[job] = {
      total: r.total,
      detPct: +(100 * r.detSum / r.total).toFixed(1),
      judgePct: +(100 * r.judgeSum / r.total).toFixed(1),
      score: +(100 * r.caseSum / r.total).toFixed(1),
      durationSec: +(r.durationMs / 1000).toFixed(1),
      wallTokPerSec: r.durationMs > 0 ? +(r.completionTokens / (r.durationMs / 1000)).toFixed(1) : 0,
      cases: r.cases,
    };
  }
  const totalCases = cached.length;
  const overallScore = +(100 * cached.reduce((a, r) => a + r.scored.caseScore, 0) / totalCases).toFixed(1);

  return {
    savedAt: new Date().toISOString(),
    model: MODEL,
    judge: JUDGE,
    total: totalCases,
    overall: { score: overallScore, pass1Sec: +(pass1Ms / 1000).toFixed(1), pass2Sec: +(pass2Ms / 1000).toFixed(1), totalSec: +((pass1Ms + pass2Ms) / 1000).toFixed(1) },
    byJob: byJobOut,
    failedCases,
  };
}

// ── Reporting ────────────────────────────────────────────────────────────────
function pad(s, w, right = true) { return right ? String(s).padStart(w) : String(s).padEnd(w); }

function printReport(current, base) {
  const cols = ["job", "cases", "det%", "judge%", "score", "wall t/s", "sec"];
  const widths = [16, 5, 5, 6, 6, 8, 6];
  console.log("");
  console.log(cols.map((c, i) => i === 0 ? pad(c, widths[i], false) : pad(c, widths[i])).join(" | "));
  console.log("-".repeat(widths.reduce((a, w) => a + w + 3, 0)));

  for (const [job, r] of Object.entries(current.byJob)) {
    const bRow = base?.byJob?.[job];
    const score = `${r.score.toFixed(1)}`;
    const dScore = bRow ? ` (${(r.score - bRow.score >= 0 ? "+" : "")}${(r.score - bRow.score).toFixed(1)})` : "";
    const row = [
      pad(job, widths[0], false),
      pad(r.total, widths[1]),
      pad(`${r.detPct.toFixed(0)}%`, widths[2]),
      pad(`${r.judgePct.toFixed(0)}%`, widths[3]),
      pad(score + dScore, widths[4]),
      pad(r.wallTokPerSec.toFixed(1), widths[5]),
      pad(r.durationSec.toFixed(0), widths[6]),
    ].join(" | ");
    console.log(row);
  }
  console.log("-".repeat(widths.reduce((a, w) => a + w + 3, 0)));
  console.log(pad("OVERALL", widths[0], false) + " | " + pad(current.total, widths[1]) + " | " + pad("", widths[2]) + " | " + pad("", widths[3]) + " | " + pad(current.overall.score.toFixed(1), widths[4]) + " | " + pad("", widths[5]) + " | " + pad(current.overall.totalSec.toFixed(0), widths[6]));

  console.log(`\nwall: pass1 ${current.overall.pass1Sec.toFixed(0)}s + pass2 ${current.overall.pass2Sec.toFixed(0)}s = ${current.overall.totalSec.toFixed(0)}s`);
  console.log(`judge: ${current.judge}`);

  if (current.failedCases.length) {
    console.log(`\nweak cases (score < 0.5): ${current.failedCases.length}`);
    for (const id of current.failedCases) console.log(`  ${id}`);
  }
}

// ── Main ─────────────────────────────────────────────────────────────────────
async function main() {
  console.log(`\nbench-jobs: model=${MODEL} judge=${JUDGE} host=${HOST} cases=${CASES.length}\n`);
  await assertJudgeInstalled();

  const existing = getModelSection(OUT, MODEL, "jobs");
  let mode = MODE;
  if (mode === "smart") {
    if (!existing) {
      mode = "save";
      console.log(`(no jobs entry for ${MODEL} at ${OUT} — saving one now)`);
    } else {
      mode = "compare";
    }
  } else if (mode === "compare" && !existing) {
    console.error(`no jobs entry for ${MODEL} at ${OUT} — run with --save first`);
    process.exit(1);
  }

  const current = await runCases();
  printReport(current, mode === "compare" ? existing : null);

  if (mode === "save") {
    writeModelSection(OUT, MODEL, "jobs", current);
    console.log(`\njobs entry saved for ${MODEL} → ${OUT}`);
  }
}

main().catch(e => { console.error(e); process.exit(1); });
