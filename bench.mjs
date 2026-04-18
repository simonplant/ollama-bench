#!/usr/bin/env node
/**
 * Performance regression harness for local Ollama models.
 *
 * Goal: catch throughput regressions from configuration changes. Not a
 * capability benchmark — every run uses identical prompts so differences
 * come from the environment, not the workload.
 *
 * Dimensions measured (each gets its own row):
 *   1. Single-stream generation throughput at 4 context sizes
 *   2. Prompt-eval throughput at those sizes (fresh prefix per run → no cache)
 *   3. Time-to-first-token (proxy: load_duration on a short warm prompt)
 *   4. Concurrency — 1/2/4/8 parallel streams, aggregate tokens/sec
 *
 * Workflow:
 *   bench-regression.mjs save       → runs and writes baseline.json
 *   bench-regression.mjs compare    → runs and diffs vs baseline.json
 *   bench-regression.mjs run        → runs and prints, no baseline touched
 *
 * Flags:
 *   --model <name>    default gemma4:26b
 *   --host <url>      default http://ollama:11434
 *   --runs <n>        per-case runs, median reported; default 3
 *   --out <path>      baseline file; default ./baseline.json
 *   --regression-pct  threshold to flag (default 5 = 5% slower than baseline)
 */

import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { execSync } from "node:child_process";

const args = process.argv.slice(2);
const cmd  = args[0] ?? "run";
const arg = (n, fb) => { const i = args.indexOf(n); return i >= 0 ? args[i + 1] : fb; };

const MODEL     = arg("--model", "gemma4:26b");
const HOST      = arg("--host",  "http://ollama:11434");
const RUNS      = parseInt(arg("--runs", "3"), 10);
const OUT       = arg("--out",   "./baseline.json");
const REG_PCT   = parseFloat(arg("--regression-pct", "5"));

// ── Fixtures ─────────────────────────────────────────────────────────────────
// Deterministic seed so "fresh prefix per run" is still reproducible across
// different invocations. Same sequence → same workload.
let seedState = 1337;
function seededRand() {
  // xorshift32
  let x = seedState;
  x ^= x << 13; x ^= x >>> 17; x ^= x << 5;
  seedState = x >>> 0;
  return (seedState & 0xffffffff) / 0x100000000;
}
function resetSeed(s) { seedState = s; }

const WORDS = (
  "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima " +
  "mike november oscar papa quebec romeo sierra tango uniform victor whiskey " +
  "xray yankee zulu aurora borealis crescent horizon kernel lattice meridian"
).split(" ");

function filler(approxTokens) {
  const out = [];
  for (let i = 0; i < approxTokens; i++) out.push(WORDS[Math.floor(seededRand() * WORDS.length)]);
  return out.join(" ");
}

const CTX_SIZES = [
  // "tiny" (< 30 tokens) was noise-dominated: variance swung ±60% between
  // identical back-to-back runs because prompt-eval duration approaches the
  // request floor. Dropped from defaults; add back via --include-tiny if
  // you specifically care about short-request overhead.
  { label: "short",  pad: 180 },
  { label: "medium", pad: 1800 },
  { label: "long",   pad: 7800 },
];

// llama.cpp's prompt cache is prefix-sequential — a single different leading
// token invalidates the cache for this prompt. Every call must therefore start
// with a fresh token, otherwise prompt-eval t/s measures cache hits rather
// than real throughput. Keyed by process-PID + cell + run so no two calls
// across any invocations of this script share a prefix.
const PROCESS_NONCE = `${Date.now().toString(36)}-${Math.floor(Math.random() * 1e9).toString(36)}`;

function mkPrompt(ctxIdx, runIdx) {
  const size = CTX_SIZES[ctxIdx];
  // Seed tied to (ctxIdx, runIdx) so the filler content is reproducible for a
  // given cell across invocations — keeps the workload size stable so t/s
  // numbers are comparable.
  resetSeed(1000 + ctxIdx * 100 + runIdx);
  const body = filler(size.pad);
  // Cache-buster first, and distinct per call, so the prompt's first token
  // differs from any other call this process or any previous process made.
  return `[${PROCESS_NONCE}/${ctxIdx}/${runIdx}]\nSummarise in two sentences:\n\n${body}`;
}

// ── Ollama API ───────────────────────────────────────────────────────────────
async function generate(prompt, { numPredict = 128, keepAlive } = {}) {
  const body = {
    model: MODEL,
    prompt,
    stream: false,
    options: { num_predict: numPredict, temperature: 0, seed: 42 },
  };
  if (keepAlive !== undefined) body.keep_alive = keepAlive;
  const res = await fetch(`${HOST}/api/generate`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  const j = await res.json();
  return {
    promptTokens: j.prompt_eval_count ?? 0,
    genTokens:    j.eval_count ?? 0,
    promptTps:    j.prompt_eval_count / (j.prompt_eval_duration / 1e9),
    genTps:       j.eval_count / (j.eval_duration / 1e9),
    totalMs:      j.total_duration / 1e6,
    loadMs:       (j.load_duration ?? 0) / 1e6,
  };
}

// ── Metrics helpers ──────────────────────────────────────────────────────────
function median(xs) {
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

// ── Scenarios ────────────────────────────────────────────────────────────────
async function scenarioSingleStream() {
  const results = [];
  for (let i = 0; i < CTX_SIZES.length; i++) {
    const pTps = [], gTps = [], total = [], load = [];
    for (let r = 0; r < RUNS; r++) {
      const out = await generate(mkPrompt(i, r));
      pTps.push(out.promptTps);
      gTps.push(out.genTps);
      total.push(out.totalMs);
      load.push(out.loadMs);
    }
    results.push({
      ctx: CTX_SIZES[i].label,
      promptTps: median(pTps),
      genTps:    median(gTps),
      totalMs:   median(total),
      loadMs:    median(load),
    });
  }
  return results;
}

async function scenarioColdStart() {
  // Force model eviction by setting keep_alive=0s on a short request, then
  // measure the NEXT request's load_duration + time-to-useful-response.
  await generate("bye", { keepAlive: "0s", numPredict: 1 });
  await new Promise(r => setTimeout(r, 1500)); // let eviction settle
  const t0 = performance.now();
  const out = await generate("Say hi in five words.", { numPredict: 16 });
  const wall = performance.now() - t0;
  return { loadMs: out.loadMs, firstPromptWallMs: wall, genTps: out.genTps };
}

async function scenarioConcurrent(levels = [1, 2, 4, 8]) {
  const results = [];
  for (const n of levels) {
    const samples = [];
    for (let r = 0; r < Math.max(1, Math.ceil(RUNS / 2)); r++) {
      const prompts = Array.from({ length: n }, (_, k) => mkPrompt(1, r * 100 + k)); // short prompts
      const t0 = performance.now();
      const outs = await Promise.all(prompts.map(p => generate(p, { numPredict: 64 })));
      const wall = (performance.now() - t0) / 1000;
      const totalGen = outs.reduce((a, o) => a + o.genTokens, 0);
      samples.push({ aggGenTps: totalGen / wall, perStreamGenTps: median(outs.map(o => o.genTps)) });
    }
    results.push({
      parallel: n,
      aggGenTps:       median(samples.map(s => s.aggGenTps)),
      perStreamGenTps: median(samples.map(s => s.perStreamGenTps)),
    });
  }
  return results;
}

// ── Environment snapshot ─────────────────────────────────────────────────────
function tryExec(cmd) {
  try { return execSync(cmd, { timeout: 5000, stdio: ["ignore", "pipe", "ignore"] }).toString().trim(); }
  catch { return null; }
}

async function env() {
  // Query Ollama's own API for its view. When run inside a container the host
  // `docker` / `nvidia-smi` won't be reachable, so use the API where possible.
  let ollamaVersion = null;
  try {
    const r = await fetch(`${HOST}/api/version`);
    if (r.ok) ollamaVersion = (await r.json()).version;
  } catch {}
  let modelDigest = null, modelParams = null, quant = null;
  try {
    const r = await fetch(`${HOST}/api/show`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ name: MODEL }) });
    if (r.ok) {
      const j = await r.json();
      modelParams = j.details?.parameter_size ?? null;
      quant = j.details?.quantization_level ?? null;
    }
  } catch {}
  // Digest is on /api/tags, not /api/show
  try {
    const r = await fetch(`${HOST}/api/tags`);
    if (r.ok) {
      const j = await r.json();
      const hit = (j.models ?? []).find(m => m.name === MODEL || m.model === MODEL);
      modelDigest = hit?.digest ?? null;
    }
  } catch {}

  // GPU state — skipped silently when nvidia-smi isn't on PATH (e.g. CPU-only
  // or when running inside a container without the nvidia runtime mapping).
  let gpu = null;
  const nvOut = tryExec("nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.current.sm --format=csv,noheader,nounits");
  if (nvOut) {
    const lines = nvOut.split("\n").filter(Boolean);
    gpu = lines.map(l => {
      const [name, driver, memTotal, memUsed, util, temp, power, clock] = l.split(",").map(s => s.trim());
      return { name, driver, memTotalMiB: +memTotal, memUsedMiB: +memUsed, utilPct: +util, tempC: +temp, powerW: +power, clockSmMHz: +clock };
    });
  }

  // Ollama server env — OLLAMA_NUM_PARALLEL governs whether concurrent
  // requests batch or serialize, so it's the #1 concurrency-shape variable.
  // Read via `docker inspect` when Ollama runs as a container; otherwise
  // fall back to the host env.
  let ollamaServerEnv = null;
  const dockerEnv = tryExec("docker inspect ollama --format '{{json .Config.Env}}' 2>/dev/null");
  if (dockerEnv) {
    try {
      const arr = JSON.parse(dockerEnv);
      const rel = arr.filter(kv => /^OLLAMA_/.test(kv));
      ollamaServerEnv = Object.fromEntries(rel.map(kv => { const i = kv.indexOf("="); return [kv.slice(0, i), kv.slice(i + 1)]; }));
    } catch {}
  }
  if (!ollamaServerEnv) {
    const relevant = ["OLLAMA_NUM_PARALLEL", "OLLAMA_MAX_LOADED_MODELS", "OLLAMA_KEEP_ALIVE", "OLLAMA_FLASH_ATTENTION", "OLLAMA_KV_CACHE_TYPE"];
    ollamaServerEnv = Object.fromEntries(relevant.map(k => [k, process.env[k] ?? null]).filter(([, v]) => v !== null));
    if (Object.keys(ollamaServerEnv).length === 0) ollamaServerEnv = null;
  }
  return {
    timestamp: new Date().toISOString(),
    model: MODEL,
    host: HOST,
    ollamaVersion,
    modelDigest,
    modelParams,
    modelQuant: quant,
    runs: RUNS,
    node: process.version,
    // May be null when running inside a minimal container image; that's fine.
    hostname: tryExec("hostname"),
    kernel: tryExec("uname -r"),
    gpu,
    ollamaServerEnv,
  };
}

// ── Reporting ────────────────────────────────────────────────────────────────
function pctDelta(cur, base) {
  if (!base || base === 0) return null;
  return ((cur - base) / base) * 100;
}
function fmtDelta(d, better = "higher") {
  if (d === null) return "     —";
  const sign = d >= 0 ? "+" : "";
  const isRegression = better === "higher" ? d < -REG_PCT : d > REG_PCT;
  const marker = isRegression ? " ⚠" : "";
  return `${sign}${d.toFixed(1)}%${marker}`.padStart(8);
}

function printSingleStream(cur, base) {
  console.log("\n[single-stream generation]");
  console.log("ctx    | prompt t/s |  gen t/s  | total ms  |  Δ prompt  |   Δ gen    |  Δ total");
  console.log("-".repeat(90));
  for (const row of cur) {
    const b = base?.singleStream?.find(r => r.ctx === row.ctx);
    console.log(
      row.ctx.padEnd(7) + "| " +
      row.promptTps.toFixed(0).padStart(10) + " | " +
      row.genTps.toFixed(1).padStart(9) + " | " +
      row.totalMs.toFixed(0).padStart(9) + " | " +
      (b ? fmtDelta(pctDelta(row.promptTps, b.promptTps)) : "     —  ") + " | " +
      (b ? fmtDelta(pctDelta(row.genTps, b.genTps))       : "     —  ") + " | " +
      (b ? fmtDelta(pctDelta(row.totalMs, b.totalMs), "lower") : "     —")
    );
  }
}

function printColdStart(cur, base) {
  console.log("\n[cold-start]");
  console.log(`load ms:              ${cur.loadMs.toFixed(0).padStart(7)}${base ? "   Δ " + fmtDelta(pctDelta(cur.loadMs, base.coldStart.loadMs), "lower") : ""}`);
  console.log(`first prompt wall ms: ${cur.firstPromptWallMs.toFixed(0).padStart(7)}${base ? "   Δ " + fmtDelta(pctDelta(cur.firstPromptWallMs, base.coldStart.firstPromptWallMs), "lower") : ""}`);
  console.log(`gen t/s:              ${cur.genTps.toFixed(1).padStart(7)}${base ? "   Δ " + fmtDelta(pctDelta(cur.genTps, base.coldStart.genTps)) : ""}`);
}

function printConcurrent(cur, base) {
  console.log("\n[concurrent streams]");
  console.log("parallel | agg t/s | per-stream t/s |  Δ agg   | Δ per-stream");
  console.log("-".repeat(66));
  for (const row of cur) {
    const b = base?.concurrent?.find(r => r.parallel === row.parallel);
    console.log(
      String(row.parallel).padStart(8) + " | " +
      row.aggGenTps.toFixed(1).padStart(7) + " | " +
      row.perStreamGenTps.toFixed(1).padStart(14) + " | " +
      (b ? fmtDelta(pctDelta(row.aggGenTps, b.aggGenTps)) : "     —  ") + " | " +
      (b ? fmtDelta(pctDelta(row.perStreamGenTps, b.perStreamGenTps)) : "     —")
    );
  }
}

function envDiff(cur, base) {
  if (!base?.env) return;
  const changes = [];
  const scalarKeys = ["ollamaVersion", "modelDigest", "modelParams", "modelQuant", "host", "kernel", "hostname"];
  for (const k of scalarKeys) if (cur[k] !== base.env[k]) changes.push(`${k}: ${base.env[k]} → ${cur[k]}`);

  // GPU — flag hardware/driver changes; ignore transient util/temp/power/clock
  // since those vary between any two runs on an idle machine.
  const curGpu = cur.gpu?.[0], baseGpu = base.env.gpu?.[0];
  if (curGpu && baseGpu) {
    for (const k of ["name", "driver", "memTotalMiB"]) {
      if (curGpu[k] !== baseGpu[k]) changes.push(`gpu.${k}: ${baseGpu[k]} → ${curGpu[k]}`);
    }
  } else if (!curGpu !== !baseGpu) {
    changes.push(`gpu: ${baseGpu ? "present" : "absent"} → ${curGpu ? "present" : "absent"}`);
  }

  // Ollama server env — these are the config knobs most likely to explain a
  // concurrency or latency shift.
  const curEnv = cur.ollamaServerEnv ?? {}, baseEnv = base.env.ollamaServerEnv ?? {};
  const envKeys = new Set([...Object.keys(curEnv), ...Object.keys(baseEnv)]);
  for (const k of envKeys) {
    if (curEnv[k] !== baseEnv[k]) changes.push(`${k}: ${baseEnv[k] ?? "(unset)"} → ${curEnv[k] ?? "(unset)"}`);
  }

  if (changes.length === 0) return;
  console.log("\n[environment changes vs baseline]");
  for (const line of changes) console.log(`  ${line}`);
}

function gpuStateSummary(env) {
  if (!env.gpu || env.gpu.length === 0) return null;
  const g = env.gpu[0];
  return `${g.name} (driver ${g.driver}) — ${g.memUsedMiB}/${g.memTotalMiB} MiB, util ${g.utilPct}%, ${g.tempC}°C, ${g.powerW}W, SM ${g.clockSmMHz} MHz`;
}

function serverEnvSummary(env) {
  const e = env.ollamaServerEnv;
  if (!e || Object.keys(e).length === 0) return null;
  return Object.entries(e).map(([k, v]) => `${k}=${v}`).join(" ");
}

// ── Main ─────────────────────────────────────────────────────────────────────
async function main() {
  const envSnap = await env();
  console.log(`bench: model=${envSnap.model} host=${envSnap.host} runs=${RUNS} ollama=${envSnap.ollamaVersion} digest=${(envSnap.modelDigest ?? "").slice(0, 12)}`);
  const gpuLine = gpuStateSummary(envSnap);
  if (gpuLine) console.log(`gpu:   ${gpuLine}`);
  const srvLine = serverEnvSummary(envSnap);
  if (srvLine) console.log(`ollama env: ${srvLine}`);
  // Warn if GPU is not idle — background work will skew numbers.
  if (envSnap.gpu?.[0] && envSnap.gpu[0].utilPct >= 10) {
    console.log(`⚠ GPU utilization ${envSnap.gpu[0].utilPct}% before run — numbers may be noisy`);
  }
  console.log("warming up…");
  await generate("ok?", { numPredict: 8 });

  console.log("\n[1/3] single-stream sizes…");
  const singleStream = await scenarioSingleStream();
  console.log("[2/3] cold-start…");
  const coldStart = await scenarioColdStart();
  console.log("[3/3] concurrent streams…");
  const concurrent = await scenarioConcurrent();

  const current = { env: envSnap, singleStream, coldStart, concurrent };

  if (cmd === "save") {
    writeFileSync(OUT, JSON.stringify(current, null, 2));
    console.log(`\nbaseline saved → ${OUT}`);
  }

  const base = (cmd === "compare" && existsSync(OUT))
    ? JSON.parse(readFileSync(OUT, "utf-8")) : null;

  if (cmd === "compare") {
    if (!base) { console.log(`\nno baseline at ${OUT} — run 'save' first`); }
    else envDiff(envSnap, base);
  }

  printSingleStream(singleStream, base);
  printColdStart(coldStart, base);
  printConcurrent(concurrent, base);

  if (cmd === "compare" && base) {
    console.log(`\nregression threshold: ${REG_PCT}% slower flags ⚠`);
  }
}

main().catch(e => { console.error(e); process.exit(1); });
