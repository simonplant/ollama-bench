#!/usr/bin/env node
/**
 * ollama-bench — unified CLI for throughput regression + tool-call probes.
 *
 * Default (no subcommand) runs the throughput benchmark with smart baseline
 * behavior: saves a baseline on first run, compares against it on every run
 * after. That's what a user expects from "just ./bench".
 *
 * Subcommands:
 *   perf [save|compare|run]   Throughput benchmark. No mode → smart default.
 *   toolcall                  Single-turn tool-call accuracy probe.
 *   multiturn                 Multi-turn tool-call probe (post-tool-result).
 *   doctor                    Preflight system audit — flags GPU/Ollama/host misconfigs.
 *   all                       perf (smart) → toolcall → multiturn, sequentially.
 *   baseline [show|clear]     Inspect or delete baseline.json.
 *   help                      Print usage.
 *
 * Flags (apply to any subcommand that uses them):
 *   --model <tag>             default gemma4:26b
 *   --host <url>              default http://ollama:11434
 *   --runs <n>                per-cell runs, default 3 (perf only)
 *   --out <path>              baseline file, default ./baseline.json
 *   --regression-pct <n>      flag threshold, default 5 (perf only)
 *   -v, --verbose             per-case output (toolcall/multiturn)
 *
 * Back-compat: `bench.mjs save|compare|run` still works, routes to `perf`.
 */

import { readFileSync, writeFileSync, existsSync, unlinkSync } from "node:fs";
import { execSync, spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));

const args = process.argv.slice(2);
// lastIndexOf so later flags override earlier ones — lets the Docker wrapper
// append --out <in-container-path> after user args without the user's host
// path (or accidental earlier value) winning.
const arg = (n, fb) => { const i = args.lastIndexOf(n); return i >= 0 ? args[i + 1] : fb; };

// Flags that consume the next argv entry as their value. Anything else
// starting with `-` is a boolean; anything else is positional.
const VALUE_FLAGS = new Set(["--model", "--host", "--runs", "--out", "--regression-pct"]);
function parseArgs(argv) {
  const flags = [], positional = [];
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (VALUE_FLAGS.has(a))        { flags.push(a, argv[++i] ?? ""); }
    else if (a.startsWith("-"))    { flags.push(a); }
    else                           { positional.push(a); }
  }
  return { flags, positional };
}
const PARSED = parseArgs(args);

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
  // request floor. Dropped from defaults.
  { label: "short",     pad: 180,  numPredict: 128  },
  { label: "medium",    pad: 1800, numPredict: 128  },
  { label: "long",      pad: 7800, numPredict: 128  },
  // Long-gen cell — same short prompt, 1024 output tokens. Catches
  // regressions in autoregressive decode that short-gen cells miss
  // (KV-cache growth, sampler overhead at depth).
  { label: "long-gen",  pad: 180,  numPredict: 1024 },
];

// Concurrency cell — medium prompt × short gen. Short gen keeps the wall
// time reasonable at parallel=8; medium prompt is representative of real
// chat turns. Edit here to change what the concurrency scenario measures.
const CONCURRENT_CELL = { ctxIdx: 1, numPredict: 64 };

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
    promptTokens:    j.prompt_eval_count ?? 0,
    genTokens:       j.eval_count ?? 0,
    promptTps:       j.prompt_eval_count / (j.prompt_eval_duration / 1e9),
    genTps:          j.eval_count / (j.eval_duration / 1e9),
    totalMs:         j.total_duration / 1e6,
    loadMs:          (j.load_duration ?? 0) / 1e6,
    evalDurationMs:  (j.eval_duration ?? 0) / 1e6,
  };
}

// ── Metrics helpers ──────────────────────────────────────────────────────────
function median(xs) {
  const s = [...xs].sort((a, b) => a - b);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function stdev(xs) {
  if (xs.length < 2) return 0;
  const m = xs.reduce((a, v) => a + v, 0) / xs.length;
  const variance = xs.reduce((a, v) => a + (v - m) ** 2, 0) / (xs.length - 1);
  return Math.sqrt(variance);
}

// Coefficient-of-variation in percent. Used as a per-cell noise floor so the
// regression threshold is `max(--regression-pct, 2 * cv_%)` — catches real
// regressions on quiet cells without false-firing on noisy ones.
function cvPct(xs) {
  if (xs.length < 2) return 0;
  const m = xs.reduce((a, v) => a + v, 0) / xs.length;
  if (m === 0) return 0;
  return 100 * stdev(xs) / m;
}

// ── Scenarios ────────────────────────────────────────────────────────────────
// On `save` runs we do extra iterations to estimate per-cell noise (cv%);
// `compare` and `run` stick to the cheaper default.
async function scenarioSingleStream(isSave) {
  const runsHere = isSave ? Math.max(RUNS, 6) : RUNS;
  const results = [];
  for (let i = 0; i < CTX_SIZES.length; i++) {
    const cell = CTX_SIZES[i];
    const pTps = [], gTps = [], total = [], load = [];
    for (let r = 0; r < runsHere; r++) {
      const out = await generate(mkPrompt(i, r), { numPredict: cell.numPredict });
      pTps.push(out.promptTps);
      gTps.push(out.genTps);
      total.push(out.totalMs);
      load.push(out.loadMs);
    }
    results.push({
      ctx: cell.label,
      promptTps: median(pTps),
      genTps:    median(gTps),
      totalMs:   median(total),
      loadMs:    median(load),
      cv: isSave ? {
        promptTps: cvPct(pTps),
        genTps:    cvPct(gTps),
        totalMs:   cvPct(total),
      } : undefined,
    });
  }
  return results;
}

async function scenarioColdStart() {
  // Force model eviction by setting keep_alive=0s on a short request, then
  // measure the NEXT request's load_duration + time-to-useful-response.
  // Prefix with PROCESS_NONCE so back-to-back runs don't share a prompt.
  await generate(`[${PROCESS_NONCE}/cold/evict] bye`, { keepAlive: "0s", numPredict: 1 });
  await new Promise(r => setTimeout(r, 1500)); // let eviction settle
  const t0 = performance.now();
  const out = await generate(`[${PROCESS_NONCE}/cold/measure] Say hi in five words.`, { numPredict: 16 });
  const wall = performance.now() - t0;
  return { loadMs: out.loadMs, firstPromptWallMs: wall, genTps: out.genTps };
}

async function scenarioConcurrent(isSave, levels = [1, 2, 4, 8]) {
  const results = [];
  const samplesHere = isSave ? Math.max(3, Math.ceil(RUNS / 2) + 2) : Math.max(1, Math.ceil(RUNS / 2));
  const { ctxIdx, numPredict } = CONCURRENT_CELL;
  for (const n of levels) {
    const samples = [];
    for (let r = 0; r < samplesHere; r++) {
      const prompts = Array.from({ length: n }, (_, k) => mkPrompt(ctxIdx, r * 100 + k));
      const t0 = performance.now();
      const outs = await Promise.all(prompts.map(p => generate(p, { numPredict })));
      const wall = (performance.now() - t0) / 1000;
      const totalGen = outs.reduce((a, o) => a + o.genTokens, 0);
      // e2e = end-to-end throughput including prompt-eval + network; what a
      // caller experiences. decode = pure decoder aggregate; isolates batcher
      // scaling by dividing out prompt-eval. max(eval_duration) is the wall
      // time of the slowest stream's decode, i.e. the span during which all
      // streams' decode overlapped.
      const maxEvalSec = Math.max(...outs.map(o => o.evalDurationMs / 1000)) || wall;
      samples.push({
        e2eGenTps:       totalGen / wall,
        decodeGenTps:    totalGen / maxEvalSec,
        perStreamGenTps: median(outs.map(o => o.genTps)),
      });
    }
    const e2es    = samples.map(s => s.e2eGenTps);
    const decodes = samples.map(s => s.decodeGenTps);
    const pers    = samples.map(s => s.perStreamGenTps);
    results.push({
      parallel: n,
      e2eGenTps:       median(e2es),
      decodeGenTps:    median(decodes),
      perStreamGenTps: median(pers),
      cv: isSave ? {
        e2eGenTps:       cvPct(e2es),
        decodeGenTps:    cvPct(decodes),
        perStreamGenTps: cvPct(pers),
      } : undefined,
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

  // GPU state — prefer host-injected CSV (Docker-sibling flow: sibling rarely
  // has nvidia-smi, so the wrapper runs it on the host and forwards results).
  // Fall back to local nvidia-smi for direct host invocation.
  let gpu = null;
  const nvOut = process.env.OLLAMA_BENCH_GPU_CSV
    || tryExec("nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.current.sm --format=csv,noheader,nounits");
  if (nvOut) {
    const lines = nvOut.split("\n").filter(Boolean);
    gpu = lines.map(l => {
      const [name, driver, memTotal, memUsed, util, temp, power, clock] = l.split(",").map(s => s.trim());
      return { name, driver, memTotalMiB: +memTotal, memUsedMiB: +memUsed, utilPct: +util, tempC: +temp, powerW: +power, clockSmMHz: +clock };
    });
  }

  // Ollama server env — OLLAMA_NUM_PARALLEL governs whether concurrent
  // requests batch or serialize, so it's the #1 concurrency-shape variable.
  // Prefer host-injected JSON (Docker-sibling flow: sibling has no docker
  // socket, so the wrapper runs `docker inspect` on the host and forwards).
  // Fall back to local `docker inspect`, then host env.
  let ollamaServerEnv = null;
  const dockerEnv = process.env.OLLAMA_BENCH_SERVER_ENV_JSON
    || tryExec("docker inspect ollama --format '{{json .Config.Env}}' 2>/dev/null");
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
  const hostname = tryExec("hostname");
  const kernel   = tryExec("uname -r");

  // Machine fingerprint — short hash over the fields that *define* which
  // physical machine produced the numbers. When run inside a container,
  // hostname is the (ephemeral) container id and gpu/server-env are null,
  // so the fallback hash is unstable across container recreates. The
  // wrapper passes in the host's /etc/machine-id via OLLAMA_BENCH_MACHINE_ID
  // to give us a stable anchor; we prefer it when present.
  const hostMachineId = process.env.OLLAMA_BENCH_MACHINE_ID || null;
  const fingerprintSource = hostMachineId
    ? JSON.stringify({ hostMachineId })
    : JSON.stringify({
        gpuName: gpu?.[0]?.name ?? null,
        gpuDriver: gpu?.[0]?.driver ?? null,
        gpuMemTotalMiB: gpu?.[0]?.memTotalMiB ?? null,
        hostname,
        kernel,
      });
  const machineId = createHash("sha256").update(fingerprintSource).digest("hex").slice(0, 12);

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
    hostname,
    kernel,
    machineId,
    hostMachineId,
    gpu,
    ollamaServerEnv,
  };
}

// ── Reporting ────────────────────────────────────────────────────────────────
function pctDelta(cur, base) {
  if (!base || base === 0) return null;
  return ((cur - base) / base) * 100;
}
// Per-cell threshold = max(user-set floor, 2× measured cv%). The 2σ multiplier
// means a regression has to exceed ~95th-percentile noise for the baseline
// cell to fire. `cvBaseline` may be undefined on old baselines — fall back
// to the global floor.
function thresholdFor(cvBaseline) {
  const noiseFloor = typeof cvBaseline === "number" ? 2 * cvBaseline : 0;
  return Math.max(REG_PCT, noiseFloor);
}
function isRegression(d, better, cvBaseline) {
  if (d === null) return false;
  const thr = thresholdFor(cvBaseline);
  return better === "higher" ? d < -thr : d > thr;
}
function fmtDelta(d, better = "higher", cvBaseline) {
  if (d === null) return "     —";
  const sign = d >= 0 ? "+" : "";
  const marker = isRegression(d, better, cvBaseline) ? " ⚠" : "";
  return `${sign}${d.toFixed(1)}%${marker}`.padStart(8);
}

// Walk every comparable cell and collect the ones past threshold. Used for the
// end-of-run verdict line so the user doesn't have to scan three tables.
function collectRegressions(cur, base) {
  const regs = [];
  const check = (label, c, b, better, cv) => {
    if (b === undefined || b === null) return;
    const d = pctDelta(c, b);
    if (isRegression(d, better, cv)) {
      const sign = d >= 0 ? "+" : "";
      regs.push(`${label} ${sign}${d.toFixed(1)}%`);
    }
  };
  for (const row of cur.singleStream ?? []) {
    const b = base?.singleStream?.find(r => r.ctx === row.ctx);
    if (!b) continue;
    check(`${row.ctx} prompt t/s`, row.promptTps, b.promptTps, "higher", b.cv?.promptTps);
    check(`${row.ctx} gen t/s`,    row.genTps,    b.genTps,    "higher", b.cv?.genTps);
    check(`${row.ctx} total ms`,   row.totalMs,   b.totalMs,   "lower",  b.cv?.totalMs);
  }
  if (base?.coldStart && cur.coldStart) {
    check("cold-start load ms",       cur.coldStart.loadMs,            base.coldStart.loadMs,            "lower");
    check("cold-start first wall ms", cur.coldStart.firstPromptWallMs, base.coldStart.firstPromptWallMs, "lower");
    check("cold-start gen t/s",       cur.coldStart.genTps,            base.coldStart.genTps,            "higher");
  }
  for (const row of cur.concurrent ?? []) {
    const b = base?.concurrent?.find(r => r.parallel === row.parallel);
    if (!b) continue;
    const bE2e   = b.e2eGenTps   ?? b.aggGenTps;
    const bE2eCv = b.cv?.e2eGenTps ?? b.cv?.aggGenTps;
    check(`n=${row.parallel} e2e t/s`,        row.e2eGenTps,       bE2e,              "higher", bE2eCv);
    check(`n=${row.parallel} decode t/s`,     row.decodeGenTps,    b.decodeGenTps,    "higher", b.cv?.decodeGenTps);
    check(`n=${row.parallel} per-stream t/s`, row.perStreamGenTps, b.perStreamGenTps, "higher", b.cv?.perStreamGenTps);
  }
  return regs;
}

function printSingleStream(cur, base) {
  console.log("\n[single-stream generation]");
  console.log("ctx       | prompt t/s |  gen t/s  | total ms  |  Δ prompt  |   Δ gen    |  Δ total");
  console.log("-".repeat(93));
  for (const row of cur) {
    const b = base?.singleStream?.find(r => r.ctx === row.ctx);
    console.log(
      row.ctx.padEnd(10) + "| " +
      row.promptTps.toFixed(0).padStart(10) + " | " +
      row.genTps.toFixed(1).padStart(9) + " | " +
      row.totalMs.toFixed(0).padStart(9) + " | " +
      (b ? fmtDelta(pctDelta(row.promptTps, b.promptTps), "higher", b.cv?.promptTps) : "     —  ") + " | " +
      (b ? fmtDelta(pctDelta(row.genTps, b.genTps), "higher", b.cv?.genTps)          : "     —  ") + " | " +
      (b ? fmtDelta(pctDelta(row.totalMs, b.totalMs), "lower", b.cv?.totalMs)        : "     —")
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
  console.log("\n[concurrent streams]   e2e = end-to-end t/s (includes prompt-eval + network)");
  console.log("                       decode = aggregate pure-decode t/s");
  console.log("parallel | e2e t/s | decode t/s | per-stream t/s |  Δ e2e   | Δ decode |  Δ per-stream");
  console.log("-".repeat(92));
  for (const row of cur) {
    const b = base?.concurrent?.find(r => r.parallel === row.parallel);
    // Support old baselines that used `aggGenTps` as the only aggregate.
    const bE2e = b ? (b.e2eGenTps ?? b.aggGenTps) : undefined;
    const bE2eCv = b ? (b.cv?.e2eGenTps ?? b.cv?.aggGenTps) : undefined;
    console.log(
      String(row.parallel).padStart(8) + " | " +
      row.e2eGenTps.toFixed(1).padStart(7) + " | " +
      row.decodeGenTps.toFixed(1).padStart(10) + " | " +
      row.perStreamGenTps.toFixed(1).padStart(14) + " | " +
      (bE2e !== undefined ? fmtDelta(pctDelta(row.e2eGenTps, bE2e), "higher", bE2eCv) : "     —  ") + " | " +
      (b?.decodeGenTps !== undefined ? fmtDelta(pctDelta(row.decodeGenTps, b.decodeGenTps), "higher", b.cv?.decodeGenTps) : "     —  ") + " | " +
      (b ? fmtDelta(pctDelta(row.perStreamGenTps, b.perStreamGenTps), "higher", b.cv?.perStreamGenTps) : "     —")
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

function readBaseline(path) {
  if (!existsSync(path)) return null;
  try {
    return JSON.parse(readFileSync(path, "utf-8"));
  } catch (e) {
    console.error(`baseline at ${path} is unreadable (${e.message}) — delete it or run './bench baseline clear'`);
    process.exit(1);
  }
}

// ── Perf runner ──────────────────────────────────────────────────────────────
// mode: "save" | "compare" | "run" | "smart"
//   smart → compare if baseline exists (same machine), save if not, run otherwise.
async function runPerf(mode) {
  const envSnap = await env();

  // Resolve smart mode before announcing, so the header says what we're doing.
  let resolved = mode;
  let base = readBaseline(OUT);
  if (mode === "smart") {
    if (!base) {
      resolved = "save";
      console.log(`(no baseline at ${OUT} — saving one now; re-run after any change to see regressions)`);
    } else if (base.env?.machineId && envSnap.machineId && base.env.machineId !== envSnap.machineId) {
      resolved = "run";
      console.log(`⚠ baseline is from a different machine (${base.env.machineId} vs ${envSnap.machineId}) — showing raw numbers, no compare.`);
    } else {
      resolved = "compare";
    }
  }

  console.log(`bench: model=${envSnap.model} host=${envSnap.host} runs=${RUNS} ollama=${envSnap.ollamaVersion} digest=${(envSnap.modelDigest ?? "").slice(0, 12)}`);
  const gpuLine = gpuStateSummary(envSnap);
  if (gpuLine) console.log(`gpu:   ${gpuLine}`);
  const srvLine = serverEnvSummary(envSnap);
  if (srvLine) console.log(`ollama env: ${srvLine}`);
  if (envSnap.gpu?.[0] && envSnap.gpu[0].utilPct >= 10) {
    console.log(`⚠ GPU utilization ${envSnap.gpu[0].utilPct}% before run — numbers may be noisy`);
  }
  console.log("warming up…");
  await generate("ok?", { numPredict: 8 });

  const isSave = resolved === "save";
  if (isSave) console.log("(save mode — extra runs to measure per-cell noise)");
  console.log("\n[1/3] single-stream sizes…");
  const singleStream = await scenarioSingleStream(isSave);
  console.log("[2/3] cold-start…");
  const coldStart = await scenarioColdStart();
  console.log("[3/3] concurrent streams…");
  const concurrent = await scenarioConcurrent(isSave);

  const current = { env: envSnap, singleStream, coldStart, concurrent };

  if (resolved === "save") {
    writeFileSync(OUT, JSON.stringify(current, null, 2));
    console.log(`\nbaseline saved → ${OUT}`);
    base = null; // don't diff a just-saved baseline against itself
  }

  if (resolved === "compare") {
    if (!base) {
      console.log(`\nno baseline at ${OUT} — run './bench perf save' first`);
    } else {
      if (base.env?.machineId && envSnap.machineId && base.env.machineId !== envSnap.machineId) {
        console.log(`\n⚠ machine mismatch: baseline from ${base.env.machineId} (${base.env.hostname ?? "?"} · ${base.env.gpu?.[0]?.name ?? "no GPU"})`);
        console.log(`                  current  from ${envSnap.machineId} (${envSnap.hostname ?? "?"} · ${envSnap.gpu?.[0]?.name ?? "no GPU"})`);
        console.log(`  Deltas below will be dominated by hardware/driver differences, not your change.`);
      }
      envDiff(envSnap, base);
    }
  } else {
    base = null;
  }

  printSingleStream(singleStream, base);
  printColdStart(coldStart, base);
  printConcurrent(concurrent, base);

  if (resolved === "compare" && base) {
    const regs = collectRegressions(current, base);
    if (regs.length === 0) {
      console.log("\n✓ no regressions");
    } else {
      const shown = regs.slice(0, 8);
      const overflow = regs.length - shown.length;
      const tail = overflow > 0 ? `, …and ${overflow} more` : "";
      console.log(`\n⚠ ${regs.length} cell${regs.length === 1 ? "" : "s"} regressed: ${shown.join(", ")}${tail}`);
    }
    const hasCv = base.singleStream?.some(r => r.cv);
    console.log(`regression threshold: ${REG_PCT}% ${hasCv ? "or 2× per-cell noise (whichever is higher)" : ""} → flags ⚠`);
  }
}

// ── Subcommand dispatch ──────────────────────────────────────────────────────
function spawnSibling(script) {
  // Forward all flags+values; drop the first positional (the subcommand).
  // Using PARSED avoids the earlier bug where the first flag-value entry
  // could be mistaken for the subcommand.
  const forwarded = [...PARSED.flags, ...PARSED.positional.slice(1)];
  return new Promise((resolve, reject) => {
    const p = spawn(process.execPath, [join(SCRIPT_DIR, script), ...forwarded], { stdio: "inherit" });
    p.on("exit", code => code === 0 ? resolve() : reject(new Error(`${script} exited ${code}`)));
  });
}

async function runAll() {
  console.log("=== [1/3] perf ===");
  await runPerf("smart");
  console.log("\n=== [2/3] toolcall ===");
  await spawnSibling("bench-toolcall.mjs");
  console.log("\n=== [3/3] multiturn ===");
  await spawnSibling("bench-multiturn.mjs");
}

function cmdBaseline(sub) {
  if (sub === "clear") {
    if (!existsSync(OUT)) { console.log(`no baseline at ${OUT}`); return; }
    unlinkSync(OUT);
    console.log(`removed ${OUT}`);
    return;
  }
  // default: show
  if (!existsSync(OUT)) { console.log(`no baseline at ${OUT} — run './bench perf save' to create one`); return; }
  const b = readBaseline(OUT);
  console.log(`baseline: ${OUT}`);
  console.log(`  saved:   ${b.env?.timestamp ?? "?"}`);
  console.log(`  model:   ${b.env?.model ?? "?"} (digest ${(b.env?.modelDigest ?? "").slice(0, 12)})`);
  console.log(`  host:    ${b.env?.host ?? "?"}`);
  console.log(`  machine: ${b.env?.machineId ?? "?"} (${b.env?.hostname ?? "?"} · ${b.env?.gpu?.[0]?.name ?? "no GPU"})`);
  console.log(`  ollama:  ${b.env?.ollamaVersion ?? "?"}`);
  if (b.singleStream) {
    console.log(`  single-stream gen t/s: ${b.singleStream.map(r => `${r.ctx}=${r.genTps.toFixed(1)}`).join("  ")}`);
  }
}

function printHelp() {
  console.log(`ollama-bench — throughput regression harness + tool-call probes

USAGE
  ./bench [subcommand] [flags]

SUBCOMMANDS
  (none)            Run throughput benchmark with smart baseline behavior:
                    saves a baseline on first run, compares on every run after.
  perf [mode]       Throughput benchmark. mode: save | compare | run | (smart).
  toolcall          Single-turn tool-call accuracy probe (22 cases).
  multiturn         Multi-turn tool-call probe, after fabricated tool result (14 cases).
  doctor            Preflight audit: persistence mode, power cap, governor, KV/FA combo, etc.
  all               Run perf (smart) + toolcall + multiturn, sequentially.
  baseline [show]   Show baseline summary (default).
  baseline clear    Delete baseline.json.
  help              This message.

FLAGS
  --model <tag>            default gemma4:26b
  --host <url>             default http://ollama:11434
  --runs <n>               per-cell runs for perf, default 3
  --out <path>             baseline file, default ./baseline.json
  --regression-pct <n>     regression threshold, default 5 (%)
  -v, --verbose            per-case output for toolcall/multiturn

EXAMPLES
  ./bench                              # smart perf run
  ./bench doctor                       # audit GPU/host/Ollama config
  ./bench perf save                    # force-save a new baseline
  ./bench toolcall -v                  # accuracy probe, per-case output
  ./bench all --model qwen3:30b        # compare a different model end-to-end
  ./bench baseline clear               # start fresh

Back-compat: 'save'/'compare'/'run' at the top level still work, routed to 'perf'.`);
}

// ── Main dispatcher ──────────────────────────────────────────────────────────
async function main() {
  const { positional } = PARSED;
  if (args.includes("--help") || args.includes("-h") || positional[0] === "help") {
    printHelp();
    return;
  }

  const cmd = positional[0] ?? "";
  const sub = positional[1] ?? "";

  switch (cmd) {
    case "":
      return runPerf("smart");
    case "perf":
      if (sub && !["save", "compare", "run"].includes(sub)) {
        console.error(`unknown perf mode: ${sub} (expected save|compare|run)`);
        process.exit(2);
      }
      return runPerf(sub || "smart");
    // Back-compat: bare save/compare/run at top level
    case "save": case "compare": case "run":
      return runPerf(cmd);
    case "toolcall":
      return spawnSibling("bench-toolcall.mjs");
    case "multiturn":
      return spawnSibling("bench-multiturn.mjs");
    case "doctor":
      return spawnSibling("bench-doctor.mjs");
    case "all":
      return runAll();
    case "baseline":
      return cmdBaseline(sub || "show");
    default:
      console.error(`unknown command: ${cmd}\n`);
      printHelp();
      process.exit(2);
  }
}

main().catch(e => { console.error(e); process.exit(1); });
