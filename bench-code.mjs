#!/usr/bin/env node
/**
 * Code probe — HumanEval (164 cases). For each case:
 *   1. Send the function signature + docstring as a completion prompt
 *   2. Extract the model's code (strip fences, handle full-def vs continuation)
 *   3. Run prompt + completion + test + check(entry_point) in a sandboxed
 *      python3 subprocess with a wall-time cap
 *   4. Exit 0 = pass
 *
 * Deterministic, no judge. Requires python3 on PATH (the bench wrapper's
 * sibling container already has it; on the host it's standard).
 *
 * Sandbox: python3 -I (isolated mode), 8s wall timeout per case, process is
 * killed on overrun. Cases run sequentially so a hung subprocess can't pile
 * up. No network restriction at the OS level — we trust the python sandbox
 * inside the bench container.
 *
 * Usage:
 *   node bench-code.mjs [--model gemma4:26b] [--host http://ollama:11434]
 *                       [--out ./baseline.json] [--save|--compare]
 *                       [--limit N] [-v|--verbose]
 *
 * Per-call timeout (generation): 240s. Override: OLLAMA_BENCH_TIMEOUT_MS.
 * Per-case timeout (execution):   8s. Override: OLLAMA_BENCH_CODE_EXEC_MS.
 */

import { readFileSync, writeFileSync, mkdtempSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { tmpdir } from "node:os";
import { spawn } from "node:child_process";
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
const LIMIT   = (() => { const v = arg("--limit", null); return v ? parseInt(v, 10) : null; })();
const VERBOSE = args.includes("-v") || args.includes("--verbose");
const MODE    = args.includes("--save")    ? "save"
              : args.includes("--compare") ? "compare"
              : "smart";

const REG_PP = 5;

function genTimeoutMs(numPredict) {
  const override = parseInt(process.env.OLLAMA_BENCH_TIMEOUT_MS ?? "", 10);
  if (Number.isFinite(override) && override > 0) return override;
  return 240_000 + (numPredict || 1024) * 100;
}
function execTimeoutMs() {
  const override = parseInt(process.env.OLLAMA_BENCH_CODE_EXEC_MS ?? "", 10);
  if (Number.isFinite(override) && override > 0) return override;
  return 8_000;
}
function withTimeout(ms) {
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), ms);
  return { signal: ac.signal, cancel: () => clearTimeout(timer) };
}

function loadJsonl(path) {
  return readFileSync(path, "utf-8").trim().split("\n").map(l => JSON.parse(l));
}

// ── Generation ───────────────────────────────────────────────────────────────
const SYSTEM = "You are an expert Python programmer. Complete the function. " +
               "Respond with only the function body or full function definition, " +
               "no prose, no explanation.";

async function generate(prompt, think, numPredict) {
  const timeoutMs = genTimeoutMs(numPredict);
  const t = withTimeout(timeoutMs);
  let res;
  try {
    res = await fetch(`${HOST}/api/generate`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: MODEL,
        system: SYSTEM,
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

// ── Code extraction ──────────────────────────────────────────────────────────
// Three shapes models commonly emit for HumanEval:
//   A. ```python\n<code>\n```          — fenced markdown
//   B. def <entry_point>(...): ...     — full function definition
//   C. <body>                          — bare continuation of the signature
function extractCode(response, entryPoint) {
  let code = response;

  // Strip markdown fences if present. Take the first python/code block.
  const fenceMatch = code.match(/```(?:python|py)?\s*\n([\s\S]*?)```/);
  if (fenceMatch) code = fenceMatch[1];

  // Trim assistant chatter that occasionally leaks past the fence.
  code = code.replace(/^\s*Here(?:'s|\sis).*?\n/i, "").trim();

  return code;
}

// Assemble the test script. If the model emitted a full `def <entry_point>`,
// we use only its code (it replaces the prompt's stub). Otherwise we treat
// the response as a continuation of the prompt's signature.
function assembleScript(prompt, modelCode, test, entryPoint) {
  const hasFullDef = new RegExp(`^\\s*def\\s+${entryPoint}\\s*\\(`, "m").test(modelCode);
  const body = hasFullDef ? modelCode : prompt + modelCode;
  // Make sure the entry-point function exists at module scope, then run the
  // dataset's check() against it.
  return `${body}\n\n${test}\n\ncheck(${entryPoint})\n`;
}

// ── Sandboxed execution ──────────────────────────────────────────────────────
async function runPython(script) {
  return new Promise((resolve) => {
    const dir = mkdtempSync(join(tmpdir(), "humaneval-"));
    const path = join(dir, "case.py");
    writeFileSync(path, script);
    // python3 -I: isolated mode (no PYTHONPATH, no user site, ignore env)
    // python3 -B: no .pyc files
    const child = spawn("python3", ["-IB", path], {
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "", stderr = "";
    child.stdout.on("data", d => { stdout += d.toString(); });
    child.stderr.on("data", d => { stderr += d.toString(); });

    const timer = setTimeout(() => {
      // SIGKILL — SIGTERM gets swallowed by infinite loops.
      child.kill("SIGKILL");
    }, execTimeoutMs());

    child.on("close", (code, signal) => {
      clearTimeout(timer);
      rmSync(dir, { recursive: true, force: true });
      if (signal === "SIGKILL") return resolve({ pass: false, reason: `timeout >${execTimeoutMs()}ms` });
      if (code === 0) return resolve({ pass: true, reason: "ok" });
      // Surface first stderr line — usually the AssertionError or syntax issue.
      const firstErr = stderr.split("\n").reverse().find(l => l.trim()) ?? `exit ${code}`;
      resolve({ pass: false, reason: firstErr.slice(0, 120) });
    });
    child.on("error", e => {
      clearTimeout(timer);
      rmSync(dir, { recursive: true, force: true });
      resolve({ pass: false, reason: `spawn: ${e.message}` });
    });
  });
}

// ── Runner ───────────────────────────────────────────────────────────────────
async function runCases() {
  let cases = loadJsonl(join(ROOT, "data", "humaneval.jsonl"));
  if (LIMIT) cases = cases.slice(0, LIMIT);

  const { think, numPredict, supports } = await thinkingParams(HOST, MODEL, 1024, 8192);
  if (supports) console.log(`(thinking model: think=${think}, num_predict=${numPredict})\n`);

  const failed = [];
  const gpuHandle = startSampler();
  const sysHandle = startSysSampler();
  const t0 = performance.now();
  let pass = 0;
  for (const c of cases) {
    let scored;
    try {
      const response = await generate(c.prompt, think, numPredict);
      const code = extractCode(response, c.entry_point);
      const script = assembleScript(c.prompt, code, c.test, c.entry_point);
      scored = await runPython(script);
    } catch (e) {
      scored = { pass: false, reason: `threw: ${e.message}` };
    }
    if (scored.pass) pass++;
    else failed.push(c.task_id);
    if (VERBOSE) console.log(`${scored.pass ? "✔" : "✘"} ${c.task_id} — ${scored.reason}`);
  }
  const durationSec = (performance.now() - t0) / 1000;
  const gpu = await stopSampler(gpuHandle);
  const sys = stopSysSampler(sysHandle);
  return {
    savedAt: new Date().toISOString(),
    model: MODEL,
    total: cases.length,
    pass,
    codePct: 100 * pass / cases.length,
    failed,
    durationSec,
    gpu,
    sys,
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
  console.log("");
  console.log(`HumanEval: ${current.pass}/${current.total} = ${fmtPct(current.codePct)}` +
              (base ? `  (Δ ${fmtDelta(current.codePct - base.codePct)} vs ${base.savedAt})` : "") +
              `   wall: ${current.durationSec.toFixed(1)}s`);
  console.log(fmtGpuSummary(current.gpu));
  console.log(fmtSysSummary(current.sys));

  if (base) {
    const baseSet = new Set(base.failed ?? []);
    const curSet  = new Set(current.failed);
    const newFails  = [...curSet].filter(f => !baseSet.has(f));
    const newPasses = [...baseSet].filter(f => !curSet.has(f));
    if (newFails.length) console.log(`\nnewly failed: ${newFails.slice(0, 10).join(", ")}${newFails.length > 10 ? ", …" : ""}`);
    if (newPasses.length) console.log(`newly passed: ${newPasses.slice(0, 10).join(", ")}${newPasses.length > 10 ? ", …" : ""}`);
  } else if (current.failed.length) {
    console.log(`\nfailures (${current.failed.length}): ${current.failed.slice(0, 15).join(", ")}${current.failed.length > 15 ? ", …" : ""}`);
  }
}

// ── Main ─────────────────────────────────────────────────────────────────────
async function main() {
  console.log(`\nbench-code: model=${MODEL} host=${HOST} (HumanEval${LIMIT ? `, first ${LIMIT}` : ""})\n`);
  const existing = getModelSection(OUT, MODEL, "code");
  let mode = MODE;
  if (mode === "smart") mode = existing ? "compare" : "save";
  else if (mode === "compare" && !existing) {
    console.error(`no code entry for ${MODEL} at ${OUT} — run with --save first`);
    process.exit(1);
  }

  const current = await runCases();
  printReport(current, mode === "compare" ? existing : null);

  if (mode === "save") {
    writeModelSection(OUT, MODEL, "code", current);
    console.log(`\ncode entry saved for ${MODEL} → ${OUT}`);
  }
}

main().catch(e => { console.error(e); process.exit(1); });
