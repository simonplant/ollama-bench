#!/usr/bin/env node
/**
 * `./bench doctor` — system audit for GPU, Ollama server, and host config.
 *
 * Each check emits one status with a fix command when actionable:
 *   ✓ ok       configured well
 *   ⚠ warn     suboptimal, advisory
 *   ✗ fail     broken or silently degraded; non-zero exit
 *   · info     neutral
 *   ? unknown  couldn't probe (e.g. sysfs unreadable from inside container)
 *
 * Read-only — never mutates GPU, Ollama, or host state.
 *
 * Host-side probes (CPU governor, extended nvidia-smi fields, swap) read
 * env vars injected by the ./bench wrapper when running Docker-sibling,
 * since the sibling can't see host sysfs or have nvidia-smi on PATH.
 * Falls back to local commands for direct-on-host invocation.
 */

import { execSync } from "node:child_process";

const args = process.argv.slice(2);
const arg = (n, fb) => { const i = args.indexOf(n); return i >= 0 ? args[i + 1] : fb; };
const HOST  = arg("--host",  "http://ollama:11434");
const MODEL = arg("--model", "gemma4:26b");

const checks = [];
const add = (section, status, title, detail, fix) => checks.push({ section, status, title, detail, fix });
const ok      = (s, t, d)      => add(s, "ok",      t, d);
const warn    = (s, t, d, fix) => add(s, "warn",    t, d, fix);
const fail    = (s, t, d, fix) => add(s, "fail",    t, d, fix);
const info    = (s, t, d)      => add(s, "info",    t, d);
const unknown = (s, t, d)      => add(s, "unknown", t, d);

function tryExec(cmd) {
  try { return execSync(cmd, { timeout: 5000, stdio: ["ignore", "pipe", "ignore"] }).toString().trim(); }
  catch { return null; }
}

// Doctor's job is to diagnose a misconfigured system — including one where
// Ollama itself is hung. Bare `fetch()` waits indefinitely; a short deadline
// turns "hang forever" into "fail the API-reachable check and report".
async function fetchWithTimeout(url, opts = {}, ms = 10_000) {
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), ms);
  try {
    return await fetch(url, { ...opts, signal: ac.signal });
  } finally {
    clearTimeout(timer);
  }
}

// ── GPU ──────────────────────────────────────────────────────────────────────
// Extended field set — superset of what perf bench captures. Mirror any field
// changes in the ./bench wrapper's OLLAMA_BENCH_GPU_EXT_CSV producer.
const GPU_FIELDS = [
  "name", "driver_version",
  "persistence_mode",
  "power.default_limit", "power.max_limit",
  "clocks_throttle_reasons.active",
];

function checkGpu() {
  const csv = process.env.OLLAMA_BENCH_GPU_EXT_CSV
    || tryExec(`nvidia-smi --query-gpu=${GPU_FIELDS.join(",")} --format=csv,noheader,nounits`);
  if (!csv) {
    unknown("gpu", "probe", "nvidia-smi unavailable and no host-injected data (OLLAMA_BENCH_GPU_EXT_CSV unset)");
    return;
  }
  const [name, driver, persistence, powerDefault, powerMax, throttleHex] =
    csv.split("\n")[0].split(",").map(s => s.trim());

  info("gpu", "device", `${name} (driver ${driver})`);

  if (/^Enabled$/i.test(persistence)) {
    ok("gpu", "persistence mode", "enabled");
  } else {
    warn("gpu", "persistence mode",
      `"${persistence}" — driver unloads between idle periods, adding ~1–3s to first request`,
      "sudo nvidia-smi -pm 1");
  }

  const pdef = parseFloat(powerDefault), pmax = parseFloat(powerMax);
  if (isFinite(pdef) && isFinite(pmax) && pmax > 0) {
    if (pdef < pmax * 0.98) {
      const pctCap = Math.round((1 - pdef / pmax) * 100);
      warn("gpu", "power limit",
        `capped at ${pdef}W of ${pmax}W max (${pctCap}% headroom unused)`,
        `sudo nvidia-smi -pl ${Math.round(pmax)}`);
    } else {
      ok("gpu", "power limit", `${pdef}W (at max)`);
    }
  }

  const throttle = parseInt(throttleHex || "0", 16);
  if (!isNaN(throttle)) {
    // NVML bits: 0x8=hw_slowdown, 0x40=hw_thermal, 0x80=hw_power_brake.
    // Idle/app-clock bits (0x1, 0x2) are normal at rest — don't flag them.
    const hw = throttle & 0xC8;
    if (hw) {
      fail("gpu", "throttle state",
        `hardware throttle active (mask 0x${hw.toString(16)}): check cooling, PSU, power limit`);
    } else {
      ok("gpu", "throttle state", "no hardware throttle");
    }
  }
}

// ── Ollama ───────────────────────────────────────────────────────────────────
async function checkOllama() {
  let ver = null;
  try {
    const r = await fetchWithTimeout(`${HOST}/api/version`);
    if (r.ok) ver = (await r.json()).version;
  } catch (e) {
    const reason = e.name === "AbortError" ? "timed out after 10s" : e.message;
    fail("ollama", "API reachable",
      `cannot reach ${HOST}/api/version (${reason})`,
      "check --host value and container networking");
    return;
  }
  if (!ver) {
    fail("ollama", "API reachable", `${HOST} responded but no version payload`, null);
    return;
  }
  info("ollama", "server version", ver);

  // Server env — prefer host-injected (Docker-sibling flow), fall back to
  // local docker inspect, finally the process environment.
  let env = null;
  const envJson = process.env.OLLAMA_BENCH_SERVER_ENV_JSON
    || tryExec("docker inspect ollama --format '{{json .Config.Env}}' 2>/dev/null");
  if (envJson) {
    try {
      const arr = JSON.parse(envJson);
      env = Object.fromEntries(
        arr.filter(kv => /^OLLAMA_/.test(kv))
           .map(kv => { const i = kv.indexOf("="); return [kv.slice(0, i), kv.slice(i + 1)]; })
      );
    } catch {}
  }
  if (!env) {
    const keys = ["OLLAMA_NUM_PARALLEL","OLLAMA_MAX_LOADED_MODELS","OLLAMA_KEEP_ALIVE",
                  "OLLAMA_FLASH_ATTENTION","OLLAMA_KV_CACHE_TYPE","OLLAMA_CONTEXT_LENGTH"];
    env = Object.fromEntries(keys.map(k => [k, process.env[k] ?? null]).filter(([,v]) => v !== null));
    if (!Object.keys(env).length) env = null;
  }

  if (!env) {
    unknown("ollama", "server env",
      "couldn't read OLLAMA_* env (no docker socket + no host-injected data) — skipping env-dependent checks");
    return;
  }

  const np = env.OLLAMA_NUM_PARALLEL;
  info("ollama", "OLLAMA_NUM_PARALLEL", np ?? "unset (server default)");

  const fa = env.OLLAMA_FLASH_ATTENTION;
  const kv = env.OLLAMA_KV_CACHE_TYPE;
  if (kv === "q4_0" && fa !== "1") {
    fail("ollama", "KV cache + flash attention",
      "OLLAMA_KV_CACHE_TYPE=q4_0 requires OLLAMA_FLASH_ATTENTION=1; otherwise Ollama falls back to f16 silently",
      "set OLLAMA_FLASH_ATTENTION=1");
  } else if (kv && fa === "1") {
    ok("ollama", "KV cache + flash attention", `${kv} with flash attention`);
  } else if (kv === "q8_0") {
    info("ollama", "KV cache", `q8_0 (half the memory of f16, minimal quality impact)`);
  } else if (kv) {
    info("ollama", "KV cache", kv);
  } else {
    info("ollama", "KV cache", "default f16 — set OLLAMA_KV_CACHE_TYPE=q8_0 to halve KV memory use");
  }

  const ka = env.OLLAMA_KEEP_ALIVE;
  if (!ka || ka === "0" || ka === "0s") {
    warn("ollama", "OLLAMA_KEEP_ALIVE",
      `"${ka ?? "unset"}" — model unloads aggressively, every first-prompt pays model-load latency`,
      "OLLAMA_KEEP_ALIVE=24h for dev; -1 to pin permanently");
  } else {
    ok("ollama", "OLLAMA_KEEP_ALIVE", ka);
  }

  info("ollama", "OLLAMA_CONTEXT_LENGTH", env.OLLAMA_CONTEXT_LENGTH ?? "unset (server default)");

  // Model presence check — not fatal, just informative.
  try {
    const r = await fetchWithTimeout(`${HOST}/api/tags`);
    if (r.ok) {
      const j = await r.json();
      const hit = (j.models ?? []).find(m => m.name === MODEL || m.model === MODEL);
      if (!hit) {
        warn("ollama", "model",
          `${MODEL} not in /api/tags — first use will pull the model`,
          `ollama pull ${MODEL}`);
      } else {
        info("ollama", "model", `${MODEL} (digest ${(hit.digest ?? "").slice(0, 12)})`);
      }
    }
  } catch {}
}

// ── Host ─────────────────────────────────────────────────────────────────────
function checkHost() {
  // CPU governor — prefer injected (newline-separated one value per core),
  // fall back to sysfs glob via shell.
  const governorRaw = process.env.OLLAMA_BENCH_CPU_GOVERNOR
    || tryExec("cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor 2>/dev/null");
  if (!governorRaw) {
    unknown("host", "CPU governor", "sysfs unreadable (likely running inside container without host sysfs mount)");
  } else {
    const govs = [...new Set(governorRaw.split("\n").map(s => s.trim()).filter(Boolean))];
    if (govs.length === 1 && govs[0] === "performance") {
      ok("host", "CPU governor", "performance (all cores)");
    } else {
      warn("host", "CPU governor",
        govs.length === 1 ? `all cores: ${govs[0]}` : `mixed: ${govs.join(", ")}`,
        "sudo cpupower frequency-set -g performance");
    }
  }

  const swapRaw = process.env.OLLAMA_BENCH_SWAP
    || tryExec("free -m | awk '/^Swap:/ {print $2, $3}'");
  if (swapRaw) {
    const [total, used] = swapRaw.trim().split(/\s+/).map(s => parseInt(s, 10));
    if (!isFinite(total) || total === 0) {
      info("host", "swap", "disabled");
    } else if (isFinite(used) && used > 100) {
      warn("host", "swap",
        `${used} MiB in use (of ${total} MiB) — if model data pages, gen t/s will tank`,
        "swapon --show; swapoff -a if RAM is sufficient");
    } else {
      ok("host", "swap", `${used ?? 0}/${total} MiB used`);
    }
  }
}

// ── Report ───────────────────────────────────────────────────────────────────
const ICONS = { ok: "✓", warn: "⚠", fail: "✗", info: "·", unknown: "?" };
const SECTION_ORDER = ["gpu", "ollama", "host"];

function printReport() {
  const bySection = {};
  for (const c of checks) (bySection[c.section] ||= []).push(c);

  for (const section of SECTION_ORDER) {
    const rows = bySection[section];
    if (!rows) continue;
    console.log(`\n[${section}]`);
    for (const c of rows) {
      const icon = ICONS[c.status] ?? "?";
      console.log(`  ${icon} ${c.title}: ${c.detail ?? ""}`);
      if (c.fix) console.log(`     fix: ${c.fix}`);
    }
  }

  const warns = checks.filter(c => c.status === "warn").length;
  const fails = checks.filter(c => c.status === "fail").length;
  if (fails === 0 && warns === 0) console.log("\nall clear");
  else console.log(`\n${fails} fail · ${warns} warn`);

  // Exit non-zero only on real failures — warns are advisory.
  if (fails > 0) process.exit(1);
}

// ── Main ─────────────────────────────────────────────────────────────────────
console.log(`doctor: model=${MODEL} host=${HOST}`);
checkGpu();
await checkOllama();
checkHost();
printReport();
