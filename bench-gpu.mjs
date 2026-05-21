// Background GPU telemetry sampler — one nvidia-smi subprocess per probe,
// streamed at 1 Hz (configurable). Aggregates to {avg, max} per field.
//
// Silent no-op when nvidia-smi is unavailable (typical inside a container
// without --gpus passthrough). Probes that integrate this should treat a
// null return from startSampler() as "no telemetry, keep going".
//
// Sampling design:
// - Uses `nvidia-smi -lms <ms>` so we get a stream from a single subprocess
//   instead of spawning every interval. Lower overhead, less jitter.
// - Buffers stdout, parses one CSV row per line. Malformed lines are dropped.
// - On stop, sends SIGTERM and waits briefly for the process to flush.

import { spawn, spawnSync } from "node:child_process";

let _checked = null;

// One-shot probe: is nvidia-smi on PATH and does it actually return data?
// Cached so probe modules can call it freely without re-execing each time.
export function gpuAvailable() {
  if (_checked != null) return _checked;
  try {
    const r = spawnSync("nvidia-smi",
      ["--query-gpu=index", "--format=csv,noheader,nounits"],
      { timeout: 2000, stdio: ["ignore", "pipe", "pipe"] });
    _checked = r.status === 0 && (r.stdout?.toString().trim().length ?? 0) > 0;
  } catch { _checked = false; }
  return _checked;
}

// Fields we sample. Index positions matter — parseRow uses positional access.
const FIELDS = [
  "utilization.gpu",        // % utilization
  "memory.used",            // MiB
  "power.draw",             // W
  "temperature.gpu",        // °C
  "clocks.current.sm",      // MHz
];

function parseRow(line) {
  const parts = line.split(/,\s*/).map(s => s.trim());
  if (parts.length < FIELDS.length) return null;
  const [util, mem, pwr, temp, clk] = parts.map(Number);
  // A row where every field is NaN means nvidia-smi printed an error line;
  // drop it rather than poison the aggregate.
  if (![util, mem, pwr, temp, clk].some(Number.isFinite)) return null;
  return { util, mem, pwr, temp, clk };
}

// Start sampling. Returns a handle to pass to stopSampler, or null if GPU
// telemetry is unavailable. intervalMs defaults to 1000.
export function startSampler({ intervalMs = 1000 } = {}) {
  if (!gpuAvailable()) return null;
  const samples = [];
  let buf = "";
  const child = spawn("nvidia-smi", [
    `--query-gpu=${FIELDS.join(",")}`,
    "--format=csv,noheader,nounits",
    "-lms", String(intervalMs),
  ], { stdio: ["ignore", "pipe", "pipe"] });

  child.stdout.on("data", (d) => {
    buf += d.toString();
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line) continue;
      const row = parseRow(line);
      if (row) samples.push(row);
    }
  });

  // Swallow stderr — `-lms` occasionally emits driver warnings that aren't
  // worth surfacing to the user mid-probe.
  child.stderr.on("data", () => {});

  // If the child dies prematurely we want startSampler to still return a
  // usable handle; stopSampler will just see whatever it captured.
  child.on("error", () => {});
  child.on("exit", () => { child._exited = true; });

  const startedAt = performance.now();
  return { child, samples, intervalMs, startedAt };
}

function aggregate(samples, key) {
  const vals = samples.map(s => s[key]).filter(Number.isFinite);
  if (!vals.length) return { avg: null, max: null };
  const sum = vals.reduce((a, b) => a + b, 0);
  return { avg: sum / vals.length, max: Math.max(...vals) };
}

// Stop sampling and return aggregated stats. Returns null if sampler was
// null (GPU unavailable) or if no samples were captured.
export async function stopSampler(handle) {
  if (!handle) return null;
  const { child, samples, startedAt } = handle;
  if (!child._exited) {
    try { child.kill("SIGTERM"); } catch {}
    // Give nvidia-smi a brief moment to flush its last row.
    await new Promise(r => setTimeout(r, 150));
    if (!child._exited) {
      try { child.kill("SIGKILL"); } catch {}
    }
  }
  if (samples.length === 0) return null;
  return {
    samples:    samples.length,
    durationS:  (performance.now() - startedAt) / 1000,
    util:       aggregate(samples, "util"),     // %
    memMiB:     aggregate(samples, "mem"),      // MiB
    powerW:     aggregate(samples, "pwr"),      // W
    tempC:      aggregate(samples, "temp"),     // °C
    clockMHz:   aggregate(samples, "clk"),      // MHz
  };
}

// Pretty-print a one-line summary for verbose probe output.
export function fmtGpuSummary(gpu) {
  if (!gpu) return "(no GPU telemetry)";
  const f = (v, suffix = "", digits = 0) => v?.avg == null
    ? "—"
    : `${v.avg.toFixed(digits)}${suffix}/${v.max.toFixed(digits)}${suffix}`;
  return `GPU avg/max: util ${f(gpu.util, "%")}  vram ${f(gpu.memMiB, "MiB")}  power ${f(gpu.powerW, "W", 1)}  temp ${f(gpu.tempC, "°C")}`;
}
