// Background CPU + system-RAM sampler. Reads /proc/stat and /proc/meminfo
// directly — no subprocess, no dependencies. Works inside any Linux container
// (host-wide stats by default in Docker, since /proc isn't namespaced for
// these files unless lxcfs is involved).
//
// Sister to bench-gpu.mjs: same start/stop/aggregate shape, separate domain.
// Probes that integrate this should treat a null return as "not Linux,
// keep going" — the rest of the report still works.

import { readFileSync } from "node:fs";

let _checked = null;

export function sysAvailable() {
  if (_checked != null) return _checked;
  try {
    readFileSync("/proc/stat", "utf-8");
    readFileSync("/proc/meminfo", "utf-8");
    _checked = true;
  } catch { _checked = false; }
  return _checked;
}

// Aggregate CPU jiffies from the first "cpu " line in /proc/stat.
// Fields: user nice system idle iowait irq softirq steal guest guest_nice.
// idle-share = (idle + iowait) / total. cpu% = 100 * (1 - idle-share).
function readCpu() {
  const line = readFileSync("/proc/stat", "utf-8").split("\n", 1)[0];
  const parts = line.trim().split(/\s+/).slice(1).map(Number);
  if (parts.length < 5) return null;
  const idle = parts[3] + parts[4]; // idle + iowait
  const total = parts.reduce((a, b) => a + b, 0);
  return { idle, total };
}

function readMem() {
  const text = readFileSync("/proc/meminfo", "utf-8");
  const get = (key) => {
    const m = text.match(new RegExp(`^${key}:\\s+(\\d+)\\s*kB`, "m"));
    return m ? Number(m[1]) : null;
  };
  const total = get("MemTotal");
  const avail = get("MemAvailable");
  if (total == null || avail == null) return null;
  return {
    totalMiB: total / 1024,
    usedMiB: (total - avail) / 1024,
  };
}

export function startSysSampler({ intervalMs = 1000 } = {}) {
  if (!sysAvailable()) return null;
  const samples = []; // { cpuPct, usedMiB }
  let prev = readCpu();
  const totalMiB = readMem()?.totalMiB ?? null;

  const timer = setInterval(() => {
    const cur = readCpu();
    const mem = readMem();
    if (cur && prev && mem) {
      const dTotal = cur.total - prev.total;
      const dIdle  = cur.idle  - prev.idle;
      const cpuPct = dTotal > 0 ? 100 * (1 - dIdle / dTotal) : null;
      if (cpuPct != null) samples.push({ cpuPct, usedMiB: mem.usedMiB });
    }
    prev = cur;
  }, intervalMs);
  // Avoid blocking process exit on a stray sampler.
  timer.unref?.();

  return { timer, samples, totalMiB, startedAt: performance.now() };
}

function aggregate(samples, key) {
  const vals = samples.map(s => s[key]).filter(Number.isFinite);
  if (!vals.length) return { avg: null, max: null };
  const sum = vals.reduce((a, b) => a + b, 0);
  return { avg: sum / vals.length, max: Math.max(...vals) };
}

export function stopSysSampler(handle) {
  if (!handle) return null;
  clearInterval(handle.timer);
  if (handle.samples.length === 0) return null;
  return {
    samples:   handle.samples.length,
    durationS: (performance.now() - handle.startedAt) / 1000,
    totalMiB:  handle.totalMiB,
    cpuPct:    aggregate(handle.samples, "cpuPct"),
    ramMiB:    aggregate(handle.samples, "usedMiB"),
  };
}

export function fmtSysSummary(sys) {
  if (!sys) return "(no sys telemetry)";
  const f = (v, suffix = "", digits = 0) => v?.avg == null
    ? "—"
    : `${v.avg.toFixed(digits)}${suffix}/${v.max.toFixed(digits)}${suffix}`;
  const totalNote = sys.totalMiB ? ` of ${sys.totalMiB.toFixed(0)}MiB` : "";
  return `CPU/RAM avg/max: cpu ${f(sys.cpuPct, "%")}  ram ${f(sys.ramMiB, "MiB")}${totalNote}`;
}
