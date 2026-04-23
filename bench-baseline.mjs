// Shared baseline file I/O. Single-file store across perf + toolcall + multiturn.
// Each caller owns a top-level key ({singleStream,coldStart,concurrent} for perf,
// `toolcall` for the single-turn probe, `multiturn` for the multi-turn probe)
// and merges non-destructively so the three flows don't clobber each other.
// Atomic via tmp+rename.

import { readFileSync, writeFileSync, existsSync, renameSync } from "node:fs";

export function read(path) {
  if (!existsSync(path)) return null;
  try { return JSON.parse(readFileSync(path, "utf-8")); }
  catch (e) {
    console.error(`baseline at ${path} is unreadable (${e.message}) — delete it or run './bench baseline clear'`);
    process.exit(1);
  }
}

export function writeMerge(path, patch) {
  const existing = existsSync(path) ? (read(path) ?? {}) : {};
  const merged = { ...existing, ...patch };
  const tmp = path + ".tmp";
  writeFileSync(tmp, JSON.stringify(merged, null, 2));
  renameSync(tmp, path);
}
