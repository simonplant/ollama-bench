// Shared baseline file I/O. Schema v2: per-model storage so one machine can
// hold a "league table" of every model that's been benched on it. Reads
// migrate v1 (flat) baselines on the fly so nothing breaks.
//
// v2 shape:
//   {
//     schemaVersion: 2,
//     machine: { machineId, hostMachineId, hostname, kernel, gpu },
//     models: {
//       "<model:tag>": {
//         savedAt: ISO,
//         perf:      { env, singleStream, coldStart, concurrent },
//         toolcall:  { ...probe results... },
//         multiturn: { ...probe results... }
//       }
//     }
//   }
//
// Atomicity preserved: every mutation goes through tmp+rename.

import { readFileSync, writeFileSync, existsSync, renameSync, unlinkSync } from "node:fs";

export const SCHEMA_VERSION = 2;

// Read raw file, migrate v1 → v2 on the fly. Returns null if missing.
// Migration is lossless and idempotent.
export function read(path) {
  if (!existsSync(path)) return null;
  let raw;
  try { raw = JSON.parse(readFileSync(path, "utf-8")); }
  catch (e) {
    console.error(`baseline at ${path} is unreadable (${e.message}) — delete it or run './bench baseline clear'`);
    process.exit(1);
  }
  return migrate(raw);
}

// v1 → v2 migration. v1 had a flat shape with one model's data at the top
// level; v2 keys by model so multiple models coexist. Migration preserves
// every v1 field — machine fields go to machine{}, model fields go under
// models[<tag>].
function migrate(b) {
  if (!b || typeof b !== "object") return b;
  if (b.schemaVersion === SCHEMA_VERSION || b.models) return b;
  const tag = b?.env?.model;
  if (!tag) return b; // nothing to migrate; leave caller to error if it cares

  const machine = {
    machineId:     b.env.machineId     ?? null,
    hostMachineId: b.env.hostMachineId ?? null,
    hostname:      b.env.hostname      ?? null,
    kernel:        b.env.kernel        ?? null,
    gpu:           b.env.gpu           ?? null,
  };
  // Per-model env split: everything in v1.env that isn't machine-anchored
  // belongs to the model entry's perf.env (timestamp, model digest, ollama
  // version, server env, etc).
  const perfEnv = {
    timestamp:       b.env.timestamp       ?? null,
    model:           b.env.model,
    modelDigest:     b.env.modelDigest     ?? null,
    modelParams:     b.env.modelParams     ?? null,
    modelQuant:      b.env.modelQuant      ?? null,
    ollamaVersion:   b.env.ollamaVersion   ?? null,
    ollamaServerEnv: b.env.ollamaServerEnv ?? null,
    host:            b.env.host            ?? null,
    runs:            b.env.runs            ?? null,
    node:            b.env.node            ?? null,
  };
  const entry = { savedAt: b.env.timestamp ?? new Date().toISOString() };
  if (b.singleStream || b.coldStart || b.concurrent) {
    entry.perf = {
      env: perfEnv,
      singleStream: b.singleStream ?? null,
      coldStart:    b.coldStart    ?? null,
      concurrent:   b.concurrent   ?? null,
    };
  }
  if (b.toolcall)  entry.toolcall  = b.toolcall;
  if (b.multiturn) entry.multiturn = b.multiturn;
  return {
    schemaVersion: SCHEMA_VERSION,
    machine,
    models: { [tag]: entry },
  };
}

// Get one model's slice for one section ("perf" | "toolcall" | "multiturn").
// Returns null when absent.
export function getModelSection(path, modelTag, section) {
  const b = read(path);
  return b?.models?.[modelTag]?.[section] ?? null;
}

// Get the machine slice (anchored hardware/host facts). Returns null when
// the file doesn't exist.
export function getMachine(path) {
  const b = read(path);
  return b?.machine ?? null;
}

// Get full model entry: { savedAt, perf?, toolcall?, multiturn? }.
export function getModel(path, modelTag) {
  const b = read(path);
  return b?.models?.[modelTag] ?? null;
}

// List every model entry. Each row: { tag, savedAt, perf?, toolcall?, multiturn? }.
// Used by the league view.
export function listModels(path) {
  const b = read(path);
  if (!b?.models) return [];
  return Object.entries(b.models).map(([tag, entry]) => ({ tag, ...entry }));
}

// Write/update one (model, section) slice non-destructively. Other models
// and other sections of the same model are preserved. Atomic.
//
// machinePatch (optional) overwrites the machine slice fields supplied —
// callers pass it on perf save so the league always reflects the box that
// produced the latest perf numbers.
export function writeModelSection(path, modelTag, section, payload, machinePatch) {
  const existing = read(path) ?? { schemaVersion: SCHEMA_VERSION, machine: {}, models: {} };
  // Defensive: migrate() may have returned a partially-shaped object if the
  // input was unrecognized; rebuild missing scaffolding.
  if (!existing.models)  existing.models  = {};
  if (!existing.machine) existing.machine = {};
  const entry = existing.models[modelTag] ?? {};
  entry[section] = payload;
  entry.savedAt = new Date().toISOString();
  existing.models[modelTag] = entry;
  if (machinePatch) existing.machine = { ...existing.machine, ...machinePatch };
  existing.schemaVersion = SCHEMA_VERSION;
  const tmp = path + ".tmp";
  writeFileSync(tmp, JSON.stringify(existing, null, 2));
  renameSync(tmp, path);
}

// Remove a single model entry. Returns true if it existed and was removed.
// File itself stays put (use unlinkSync directly to nuke everything).
export function removeModel(path, modelTag) {
  const b = read(path);
  if (!b?.models?.[modelTag]) return false;
  delete b.models[modelTag];
  const tmp = path + ".tmp";
  writeFileSync(tmp, JSON.stringify(b, null, 2));
  renameSync(tmp, path);
  return true;
}

// Convenience: delete the whole baseline file. No-op if missing.
export function clearAll(path) {
  if (existsSync(path)) unlinkSync(path);
}
