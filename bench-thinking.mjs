// Thinking-mode detection + per-call budget adjustment for capability probes.
//
// Ollama exposes a `capabilities` array on /api/show; thinking-capable models
// (qwen3.x, gpt-oss, nemotron-3, gemma4, deepseek-r1) include "thinking". For
// those, calls to /api/generate must (a) opt in with `think: true` so reasoning
// goes into a separate `thinking` field instead of eating the visible
// `response`, and (b) raise `num_predict` enough that reasoning + answer fit.
//
// Env overrides (compose with bench.mjs `OLLAMA_BENCH_THINK`):
//   OLLAMA_BENCH_THINK=0  → force off everywhere
//   OLLAMA_BENCH_THINK=1  → force on (subject to model support)
// Default behaviour is auto: on iff the model reports the capability.

const cache = new Map();

export async function detectThinking(host, model) {
  const key = `${host}::${model}`;
  if (cache.has(key)) return cache.get(key);
  let supports = false;
  try {
    const r = await fetch(`${host}/api/show`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model }),
    });
    if (r.ok) {
      const j = await r.json();
      supports = Array.isArray(j.capabilities) && j.capabilities.includes("thinking");
    }
  } catch {
    // Treat unreachable /api/show as non-thinking; the request will fail loudly
    // later if the model itself is also unreachable.
  }
  cache.set(key, supports);
  return supports;
}

function envFlag(name) {
  const v = process.env[name];
  if (v == null || v === "") return null;
  return v === "0" || v.toLowerCase() === "false" ? false : true;
}

// Returns the `think` value to put in the request body, plus the suggested
// num_predict. `baseNumPredict` is the probe's default (used when thinking is
// off); `thinkingNumPredict` is the larger budget for when reasoning is on.
export async function thinkingParams(host, model, baseNumPredict, thinkingNumPredict) {
  const override = envFlag("OLLAMA_BENCH_THINK");
  const supports = await detectThinking(host, model);
  const want = override == null ? supports : (override && supports);
  return {
    think: want,
    numPredict: want ? thinkingNumPredict : baseNumPredict,
    supports,
  };
}
