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

// Sampling profile for a request. Thinking models degrade under greedy
// decoding — repetition loops and degenerate output, per Qwen3 and DeepSeek-R1
// guidance — so when thinking is on we use the widely-recommended reasoning
// profile (temp 0.6 / top_p 0.95 / top_k 20) instead of temp 0. Non-thinking
// runs stay greedy for deterministic grading. OLLAMA_BENCH_GREEDY=1 forces
// greedy everywhere for fully reproducible runs (at some accuracy cost on
// thinking models). Spread into the request `options` (native) or body (OpenAI).
export function samplingFor(think) {
  if (process.env.OLLAMA_BENCH_GREEDY === "1") return { temperature: 0 };
  return think
    ? { temperature: 0.6, top_p: 0.95, top_k: 20 }
    : { temperature: 0 };
}

// Strip inline reasoning blocks from a model's visible output. With think /
// reasoning_effort enabled Ollama returns reasoning in a separate field and
// `response`/`content` is clean — but some models/templates leak
// <think>…</think> (or <thinking>/<reasoning>) into the visible text, and a
// model that always reasons (e.g. deepseek-r1) does so even when thinking is
// forced off. Answer extractors run on the post-strip text so a leaked chain
// can't poison parsing (e.g. grabbing a number or option letter from the
// reasoning instead of the final answer). No-op when no tags are present.
export function stripThinking(text) {
  if (!text) return text ?? "";
  return String(text)
    // Closed reasoning blocks anywhere in the text.
    .replace(/<(think|thinking|reasoning)>[\s\S]*?<\/\1>/gi, "")
    // Dangling open block (reasoning truncated by num_predict, never closed):
    // drop from the tag to end — there's no final answer past it anyway.
    .replace(/<(think|thinking|reasoning)>[\s\S]*$/i, "")
    .trim();
}
