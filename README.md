# ollama-bench

A regression harness for local [Ollama](https://ollama.com) models. Catches throughput regressions from configuration changes — not a capability benchmark.

The problem it solves: you update Ollama, swap a model digest, move the runtime from host to container, tweak `num_ctx`, or touch the Docker network topology — and your local agent feels a bit slower. Was it actually slower? By how much? What changed? Without a fixed measurement discipline you can't answer any of that.

## What it measures

Four dimensions, each chosen because it isolates a class of regression:

1. **Single-stream generation** at 3 prompt sizes (short / medium / long ≈ 200 / 2K / 8K tokens). Median over N runs. Every call starts with a per-process nonce so the entire prompt prefix differs from any other call — defeats llama.cpp's prefix cache so prompt-eval numbers reflect real compute, not cache hits.
2. **Cold start.** Forces eviction with `keep_alive: "0s"`, then measures `load_duration` + wall time for the next request. This is the number your user pays when the model has fallen out of VRAM.
3. **Concurrency.** 1 / 2 / 4 / 8 parallel streams. Separates per-stream throughput from aggregate throughput. On a single GPU, scaling here is dominated by `OLLAMA_NUM_PARALLEL`.
4. **Environment snapshot.** Ollama version, model digest + quant, GPU state (name, driver, VRAM, util, temp, power, SM clock), and the Ollama server's `OLLAMA_*` env vars. Stored in the baseline so a regression run diffs *what changed* alongside *how it changed*.

## Quick start

```
node bench.mjs save --model gemma4:26b --host http://localhost:11434
# …change something…
node bench.mjs compare --model gemma4:26b --host http://localhost:11434
```

Output on `compare` flags any cell more than 5% slower than the baseline with `⚠`, and prints an `[environment changes vs baseline]` section listing any scalar that differs — Ollama version, GPU driver, `OLLAMA_NUM_PARALLEL`, etc.

Run `save` after any intentional upgrade (new Ollama, new model, new topology). Run `compare` before and after every other change. A full run is ~2 minutes for a 26B model on a single modern GPU.

## Requirements

- Node 20+ (uses built-in `fetch`). No npm install.
- An Ollama endpoint reachable from the machine you run `bench.mjs` on.
- Optional: `nvidia-smi` on PATH for GPU state capture.
- Optional: `docker` on PATH to read `OLLAMA_*` env vars from a containerized Ollama.

## Options

| flag | default | meaning |
|---|---|---|
| `--model` | `gemma4:26b` | model tag |
| `--host` | `http://localhost:11434` | Ollama base URL |
| `--runs` | `3` | per-cell runs, median reported |
| `--out` | `./baseline.json` | baseline file |
| `--regression-pct` | `5` | flag when slower by more than this % |

## When Ollama is on a private Docker network

If Ollama runs inside Docker with no host port publish (e.g., a compose bridge where only sibling containers have network access), use the `./bench` wrapper — it pipes the script into a sibling container that *can* reach Ollama, runs it there, and copies the baseline back to the host.

```
OLLAMA_BENCH_CONTAINER=my-app ./bench save
OLLAMA_BENCH_CONTAINER=my-app ./bench compare
```

Environment variables:
- `OLLAMA_BENCH_CONTAINER` — the sibling container name (must have Node 20+ and network access to Ollama). Defaults try `openclaw` then `engine-openclaw-1`.
- `OLLAMA_BENCH_REMOTE_DIR` — writable path inside the container (default `/tmp`).

Most users want the direct `node bench.mjs` invocation above; the wrapper is for private-network setups.

## Interpretation notes

- **Per-stream tokens/sec** stays roughly constant as concurrency rises when `OLLAMA_NUM_PARALLEL=1` (Ollama's historical default) — requests are serialized on a single GPU. Aggregate t/s grows sub-linearly. Set `OLLAMA_NUM_PARALLEL` higher and you'll get batching; the concurrency scaling curve will look different. The env snapshot captures this so you know which regime you measured.
- **Noise floor** on a warm, idle machine is typically ±3–4% for throughput and ±1.5% for wall time. Set `--regression-pct` above the noise floor for your hardware. If the tool prints a ⚠ on one cell but the rest of the run looks clean, run `compare` again before panicking.
- **GPU must be idle.** If another process is using the GPU (image generation, game, another model), numbers are garbage. The tool warns when GPU utilization is ≥10% before a run but doesn't refuse.
- **Baselines are machine-specific.** `baseline.json` has meaning only against the machine that produced it — GPU model, VRAM, driver, kernel, Ollama config. Don't commit baselines to a repo that multiple people use. The env-diff output lists hardware changes explicitly if you try.
- **Absolute numbers are indicative, deltas are the product.** This is a regression tool — the headline t/s on any individual run will vary with cooling, clock boost, background noise. What matters is whether the same machine gets slower than *itself*.

## What it doesn't measure

- **Capability.** Tool-call accuracy, JSON schema fidelity, instruction following, factual accuracy. See `bench-toolcall.mjs` for a small capability probe if that's what you want.
- **Very long context.** Measured up to ~8K prompt tokens. Scale past that is model-dependent; add entries to `CTX_SIZES` if you care about 32K/64K/128K.
- **Cross-model comparison.** It tells you whether *your* model got slower. Comparing apples to oranges is a different problem.
- **OpenAI-compat vs. native API.** All measurements use `/api/generate`. If your app uses `/v1/chat/completions` with tool schemas, those have different overhead that this tool won't catch.

## The tool-call probe (bonus)

`bench-toolcall.mjs` is a separate script that checks whether a model can pick the right tool, produce valid JSON matching declared schemas, and decline when no tool fits. It uses the OpenAI-compat endpoint at `/v1/chat/completions`. It's not a regression harness — results are qualitative and it won't flag small changes.

```
# host-native, if Ollama is reachable from the host
node bench-toolcall.mjs --model gemma4:26b -v

# private-network setup (mirrors the ./bench wrapper)
OLLAMA_BENCH_CONTAINER=my-app ./bench-toolcall --model gemma4:26b -v
```

## License

MIT. See `LICENSE`.
