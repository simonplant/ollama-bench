# ollama-bench

A regression harness for local [Ollama](https://ollama.com) models. Catches throughput regressions from configuration changes — not a capability benchmark.

The problem it solves: you update Ollama, swap a model digest, move the runtime from host to container, tweak `num_ctx`, or touch the Docker network topology — and your local agent feels a bit slower. Was it actually slower? By how much? What changed? Without a fixed measurement discipline you can't answer any of that.

## What it measures

Four dimensions, each chosen because it isolates a class of regression:

1. **Single-stream generation** at 4 prompt sizes (tiny / short / medium / long ≈ 20 / 200 / 2K / 8K tokens). Median over N runs. Unique pseudo-random filler per run, so every call is a KV-cache miss — otherwise prompt-eval duration collapses to zero and the numbers lie.
2. **Cold start.** Forces eviction with `keep_alive: "0s"`, then measures `load_duration` + wall time for the next request. This is the number your user pays when the model has fallen out of VRAM.
3. **Concurrency.** 1 / 2 / 4 / 8 parallel streams. Separates per-stream throughput from aggregate throughput — useful because Ollama serializes rather than dynamic-batches, and this lets you see it.
4. **Environment snapshot.** Ollama version, model digest + quant, kernel, hostname. Stored in the baseline file so a regression run diffs *what changed* alongside *how it changed*.

## Workflow

```
./bench save       # runs the full suite, writes baseline.json
./bench compare    # reruns and prints deltas; flags >5% regressions with ⚠
./bench run        # runs without touching baseline
```

Run `save` after any intentional upgrade (new Ollama, new model, new topology). Run `compare` before and after every other change.

Full run is ~2 minutes for a 26B model on a single GPU.

## Requirements

- Node 20+ (uses built-in `fetch`). No npm install.
- An Ollama endpoint reachable from the machine you run `bench.mjs` on.
- `docker` on PATH if you use the `./bench` wrapper to run inside a container.

## Usage directly (no wrapper)

```
node bench.mjs save --model gemma4:26b --host http://localhost:11434 --runs 3
node bench.mjs compare --model gemma4:26b --host http://localhost:11434
```

## Usage via the wrapper

If your Ollama runs on a Docker network not reachable from the host (e.g., a compose-private bridge with no host port publish), use `./bench` — it pipes the script into a sibling container that *can* reach Ollama, runs it there, and copies the baseline back to the host.

```
OLLAMA_BENCH_CONTAINER=my-app ./bench save
OLLAMA_BENCH_CONTAINER=my-app ./bench compare
```

Environment variables:
- `OLLAMA_BENCH_CONTAINER` — the sibling container name (must have Node 20+ and network access to Ollama). Defaults try `openclaw` then `engine-openclaw-1`.
- `OLLAMA_BENCH_REMOTE_DIR` — writable path inside the container (default `/tmp`).

## Options

| flag | default | meaning |
|---|---|---|
| `--model` | `gemma4:26b` | model tag |
| `--host` | `http://ollama:11434` | Ollama base URL |
| `--runs` | `3` | per-cell runs, median reported |
| `--out` | `./baseline.json` | baseline file |
| `--regression-pct` | `5` | flag when slower by more than this % |

## What it doesn't measure

- **Capability.** Tool-call accuracy, JSON schema fidelity, instruction following, factual accuracy. See `bench-toolcall.mjs` for a small capability probe if that's what you want.
- **Very long context.** Measured up to ~8K prompt tokens. Scale past that is model-dependent; add more `CTX_SIZES` entries if you care.
- **Cross-model comparison.** It tells you whether *your* model got slower. Comparing apples to oranges is a different problem.

## Interpretation notes

- **Per-stream tokens/sec** stays roughly constant as concurrency rises because Ollama runs requests sequentially on a single GPU. **Aggregate tokens/sec** grows sub-linearly. This is Ollama's design, not a bug — plan for it if you need multi-user throughput on one GPU.
- **Noise floor** on a warm, idle machine is typically ±3–4% for throughput and ±1.5% for wall time. Set `--regression-pct` above the noise floor for your hardware.
- **First calls after `./bench save`** on a compare run tend to run slightly hot (cache effects from the save). If you want a clean compare, wait 30s or re-save immediately before comparing.

## The tool-call probe (bonus)

`bench-toolcall.mjs` is a separate ~200-case script that checks whether a model can pick the right tool, produce valid JSON matching declared schemas, and decline when no tool fits. It uses the OpenAI-compat endpoint at `/v1/chat/completions`. It's not a regression harness — results are qualitative and it won't flag small changes.

```
node bench-toolcall.mjs --model gemma4:26b -v
```

## License

MIT. See `LICENSE`.
