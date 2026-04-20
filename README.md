# ollama-bench

A regression harness for local [Ollama](https://ollama.com) models, plus two small tool-calling capability probes. Designed to catch throughput regressions from configuration changes on a single machine over time — not to rank models against each other.

The problem it solves: you update Ollama, swap a model digest, move the runtime from host to container, tweak `num_ctx`, or touch the Docker network topology — and your local agent feels a bit slower. Was it actually slower? By how much? What changed? Without a fixed measurement discipline you can't answer any of that.

## Quick start

```
./bench                    # First run: save baseline + show throughput numbers.
                           # Every run after: compare against baseline, flag regressions.
```

That's it. The default does the right thing — no subcommand, no flags, no `save` vs `compare` to remember. If you change anything (Ollama version, `OLLAMA_NUM_PARALLEL`, topology, GPU driver, model digest), re-run `./bench` and read the deltas.

Full CLI:

```
./bench                    Smart throughput run (save first time, compare after).
./bench perf [mode]        Explicit mode: save | compare | run. Default smart.
./bench toolcall           Single-turn tool-call accuracy probe.
./bench multiturn          Multi-turn probe — what happens after a tool returns.
./bench all                perf (smart) + toolcall + multiturn, sequentially.
./bench baseline [show]    Print baseline summary.
./bench baseline clear     Delete baseline.json.
./bench --help             Usage.
```

Flags: `--model` (default `gemma4:26b`), `--host` (default `http://ollama:11434`), `--runs` (perf, default 3), `--out` (baseline path, default `./baseline.json`), `--regression-pct` (default 5), `-v` (verbose, probes only).

## What the throughput run measures

Four dimensions, each chosen because it isolates a class of regression:

1. **Single-stream generation** at 3 prompt sizes (short / medium / long ≈ 200 / 2K / 8K tokens) plus a long-gen cell (short prompt, 1024 output tokens to catch KV-cache and sampler-at-depth regressions). Median over N runs. Every call starts with a per-process nonce so the entire prompt prefix differs from any other call — defeats llama.cpp's prefix cache so prompt-eval numbers reflect real compute, not cache hits.
2. **Cold start.** Forces eviction with `keep_alive: "0s"`, then measures `load_duration` + wall time for the next request. The number your user pays when the model has fallen out of VRAM.
3. **Concurrency.** 1 / 2 / 4 / 8 parallel streams at a medium prompt × 64-token gen (edit `CONCURRENT_CELL` in `bench.mjs` to change the cell). Three per-row metrics: `e2e t/s` = end-to-end wall-time throughput (prompt-eval + network included); `decode t/s` = aggregate pure-decoder throughput (`sum(eval_count) / max(eval_duration)`); `per-stream t/s` = median of per-request decode speed. At parallel=1, `decode t/s` should match `per-stream t/s` within noise — a gap means prompt-eval or network is dominating. Scaling on a single GPU is dominated by `OLLAMA_NUM_PARALLEL`.
4. **Environment snapshot.** Ollama version, model digest + quant, GPU state (name, driver, VRAM, util, temp, power, SM clock), and the Ollama server's `OLLAMA_*` env vars. Stored in the baseline so a regression run diffs *what changed* alongside *how it changed*.

On `compare` (and the smart default when a baseline exists), any cell more than 5% slower than the baseline gets a `⚠`, and an `[environment changes vs baseline]` section lists any scalar that differs — Ollama version, GPU driver, `OLLAMA_NUM_PARALLEL`, etc. Baselines include per-cell noise floors (2σ), so the threshold auto-widens on naturally noisy cells.

A full run is ~2 minutes for a 26B model on a single modern GPU.

## Tool-calling probes

Separate from throughput. These characterize a model's *capability*, not its speed. Useful when picking a model or diagnosing agent loops.

### `./bench toolcall` — single-turn

22 cases checking tool selection, schema fidelity, and correct declination when no tool fits. Three categories:
- `simple` — one obvious tool should be called
- `multiple` — disambiguate across tools
- `relevance` — no tool should be called

Passes per category. Add `-v` for per-case output.

### `./bench multiturn` — two-turn with fabricated tool results

14 cases that test what happens **after** a tool returns. The user prompt triggers a tool call; the probe injects a fabricated tool result; the model's next turn is scored. This catches the real failure mode local models hit:

- **Synthesis** — tool returned useful data; model should summarize.
- **Empty** — tool returned `[]`; model should say "no results", NOT re-call.
- **Error** — tool returned `{error: ...}`; model should surface or handle.
- **Chain** — tool 1 succeeded; model should legitimately call tool 2.

Scored failures are specific: `LOOP: re-called X with identical args`, `LOOP: re-called X with different args`, `unexpected tool call: Y`, `expected synthesis, got 0-char content`. When you switch models (qwen, llama, mistral), these categories show where each model breaks.

## Requirements

- Node 20+ (uses built-in `fetch`). No npm install.
- An Ollama endpoint reachable from the machine you run on.
- Optional: `nvidia-smi` on PATH for GPU state capture.
- Optional: `docker` on PATH to read `OLLAMA_*` env vars from a containerized Ollama.

## Running when Ollama is on a private Docker network

If Ollama runs inside Docker with no host port publish (e.g., a compose bridge where only sibling containers have network access), use the `./bench` wrapper — it pipes the scripts into a sibling container that *can* reach Ollama, runs them there, and copies the baseline back to the host. All subcommands work identically through the wrapper.

```
OLLAMA_BENCH_CONTAINER=my-app ./bench
OLLAMA_BENCH_CONTAINER=my-app ./bench toolcall
```

Environment variables:
- `OLLAMA_BENCH_CONTAINER` — sibling container name (needs Node 20+ and network access to Ollama). Defaults try `openclaw` then `engine-openclaw-1`.
- `OLLAMA_BENCH_REMOTE_DIR` — writable path inside the container (default `/tmp`).
- `OLLAMA_BENCH_MACHINE_ID` — stable machine fingerprint seed. The wrapper auto-populates it from `/etc/machine-id` (or `/var/lib/dbus/machine-id`) so the baseline survives `docker compose up --force-recreate` of the sibling container. Set it manually if your host doesn't expose one.

If the host can reach Ollama directly, skip the wrapper and run `node bench.mjs` instead.

## Interpretation notes

- **Per-stream tokens/sec** stays roughly constant as concurrency rises when `OLLAMA_NUM_PARALLEL=1` (Ollama's historical default) — requests are serialized on a single GPU. Aggregate t/s grows sub-linearly. Set `OLLAMA_NUM_PARALLEL` higher and you'll get batching; the concurrency scaling curve will look different. The env snapshot captures this so you know which regime you measured.
- **Noise floor** on a warm, idle machine is typically ±3–4% for throughput and ±1.5% for wall time. The baseline measures per-cell noise directly and widens the threshold to `max(--regression-pct, 2× cv%)`. If the tool prints a ⚠ on one cell but the rest of the run looks clean, run again before panicking.
- **GPU must be idle.** If another process is using the GPU (image generation, game, another model), numbers are garbage. The tool warns when GPU utilization is ≥10% before a run but doesn't refuse.
- **Baselines are machine-specific.** `baseline.json` has meaning only against the machine that produced it — GPU model, VRAM, driver, kernel, Ollama config. The tool fingerprints the machine and warns + skips compare if you re-run against a baseline from a different host.
- **Absolute numbers are indicative, deltas are the product.** This is a regression tool — the headline t/s on any individual run will vary with cooling, clock boost, background noise. What matters is whether the same machine gets slower than *itself*.

## What it doesn't measure

- **Capability throughput (tok/s on agentic workloads).** The perf harness uses `/api/generate` with fixed prompts. If your app uses `/v1/chat/completions` with tool schemas, that path has different overhead that perf won't catch — the tool-call probes exercise that path but score correctness, not speed.
- **Very long context.** Measured up to ~8K prompt tokens. Scale past that is model-dependent; edit `CTX_SIZES` in `bench.mjs` if you care about 32K/64K/128K.
- **Cross-model comparison.** It tells you whether *your* model got slower. Comparing apples to oranges is a different problem.

## License

MIT. See `LICENSE`.
