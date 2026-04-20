# ollama-bench

A regression harness for local [Ollama](https://ollama.com) models, plus two tool-calling capability probes. Catches throughput regressions from configuration changes on a single machine over time — not a cross-model ranking tool.

The problem it solves: you update Ollama, swap a model digest, rebuild the container, tweak `num_ctx`, or touch the Docker network topology — and your local agent feels a bit slower. Was it actually slower? By how much? What changed? Without fixed measurement discipline you can't answer any of that.

Assumes a Docker-native setup: Ollama runs inside a container on a user-defined bridge network and is only reachable from sibling containers. The `./bench` wrapper pipes the harness into a sibling, runs it there, and syncs the baseline back to the host. If you run Ollama natively on the host instead, see [Direct host invocation](#direct-host-invocation) below.

## Quickstart

```bash
cd ~/bench
./bench                 # first run saves baseline; every run after compares
./bench all             # perf + both tool probes, end-to-end
./bench toolcall -v     # per-case output when debugging a capability
./bench baseline show   # what's in the current baseline
```

That's the whole workflow. After you change anything meaningful — Ollama version, `OLLAMA_NUM_PARALLEL`, model digest, GPU driver, Docker network — rerun `./bench` and read the deltas. To start fresh: `./bench baseline clear`.

The wrapper auto-detects a sibling container named `openclaw` or `engine-openclaw-1` (override with `OLLAMA_BENCH_CONTAINER=<name>`). The machine fingerprint survives `docker compose up --force-recreate` because the wrapper reads `/etc/machine-id` from the host and passes it in.

## CLI

```
./bench                      Smart throughput run (save first time, compare after).
./bench perf [mode]          Explicit mode: save | compare | run. Default smart.
./bench toolcall             Single-turn tool-call accuracy probe (22 cases).
./bench multiturn            Multi-turn probe — what happens after a tool returns (14 cases).
./bench all                  perf (smart) → toolcall → multiturn, sequentially.
./bench baseline [show]      Print baseline summary.
./bench baseline clear       Delete baseline.json.
./bench --help               Usage.
```

Flags (any order, any subcommand):
- `--model <tag>` — default `gemma4:26b`
- `--host <url>` — default `http://ollama:11434` (Docker service name; reachable from the sibling container)
- `--runs <n>` — per-cell runs for perf, default 3
- `--out <path>` — baseline file, default `./baseline.json`
- `--regression-pct <n>` — regression threshold in %, default 5
- `-v`, `--verbose` — per-case output for toolcall / multiturn

## Container setup

Environment variables the wrapper respects:
- `OLLAMA_BENCH_CONTAINER` — sibling container name (needs Node 20+ and network access to Ollama). Auto-tries `openclaw` then `engine-openclaw-1`.
- `OLLAMA_BENCH_OLLAMA_CONTAINER` — name of the Ollama container whose `OLLAMA_*` env the wrapper reads via `docker inspect` for the env snapshot. Default `ollama`.
- `OLLAMA_BENCH_REMOTE_DIR` — writable path inside the container (default `/tmp`).
- `OLLAMA_BENCH_MACHINE_ID` — stable fingerprint seed. Auto-populated from `/etc/machine-id` (or `/var/lib/dbus/machine-id`). Set manually only if your host doesn't expose one.

The wrapper also runs `nvidia-smi` and `docker inspect <ollama>` on the host and forwards the results into the sibling as `OLLAMA_BENCH_GPU_CSV` / `OLLAMA_BENCH_SERVER_ENV_JSON`. `bench.mjs` prefers these over its own in-container probes, so GPU state and `OLLAMA_NUM_PARALLEL` always land in the baseline even when the sibling has no `nvidia-smi` or docker socket.

The wrapper copies all four harness files (`bench.mjs`, `bench-toolcall.mjs`, `bench-multiturn.mjs`, `bench-tools.mjs`) into the sibling under `$OLLAMA_BENCH_REMOTE_DIR` and fails fast with a named file list if any are missing on the host. Baseline syncs back to `./baseline.json` on the host only when content changed during the run — no mtime churn on no-op invocations.

## What the throughput run measures

Four dimensions, each chosen because it isolates a class of regression:

1. **Single-stream generation** at 3 prompt sizes (short / medium / long ≈ 200 / 2K / 8K tokens) plus a long-gen cell (short prompt, 1024 output tokens to catch KV-cache and sampler-at-depth regressions). Median over N runs. Every call starts with a per-process nonce so the entire prompt prefix differs from any other call — defeats llama.cpp's prefix cache so prompt-eval numbers reflect real compute, not cache hits.
2. **Cold start.** Forces eviction with `keep_alive: "0s"`, then measures `load_duration` + wall time for the next request. The number your user pays when the model has fallen out of VRAM.
3. **Concurrency.** 1 / 2 / 4 / 8 parallel streams at a medium prompt × 64-token gen (edit `CONCURRENT_CELL` in `bench.mjs` to change the cell). Three per-row metrics:
   - `e2e t/s` — end-to-end wall-time throughput (prompt-eval + network included). What a caller experiences.
   - `decode t/s` — aggregate pure-decoder throughput, `sum(eval_count) / max(eval_duration)`. Isolates batcher scaling from prompt-eval overhead.
   - `per-stream t/s` — median of per-request decode speed.
   At parallel=1, `decode t/s` should match `per-stream t/s` within noise — a gap means prompt-eval or network is dominating. Above parallel=1, `decode t/s` scaling is dominated by `OLLAMA_NUM_PARALLEL`.
4. **Environment snapshot.** Ollama version, model digest + quant, GPU state (name, driver, VRAM, util, temp, power, SM clock), and the Ollama server's `OLLAMA_*` env vars read via `docker inspect`. Stored in the baseline so a regression run diffs *what changed* alongside *how it changed*.

On a compare run, any cell more than 5% worse than baseline gets a `⚠`, and an `[environment changes vs baseline]` section lists any scalar that differs — Ollama version, GPU driver, `OLLAMA_NUM_PARALLEL`, etc. Baselines include per-cell noise floors (2σ), so the threshold auto-widens on naturally noisy cells: the effective threshold is `max(--regression-pct, 2× cv%)`.

A full run is ~2 minutes for a 26B model on a single modern GPU.

## Tool-calling probes

Separate from throughput. These characterize a model's *capability*, not its speed. Useful when picking a model or diagnosing agent loops.

### `./bench toolcall` — single-turn

22 cases checking tool selection, schema fidelity, and correct declination when no tool fits. Three categories:
- `simple` — one obvious tool should be called
- `multiple` — disambiguate across tools
- `relevance` — no tool should be called

Reports pass% and schema% per category. `schema%` is scored against whichever tool the model *actually* called, so a wrong-tool pick with well-formed args still registers valid schema fidelity — only ill-formed JSON or missing required keys counts as a schema failure. Add `-v` for per-case output.

### `./bench multiturn` — two-turn with fabricated tool results

14 cases that test what happens **after** a tool returns. The user prompt triggers a tool call; the probe injects a fabricated tool result; the model's next turn is scored. Catches the real failure mode local models hit:

- **Synthesis** — tool returned useful data; model should summarize.
- **Empty** — tool returned `[]`; model should say "no results", NOT re-call.
- **Error** — tool returned `{error: ...}`; model should surface or handle.
- **Chain** — tool 1 succeeded; model should legitimately call tool 2.

Scored failures are specific: `LOOP: re-called X with identical args`, `LOOP: re-called X with different args`, `unexpected tool call: Y`, `expected synthesis, got 0-char content`. When you switch models (qwen, llama, mistral), these categories show where each model breaks.

Both probes share the same tool catalogue from `bench-tools.mjs`.

## Interpretation notes

- **Per-stream tokens/sec** stays roughly constant as concurrency rises when `OLLAMA_NUM_PARALLEL=1` (Ollama's historical default) — requests are serialized on a single GPU. `decode t/s` grows sub-linearly. Set `OLLAMA_NUM_PARALLEL` higher and you'll get batching; the scaling curve will look different. The env snapshot captures this so you know which regime you measured.
- **Noise floor** on a warm, idle machine is typically ±3–4% for throughput and ±1.5% for wall time. The baseline measures per-cell noise directly and widens the threshold to `max(--regression-pct, 2× cv%)`. If one cell flags ⚠ but the rest of the run is clean, re-run before panicking.
- **GPU must be idle.** If another process is using it (image generation, game, another model), numbers are garbage. The tool warns when GPU utilization is ≥10% before a run but doesn't refuse.
- **Baselines are machine-specific.** `baseline.json` has meaning only against the machine that produced it. The tool fingerprints the machine (via host `/etc/machine-id` when available, GPU+hostname+kernel otherwise) and skips compare with a warning when a baseline is from a different machine.
- **Absolute numbers are indicative, deltas are the product.** The headline t/s on any individual run varies with cooling, clock boost, background noise. What matters is whether the same machine gets slower than *itself*.

## What it doesn't measure

- **Capability throughput on agentic workloads.** The perf harness uses `/api/generate` with fixed prompts. If your app uses `/v1/chat/completions` with tool schemas, that path has different overhead the perf run won't catch — the tool-call probes exercise that path but score correctness, not speed.
- **Very long context.** Measured up to ~8K prompt tokens. Scale past that is model-dependent; edit `CTX_SIZES` in `bench.mjs` if you care about 32K/64K/128K.
- **Cross-model comparison.** It tells you whether *your* model got slower. Comparing apples to oranges is a different problem.

## Direct host invocation

If Ollama is reachable from the host directly (native install, or a container with a published port), skip the `./bench` wrapper and invoke the CLI entry point:

```bash
node bench.mjs --host http://localhost:11434
node bench.mjs toolcall --model qwen3:30b -v
```

Every subcommand and flag works the same way. You lose the auto-populated machine fingerprint from `/etc/machine-id` — bench.mjs falls back to hashing GPU info + hostname + kernel, which is stable on a bare-metal host.

## Requirements

- Node 20+ (uses built-in `fetch`). No `npm install`.
- An Ollama endpoint reachable from where the CLI actually runs (the sibling container, in the default flow).
- Optional: `nvidia-smi` on PATH inside the sibling for GPU state capture.
- Optional: `docker` on PATH on the host for `OLLAMA_*` env var inspection.

## Files

- `bench` — host-side wrapper for the Docker-sibling flow.
- `bench.mjs` — unified CLI entry point (perf + subcommand dispatch).
- `bench-toolcall.mjs` — single-turn tool-call probe.
- `bench-multiturn.mjs` — two-turn tool-call probe.
- `bench-tools.mjs` — shared tool catalogue, imported by both probes.
- `baseline.json` — saved perf baseline. Gitignored; one per machine.

## License

MIT. See `LICENSE`.
