# ollama-bench

A measurement harness for local [Ollama](https://ollama.com) on a single machine. Three things in one tool:

1. **Throughput regression check.** Same model, same box, same workload — did your last config change make it slower?
2. **Per-machine model league.** Bench multiple models on the same hardware and rank them by what matters: gen t/s, TTFT, cold-load, concurrency, tool-use accuracy.
3. **Tool-calling capability probes.** Single-turn (does the model pick the right tool?) and multi-turn (does it loop after a tool returns?).

Assumes a Docker-native setup: Ollama runs inside a container on a user-defined bridge network, reachable only from sibling containers. The `./bench` wrapper pipes the harness into a sibling, runs it there, and syncs results back to the host. For native-host Ollama, see [Direct host invocation](#direct-host-invocation).

## Quickstart

```bash
./bench                       # smart perf run for default model (save first time, compare after)
./bench rank --model X        # bench X end-to-end and add it to the league
./bench league                # ranked table across every model on this machine
./bench doctor                # preflight audit of GPU/Ollama/host config
./bench all                   # perf + toolcall + multiturn for current --model
./bench --help                # full CLI
```

## CLI

```
./bench                       Smart perf run for --model (save if no entry, compare otherwise).
./bench perf [save|compare|run]   Explicit perf mode. Default smart.
./bench toolcall              Single-turn tool-call accuracy probe (22 cases).
./bench multiturn             Multi-turn probe — what happens after a tool returns (14 cases).
./bench doctor                Audit: GPU persistence/power/throttle, Ollama env, host governor/swap.
./bench all                   perf (smart) + toolcall + multiturn for --model.
./bench rank                  Bench --model end-to-end and (re)write its slice in the league.
./bench league                Ranked table across all models benched on this machine.
./bench baseline show         Per-model summary of what's in the baseline.
./bench baseline clear        Delete baseline.json.
./bench baseline clear <tag>  Drop one model's entry.
./bench --help                Usage.
```

Flags (any order, any subcommand):

- `--model <tag>` — default `gemma4:26b`
- `--host <url>` — default `http://ollama:11434`
- `--runs <n>` — per-cell runs for perf, default 3
- `--out <path>` — baseline file, default `./baseline.json`
- `--regression-pct <n>` — regression threshold in %, default 5
- `-v` / `--verbose` — per-case output for toolcall / multiturn
- `OLLAMA_BENCH_TIMEOUT_MS` — env override for per-call request timeout

## What the perf run measures

Four dimensions, each chosen because it isolates a class of regression:

1. **Single-stream generation** at three prompt sizes (short / medium / long ≈ 200 / 2K / 8K tokens) plus a long-gen cell (short prompt, 1024 output tokens). Median over N runs. Reports prompt t/s, gen t/s, TTFT (time to first decoded token), and total wall ms. Every call starts with a per-process nonce so the prompt prefix differs from any other call — defeats llama.cpp's prefix cache so prompt-eval reflects real compute.
2. **Cold start.** Forces eviction with `keep_alive: "0s"`, then measures load duration, full-reply wall time, and TTFT for the next request. Median over 3 cycles.
3. **Concurrency.** 1 / 2 / 4 / 8 parallel streams at a medium prompt × 64-token gen. Three per-row metrics: `e2e t/s` (end-to-end wall throughput, what a caller experiences), `decode t/s` (aggregate pure-decoder, isolates batcher scaling from prompt-eval), `per-stream t/s` (median per-request).
4. **Environment snapshot.** Ollama version, model digest + quant, GPU state (name, driver, VRAM, util, temp, power, SM clock), and the Ollama server's `OLLAMA_*` env vars. Stored alongside the numbers so a regression run diffs *what changed* alongside *how it changed*.

Compare mode flags any cell more than 5% worse than baseline with ⚠. Baselines include per-cell noise floors (2σ); the effective threshold is `max(--regression-pct, 2× cv%)`. An `[environment changes vs baseline]` block lists scalar deltas (Ollama version, GPU driver, `OLLAMA_NUM_PARALLEL`, etc.).

A full perf run takes ~2 minutes for a 26B model on a single modern GPU.

## Model league

`baseline.json` is keyed by model: one machine, many models. `./bench rank --model <tag>` benches a model end-to-end and writes its slice. `./bench league` prints the ranked table:

| column | source |
|---|---|
| `model` | tag |
| `params` / `quant` | from `/api/show` |
| `gen t/s` | short-prompt single-stream median |
| `ttft ms` | short-prompt time to first token |
| `cold load` | median load duration after forced eviction |
| `n=4 t/s` | per-stream throughput at parallel=4 |
| `tool %` / `multi %` | toolcall + multiturn pass rate |
| `age` | days since this entry was last refreshed; ⚠ at >30d |

Sorted by `gen t/s` descending. Entries refresh independently — re-running `rank` for one model doesn't touch the others.

## Tool-calling probes

Separate from throughput. These score capability, not speed.

### `toolcall` — single-turn (22 cases)

Three categories:
- `simple` — one obvious tool should be called
- `multiple` — disambiguate across tools (some cases accept alternates)
- `relevance` — no tool should be called

Reports pass% and schema% per category. `schema%` scores arguments against whichever tool the model actually called — a wrong-tool pick with well-formed args still counts as valid schema fidelity. Compare deltas flag drops past `±5pp` to filter run-to-run noise.

### `multiturn` — two-turn with fabricated tool results (14 cases)

Initial prompt triggers a tool call; the probe injects a fabricated tool result; the model's next turn is scored. Four categories:

- **Synthesis** — tool returned useful data; model should summarize.
- **Empty** — tool returned `[]`; model should say "no results", not re-call.
- **Error** — tool returned `{error: ...}`; model should surface or handle.
- **Chain** — tool 1 succeeded; model should legitimately call tool 2.

Failure signatures are specific: `LOOP: re-called X with identical args`, `LOOP: re-called X with different args`, `unexpected tool call: Y`, `expected synthesis, got 0-char content`.

Both probes share `bench-tools.mjs` (LifeOps-shaped: email, calendar, tasks, quote, web search) and use a 180s per-call timeout (`OLLAMA_BENCH_TIMEOUT_MS` to override).

## Doctor

`./bench doctor` audits the system before you bench it. Each check emits ✓ ok / ⚠ warn / ✗ fail / · info / ? unknown, with a fix command when actionable:

- **GPU**: persistence mode, power-limit headroom, hardware throttle bits.
- **Ollama**: API reachable (10s timeout), server version, `OLLAMA_NUM_PARALLEL`, KV cache + flash attention combo (catches the silent f16 fallback when `KV_CACHE_TYPE=q4_0` without `FLASH_ATTENTION=1`), `KEEP_ALIVE`, `CONTEXT_LENGTH`, model presence.
- **Host**: CPU governor, swap usage.

Exits non-zero only on real failures; warns are advisory.

## Container setup

Wrapper environment variables:

- `OLLAMA_BENCH_CONTAINER` — sibling container name (Node 20+, network access to Ollama). Auto-tries `openclaw` then `engine-openclaw-1`.
- `OLLAMA_BENCH_OLLAMA_CONTAINER` — name of the Ollama container whose `OLLAMA_*` env the wrapper reads via `docker inspect`. Default `ollama`.
- `OLLAMA_BENCH_REMOTE_DIR` — writable path inside the sibling. Default `/tmp`.
- `OLLAMA_BENCH_MACHINE_ID` — stable machine fingerprint seed. Auto-populated from `/etc/machine-id` (or `/var/lib/dbus/machine-id`).
- `OLLAMA_BENCH_TIMEOUT_MS` — per-call request timeout override.

The wrapper runs `nvidia-smi` and `docker inspect <ollama>` on the host and forwards results into the sibling as `OLLAMA_BENCH_GPU_CSV` / `OLLAMA_BENCH_GPU_EXT_CSV` / `OLLAMA_BENCH_SERVER_ENV_JSON` / `OLLAMA_BENCH_CPU_GOVERNOR` / `OLLAMA_BENCH_SWAP`. The harness prefers injected data over its own probes, so GPU state, host config, and `OLLAMA_NUM_PARALLEL` always land in the baseline even when the sibling can't see them.

All harness files (`bench.mjs`, `bench-toolcall.mjs`, `bench-multiturn.mjs`, `bench-tools.mjs`, `bench-doctor.mjs`, `bench-baseline.mjs`) are copied into the sibling under `$OLLAMA_BENCH_REMOTE_DIR`. The wrapper fails fast with a named file list if any are missing on the host. Baseline syncs back to `./baseline.json` only when content changed during the run.

## Interpretation notes

- **Per-stream t/s** stays roughly constant as concurrency rises when `OLLAMA_NUM_PARALLEL=1` — requests are serialized on a single GPU. `decode t/s` grows sub-linearly. Raise `OLLAMA_NUM_PARALLEL` for batching. The env snapshot captures which regime you measured.
- **Noise floor** on a warm, idle machine is typically ±3–4% for throughput, ±1.5% for wall time. Baselines measure per-cell noise directly and widen the threshold to `max(--regression-pct, 2× cv%)`.
- **GPU must be idle.** Other processes on the GPU make the numbers garbage. Doctor warns at ≥10% utilization before a run.
- **Baselines are machine-specific.** The tool fingerprints the machine (host `/etc/machine-id` when available, GPU + hostname + kernel otherwise). Compare against a different machine prints a warning and skips the diff.
- **Absolute numbers vary** with cooling, clock boost, background noise. Deltas against the same machine are the signal.

## What it doesn't measure

- **Agentic workload throughput.** Perf uses `/api/generate` with fixed prompts. Tool-call paths (`/v1/chat/completions` with tools) have different overhead — the probes exercise that path but score correctness, not speed.
- **Very long context.** Measured up to ~8K prompt tokens. For 32K/64K/128K, edit `CTX_SIZES` in `bench.mjs`.

## Direct host invocation

If Ollama is reachable from the host directly (native install, or a container with a published port), skip the wrapper:

```bash
node bench.mjs --host http://localhost:11434
node bench.mjs rank --model qwen3:30b --host http://localhost:11434
node bench.mjs toolcall --model qwen3:30b -v --host http://localhost:11434
```

Every subcommand and flag works the same. Without the wrapper you lose the host-injected env (GPU CSV, server env, CPU governor) — the harness falls back to its own probes, which work on a bare-metal host.

## Requirements

- Node 20+ (built-in `fetch`). No `npm install`.
- An Ollama endpoint reachable from where the CLI runs.
- Optional: `nvidia-smi` for GPU state capture.
- Optional: `docker` CLI for `OLLAMA_*` env inspection.

## Files

- `bench` — host-side wrapper for the Docker-sibling flow.
- `bench.mjs` — unified CLI entry point (perf + subcommand dispatch + league).
- `bench-toolcall.mjs` — single-turn tool-call probe.
- `bench-multiturn.mjs` — multi-turn tool-call probe.
- `bench-tools.mjs` — shared tool catalogue.
- `bench-doctor.mjs` — preflight system audit.
- `bench-baseline.mjs` — per-model baseline I/O.
- `baseline.json` — one per machine, keyed by model. Gitignored.

## License

MIT. See `LICENSE`.
