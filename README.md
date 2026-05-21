# ollama-bench

Measurement harness for local [Ollama](https://ollama.com). Three subsystems:

1. **Throughput regression check** against a per-machine baseline.
2. **Per-machine model league** ranking models on the same hardware.
3. **Tool-calling capability probes** — single-turn and multi-turn.

Run mode: a `./bench` bash wrapper that copies the harness into a sibling Docker container with network access to Ollama and runs it there, syncing the baseline back to the host on exit. Native-host Ollama is supported via [direct invocation](#direct-host-invocation).

## Quickstart

```bash
./bench                       # smart perf run for default model
./bench rank --model X        # add or refresh model X in the league
./bench league                # ranked table across all benched models
./bench routes                # best model per job role
./bench doctor                # preflight system audit
./bench all                   # perf + toolcall + multiturn + jobs for --model
./bench --help                # full CLI
```

## CLI

```
./bench                       Smart perf run: save if no baseline entry, compare otherwise.
./bench perf [save|compare|run]   Explicit perf mode (default smart).
./bench toolcall              Single-turn tool-call probe (22 cases).
./bench multiturn             Multi-turn tool-call probe (14 cases).
./bench jobs                  Judge-scored quality probe across job roles.
./bench doctor                System audit (GPU / Ollama / host).
./bench all                   perf + toolcall + multiturn + jobs.
./bench rank                  Bench --model end-to-end, write its league slice.
./bench league                Ranked table across all models.
./bench routes                Best model per job, plus quality-per-second.
./bench baseline show         Per-model summary of the baseline.
./bench baseline clear        Delete baseline.json.
./bench baseline clear <tag>  Drop one model's entry.
```

Flags:

| Flag | Default | Effect |
|---|---|---|
| `--model <tag>` | `gemma4:26b` | Target model. |
| `--host <url>` | `http://ollama:11434` | Ollama endpoint. |
| `--runs <n>` | `3` | Per-cell runs (perf). |
| `--out <path>` | `./baseline.json` | Baseline file. |
| `--regression-pct <n>` | `5` | Regression threshold (%). |
| `--concurrent-levels <csv>` | auto | Override stage-3 parallel levels. |
| `--no-concurrent` | off | Skip stage 3. Same as `OLLAMA_BENCH_NO_CONCURRENT=1`. |
| `--judge <tag>` | `gemma4:31b` | Judge model for `jobs`. Auto-swaps to `gpt-oss:20b` when target == judge. Must already be pulled. |
| `-v` / `--verbose` | off | Per-case output for toolcall / multiturn / jobs. |

Env overrides:

| Var | Effect |
|---|---|
| `OLLAMA_BENCH_TIMEOUT_MS` | Per-call request timeout. |
| `OLLAMA_BENCH_NO_CONCURRENT` | Skip stage 3 (equivalent to `--no-concurrent`). |
| `OLLAMA_BENCH_THINK` | `1` to enable reasoning-mode (off by default). |
| `OLLAMA_BENCH_JUDGE` | Default judge tag for the `jobs` probe. |

## Perf measurements

1. **Single-stream**: short / medium / long prompts (~200 / 2K / 8K tokens) plus a long-gen cell (short prompt, 1024 output tokens). Median over `--runs`. Reports prompt t/s, gen t/s, TTFT, total wall ms. Each call starts with a per-process nonce so llama.cpp's prefix cache can't hide prompt-eval.
2. **Cold start**: forces eviction with `keep_alive: "0s"`, then measures load duration, full-reply wall, and TTFT on the next request. Median over 3 cycles.
3. **Concurrency**: 1 / 2 / 4 / 8 parallel streams at medium prompt × 64-token gen. Reports `e2e t/s`, `decode t/s`, `per-stream t/s`. Levels auto-cap by `OLLAMA_NUM_PARALLEL`, model params (≥30B → max n=4, ≥45B → n=2, ≥65B → skip), and post-warmup VRAM (≥75% / ≥85% / ≥95% tighten further). `OLLAMA_NUM_PARALLEL=1` skips the stage. `--concurrent-levels` overrides; `--no-concurrent` skips.
4. **Environment snapshot**: Ollama version, model digest + quant, GPU state (name, driver, VRAM, util, temp, power, SM clock), Ollama server `OLLAMA_*` env.

`compare` mode flags cells worse than `max(--regression-pct, 2 × cv%)` with ⚠. Environment-change deltas (Ollama version, GPU driver, `OLLAMA_NUM_PARALLEL`, etc.) print in a separate block.

## Model league

`baseline.json` is keyed by model. `./bench rank --model <tag>` benches end-to-end and writes the slice. `./bench league` prints:

| Column | Source |
|---|---|
| `model` | tag |
| `params` / `quant` | `/api/show` |
| `gen t/s` | short-prompt single-stream median |
| `ttft ms` | short-prompt time to first token |
| `cold load` | median load after forced eviction |
| `n=4 t/s` | per-stream throughput at parallel=4 |
| `tool %` / `multi %` | toolcall + multiturn pass rate |
| `jobs` | overall quality score |
| `age` | days since last refresh (⚠ at >30d) |

Sorted by `gen t/s` descending. `./bench routes` regroups the same data by job role with a quality-per-second view.

## Tool-calling probes

### `toolcall` — single-turn (22 cases)

Categories: `simple` (one obvious tool), `multiple` (disambiguate; alternates accepted via `altNames`), `relevance` (no tool should be called). Reports pass% and schema% per category. `schema%` scores arguments against the tool the model actually called.

### `multiturn` — two-turn (14 cases)

Initial prompt → expected first tool call → fabricated tool result injected → second turn scored. Categories:

- **Synthesis** — tool returned data; model should summarize.
- **Empty** — tool returned `[]`; model should say "no results".
- **Error** — tool returned `{error: ...}`; model should surface or handle.
- **Chain** — tool 1 succeeded; model should call tool 2.

Failure signatures: `LOOP: re-called X with identical args`, `LOOP: re-called X with different args`, `unexpected tool call: Y`, `expected synthesis, got 0-char content`.

Both probes share `bench-tools.mjs` (email, calendar, tasks, quote, web search). Per-call timeout: 180s (`OLLAMA_BENCH_TIMEOUT_MS` overrides).

## Jobs probe

`./bench jobs` (or `./bench rank ...` which includes it) runs a judge-scored quality probe across job roles: `trading_brief`, `x_analysis`, `document_prep`, `hard_toolcall`, `reasoning`. Hybrid scoring per case:

- `deterministic`: JSON parse, required keys, gold-label match, numeric tolerance
- `judge`: a separate model rates the response on a 0–3 rubric
- `caseScore = 0.5 × deterministic + 0.5 × (judge / 3)`, in `[0, 1]`

Two-pass under `OLLAMA_MAX_LOADED_MODELS=1`: pass 1 generates all responses with the candidate; pass 2 loads the judge once and scores. Per-call timeout: 240s.

## Doctor

`./bench doctor` emits ✓ ok / ⚠ warn / ✗ fail / · info / ? unknown per check, with a fix command when actionable.

- **GPU**: persistence mode, power-limit headroom, hardware/SW throttle bits.
- **Ollama**: API reachable (10s timeout), server version, `OLLAMA_NUM_PARALLEL`, KV cache + flash attention combo, `KEEP_ALIVE`, `CONTEXT_LENGTH`, model presence.
- **Host**: CPU governor, swap usage.

Exits non-zero only on `fail`.

## Container setup

Wrapper env vars:

| Var | Default | Purpose |
|---|---|---|
| `OLLAMA_BENCH_CONTAINER` | auto-detect | Sibling container name (Node 20+, on Ollama's network). |
| `OLLAMA_BENCH_OLLAMA_CONTAINER` | `ollama` | Ollama container name — `docker inspect`ed for `OLLAMA_*` env. |
| `OLLAMA_BENCH_REMOTE_DIR` | `/tmp` | Writable path in the sibling for scripts + baseline. |
| `OLLAMA_BENCH_MACHINE_ID` | `/etc/machine-id` | Machine fingerprint seed. |
| `OLLAMA_BENCH_TIMEOUT_MS` | — | Per-call request timeout override. |

Auto-detect: lists running containers on the Ollama container's user-defined network(s), excludes the Ollama container itself, picks the unique survivor that has `node` on PATH. Bails out if zero or multiple match.

The wrapper runs `nvidia-smi` and `docker inspect <ollama>` on the host and forwards results via `OLLAMA_BENCH_GPU_CSV` / `OLLAMA_BENCH_GPU_EXT_CSV` / `OLLAMA_BENCH_SERVER_ENV_JSON` / `OLLAMA_BENCH_CPU_GOVERNOR` / `OLLAMA_BENCH_SWAP` / `OLLAMA_BENCH_PERSISTENCED`. The harness uses injected data when present, falls back to local probes otherwise.

All harness files (`bench.mjs`, `bench-toolcall.mjs`, `bench-multiturn.mjs`, `bench-jobs.mjs`, `bench-tools.mjs`, `bench-doctor.mjs`, `bench-baseline.mjs`) are copied into the sibling under `$OLLAMA_BENCH_REMOTE_DIR` per run. Baseline syncs back to `./baseline.json` only when content changed (atomic: temp file + rename).

## Direct host invocation

When Ollama is reachable from the host directly:

```bash
node bench.mjs --host http://localhost:11434
node bench.mjs rank --model nemotron3:33b --host http://localhost:11434
node bench.mjs toolcall --model nemotron3:33b -v --host http://localhost:11434
```

All subcommands and flags work identically. Without the wrapper the harness uses its own GPU / server-env probes.

## Requirements

- Node 20+ (built-in `fetch`). No `npm install`.
- An Ollama endpoint reachable from the runner.
- Optional: `nvidia-smi` (GPU state), `docker` CLI (`OLLAMA_*` env inspection).

## Not measured

- Agentic workload throughput. Perf uses `/api/generate`; the tool-call paths score correctness, not speed.
- Contexts above ~8K prompt tokens. Edit `CTX_SIZES` in `bench.mjs` to extend.

## Files

| File | Role |
|---|---|
| `bench` | Host wrapper (Docker-sibling flow). |
| `bench.mjs` | CLI entry point + perf scenarios + league/routes/baseline commands. |
| `bench-toolcall.mjs` | Single-turn tool-call probe. |
| `bench-multiturn.mjs` | Multi-turn tool-call probe. |
| `bench-jobs.mjs` | Judge-scored quality probe. |
| `bench-tools.mjs` | Shared tool catalogue. |
| `bench-doctor.mjs` | System audit. |
| `bench-baseline.mjs` | Per-model baseline I/O. |
| `baseline.json` | One per machine, keyed by model. Gitignored. |

## License

MIT. See `LICENSE`.
