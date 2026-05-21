# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

The user-facing usage docs live in `README.md` — read it first for CLI surface, what each subcommand measures, and interpretation guidance. This file covers architecture, conventions, and how to develop on the code.

## Architecture

Two-tier execution model:

1. **Host wrapper** (`bench`, bash) — runs on the box where Ollama lives. It picks a sibling container with Node + network access to the Ollama container, copies the `.mjs` files in, runs `nvidia-smi` and `docker inspect <ollama>` on the host, forwards results as `OLLAMA_BENCH_*` env vars, then invokes `node bench.mjs` inside the sibling. Baseline syncs back to the host on exit if content changed.
2. **Node harness** (`bench.mjs` + siblings) — pure Node 20+ stdlib, no `npm install`, no build step. Runs inside whichever container the wrapper picked, or directly on the host via `node bench.mjs --host …`.

Every `.mjs` is independently executable and parses its own CLI args. `bench.mjs` is the unified entry point that dispatches to subcommand modules via `spawn`, but you can also run e.g. `node bench-doctor.mjs` standalone. Only two modules are *imported* across files: `bench-baseline.mjs` (read/write/migrate the per-machine baseline) and `bench-tools.mjs` (shared tool catalogue for `toolcall` + `multiturn`). The agent probe inlines its own catalogue because its tools mutate world state.

### Module map

| File | Role |
|---|---|
| `bench` | Host-side wrapper: container detection, env forwarding, baseline sync |
| `bench.mjs` | Subcommand dispatcher + perf benchmark + league + routes + baseline views |
| `bench-doctor.mjs` | Preflight audit — emits ok/warn/fail/info/unknown checks |
| `bench-toolcall.mjs` | Single-turn tool-call probe (22 cases, custom) |
| `bench-multiturn.mjs` | Two-turn probe with fabricated tool results (14 cases, custom) |
| `bench-agent.mjs` | Multi-turn ReAct agent over mutable synthetic world (~13 cases) |
| `bench-math.mjs` | GSM8K + MATH-500 (numeric reasoning) |
| `bench-code.mjs` | HumanEval, executed in a sandboxed python3 subprocess |
| `bench-knowledge.mjs` | MMLU-Pro (10-option MCQ, stratified) |
| `bench-ifeval.mjs` | IFEval (JS verifier port for ~25 constraint types) |
| `bench-data.mjs` | WikiTableQuestions (table QA) + WikiSQL (SQL gen, executed via node:sqlite) |
| `bench-tools.mjs` | Tool catalogue shared by toolcall + multiturn |
| `bench-baseline.mjs` | Baseline I/O — schema v2 (per-machine, keyed by model), auto-migrates v1 |
| `bench-gpu.mjs` | Background `nvidia-smi -lms` sampler, aggregated `{avg, max}` per field |
| `scripts/fetch-data.mjs` | One-shot data acquisition (HF datasets-server API → sampled JSONL) |
| `data/*.jsonl` | Vendored benchmark samples — gsm8k, math500, humaneval, mmlupro, ifeval, wtq, wikisql |

### Capability matrix

The league's composite `iq` score is the equal-weighted mean of every capability column that exists for the model. New probes feed into it via the `CAPABILITIES` table in `bench.mjs` — `{key, label, section, pct: entry → 0..100}`. Add a probe? Append a row there and the league + routes views pick it up automatically.

### Data flow conventions

- **Host-injected env wins over self-probe.** The wrapper sets `OLLAMA_BENCH_GPU_CSV`, `OLLAMA_BENCH_GPU_EXT_CSV`, `OLLAMA_BENCH_SERVER_ENV_JSON`, `OLLAMA_BENCH_CPU_GOVERNOR`, `OLLAMA_BENCH_SWAP`. The harness prefers these and falls back to its own probes only when absent.
- **Baseline is per-machine and per-model.** Machine fingerprint comes from `OLLAMA_BENCH_MACHINE_ID` (seeded from `/etc/machine-id` by the wrapper) or, when absent, a hash of GPU + hostname + kernel. Compares against a different machine print a warning and skip the diff.
- **Prefix cache defeat.** Every perf `/api/generate` prompt starts with `PROCESS_NONCE` + per-run seed, so llama.cpp's prefix cache can't make prompt-eval look free.
- **TTFT requires streaming.** The single-stream and cold-start cells use `stream: true` to capture time-to-first-token.
- **Capability probes are deterministic-graded.** No judge model in the default path. Adding a probe that needs an LLM grader? Make it opt-in, not the default — the value of the suite is that it runs without a second model competing for VRAM.
- **Data is vendored, not fetched at runtime.** `data/*.jsonl` files ship with the repo. The fetch script (`scripts/fetch-data.mjs`) regenerates them with a fixed seed (`SEED=42`) so anyone running it produces identical samples. HF's datasets-server rate-limits aggressively (~5K rows for MMLU-Pro before 429s); the script handles partial fetches gracefully.
- **Sample sizes are intentional.** ~50 cases per probe (164 for HumanEval, full set). At p=0.5 that's ~±7pp noise — enough to rank distinct models, not enough to compare to published frontier-model numbers. Don't scale up without thinking about runtime budget.
- **GPU telemetry is best-effort.** `bench-gpu.mjs` checks for `nvidia-smi` once and silently no-ops if absent. When using the wrapper, the bench *container* needs nvidia-smi (not just the host) — the host injects a one-shot snapshot via env vars but the continuous sampler runs from inside the process. League/baseline omit GPU columns when no model has telemetry.

## Development

No build, no test suite, no lint config. Iteration loop is literally edit-and-rerun:

```bash
# Smoke the perf path against a small model
node bench.mjs --model gpt-oss:20b --host http://localhost:11434 --runs 1

# Smoke a capability probe with a few cases
node bench-math.mjs    --model gpt-oss:20b --host http://localhost:11434 --limit 5 -v
node bench-code.mjs    --model gpt-oss:20b --host http://localhost:11434 --limit 5 -v
node bench-knowledge.mjs --model gpt-oss:20b --host http://localhost:11434 --limit 5 -v
node bench-ifeval.mjs  --model gpt-oss:20b --host http://localhost:11434 --limit 5 -v
node bench-data.mjs    --model gpt-oss:20b --host http://localhost:11434 --limit 5 -v

# Smoke the agent (one category at a time keeps it tight)
node bench-agent.mjs   --model gpt-oss:20b --host http://localhost:11434 --cat recovery -v

# Smoke a tool-use probe in isolation
node bench-toolcall.mjs --model gpt-oss:20b --host http://localhost:11434 -v
node bench-multiturn.mjs --model gpt-oss:20b --host http://localhost:11434 -v
node bench-doctor.mjs  --host http://localhost:11434

# Smoke the wrapper end-to-end (on devbox)
OLLAMA_BENCH_CONTAINER=bench ./bench doctor

# Refresh vendored benchmark data (one-shot; HF datasets-server rate-limits)
node scripts/fetch-data.mjs              # skips files that exist
node scripts/fetch-data.mjs --force      # re-fetch everything
```

On `devbox` the sibling container is named `bench` (compose-managed alongside Ollama on `ollama_default`). The wrapper's default container search (`openclaw`, `engine-openclaw-1`) is dead history — always pass `OLLAMA_BENCH_CONTAINER=bench` there.

### Conventions worth respecting

- **Strict numeric flag parsing.** `parsePosInt` / `parseNonNegFloat` exist because earlier versions of the harness silently produced NaN from `--runs foo` and skipped loops. Keep new numeric flags routed through these — fail fast beats a garbage baseline.
- **`lastIndexOf` for flag lookup in `bench.mjs`.** Lets the wrapper append `--out <in-container-path>` after the user's args without the user's value (or an earlier accidental value) winning. Don't switch to `indexOf` here.
- **Compare deltas have a noise floor.** Throughput regressions are flagged at `max(--regression-pct, 2× cv%)` — per-cell noise is measured at save time. Don't hard-code threshold logic; route through `thresholdFor` / `isRegression`.
- **Schema migrations are read-side, not write-side.** `bench-baseline.mjs::migrate` upgrades older flat-shaped baselines on read. New schema bumps should follow that pattern (read-time migration, write current shape) so old files keep working.
- **No new runtime deps.** Stay on Node stdlib. `fetch` is built in on 20+, `crypto`/`fs`/`child_process` cover everything else used.

## Wrapper env vars (host → sibling)

The wrapper reads/derives these on the host and exports them into the container. Listed here because they're the contract between the two tiers:

| Var | Purpose | Default |
|---|---|---|
| `OLLAMA_BENCH_CONTAINER` | Sibling container name (Node + Ollama-network access) | tries `openclaw`, `engine-openclaw-1`, then prefix match |
| `OLLAMA_BENCH_OLLAMA_CONTAINER` | Container whose `OLLAMA_*` env we capture via `docker inspect` | `ollama` |
| `OLLAMA_BENCH_REMOTE_DIR` | Writable path inside sibling for scripts + baseline | `/tmp` |
| `OLLAMA_BENCH_MACHINE_ID` | Stable fingerprint seed | `/etc/machine-id` or `/var/lib/dbus/machine-id` |
| `OLLAMA_BENCH_TIMEOUT_MS` | Per-call request timeout override | none |
| `OLLAMA_BENCH_GPU_CSV` / `_GPU_EXT_CSV` | `nvidia-smi --query-gpu=…` output | injected by wrapper |
| `OLLAMA_BENCH_SERVER_ENV_JSON` | `docker inspect <ollama>` env array | injected by wrapper |
| `OLLAMA_BENCH_CPU_GOVERNOR` / `_SWAP` | host `/sys` + `/proc/swaps` | injected by wrapper |
