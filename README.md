# ollama-bench

Measurement harness for local [Ollama](https://ollama.com). Four subsystems:

1. **Throughput regression check** against a per-machine baseline.
2. **Capability probes** — math, code, knowledge, instruction-following, data analysis. Public open benchmarks, deterministic grading.
3. **Tool-use probes** — single-turn, multi-turn, and a full agentic ReAct loop with mutable world state.
4. **Per-machine model league** with a composite `iq` score and a speed/intelligence Pareto view.

Run mode: a `./bench` bash wrapper that copies the harness into a sibling Docker container with network access to Ollama and runs it there, syncing the baseline back to the host on exit. Native-host Ollama is supported via [direct invocation](#direct-host-invocation).

## Quickstart

```bash
./bench                       # smart perf run for default model
./bench rank --model X        # add or refresh model X in the league
./bench rank-all              # iterate every installed model (overnight)
./bench league                # ranked table across all benched models (sorted by iq)
./bench league --sort qps     # sorted by intelligence × throughput
./bench routes                # best model per capability + Pareto view
./bench doctor                # preflight system audit
./bench all                   # perf + every probe for --model
./bench --help                # full CLI
```

## CLI

```
./bench                       Smart perf run: save if no baseline entry, compare otherwise.
./bench perf [save|compare|run]

CAPABILITY PROBES (deterministic, public benchmarks):
./bench math                  GSM8K (50) + MATH-500 (30) — numeric reasoning.
./bench code                  HumanEval (164) — Python, executed in a sandbox.
./bench mmlu                  MMLU-Pro (50, stratified) — broad knowledge MCQ.
./bench ifeval                IFEval (50) — verifiable instruction-following constraints.
./bench data                  WikiTableQuestions (30) + WikiSQL (30) — table QA + SQL gen.

TOOL-USE PROBES:
./bench toolcall              Single-turn tool-call probe (22 cases, custom).
./bench multiturn             Two-turn tool-call probe (14 cases, custom).
./bench agent                 Multi-turn ReAct agent over synthetic world state (13 cases).

ORCHESTRATION:
./bench doctor                Preflight system audit.
./bench all                   perf + every probe.
./bench rank                  bench end-to-end and write the league slice.
./bench rank-all              iterate every installed Ollama model. Skips embed-only.
                              Flags: --skip-if-newer-than N, --include p1,p2, --exclude p1,p2, --log path.
./bench league                Ranked table across all models. --sort iq|speed|qps.
./bench routes                Per-capability winners + speed/intelligence Pareto.
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
| `--concurrent-levels <csv>` | auto | Override perf stage-3 parallel levels. |
| `--no-concurrent` | off | Skip perf stage 3. |
| `--limit <n>` | none | Cap cases per probe (smoke-test mode). |
| `--cell <name>` | all | For multi-cell probes (`math`, `data`): run only one cell. |
| `--cat <name>` | all | For `agent`: run only one category. |
| `--real-web` | off | Agent's web_search hits real DuckDuckGo (else fixture). |
| `--sort iq\|speed\|qps` | `iq` | League sort key. |
| `-v` / `--verbose` | off | Per-case output. |

Env overrides:

| Var | Effect |
|---|---|
| `OLLAMA_BENCH_TIMEOUT_MS` | Per-call request timeout. |
| `OLLAMA_BENCH_CODE_EXEC_MS` | HumanEval sandbox wall timeout per case (default 8 000). |
| `OLLAMA_BENCH_NO_CONCURRENT` | Skip perf stage 3. |
| `OLLAMA_BENCH_THINK` | `1` to enable reasoning-mode. |

## GPU telemetry

Every probe and the perf benchmark run a background `nvidia-smi` sampler at 1 Hz capturing GPU utilization, VRAM used, power draw, temperature, and SM clock. Aggregated to `{avg, max}` per field and stored in the baseline under each section's `gpu` key. Silent no-op when `nvidia-smi` is unavailable (e.g., inside a container without GPU passthrough).

The league surfaces three GPU columns when telemetry exists:
- `W` — average power draw across the perf run
- `VRAM` — average VRAM used in GiB
- `t/s/W` — tokens-per-second-per-watt, the efficiency metric

`./bench baseline show` prints avg/peak power and peak temperature per probe section.

## Overnight runs

`./bench rank-all` discovers every installed Ollama model via `/api/tags`, filters out embed-only models, and runs `rank` for each in sequence. Each model is a fresh subprocess — a failure in one doesn't kill the batch, and Ollama gets a clean state between models. Designed for unattended overnight execution.

```bash
./bench rank-all --log overnight.log               # log timestamped progress
./bench rank-all --skip-if-newer-than 1            # skip models benched today
./bench rank-all --include nemotron,qwen          # only these tag patterns
./bench rank-all --exclude embed                  # skip anything with "embed"
```

Per-model runtime varies by model size: ~30 min for a 20B, ~90+ min for a 33B. Six models on a single GPU typically completes in 6–12 hours. The final league snapshot prints at the end of the batch.

## Perf measurements

1. **Single-stream**: short / medium / long prompts (~200 / 2K / 8K tokens) plus a long-gen cell (short prompt, 1024 output tokens). Median over `--runs`. Reports prompt t/s, gen t/s, TTFT, total wall ms. Each call starts with a per-process nonce so llama.cpp's prefix cache can't hide prompt-eval.
2. **Cold start**: forces eviction with `keep_alive: "0s"`, then measures load duration, full-reply wall, and TTFT on the next request. Median over 3 cycles.
3. **Concurrency**: 1 / 2 / 4 / 8 parallel streams at medium prompt × 64-token gen. Reports `e2e t/s`, `decode t/s`, `per-stream t/s`. Auto-caps by `OLLAMA_NUM_PARALLEL`, model params, post-warmup VRAM.
4. **Environment snapshot**: Ollama version, model digest + quant, GPU state, server `OLLAMA_*` env.

`compare` mode flags cells worse than `max(--regression-pct, 2 × cv%)` with ⚠.

## Capability probes

All five are deterministic-graded against public open benchmarks. No judge model. Data is vendored under `data/*.jsonl` — regenerate with `node scripts/fetch-data.mjs`.

### `math` — GSM8K + MATH-500

- **gsm8k** (50 sampled cases): grade-school word problems. Prompt asks for `#### <answer>`; grader extracts numeric value.
- **math500** (30 sampled, stratified by level): competition math. Prompt asks for `\boxed{}`; grader normalizes LaTeX + tries numeric match.

### `code` — HumanEval

164 Python function-completion problems. Model emits code; harness assembles `prompt + completion + test + check(entry_point)` and runs it under `python3 -IB` in a temp dir with an 8s wall timeout. Exit 0 = pass.

### `mmlu` — MMLU-Pro

50 sampled questions stratified across categories. 10-option MCQ with harder distractors than original MMLU. Direct-prompted (no CoT) — extracts the answer letter A–J via several patterns. Per-category breakdown in the report.

### `ifeval` — IFEval

50 sampled cases. Each case has one or more verifiable constraints (`length_constraints:number_words`, `detectable_format:json_format`, `punctuation:no_comma`, etc.). Constraints are checked programmatically — a JS port of the upstream verifier semantics. Per-instruction pass rate in the report.

### `data` — WikiTableQuestions + WikiSQL

- **table_qa** (30 WikiTableQuestions cases): markdown-rendered table + NL question. Free-form answer compared to gold answer set (string + numeric normalized).
- **sql_gen** (30 WikiSQL cases): schema + question → model SQL → executed via Node's built-in `node:sqlite` against an in-memory table loaded from the case fixture. Result rows compared (set-wise, numeric tolerant) to a precomputed gold answer list (gold computed from WikiSQL's structured `sql` field to sidestep `human_readable` quoting quirks).

## Tool-use probes

### `toolcall` — single-turn (22 cases, custom)

Categories: `simple` (one obvious tool), `multiple` (disambiguate), `relevance` (no tool should fire). Reports pass% and schema% per category.

### `multiturn` — two-turn (14 cases, custom)

Initial prompt → expected first tool call → fabricated tool result injected → second turn scored. Categories: synthesis / empty / error / chain.

### `agent` — full ReAct loop (13 cases)

Multi-turn agent loop (up to 8 turns), tools mutate a per-case copy of a synthetic world (~20 emails, ~10 calendar events, a tasks list, a small e-commerce sqlite db). Tool catalogue: email (inbox/search/read/send), calendar (events/create), task (list/create), web_search, list_tables/describe_table/query_data.

Categories:

- **workflow** (4 cases): multi-step goal completion. *"Find tomorrow's first meeting, send the organizer a confirmation."* Grading: did the right email get sent? was the right task created?
- **recovery** (3 cases): handle tool errors, empties, malformed arg requirements without looping. *Search fails on first call — does the agent fall back to inbox?*
- **triage** (2 cases): sift signal from noise across many tool results. *"Summarize what legal and Bob said about the Acme contract" — needs to filter ~20 emails.*
- **data_analysis** (3 cases): list tables → describe → query → reason → answer. *"Top 3 product categories by May 2026 revenue, excluding returned orders."*

Loop detection: same tool+args repeated immediately fails the case. Web search defaults to fixture results for repeatability; pass `--real-web` to hit DuckDuckGo's HTML endpoint live.

## League and routes

`baseline.json` stores per-model probe results. `./bench league` prints a table with a composite **`iq`** column — the equal-weighted mean of every capability percentage that exists for the model:

| Column | Source |
|---|---|
| `model` / `params` | `/api/show` |
| `gen t/s` | short-prompt single-stream median |
| `tool %` / `multi %` | toolcall + multiturn pass rate |
| `math %` / `code %` / `mmlu %` / `ifeval %` / `data %` / `agent %` | capability probes |
| `iq` | equal-weighted mean of the above |
| `age` | days since last refresh |

Sort key controlled by `--sort`:
- `iq` (default) — intelligence first
- `speed` — short-prompt gen t/s
- `qps` — `iq × gen t/s ÷ 100`, the speed/intelligence Pareto pick

`./bench routes` rebuilds two views from the same data:
- **By capability** — winner per dimension (best math model, best code model, etc.) with the runner-up gap.
- **Speed/intelligence Pareto** — every model ranked by `iq × gen t/s`. Surfaces "good-enough fast" picks that pure-iq sort hides.

## Doctor

`./bench doctor` emits ✓ ok / ⚠ warn / ✗ fail / · info / ? unknown per check, with a fix command when actionable. Exits non-zero only on `fail`.

## Container setup

Wrapper env vars:

| Var | Default | Purpose |
|---|---|---|
| `OLLAMA_BENCH_CONTAINER` | auto-detect | Sibling container (Node 20+, on Ollama's network, python3 for code probe). |
| `OLLAMA_BENCH_OLLAMA_CONTAINER` | `ollama` | Ollama container — `docker inspect`ed for `OLLAMA_*` env. |
| `OLLAMA_BENCH_REMOTE_DIR` | `/tmp` | Writable path in the sibling for scripts + data + baseline. |
| `OLLAMA_BENCH_MACHINE_ID` | `/etc/machine-id` | Machine fingerprint seed. |

Auto-detect: lists containers on the Ollama container's user-defined network(s), picks the unique survivor with `node` on PATH.

The wrapper copies the harness files **and** `data/*.jsonl` into the sibling under `$OLLAMA_BENCH_REMOTE_DIR` per run. Baseline syncs back to `./baseline.json` only when content changed.

## Direct host invocation

```bash
node bench.mjs --host http://localhost:11434
node bench.mjs rank --model nemotron3:33b --host http://localhost:11434
node bench.mjs math --limit 10 -v --host http://localhost:11434
node bench.mjs agent --cat recovery -v --host http://localhost:11434
```

All subcommands work identically.

## Requirements

- Node 20+ (built-in `fetch`). Node 22.5+ for `bench-data.mjs` (`node:sqlite`).
- An Ollama endpoint reachable from the runner.
- `python3` on PATH (HumanEval grading).
- Optional: `nvidia-smi`, `docker` CLI.

## Not measured

- Agentic workload *throughput*. Perf uses `/api/generate`; the tool-use probes score correctness, not speed.
- Contexts above ~8K prompt tokens in perf. Edit `CTX_SIZES` to extend.
- Multimodal (vision / audio).

## Files

| File | Role |
|---|---|
| `bench` | Host wrapper (Docker-sibling flow). |
| `bench.mjs` | CLI entry point + perf scenarios + league/routes/baseline. |
| `bench-toolcall.mjs` | Single-turn tool-call probe. |
| `bench-multiturn.mjs` | Multi-turn tool-call probe. |
| `bench-agent.mjs` | Multi-turn ReAct agent over synthetic world state. |
| `bench-math.mjs` | GSM8K + MATH-500. |
| `bench-code.mjs` | HumanEval (sandboxed). |
| `bench-knowledge.mjs` | MMLU-Pro. |
| `bench-ifeval.mjs` | IFEval (JS verifier port). |
| `bench-data.mjs` | WikiTableQuestions + WikiSQL. |
| `bench-tools.mjs` | Tool catalogue (toolcall + multiturn). |
| `bench-doctor.mjs` | System audit. |
| `bench-baseline.mjs` | Per-model baseline I/O. |
| `bench-gpu.mjs` | Background nvidia-smi sampler shared by every probe. |
| `scripts/fetch-data.mjs` | One-shot benchmark data acquisition. |
| `data/*.jsonl` | Vendored benchmark samples. |
| `baseline.json` | One per machine, keyed by model. Gitignored. |

## License

MIT. See `LICENSE`.
