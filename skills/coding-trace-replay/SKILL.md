---
name: coding-trace-replay
description: "Configure, run, and read a TraceLab replay against a live inference server. Use when choosing a trace format or frontend (session-execution-v2, session, independent), generating a canonical trace with tracegen, picking a session context policy (trace-reported versus prefix-preserving/monotonic), setting arrival mode and concurrency (--arrival-mode, --rate, --max-concurrency, --max-items), choosing a wire backend (openai, vllm-tokens, sglang-tokens) and the server flags it requires, launching vLLM or SGLang for a replay, diagnosing prefix-cache preflight failures, cumulative-streaming or echoed-prompt errors, or interpreting session_runner_output.jsonl and summary.json."
---

# Coding Trace Replay

## Overview

Use this skill to drive `replay/` — the `session_runner` load generator and the
`tracegen` materializer. This is the *replay* side of the repo, distinct from the
dataset pipeline (`$coding-trace-collect` → `$coding-trace-normalize` →
`$coding-trace-sanitize` → `$coding-trace-analyze`) that produces the traces it
consumes.

`replay/README.md` is the authoritative reference and stays authoritative. This
skill is the decision path through it: which value on each axis, what that
choice commits you to, and which failures mean a real problem versus a
misconfiguration.

The runner replaces private prompt text with synthetic token IDs from a corpus
and preserves only workload *shape* — lengths, release timing, session ordering,
tool waits. Never describe its output as reproducing original prompts or model
answers.

## First Steps

1. Work from the repo root (`pyproject.toml`, `replay/`, `trace/`, `artifacts/`).
2. Build once: `cargo build --release --manifest-path replay/Cargo.toml`. The
   binaries are `session_runner` and `tracegen`.
3. **Always dry-run first**: add `--dry-run` to parse the trace, print the
   workload summary, and contact no server. It catches schema and axis errors in
   under a second.
4. Decide all four axes explicitly before running. Defaults are `--trace-format
   session --session-context-policy trace-reported --arrival-mode trace-timed
   --backend openai`; a default that was never chosen is the usual source of a
   result nobody can interpret later.
5. Use `uv run python ...` for anything on the Python side (the alignment
   launcher under `alignment/load_generator/`).

## The Four Axes

Read `replay/README.md` § *Configuration axes* for the full tables. The decisions:

### Axis 1 — trace format

| Choose | When |
|---|---|
| `session-execution-v2` | The run will be compared against a simulated run, or reproduced later. The prompt split is materialized in the file, so nothing is left for a runtime flag to change. |
| `session` | Exploring how a context policy changes the workload. This is the raw schema whose `prefix_len` is what the source *reported*. |
| `independent` | One-shot requests with no shared context. |

### Axis 2 — session context policy

Only meaningful for `--trace-format session`, and **rejected outright** with
`session-execution-v2`, whose policy was resolved at generation time.

- `trace-reported` — keep the reported prefix/input split; prefix the replayed
  conversation cannot supply is folded into fresh input.
- `monotonic` (alias `prefix-preserving`) — keep the longest prefix the
  conversation can actually supply, up to the trace's total, resetting only on
  a major compaction. Requires exact generated token IDs, so it constrains
  axis 4.

### Axis 3 — arrival and capacity

Two independent sub-axes. Do not conflate them; they were deliberately split.

- **When**: `--arrival-mode trace-timed` (default, replays recorded offsets,
  rescalable with `--rate`) or `saturated` (ignores recorded offsets entirely).
  `saturated` is rejected with `--rate`.
- **How many**: `--max-concurrency N`, optional, valid under either mode. It
  caps *workload units* — a session counts while it waits on a tool, when it has
  no request in flight at all.

### Axis 4 — wire backend

All three send the prompt as token IDs and are identical on the input side; they
differ in whether the server detokenizes on the way out.

| Backend | Endpoint | Required server flags |
|---|---|---|
| `openai` | `{base_url}/completions`, base URL ends in `/v1` | prefix caching + prompt-token usage details |
| `vllm-tokens` | `{base_url}/inference/v1/generate`, no `/v1` | `--tokens-only` |
| `sglang-tokens` | `{base_url}/generate`, no `/v1` | `--skip-tokenizer-init` and `--stream-output` (newer builds: `--incremental-streaming-output`) |

## Generating A Canonical Trace

Materialize a raw session CSV once, then hand the same bytes to every consumer:

```bash
cargo run --release --manifest-path replay/Cargo.toml --bin tracegen -- \
  --source raw_sessions.csv \
  --policy trace-reported \
  --max-sessions 200 \
  --out trace/execution.csv
```

Writes `execution.csv`, `execution.manifest.json`, and `execution.plan.json`.

**Read the manifest before quoting any cache number.** `folded_prefix_tokens`
is how much prefix the source attributed to cache that the replay must actually
prefill; on real coding-agent data this is dominated by first rounds, where the
agent resumed from context the published trace does not contain. A large fold is
not a bug, but a hit rate reported without it is misleading.

The raw source CSV comes from `$coding-trace-analyze`
(`artifacts/trace_facts/csv_export/convert.py`). Session identity there is
`(project, session_file, session_id)` — grouping by `session_id` alone merges
distinct sessions.

## Running

```bash
# 1. Parse only. No server contacted.
./replay/target/release/session_runner \
  --trace trace/execution.csv --trace-format session-execution-v2 \
  --text-file corpus.txt --tokenizer <hf-model-or-path> --model <served-name> \
  --dry-run

# 2. Replay a canonical trace through native vLLM token-in/token-out.
./replay/target/release/session_runner \
  --trace trace/execution.csv --trace-format session-execution-v2 \
  --text-file corpus.txt --tokenizer <hf-model-or-path> --model <served-name> \
  --base-url http://127.0.0.1:8000 --backend vllm-tokens \
  --log-path out/session_runner_output.jsonl --summary-path out/summary.json

# 3. Saturate under a session cap instead of replaying the timeline.
#    ... --arrival-mode saturated --max-concurrency 8
```

`--tokenizer` must match the served model: it sizes the synthetic corpus in the
server's own token space.

## Interpretation Guidance

- **A cap that changes nothing means the units never overlapped**, not that the
  cap was ignored. Compare the trace's arrival spacing against per-unit
  duration; use `--arrival-mode saturated` to force overlap.
- `folded_prefix_tokens` (planned, a trace property) and
  `prefix_shortfall_tokens` (unplanned, a run property) are both fresh work and
  neither is ever counted as a cache hit. A nonzero shortfall means a short or
  failed round upstream — investigate it rather than averaging over it.
- Compare `planned_prefix_hit_rate` against the server's
  `cached_prompt_tokens / prompt_tokens`. A gap is the interesting quantity;
  a run reported without both sides is not evidence of anything.
- Timing is client-observed. TTFT and TPOT include client queueing and network,
  and a unit held back by `--max-concurrency` starts its clock when the permit
  is granted, not when the trace says it arrived.
- `output_len` is sent as `max_tokens` with `ignore_eos: true`, so output length
  is the trace's, never the model's stopping point. Do not read output length as
  a model behaviour.
- Failed rounds stop their session by default (`--stop-session-on-error`), so a
  failure early in a long session removes every later round from the run. Check
  the completed-round count against the workload summary before comparing
  aggregates.

## Troubleshooting

Full table in `replay/README.md` § *Troubleshooting*. The ones that are
configuration rather than server problems:

| Symptom | Action |
|---|---|
| `prefix-cache preflight failed` | The server does not report cached prompt tokens. vLLM needs `--enable-prompt-tokens-details`; SGLang reports them by default. This is a hard abort by design — without it every hit rate silently reads zero. |
| `carries no context policy` | Drop `--session-context-policy`; regenerate with `tracegen --policy` instead. |
| `monotonic ... requires exact generated token IDs` | Axis 2 and axis 4 disagree. Use a native token backend, or `openai` against a server honouring `return_token_ids`. Never work around it with text re-tokenization. |
| `server streamed cumulative output` | SGLang is in default cumulative mode. Relaunch with `--stream-output`. TraceLab fails rather than reporting the distorted late-token latency. |
| `--rate` rejected with `saturated` | Pick one. To bound a saturated run, use `--max-concurrency`. |

## Reporting Guidance

When summarizing a replay run:

- Lead with the four axis values, the trace path, and — for a canonical trace —
  its manifest's policy and fold count. A run reported without its axes cannot
  be reproduced or compared.
- State attempted versus completed rounds, and name any session that stopped
  early.
- Report planned and server-observed prefix hit rates together, never one alone.
- Call timing client-observed, and say whether a concurrency cap was in effect.
- Do not describe the workload as replaying real prompts. It replays shapes.
