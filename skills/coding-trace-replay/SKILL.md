---
name: coding-trace-replay
description: "Configure, run, and read a req-frontend replay against a live inference server. Use when choosing a trace format or frontend (session, independent), generating a canonical trace with tracegen and picking its context policy (trace-reported versus monotonic), setting arrival mode and concurrency (--arrival-mode, --rate, --max-concurrency, --max-items), choosing a wire backend (openai, vllm-tokens, sglang-tokens) and the server flags it requires, launching vLLM or SGLang for a replay, diagnosing prefix-cache preflight failures, cumulative-streaming or echoed-prompt errors, or interpreting session_runner_output.jsonl and summary.json."
---

# Coding Trace Replay

## Overview

Use this skill to drive this repo — the `session_runner` load generator and the
`tracegen` materializer. This is the *replay* side of the pipeline, distinct
from the dataset side that produces the raw traces it consumes; that lives in
[TraceLab](https://github.com/uw-syfi/TraceLab).

`README.md` is the authoritative reference and stays authoritative. This skill
is the decision path through it: which value on each axis, what that choice
commits you to, and which failures mean a real problem versus a
misconfiguration.

The runner replaces private prompt text with synthetic token IDs from a corpus
and preserves only workload *shape* — lengths, release timing, session ordering,
tool waits. Never describe its output as reproducing original prompts or model
answers.

## First Steps

1. Work from the repo root (`Cargo.toml`, `src/`, `examples/`).
2. Build once: `cargo build --release`. The binaries are `session_runner` and
   `tracegen`.
3. **Always dry-run first**: add `--dry-run` to parse the trace, print the
   workload summary, and contact no server. It catches schema and axis errors in
   under a second.
4. Decide all three axes explicitly before running. Defaults are `--trace-format
   session --arrival-mode trace-timed --backend openai`; a default that was never
   chosen is the usual source of a result nobody can interpret later.

## The Three Axes

Read `README.md` § *Configuration axes* for the full tables. The decisions:

### Axis 1 — trace format

| Choose | When |
|---|---|
| `session` | Multi-round conversations. Reads a canonical `session-execution-v2` file whose prompt split is already materialized, so nothing is left for a runtime flag to change. A raw CSV is rejected at parse — run it through `tracegen` first. |
| `independent` | One-shot requests with no shared context. |

The prefix/append split is **not** an axis of the run. It is chosen once with
`tracegen --policy` and recorded in the trace's manifest; see *Generating A
Canonical Trace* below.

### Axis 2 — arrival and capacity

Two independent sub-axes. Do not conflate them; they were deliberately split.

- **When**: `--arrival-mode trace-timed` (default, replays recorded offsets,
  rescalable with `--rate`) or `saturated` (ignores recorded offsets entirely).
  `saturated` is rejected with `--rate`.
- **How many**: `--max-concurrency N`, optional, valid under either mode. It
  caps *workload units* — a session counts while it waits on a tool, when it has
  no request in flight at all.

### Axis 3 — wire backend

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
cargo run --release --bin tracegen -- \
  --source examples/multi_session_example.csv \
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

The raw source CSV is `session-rounds-v2` from TraceLab's exporter,
`artifacts/trace_facts/csv_export/convert.py`. That exporter is deterministic
and carries no timeline — the corpus has no arrival timestamps.

So `tracegen` invents one, and every knob that shapes the workload lives here
rather than upstream: `--arrival-rate` (default `1.0`/s), `--arrival-pattern`
(`poisson` or `constant`), `--session-order` (`source` or `shuffle`),
`--max-sessions`, and `--seed`. All five land in the manifest, so a trace plus
its manifest and source hash is a complete recipe.

Selection runs before arrivals are drawn, so `--max-sessions 200` offers the
same rate as the full file instead of its densest 200-session slice.

Downstream, `session_runner --rate` rescales the recorded timeline and
`--arrival-mode saturated` discards it. Neither can tell you what rate the file
was generated at — read the manifest.

## Running

```bash
# 1. Parse only. No server contacted.
./target/release/session_runner \
  --trace trace/execution.csv --trace-format session \
  --text-file corpus.txt --tokenizer <hf-model-or-path> --model <served-name> \
  --dry-run

# 2. Replay a canonical trace through native vLLM token-in/token-out.
./target/release/session_runner \
  --trace trace/execution.csv --trace-format session \
  --text-file corpus.txt --tokenizer <hf-model-or-path> --model <served-name> \
  --base-url http://127.0.0.1:8000 --backend vllm-tokens \
  --log-path out/session_runner_output.jsonl --summary-path out/summary.json \
  --timeline-path out/timeline.parquet

# 3. Saturate under a session cap instead of replaying the timeline.
#    ... --arrival-mode saturated --max-concurrency 8

# 4. Hold the run to an objective and report attainment.
#    ... --slo 'ttft_ms=500,tpot_ms=50'

# 5. Replay a trace carrying its own per-round deadlines.
#    ... --trace-tags slo        (the file must then have deadline_ms + priority)

# 6. Find the rate where the server stops keeping up. Ramps, bisects, densifies.
./target/release/sweep --mode max-sustainable-rate --out out/sweep \
  --start-rate 5 --max-rate 800 \
  <every session_runner flag above, except --rate>

# 7. Find the most work it will ever do. A different number, past that rate.
#    ... --mode peak-throughput
```

`--tokenizer` must match the served model: it sizes the synthetic corpus in the
server's own token space.

## Interpretation Guidance

- **A cap that changes nothing means the units never overlapped**, not that the
  cap was ignored. Compare the trace's arrival spacing against per-unit
  duration; use `--arrival-mode saturated` to force overlap.
- `folded_prefix_tokens` lives in the **manifest** (planned, a trace property);
  `prefix_shortfall_tokens` lives in the **JSONL** (unplanned, a run property).
  Both are fresh work and neither is ever counted as a cache hit. A nonzero
  shortfall means a short or failed round upstream — investigate it rather than
  averaging over it.
- Compare `planned_prefix_hit_rate` against the server's
  `cached_prompt_tokens / prompt_tokens`. A gap is the interesting quantity;
  a run reported without both sides is not evidence of anything.
- Timing is client-observed. TTFT and TPOT include client queueing and network,
  and a unit held back by `--max-concurrency` starts its clock when the permit
  is granted, not when the trace says it arrived.
- `output_len` is sent as `max_tokens` with `ignore_eos: true`, so output length
  is the trace's, never the model's stopping point. Do not read output length as
  a model behaviour.
- Failed rounds stop their session by default (`--stop-session-on-error false`
  to change it), so a failure early in a long session removes every later round
  from the run. Check the completed-round count against the workload summary
  before comparing aggregates.
- The Parquet timeline (`--timeline-path`, on by default) is **one row per
  arrival, not per token**: a chunk carrying four ids is one row with
  `tokens = 4`. Averaging its rows as if each were a token measurement is the
  one mistake the schema is shaped to prevent.
- Check `timeline.dropped_requests` in `summary.json` before drawing anything
  from the timeline. Nonzero means the writer fell behind and the file is a
  sample of the run, not a record of it.
- `slo.attainment` is the fraction of steps that met **every** declared bound,
  not a percentile target. Read `unmeasured_steps` alongside it: those steps
  count against attainment but were failures or metrics that did not exist, not
  responses that came back slowly. A trace can declare its own objective in a
  `<trace>.slo.json` sidecar; `--slo` overrides it and prints that it did, so
  check `slo.source` before attributing a number to a trace.
- A trace's columns are checked against what it was **declared** to be
  (`--trace-kind`, `--trace-tags`), and an undeclared column is an error. If a
  run refuses a header, the file is not malformed — the declaration is wrong.
  Read the `expected exactly:` list in the message and add the tag the file
  needs, rather than editing the file.
- `slo.declared_deadline` counts only the rows that set `deadline_ms`. Read
  `declared_steps` against the run's total before quoting its rate: an
  almost-empty column produces a rate over a handful of rows.
- `knee` and `peak` answer different questions and are usually different rates.
  "How much load can it take" is the knee; "how much can it produce" is the peak,
  which lives past the knee and is normally a plateau. Quote the one that was
  asked for, and read `peak.plateau_low_rate` when someone wants the cheapest
  rate that still reaches peak throughput.
- `peak.decline_from_peak` above noise means the server got *worse* under more
  load, not merely no better. That is preemption or thrashing, and it is worth
  investigating rather than averaging away.
- A sweep's `knee.outcome` of `never_crossed` or `always_crossed` is a **result**,
  not a failure: the knee is outside the searched range. Neither reports a rate
  at the range's edge as the knee. Widen `--max-rate` / `--min-rate` and re-run —
  completed points are reused, so it costs only the new ones.
- Check `sweep.json`'s `contamination_warning` before comparing points. Nonzero
  means some point started warm on its predecessor's content, and the
  prefix-cache numbers across points are not comparable.
- The sweep's `points` array is every rate measured, including ones the search
  discarded on the way; `curve` is the same points sorted by rate. Plot `curve`.

## Troubleshooting

Full table in `README.md` § *Troubleshooting*. The ones that are
configuration rather than server problems:

| Symptom | Action |
|---|---|
| `prefix-cache preflight failed` | The server does not report cached prompt tokens. vLLM needs `--enable-prompt-tokens-details`; SGLang reports them by default. This is a hard abort by design — without it every hit rate silently reads zero. |
| `failed to parse a session-execution-v2 row` | The trace is a raw, unmaterialized CSV. Generate a canonical one with `tracegen` first. |
| `server streamed cumulative output` | SGLang is in default cumulative mode. Relaunch with `--stream-output`. req-frontend fails rather than reporting the distorted late-token latency. |
| `--rate` rejected with `saturated` | Pick one. To bound a saturated run, use `--max-concurrency`. |

## Reporting Guidance

When summarizing a replay run:

- Lead with the three axis values, the trace path, and — for a canonical trace —
  its manifest's policy and fold count. A run reported without its axes cannot
  be reproduced or compared.
- State attempted versus completed rounds, and name any session that stopped
  early.
- Report planned and server-observed prefix hit rates together, never one alone.
- Call timing client-observed, and say whether a concurrency cap was in effect.
- Do not describe the workload as replaying real prompts. It replays shapes.
