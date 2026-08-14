<div align="center">

<h1>req-frontend</h1>

**Replay real coding-agent workload shapes against an inference server.**

Session chains · Independent requests · Exact token-ID prompts · Prefix-cache auditing · TTFT/TPOT

[Quickstart](#quickstart) ·
[Architecture](ARCHITECTURE.md) ·
[中文架构](ARCHITECTURE.zh-CN.md) ·
[Engine setup](#engine-side-setup-guide) ·
[Configuration axes](#configuration-axes) ·
[CSV formats](#input-csv-formats) ·
[Backends](#request-backends) ·
[Metrics](#metrics) ·
[CLI reference](#cli-reference) ·
[Troubleshooting](#troubleshooting)

</div>

---

## What this runner does

`session_runner` converts a typed CSV workload into streaming generation
requests. It preserves the trace's request lengths, release timing, session
ordering, and tool waits while replacing private prompt contents with synthetic
token IDs from a user-supplied text corpus. This runner reproduces workload
shape, not original private text or model answers.

It reads a trace; it does not collect one. The public coding-agent corpus these
traces are usually derived from lives in [TraceLab][tracelab], which exports raw
session rounds; `tracegen` here turns those into the canonical execution trace
`session_runner` replays. This repository was extracted from TraceLab's
`replay/` directory and carries that history.

[tracelab]: https://github.com/uw-syfi/TraceLab

## Configuration axes

A run is described by three independent choices. Each axis answers a different
question, and nothing in one axis implies a value in another — pick one value
per axis.

| # | Axis | Selected by | Supported values |
|---|---|---|---|
| 1 | **Input-file format** — request family, row schema, and topology | `--input-file-format` | Complete names such as `text-generation-session-execution-v2` |
| 2 | **Arrival and load control** — when top-level units are released, how many run at once | `--arrival-mode`, CSV `arrival_time`, `--rate`, `--max-concurrency`, `--max-items` | `trace-timed` (default) or `saturated`, each with an optional cap |
| 3 | **Wire backend** — endpoint and output representation | `--backend` | `openai` (default), `vllm-tokens`, `sglang-tokens` |

The prefix/append split a session round replays is **not** an axis of a run. It
is resolved once, when the canonical trace is generated, and recorded in that
file's manifest — see
[Canonical execution CSV](#canonical-execution-csv--session-execution-v2).

### Axis 1 — input-file format

Two typed frontends with separate schemas. An independent request is never
rewritten into a session row with placeholder fields, and a session round is
never flattened into a standalone one.

| Value | Row means | Execution |
|---|---|---|
| `text-generation-session-execution-v2` | One **already-materialized** text-generation round | Rounds are closed-loop: submit round `i`, await its response, wait `tool_wait_after_ms`, then submit round `i + 1` |
| `text-generation-independent` | One standalone text-generation request | Each row releases independently |
| `independent` | One standalone request | One-shot; rows never share context |

`session` reads a canonical `session-execution-v2` file: its `prefix_len` is
guaranteed to exist by the time the round runs, so the runtime has nothing left
to decide and a simulator reading the same bytes reaches the same plan. Generate
one from a raw CSV with [`tracegen`](#generating-a-canonical-trace); a raw,
unmaterialized CSV is rejected at parse rather than half-read.

### Axis 2 — arrival and load control

Applies uniformly to whichever top-level unit axis 1 selected — a *session* or
an *independent request* — and never changes prompt content or context reuse.

This axis has two sub-axes that compose freely — *when* a unit may start, and
*how many* may run — plus a selection control.

| Control | Effect |
|---|---|
| `--arrival-mode trace-timed` | Default. Release offsets are replayed from CSV `arrival_time`. For `session`, only the first sorted round's value releases the session |
| `--arrival-mode saturated` | Recorded arrivals are ignored: every unit is eligible from the start. Without a cap this submits the whole workload at once; with one it is a closed-loop generator. Rejected together with `--rate`, which rescales a timeline this mode discards |
| `--rate N` | Rescale all arrivals to `N` units/s, preserving relative gaps and simultaneous-arrival bursts. Needs at least two distinct arrival times, measured after `--max-items` has been applied |
| `--max-concurrency N` | Bound concurrently active units, under either arrival mode. One session holds its slot across all of its rounds **and its tool waits** — while waiting on a tool it has no request in flight but is still occupying a slot |
| `--max-items N` | Keep the first `N` units in trace order, applied before `--rate` is measured. Rows are kept in file order; `independent` keeps the first `N` CSV rows |

`--max-concurrency` bounds *workload units*, not HTTP requests in flight. There
is deliberately no separate in-flight cap: a session is the unit a coding agent
actually is, and capping requests instead would let a third conversation start
while a second one is merely waiting on a tool.

Under a cap, units take slots strictly in trace order. Without that rule the
winner of a freed slot would be whichever task the async runtime happened to
poll first, so two runs of the same trace could admit different sessions — and
no comparison against a simulated run of the same trace would mean anything.
Since the trace is ordered by arrival, waiting for your turn never means waiting
for a unit that arrives after you.

### Axis 3 — wire backend

Transport only: endpoint, payload, and response parsing. It does not change
workload shape.

| Value | Endpoint | Output representation |
|---|---|---|
| `openai` | `POST {base_url}/completions` | Text plus vLLM's optional `return_token_ids` extension |
| `vllm-tokens` | `POST {base_url}/inference/v1/generate` | Native token-ID deltas; server must run with `--tokens-only` |
| `sglang-tokens` | `POST {base_url}/generate` | Native `output_ids` deltas; server must run with `--skip-tokenizer-init` and `--stream-output` |

All three send the prompt as token IDs; they differ only in whether the server
detokenizes. See [Request backends](#request-backends) for the comparison.

Each native token endpoint additionally requires its own server launch flags,
listed with the backend in [Request backends](#request-backends). Those are
server-side prerequisites, not constraints between axes — the axes carry none
between them, and the two rules that read like exceptions (`--rate` needs two
distinct arrival times, and is rejected alongside `--arrival-mode saturated`)
both live inside axis 2.

### Always on, not configurable

Independent of every axis above, a live run:

- sends prompts as explicit token-ID arrays built from `--text-file`, with a
  distinct pool offset per workload unit so cross-unit prefix sharing is never
  fabricated;
- sends `output_len` as `max_tokens` with `ignore_eos: true`, so output length
  is the trace's, not the model's stopping point;
- requires server-side prefix caching plus cached-prompt-token usage details,
  and aborts on a two-request preflight if they are missing — this holds for
  `independent` workloads too, because the accounting is what proves the
  planned reuse actually happened.

For what is deliberately *not* implemented, see [Current scope](#current-scope).

## Quickstart

Run these commands from the repository root.

### 1. Build

```bash
cargo build --release --bin session_runner
```

### 2. Parse and inspect a trace without a server

```bash
mkdir -p "$TMPDIR/req-frontend"

./target/release/session_runner \
  --trace examples/session_execution_v2_example.csv \
  --input-file-format text-generation-session-execution-v2 \
  --text-file unused-in-dry-run \
  --tokenizer unused-in-dry-run \
  --model dry-run \
  --dry-run \
  --max-model-len 131072 \
  --summary-path "$TMPDIR/req-frontend/dry-run-summary.json"
```

`--text-file`, `--tokenizer`, and `--model` remain required CLI arguments, but
dry-run returns before loading a tokenizer or corpus and never contacts a
server.

Dry-run performs static inspection only. It:

- parses required columns and field types;
- groups and sorts session rows;
- applies `--max-items` and optional arrival-rate scaling;
- reports workload counts, length maxima, output totals, arrivals, and waits;
- reports the first trace-target prompt plus target output that reaches
  `--max-model-len`.

It does **not** check duplicate round indices, cumulative session consistency,
the synthetic corpus, tokenizer/server identity, backend capabilities,
prefix-cache telemetry, exact output IDs, or live server behavior. Live replay
loads the corpus and checks cache telemetry, output IDs, and actual prompt
overflow. Duplicate indices, cumulative consistency, and tokenizer/server
identity are not automatically proven today.

### 3. Replay through the OpenAI-compatible backend

```bash
./target/release/session_runner \
  --trace examples/session_execution_v2_example.csv \
  --input-file-format text-generation-session-execution-v2 \
  --text-file /path/to/large-text-corpus \
  --tokenizer /path/to/model-or-tokenizer.json \
  --model meta-llama/Meta-Llama-3-8B \
  --backend openai \
  --base-url http://127.0.0.1:8000/v1 \
  --max-model-len 131072 \
  --skip-when-reaching-limit \
  --log-path "$TMPDIR/req-frontend/requests.jsonl" \
  --summary-path "$TMPDIR/req-frontend/summary.json"
```

Before any measured requests, every live run performs a two-request
prefix-cache preflight. The server must enable prefix caching and report cached
prompt-token details. For vLLM that means prefix caching left enabled (do not
pass `--no-enable-prefix-caching`) plus `--enable-prompt-tokens-details`, or
the equivalent `ENABLE_PROMPT_TOKENS_DETAILS=1`.

### 4. Use native vLLM token-in/token-out

Launch vLLM with `--tokens-only`, then change the client arguments to:

```bash
--backend vllm-tokens \
--base-url http://127.0.0.1:8000
```

This backend disables server-side detokenization and is the preferred path when
TTFT/TPOT must exclude detokenization work.

## Engine-side setup guide

req-frontend measures the full client-visible streaming path, so engine setup is
part of the measurement contract. The model engine and its HTTP frontend are
separate capacity boundaries: TP/DP and batching control EngineCore execution,
while vLLM API processes parse requests, receive engine outputs, serialize SSE
events, and drain them to clients.

### Recommended vLLM launch

For the OpenAI-compatible backend, the local TP4 measurement setup is:

```bash
python -m vllm.entrypoints.cli.main serve \
  meta-llama/Meta-Llama-3-8B \
  --tensor-parallel-size 4 \
  --api-server-count 8 \
  --stream-interval 1 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --disable-uvicorn-access-log
```

Use `--base-url http://127.0.0.1:8000/v1 --backend openai` on the req-frontend
side. For native token-in/token-out, add `--tokens-only` to the server command
and use `--base-url http://127.0.0.1:8000 --backend vllm-tokens`.

| Server setting | Why req-frontend needs it |
|---|---|
| `--enable-prefix-caching` | Enables the cache behavior audited by the mandatory two-request preflight. |
| `--enable-prompt-tokens-details` | Returns cached-token usage needed to prove the preflight and report cache alignment. |
| `--stream-interval 1` | Requests one-token streaming cadence. It does not guarantee one SSE event per token if the API process falls behind. |
| `--api-server-count N` | Adds independent HTTP API **processes**, not threads, for request parsing and streamed-output drain. |
| `--tokens-only` | Enables `/inference/v1/generate` and removes server-side detokenization from the native-token path. |

### Recommended SGLang launch

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Meta-Llama-3-8B \
  --tp 4 \
  --host 0.0.0.0 --port 30000 \
  --skip-tokenizer-init \
  --stream-output
```

Use `--base-url http://127.0.0.1:30000 --backend sglang-tokens` on the
req-frontend side — no `/v1` suffix, because `/generate` is a native route.

| Server setting | Why req-frontend needs it |
|---|---|
| `--skip-tokenizer-init` | Accepts `input_ids` and returns `output_ids` with no detokenization. The counterpart of vLLM's `--tokens-only`. OpenAI-compatible routes stop working on this server. |
| `--stream-output` | Streams disjoint deltas. Without it SGLang resends the full output every chunk, which distorts late-token latency; req-frontend detects this and fails rather than reporting it. Newer SGLang renames it `--incremental-streaming-output`. |

SGLang's radix prefix cache is on by default and reports `cached_tokens` in
`meta_info`, so the preflight needs no extra flag — unlike vLLM, which needs
`--enable-prompt-tokens-details`.

### API process sizing

Do not confuse vLLM API processes with req-frontend concurrency or req-frontend's
Tokio worker threads:

| Boundary | Control | Meaning |
|---|---|---|
| req-frontend arrival scheduler | `--arrival-mode`, CSV arrivals and `--rate` | When top-level workload units become runnable. |
| req-frontend active work | `--max-concurrency` | Maximum active sessions or independent requests, counted across tool waits. |
| req-frontend runtime | Tokio workers, reported under `client_runtime` | Polling release and HTTP client tasks. |
| vLLM HTTP frontend | `--api-server-count N` | Number of API processes draining EngineCore outputs and emitting streams. |
| vLLM EngineCore | TP/DP, batching, and token-budget flags | Model execution and engine-side queueing. |

One API process can become the bottleneck at high request rates even while the
GPU engine has capacity. Increasing req-frontend `--max-concurrency` does not fix
that server-side bottleneck. Typical evidence is:

- `arrival_release_lag_ms` remains small, proving the client released on time;
- API first-output wait grows far beyond EngineCore TTFT;
- SSE events begin carrying multiple output token IDs;
- client TTFT inflates and client TPOT diverges from EngineCore TPOT.

Eight API processes are the validated setting for the local TP4 Llama3-8B
10--300 requests/s sweep; they are not a universal default. For another host or
workload, increase the process count until API-side wait and SSE coalescing no
longer grow, then report the selected count with the benchmark results.

### Verify the effective server path

Do not trust argument parsing alone. The startup log must confirm the requested
process count, for example:

```text
Started 8 API server processes
ApiServer_0 ... ApiServer_7
```

In the vLLM fork used for alignment,
`python -m vllm.entrypoints.openai.api_server` accepts
`--api-server-count` but still enters the single-server path. Launch through
`vllm serve` or `python -m vllm.entrypoints.cli.main serve` to select the real
multi-API path. Before measuring, also confirm that the req-frontend prefix-cache
preflight passes and that the server reports the intended model, TP/DP layout,
prefix caching, token mode, and `stream_interval`.

## Input CSV formats

Select the complete parser explicitly with
`--input-file-format text-generation-session-execution-v2` or
`--input-file-format text-generation-independent`. The schemas are intentionally separate: independent requests are
not converted into fake session rows with placeholder fields, and a header is
never used to guess which schema a file is.

### Canonical execution CSV — `session-execution-v2`

The only session input, selected by `--input-file-format text-generation-session-execution-v2`. Every column is
already materialized, so the file is the whole contract and the runner has
nothing left to decide:

```csv
request_id,session_id,round_idx,arrival_time_ms,prefix_len,input_len,output_len,tool_wait_after_ms
session_0_round_000000,0,0,0.000000,0,14438,157,2672.000000
session_0_round_000001,0,1,0.000000,14435,1724,91,50.000000
```

| Column | Unit/type | Contract |
|---|---|---|
| `request_id` | String | Exactly `session_{session_id}_round_{round_idx:06}`. Validated, not trusted. |
| `session_id` | String | Opaque session identity. |
| `round_idx` | Non-negative integer | Contiguous from `0` within each session. |
| `arrival_time_ms` | Milliseconds, 6 decimals | Identical on every row of a session. The earliest arrival in the file is `0`. |
| `prefix_len` | Tokens | Reusable prefix that **will exist** when the round runs. `0` on round `0`. |
| `input_len` | Tokens | Fresh tokens appended this round. May be `0` only when a prefix exists. |
| `output_len` | Tokens | Sent as `max_tokens` with EOS ignored. |
| `tool_wait_after_ms` | Milliseconds, 6 decimals | Delay before the next round of the same session. |

What the format buys, and what it costs:

- **No runtime policy.** There is no context-policy flag on the runner at all.
  Two runtimes replaying the file do identical work without having to agree on
  anything beyond the bytes.
- **Arrivals are part of the file.** The corpus behind these traces has no
  session arrival times, so the timeline is synthesized once at generation and
  recorded in the manifest. A trace starts at its own origin.
- **Row order is the contract.** Rows of a session are contiguous, sessions are
  nondecreasing by arrival, and no consumer may re-sort by identifier — dense
  internal IDs are assigned in file order on both sides.
- The cost is that the fold decision is frozen. To ask "what if the prefix
  assumption were different", regenerate the file, do not switch a flag.

#### Generating a canonical trace

`tracegen` is a registry of generators, one subcommand each. They share
everything that is true of *every* canonical trace — validation, the CSV writer,
the plan, and the manifest's totals — and differ only in where the rows come
from. A new way of producing a trace is a new file under `bin/tracegen/generator/`,
not a new binary.

| Generator | Rows come from | Use when |
|---|---|---|
| `coding-session` | a raw `session-rounds-v2` CSV, materialized under a context policy | You are replaying something that was recorded |
| `synthetic` | distributions, no corpus at all | You want to vary one dimension and hold the rest fixed |

```bash
cargo run --release --bin tracegen -- coding-session \
  --source examples/multi_session_example.csv \
  --policy trace-reported \
  --max-sessions 200 \
  --out trace/execution.csv
```

It writes `execution.csv` plus `execution.manifest.json` and
`execution.plan.json` beside it. The manifest records which generator ran, the
totals counted from the rows that were actually emitted, and — under
`parameters` — everything that generator needed to produce them: for
`coding-session` the source hash, the policy and its thresholds, the selection
rule, the arrival synthesis, and how many tokens were folded from prefix into
fresh input. Read it before quoting any cache number, because a fold that is
large is not a bug but does mean the source attributed to cache what the replay
must actually prefill. The plan is the normalized per-round expansion, which is
what a simulator compares against to prove both sides scheduled the same work.

The totals live outside `parameters` because they are derived from the file
rather than reported by whatever made it; a generator cannot mis-state them.
Between them and `parameters`, the manifest is a complete recipe: given the same
inputs, the flags it records regenerate the trace byte for byte.

#### Drawing a trace instead of recording one

```bash
cargo run --release --bin tracegen -- synthetic \
  --sessions 500 \
  --rounds 'uniform:1..8' \
  --input-len 'lognormal:1024,0.8' \
  --output-len 'lognormal:256,0.7' \
  --compaction-probability 0.1 \
  --arrival-rate 4 \
  --out trace/synthetic.csv
```

Every length knob takes a distribution: `512` or `fixed:512` for a constant,
`uniform:256..1024` for an even sweep, `lognormal:2048,0.8` for the long right
tail real prompt and completion lengths actually have. The lognormal is
parameterized by its **median**, not by the underlying normal's mean, so
`lognormal:2048,0.8` is a claim you can check against a corpus.

A drawn trace carries the same guarantees as a recorded one — a round only
reuses a prefix an earlier round actually produced, and the same seed produces
the same file — but it is not a model of anything. It tells you how a deployment
responds to the shape you asked for, which is a different claim from telling you
how it responds to real traffic. Say which one you measured.

`--compaction-probability` is drawn per round; the manifest records the count
that *came out* (`compaction_rounds`), because on any finite file that differs
from the probability that went in, and the count is the property of this file.

#### Shaping the workload

These apply to `coding-session`, which materializes a corpus that has no
timeline of its own.

The raw trace says what each session *did*, never when it arrived — the corpus
has no arrival timestamps. So the timeline is invented here, and so is the
choice of which sessions to keep:

| Flag | Default | Effect |
|---|---|---|
| `--arrival-rate` | `1.0` | Session arrivals per second. |
| `--arrival-pattern` | `poisson` | `poisson` for exponential gaps, `constant` for even spacing. |
| `--session-order` | `source` | `source` keeps the file's order; `shuffle` permutes first. |
| `--max-sessions` | all | Keep the first N of the emitted order. |
| `--seed` | `0` | Drives both the shuffle and the Poisson gaps. |

Selection runs *before* arrivals are drawn, so `--max-sessions` shortens a trace
without compressing it: a 200-session cut offers the same rate as the full file
rather than the densest slice of it.

Randomness is a xoshiro256\*\* stream implemented in `bin/tracegen/arrivals.rs`
rather than taken from a crate, and its output is pinned by a test. Reproducing
a published trace years from now means reproducing that exact bit stream, which
a dependency free to change its default algorithm cannot promise.

Downstream, `session_runner --rate` rescales this recorded timeline and
`--arrival-mode saturated` ignores it. Neither can recover what rate the file
was generated at; that is what the manifest is for.

#### Choosing a policy

`--policy` is the one real choice, and it decides only what to preserve when the
raw trace and the replayable conversation disagree. Both values keep
`prefix_len + input_len` equal to the raw round's total, so neither changes the
workload's prompt shape — only its cache assumption.

Write `C` for the context the replay owns going in (the previous prompt *plus*
the model's real output, because a server's prefix cache holds both) and
`T = prefix_len + input_len` for the raw round's total prompt.

| Policy | Emits | Use when |
|---|---|---|
| `trace-reported` (default) | `prefix_len = min(raw_prefix, C)`, with the shortfall folded into `input_len` | You want the trace's own prompt shape, honestly cold where the source was warm |
| `monotonic` | `prefix_len = min(C, T)`, the remainder as fresh input | You want maximum realistic cache reuse |

Folding under `trace-reported` is the common case, not a fallback: a real coding
agent resumes from a system prompt and history the published trace does not
contain, so a session's first round usually reports a large prefix against an
empty conversation. Those tokens become prefill work, and the manifest counts
them.

`monotonic` starts over only on a **major compaction** — the context must drop by
at least 64,000 tokens *and* by at least 50% of `C`. Both thresholds must hold: a
large absolute drop out of a much larger context is ordinary trimming, and a
large relative drop out of a small context is noise. A reduction that misses
either one truncates `C` to `T` while keeping its exact prefix. In the generated
file a compaction round is visible as `prefix_len = 0` at `round_idx > 0`.

| `C` | `T` | Decision | `prefix_len` | `input_len` |
|---:|---:|---|---:|---:|
| 576 | 704 | Grow | 576 | 128 |
| 100,000 | 90,000 | Truncate: small reduction | 90,000 | 0 |
| 200,000 | 130,000 | Truncate: drop is under 50% | 130,000 | 0 |
| 140,000 | 70,000 | Major compaction | 0 | 70,000 |

### Independent-request CSV

The independent-request frontend accepts:

```csv
id,input_len,output_len,arrival_time
request-0,512,128,0
request-1,1024,64,10
```

All four columns are required. `arrival_time` is a millisecond offset from the
start of replay. `input_len` is the full prompt length, and `output_len` is the
requested maximum output length. Rows remain independent and never share
session context.

Select this frontend with `--input-file-format text-generation-independent`. Its schema and runtime
semantics are generic and do not depend on any simulator or trace producer.
There is no canonical variant of it: with no context to carry forward, there is
nothing for a policy to materialize.

### What an input file declares about itself

A file is read against what it says it is, never against what its header looks
like. Its complete format and orthogonal tags are shared with the simulator
through `src/schema/`:

| flag | default | meaning |
| --- | --- | --- |
| `--input-file-format` | `text-generation-session-execution-v2` | request family, base columns, loader, and structural rules |
| `--trace-tags` | none | orthogonal additions, comma-separated |

The declaration fixes the exact column set, and both directions are errors: a
missing column means the file cannot be read, and an **unexpected** one means the
file describes something the run was not told about. That second case is the one
worth having — an undeclared column is data whose author expected it to matter,
and serde would have silently dropped it.

`slo` and `priority` are independent tags. They add these columns to whichever
complete format declares them, including the canonical one:

| column | meaning |
| --- | --- |
| `ttft_slo_ms` | this row's TTFT upper bound; blank means this row declares none |
| `tpot_slo_ms` | this row's TPOT upper bound; blank means this row declares none |
| `e2e_slo_ms` | this row's submission-to-completion upper bound; blank means this row declares none |
| `priority` | added only by the `priority` tag; carried into the log and never acted on by this client |

```bash
--input-file-format text-generation-session-execution-v2 --trace-tags slo,priority
```

Selecting a format or tag this client cannot execute is refused by name rather
than half-attempted: a media format (no prompt builder exists yet, so replaying one
would mean inventing content the trace never described), the `speculative` tag
(an acceptance rate is a simulation input; against a real server it is measured,
not imposed), and the `session` tag on an independent format (multi-round replay
uses the canonical text session format). The taxonomy is shared with a simulator that has more of it
implemented, so parsing a declaration is not a promise that a replay can carry
it out.

## Arrival scheduling and load control

Release is two independent decisions: *when* a unit may start, and *how many*
may run at once. Neither implies the other, and the runner keeps them separate
so that "replay the recorded timeline, but never more than eight conversations
at once" is expressible.

### When: `--arrival-mode`

Under `trace-timed` (the default) and without `--rate`, top-level units use the
CSV arrival offsets unchanged:

- `session`: one arrival per session, taken from the first sorted round.
- `independent`: one arrival per request.

Under `saturated` the recorded offsets are ignored entirely and every unit is
eligible from the start, so what actually paces the run is `--max-concurrency`
plus how fast the server answers. This is the mode to use when the question is
"what does this workload look like at saturation" rather than "what did this
timeline do". Since it discards the timeline, it is rejected with `--rate`.

### How many: `--max-concurrency`

The cap counts *top-level units*, not requests:

- `session`: one permit per session, acquired after its arrival and held across
  every round **and every tool wait** until the session ends. A session waiting
  on a tool has no request in flight and still owns its slot.
- `independent`: one permit per request.

Permits are handed out in trace order, so which unit takes a freed slot is a
property of the trace rather than of the async runtime's scheduling that run.

### Rescaling the timeline: `--rate`

`--rate N` rescales arrivals to `N` sessions/s or requests/s. The trace rate is:

```text
(unit_count - 1) / (max_arrival - min_arrival)
```

All arrival offsets are multiplied by `trace_rate / requested_rate`, preserving
relative gaps and simultaneous-arrival bursts. Rate scaling is applied after
`--max-items` and requires at least two selected units with distinct arrival
times.

Every flag, including the ones outside this axis, is listed in the
[CLI reference](#cli-reference).

## Request backends

All three backends submit the prompt as explicit token IDs, so they are
**identical on the input side**: the server's prefix-cache keys are the exact
ids req-frontend constructed. They differ only in what comes back, and the
difference is not whether generated token IDs are available — they are, on all
three — but whether the server performs detokenization at all.

| `--backend` | Endpoint | Prompt on the wire | Output on the wire | Detokenization in the measured path |
|---|---|---|---|---|
| `openai` | `POST {base_url}/completions` | Token-ID array | Text, plus echoed IDs via `return_token_ids` | **Yes** (output side) |
| `vllm-tokens` | `POST {base_url}/inference/v1/generate` | `token_ids` | Token-ID deltas | No |
| `sglang-tokens` | `POST {base_url}/generate` | `input_ids` | `output_ids` deltas | No |

So `openai` is **token-in, but not token-out**: the server still decodes, and
the echoed IDs ride alongside the text rather than replacing it. In vLLM only
the tokens-only path disables decoding —
`sampling_params.detokenize = False` appears exactly once in the tree, behind
the `--tokens-only` flag that serves `/inference/v1/generate`. The OpenAI
completions path never sets it.

Pick accordingly: `vllm-tokens` and `sglang-tokens` are the two comparable
high-fidelity paths, and `openai` is the portable fallback whose TTFT/TPOT
include decode cost.

### OpenAI-compatible completions

```text
--backend openai
--base-url http://HOST:PORT/v1
POST {base_url}/completions
```

The request carries the prompt as an integer token-ID array, plus
`return_token_ids: true`, streaming usage, `ignore_eos: true`, and the selected
sampling settings. `return_token_ids` is a vLLM extension, not part of the
standard OpenAI completions contract. A server that ignores it still replays, but
its sessions continue from re-encoded output text — see
[Carried context and output token IDs](#carried-context-and-output-token-ids).

### Native vLLM tokens

```text
--backend vllm-tokens
--base-url http://HOST:PORT
POST {base_url}/inference/v1/generate
```

The server must be launched with `--tokens-only`. Requests contain `token_ids`
and nested `sampling_params`; streamed responses contain token-ID deltas. This
path forces `SamplingParams.detokenize = false` and removes detokenization from
the measured response path.

### Native SGLang tokens

```text
--backend sglang-tokens
--base-url http://HOST:PORT
POST {base_url}/generate
```

The server must be launched with **two** flags:

| Server flag | Why req-frontend requires it |
|---|---|
| `--skip-tokenizer-init` | The counterpart of vLLM's `--tokens-only`. The server accepts `input_ids` and returns `output_ids` without ever detokenizing. OpenAI-compatible endpoints stop working on that server, so use `/generate`. |
| `--stream-output` | Makes streamed chunks disjoint deltas. SGLang's default resends the entire output in every chunk, which is O(n²) bytes over the stream and inflates late-token latency — a measurement artifact, not just a parsing inconvenience. Newer SGLang renames this to `--incremental-streaming-output` and keeps `--stream-output` as a deprecated alias. |

The request carries `input_ids` and nested `sampling_params`
(`max_new_tokens`, `temperature`, `ignore_eos`). It deliberately omits two
fields:

- **no `model`** — an SGLang server hosts exactly one model, so `--model` is
  accepted but unused by this backend;
- **no `return_logprob`** — `output_ids` is a native top-level response field.
  Recovering IDs out of per-token logprobs instead would add compute and
  serialization to the very path being timed.

Token accounting is read from `meta_info` (`prompt_tokens`,
`completion_tokens` or `output_tokens`, `cached_tokens`) rather than an OpenAI
`usage` object, and `finish_reason` is accepted as either a string or a
`{"type": ...}` object.

Two guards fail the round rather than report a polluted number:

- a chunk that repeats every token delivered so far means the server is still
  in cumulative mode, and names the missing flag in the error;
- if the server streams more generated IDs than its own `completion_tokens`
  count, the excess is dropped **only** when it provably equals the prompt's
  tail (an echo reported in sgl-project/sglang#10896); anything else is an
  unexplained mismatch and fails. Trimmed tokens are recorded as
  `echoed_prompt_tokens`.

### Carried context and output token IDs

A session's next prompt is built from the previous prompt plus the model's real
output tokens, so that the previous-output region of the next prefix is
byte-identical to what the server cached and stays cache-hittable. That is the
only reason `prefix_len` can be claimed as reuse at all.

Both native token backends return output IDs by protocol. The `openai` backend
asks for them with vLLM's non-standard `return_token_ids` extension; a server
that ignores the extension leaves the runner to re-encode the output text and
carry those reconstructed IDs forward instead.

**Re-encoded IDs are not guaranteed to round-trip.** When they differ from what
the server actually generated, the next round's prefix no longer matches the
cached one, so the reuse the trace planned does not land — and nothing in the
run reports it. Prefer a native token backend, or an `openai` endpoint that
honours `return_token_ids`, whenever a cache number is going to be quoted. Note
also that under `vllm-tokens` / `sglang-tokens` no text is streamed at all, so
there is nothing to re-encode: a round that loses its IDs there carries an empty
output forward and shows up as a large `prefix_shortfall_tokens` on the next one.

### Alignment profile configuration

The alignment launcher exposes the same choices:

```yaml
workload:
  frontend:
    type: session
    path: ../../trace/execution.csv
  backend:
    type: vllm_tokens
  text_file: ../../trace/prompts.txt
  tokenizer: meta-llama/Meta-Llama-3-8B
```

Selecting `vllm_tokens` makes the launcher add `--tokens-only` to the paired
vLLM server. Omitting `backend` keeps the backward-compatible `openai` default.

## Synthetic token corpus

`--text-file` supplies content, while the CSV supplies shape. req-frontend tokenizes
non-empty corpus lines once with `add_special_tokens = false`, stores the
resulting IDs in a shared pool, and performs all later prompt construction in
ID space. The tokenizer must match the served model.

Each workload unit starts at a different pool offset to avoid fabricated
cross-session prefix sharing. The default pool size is at least:

```text
max(2 * longest_trace_prompt, workload_unit_count, 100,000,000 tokens)
```

The 100M-token floor consumes about 400 MB for the `u32` ID pool and generally
requires roughly 400–600 MB or more of source text. `--token-pool-limit` can
reduce it. If the corpus produces fewer IDs than the longest prompt, req-frontend
warns that content will repeat within a request and may distort prefix-cache
measurements. Monotonic construction also keeps every actual prompt at the
trace-reported target `prefix_len + input_len`.

The corpus is tokenized line by line, so concatenated pool IDs need not equal a
single tokenizer call over the original whole file. This creates synthetic
boundary transitions but no client/server mismatch: the exact resulting IDs are
sent directly to the server.

For large-context tests, a large public corpus such as `enwik9` is suitable.
req-frontend does not bundle it; obtain it from its original distributor and follow
the applicable license.

## Context limits and prefix-cache preflight

### Context limits

`--max-model-len N` adds context-limit information to dry-run output.
`--skip-when-reaching-limit` requires that flag and reserves at least one token
of headroom. A live request is skipped when:

```text
actual_prompt_len + output_len_target >= max_model_len
```

Equality deliberately skips. req-frontend does not silently shorten the requested
output to fit. An independent request is logged as skipped and replay continues.
A skipped session round is logged and that session terminates, because the
missing model output makes subsequent context continuation untrustworthy.

`--skip-on-context-limit` and the older `--fail-on-context-overflow` are
compatibility aliases for the same behavior.

The requested context length is `prefix_len + input_len + output_len`. The live
guard uses the actually constructed prompt plus target output, which differs from
the trace's own numbers only when `prefix_shortfall_tokens` is nonzero.

### Prefix-cache preflight

Before every non-dry run, req-frontend sends the same 512-token-or-smaller probe
twice and requires the second response to report a positive cached-token count.
A single probe cannot separate "prefix caching is off" from "the cache is
cold", so the run aborts unless the second response proves a hit. Both probe
requests carry `X-data-parallel-rank: 0` so vLLM data-parallel deployments hit
the same cache shard.

The probe is taken from the **tail** of the synthetic token pool, not its head.
Workload unit 0 seeds at pool offset 0, so a head probe would warm that unit's
first prompt and fabricate a cache hit in the measured population.

This means a live run requires all of the following, even for an
independent-request workload:

- prefix caching enabled;
- streaming usage returned;
- cached prompt-token details present in usage.

Preflight is the only place missing telemetry is fatal. Once past it, a
per-request response that carries a usage block but no cache detail is recorded
as `cached_prompt_tokens: 0` — a real "nothing was cached" reading on servers
that omit the field when the count is zero. Preflight exists precisely so that
this zero-fill cannot mean "the server never reports cache detail at all".

## Output contracts

### Per-request JSONL — schema v11

`--log-path` receives one typed record per attempted request. Session and
independent-request source data are tagged variants rather than one sparse
object.

v11 replaces the single per-row completion deadline with metric-specific
`declared_ttft_slo_ms`, `declared_tpot_slo_ms`, and `declared_e2e_slo_ms`.
`declared_priority` now comes from a separate `priority` tag. Every undeclared or
blank value is absent rather than null, so a reader can distinguish a bound the
row never declared from a bound it violated.

v9 is the first non-additive revision: it **removes** three session fields that
described a decision the runtime no longer makes. `session_context_policy`,
`folded_prefix_tokens`, and `major_compaction` were all properties of how a raw
trace had been materialized, and materialization now happens once in `tracegen`,
which reports all three in the canonical trace's manifest — as a policy name, a
run-wide fold total, and a compaction count. A per-round compaction is still
recoverable from the trace itself: it is `prefix_len = 0` at `round_idx > 0`.

What remains on the record is the run's own behaviour:

- `prefix_shortfall_tokens` — prefix the trace declared that the *live*
  conversation could not supply, filled with fresh ids instead. Now the only
  place a live run departs from the file it is replaying, so a nonzero value
  means a short or failed round upstream, never a trace property. It is never
  counted as a cache hit.
- `derived_prefix_len` / `derived_append_len` — the split actually built. Equal
  to the trace's own numbers unless `prefix_shortfall_tokens` is nonzero.

v7 added `outcome.echoed_prompt_tokens`: leading generated IDs that repeated the
prompt tail and were dropped before carry-forward. It is `0` on every server
that does not echo, which is all of them today apart from the SGLang case
described under [Request backends](#request-backends). Consumers reading
`outcome.status` or `outcome.request_id` are unaffected by any of this.

Abbreviated session example:

<details>
<summary><b>View a schema-v11 session record</b></summary>

```json
{
  "schema_version": 11,
  "source": {
    "type": "session_round",
    "data": {
      "session_id": "0",
      "round_idx": 1,
      "prefix_len": 576,
      "input_len": 128,
      "target_prompt_len": 704,
      "prompt_len": 704,
      "derived_prefix_len": 576,
      "derived_append_len": 128,
      "prefix_shortfall_tokens": 0,
      "planned_prefix_hit_rate": 0.8181818182,
      "output_len_target": 64,
      "tool_wait_after_ms": 100.0,
      "arrival_time_ms": 0.0,
      "declared_ttft_slo_ms": 500.0,
      "declared_tpot_slo_ms": 50.0,
      "declared_e2e_slo_ms": 2000.0,
      "declared_priority": 0
    }
  },
  "outcome": {
    "request_id": "session_0_round_000001",
    "status": "SUCCESS",
    "output_len_actual": 64,
    "first_token_id_ms": 18.4,
    "token_delivery_tpot_ms": 3.2,
    "response_complete_ms": 220.1,
    "total_duration_ms": 220.3,
    "server_usage": {
      "prompt_tokens": 704,
      "completion_tokens": 64,
      "cached_prompt_tokens": 576,
      "uncached_prompt_tokens": 128,
      "prefix_hit_rate": 0.8181818182
    }
  }
}
```

</details>

Independent-request records use
`source.type = "independent_request"` and include
`arrival_release_lag_ms`, measured from scheduled arrival until the Tokio task
resumes, before any configured concurrency semaphore wait.

`outcome.request_id` is also sent as the `x-request-id` header, so it is the
join key against server-side logs. Its shape depends on the frontend:

| Frontend | `request_id` | Example |
|---|---|---|
| `session` | `session_{session_id}_round_{round_idx:06}` | `session_0_round_000001` |
| `independent` | `independent_{id}` | `independent_request-0` |

Both frontends namespace their ids with the frontend name. That prefix is
this client's, not the corpus's: the `session_id` column keeps whatever the dataset
called the session, which in the published corpus is a bare integer that would
otherwise make `0_round_000001` say nothing about what it identifies. Only
`round_idx` is zero-padded.

A canonical trace carries `request_id` as a column, and the runner validates that
it matches this form rather than trusting it — so the join key against server logs
is the same string in the trace, the client log, and the server log.

### Run summary

`--summary-path` writes one JSON document containing:

- `workload`: parsed shape and optional trace-target overflow information;
- `replay.common`: success/failure counts, throughput, TTFT, TPOT, and E2E;
- `replay.prefix_cache`: session-only planned-versus-server cache accounting;
- `client_runtime`: Tokio worker count and sampled global injection-queue peak;
- `timeline`: what the per-event timeline recorded, and what it had to drop;
- `slo`: attainment against the declared objective, or `null` when none was.

The queue-depth metric does not include every worker-local runnable queue. Use
it together with `arrival_release_lag_ms` and OS CPU/thread evidence.

### Per-event timeline — Parquet

`--timeline-path` (on by default; `--timeline false` disables) records **when
each thing arrived** on each request's stream:

| column | type | meaning |
| --- | --- | --- |
| `request_id` | `utf8` | the request this arrival belongs to |
| `seq` | `uint32` | position within that request's stream, from 0 |
| `elapsed_ms` | `float32` | milliseconds after the request was sent |
| `kind` | `utf8` | `tokens`, `usage`, `finish`, or `other` |
| `tokens` | `uint16` | tokens delivered by **this one arrival** |
| `cumulative_tokens` | `uint32` | tokens delivered by this arrival and all earlier ones |

**One row per arrival, not per token.** A chunk carrying four token ids is one
observable instant; four rows sharing a timestamp would invite a reader to
average them as four measurements. `tokens` says how many that one arrival
carried — the same reasoning that keeps `first_token_event_tokens` out of TPOT's
denominator.

`kind` is what lets this generalize. A multimodal pipeline reporting per-stage
progress adds kinds here rather than a second file format.

Measured at 4.7 bytes/row after dictionary encoding and zstd, so the full
357k-round corpus (~187M output tokens) lands under a gigabyte.

#### It cannot slow submission

The measurement is free — the fold already times every event. The write path is
arranged so the recording is too:

- events are pushed into a `Vec` preallocated to the request's target output
  length, so nothing allocates, formats or writes while a response streams;
- the whole `Vec` is handed off **once per request**, not once per token;
- the handoff is a `try_send` on a bounded channel. A full channel drops that
  request's timeline and counts it, because a slow disk must never become
  backpressure on the thing being measured;
- all Arrow encoding and compression happens on a thread of the writer's own.
  Not an async task: zstd is blocking CPU work, and on a runtime worker it
  competes with the requests being timed. That was measurable — see below.

This claim is maintained by `selfcheck`, not by a pasted benchmark table. It
alternates timeline-on and timeline-off runs against `tools/stub_server.py`,
then checks the p50/p99 difference and requires zero dropped request timelines:

```bash
cargo run --release --bin selfcheck -- \
  --tokenizer /path/to/tokenizer.json \
  --out "$TMPDIR/req-frontend-selfcheck"
```

The same harness separately checks scheduled release lag, rate scaling, TTFT,
TPOT, end-to-end time, prompt/output lengths, server/client token accounting,
and planned prefix hits. Every result, expected bound, and tolerance rationale
is written to `selfcheck.json`; any failed claim makes the command exit nonzero.

`timeline.dropped_requests` in the run summary is nonzero exactly when the file
is a sample of the run rather than a record of it. A run also prints a warning
in that case, so a lossy timeline says so instead of quietly under-reporting.

### Service-level objectives

An objective is a set of **upper bounds**, and the number it produces is an
**attainment rate**: the fraction of steps that met every bound declared.

```bash
--slo 'ttft_ms=500,tpot_ms=50'
```

| metric | bound on |
| --- | --- |
| `ttft_ms` | time to first token, from the moment the request was sent |
| `tpot_ms` | client-observed delivery time per output token after the first timed event |
| `e2e_ms` | submission to completion |

Not a percentile target. "p99 TTFT under 500 ms" hides how many requests were
bad, and a run of two hundred thousand rounds has room to hide a great deal.

Three scopes are supported, and the summary records which applied:

| scope | how | `slo.source` |
| --- | --- | --- |
| global | `--slo` on the command line | `global` |
| per trace | a `<trace>.slo.json` sidecar beside the trace file | `trace` |
| per request | the `slo` tag's three metric-specific columns | absent — see below |

The first two set the same bounds for every row, which is why they are a
sidecar rather than a column repeated N times. The third is genuinely per-row
and reported separately, under `slo.declared_slo`:

```text
slo attainment | source=TraceRows steps=30 overall=0.7333 |
  declared_slo 0.5556 (18 of 30 rows, 8 violated, 0 unmeasured)
```

Each row is judged only on the TTFT, TPOT, and E2E bounds it filled in. A row
with all three cells blank is outside the per-request denominator; counting it
as attained would let an almost-empty declaration report near-perfect
attainment. A step is attained overall when it met every applicable run-level
and per-request bound. A trace declaring the tag needs no `--slo` to produce
attainment.

The sidecar is the same convention `.manifest.json` and `.plan.json` already
use, so a trace and everything true about it travel together:

```json
{ "ttft_ms": 500, "tpot_ms": 50 }
```

`--slo` overrides a sidecar and says so on stderr, because a trace that declares
an objective and a run that ignores it is exactly where an unaccountable number
comes from. A misspelled metric name, an empty spec, and a sidecar declaring no
bounds are all errors — a run that reports 100% attainment because it was
quietly asked for nothing is the worst failure available here.

Every step is judged on all declared bounds, and the fold happens in the same
pass that produces the replay percentiles, on the same clocks. Three outcomes
per metric:

| | meaning |
| --- | --- |
| attained | measured at or under the bound |
| violated | measured past the bound |
| unmeasured | the step failed, or succeeded without producing that metric (a one-event response has no measurable TPOT) |

Unmeasured counts **against** attainment and is reported separately, so nobody
reads "94% attained" as "6% were slow" when it was "6% never answered". A failed
step is never attained: a response that did not arrive did not arrive on time
either.

```text
slo attainment | source=Global steps=48 overall=0.6667 |
  ttft_ms<=5 1.0000 (0 violated, 0 unmeasured) |
  e2e_ms<=60 0.6667 (16 violated, 0 unmeasured)
```

`SloSpec` and the attainment fold live in `src/schema/slo.rs`, alongside the
trace schemas rather than in the runtime, because a measured replay and a
simulated run must report the same number for the same trace.

## Rate sweeps — `sweep`

A grid sweep asks you to know the answer before you start: too coarse and the
knee falls in a gap, too fine and most of the points are spent far from it. The
`sweep` binary ramps by doubling until the boundary flips, bisects back to locate
it, then spends its remaining points **at** the knee.

```bash
cargo build --release --bin sweep

./target/release/sweep --mode max-sustainable-rate --out out/sweep \
  --start-rate 5 --max-rate 800 --tolerance 0.05 --densify-points 3 \
  --trace trace/execution.csv --input-file-format text-generation-session-execution-v2 \
  --text-file corpus.txt --tokenizer <hf-model-or-path> --model <served-name> \
  --base-url http://127.0.0.1:8000 --backend vllm-tokens
```

Every point is a full run in **this process**, which is what lets a twenty-point
sweep pay for the tokenizer and the hundred-million-token synthetic corpus once
instead of twenty times.

### Modes

| `--mode` | Question | Answer shape |
| --- | --- | --- |
| `max-sustainable-rate` | how much load can this deployment take? | one rate: `knee` |
| `peak-throughput` | how much work can it produce? | a value and a rate span: `peak` |
| `max-rate-under-slo` | how much load can it take *and still be acceptable*? | one rate: `knee` |
| `grid` | run these rates | no search |

**The first two are not the same question, and usually not the same rate.** The
sustainable rate is where the server stops keeping up. Peak throughput lives
*past* that point — on a server whose batch grows with load, throughput keeps
climbing long after latency has stopped being acceptable — and it is normally a
plateau rather than a point. Reporting one number for both would answer
whichever question you did not ask.

#### `max-sustainable-rate`

Crossed when delivered request throughput falls more than `--max-shortfall`
behind the offered rate. Ramp, bisect, densify.

**A boundary judges one point, alone** — no history, no neighbours. That
restriction is deliberate, and it came from getting it wrong first. "Throughput
stopped rising over the best seen so far" reads like the textbook definition of
saturation and is order-dependent: the same rate judged before and after a higher
one gives opposite answers, so a bisection narrows on an artifact of visit order.
Sorting the points by rate fixes the order-dependence but not the deeper problem
— a *relative-gain* test is not a property of a rate at all, but of that rate
plus whichever lower rates you happened to measure, and bisection's whole job is
to keep changing that set. On the verified curve below, the relative-gain
formulation puts the knee at ~80/s where the true value is ~48/s, purely because
the doubling ramp landed on 80.

Stating it as *delivered against offered* is order-independent, grid-independent,
and the more direct claim: a saturated server is one that cannot complete work as
fast as it arrives. It is also exactly the sorted-curve test — the delivered
curve's distance from `y = x` — expressed so one point can be judged without its
neighbours.

One artifact to know about: the run window runs from the first submission to the
last completion, so it always includes one request's latency after the last
arrival. Delivered throughput therefore falls short of offered by roughly
`latency × rate / units` even on a server that kept up perfectly. Use enough
workload units that the arrival span dominates that tail.

#### `peak-throughput`

Ramps while throughput keeps improving by `--min-gain`, stopping after
`--patience` consecutive points that do not — more than one, so a single noisy
run does not end the search. It deliberately keeps going past the sustainable
rate, because that is where the peak is.

Reported as a region: `peak_throughput` and the rate that produced it, plus
`plateau_low_rate` / `plateau_high_rate`, the *contiguous* span within
`--plateau-tolerance` of the peak. The lower edge is the actionable number — the
cheapest rate that still gets peak throughput — so densification spends its
points closing the octave-wide gap below it rather than drawing more of the flat
top.

`decline_from_peak` is how far throughput at the highest rate measured had fallen
below the peak. Positive beyond noise means the server got *worse* under more
load — preemption, cache thrashing — which is a finding a sweep that stopped at
the knee could never make.

`--peak-metric` chooses `output-tokens` (default) or `requests`. With variable
output lengths these peak in different places.

#### `max-rate-under-slo`

Crossed when SLO attainment falls below `--target-attainment`.
`--attainment-metric declared-slo` watches the trace's own per-request metric
bounds instead of the run-level objective. Either way the run must have an
objective; a sweep whose every point reports `null` attainment fails rather than
ramping to the ceiling and announcing a knee it never tested for.

### What comes out

`sweep.json` records **every** point in the order it was measured — including
ones the search discarded — plus the same points sorted by rate as `curve`, the
phase that asked for each rate, the verdict in words, and the located knee:

```json
{
  "knob": "rate",
  "objective": "request_throughput_per_s",
  "boundary": { "mode": "max-sustainable-rate", "max_shortfall": 0.1 },
  "knee": {
    "outcome": "located",
    "last_good_rate": 47.5,
    "first_bad_rate": 50.0,
    "bracket_width": 0.05
  },
  "peak": null
}
```

`knee` is set by the boundary-searching modes and `peak` by `peak-throughput`;
they are separate fields because they are separate answers.

Three knee outcomes, and the two that are not `located` are findings rather than
failures: `never_crossed` means the knee is above `--max-rate`, and
`always_crossed` means it is below `--min-rate`. Neither is reported as a knee at
the range's edge, because that would be a fabrication. `peak` likewise reports
`still_rising_at_max_rate` rather than calling the ceiling's throughput the peak.

Both modes run against `tools/stub_server.py --capacity 4 --chunk-delay-ms 2`
(400 requests × 40 tokens, so capacity is arithmetic) give the two different
answers they should:

| mode | answer |
| --- | --- |
| `max-sustainable-rate` | knee bracketed to [47.5, 50.0]/s |
| `peak-throughput` | 43.3 req/s, flat from 50/s to 320/s |

Each point gets its own directory under `points/rate_*/` with the full
`requests.jsonl`, `summary.json` and `timeline.parquet` of that run, plus a
`point.json` written **last and only on success** — its presence is what a
resumed sweep reads as "already done". Re-running the same command reuses those
points; `--no-resume` re-measures everything.

### Server state between points

Point *k+1* would otherwise start warm on point *k*'s content, and its measured
prefix-cache rate would not be comparable to anything. The sweep calls the
endpoint's cache reset before each point (vLLM's `/reset_prefix_cache`, which
sits on the server root beside the API rather than inside it, so a `/v1` base URL
is stripped) and records per point what happened:

| `cache_reset.state` | meaning |
| --- | --- |
| `done` | the endpoint accepted the reset |
| `unsupported` | this backend exposes no reset this repo has verified |
| `failed` | the endpoint has one and it did not work |

Anything other than `done` on any point produces a `contamination_warning` in
`sweep.json` and on stderr, naming how many points were affected — a
contaminated curve reported as a clean one is the failure worth preventing here.
SGLang is `unsupported`: it has `/flush_cache`, but not with the same meaning on
every version, and a wrong guess would report a reset that never happened.

## Plotting a sweep — `viz/`

An optional Python sidecar with its own `pyproject.toml`, run under `uv`. It is
never invoked by a sweep and nothing in `src/` knows it exists: a missing
plotting dependency must not be able to cost anyone a measurement.

```bash
cd viz
uv run viz ../out/sweep            # or the sweep.json inside it
```

Four figures, written beside the report under `figures/`:

| File | Shows |
|---|---|
| `throughput.png` | delivered steps against offered load, with the knee bracket or peak plateau shaded, and the dashed reference a perfect server would have followed |
| `attainment.png` | SLO attainment against offered load — the run's objective and per-request metric bounds as separate series, never merged |
| `latency.png` | TTFT, TPOT and end-to-end distributions per rate, as boxes with p99 marked |
| `arrivals_rate_*.png` | when each token actually arrived, per request, for the lowest and highest rates measured |

What the figures will not do is as much the point as what they will. A metric no
point reported produces a figure saying so rather than empty axes; a null point
is counted, never interpolated across; a knee that was `never_crossed` is not
drawn at the edge of the range; and every figure prints the sweep's caveats —
cache contamination, dropped timelines, reused points — under the axes, so a
contaminated run cannot be screenshotted without them.

The token-arrival figure is drawn as a **step** function with a marker per
arrival, because a chunk carrying four ids is one observable instant and not
four. The flat treads are the waits and the risers are the arrivals; the riser
height is the chunk size.

```bash
uv run pytest                      # the transforms; the plots are checked by eye
```

## Metrics

### Timing boundaries

| Metric | Definition |
|---|---|
| `first_token_id_ms` | HTTP send to the first event carrying generated token IDs. |
| `first_token_ms` | HTTP send to first non-empty text event; retained as a legacy/fallback boundary. |
| `token_delivery_tpot_ms` | First-to-last token-ID event span divided by tokens delivered after the first event. |
| `response_complete_ms` | HTTP send to SSE `[DONE]` or EOF, before output re-tokenization and log shaping. |
| `terminal_tail_ms` | Response-completion time after the last token-ID event. |
| `total_duration_ms` | Full client step from request entry through response processing and bookkeeping. |

Canonical summary TTFT prefers `first_token_id_ms` and falls back to
`first_token_ms` only when token IDs are unavailable. The summary exposes
`ttft_token_id_steps` and `ttft_text_fallback_steps` so mixed populations remain
visible.

Canonical TPOT is token-event delivery cadence, not per-token ITL. One SSE event
may contain several IDs; all IDs in the first event share one observable time
boundary and are excluded from the denominator. The historical
completion-amortized calculation remains under
`completion_amortized_tpot_*`; it includes terminal and client-side tail work
and is not canonical TPOT.

The two also anchor on different first-token boundaries. Canonical TTFT prefers
`first_token_id_ms`; `completion_amortized_tpot_*` prefers `first_token_ms` and
only falls back to the token-ID boundary. On the `openai` backend both
boundaries exist and differ slightly, so the audit metric is anchored on the
text event while TTFT is anchored on the ID event. On `vllm-tokens` there are no
text events and both use the ID boundary.

### Throughput and the run window

```text
run_duration_ms         = max(complete_timestamp) - min(submit_timestamp)
request_throughput_per_s      = success_steps / run_duration_s
output_token_throughput_per_s = successful output tokens / run_duration_s
```

The window spans **every attempted** request, while both numerators count
**successes only**. A failed or context-skipped request therefore widens the
denominator without contributing to the numerator, which reads as lower
throughput rather than as an error. Check `failed_steps` and
`context_overflow_steps` before quoting a throughput number.

`run_duration_ms` is also a client-side span across wall-clock timestamps, not
a server-side serving window: it includes trace arrival gaps, tool waits, and
any time spent blocked on `--max-concurrency`.

### Offered units against delivered steps

The two sides of a saturation test are counted in different things, and the
conversion is not optional:

```text
offered_step_rate = rate * steps_per_workload_unit
```

`--rate` offers **workload units** per second — a session for a session trace, a
request for an independent one — while every throughput above counts **steps**,
and a session issues several rounds. `steps_per_workload_unit` is the trace's
own ratio, reported in `RunMetrics` and on every curve entry, and it says what a
server that kept up perfectly would have had to deliver.

Comparing the raw numbers instead reads a saturated server as keeping up with
room to spare, by exactly the mean rounds per session. On a two-round trace that
put the reported knee at more than twice the true one.

### Prefix-cache accounting

For session rounds:

```text
planned_prefix_hit_rate = derived_prefix_len / prompt_len
server_prefix_hit_rate = server_cached_prompt_tokens / server_prompt_tokens
server_prefix_hit_rate_delta = server_prefix_hit_rate - planned_prefix_hit_rate
```

Aggregate summaries compare only rounds for which the server reports both
cached and total prompt tokens. In practice that is every round with a usage
block, because a missing cache detail is recorded as zero cached tokens; a
round drops out of the comparison only when the response carried no usage at
all. `planned_*_for_measured_cache_steps` re-accumulates the plan over exactly
the compared rounds, so the delta is never a plan/measurement population
mismatch.

vLLM's cache blocks can make measured cached tokens differ slightly from a
token-level plan.

## Troubleshooting

| Symptom | Meaning and action |
|---|---|
| `prefix-cache preflight failed` | Enable prefix caching and prompt-token usage details — for vLLM, `--enable-prompt-tokens-details` (or `ENABLE_PROMPT_TOKENS_DETAILS=1`) with prefix caching left on. Confirm both probe requests reach the same server/cache shard. |
| `failed to parse a session-execution-v2 row` | The file is a raw, unmaterialized CSV. Run it through `tracegen coding-session` first; `--input-file-format text-generation-session-execution-v2` reads canonical traces only. |
| `server streamed cumulative output` | The SGLang server is in its default cumulative streaming mode. Relaunch it with `--stream-output` (named `--incremental-streaming-output` in newer builds). |
| `the extra leading ids do not match the prompt tail` | The server streamed more generated IDs than its own `completion_tokens` count, and the excess is not an echo of the prompt. req-frontend refuses to guess what those IDs are; inspect the raw response before trusting the run. |
| `--rate ... --arrival-mode saturated` rejected | One rescales the recorded timeline, the other discards it. To bound a saturated run, use `--max-concurrency`. |
| `--max-concurrency` appears to change nothing | The units never overlap, so the cap never binds. Check the trace's arrival spacing against its per-unit duration; to force overlap, use `--arrival-mode saturated`. |
| `cannot apply --rate` | The selected workload has fewer than two distinct top-level arrival times. |
| Token-pool repetition warning | Supply a larger corpus or increase `--token-pool-limit`. |
| `SKIPPED_CONTEXT_OVERFLOW` | Compatibility status name: actual prompt plus target output reached `--max-model-len` while `--skip-when-reaching-limit` was enabled. The request was not sent. |
| TTFT/TPOT missing | The response carried insufficient token or text events for that metric's denominator. Inspect per-request event counters and usage. |

Use `--dry-run` first for every new trace. It catches CSV/type errors and reports
the selected workload shape without consuming model capacity.

## CLI reference

Every flag `session_runner` accepts. The axis columns map back to
[Configuration axes](#configuration-axes).

### Required

| Flag | Value | Notes |
|---|---|---|
| `--trace` | Path | Source CSV, interpreted by `--input-file-format` |
| `--text-file` | Path | Synthetic token corpus. Required even for `--dry-run`, which never opens it |
| `--tokenizer` | Path or HF repo id | `tokenizer.json`, a directory containing one, or a repo id to download. Must match the served model |
| `--model` | String | Model name placed in the request payload. Accepted but unused by `sglang-tokens`, whose server hosts one model and takes no model field |

### Axis selection

| Flag | Default | Values |
|---|---|---|
| `--input-file-format` | `text-generation-session-execution-v2` | Complete family-specific format. The HTTP client currently executes the two `text-generation-*` formats |
| `--trace-tags` | none | Comma-separated. `slo` adds TTFT/TPOT/E2E bounds; `priority` adds scheduling priority |
| `--backend` | `openai` | `openai`, `vllm-tokens`, `sglang-tokens` — axis 3 |
| `--base-url` | `http://127.0.0.1:8000/v1` | Include `/v1` for `openai`, omit it for the native token endpoints |

### Load control (axis 2)

| Flag | Default | Notes |
|---|---|---|
| `--arrival-mode` | `trace-timed` | `trace-timed`, `saturated`. `saturated` is rejected with `--rate` |
| `--max-items N` | unlimited | Alias `--max-sessions`. See the axis-3 table for its per-frontend selection order |
| `--rate N` | trace arrivals unchanged | Units/s. Needs at least two distinct arrival times after `--max-items` |
| `--max-concurrency N` | unlimited | Must be greater than `0`; `0` is rejected at startup. Caps active units — a session counts while it waits on a tool |

### Generation

| Flag | Default | Notes |
|---|---|---|
| `--temperature X` | `0` | Applies to both backends |
| `--stream-idle-timeout-secs N` | `600` | Fail the request when no stream chunk arrives within this interval |

### Context guard

| Flag | Default | Notes |
|---|---|---|
| `--max-model-len N` | unset | Enables dry-run overflow reporting on trace targets |
| `--skip-when-reaching-limit` | off | Requires `--max-model-len`. Aliases: `--skip-on-context-limit`, `--fail-on-context-overflow` |

### Failure handling

| Flag | Default | Notes |
|---|---|---|
| `--stop-session-on-error` | `true` | A session stops after its first failed round. Takes an explicit value: `--stop-session-on-error false` |

A context-limit skip always ends its session, independently of this flag.

### Output

| Flag | Default | Notes |
|---|---|---|
| `--log-path` | `session_runner_output.jsonl` | Per-request JSONL, flushed per record |
| `--summary-path` | unset | No JSON summary is written unless given. Also written by `--dry-run` |
| `--dry-run` | off | Static inspection only; returns before the tokenizer, corpus, and preflight |
| `--token-pool-limit N` | see [Synthetic token corpus](#synthetic-token-corpus) | Cap on synthetic pool size |
| `--timeline` | `true` | Per-event Parquet timeline. Takes an explicit value: `--timeline false` |
| `--timeline-path` | `session_runner_timeline.parquet` | Where that timeline is written |
| `--slo` | trace sidecar, else none | `ttft_ms=500,tpot_ms=50`. Overrides a `<trace>.slo.json` sidecar. See [Service-level objectives](#service-level-objectives) |

## Repository structure

<details>
<summary><b>Contributor-facing module map</b></summary>

```text
.
├── examples/                 canonical trace + raw tracegen inputs
├── skills/                   agent-facing operating guide for a replay
├── tools/                    stub server for measuring the client alone
├── viz/                      optional Python plotting; nothing here depends on it
├── src/
│   ├── lib.rs                the public library: the trace schemas + run_once
│   ├── release.rs            shared trace-timed/saturated eligibility vocabulary
│   ├── schema/
│   │   ├── input_file_schema.rs  complete format + orthogonal tags
│   │   ├── format/           one typed row/validator/loader per family format
│   │   ├── family/           media and omni family-specific declarations
│   │   └── tag/              per-request SLO, priority, speculative declarations
│   ├── workload.rs           runtime selection, item limit, arrival scaling, stats
│   ├── slo.rs                runtime measurement, attainment, and aggregation
│   ├── runner.rs             one run: validate, load, preflight, fan out, fold
│   ├── slo_source.rs         --slo, the trace's sidecar, and which one won
│   ├── main.rs               argument parsing; one call into the library
│   ├── cli.rs                public CLI contract
│   ├── backend/
│   │   ├── mod.rs            normalized request/response vocabulary + Backend
│   │   ├── wire/             one file per protocol: openai, vllm, sglang
│   │   ├── client.rs         the protocol-blind streaming engine
│   │   ├── stream.rs         the response fold, testable without a server
│   │   ├── integrity.rs      may this response be believed at all
│   │   └── preflight.rs      the prefix-cache gate run once before a workload
│   ├── bin/tracegen/
│   │   ├── main.rs           shared path: validate, write CSV, plan, manifest
│   │   ├── generator/
│   │   │   ├── mod.rs        the Generator trait and the registry
│   │   │   ├── coding_session.rs  raw recorded trace -> canonical rows
│   │   │   ├── synthetic.rs  rows drawn from distributions, no corpus
│   │   │   └── distribution.rs   fixed / uniform / lognormal, parsed and recorded
│   │   ├── arrivals.rs       seeded arrival synthesis + session selection
│   │   └── policy.rs         context-policy arithmetic (coding-session only)
│   ├── bin/sweep/
│   │   ├── main.rs           CLI, orchestration, sweep.json
│   │   ├── search.rs         ramp -> bisect -> densify: locate a boundary
│   │   ├── peak.rs           ramp past it: locate a maximum and its plateau
│   │   ├── boundary.rs       what "crossed" means, judged one point at a time
│   │   └── point.rs          one point: reset the server, run, record, resume
│   ├── bin/selfcheck/            executable fidelity claims against the stub
│   ├── executor/
│   │   ├── mod.rs            run policy, shared state, counters, status task
│   │   ├── admission.rs      declaration-order admission under a cap
│   │   ├── session.rs        ordered closed-loop session executor
│   │   └── independent.rs    one-shot independent-request executor
│   ├── trace/
│   │   ├── mod.rs            frontend dispatch + arrival-rate scaling
│   │   ├── session.rs        canonical session trace loader
│   │   └── independent.rs    independent-request CSV parser
│   ├── tokens.rs             synthetic ID pool + session prompt assembly
│   ├── record.rs             versioned typed JSONL records
│   ├── timeline/
│   │   ├── mod.rs            per-event vocabulary + the non-blocking handoff
│   │   └── writer.rs         Arrow/Parquet, entirely off the request path
│   ├── summary.rs            run-level metric aggregation
│   ├── workload.rs           dry-run workload summaries
│   └── util.rs               shared timing/ratio helpers
└── Cargo.toml
```

Frontends own source semantics and produce distinct workload variants. Backends
own only endpoint, payload, and response parsing. The shared generation client
accepts normalized token-generation requests; it does not depend on either
frontend's row type.

`src/lib.rs` exposes two things. `schema/`, because what a trace file is — which
kinds exist, what columns each one obliges a file to carry — is the artifact
other programs must agree with us about; a simulator reading the same file links
the library rather than reimplementing the taxonomy, and adding a tenth kind is
then an edit in one repository. And `run_once`, because a run is not only
something a person starts from a shell: a sweep drives dozens of them, and doing
that in one process is what lets it pay for the tokenizer and the synthetic token
pool once instead of per point. `session_runner` and `tracegen` are binaries
built on top.

A consumer that only reads trace files takes the schemas without the runtime:

```toml
req-frontend = { path = "...", default-features = false }
```

That drops reqwest, tokio, tokenizers and hf-hub, leaving `anyhow`, `csv`,
`serde` and `serde_json` — because a program that never sends a request should
not compile an HTTP client to find out what a column is called.

</details>

## Current scope

The supported configuration surface is the three axes in
[Configuration axes](#configuration-axes), plus context guards, prefix-cache
accounting, and the client-observed TTFT / token-event TPOT / E2E / throughput
metrics described above.

Not currently provided:

- an OpenAI Chat Completions backend;
- raw private prompt/tool-result reconstruction;
- per-token timestamp dumps;
- TTFT/TPOT SLO pass/fail policy;
- block-level Prometheus cache telemetry.

This code is licensed under Apache 2.0; see the repository-level
[`LICENSE`](LICENSE).
