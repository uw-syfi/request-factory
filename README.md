<div align="center">

<h1>TraceLab Replay</h1>

**Replay real coding-agent workload shapes against an inference server.**

Session chains · Independent requests · Exact token-ID prompts · Prefix-cache auditing · TTFT/TPOT

[Quickstart](#quickstart) ·
[Engine setup](#engine-side-setup-guide) ·
[Configuration axes](#configuration-axes) ·
[CSV formats](#input-csv-formats) ·
[Session policies](#session-context-policies) ·
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

## Configuration axes

A run is described by four independent choices. Each axis answers a different
question, and nothing in one axis implies a value in another — pick one value
per axis.

| # | Axis | Selected by | Supported values |
|---|---|---|---|
| 1 | **Trace format** — what a CSV row is, and whether requests chain | `--trace-format` | `session`, `independent` |
| 2 | **Session context policy** — how a round's prompt reuses the previous one, i.e. the prefix-cache assumption | `--session-context-policy` | `trace-reported` (default), `monotonic` |
| 3 | **Arrival and load control** — when top-level units are released, how many run at once | `--arrival-mode`, CSV `arrival_time`, `--rate`, `--max-concurrency`, `--max-items` | `trace-timed` (default) or `saturated`, each with an optional cap |
| 4 | **Wire backend** — endpoint and output representation | `--backend` | `openai` (default), `vllm-tokens`, `sglang-tokens` |

### Axis 1 — trace format

Two typed frontends with separate schemas, and two accepted session schemas. An
independent request is never rewritten into a session row with placeholder
fields.

| Value | Row means | Execution |
|---|---|---|
| `session-execution-v2` | One **already-materialized** round of a session | Same execution as `session`, but the prompt split is fixed in the file rather than derived at run time |
| `session` | One round of a multi-round session, as the source reported it | Rounds of one session are closed-loop: submit round `i`, await its full response, wait `tool_wait_after_ms`, then submit round `i + 1` |
| `independent` | One standalone request | One-shot; rows never share context |

Prefer `session-execution-v2` for anything compared against a simulated run:
it is the only format whose `prefix_len` is guaranteed to exist when the round
runs, so both runtimes replay identical work without agreeing on a policy. Use
raw `session` when you are exploring how a policy changes the workload — that
is the choice the canonical format has already made for you.

### Axis 2 — session context policy (prefix-cache assumption)

This axis, and only this axis, decides how much of the previous prompt is
carried into the next one — that is, the prefix-cache hit shape the workload
plans for. **Only the `session` frontend reads it.** Independent requests share
no context by construction, so the default `trace-reported` is accepted and
simply unused there, while `monotonic` with `--trace-format independent` is
rejected at startup rather than silently ignored.

| Value | Next prompt is | Planned reuse |
|---|---|---|
| `trace-reported` | `previous_context[0:prefix_len] + fresh_ids(input_len)` | Exactly the split the trace reported |
| `monotonic` | Prior prompt + exact model output IDs, truncated or grown to `prefix_len + input_len`, reset only on major compaction | The longest available prefix up to the trace target |

### Axis 3 — arrival and load control

Applies uniformly to whichever top-level unit axis 1 selected — a *session* or
an *independent request* — and never changes prompt content or context reuse.

This axis has two sub-axes that compose freely — *when* a unit may start, and
*how many* may run — plus a selection control.

| Control | Effect |
|---|---|
| `--arrival-mode trace-timed` | Default. Release offsets are replayed from CSV `arrival_time`. For `session`, only the first sorted round's value releases the session |
| `--arrival-mode saturated` | Recorded arrivals are ignored: every unit is eligible from the start. Without a cap this submits the whole workload at once; with one it is a closed-loop generator. Rejected together with `--rate`, which rescales a timeline this mode discards |
| `--rate N` | Rescale all arrivals to `N` units/s, preserving relative gaps and simultaneous-arrival bursts. Needs at least two distinct arrival times |
| `--max-concurrency N` | Bound concurrently active units, under either arrival mode. One session holds its slot across all of its rounds **and its tool waits** — while waiting on a tool it has no request in flight but is still occupying a slot |
| `--max-items N` | Keep `N` units, applied before `--rate` is measured. **Not the earliest `N` arrivals**: `session` keeps the lexicographically smallest `N` session IDs (`session-10` sorts before `session-2`), `independent` keeps the first `N` CSV rows |

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

### Axis 4 — wire backend

Transport only: endpoint, payload, and response parsing. It does not change
workload shape.

| Value | Endpoint | Output representation |
|---|---|---|
| `openai` | `POST {base_url}/completions` | Text plus vLLM's optional `return_token_ids` extension |
| `vllm-tokens` | `POST {base_url}/inference/v1/generate` | Native token-ID deltas; server must run with `--tokens-only` |
| `sglang-tokens` | `POST {base_url}/generate` | Native `output_ids` deltas; server must run with `--skip-tokenizer-init` and `--stream-output` |

All three send the prompt as token IDs; they differ only in whether the server
detokenizes. See [Request backends](#request-backends) for the comparison.

### Cross-axis constraints

The axes are independent with exactly five exceptions, all enforced by the
runner:

- `monotonic` (axis 2) requires `--trace-format session` (axis 1). The
  combination is rejected before the trace is even loaded.
- Any explicit `--session-context-policy` (axis 2) is rejected with
  `--trace-format session-execution-v2` (axis 1): that trace carries no policy
  switch, because the policy was resolved when the file was generated and is
  recorded in its manifest.
- `monotonic` (axis 2) requires exact generated token IDs, so axis 4 must be a
  native token endpoint (`vllm-tokens`, `sglang-tokens`) or an `openai`
  endpoint that honours `return_token_ids`. A server that ignores the extension
  can still serve `trace-reported`. This one fails per round at replay time,
  not at startup.
- `--rate` (axis 3) needs a trace with at least two distinct arrival times,
  after `--max-items` has been applied.
- `--rate` (axis 3) is rejected with `--arrival-mode saturated` (axis 3): one
  rescales the recorded timeline, the other throws it away.

Each native token endpoint additionally requires its own server launch flags,
listed with the backend in [Request backends](#request-backends). Those are
server-side prerequisites rather than constraints between axes.

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

Run these commands from the TraceLab repository root.

### 1. Build

```bash
cargo build --release --manifest-path replay/Cargo.toml --bin session_runner
```

### 2. Parse and inspect a trace without a server

```bash
mkdir -p "$TMPDIR/tracelab-replay"

./replay/target/release/session_runner \
  --trace replay/examples/multi_session_example.csv \
  --trace-format session \
  --text-file unused-in-dry-run \
  --tokenizer unused-in-dry-run \
  --model dry-run \
  --dry-run \
  --max-model-len 131072 \
  --summary-path "$TMPDIR/tracelab-replay/dry-run-summary.json"
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
./replay/target/release/session_runner \
  --trace replay/examples/multi_session_example.csv \
  --trace-format session \
  --session-context-policy trace-reported \
  --text-file /path/to/large-text-corpus \
  --tokenizer /path/to/model-or-tokenizer.json \
  --model meta-llama/Meta-Llama-3-8B \
  --backend openai \
  --base-url http://127.0.0.1:8000/v1 \
  --max-model-len 131072 \
  --skip-when-reaching-limit \
  --log-path "$TMPDIR/tracelab-replay/requests.jsonl" \
  --summary-path "$TMPDIR/tracelab-replay/summary.json"
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

TraceLab measures the full client-visible streaming path, so engine setup is
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

Use `--base-url http://127.0.0.1:8000/v1 --backend openai` on the TraceLab
side. For native token-in/token-out, add `--tokens-only` to the server command
and use `--base-url http://127.0.0.1:8000 --backend vllm-tokens`.

| Server setting | Why TraceLab needs it |
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
TraceLab side — no `/v1` suffix, because `/generate` is a native route.

| Server setting | Why TraceLab needs it |
|---|---|
| `--skip-tokenizer-init` | Accepts `input_ids` and returns `output_ids` with no detokenization. The counterpart of vLLM's `--tokens-only`. OpenAI-compatible routes stop working on this server. |
| `--stream-output` | Streams disjoint deltas. Without it SGLang resends the full output every chunk, which distorts late-token latency; TraceLab detects this and fails rather than reporting it. Newer SGLang renames it `--incremental-streaming-output`. |

SGLang's radix prefix cache is on by default and reports `cached_tokens` in
`meta_info`, so the preflight needs no extra flag — unlike vLLM, which needs
`--enable-prompt-tokens-details`.

### API process sizing

Do not confuse vLLM API processes with TraceLab concurrency or TraceLab's
Tokio worker threads:

| Boundary | Control | Meaning |
|---|---|---|
| TraceLab arrival scheduler | `--arrival-mode`, CSV arrivals and `--rate` | When top-level workload units become runnable. |
| TraceLab active work | `--max-concurrency` | Maximum active sessions or independent requests, counted across tool waits. |
| TraceLab runtime | Tokio workers, reported under `client_runtime` | Polling release and HTTP client tasks. |
| vLLM HTTP frontend | `--api-server-count N` | Number of API processes draining EngineCore outputs and emitting streams. |
| vLLM EngineCore | TP/DP, batching, and token-budget flags | Model execution and engine-side queueing. |

One API process can become the bottleneck at high request rates even while the
GPU engine has capacity. Increasing TraceLab `--max-concurrency` does not fix
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

In the vLLM fork used by TraceLab alignment,
`python -m vllm.entrypoints.openai.api_server` accepts
`--api-server-count` but still enters the single-server path. Launch through
`vllm serve` or `python -m vllm.entrypoints.cli.main serve` to select the real
multi-API path. Before measuring, also confirm that the TraceLab prefix-cache
preflight passes and that the server reports the intended model, TP/DP layout,
prefix caching, token mode, and `stream_interval`.

## Input CSV formats

Select the parser explicitly with `--trace-format session-execution-v2`,
`--trace-format session`, or `--trace-format independent`. The schemas are
intentionally separate: independent requests are not converted into fake session
rows with placeholder fields, and a header is never used to guess which schema a
file is.

### Canonical execution CSV — `session-execution-v2`

The recommended input. Every column is already materialized, so the file is the
whole contract and the runner has nothing left to decide:

```csv
request_id,session_id,round_idx,arrival_time_ms,prefix_len,input_len,output_len,tool_wait_after_ms
1150_round_000000,1150,0,0.000000,0,4211,151,54.000000
1150_round_000001,1150,1,0.000000,4362,238,147,61.000000
```

| Column | Unit/type | Contract |
|---|---|---|
| `request_id` | String | Exactly `{session_id}_round_{round_idx:06}`. Validated, not trusted. |
| `session_id` | String | Opaque session identity. |
| `round_idx` | Non-negative integer | Contiguous from `0` within each session. |
| `arrival_time_ms` | Milliseconds, 6 decimals | Identical on every row of a session. The earliest arrival in the file is `0`. |
| `prefix_len` | Tokens | Reusable prefix that **will exist** when the round runs. `0` on round `0`. |
| `input_len` | Tokens | Fresh tokens appended this round. May be `0` only when a prefix exists. |
| `output_len` | Tokens | Sent as `max_tokens` with EOS ignored. |
| `tool_wait_after_ms` | Milliseconds, 6 decimals | Delay before the next round of the same session. |

What the format buys, and what it costs:

- **No runtime policy.** `--session-context-policy` is rejected outright. Two
  runtimes replaying the file do identical work without having to agree on
  anything beyond the bytes.
- **Rebased arrivals.** A canonical trace starts at its own origin, so a
  three-session subset of a month-long dataset does not open with weeks of dead
  time.
- **Row order is the contract.** Rows of a session are contiguous, sessions are
  nondecreasing by arrival, and no consumer may re-sort by identifier — dense
  internal IDs are assigned in file order on both sides.
- The cost is that the fold decision is frozen. To ask "what if the prefix
  assumption were different", regenerate the file, do not switch a flag.

Generate one from a raw session trace with `tracegen`:

```bash
cargo run --release --manifest-path replay/Cargo.toml --bin tracegen -- \
  --source raw_sessions.csv \
  --policy trace-reported \
  --max-sessions 200 \
  --out trace/execution.csv
```

It writes `execution.csv` plus `execution.manifest.json` and
`execution.plan.json` beside it. The manifest records the source hash, the
policy and its thresholds, the selection rule, how many tokens were folded from
prefix into fresh input, and the planned prefix hit rate — read it before
quoting any cache number, because a fold that is large is not a bug but does
mean the source attributed to cache what the replay must actually prefill. The
plan is the normalized per-round expansion, which is what a simulator compares
against to prove both sides scheduled the same work.

`--policy` is the one real choice:

| Policy | Keeps | Use when |
|---|---|---|
| `trace-reported` | The reported prefix/input split; any prefix the replayed conversation cannot supply is folded into fresh input | You want the trace's own prompt shape, honestly cold where the source was warm |
| `prefix-preserving` | The longest prefix the conversation can actually supply, truncated to the trace's total, reset only on major compaction | You want maximum realistic cache reuse |

### Session CSV

The raw, pre-materialization schema. Its `prefix_len` is whatever the source
reported, which is why this format still needs a runtime policy:

```csv
id,input_len,output_len,arrival_time,round_idx,tool_wait_after_ms,prefix_len
session-0,512,64,0,0,0,0
session-0,128,64,0,1,100,576
session-1,400,48,250,0,0,0
```

`session_id` is accepted as an alias for `id`. A compact header without
`arrival_time` is also valid because its default is `0`:

```csv
session_id,round_idx,prefix_len,input_len,output_len,tool_wait_after_ms
```

| Column | Required | Unit/type | Contract |
|---|---:|---|---|
| `id` or `session_id` | Yes | String | Session identity. Supply one name, not both. |
| `arrival_time` | No | Milliseconds, default `0` | Release offset for the session. The first round after sorting by `round_idx` controls the session's release. |
| `round_idx` | Yes | Non-negative integer | Ordering key within the session. Use a unique value for every round; duplicates are not currently rejected. |
| `prefix_len` | Yes | Tokens | Trace-reported retained-prefix length. Its operational meaning depends on the selected session context policy. |
| `input_len` | Yes | Tokens | Trace-reported newly appended input length. In `monotonic`, it contributes to the target total rather than forcing the actual append. |
| `output_len` | Yes | Tokens | Requested maximum output length. TraceLab sends it as `max_tokens` with EOS ignored. |
| `tool_wait_after_ms` | Yes | Milliseconds | Delay after this round completes and before the next round in the same session starts. |

Parsing and execution rules:

1. Rows are grouped by session ID.
2. Each session is sorted by `round_idx`; with unique indices, input row order is irrelevant.
3. The session waits for the complete response to round `i`, then waits
   `tool_wait_after_ms`, before submitting round `i + 1`.
4. Only the first sorted round's `arrival_time` releases the session. For a
   canonical trace, repeat the same session arrival on every row.
5. Use finite, non-negative arrival and wait values. Negative values currently
   behave as zero at the corresponding wait boundary and should not be relied on
   as part of the public format.

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

Select this frontend with `--trace-format independent`. Its schema and runtime
semantics are generic and do not depend on any simulator or trace producer.
There is no canonical variant of it: with no context to carry forward, there is
nothing for a policy to materialize.

## Session context policies

Select a policy with `--session-context-policy`. It is used only by the
`session` frontend.

| Policy | CLI value | Trace fields control | Small target reduction | Output-ID requirement |
|---|---|---|---|---|
| Trace-reported, default | `trace-reported` | Exact prefix/append split | Truncate to `prefix_len` | Exact IDs preferred; text fallback remains for backward compatibility |
| Monotonic | `monotonic` | Target total `prefix_len + input_len` | Truncate to the target while retaining its exact prefix | Exact server IDs required |

### `trace-reported` — reproduce the trace split

For each row, TraceLab constructs:

```text
prompt = previous_context[0:prefix_len] + fresh_ids(input_len)
```

After the response, `previous_context` becomes the submitted prompt plus the
actual output. If the carried context is shorter than `prefix_len`, TraceLab
fills the missing region with fresh synthetic IDs so the requested length can
still be sent.

That fill preserves shape but cannot represent a real cache hit. A
cache-alignment experiment should therefore use a cumulative-consistent trace
where every reported prefix is available from the preceding prompt and output.
Dry-run parses the row types and summarizes their lengths but does not currently
reject this semantic inconsistency.

### `monotonic` — preserve the longest reusable prefix at the trace target

Let:

```text
C = previous submitted prompt + exact actual output IDs
T = prefix_len + input_len                  # trace target total
drop = max(0, C.len - T)
```

The next prompt is constructed as follows:

```text
major_compaction = drop >= 64,000 and drop / C.len >= 50%

if major_compaction:
    prompt = fresh_ids(T)
    derived_prefix_len = 0
    derived_append_len = T
else:
    derived_prefix_len = min(C.len, T)
    derived_append_len = T - derived_prefix_len
    prompt = C[0:derived_prefix_len] + fresh_ids(derived_append_len)
```

Consequences:

- Normal growth reuses the entire preceding prompt and model output.
- A micro-compaction, or any reduction that misses either major threshold,
  truncates `C` to `T` while preserving the longest exact reusable prefix.
- A major compaction starts an unrelated target-length context.
- The actual prompt length is always exactly `T`.
- The raw trace fields remain in the log, alongside the actual derived
  prefix/append decision.

Examples:

| Current `C` | Trace target `T` | Decision | Actual prompt | Derived prefix | Fresh append |
|---:|---:|---|---:|---:|---:|
| 576 | 704 | Grow | 704 | 576 | 128 |
| 100,000 | 90,000 | Truncate small reduction | 90,000 | 90,000 | 0 |
| 140,000 | 70,000 | Major compaction | 70,000 | 0 | 70,000 |
| 200,000 | 130,000 | Truncate: drop is under 50% | 130,000 | 130,000 | 0 |

### Token-ID fidelity

Monotonic replay never reconstructs continued context from output text:

```text
prior prompt IDs + server output IDs + synthetic append IDs -> next prompt IDs
```

The `openai` backend requests vLLM's non-standard `return_token_ids` extension.
The `vllm-tokens` backend returns token IDs by protocol. With either backend, a
successful response is allowed to continue only when token IDs are present and
their total agrees with `completion_tokens` when usage reports that count.
Otherwise the round is marked `FAILED` and the session stops before the next
prompt.

The backend may still re-tokenize output text for diagnostics and the legacy
`trace-reported` fallback. Those reconstructed IDs are never committed by
`monotonic`.

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
ids TraceLab constructed. They differ only in what comes back, and the
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
standard OpenAI completions contract. A server that ignores it can serve
`trace-reported` replay, but cannot successfully continue a `monotonic` session.

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

| Server flag | Why TraceLab requires it |
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

### Alignment profile configuration

The alignment launcher exposes the same choices:

```yaml
workload:
  frontend:
    type: session
    path: ../../trace/sessions.csv
    context_policy: monotonic
  backend:
    type: vllm_tokens
  text_file: ../../trace/prompts.txt
  tokenizer: meta-llama/Meta-Llama-3-8B
```

Selecting `vllm_tokens` makes the launcher add `--tokens-only` to the paired
vLLM server. Omitting `backend` keeps the backward-compatible `openai` default.

## Synthetic token corpus

`--text-file` supplies content, while the CSV supplies shape. TraceLab tokenizes
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
reduce it. If the corpus produces fewer IDs than the longest prompt, TraceLab
warns that content will repeat within a request and may distort prefix-cache
measurements. Monotonic construction also keeps every actual prompt at the
trace-reported target `prefix_len + input_len`.

The corpus is tokenized line by line, so concatenated pool IDs need not equal a
single tokenizer call over the original whole file. This creates synthetic
boundary transitions but no client/server mismatch: the exact resulting IDs are
sent directly to the server.

For large-context tests, a large public corpus such as `enwik9` is suitable.
TraceLab does not bundle it; obtain it from its original distributor and follow
the applicable license.

## Context limits and prefix-cache preflight

### Context limits

`--max-model-len N` adds context-limit information to dry-run output.
`--skip-when-reaching-limit` requires that flag and reserves at least one token
of headroom. A live request is skipped when:

```text
actual_prompt_len + output_len_target >= max_model_len
```

Equality deliberately skips. TraceLab does not silently shorten the requested
output to fit. An independent request is logged as skipped and replay continues.
A skipped session round is logged and that session terminates, because the
missing model output makes subsequent context continuation untrustworthy.

`--skip-on-context-limit` and the older `--fail-on-context-overflow` are
compatibility aliases for the same behavior.

For both policies, the requested context length is
`prefix_len + input_len + output_len`. The policies differ in token identity
and the derived prefix/append split, not total prompt length. The live guard
still uses the actually constructed prompt plus target output.

### Prefix-cache preflight

Before every non-dry run, TraceLab sends the same 512-token-or-smaller probe
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

### Per-request JSONL — schema v8

`--log-path` receives one typed record per attempted request. Session and
independent-request source data are tagged variants rather than one sparse
object.

v8 adds two prefix-accounting fields to session rounds, which together separate
a *planned* departure from the source trace from an *unplanned* one at run time:

- `folded_prefix_tokens` — prefix the source reported that the replayed
  conversation never produced, folded into fresh input. Large and expected on a
  session's first round, since a real coding agent resumes from context the
  published trace does not contain. Under `session-execution-v2` this is already
  baked into the file and the manifest reports the total.
- `prefix_shortfall_tokens` — planned prefix the *live* conversation could not
  supply, filled with fresh ids instead. This is the one place a live run
  departs from its materialized plan, so a nonzero value means a short or failed
  round upstream, not a trace property. Neither field is ever counted as a cache
  hit.

v7 added `outcome.echoed_prompt_tokens`: leading generated IDs that repeated the
prompt tail and were dropped before carry-forward. It is `0` on every server
that does not echo, which is all of them today apart from the SGLang case
described under [Request backends](#request-backends). Both additions are purely
additive; consumers reading `outcome.status` or `outcome.request_id` are
unaffected.

Abbreviated session example:

<details>
<summary><b>View a schema-v8 session record</b></summary>

```json
{
  "schema_version": 8,
  "source": {
    "type": "session_round",
    "data": {
      "session_id": "session-0",
      "round_idx": 1,
      "prefix_len": 576,
      "input_len": 128,
      "target_prompt_len": 704,
      "prompt_len": 704,
      "session_context_policy": "monotonic",
      "derived_prefix_len": 576,
      "derived_append_len": 128,
      "folded_prefix_tokens": 0,
      "prefix_shortfall_tokens": 0,
      "major_compaction": false,
      "planned_prefix_hit_rate": 0.8181818182,
      "output_len_target": 64,
      "tool_wait_after_ms": 100.0,
      "arrival_time_ms": 0.0
    }
  },
  "outcome": {
    "request_id": "session-0_round_000001",
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
| `session`, `session-execution-v2` | `{session_id}_round_{round_idx:06}` | `session-0_round_000001` |
| `independent` | `independent_{id}` | `independent_request-0` |

Under `session-execution-v2` the file carries `request_id` as a column, and the
runner validates that it matches this form rather than trusting it — so the join
key against server logs is the same string in the trace, the client log, and the
server log.

### Run summary

`--summary-path` writes one JSON document containing:

- `workload`: parsed shape and optional trace-target overflow information;
- `replay.common`: success/failure counts, throughput, TTFT, TPOT, and E2E;
- `replay.prefix_cache`: session-only planned-versus-server cache accounting;
- `client_runtime`: Tokio worker count and sampled global injection-queue peak.

The queue-depth metric does not include every worker-local runnable queue. Use
it together with `arrival_release_lag_ms` and OS CPU/thread evidence.

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
| `monotonic session context requires exact generated token IDs` | The endpoint omitted token IDs or their count disagreed with `completion_tokens`. Use a vLLM endpoint supporting `return_token_ids`, or use `vllm-tokens`. Do not work around this with text re-tokenization. |
| `server streamed cumulative output` | The SGLang server is in its default cumulative streaming mode. Relaunch it with `--stream-output` (named `--incremental-streaming-output` in newer builds). |
| `the extra leading ids do not match the prompt tail` | The server streamed more generated IDs than its own `completion_tokens` count, and the excess is not an echo of the prompt. TraceLab refuses to guess what those IDs are; inspect the raw response before trusting the run. |
| `--trace-format session-execution-v2 carries no context policy` | The canonical trace resolved its policy at generation time. Drop `--session-context-policy`, or regenerate with `tracegen --policy`. |
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
| `--trace` | Path | Source CSV, interpreted by `--trace-format` |
| `--text-file` | Path | Synthetic token corpus. Required even for `--dry-run`, which never opens it |
| `--tokenizer` | Path or HF repo id | `tokenizer.json`, a directory containing one, or a repo id to download. Must match the served model |
| `--model` | String | Model name placed in the request payload. Accepted but unused by `sglang-tokens`, whose server hosts one model and takes no model field |

### Axis selection

| Flag | Default | Values |
|---|---|---|
| `--trace-format` | `session` | `session-execution-v2`, `session`, `independent` — axis 1 |
| `--session-context-policy` | `trace-reported` | `trace-reported`, `monotonic` (alias of `prefix-preserving`) — axis 2. `monotonic` requires `--trace-format session`; any explicit value is rejected with `session-execution-v2` |
| `--backend` | `openai` | `openai`, `vllm-tokens`, `sglang-tokens` — axis 4 |
| `--base-url` | `http://127.0.0.1:8000/v1` | Include `/v1` for `openai`, omit it for the native token endpoints |

### Load control (axis 3)

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
| `--stop-session-on-error` | `true` | A session stops after its first failed round. This is a set-true flag with a `true` default, so it cannot currently be switched off from the command line |

A `monotonic` round that loses exact output IDs always ends its session,
independently of this flag, because the next prompt would otherwise be built on
re-tokenized text. A context-limit skip likewise always ends its session.

### Output

| Flag | Default | Notes |
|---|---|---|
| `--log-path` | `session_runner_output.jsonl` | Per-request JSONL, flushed per record |
| `--summary-path` | unset | No JSON summary is written unless given. Also written by `--dry-run` |
| `--dry-run` | off | Static inspection only; returns before the tokenizer, corpus, and preflight |
| `--token-pool-limit N` | see [Synthetic token corpus](#synthetic-token-corpus) | Cap on synthetic pool size |

## Repository structure

<details>
<summary><b>Contributor-facing module map</b></summary>

```text
replay/
├── examples/                 small session CSVs
├── src/
│   ├── main.rs               argument validation, preflight, task fan-out
│   ├── cli.rs                public CLI contract
│   ├── backend.rs            backend adapters + shared streaming engine
│   ├── executor/
│   │   ├── mod.rs            shared run state, progress counters, status task
│   │   ├── session.rs        ordered closed-loop session executor
│   │   └── independent.rs    one-shot independent-request executor
│   ├── trace/
│   │   ├── mod.rs            frontend dispatch + arrival-rate scaling
│   │   ├── session.rs        session CSV parser
│   │   └── independent.rs    independent-request CSV parser
│   ├── tokens.rs             synthetic ID pool + session prompt policies
│   ├── record.rs             versioned typed JSONL records
│   ├── summary.rs            run-level metric aggregation
│   ├── workload.rs           dry-run workload summaries
│   └── util.rs               shared timing/ratio helpers
└── Cargo.toml
```

Frontends own source semantics and produce distinct workload variants. Backends
own only endpoint, payload, and response parsing. The shared generation client
accepts normalized token-generation requests; it does not depend on either
frontend's row type.

</details>

## Current scope

The supported configuration surface is the four axes in
[Configuration axes](#configuration-axes), plus context guards, prefix-cache
accounting, and the client-observed TTFT / token-event TPOT / E2E / throughput
metrics described above.

Not currently provided:

- an OpenAI Chat Completions backend;
- raw private prompt/tool-result reconstruction;
- per-token timestamp dumps;
- TTFT/TPOT SLO pass/fail policy;
- block-level Prometheus cache telemetry.

TraceLab code is licensed under Apache 2.0; see the repository-level
[`LICENSE`](../LICENSE).
