# req-frontend: from input file to replay report

This document explains `req-frontend`'s own end-to-end data flow: how one run
declares an input file, parses and validates its rows, builds a replay workload,
releases requests, talks to a serving backend, and records the result. SLO,
priority, session, and speculative metadata are local concerns along that path;
none of them is the architecture's organizing axis.

## 1. The complete path through one run

`runner::run_once_reusing` is the wiring root:

```text
YAML config
  │ launcher: validate → resolve paths → build argv
  ▼
Rust Args
  │
  ├─ parse --input-file-format + --trace-tag
  ▼
InputFileSchema
  │  declares the file-wide format, request family, and legal columns
  ▼
format::<family>::load(path, schema)
  │  header validation → CSV decoding → row validation → structural validation
  ▼
typed file contents
  ├─ Vec<IndependentRequest>
  └─ SessionPlans = Vec<(session_id, Vec<SessionRound>)>
  ▼
ReplayWorkload
  │  --max-items, --rate, arrival mode, workload summary
  ▼
executor
  │  arrival wait → admission → prompt construction → context-limit decision
  ▼
GenerationClient
  │  backend-specific JSON → HTTP stream → normalized events
  ▼
GenerationResult
  │
  ├─ StepLog channel ──────→ JSONL + replay/SLO aggregation
  └─ TimelineSink channel ─→ per-event Parquet (optional)
  ▼
RunSummary (optionally written as summary JSON and also returned to the caller)
```

Each layer answers one question:

| Layer | Question | Main output |
|---|---|---|
| schema declaration | What does the file claim to be? | `InputFileSchema` |
| format loader | Does the whole file satisfy that claim? | typed rows or grouped sessions |
| workload | How will this run use the validated contents? | `ReplayWorkload` |
| executor | When and under which dependencies does each unit run? | a concrete generation attempt |
| backend | How is a request sent and its text or media stream normalized? | `GenerationResult` |
| output | What actually happened? | `StepLog`, timeline, `RunSummary` |

## 2. The user interface is task plus structured YAML

Operators do not assemble dozens of Rust flags. The supported entry points are:

```bash
uv run python -m launcher run configs/run.yaml
uv run python -m launcher sweep configs/sweep.yaml
uv run python -m launcher tracegen configs/tracegen.yaml
uv run python -m launcher selfcheck configs/selfcheck.yaml
```

Both tasks share the same nested blocks:

```text
input       file, complete format, and tags
corpus      text corpus, tokenizer, and token-pool limit (text replay only)
server      endpoint, backend, model, and sampling
replay      arrival, capacity, context-limit, and failure policy
measurement timeline and optional run-level SLO
output      one output directory with stable artifact filenames
```

`sweep` adds a `search` block for its mode, rate range, and stopping rules. The
launcher strictly rejects unknown keys and bad value types, resolves paths,
builds the corresponding Rust binary, and lowers resolved values to internal
argv. YAML is the supported operator contract; Rust flags are the internal
launcher-to-engine interface.

`tracegen` selects `synthetic` or `coding-session` with `generator.type` and
puts generator-specific fields in that block. `selfcheck` has an independent
config for its tokenizer, output directory, pair count, and owned loopback
port. Every Rust execution mode therefore shares one launcher lifecycle.

The launcher does not read CSV, build prompts, or compute metrics. It owns only
the lifecycle around a run and terminal presentation. Complete engine output is
preserved in `terminal.log`; the default terminal view shows build, replay
progress, final workload/success/throughput/latency/cache metrics, and artifact
paths.

## 3. Startup declares the whole input file

A CSV header cannot determine its own meaning. The same `input_len` can mean an
independent request's complete prompt or the fresh suffix of a session round.
The CLI therefore selects one complete format:

```text
--input-file-format text-generation-independent
--input-file-format text-generation-session-execution-v2
--input-file-format image-to-text-independent
```

Startup combines that format with optional tags:

```rust
InputFileSchema {
    input_file_format: InputFileFormat,
    tags: Vec<TraceTag>,
}
```

The concepts relate as follows:

- `InputFileFormat` selects physical columns, a loader, structural rules, and a
  `RequestFamily`;
- `RequestFamily` is derived from the format, not selected by another CLI
  argument and not allowed to vary by row;
- `TraceTag` adds a family-orthogonal column bundle such as `slo` or `priority`;
- `InputFileSchema` is the exact header contract after combining the base
  format with its legal tags.

### The benchmark boundary is modality-compositional

`RequestFamily` and the existing CSV formats remain exact descriptions of
trace artifacts. New asset-backed benchmark adapters additionally lower their
source data into `RequestSpec`, whose inputs and outputs are independent typed
lists:

```text
RequestSpec
  inputs:  [Text, Image, Audio, Video, Tensor] (ordered, repeatable)
  outputs: [Text, Image, Audio, Video, Tensor]
```

`CapabilityProfile` validates those lists against a backend as sets plus two
composition flags. The runtime therefore grows by input encoder, output
observer, and protocol adapter—not by a matrix of modality pairs. Concrete
pair-specific validation is still allowed when a model has coupled semantics.

Asset-backed executors compose the common scheduling state with their own
client and `AssetStore`; text executors compose it with the tokenizer-backed
token pool. This prevents every modality from inheriting text-only startup,
prefix-cache preflight, or corpus requirements.

See [Adding modality-compositional benchmarks](docs/ADDING_BENCHMARKS.md) for
the stable request contract and extension checklist.

For example, `text-generation-session-execution-v2` already expresses both the
text-generation family and the session-execution layout. There is no partial
`SessionExecutionV2 + ImageToText` state and no family inference from headers.

## 4. What happens after schema parsing

`InputFileSchema` declares the contract; `workload::load_workload` opens the
file. It dispatches once on the complete format:

```text
TextGenerationIndependent
  └─ format/text_generation/independent.rs::load
       └─ Vec<IndependentRequest>

TextGenerationSessionExecutionV2
  └─ format/text_generation/session.rs::load
       └─ SessionPlans
```

Every family-format module owns:

```text
COLUMNS
typed Row / runtime-ready row type
per-row validation
load(path, InputFileSchema)
```

`format/load_utils.rs` shares only mechanical work: opening CSV, checking the
header, walking records, and parsing tag columns. It does not select a family
or produce a generic request union.

### Independent files

Each row is decoded and validated into an `IndependentRequest`:

```text
CSV record
  ↓ decode base fields + declared tag fields
IndependentRequest
  ├─ id / arrival_time
  ├─ input_len / output_len
  ├─ per-request SLO (when declared)
  └─ priority (when declared)
```

The loader returns `Vec<IndependentRequest>` in file order.

### Session files

The session loader decodes an `ExecutionRow`, combines it with declared tags,
and then validates whole-file structure:

- rows for one session are contiguous;
- `round_idx` starts at zero and is consecutive;
- round zero declares no prefix;
- every round in a session has the same arrival;
- session blocks are ordered by arrival;
- request ids are unique.

It then returns:

```rust
type SessionPlans = Vec<(String, Vec<SessionRound>)>;
```

Grouping belongs to format parsing because round order, prefix, and tool wait
are meanings encoded in the bytes. They are not optional replay policy.

### Valid schema does not imply executable by this client

The shared schema defines several request families. The runtime executes the
two text formats and asset-backed `multimodal-independent-v1`; shape-only media
CSV formats remain parseable but non-executable because dimensions and token
counts are not media content. `load_workload` rejects them at the runtime
boundary rather than inventing assets.

## 5. Why `ReplayWorkload` still exists after loading

The loaders return different Rust types:

```text
independent::load(...) → Vec<IndependentRequest>
session::load(...)     → SessionPlans
multimodal::load(...)  → Vec<RequestSpec>
```

Because `--input-file-format` is known only at runtime, `load_workload` cannot
choose one of those return types at compile time. `ReplayWorkload` is only the
sum type that carries either result:

```rust
enum ReplayWorkload {
    IndependentRequests(Vec<IndependentRequest>),
    Sessions(SessionPlans),
    MultimodalRequests(Vec<RequestSpec>),
}
```

It does not parse rows again or introduce another schema. Each variant directly
owns one loader's output. One whole file selects one variant, and the runner
enters only the corresponding executor.

The branch itself is real: independent requests release separately, multimodal
requests need asset preparation, and session rounds execute as a
predecessor-ordered closed loop. Removing the enum
would either move the same `match` into `runner.rs` and duplicate later setup,
or replace it with a more elaborate trait abstraction. Flattening sessions into
independent rows would instead lose dependency and tool-wait semantics.

This is therefore the smallest runtime branch, not a second input model. If two
formats eventually share exactly the same execution shape, their variants
should be merged or removed.

`workload.rs` then applies operations that belong to this run rather than to
the input format:

1. validate the whole file, then apply `--max-items`;
2. calculate the arrival rate of top-level workload units;
3. rescale top-level arrival offsets when `--rate` is set;
4. build `WorkloadSummary`, including unit count, step count, maximum lengths,
   and context-limit information.

Validation deliberately precedes truncation, so `--max-items 1` cannot hide a
malformed row later in the file.

A session workload counts sessions as workload units and rounds as steps. In an
independent workload both are requests. Offered unit rate must be converted by
`steps_per_workload_unit` before it is compared with delivered step throughput.

## 6. Preparing execution

Before tasks start, `runner.rs` performs one-time setup:

```text
WorkloadSummary / dry-run early return
  ├─ text → tokenizer + synthetic token pool → prefix-cache preflight
  └─ multimodal → verify/read/hash/base64 assets before run_start
  ↓
construct GenerationClient and its protocol adapter
  ↓
create AppState, bounded dispatcher, log channel, optional timeline channel
```

`--dry-run` returns before corpus construction and network access. It validates
the input and workload shaping, not the serving endpoint.

`CorpusCache` can reuse the tokenizer and synthetic corpus across text sweep
points. The text backend preflight confirms that the server exposes required
prefix-cache usage before replay. Multimodal replay has no corpus or cache
preflight; it validates backend capabilities and prepares immutable assets
before starting the arrival clock.

## 7. `executor/` owns release, dependencies, and admission

The runner's central `JoinSet` dispatcher starts at most the configured number
of top-level workload tasks and follows one of three paths selected by
`ReplayWorkload`. It does not park one task per trace row.

### Multimodal independent request

```text
wait for request arrival
  → send prevalidated ordered text/media parts
  → fold the streamed text response through the shared client
  → StepLog with modalities and asset byte counts
```

### Independent request

```text
wait for request arrival
  → draw input_len synthetic tokens
  → context-limit check
  → GenerationClient::run_step
  → StepLog
```

Each independent request releases and holds capacity independently.

### Session

```text
wait for session arrival
  → hold one dispatcher slot for the entire session
  → for each round in order:
       build prompt from carried context
       context-limit check
       GenerationClient::run_step
       carry real output token ids into the next round
       wait tool_wait_after_ms
  → release the slot when the session ends
```

Later rounds are not independently released by a recorded wall-clock arrival.
They form a closed-loop chain and wait for their predecessor and tool delay.
The session holds its capacity slot across every round and tool wait; that is
the current concurrency contract.

`arrival_mode=trace-timed` honors recorded arrival offsets. `arrival_mode=saturated`
ignores that timeline and moves units into admission as soon as possible.
`--max-concurrency` sizes the dispatch window. Completion admits exactly the
next canonical unit, so order is deterministic and scheduler memory is
proportional to active concurrency rather than trace length. For saturated
scale-out, launcher processes partition canonical top-level ordinals; sessions
remain indivisible.

## 8. `tokens.rs` materializes actual prompts

Input rows carry lengths and prefix relationships, not concrete token ids for
this replay.

An independent request draws `input_len` tokens from the shared synthetic
pool. A session round uses `PromptBuilder`:

```text
previous realized context[..prefix_len]
+ fresh synthetic tokens[input_len]
= prompt ids sent to the server
```

After a round, the builder prefers the server's real output token ids for the
next context instead of inventing output. `prefix_len` is the planned reusable
prefix; it does not assert that the server hit its cache. Server-reported cached
prompt tokens are recorded separately.

## 9. `backend/` normalizes serving protocols

The executor submits a backend-neutral request:

```rust
GenRequest {
    request_id,
    prompt: Prompt::Tokens(...),
    max_tokens,
    ...
}
```

`backend/wire/` owns typed token/text wire differences among OpenAI, vLLM
native-token, and SGLang native-token endpoints. It serializes borrowed request
structs and parses SSE bytes without an intermediate JSON DOM.
`GenerationClient` owns the shared async
lifecycle:

1. build and send a payload;
2. consume the response stream;
3. normalize wire objects into `StreamEvent`;
4. fold text, token ids, usage, finish reason, and failures;
5. check prompt echo and token accounting;
6. return a backend-neutral result. Generated media uses `media_client.rs`,
   which normalizes JSON-carried images, base64 chat-audio deltas, and raw PCM
   streams into first-output, byte, duration, RTF, artifact, and timeline fields.

```rust
GenerationResult {
    outcome: GenerationOutcome,
    output_ids: Vec<u32>,
    timeline: Vec<TimelineEvent>,
}
```

This layer observes submission, send, first text/token id, last token id, and
response-completion clocks. TTFT, TPOT, E2E, and arrival-release lag are derived
from explicit clocks in the outcome; they do not organize the whole pipeline.

## 10. Results flow into logs and summary

The executor combines the source declaration with the result:

```text
IndependentRequest / SessionRound
            +
GenerationOutcome
            ↓
         StepLog
```

Two output paths avoid blocking request execution:

- the log channel lets `summary::write_logs` fold replay metrics, prefix-cache
  metrics, and optional SLO attainment, and optionally persist per-step JSONL;
- the optional timeline channel encodes per-event Parquet in a separate
  blocking writer. A full channel drops timeline samples instead of applying
  disk or Arrow backpressure to the measured submission path.

After every workload task and writer finishes, the runner constructs:

```rust
RunSummary {
    workload,
    replay,
    client_runtime,
    timeline,
    slo,
}
```

It is returned as a library value and may also be written to `--summary-path`.
The final product is therefore broader than an SLO report: it covers input
shape, replay outcome, client runtime, timeline completeness, throughput,
latency, prefix-cache fidelity, and SLO attainment when one was declared.

## 11. Tags are auxiliary declarations carried through the path

Tags enter typed rows during schema parsing and remain attached to the source
record through output:

```text
slo         → ttft_slo_ms, tpot_slo_ms, e2e_slo_ms
priority    → priority
session     → session-related columns for the native independent layout
speculative → accept_rate
```

They do not imply one another. Priority is a scheduling hint, not an SLO, and
the three SLO metrics are declared independently per request. Root-level
`slo.rs` compares declared bounds with measured timings after execution; it
does not own schema loading or execution.

## 12. Binary and library boundaries

| Entry point | Role |
|---|---|
| `run` / `session_runner` | executes one live replay |
| `sweep` | calls the same `run_once_reusing` repeatedly, searches a rate/SLO boundary, and reuses the corpus |
| `tracegen` | materializes canonical session input files from generator sources |
| `selfcheck` | validates release, stream measurement, and cache accounting against a controlled stub |

The main binary and sweep share one runner instead of maintaining two replay
paths. Another consumer may share the schema's format contracts and loaders
without inheriting this HTTP client's `ReplayWorkload`, token construction, or
execution policy.

## Owner map

| Path | Sole responsibility |
|---|---|
| `launcher/` | YAML validation, argv/build/run lifecycle, and terminal UI |
| `schema/input_file_schema.rs` | combine a complete format and tags into an exact file contract |
| `schema/format/` | decode, validate, and organize family-specific typed contents |
| `schema/family/` | family-specific declared value types |
| `schema/tag/` | orthogonal per-row declaration types |
| `workload.rs` | runtime dispatch, truncation, rate scaling, and workload summary |
| `runner.rs` | wiring and lifecycle for one run |
| `executor/` | arrival release, session dependency, admission, request lifecycle |
| `tokens.rs` | concrete token ids and session-context carry-forward |
| `backend/wire/` | protocol-specific JSON shaping and parsing |
| `backend/client.rs` | shared token/text HTTP streaming engine and integrity measurements |
| `backend/media_client.rs` | generated image/audio transport and modality-neutral measurements |
| `record.rs` | per-step source-plus-outcome JSONL contract |
| `timeline.rs` | optional per-event recording |
| `summary.rs` | replay, runtime, timeline, and run-level aggregation |
| `slo.rs` | optional declared-versus-measured SLO evaluation |

The shortest useful mental model is:

```text
Schema says what the file is.
The format loader turns bytes into validated typed contents.
Workload decides how this run replays them.
Executor turns the plan into actual requests under time and dependency rules.
Backend turns requests into stream outcomes.
Record and Summary preserve what actually happened.
```
