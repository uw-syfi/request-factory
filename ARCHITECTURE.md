# Request frontend, from CSV to SLO report

This document explains the system bottom-up: what exists in an input file, how
the file's declaration gives those bytes meaning, how rows become typed
requests, and how declarations are eventually compared with measurements.

## The whole path

```text
CSV cells
   ↓
InputFileFormat + TraceTag
   ↓
InputFileSchema
   ↓
validated, parsed row
   ↓
Workload: independent requests or session steps
   ↓
ScheduledRequest
   ↓
Request
   ↓
ActiveRequest
   ↓
worker execution and telemetry
   ↓
JSONL / request_slo.parquet / SloSummary
```

The first four layers live in this repository. `ScheduledRequest`, `Request`,
`ActiveRequest`, and `request_slo.parquet` are the corresponding downstream
VibeSim concepts. The important property across the boundary is that both
consumers use the same input declaration and therefore assign the same meaning
to the same CSV.

## 1. CSV cells have no meaning by themselves

Consider this row:

```csv
id,input_len,output_len,arrival_time,ttft_slo_ms,tpot_slo_ms,e2e_slo_ms,priority
0,512,64,0.0,300,25,1200,3
```

The header still does not answer all the questions a consumer must answer:

- Is the row an independent request or one round of a session?
- Is this text generation, image generation, or another request family?
- Were the SLO and priority columns intentionally declared?
- Does `input_len` mean the whole prompt or only its fresh suffix?

The consumer therefore reads the file against an explicit declaration. It does
not infer the declaration from the header.

## 2. A complete format determines its request family

| Type | Role | Question answered |
|---|---|---|
| `InputFileFormat` | complete format | Which family, columns, loader, and structural rules apply? |
| `RequestFamily` | format property | What kind of request does every row describe? |
| `TraceTag` | optional schema extension | Which orthogonal column bundles does the file carry? |

`InputFileSchema` combines the axes and resolves the exact expected header:

```text
complete family-specific format
+ declared tag columns
= exact input-file schema
```

There is no generic `Native` format that must be paired with a second family
selector. Names such as `text-generation-independent` and
`text-generation-session-execution-v2` are complete choices.

### `InputFileFormat`: family and representation together

A format selects a request family, physical row contract, and parser. The canonical
`session-execution-v2` format, for example, carries already-materialized session
execution facts:

```text
request_id
session_id
round_idx
arrival_time_ms
prefix_len
input_len
output_len
tool_wait_after_ms
```

Its family is text generation by construction. An impossible pairing such as
`SessionExecutionV2 + ImageToText` cannot be represented.

### `RequestFamily`: what every request in the file is

One format applies to one whole CSV. `RequestFamily` cannot differ by row and is
returned by `InputFileFormat::request_family()` rather than parsed separately.
For example:

```text
RequestFamily::TextGeneration
        ↓ startup selects one typed parser
TextGenerationDefinition
```

Rows may differ in lengths, arrival times, SLOs, priority, and session facts,
but they may not switch from text generation to image generation halfway
through the file. A mixed-family workload uses separate typed files and a
higher-level orchestrator.

Keeping the family at file scope lets downstream Rust code preserve it in the
type system:

```text
TraceFrontend<TextGenerationDefinition>
        ↓
RequestStore<TextGenerationDefinition>
        ↓
text-generation worker
```

The worker does not repeatedly match a per-row request-family enum.

### `TraceTag`: optional, orthogonal column bundles

Tags add columns without changing the request family or base row format:

```text
session
└── session_id, prefix_kv, tool_wait_after_ms

slo
└── ttft_slo_ms, tpot_slo_ms, e2e_slo_ms

priority
└── priority

speculative
└── accept_rate
```

`slo` and `priority` are deliberately separate. An SLO states how quickly a
request should be served; priority is a scheduling-policy hint. Neither implies
the other.

## 3. Header validation closes the declaration

Before parsing any row, `InputFileSchema` computes the exact expected columns
and compares them with the file header.

Both directions are errors:

- A missing column means the file cannot supply what it declared.
- An unexpected column means the file contains semantics the run was not told
  to consume.

For example, a `priority` column without the `priority` tag is rejected rather
than silently ignored. A declaration carrying `slo` requires all three SLO
columns, although each individual cell may be blank.

This is the boundary where the complete format and its tags become one exact
input-file schema.

## 4. Parsing turns cells into typed declarations

After header validation, each tag is parsed into its own type.

Per-request metric bounds become:

```rust
RequestSlo {
    ttft_slo_ms: Option<f64>,
    tpot_slo_ms: Option<f64>,
    e2e_slo_ms: Option<f64>,
}
```

Scheduling policy becomes:

```rust
RequestPriority {
    priority: Option<i64>,
}
```

Every SLO field is independently optional. A blank cell means that the request
declares no bound for that metric. It does not mean zero, and it is not an
automatic success. A present bound must be finite and greater than zero.

## 5. Format loaders own structure; `workload.rs` owns replay operations

The schema hierarchy separates declarations, physical formats, families, and
orthogonal tags:

```text
schema/
├── input_file_schema.rs  adds tags and verifies the exact header
├── format/
│   ├── text_generation/  independent and session-execution-v2
│   ├── image_to_text.rs, video_to_text.rs, ...
│   │                     each owns columns, typed row, validation, and load
│   └── load_utils.rs     shared CSV/tag mechanics only
├── family/               family-specific declared values
└── tag/                  SLO, priority, and speculative declarations
```

Session grouping belongs to the canonical format loader because contiguity,
round indices, predecessor order, and shared session arrival are properties of
the file's bytes. They are not optional replay policy. `SessionRound` is the
validated value produced by combining the base execution row with its declared
tags:

```text
CSV row
  ↓ session parser
ExecutionRow + RequestSlo + RequestPriority
  ↓ format validation and grouping
SessionRound
```

`workload.rs` starts after that boundary. It selects the one format loader,
then applies runtime-only operations such as `--max-items`, `--rate`, and
workload summaries. The whole input is validated before truncation, so a small
`--max-items` cannot hide malformed later rows.

## 6. A parsed row becomes a scheduled request

In VibeSim, the corresponding normalized value is:

```rust
ScheduledRequest<Definition> {
    release,
    slo,
    scheduling,
    definition,
}
```

Each field has one owner:

| Field | Meaning |
|---|---|
| `definition` | What family-specific work should be performed? |
| `release` | When may the request enter, and what predecessor/tool wait gates it? |
| `slo` | What TTFT, TPOT, and E2E duration bounds did this request declare? |
| `scheduling` | What scheduling priority did it declare? |

The distinction is important:

```text
definition  = what to do
release     = when it may enter
slo         = how quickly it should finish each measured obligation
scheduling  = how a policy may rank it in a queue
```

## 7. Release constructs the immutable request

When replay releases a scheduled row, it stamps the actual arrival time and
constructs:

```rust
Request<Definition> {
    core: RequestCore {
        id,
        arrival_time,
        slo: SloContract,
        scheduling: SchedulingContract,
    },
    definition,
}
```

`SloContract` contains three optional durations. `SchedulingContract` contains
priority only. SLOs are not converted into absolute scheduling deadlines.

Family-specific work remains in `Definition`, so a
`Request<TextGenerationDefinition>` cannot accidentally enter an image worker.

## 8. Execution adds mutable state and observations

An executing request is wrapped in:

```rust
ActiveRequest<Definition> {
    request,
    progress,
    lifecycle,
    telemetry,
}
```

The layers remain separate:

| Field | Owner |
|---|---|
| `request` | Immutable input declaration |
| `progress` | Work completed so far |
| `lifecycle` | Admission, stage, and completion state |
| `telemetry` | First/last/per-output timing observations |

For example, the declared TTFT bound lives in `request.core.slo`, while the
actual first-output time lives in telemetry. Only after execution can the two
be compared.

Current workers preserve SLO and priority, but they do not yet change execution
policy based on them. Carrying a field faithfully is not the same as having an
execution consumer for it.

## 9. Logs place declarations beside measurements

The replay client writes its public JSONL record, and VibeSim writes one
terminal or sim-end partial row per request to `request_slo.parquet`.

The downstream SLO record preserves the declarations:

```text
declared_ttft_slo_ms
declared_tpot_slo_ms
declared_e2e_slo_ms
```

beside measurements such as:

```text
ttft_ms
tpot_mean_ms
arrival_time_ms
finish_decode_time_ms
```

E2E is submission to completion and can be derived as:

```text
finish_decode_time_ms - arrival_time_ms
```

The metric relationships are exact:

```text
declared_ttft_slo_ms  ↔ measured TTFT
declared_tpot_slo_ms  ↔ measured TPOT
declared_e2e_slo_ms   ↔ measured E2E
```

One metric's bound cannot stand in for another.

## 10. Summary folds per-request verdicts

For each bound a request actually declared:

```text
attained = measurement <= bound
```

If a request declares multiple bounds, it attains its combined per-request SLO
only when every declared bound is met. For example:

```text
declared: TTFT <= 300 ms, TPOT <= 25 ms, no E2E bound
measured: TTFT = 240 ms, TPOT = 31 ms

TTFT verdict    = attained
TPOT verdict    = violated
combined verdict = violated
```

A request with no bound for a metric is excluded from that metric's
denominator. If it has no per-request bound at all and no run-level objective,
it is also excluded from overall SLO attainment; “asked for nothing” must not
be reported as “attained everything.”

## Owner map

| Concept | Question | Owner |
|---|---|---|
| `InputFileFormat` | Which family-specific format is this file? | `schema/format/` |
| `RequestFamily` | What request family does this whole file contain? | `schema/` and startup dispatch |
| `TraceTag` | Which optional column bundles are present? | `schema/` |
| `InputFileSchema` | What exact input contract was declared? | `schema/` |
| `schema/format/` | How are bytes validated and organized? | format loader |
| `workload.rs` | How is validated input replayed for this run? | runtime workload operations |
| `Definition` | What work should this family perform? | typed request |
| `ReleaseMetadata` | When may it enter? | replay scheduler |
| `SloContract` | What per-metric duration bounds apply? | request core |
| `SchedulingContract` | What queue-ranking policy hint applies? | request core/admission |
| `Progress` | How much work has completed? | active request |
| `Telemetry` | What timing was observed? | active request |
| JSONL / `request_slo.parquet` | What was declared and measured? | logging |
| `SloSummary` | What fraction met the applicable bounds? | reporting |

The shortest useful mental model is:

```text
InputFileSchema defines the file.
Format loaders prove that the bytes match it.
Workload applies this run's replay operations.
ScheduledRequest waits for release.
Request preserves the declaration.
ActiveRequest holds execution state.
Telemetry records what happened.
Summary compares measurements with SLOs.
```
