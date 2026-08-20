---
name: coding-trace-replay
description: Configure, generate, run, sweep, validate, and interpret req-frontend workloads through its structured YAML launcher. Use for `python -m launcher` tasks (`run`, `sweep`, `tracegen`, `selfcheck`), choosing complete input-file formats, canonicalizing recorded or synthetic session traces, selecting arrival/concurrency and OpenAI/vLLM/SGLang backends, launching a compatible real serving endpoint, diagnosing prefix-cache or streaming failures, rendering sweep figures, and reading requests.jsonl, summary.json, timeline.parquet, sweep.json, trace manifests, or selfcheck reports.
---

# Coding Trace Replay

## Use the supported interface

Work from the `req-frontend` repository root. Represent every operator action as
one task plus one YAML file:

```bash
uv run python -m launcher run CONFIG.yaml
uv run python -m launcher sweep CONFIG.yaml
uv run python -m launcher tracegen CONFIG.yaml
uv run python -m launcher selfcheck CONFIG.yaml
```

Start from the matching file under `configs/`. Use launcher `--dry-run` to
validate and print the resolved internal command without executing it. Do not
teach users to assemble the underlying Rust flags; those flags are the internal
launcher-to-engine interface.

The launcher owns YAML validation, path resolution, build/run lifecycle,
`terminal.log`, and the result panel. Rust owns file schemas, workload shaping,
release, prompt construction, HTTP streaming, integrity checks, and metrics.

## Select the task

| Task | Use it for | Primary output |
|---|---|---|
| `tracegen` | Materialize one canonical session trace | CSV + manifest + plan |
| `run` | Replay one workload against a serving endpoint | requests JSONL + summary + timeline |
| `sweep` | Search or grid over offered arrival rate | `sweep.json` + point directories + optional figures |
| `selfcheck` | Verify client measurement fidelity against the owned timing stub | `selfcheck.json` |

Use `selfcheck` to test the client, not server performance. Use `run` or `sweep`
for real vLLM/SGLang evidence.

## Generate before replay when the source is raw

Choose `generator.type: coding-session` for recorded coding-agent sessions and
`generator.type: synthetic` for controlled shapes. Never describe a synthetic
trace as production-representative.

For `coding-session`, choose and report:

- `policy`: how raw prefix claims become replayable context;
- `session_order`, `arrival_rate`, `arrival_pattern`, and `seed`: the invented
  arrival timeline;
- `max_sessions`: selection bound, when used.

Read the emitted manifest before quoting cache numbers. For a recorded source,
report `parameters.folded_prefix_tokens`; those tokens were attributed to cache
by the source but must be prefetched by the canonical replay.

For `synthetic`, report the input/output/round distributions, arrival process,
compaction probability, and seed. The same seed and config must reproduce the
same canonical bytes.

## Configure `run` and `sweep`

Use the blocks in `configs/run.example.yaml` and `configs/sweep.example.yaml`:

- `input`: select one complete `format` for the whole file and declare only the
  tag column bundles the file actually carries;
- `corpus`: use a tokenizer matching the served model and a text corpus large
  enough not to fabricate repeated prompt content;
- media inputs may declare `synthetic` (a shape) instead of `asset` (a file);
  use it for capacity runs where content does not matter, pin `seed` for
  byte-identical content across requests and omit it for unique content;
- `server`: for multimodal formats the media surfaces are `openai-chat`,
  `openai-images`, `openai-image-edits`, `openai-videos`, `openai-speech`,
  `openai-transcriptions`, and `openai-translations`; a surface the chosen
  dialect does not serve is rejected at startup, not mid-run;
- `server`: select backend, endpoint, model, and temperature; for multimodal
  formats also set `dialect` (`openai`, `vllm`, `vllm-omni`, `sglang-omni`,
  `mstar`, `dynamo`) — `backend` picks the endpoint, `dialect` picks the field
  names and knob placement, and a mismatch fails silently because most
  servers ignore unrecognized fields;
- `replay`: select arrival timeline, capacity, context limit, and failure policy;
- `measurement`: select timeline recording and optional metric-specific SLO;
- `output`: select one artifact directory;
- `search` (sweep only): select the search question and rate range.

One file has one `InputFileFormat`; request family cannot vary by row. Use
`text-generation-session-execution-v2` for canonical closed-loop rounds and
`text-generation-independent` for standalone requests.

Keep arrival and capacity separate:

- `trace-timed` honors recorded arrival offsets and can be rescaled with
  `replay.rate`;
- `saturated` ignores recorded arrivals and cannot be combined with a rate;
- `max_concurrency` caps top-level units. A session holds its slot across all
  rounds and tool waits.

A sweep owns `rate`; omit `replay.rate`. Choose the question precisely:

- `max-sustainable-rate`: highest offered rate whose delivered step throughput
  keeps up after unit-to-step conversion;
- `peak-throughput`: maximum production rate, normally past the knee;
- `max-rate-under-slo`: highest rate preserving the target attainment;
- `grid`: only the explicitly listed rates.

## Verify the serving contract

Before a real run, inspect the exact process, model, endpoint, and backend. Do
not infer compatibility from a listening port.

| Backend | Base URL shape | Required server behavior |
|---|---|---|
| `openai` | ends in `/v1` | `/completions`, raw token-array prompt, streaming usage |
| `vllm-tokens` | no `/v1` | vLLM launched with `--tokens-only` |
| `sglang-tokens` | no `/v1` | `--skip-tokenizer-init` and incremental streaming output |

All live runs require prefix caching and cached-prompt-token details. vLLM needs
`--enable-prompt-tokens-details`; leave prefix caching enabled. The mandatory
two-request preflight must pass before treating cache metrics as evidence.

Do not restart or stop an existing server without explicit authorization. If a
dedicated test server is authorized, use an isolated port/GPU, keep long-lived
execution in `tmux`, and clean up only that owned process after the run.

## Execute and preserve evidence

1. Run launcher `--dry-run` to catch YAML errors and inspect resolved paths.
2. For static trace inspection without a server, set `replay.dry_run: true` and
   run the `run` task normally.
3. Confirm real endpoint readiness and model identity.
4. Execute the YAML without rewriting it into ad hoc flags.
5. Read the final panel, then inspect machine-readable artifacts before making
   claims. Use `terminal.log` for complete engine diagnostics.

The runner replaces private prompt text with synthetic token IDs. Say it
replays workload shapes—lengths, arrivals, session ordering, prefixes, and tool
waits—not original prompts or answers.

## Interpret outputs

For a run, always report:

- input format, trace/manifest identity, backend, arrival mode, and concurrency;
- attempted/success/failed steps and early-stopped sessions;
- request/output-token throughput plus TTFT and TPOT;
- planned and server-measured prefix hit rates together;
- timeline dropped-request count;
- SLO source and applicable denominator, when declared.

Distinguish these values:

- manifest `folded_prefix_tokens`: generation-time source correction;
- JSONL `prefix_shortfall_tokens`: runtime context missing after an earlier
  short or failed round;
- planned `prefix_len`: cache-eligible context, not proof of a cache hit;
- server `cached_prompt_tokens`: observed cache reuse.

The timeline is one row per streamed event, not one row per token. Check
`timeline.dropped_requests` before analysis; nonzero means it is sampled.

For a sweep, read `points` as measurement order and `curve` as rate order.
`knee` and `peak` answer different questions. `never_crossed` and
`always_crossed` locate the knee outside the searched range rather than at an
edge. Read `contamination_warning` before comparing cache results across points.

## Diagnose failures

| Symptom | Diagnosis/action |
|---|---|
| YAML duplicate/unknown/type error | Fix the named config path; never remove strict validation |
| `prefix-cache preflight failed` | Server lacks cache behavior or cached-token telemetry; fix server launch |
| HTTP route/model error | Confirm backend-specific base URL and served model from `/v1/models` or server config |
| Generation knobs appear ignored | Wrong `server.dialect`: knob placement differs per server (flat vs `extra_body` vs `nvext`) and unknown fields are dropped without error |
| cumulative streaming error | Enable incremental streaming output on SGLang |
| echoed-prompt/token-count error | Treat the response as untrustworthy; inspect server output-ID support |
| `rate` with `saturated` | Choose a timeline or saturation; use `max_concurrency` to bound saturation |
| context overflow | Fix `replay.context`, trace shape, or model limit; do not hide it in aggregation |
| fewer completed rounds than planned | Inspect the first failed round and `stop_session_on_error` policy |

## Maintain this skill with the interface

When changing launcher tasks, YAML keys, backend requirements, artifact names,
or result semantics, update in the same change:

1. `launcher/` validation and tests;
2. matching `configs/*.example.yaml`;
3. `README.md` and architecture documents;
4. this skill's task selection, workflow, and diagnosis guidance.

Keep detailed field enumeration in tested example YAML and README. Keep this
skill as the concise decision workflow; do not copy the Rust CLI reference back
into it.
