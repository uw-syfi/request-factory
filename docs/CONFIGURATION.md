# Configuration reference

The supported operator interface is one launcher task plus one YAML file:

```bash
uv run python -m launcher {run|sweep|tracegen|selfcheck} CONFIG.yaml
```

Paths are resolved relative to the YAML file. Unknown and duplicate keys are
errors. Use launcher `--dry-run` to validate and print the resolved engine
command without building or running it.

Launcher flags apply to every task: `--build-type {release,debug}` defaults to
`release`; `--dry-run` stops after validation/command rendering; and
`--show-engine-output` streams the full engine output that is otherwise kept in
`terminal.log`.

## `run` and shared `sweep` blocks

`sweep` accepts the same blocks, except that `search` owns the rate,
`replay.processes` must be `1`, and `replay.dry_run` is unavailable.

### Input and corpus

| Key | Default | Meaning |
|---|---|---|
| `input.trace` | required | CSV or JSONL workload path |
| `input.format` | required | Executable values: `text-generation-session-execution-v2`, `text-generation-independent`, `multimodal-independent-v1` |
| `input.tags` | `[]` | `slo` and/or `priority` column bundles for text CSV |
| `corpus.text_file` | required for text | Text used to construct synthetic token-ID prompts |
| `corpus.tokenizer` | required for text | `tokenizer.json`, model directory, or Hugging Face repository ID |
| `corpus.token_pool_limit` | workload-derived | Positive cap on the token pool |

Omit `corpus` for multimodal JSONL. Eight other `InputFileFormat` values describe
shape-only media CSV schemas shared with simulators, but the live runner rejects
them because they do not contain media bytes: `image-to-text-independent`,
`video-to-text-independent`, `audio-to-text-independent`,
`text-to-image-independent`, `text-to-video-independent`,
`text-to-speech-independent`, `image-to-video-independent`, and
`omni-generation-independent`.

### Server

| Key | Default | Meaning |
|---|---|---|
| `server.model` | required | Served model name; accepted but not sent by `sglang-tokens` |
| `server.base_url` | `http://127.0.0.1:8000/v1` | Include `/v1` for OpenAI-shaped surfaces; omit it for native token endpoints |
| `server.temperature` | `0.0` | Sampling temperature where supported |
| `server.backend` | `openai` | Endpoint and response surface; values listed below |
| `server.dialect` | `openai` | Serving-system vocabulary and framing; `openai`, `vllm`, `vllm-omni`, `sglang-omni`, `mstar`, or `dynamo` |

Backends are `openai`, `vllm-tokens`, `sglang-tokens`, `openai-chat`,
`openai-images`, `openai-speech`, `openai-image-edits`, `openai-videos`,
`openai-transcriptions`, `openai-translations`, and `openai-realtime`. The last
eight require `multimodal-independent-v1`. `backend` chooses the endpoint;
`dialect` chooses field names, knob placement, route suffixes, and stream events.
See the support matrix in the [README](../README.md#serverdialect-which-vocabulary-not-which-endpoint).

### Replay and measurement

| Key | Default | Meaning |
|---|---|---|
| `replay.arrival_mode` | `trace-timed` | `trace-timed` honors/rescales arrivals; `saturated` releases everything immediately |
| `replay.rate` | trace rate | Positive workload units/s; incompatible with `saturated` and owned by `search` during sweeps |
| `replay.max_items` | all | Keep the first N units after validating the full file |
| `replay.max_concurrency` | unlimited | Active top-level units; a session holds its slot across rounds and tool waits |
| `replay.processes` | `1` | Process shards; values above one require a saturated `run` |
| `replay.runtime_worker_threads` | min(host CPUs, 16) | Tokio workers per process |
| `replay.stream_idle_timeout_seconds` | `600` | Fail after this many seconds without a stream chunk |
| `replay.stop_session_on_error` | `true` | Stop later rounds after the first failed session round |
| `replay.dry_run` | `false` | Validate and summarize the workload without tokenizer or network access |
| `replay.context.max_model_len` | unset | Report context overflow; text replay only |
| `replay.context.on_limit` | `send` | `send` anyway or `skip` while reserving one context token |
| `measurement.timeline` | `true` | Write per-event Parquet without blocking submission |
| `measurement.request_log` | `true` | Write schema-v15 per-request JSONL; disable for summary-only capacity tests |
| `measurement.slo.ttft_ms` | unset | Run-level TTFT upper bound |
| `measurement.slo.tpot_ms` | unset | Run-level token-delivery TPOT upper bound |
| `measurement.slo.e2e_ms` | unset | Run-level submission-to-completion upper bound |

### Output

| Key | Default | Meaning |
|---|---|---|
| `output.directory` | `out/run` or `out/sweep` | Artifact root |
| `output.requests` | `requests.jsonl` | Request-log filename |
| `output.summary` | `summary.json` | Run-summary filename |
| `output.timeline` | `timeline.parquet` | Timeline filename |
| `output.terminal` | `terminal.log` | Complete engine output filename |
| `output.artifacts` | `artifacts` | Generated-media directory name |

Output names must be single path components. Every launcher run also snapshots
`launcher-config.yaml` and `command.txt`.

## `sweep`-only keys

| Key | Default | Meaning |
|---|---|---|
| `search.mode` | `max-sustainable-rate` | `max-sustainable-rate`, `peak-throughput`, `max-rate-under-slo`, or `grid` |
| `search.start_rate` | `1.0` | First offered workload units/s |
| `search.max_rate` / `min_rate` | `4096.0` / `0.001` | Search bounds |
| `search.tolerance` | `0.05` | Relative knee-bracket stopping width |
| `search.densify_points` | `3` | Extra points across a located knee |
| `search.max_shortfall` | `0.10` | Sustainable-rate allowed delivered-throughput shortfall |
| `search.min_gain` | `0.03` | Peak mode improvement still counted as rising |
| `search.patience` | `2` | Consecutive non-improving peak points before stopping |
| `search.plateau_tolerance` | `0.02` | Fraction below peak still considered on its plateau |
| `search.peak_metric` | `output-tokens` | `output-tokens` or `requests` |
| `search.target_attainment` | `0.99` | Required attainment for `max-rate-under-slo` |
| `search.attainment_metric` | `overall` | `overall` or `declared-slo` |
| `search.rates` | `[]` | Positive rates required by `grid` |
| `search.resume` | `true` | Reuse successful point directories |
| `visualization.enabled` | `false` | Run `viz` after a successful sweep |

## `tracegen`

`output.trace` is required; `output.terminal` defaults to `terminal.log`.

For `generator.type: synthetic`, available keys are `sessions` (`100`),
`rounds` (`uniform:1..8`), `input_len` (`lognormal:1024,0.8`), `output_len`
(`lognormal:256,0.7`), `tool_wait_ms` (`lognormal:500,1.0`),
`compaction_probability` (`0`), `arrival_rate` (`1.0`), `arrival_pattern`
(`poisson` or `constant`), and `seed` (`0`). Length fields accept a positive
integer, `fixed:N`, `uniform:A..B`, or `lognormal:MEDIAN,SIGMA`.

For `generator.type: coding-session`, `source` is required. Other keys are
`source_schema` (`session-rounds-v2`, currently the only value), `policy`
(`trace-reported` or `monotonic`), `max_sessions` (all), `session_order`
(`source` or `shuffle`), `arrival_rate` (`1.0`), `arrival_pattern` (`poisson` or
`constant`), and `seed` (`0`).

## `selfcheck`

`tokenizer.path` is required. `checks.pairs` defaults to `2`, `checks.port` to
`8271`, `output.directory` to `out/selfcheck`, and `output.terminal` to
`terminal.log`. The harness owns the loopback port while it validates timing and
measurement fidelity against the controlled stub.
