# Replay Runner

`session_runner` replays typed workloads against OpenAI-compatible inference servers
(vLLM today, SGLang and others via the pluggable backend). The source schema is
selected explicitly with `--trace-format`; each frontend produces its own workload
variant instead of filling a universal row with zero/null placeholders.

## Trace Frontends

### `--trace-format session` (default)

Session-aware closed-loop replay preserves ordered chains of rounds:

```text
send round i -> wait for full LLM response -> sleep tool_wait_after_ms -> send round i+1
```

The session frontend preserves `prefix_len`, appended input length, target decode
length, round order, arrival, and tool/user waits. The VibeSim frontend preserves
independent request input/output shapes and arrivals. Both use a synthetic text
corpus, so raw private prompts are never replayed.

Supported session headers:

```csv
session_id,round_idx,prefix_len,input_len,output_len,tool_wait_after_ms
```

It also supports the canonical TraceLab export emitted by
`artifacts/trace_facts/csv_export`:

```csv
id,input_len,output_len,arrival_time,round_idx,tool_wait_after_ms,prefix_len
```

Fields:

- `session_id` / `id`: session identifier.
- `arrival_time`: synthetic session arrival time in milliseconds. Missing values default to `0`.
- `prefix_len`: number of prior context tokens kept for this round.
- `input_len`: number of new synthetic input tokens appended for this round.
- `output_len`: `max_tokens` sent to vLLM.
- `tool_wait_after_ms`: sleep after this round completes before the next round in the same session.

### `--trace-format vibesim`

VibeSim's L7 trace is an independent-request workload:

```csv
id,input_len,output_len,arrival_time
```

Each row remains a `VibeSimRequest`; it is not converted into a fake session with
round/prefix/tool-wait fields. `arrival_time` is interpreted in milliseconds,
matching VibeSim's trace frontend.

Frontend layout:

```text
src/trace/mod.rs       format dispatch + ReplayWorkload enum
src/trace/session.rs   session CSV → Sessions(...)
src/trace/vibesim.rs   VibeSim CSV → IndependentRequests(...)
```

Adding a source format means adding a parser module and a typed
`ReplayWorkload` variant. Adding a future modality should likewise add a typed
request/workload variant and backend capability; do not extend the session row
with unrelated optional fields.

Executor layout mirrors the workload variants:

```text
src/executor/mod.rs           shared run state, progress, and concurrency gate
src/executor/session.rs       ordered closed-loop session replay
src/executor/independent.rs   one-shot independent-request replay
```

Both executors call the same source-agnostic text-generation backend, but each
owns its source semantics and constructs its own typed log record. Adding a
frontend does not add imports or nullable source fields to `backend.rs`.

## Arrival Rate (`--rate`)

By default, the runner releases top-level workload units at the `arrival_time`
offsets stored in the CSV. Pass `--rate N` for a target of `N` sessions/s in
`session` mode or `N` requests/s in `vibesim` mode:

```bash
--rate 8
```

The runner measures the trace rate from the mean inter-session interval,
`(session_count - 1) / (last_arrival - first_arrival)`, then multiplies every arrival offset by
`trace_rate / requested_rate`. For example, requesting 8 sessions/s for a trace measured at
2 sessions/s divides every arrival time by 4. Relative gaps and simultaneous-arrival bursts are
preserved. The measured rate, target rate, and applied time scale are printed at startup.

Rate scaling requires at least two selected workload units with distinct arrival
times. `--rate` is applied after `--max-items`, so the measured trace rate
describes exactly the selected workload.

Examples live at `examples/session_workload_example.csv` (single session),
`examples/multi_session_example.csv` (3 sessions with arrival times, for a quick multi-session
run), and `examples/multi_session_large.csv` (48 sessions / 303 rounds with cumulative-consistent
prefixes, for an end-to-end prefix-cache hit-rate measurement). In the large example each round's
`prefix_len` equals the prior round's full context, so the planned hit rate is the true achievable
rate and the server-measured aggregate matches it within vLLM's 16-token block alignment.

## Text Corpus (`--text-file`)

The runner only needs the *token shape* of text, not its meaning, so any large UTF-8 text file works: your own code/logs, a Project Gutenberg book, a Wikipedia dump, etc. By default the pool auto-sizes to the workload — large enough that no single request repeats content and every session gets a distinct content window — with a floor of `100M` tokens (~400 MB of `u32`, ~400–600 MB of source text). Override with `--token-pool-limit`. The corpus must therefore supply at least that many tokens; the rest of a larger file is never read. If the resulting pool is still shorter than the longest prompt, the runner warns that synthetic content will repeat.

A convenient, widely used option is **enwik9**: the first 10^9 bytes of English Wikipedia from the Large Text Compression Benchmark. It is **not bundled** with this repository. Since enwik9 is derived from Wikipedia content, users should download it from the original source and comply with the applicable license terms.

```bash
curl -O http://mattmahoney.net/dc/enwik9.zip
unzip enwik9.zip   # -> ./enwik9 (~1 GB; tokenized up to the pool size, ~250M tokens available)
```

Then pass `--text-file ./enwik9`. Any other sufficiently large UTF-8 text file works just as well. For million-token sessions, prefer a large corpus like enwik9 so the pool can reach its full size.

## Request Path

The runner targets an OpenAI-compatible completions endpoint:

```text
POST {base_url}/completions
```

Pass `--base-url http://HOST:PORT/v1`. The wire protocol is selected with `--backend` (default `openai`, which covers vLLM and SGLang's OpenAI endpoint); the endpoint path, request body, and response parsing all live behind a `Backend` adapter in `src/backend.rs`, so adding a server (e.g. SGLang's native `/generate`) is a new adapter, not a rewrite. The backend accepts only normalized generation inputs and returns a common `GenerationOutcome`; it does not know `SessionStep`, `VibeSimRequest`, or future frontend types. The runner submits the exact prompt **token ids** directly (OpenAI's `prompt` accepts an integer array), so there is no client-side decode and the server's prefix-cache keys match the ids we built. With recent vLLM it also sets `return_token_ids` to carry the model's exact output tokens forward across rounds; servers that ignore the flag fall back to re-encoding the output text (a few tokens of drift).

## Typed Log Contract

Each JSONL line is a versioned envelope containing a tagged source record and a
source-agnostic generation outcome:

```json
{
  "schema_version": 2,
  "source": {
    "type": "vibe_sim_request",
    "data": {"id": "req-1", "input_len": 16, "prompt_len": 16,
             "output_len_target": 4, "arrival_time_ms": 12.5}
  },
  "outcome": {"request_id": "vibesim_req-1", "status": "SUCCESS", "...": "..."}
}
```

Session-only fields (`session_id`, `round_idx`, `prefix_len`, tool wait, planned
cache rate) exist only inside the `session_round` variant. They are not emitted
as null/zero fields on VibeSim requests. Optional fields inside `outcome` mean a
server observation was unavailable, not that the selected source lacks that
concept.

The run summary follows the same rule: `replay.kind` is `sessions` with a real
`prefix_cache` block, or `independent_requests` without that block.

## Build

```bash
cargo build --release --manifest-path replay/Cargo.toml --bin session_runner
```

## Dry Run

Dry-run mode validates and summarizes the CSV without contacting vLLM:

```bash
cargo run --manifest-path replay/Cargo.toml --bin session_runner -- \
  --trace replay/examples/session_workload_example.csv \
  --trace-format session \
  --text-file /path/to/text-corpus \
  --tokenizer /path/to/tokenizer.json \
  --model qwen3.6-35b-a3b-fp8 \
  --dry-run \
  --max-model-len 65536
```

`--text-file` and `--tokenizer` are still required by the CLI, but dry-run mode returns before loading them.

## Run Against vLLM

```bash
cargo run --release --manifest-path replay/Cargo.toml --bin session_runner -- \
  --trace replay/examples/session_workload_example.csv \
  --trace-format session \
  --text-file /path/to/text-corpus \
  --tokenizer /path/to/tokenizer.json \
  --model qwen3.6-35b-a3b-fp8 \
  --base-url http://127.0.0.1:60995/v1 \
  --stream-idle-timeout-secs 7200 \
  --max-model-len 65536 \
  --max-concurrency 1 \
  --summary-path /tmp/session_runner_summary.json \
  --log-path /tmp/session_runner.jsonl
```

Useful controls:

```bash
# Validate against a model context limit and report the first overflowing round.
--dry-run --max-model-len 131072

# Bound concurrent top-level workload units while still respecting arrival_time.
--max-concurrency 128

# Scale trace arrival offsets to a target of 8 sessions per second.
--rate 8

# Skip rounds that exceed a known model context limit instead of sending them to vLLM.
--max-model-len 131072 --fail-on-context-overflow

# Write one JSON summary containing workload stats and replay latency stats.
--summary-path /tmp/session_runner_summary.json
```

## Run Metrics

The typed `replay.*.common` summary reports the same end-to-end metrics for
session rounds and independent requests:

- `run_duration_ms`: wall-clock time from the earliest request submission to
  the latest completion, including arrival gaps and tool waits that occur
  between them.
- `request_throughput_per_s`: successful requests divided by run duration.
- `output_token_throughput_per_s`: actual output tokens from successful
  requests divided by run duration.
- `tpot_measured_steps` and `tpot_ms_{avg,p50,p90,max}`: per-request time per
  output token, measured only for successful requests with TTFT and at least two
  output tokens. For one request, `TPOT = (total_duration_ms - first_token_ms) /
  (output_len_actual - 1)`.

TPOT describes average decode cadence after the first token. The runner does
not report inter-token latency because it does not yet retain every streaming
chunk timestamp; `chunk_count` alone is not enough to reconstruct it.

## Prefix-Cache Accounting

The JSONL log includes per-round planned-vs-server cache fields:

- `planned_prefix_hit_rate`: `prefix_len / (prefix_len + input_len)` from the workload.
- `server_cached_prompt_tokens`: cached prompt tokens reported by vLLM usage, when available.
- `server_prefix_hit_rate`: `server_cached_prompt_tokens / server_prompt_tokens`, when available.
- `server_prefix_hit_rate_delta`: server hit rate minus planned hit rate for that round.

The runner always requests streaming usage and treats usage-present-but-cache-detail-absent as zero cached tokens (servers omit `prompt_tokens_details` when nothing was cached). For this to be meaningful, the server must report prompt-token details and have prefix caching enabled. With the Qwen helper script, start vLLM with both:

```bash
ENABLE_PROMPT_TOKENS_DETAILS=1 ENABLE_PREFIX_CACHING=1 web/ai_infra/serve_qwen36_35b_a3b_fp8_vllm.sh
```

Before replaying, the runner sends a two-request probe that forces a guaranteed prefix-cache hit and **aborts the run** if the server does not report cached prompt tokens. This fails fast on a server launched without prompt-token details (vLLM: `--enable-prompt-tokens-details`) or without prefix caching, instead of silently logging 0% hit rates. Dry-run mode skips the probe.

If the Qwen model is not already present locally, starting vLLM may download a large Hugging Face model and may execute model repository code depending on the serve flags.

## Current Scope

Implemented:

- typed `session` and `vibesim` trace frontends
- distinct session and independent-request workload variants (no sparse universal row)
- both `session_id` and canonical `id` CSV schemas
- workload summary and dry-run validation
- per-session ordered replay
- optional session-start scheduling from `arrival_time`
- optional target workload arrival-rate scaling with `--rate`
- optional top-level workload concurrency limit
- optional model-context validation and overflow skipping
- session-internal closed-loop timing
- `prefix_len + input_len` prompt construction
- direct token-id prompt submission (no client-side decode) + exact output-id carry-forward via vLLM `return_token_ids` (re-encode fallback)
- pluggable backend adapter (OpenAI-compatible today; vLLM and SGLang OpenAI endpoints)
- OpenAI-compatible streaming completions request
- startup prefix-cache preflight that aborts when the server reports no cached tokens
- TTFT and total latency logging
- JSON run summary output
- versioned JSONL output with tagged source records and a common generation outcome
- planned vs. server-reported prefix cache hit-rate logging

Not implemented yet:

- SGLang native `/generate` backend adapter
- TTFT/TPOT SLO judgment
- per-token timeline dump
- raw trace prompt/tool-result text reconstruction
- block-level Prometheus prefix-cache metric collection
