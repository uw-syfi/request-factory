# Load-generator performance

The load generator has two capacity tiers:

- A single `session_runner` process defaults to at most eight Tokio workers. Use
  `--runtime-worker-threads` to override that value after profiling the host.
- A saturated YAML run can set `replay.processes` to partition top-level trace
  units across independent processes. The launcher writes each process's logs,
  timeline, and summary under `output/shards/` and writes aggregate throughput to
  the configured summary path.

```yaml
replay:
  arrival_mode: saturated
  max_concurrency: 256
  runtime_worker_threads: 8
  processes: 4

measurement:
  timeline: false
  request_log: false
```

Run it with the normal launcher:

```bash
uv run python -m launcher run configs/run.example.yaml --build-type release
```

`processes` is deliberately limited to saturated runs. Timed arrivals need a
cross-process start barrier before their latency and release-lag measurements can
be combined honestly. Each process receives canonical top-level units whose
ordinal satisfies `ordinal % processes == shard_index`; a session is never split
between processes. Rate rescaling happens before partitioning, so direct uses of
`--shard-count` and `--shard-index` retain the aggregate trace rate.

Set `measurement.request_log: false` for a pure capacity test. Summaries are still
folded from every request, but the runner does not serialize and persist the full
JSONL record. At hundreds of thousands of requests per second, full request logs
can require hundreds of MB/s of storage bandwidth; that is an output-system limit,
not an HTTP generation limit. Keep request logging enabled when per-request audit
records are part of the benchmark contract.

## Profile and benchmark method

The optimization work used a release build and the in-repository
`loadgen_perf_server`, with server and client pinned to disjoint CPU sets. The
capacity workload was 100,000 independent one-input-token, one-output-token
requests, saturated at concurrency 256, with timeline collection disabled.
Five-run medians were used for A/B comparisons. Linux `perf` was unavailable on
the host (`perf_event_paranoid=3`), so Callgrind and `strace -c` supplied the CPU
and syscall profiles.

Profile findings and corresponding changes:

- Per-request log flushes produced 12,517 `write` calls for 10,000 requests.
  Batched blocking writes reduced that to 2,778.
- Eager task creation consumed 428.9 MiB peak RSS for 100,000 queued requests.
  Bounded dispatch reduced it to 46.4 MiB, a 9.25x reduction.
- Callgrind attributed about 29% of `run_step` instructions to request JSON
  construction/serialization and response DOM parsing. Typed byte-level wire
  data raised the pinned median from 111,725 to 130,976 requests/s (+17.2%).
- A runtime sweep found a contention knee around eight workers. Eight workers
  delivered a 122,507 requests/s median versus 64,604 at 32 workers (1.90x).

The original pinned Rust-responder baseline was 26,912 requests/s. Three
four-process launcher runs against one 16-worker responder completed the same
100,000-request trace at 296,547, 301,251, and 311,449 aggregate requests/s. The
301,251 median is an 11.2x increase, with full request logging enabled. A separate
four-client, four-endpoint run reached 303,684 requests/s. The responder was
intentionally CPU-only; real deployment capacity remains bounded by the server,
network, request size, response event count, and enabled measurement outputs.

For reproducible comparisons, record CPU affinity, Tokio worker count, request
shape, concurrency, timeline/request-log settings, backend topology, and the
median of multiple runs. Never compare a one-token summary-only capacity number
directly with an event-heavy, durable-logging benchmark.
