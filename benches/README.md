# Component benchmark scope

These Criterion benchmarks answer narrow implementation questions. Their rates
are component ceilings, not load-generator requests/s, and must not be presented
as backend, network, latency, or end-to-end capacity.

Run release benchmarks on an otherwise idle host and pin them to the same CPUs
when comparing revisions:

```bash
taskset -c 16-23 cargo bench --features bench-internals --bench dispatch
taskset -c 16-23 cargo bench --features bench-internals --bench vllm_wire
cargo build --release --features transport-bench --bin loadgen_transport_bench
taskset -c 16-31,48-63 target/release/loadgen_transport_bench \
  --transport reqwest-pool --endpoint http://127.0.0.1:18080/inference/v1/generate
taskset -c 16-31,48-63 target/release/loadgen_transport_bench \
  --transport hyper-pool --endpoint http://127.0.0.1:18080/inference/v1/generate
taskset -c 16-31,48-63 target/release/loadgen_transport_bench \
  --transport dedicated-http1 --endpoint http://127.0.0.1:18080/inference/v1/generate
```

Keep Criterion's saved baseline or record at least five independent command
runs and compare medians. CPU model, affinity, Rust version, benchmark revision,
and command line belong beside any reported number.

## `dispatch`

The primary `ready` case measures dispatcher bookkeeping and task lifecycle for
immediately ready synthetic units. `yield_once` adds exactly one scheduler
wakeup per unit. The benchmark compares the production one-spawn/one-join-per-unit
algorithm with a persistent-worker candidate across declared item counts and
concurrency windows. The candidate is benchmark-only; it is not evidence that
the production runner improved.

It includes runtime task spawn/join, the persistent workers' shared-iterator
lock, one shared-state clone/drop and atomic checksum update per item, and—in
the secondary case—one yield/wakeup. It excludes trace parsing, arrival timers,
prompt construction, HTTP/Reqwest,
JSON/SSE work, executor state, logging, timeline recording, summary folding,
server work, networking, and latency correctness. It therefore identifies
scheduler overhead but cannot predict the end-to-end improvement by itself.
The equal-work Criterion comparison measured roughly 7–13x higher component
throughput for the candidate, while its pinned end-to-end median was unchanged
(128,528 versus 128,057 requests/s), so it was not adopted.

## `vllm_wire`

The serialization cases invoke the production typed vLLM request adapter with
1, 128, and 4,096 token IDs. The parsing cases invoke the production typed event
adapter for a one-token event, a 32-token event, and a usage-only event.

Serialization includes serde JSON encoding and output-buffer allocation. Its
Criterion throughput is input token IDs per second, not requests per second.
Parsing includes typed deserialization, token-vector/string allocation where
applicable, and normalization into the internal event. Its Criterion throughput
is input JSON bytes per second, not events per second. Both exclude Reqwest
request building, headers, HTTP body transmission, networking, SSE
framing/chunk scanning, timestamps, stream folding, tokenization, logging,
summaries, and backend work.

## `chat_wire`

The chat body cannot be a derived struct: `max_tokens` is sent under every name
the dialect lists, `ignore_eos` is dialect-conditional, and one encoding moves
media from the message content to keys at the request root. It is therefore a
hand-written `Serialize` over borrowed inputs, and this benchmark exists because
the obvious way to write dialect-driven field names -- build a `serde_json::Value`
and index it -- is 4.4x slower on text and 4.8x slower with four images.

The cases cover 0, 1, and 4 image parts under the `vllm` dialect, plus one case
under `sglang-omni`, whose `TopLevelLists` encoding walks a different branch of
the same serializer. Each measures shaping plus JSON encoding plus output-buffer
allocation, and excludes everything `vllm_wire` excludes.

## `loadgen_transport_bench`

This external-responder component benchmark compares the production-style shared
Reqwest pool, a shared Hyper pool that helps isolate Reqwest wrapper overhead,
and a candidate in which each long-lived worker owns one Hyper HTTP/1 connection.
Every worker performs one untimed warm-up request and reports whether it
succeeded. The clock starts only after all workers are ready, then a broadcast
releases them to send their fixed shares sequentially. All variants build an
HTTP request, transmit the same pre-serialized one-token vLLM payload, receive
the complete chunked SSE body, and discard its bytes.

It includes Tokio scheduling of the fixed worker fleet, HTTP request construction,
pool checkout/return in the Reqwest case, TCP syscalls, HTTP/1 and chunked-body
decoding, loopback or selected network transport, and the responder's time. It
excludes workload parsing, prompt generation, JSON serialization, SSE event
framing/parsing, stream metrics, result records, logging, timeline, summaries,
arrival policy, session dependencies, retries, TLS, and initial connection
establishment from the timed region. The Reqwest case matches the production
client's one-hour overall request timeout; the direct Hyper cases do not impose
an application timeout. Both Hyper cases support `http://` only.
Therefore it is a transport comparison against the named responder, not an
end-to-end load-generator or general Internet HTTP benchmark.

A preliminary pinned localhost comparison used 1,000,000 requests, 32
connections, 16 runtime workers, responder CPUs 0–15, and client CPUs
16–31,48–63. Three Reqwest-pool runs delivered 213,298, 282,971, and 273,067
requests/s (median 273,067); the dedicated HTTP/1 candidate delivered 290,477,
252,982, and 223,511 requests/s (median 252,982). The candidate was both noisier
and slower by median, so it was not adopted in production. These are
responder-specific component results, not the load generator's end-to-end
capacity. They also predate the corrected start broadcast and should be rerun
with at least five samples before being used as a regression baseline.

## End-to-end acceptance

A component win lands only when the pinned `loadgen_perf_server` workload also
improves. That run includes request construction, HTTP transport, SSE framing and
parsing, executor bookkeeping, enabled measurements, logging configuration, and
the CPU responder. Its documentation must state request shape, concurrency,
runtime workers, affinity, responder topology, and which output paths are enabled.
