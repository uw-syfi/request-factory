# Testing and benchmarking

The repository separates correctness, measurement fidelity, component cost, and
end-to-end capacity. A number from one layer must not be presented as evidence
for another.

## Correctness suites

Run the standard checks from the repository root:

```bash
cargo test -q
cargo test -q --no-default-features
cargo fmt --check
cargo clippy --all-targets
uv run pytest -q
uv run ruff check .
(cd viz && uv run pytest -q)
```

Rust tests live beside their modules. They cover exact input schemas, trace
validation, scheduling/admission, wire serialization and parsing, dialect
profiles, stream integrity, records, summaries, sweeps, and trace generation.
The no-default-features run protects the lightweight schema-only library.

Python tests cover four boundaries:

| Area | Main tests | What is proved |
|---|---|---|
| Launcher | `tests/test_launcher.py` | Strict YAML types/keys, path resolution, argv lowering, process sharding, UI |
| Materializers | `test_*_benchmark.py`, `test_synthetic_content.py` | Deterministic selection, manifests, hashes, canonical request shapes |
| HTTP media surfaces | `test_mock_server.py`, `test_multimodal_replay.py` | Dialect enforcement plus chat, image, speech, edit, video, transcription, and translation request/response paths |
| Realtime | `tests/test_realtime.py` | OpenAI, vLLM-Omni, and SGLang-Omni event names, payload fields, turn styles, and loud failure for a mismatched dialect |

`tools/mock_multimodal_server.py` is intentionally stricter than a permissive
OpenAI-compatible server: it rejects the wrong media encoding or knob envelope.
These tests validate client protocol behavior, not model quality, GPU
preprocessing, or compatibility with every server release.

## Measurement self-check

`selfcheck` runs the production replay path against `tools/stub_server.py`, whose
prefill delay, chunk delay, capacity, token counts, and cache behavior are known:

```bash
uv run python -m launcher selfcheck configs/selfcheck.example.yaml
```

It checks release lag, rate scaling, TTFT, TPOT, E2E, prompt/output accounting,
planned prefix reuse, and the difference between timeline-on and timeline-off
runs. Bounds and results are recorded in `selfcheck.json`; any failed claim exits
nonzero.

## Component benchmarks

Criterion benchmarks isolate implementation costs:

```bash
cargo bench --features bench-internals --bench dispatch
cargo bench --features bench-internals --bench vllm_wire
cargo bench --features bench-internals --bench chat_wire
```

`dispatch` measures task lifecycle/admission, `vllm_wire` measures typed native
token serialization/parsing, and `chat_wire` measures dialect-driven borrowed
serialization. They exclude most or all HTTP, server, logging, and workload
costs. See [the component scope](../benches/README.md) before quoting their rates.

The external-responder transport comparison is built separately:

```bash
cargo build --release --features transport-bench --bin loadgen_transport_bench
target/release/loadgen_transport_bench --help
```

It compares shared Reqwest, shared Hyper, and dedicated HTTP/1 connections; it is
not a full replay benchmark.

## End-to-end performance and benchmark workloads

Load-generator capacity is measured with a release build against
`loadgen_perf_server`, recording request shape, concurrency, Tokio workers, CPU
affinity, enabled output paths, topology, and medians across repeated runs. The
current method and historical results are in
[Load-generator performance](LOADGEN_PERFORMANCE.md). Component improvements are
accepted only when this path also improves.

Dataset materializers create reproducible serving workloads rather than client
microbenchmarks:

- Food101: image-to-text serving; labels are emitted but accuracy is not scored.
- VBench: text/image-to-image generation inputs; no VBench quality scoring.
- Seed-TTS: text-to-audio serving; no WER or speaker-similarity scoring.
- Synthetic media: controlled byte shape/rate for capacity, not content quality.

For live-server claims, preserve the resolved YAML, command, request log,
summary, timeline settings, server launch/configuration, and dataset manifest.
Mock success alone is not evidence that a particular model honors every
model-specific knob.
