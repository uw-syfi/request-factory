<div align="center">

# Request Factory

**Replay realistic text and multimodal workloads against AI inference servers.**

[![CI](https://github.com/uw-syfi/request-factory/actions/workflows/ci.yml/badge.svg)](https://github.com/uw-syfi/request-factory/actions/workflows/ci.yml)

</div>

Request Factory is a load generator and benchmark runner for OpenAI-compatible,
vLLM, and SGLang inference servers. It turns recorded or synthetic workloads
into reproducible streaming requests and captures the performance of the full
client-visible path.

## What you can do

- Replay independent requests or multi-round sessions while preserving arrival
  timing, session order, output lengths, and tool waits.
- Exercise text, image, audio, video, transcription, and realtime APIs.
- Generate synthetic workloads or prepare benchmark datasets such as Food-101,
  VBench, and Seed-TTS.
- Find sustainable serving rates with automated load sweeps.
- Measure throughput, time to first token/output, time per output token, end-to-end
  latency, errors, and prefix-cache behavior.
- Save reproducible run artifacts, including the resolved configuration,
  request logs, summaries, and event timelines.

## Quick start

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- A Rust toolchain
- A running inference server and its matching tokenizer

Clone the repository, then copy the example run configuration:

```bash
git clone https://github.com/uw-syfi/request-factory.git
cd request-factory
cp configs/run.example.yaml configs/run.local.yaml
```

Edit `configs/run.local.yaml` with your server URL, served model name,
tokenizer, corpus, and output directory. Validate the configuration without
contacting the server:

```bash
uv run python -m launcher run configs/run.local.yaml --dry-run
```

Then run the workload:

```bash
uv run python -m launcher run configs/run.local.yaml
```

The launcher builds the Rust runner automatically and writes results to the
configured output directory. The default artifacts include a JSON summary,
per-request JSONL, a Parquet event timeline, the resolved configuration, and
the complete terminal log.

## Other workflows

```bash
# Generate a workload trace
uv run python -m launcher tracegen configs/tracegen.example.yaml

# Search for a server's sustainable request rate
uv run python -m launcher sweep configs/sweep.example.yaml

# Validate measurement behavior against the included local stub
uv run python -m launcher selfcheck configs/selfcheck.example.yaml
```

Example configurations for text, multimodal, realtime, and benchmark workloads
are available in [`configs/`](configs/).

## Learn more

- [Configuration reference](docs/CONFIGURATION.md)
- [Architecture](ARCHITECTURE.md)
- [Adding benchmarks](docs/ADDING_BENCHMARKS.md)
- [Load-generator performance](docs/LOADGEN_PERFORMANCE.md)
- [Testing](docs/TESTING.md)
- Benchmark guides: [Food-101](docs/FOOD101.md), [VBench](docs/VBENCH.md), and
  [Seed-TTS](docs/SEED_TTS.md)

## License

[Apache License 2.0](LICENSE)
