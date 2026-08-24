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

## Supported benchmarks and modalities

| Workload | Input | Output | What it measures |
|---|---|---|---|
| Text trace replay | Text | Text | Independent requests or closed-loop, multi-round sessions |
| Synthetic multimodal | Image, audio, or video + text | Text | Capacity at controlled media sizes and arrival rates |
| [Food-101](docs/FOOD101.md) | Image + text | Text | Image-to-text serving performance |
| [VBench T2I](docs/VBENCH.md) | Text | Image | Text-to-image serving performance |
| [VBench I2I](docs/VBENCH.md) | Image + text | Image | Image-to-image serving performance |
| [Seed-TTS](docs/SEED_TTS.md) | Text, optionally reference audio | Audio | Text-to-speech serving performance |

The included dataset adapters create reproducible load-generator inputs; they
do not currently score model output quality.

## Server interfaces

Request Factory separates the API surface from the serving-system dialect.
`server.backend` selects an endpoint such as chat, image generation, speech, or
realtime. `server.dialect` selects the exact request and response format used by
the server.

| `server.dialect` | Multimodal request format | Model parameters | Supported OpenAI-shaped surfaces |
|---|---|---|---|
| `openai` | OpenAI content parts | Standard fields | Chat, images, image edits, video, speech, transcription, translation, realtime |
| `vllm` | URL-based media parts | Flat fields | Chat, speech, transcription, translation |
| `vllm-omni` | URL-based media parts | Nested under `extra_body` | Chat, images, image edits, video, speech, realtime |
| `sglang-omni` | Top-level `images`, `audios`, and `videos` arrays | Flat fields | Chat, speech, transcription, translation, realtime |
| `mstar` | URL-based media parts | Flat fields | Chat, images, image edits, video, speech |
| `dynamo` | URL-based media parts | Nested under `nvext` | Chat, images, speech |

Text replay also supports OpenAI-compatible completions, vLLM's native token
endpoint, and SGLang's native token endpoint. Request Factory validates the
selected surface and dialect before a run so incompatible combinations fail
early. See the [configuration reference](docs/CONFIGURATION.md) for backend
names and options.

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
