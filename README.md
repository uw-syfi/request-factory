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

## Supported modalities and benchmarks

| Input → output | Example benchmarks and workloads |
|---|---|
| Text → text | [TraceLab](https://github.com/uw-syfi/TraceLab) coding-agent traces; synthetic independent requests and multi-round sessions |
| Image + text → text | [Food-101](docs/FOOD101.md); synthetic image capacity workloads |
| Audio + text → text | Synthetic audio capacity workloads |
| Video + text → text | Synthetic video capacity workloads |
| Text → image | [VBench T2I](docs/VBENCH.md) |
| Image + text → image | [VBench I2I](docs/VBENCH.md) |
| Text → audio | [Seed-TTS](docs/SEED_TTS.md) |

## Server interfaces

Request Factory supports:

- OpenAI-compatible text, chat, image, video, speech, transcription,
  translation, and realtime APIs.
- vLLM's OpenAI-compatible and native token interfaces, including vLLM-Omni.
- SGLang's native token interface and SGLang-Omni APIs.
- [M*](https://mstar.stanford.edu/) and NVIDIA Dynamo multimodal APIs.

Multimodal servers often use different payload fields and streaming events for
the same API surface. Request Factory provides dialect profiles for OpenAI,
vLLM, vLLM-Omni, SGLang-Omni, M*, and Dynamo, and validates incompatible
combinations before a run. See the
[configuration reference](docs/CONFIGURATION.md) for details.

## Quick start

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
- A Rust toolchain

### 1. Start an LLM serving system

Choose one of these serving options with a model that supports at least a 45K
token context window.

#### Option A: vLLM

```bash
python -m vllm.entrypoints.cli.main serve MODEL \
  --stream-interval 1 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details
```

#### Option B: SGLang

```bash
python -m sglang.launch_server \
  --model-path MODEL \
  --host 0.0.0.0 --port 30000 \
  --skip-tokenizer-init \
  --stream-output
```

### 2. Run Request Factory

Clone the repository:

```bash
git clone https://github.com/uw-syfi/request-factory.git
cd request-factory
```

The quick start replays the first session from the bundled
[TraceLab](https://github.com/uw-syfi/TraceLab) coding-agent workload,
preserving its round order, prompt and output lengths, prefix reuse, and tool
waits.

Save the configuration matching your server as `configs/run.local.yaml` and
replace `MODEL` with the same model ID or local path.

#### vLLM configuration

```yaml
input:
  trace: ../examples/session_execution_v2_example.csv
  format: text-generation-session-execution-v2

corpus:
  text_file: ../README.md
  tokenizer: MODEL

server:
  backend: openai
  base_url: http://127.0.0.1:8000/v1
  model: MODEL

replay:
  max_concurrency: 8
  max_items: 1

output:
  directory: ../out/vllm
```

#### SGLang configuration

```yaml
input:
  trace: ../examples/session_execution_v2_example.csv
  format: text-generation-session-execution-v2

corpus:
  text_file: ../README.md
  tokenizer: MODEL

server:
  backend: sglang-tokens
  base_url: http://127.0.0.1:30000
  model: MODEL

replay:
  max_concurrency: 8
  max_items: 1

output:
  directory: ../out/sglang
```

Validate the configuration without contacting the server:

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
