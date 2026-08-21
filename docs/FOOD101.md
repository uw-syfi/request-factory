# Food101 image-to-text replay

This adapter materializes the Food-101 images used for BAGEL image-to-text
serving evaluation into req-frontend's canonical
`multimodal-independent-v1` artifact. It is a serving-performance workload,
not an accuracy harness and not a dependency on M*.

## Materialize

```bash
uv run python -m benchmarks food101 \
  --dataset-dir data/food-101 \
  --output-dir out/food101 \
  --download \
  --split test \
  --limit 1000 \
  --seed 0 \
  --arrival-rate 10 \
  --max-tokens 64
```

`--download` fetches and safely extracts the official ETH Zurich archive. An
interrupted `.partial` download resumes with an HTTP range request. Omit the
flag when `--dataset-dir` already contains an extracted Food-101 tree.

Selection is a seeded shuffle of the named split followed by `--limit`. The
same source split, seed, and limit select the same examples. Materialization
does not copy, resize, or transcode JPEGs.

The output directory contains:

| File | Contract |
|---|---|
| `requests.jsonl` | Canonical requests with image hash, original image path, constant classification prompt, arrival, and text output target |
| `labels.jsonl` | Request ID to normalized Food-101 class label; reserved for a future quality evaluator |
| `manifest.json` | Source URL/archive hash, split hash, seed, selection size, prompt/load controls, and output hashes |

## Validate without a model

The mock is a threaded CPU HTTP server. It rejects malformed message content,
decodes every data URL, requires non-empty media, records received byte counts,
and streams text and a final usage event.

```bash
uv run python tools/mock_multimodal_server.py \
  --dialect vllm --port 8000 \
  --log-path /tmp/food101-mock.jsonl --chunk-delay-ms 2

uv run python -m launcher run configs/food101.example.yaml
```

This exercises dataset loading, SHA-256 verification, MIME handling, base64
encoding, arrival release, admission, concurrent HTTP submission, SSE parsing,
request logs, summaries, and optional Parquet timelines. It does not validate
BAGEL model correctness or GPU preprocessing performance.

## Replay against vLLM

Serve a vLLM BAGEL configuration that exposes OpenAI-compatible chat
completions, then set these fields in the example config:

```yaml
server:
  backend: openai-chat
  base_url: http://HOST:PORT/v1
  model: THE_SERVED_BAGEL_MODEL_NAME
```

No corpus or tokenizer block belongs in this run. req-frontend sends original
image bytes and text; the system under test chooses image preprocessing,
encoded-token expansion, batching, and caching. Prefix-cache preflight and
prefix-hit summaries remain text-session concerns and are not fabricated for
this independent image workload.

This benchmark intentionally observes streamed text output. The runtime also
implements image, audio, and video generation plus transcription, translation,
and realtime surfaces; tensor output still needs a concrete protocol and
observer.
