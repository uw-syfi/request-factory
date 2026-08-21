# Seed-TTS text-to-audio replay

This adapter turns the official Seed-TTS evaluation metadata into deterministic
`multimodal-independent-v1` requests for Qwen3-Omni, Orpheus, and future TTS
backends. Seed-TTS contains English Common Voice and Mandarin DiDiSpeech-2
examples. Each row has a target sentence plus a reference transcript and WAV
for zero-shot voice conditioning.

## M* parity and full benchmark semantics

M* uses the target sentence as a plain text-to-speech serving request. It does
not send the reference transcript or WAV, so its measurements exercise the TTS
path but do not evaluate zero-shot voice-cloning fidelity. This adapter uses
that behavior by default. Every request also carries M*'s exact Qwen system
prompt as a typed `system` input. Chat backends preserve that role; the
`openai-speech` adapter ignores system inputs and sends only the target text, so
one materialized artifact works for both Qwen3-Omni and Orpheus.

Use `--include-reference-audio` to emit ordered reference-text, reference-audio,
and target-text inputs for a backend that implements voice conditioning. The
reference voice metadata is always retained in `labels.jsonl`, even in M*
compatibility mode. Quality scoring such as WER and speaker similarity remains
outside req-frontend's serving-performance scope.

## Acquire and materialize

The official repository publishes its approximately 1.2 GB test archive through
Google Drive. The download helper uses `gdown`; with the CLI integration in
place, an isolated invocation is:

```bash
uv run --with gdown python -m benchmarks seed-tts \
  --dataset-dir data/seedtts_testset \
  --output-dir out/seed-tts-en \
  --download \
  --set en \
  --limit 160 \
  --arrival-rate 10 \
  --sample-rate-hz 24000 \
  --max-tokens 256
```

Omit `--download` when `--dataset-dir` already has the official layout:

```text
seedtts_testset/
├── en/
│   ├── meta.lst
│   └── prompt-wavs/*.wav
└── zh/
    ├── meta.lst
    ├── hardcase.lst
    └── prompt-wavs/*.wav
```

Supported set names are `en`, `zh`, `zh-hard`, `en-cross`, and `zh-cross`.
The standard English set is the default and matches M*'s current loader default.
The default `--selection mstar-order` takes rows in metadata order, just as M*
does. Use `--selection shuffled --seed 0` when a deterministic sample across the
whole set is more useful than exact workload parity.

The archive is downloaded atomically, checked as a readable tar archive, hashed,
and extracted without accepting links, devices, or paths outside the destination.
The official source does not publish a checksum, so the first successful local
download establishes the archive hash recorded in `manifest.json`.

## Artifacts

| File | Contract |
|---|---|
| `requests.jsonl` | Target text, deterministic arrival, and audio output specification; optionally includes reference conditioning |
| `labels.jsonl` | Original utterance ID, target text, reference transcript/WAV path and hash, and optional ground-truth WAV |
| `manifest.json` | Official source links, archive/metadata/artifact hashes, selected-record hash, seed, set, load controls, sample rate, and conditioning mode |

Selection is the metadata-order prefix used by M* or an optional seeded shuffle,
followed by `--limit`. Audio is never copied, resampled, or transcoded. The
requested output sample rate defaults to 24 kHz, matching Qwen3-Omni's waveform
output and Orpheus/SNAC serving conventions. The generated-audio limit defaults
to M*'s 256 tokens. Both can be overridden explicitly. Use `--voice Ethan` (or
another server-supported name) to put a voice on every output specification;
otherwise the serving backend chooses its default.

## Replay

For Qwen3-Omni's audio-capable chat API:

```bash
uv run python -m launcher run configs/seed-tts-qwen3-omni.example.yaml
```

The checked-in example selects `dialect: mstar`. Use `vllm-omni` or
`sglang-omni` when that stack serves the model; the dialect controls media
placement, knob nesting, and streamed-audio event shape.

For Orpheus through M*'s `/v1/audio/speech` service or another conforming
OpenAI-compatible wrapper:

```bash
uv run python -m launcher run configs/seed-tts-orpheus.example.yaml
```

This example also selects `mstar`, matching the service named above.

The upstream Canopy Labs Orpheus repository does not itself expose that
standard HTTP endpoint, so it requires M*'s service or a compatible wrapper.
Both paths stream generated audio, record first-output latency, byte and audio
throughput, duration and real-time factor, and can save returned audio after
the measured response completes. The launcher examples use saturated replay,
which is continuous closed-loop load at the configured concurrency; M*'s
offline harness instead executes discrete concurrency-sized waves after three
warmup waves.

## Data access and licensing

The official Seed-TTS repository links the archive but does not include a
top-level license file. The M* paper's dataset table identifies Seed-TTS as
CC BY 4.0. Verify the terms applicable to the underlying Common Voice and
DiDiSpeech-2 samples before redistributing either source audio or materialized
artifacts. This repository stores paths and hashes; it does not vendor audio.

## Integration API

`benchmarks.seed_tts` intentionally has no dependency on the shared CLI module:

- `add_seed_tts_parser(subparsers)` registers the command and arguments.
- `options_from_namespace(arguments)` creates `SeedTTSOptions`.
- `materialize_seed_tts(options)` writes the three artifacts.
- `acquire_seed_tts(dataset_dir)` downloads and safely extracts the source.
- `load_seed_tts(dataset_dir, evaluation_set)` parses an existing source tree.

The runtime supports Qwen3-Omni audio returned by streaming chat completions and
raw PCM16 returned by an OpenAI-compatible speech endpoint. Backend capability
validation rejects incompatible input/output combinations before replay.
