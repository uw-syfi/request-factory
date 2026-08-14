# Adding modality-compositional benchmarks

The public benchmark boundary is any-input to any-output. A benchmark adapter
describes request semantics; it does not implement one executor for each pair
of modalities.

## Canonical request model

`schema::RequestSpec` contains an ordered list of `InputPart` values and one or
more `OutputSpec` values. Both enums currently cover text, image, audio, video,
and tensor data. Repeated and mixed inputs are legal. A role-aware `system`
input is text for capability checking, must precede every user input, and is
encoded as a system message by chat transports. Plain prompt transports ignore
it, which lets one text-to-audio artifact target both chat and speech APIs.

On disk, `multimodal-independent-v1` is JSON Lines with one `RequestSpec` per
non-empty line. The loader validates every record, unique IDs, and nondecreasing
arrival times. Nested modality data stays native JSON rather than escaped text
inside a CSV cell.

```json
{
  "id": "example-1",
  "arrival_time_ms": 0,
  "inputs": [
    {"type": "system", "text": "Answer precisely."},
    {
      "type": "image",
      "asset": {
        "path": "assets/example.jpg",
        "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "media_type": "image/jpeg"
      }
    },
    {"type": "text", "text": "What is shown?"}
  ],
  "outputs": [{"type": "text", "max_tokens": 64}]
}
```

Paths are relative to the request artifact. Dataset materializers should hash
every referenced file and preserve source identifiers, split, selection seed,
and source revision in a provenance manifest. They must not resize, re-encode,
or otherwise mutate source media: preprocessing belongs to the system under
test.

## Why adding modalities is not N by N

The execution path has three independently extensible pieces:

```text
InputPart encoders -> protocol/backend adapter -> modality-neutral output events
```

A protocol adapter publishes a `CapabilityProfile`: accepted input modalities,
produced output modalities, and whether it supports mixed inputs or multiple
outputs. That profile validates a complete request before replay. Adding image
input extends an encoder once; it does not create image-to-text,
image-to-audio, image-to-video, and image-to-tensor executors.

Pair-specific code is reserved for interactions that really differ, such as an
editing endpoint that couples an input image with output mask controls or a
robotics rollout that feeds each tensor output into the next request. Such an
adapter performs an additional validation after the general capability check.

## Runtime extension checklist

Most benchmark additions should contain only a Python materializer and fixture:

1. deterministically select source examples;
2. emit canonical requests and immutable asset references;
3. emit provenance, including source URL/revision and hashes;
4. choose an existing backend adapter (`openai-chat`, `openai-images`, or
   `openai-speech`) and validate its input/output capability set;
5. test conversion against a tiny checked-in fixture and a downloaded source
   sample.

Rust changes are needed only when the benchmark requires a new modality, wire
protocol, output observer, dependency topology, or measurement primitive. In
that case:

1. add the modality to `InputPart` or `OutputSpec`;
2. teach the relevant protocol adapter to encode or observe it;
3. add it to that adapter's `CapabilityProfile`;
4. add wire-shape and mock-server tests;
5. expose first-output, byte, duration, and artifact metadata through the shared
outcome rather than introducing benchmark-only timing fields;
6. update this document and the backend support table.

`assets::AssetStore` resolves asset paths relative to the request artifact,
verifies SHA-256 and MIME declarations, caches immutable bytes, and can produce
data URLs for OpenAI-compatible transports. Preload assets before starting the
arrival clock so disk reads and verification do not contaminate request
latency.
