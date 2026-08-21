"""Synthetic multimodal requests: load without a dataset.

A benchmark adapter turns a real corpus into requests. This one turns *numbers*
into requests -- how many, how big, how fast -- for the case where the question
is capacity rather than accuracy: how does the server behave at 4 MB of image
per request, or 30 s of audio, at N requests per second.

The engine generates the bytes at load time (see ``src/synthetic.rs``); this
materializer only declares their shape, so a trace stays small no matter how
large the media it describes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MODALITIES = ("image", "audio", "video")


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "synthetic", help="generate multimodal requests with synthetic media content"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--modality", choices=MODALITIES, default="image")
    parser.add_argument("--arrival-rate", type=float, default=1.0)
    parser.add_argument(
        "--prompt", default="Describe the input.", help="text part sent alongside the media"
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    # Shape. Only the flags for the chosen modality are read.
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--sample-rate-hz", type=int, default=24_000)
    parser.add_argument("--duration-ms", type=int, default=1_000)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument(
        "--content-seed",
        type=int,
        help=(
            "pin generated content to this seed so every request carries "
            "byte-identical media; omit for content unique per request"
        ),
    )


def _synthetic_spec(arguments: argparse.Namespace) -> dict:
    spec: dict[str, object] = {}
    if arguments.content_seed is not None:
        spec["seed"] = arguments.content_seed
    if arguments.modality in ("image", "video"):
        spec["width"] = arguments.width
        spec["height"] = arguments.height
    if arguments.modality == "audio":
        spec["sample_rate_hz"] = arguments.sample_rate_hz
        spec["duration_ms"] = arguments.duration_ms
    if arguments.modality == "video":
        spec["frames"] = arguments.frames
        spec["fps"] = arguments.fps
    return spec


def _generated_bytes(arguments: argparse.Namespace) -> int:
    """Exactly what one generated input will weigh.

    Exact rather than approximate so an operator can size a run before making
    it. Mirrors the encoders in ``src/synthetic.rs``; a test asserts the two
    agree by measuring what the server actually receives, because a size
    estimate that silently drifts from the generator is worse than none.
    """
    if arguments.modality == "image":
        # 8 signature + IHDR(25) + IDAT(12 + zlib) + IEND(12); zlib is a 2-byte
        # header, a 5-byte stored-block header per 65535 bytes, and adler32.
        raw = arguments.height * (arguments.width * 3 + 1)
        blocks = max(-(-raw // 0xFFFF), 1)
        return 57 + 2 + 4 + blocks * 5 + raw
    if arguments.modality == "audio":
        # 44-byte canonical WAV header plus 16-bit mono samples.
        return 44 + arguments.sample_rate_hz * arguments.duration_ms // 1000 * 2
    # 28-byte ftyp box plus an 8-byte mdat header around the payload.
    return 36 + max(arguments.width * arguments.height * 3 // 8, 1) * arguments.frames


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize(arguments: argparse.Namespace) -> int:
    if arguments.count <= 0:
        raise ValueError("--count must be greater than zero")
    if arguments.arrival_rate <= 0:
        raise ValueError("--arrival-rate must be greater than zero")
    if arguments.max_tokens <= 0:
        raise ValueError("--max-tokens must be greater than zero")
    spec = _synthetic_spec(arguments)
    for key, value in spec.items():
        if isinstance(value, (int, float)) and key != "seed" and value <= 0:
            raise ValueError(f"--{key.replace('_', '-')} must be greater than zero")

    output_dir = arguments.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    requests_path = output_dir / "requests.jsonl"
    interval_ms = 1000.0 / arguments.arrival_rate
    with requests_path.open("w") as requests:
        for index in range(arguments.count):
            request = {
                "id": f"synthetic-{arguments.modality}-{index:06d}",
                "arrival_time_ms": round(index * interval_ms, 3),
                "inputs": [
                    {"type": arguments.modality, "synthetic": spec},
                    {"type": "text", "text": arguments.prompt},
                ],
                "outputs": [{"type": "text", "max_tokens": arguments.max_tokens}],
            }
            requests.write(json.dumps(request, separators=(",", ":")) + "\n")

    per_request = _generated_bytes(arguments)
    manifest = {
        "schema_version": 1,
        "benchmark": "synthetic-multimodal",
        "dataset": {
            "name": "synthetic",
            # No source corpus and no per-asset digests: content is a function
            # of the shape and seed recorded here, which is its whole provenance.
            "content_seed": arguments.content_seed,
            "content_is_unique_per_request": arguments.content_seed is None,
        },
        "request": {
            "format": "multimodal-independent-v1",
            "modality": arguments.modality,
            "synthetic": spec,
            "arrival_rate_per_s": arguments.arrival_rate,
            "count": arguments.count,
            "max_tokens": arguments.max_tokens,
            "prompt": arguments.prompt,
        },
        "generated": {
            "bytes_per_request": per_request,
            "bytes_per_second": round(per_request * arguments.arrival_rate),
        },
        "artifacts": {
            "requests": requests_path.name,
            "requests_sha256": _sha256(requests_path),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"wrote {arguments.count} requests to {requests_path} "
        f"({per_request} bytes of {arguments.modality} per request, "
        f"{round(per_request * arguments.arrival_rate)} bytes/s at "
        f"{arguments.arrival_rate}/s)"
    )
    return 0
