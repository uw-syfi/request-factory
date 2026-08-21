"""Materialize supported benchmark datasets into canonical request artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

from benchmarks import seed_tts, synthetic, vbench

FOOD101_URL = "https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
FOOD101_ARCHIVE_BYTES = 4_996_278_331


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m benchmarks")
    subparsers = parser.add_subparsers(dest="benchmark", required=True)
    food101 = subparsers.add_parser(
        "food101", help="materialize Food101 image-to-text requests for BAGEL"
    )
    food101.add_argument("--dataset-dir", type=Path, required=True)
    food101.add_argument("--output-dir", type=Path, required=True)
    food101.add_argument("--download", action="store_true")
    food101.add_argument("--split", choices=("train", "test"), default="test")
    food101.add_argument("--limit", type=int)
    food101.add_argument("--seed", type=int, default=0)
    food101.add_argument("--arrival-rate", type=float, default=1.0)
    food101.add_argument("--max-tokens", type=int, default=64)
    food101.add_argument(
        "--prompt",
        default="What food is shown in this image? Answer with the dish name only.",
    )
    vbench.add_parser(subparsers)
    seed_tts.add_seed_tts_parser(subparsers)
    synthetic.add_parser(subparsers)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with tarfile.open(archive, "r|gz") as reader:
        for member in reader:
            if not (member.isfile() or member.isdir()):
                raise ValueError(
                    f"archive contains unsupported link/device entry: {member.name!r}"
                )
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"archive member escapes destination: {member.name!r}")
            reader.extract(member, destination, filter="data")


def _food101_tree_is_complete(dataset_dir: Path) -> bool:
    identifiers: list[str] = []
    for split in ("train", "test"):
        metadata = dataset_dir / "meta" / f"{split}.txt"
        if not metadata.is_file():
            return False
        identifiers.extend(
            line.strip() for line in metadata.read_text().splitlines() if line.strip()
        )
    return len(identifiers) == 101_000 and all(
        (dataset_dir / "images" / f"{identifier}.jpg").is_file()
        for identifier in identifiers
    )


def _download_food101(dataset_dir: Path) -> tuple[Path, str | None]:
    if _food101_tree_is_complete(dataset_dir):
        archive = dataset_dir.parent / "food-101.tar.gz"
        return dataset_dir, _sha256(archive) if archive.is_file() else None
    parent = dataset_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    archive = parent / "food-101.tar.gz"
    partial = archive.with_suffix(archive.suffix + ".partial")
    if archive.is_file() and archive.stat().st_size != FOOD101_ARCHIVE_BYTES:
        if partial.exists():
            raise ValueError(f"both incomplete archive files exist: {archive} and {partial}")
        archive.replace(partial)
    if not archive.is_file():
        offset = partial.stat().st_size if partial.is_file() else 0
        if offset > FOOD101_ARCHIVE_BYTES:
            raise ValueError(
                f"partial archive is larger than expected: {offset} > {FOOD101_ARCHIVE_BYTES}"
            )
        request = urllib.request.Request(FOOD101_URL)
        if offset:
            request.add_header("Range", f"bytes={offset}-")
            print(
                f"resuming {FOOD101_URL} at byte {offset} -> {archive}",
                file=sys.stderr,
            )
        else:
            print(f"downloading {FOOD101_URL} -> {archive}", file=sys.stderr)
        response = urllib.request.urlopen(request)
        append = offset > 0 and response.status == 206
        if offset and not append:
            offset = 0
        with response, partial.open("ab" if append else "wb") as writer:
            shutil.copyfileobj(response, writer, length=1024 * 1024)
        downloaded = partial.stat().st_size
        if downloaded != FOOD101_ARCHIVE_BYTES:
            raise ValueError(
                f"incomplete Food101 archive: expected {FOOD101_ARCHIVE_BYTES} bytes, got {downloaded}; rerun to resume"
            )
        partial.replace(archive)
    archive_hash = _sha256(archive)
    print(f"extracting {archive}", file=sys.stderr)
    staging = parent / ".food-101-extracting"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    _safe_extract(archive, staging)
    extracted = staging / "food-101"
    if extracted != dataset_dir:
        if dataset_dir.exists():
            raise ValueError(
                f"incomplete dataset destination already exists: {dataset_dir}; remove it before retrying extraction"
            )
        extracted.rename(dataset_dir)
    staging.rmdir()
    if not _food101_tree_is_complete(dataset_dir):
        raise ValueError(f"extracted Food101 tree is incomplete: {dataset_dir}")
    return dataset_dir, archive_hash


def _materialize_food101(arguments: argparse.Namespace) -> int:
    if arguments.limit is not None and arguments.limit <= 0:
        raise ValueError("--limit must be greater than zero")
    if arguments.arrival_rate <= 0:
        raise ValueError("--arrival-rate must be greater than zero")
    if arguments.max_tokens <= 0:
        raise ValueError("--max-tokens must be greater than zero")

    dataset_dir = arguments.dataset_dir.expanduser().resolve()
    archive_hash = None
    if arguments.download:
        dataset_dir, archive_hash = _download_food101(dataset_dir)
    split_file = dataset_dir / "meta" / f"{arguments.split}.txt"
    if not split_file.is_file():
        raise ValueError(
            f"Food101 split metadata not found at {split_file}; pass --download or the extracted food-101 directory"
        )
    identifiers = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    random.Random(arguments.seed).shuffle(identifiers)
    if arguments.limit is not None:
        identifiers = identifiers[: arguments.limit]

    output_dir = arguments.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    requests_path = output_dir / "requests.jsonl"
    labels_path = output_dir / "labels.jsonl"
    interval_ms = 1000.0 / arguments.arrival_rate
    asset_hashes: list[str] = []
    with requests_path.open("w") as requests, labels_path.open("w") as labels:
        for index, identifier in enumerate(identifiers):
            image = dataset_dir / "images" / f"{identifier}.jpg"
            if not image.is_file():
                raise ValueError(f"Food101 image missing: {image}")
            digest = _sha256(image)
            asset_hashes.append(digest)
            relative_image = Path(os.path.relpath(image, output_dir))
            request = {
                "id": f"food101-{arguments.split}-{identifier.replace('/', '-')}",
                "arrival_time_ms": round(index * interval_ms),
                "inputs": [
                    {
                        "type": "image",
                        "asset": {
                            "path": relative_image.as_posix(),
                            "sha256": digest,
                            "media_type": "image/jpeg",
                        },
                    },
                    {"type": "text", "text": arguments.prompt},
                ],
                "outputs": [{"type": "text", "max_tokens": arguments.max_tokens}],
            }
            requests.write(json.dumps(request, separators=(",", ":")) + "\n")
            label = identifier.split("/", 1)[0].replace("_", " ")
            labels.write(json.dumps({"id": request["id"], "label": label}) + "\n")

    manifest = {
        "schema_version": 1,
        "benchmark": "bagel-image-to-text-food101",
        "dataset": {
            "name": "Food-101",
            "source_url": FOOD101_URL,
            "archive_sha256": archive_hash,
            "split": arguments.split,
            "split_metadata_sha256": _sha256(split_file),
        },
        "selection": {
            "seed": arguments.seed,
            "limit": arguments.limit,
            "selected_examples": len(identifiers),
        },
        "request": {
            "format": "multimodal-independent-v1",
            "arrival_rate_per_s": arguments.arrival_rate,
            "max_tokens": arguments.max_tokens,
            "prompt": arguments.prompt,
        },
        "artifacts": {
            "requests": requests_path.name,
            "requests_sha256": _sha256(requests_path),
            "labels": labels_path.name,
            "labels_sha256": _sha256(labels_path),
            "selected_assets_sha256": hashlib.sha256(
                "\n".join(asset_hashes).encode()
            ).hexdigest(),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {len(identifiers)} requests to {requests_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.benchmark == "food101":
            return _materialize_food101(arguments)
        if arguments.benchmark == "vbench":
            return vbench.materialize_from_arguments(arguments)
        if arguments.benchmark == "seed-tts":
            return seed_tts.materialize_seed_tts(
                seed_tts.options_from_namespace(arguments)
            )
        if arguments.benchmark == "synthetic":
            return synthetic.materialize(arguments)
    except (
        EOFError,
        OSError,
        ValueError,
        subprocess.CalledProcessError,
        tarfile.TarError,
    ) as error:
        print(f"benchmark error: {error}", file=sys.stderr)
        return 2
    raise AssertionError(arguments.benchmark)


if __name__ == "__main__":
    raise SystemExit(main())
