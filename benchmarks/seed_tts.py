"""Materialize the official Seed-TTS evaluation set for serving replay.

M* uses Seed-TTS as a text distribution for text-to-speech serving and drops
the reference voice conditioning carried by the original benchmark.  This
module makes that behavior the default while retaining the reference metadata
in ``labels.jsonl``.  Full zero-shot conditioning can be requested explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tarfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

SEED_TTS_GDRIVE_FILE_ID = "1GlSjVfSHkW3-leKKBlfrjuuTGqQ_xaLP"
SEED_TTS_SOURCE_URL = (
    "https://drive.google.com/file/d/"
    f"{SEED_TTS_GDRIVE_FILE_ID}/edit"
)
SEED_TTS_REPOSITORY_URL = "https://github.com/BytedanceSpeech/seed-tts-eval"
DEFAULT_SAMPLE_RATE_HZ = 24_000
DEFAULT_MAX_TOKENS = 256
MSTAR_QWEN_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)

_SETS = {
    "en": Path("en/meta.lst"),
    "zh": Path("zh/meta.lst"),
    "zh-hard": Path("zh/hardcase.lst"),
    "en-cross": Path("en/non_para_reconstruct_meta.lst"),
    "zh-cross": Path("zh/non_para_reconstruct_meta.lst"),
}


class SeedTTSError(ValueError):
    """A user-actionable dataset or materialization error."""


@dataclass(frozen=True)
class SeedTTSOptions:
    dataset_dir: Path
    output_dir: Path
    evaluation_set: str = "en"
    limit: int | None = None
    selection: str = "mstar-order"
    seed: int = 0
    arrival_rate: float = 1.0
    sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ
    max_tokens: int = DEFAULT_MAX_TOKENS
    voice: str | None = None
    include_reference_audio: bool = False
    download: bool = False


@dataclass(frozen=True)
class SeedTTSRecord:
    utterance_id: str
    prompt_text: str
    prompt_audio: Path
    target_text: str
    target_audio: Path | None


def add_seed_tts_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Register ``seed-tts`` on the benchmark CLI's subparser collection."""

    parser = subparsers.add_parser(
        "seed-tts", help="materialize Seed-TTS text-to-audio requests"
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--set", choices=tuple(_SETS), default="en")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--selection",
        choices=("mstar-order", "shuffled"),
        default="mstar-order",
        help="select metadata rows in M* order or by a deterministic shuffle",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arrival-rate", type=float, default=1.0)
    parser.add_argument("--sample-rate-hz", type=int, default=DEFAULT_SAMPLE_RATE_HZ)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="maximum generated audio tokens (M* uses 256)",
    )
    parser.add_argument(
        "--voice",
        help="optional Qwen speaker or OpenAI-compatible speech voice",
    )
    parser.add_argument(
        "--include-reference-audio",
        action="store_true",
        help="emit reference transcript/audio inputs for zero-shot voice cloning",
    )
    return parser


def options_from_namespace(arguments: argparse.Namespace) -> SeedTTSOptions:
    """Translate the shared CLI namespace into this module's typed options."""

    return SeedTTSOptions(
        dataset_dir=arguments.dataset_dir,
        output_dir=arguments.output_dir,
        evaluation_set=arguments.set,
        limit=arguments.limit,
        selection=arguments.selection,
        seed=arguments.seed,
        arrival_rate=arguments.arrival_rate,
        sample_rate_hz=arguments.sample_rate_hz,
        max_tokens=arguments.max_tokens,
        voice=arguments.voice,
        include_reference_audio=arguments.include_reference_audio,
        download=arguments.download,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with tarfile.open(archive, "r:*") as reader:
        for member in reader:
            if not (member.isfile() or member.isdir()):
                raise SeedTTSError(
                    f"archive contains unsupported link/device entry: {member.name!r}"
                )
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise SeedTTSError(f"archive member escapes destination: {member.name!r}")
            reader.extract(member, destination, filter="data")


def _default_gdown(file_id: str, output: Path) -> None:
    try:
        import gdown  # type: ignore[import-not-found]
    except ImportError as error:
        raise SeedTTSError(
            "Seed-TTS download requires gdown; run with `uv run --with gdown`, "
            "or download the official archive and pass its extracted directory"
        ) from error
    result = gdown.download(id=file_id, output=str(output), quiet=False)
    if result is None:
        raise SeedTTSError("gdown did not produce the Seed-TTS archive")


def acquire_seed_tts(
    dataset_dir: Path,
    *,
    downloader: Callable[[str, Path], None] | None = None,
) -> tuple[Path, str | None]:
    """Download and safely extract the official archive when necessary.

    Returns the resolved extracted root and the archive SHA-256 when the
    archive is available.  ``downloader`` is injectable for offline tests.
    """

    dataset_dir = dataset_dir.expanduser().resolve()
    if (dataset_dir / "en/meta.lst").is_file():
        archive = dataset_dir.parent / "seedtts_testset.tar"
        return dataset_dir, _sha256(archive) if archive.is_file() else None
    if dataset_dir.exists():
        raise SeedTTSError(
            f"incomplete Seed-TTS destination already exists: {dataset_dir}"
        )

    parent = dataset_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    archive = parent / "seedtts_testset.tar"
    partial = archive.with_suffix(".tar.partial")
    if not archive.is_file():
        if partial.exists():
            partial.unlink()
        (downloader or _default_gdown)(SEED_TTS_GDRIVE_FILE_ID, partial)
        if not partial.is_file() or not tarfile.is_tarfile(partial):
            raise SeedTTSError(
                f"downloaded Seed-TTS file is not a readable tar archive: {partial}"
            )
        partial.replace(archive)

    archive_hash = _sha256(archive)
    staging = parent / ".seedtts-extracting"
    if staging.exists():
        raise SeedTTSError(
            f"stale Seed-TTS extraction directory exists: {staging}; remove it and retry"
        )
    staging.mkdir()
    try:
        _safe_extract(archive, staging)
        candidates = [
            path.parent.parent
            for path in staging.rglob("en/meta.lst")
            if path.is_file()
        ]
        if len(candidates) != 1:
            raise SeedTTSError(
                "official archive must contain exactly one dataset root with en/meta.lst"
            )
        extracted = candidates[0]
        if extracted == staging:
            dataset_dir.mkdir()
            for child in tuple(staging.iterdir()):
                if child != dataset_dir:
                    shutil.move(str(child), dataset_dir / child.name)
        else:
            extracted.rename(dataset_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    if not (dataset_dir / "en/meta.lst").is_file():
        raise SeedTTSError(f"extracted Seed-TTS tree is incomplete: {dataset_dir}")
    return dataset_dir, archive_hash


def _resolve_dataset_path(meta_path: Path, raw_path: str, at: str) -> Path:
    relative = Path(raw_path)
    if relative.is_absolute():
        raise SeedTTSError(f"{at}: audio path must be relative: {raw_path!r}")
    language_root = meta_path.parent.resolve()
    resolved = (language_root / relative).resolve()
    if language_root not in resolved.parents:
        raise SeedTTSError(f"{at}: audio path escapes the language directory")
    if not resolved.is_file():
        raise SeedTTSError(f"{at}: audio file is missing: {resolved}")
    return resolved


def load_seed_tts(dataset_dir: Path, evaluation_set: str = "en") -> tuple[Path, list[SeedTTSRecord]]:
    """Parse one official pipe-separated evaluation metadata file."""

    if evaluation_set not in _SETS:
        raise SeedTTSError(
            f"unknown Seed-TTS set {evaluation_set!r}; expected one of {tuple(_SETS)}"
        )
    dataset_dir = dataset_dir.expanduser().resolve()
    meta_path = dataset_dir / _SETS[evaluation_set]
    if not meta_path.is_file():
        raise SeedTTSError(f"Seed-TTS metadata not found: {meta_path}")

    records: list[SeedTTSRecord] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(meta_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        columns = [column.strip() for column in line.split("|")]
        if len(columns) not in (4, 5):
            raise SeedTTSError(
                f"{meta_path}:{line_number}: expected 4 or 5 pipe-separated columns"
            )
        utterance_id, prompt_text, prompt_audio_raw, target_text = columns[:4]
        if not all((utterance_id, prompt_text, prompt_audio_raw, target_text)):
            raise SeedTTSError(f"{meta_path}:{line_number}: required column is empty")
        if utterance_id in seen_ids:
            raise SeedTTSError(
                f"{meta_path}:{line_number}: duplicate utterance id {utterance_id!r}"
            )
        seen_ids.add(utterance_id)
        at = f"{meta_path}:{line_number}"
        prompt_audio = _resolve_dataset_path(meta_path, prompt_audio_raw, at)
        target_audio = None
        if len(columns) == 5 and columns[4]:
            target_audio = _resolve_dataset_path(meta_path, columns[4], at)
        records.append(
            SeedTTSRecord(
                utterance_id=utterance_id,
                prompt_text=prompt_text,
                prompt_audio=prompt_audio,
                target_text=target_text,
                target_audio=target_audio,
            )
        )
    if not records:
        raise SeedTTSError(f"Seed-TTS metadata contains no usable rows: {meta_path}")
    return meta_path, records


def _asset(path: Path, output_dir: Path, media_type: str = "audio/wav") -> dict:
    return {
        "path": Path(os.path.relpath(path, output_dir)).as_posix(),
        "sha256": _sha256(path),
        "media_type": media_type,
    }


def materialize_seed_tts(options: SeedTTSOptions) -> int:
    """Write requests, labels, and a provenance manifest."""

    if options.limit is not None and options.limit <= 0:
        raise SeedTTSError("limit must be greater than zero")
    if options.selection not in ("mstar-order", "shuffled"):
        raise SeedTTSError("selection must be 'mstar-order' or 'shuffled'")
    if not math.isfinite(options.arrival_rate) or options.arrival_rate <= 0:
        raise SeedTTSError("arrival rate must be greater than zero")
    if options.sample_rate_hz <= 0:
        raise SeedTTSError("sample rate must be greater than zero")
    if options.max_tokens <= 0:
        raise SeedTTSError("max tokens must be greater than zero")
    if options.voice is not None and not options.voice.strip():
        raise SeedTTSError("voice must not be empty")

    dataset_dir = options.dataset_dir.expanduser().resolve()
    archive_hash = None
    if options.download:
        dataset_dir, archive_hash = acquire_seed_tts(dataset_dir)
    else:
        adjacent_archive = dataset_dir.parent / "seedtts_testset.tar"
        if adjacent_archive.is_file():
            archive_hash = _sha256(adjacent_archive)
    meta_path, records = load_seed_tts(dataset_dir, options.evaluation_set)
    if options.selection == "shuffled":
        random.Random(options.seed).shuffle(records)
    if options.limit is not None:
        records = records[: options.limit]

    output_dir = options.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    requests_path = output_dir / "requests.jsonl"
    labels_path = output_dir / "labels.jsonl"
    interval_ms = 1000.0 / options.arrival_rate
    selected_provenance: list[dict] = []
    with requests_path.open("w", encoding="utf-8") as requests, labels_path.open(
        "w", encoding="utf-8"
    ) as labels:
        for index, record in enumerate(records):
            prompt_asset = _asset(record.prompt_audio, output_dir)
            request_id = (
                f"seed-tts-{options.evaluation_set}-{record.utterance_id}"
            )
            inputs: list[dict] = [
                {"type": "system", "text": MSTAR_QWEN_SYSTEM_PROMPT}
            ]
            if options.include_reference_audio:
                inputs.extend(
                    [
                        {"type": "text", "text": record.prompt_text},
                        {"type": "audio", "asset": prompt_asset},
                    ]
                )
            inputs.append({"type": "text", "text": record.target_text})
            request = {
                "id": request_id,
                "arrival_time_ms": round(index * interval_ms, 6),
                "inputs": inputs,
                "outputs": [{
                    "type": "audio",
                    "sample_rate_hz": options.sample_rate_hz,
                    "max_tokens": options.max_tokens,
                    **({"voice": options.voice} if options.voice is not None else {}),
                }],
            }
            label = {
                "id": request_id,
                "utterance_id": record.utterance_id,
                "evaluation_set": options.evaluation_set,
                "target_text": record.target_text,
                "voice": {
                    "reference_text": record.prompt_text,
                    "reference_audio": prompt_asset,
                },
                "target_audio": (
                    _asset(record.target_audio, output_dir)
                    if record.target_audio is not None
                    else None
                ),
            }
            requests.write(json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n")
            labels.write(json.dumps(label, ensure_ascii=False, separators=(",", ":")) + "\n")
            selected_provenance.append(
                {
                    "utterance_id": record.utterance_id,
                    "prompt_audio_sha256": prompt_asset["sha256"],
                    "target_audio_sha256": (
                        label["target_audio"]["sha256"]
                        if label["target_audio"] is not None
                        else None
                    ),
                }
            )

    selection_hash = hashlib.sha256(
        json.dumps(
            selected_provenance,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "benchmark": "seed-tts-text-to-audio",
        "dataset": {
            "name": "Seed-TTS Eval",
            "repository_url": SEED_TTS_REPOSITORY_URL,
            "source_url": SEED_TTS_SOURCE_URL,
            "archive_sha256": archive_hash,
            "evaluation_set": options.evaluation_set,
            "metadata_sha256": _sha256(meta_path),
        },
        "selection": {
            "strategy": options.selection,
            "seed": options.seed if options.selection == "shuffled" else None,
            "limit": options.limit,
            "selected_examples": len(records),
            "selected_records_sha256": selection_hash,
        },
        "request": {
            "format": "multimodal-independent-v1",
            "arrival_rate_per_s": options.arrival_rate,
            "sample_rate_hz": options.sample_rate_hz,
            "max_tokens": options.max_tokens,
            "voice": options.voice,
            "system_prompt": MSTAR_QWEN_SYSTEM_PROMPT,
            "conditioning": (
                "reference_transcript_and_audio"
                if options.include_reference_audio
                else "target_text_only_mstar_compatible"
            ),
        },
        "artifacts": {
            "requests": requests_path.name,
            "requests_sha256": _sha256(requests_path),
            "labels": labels_path.name,
            "labels_sha256": _sha256(labels_path),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(records)} requests to {requests_path}", file=sys.stdout)
    return 0
