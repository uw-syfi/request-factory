from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from benchmarks.__main__ import main as benchmark_main
from benchmarks.seed_tts import (
    MSTAR_QWEN_SYSTEM_PROMPT,
    SeedTTSError,
    SeedTTSOptions,
    acquire_seed_tts,
    load_seed_tts,
    materialize_seed_tts,
)


def _fixture(root: Path) -> Path:
    dataset = root / "seedtts_testset"
    for locale in ("en", "zh"):
        (dataset / locale / "prompt-wavs").mkdir(parents=True)
    rows = []
    for index in range(5):
        audio = dataset / "en" / "prompt-wavs" / f"speaker-{index}.wav"
        audio.write_bytes(b"RIFF" + bytes([index]) * 32)
        rows.append(
            f"utt-{index}|Reference {index}|prompt-wavs/speaker-{index}.wav|Target {index}"
        )
    (dataset / "en" / "meta.lst").write_text("\n".join(rows) + "\n")
    (dataset / "zh" / "meta.lst").write_text(
        "zh-0|参考文本|prompt-wavs/zh.wav|需要合成的文本\n"
    )
    (dataset / "zh" / "prompt-wavs" / "zh.wav").write_bytes(b"RIFFzh")
    return dataset


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_mstar_mode_is_deterministic_text_to_audio_and_preserves_voice_metadata(
    tmp_path: Path,
) -> None:
    dataset = _fixture(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    common = {
        "dataset_dir": dataset,
        "evaluation_set": "en",
        "limit": 3,
        "selection": "shuffled",
        "seed": 17,
        "arrival_rate": 2.0,
        "sample_rate_hz": 24_000,
    }
    assert materialize_seed_tts(SeedTTSOptions(output_dir=first, **common)) == 0
    assert materialize_seed_tts(SeedTTSOptions(output_dir=second, **common)) == 0

    first_requests = _jsonl(first / "requests.jsonl")
    second_requests = _jsonl(second / "requests.jsonl")
    assert [request["id"] for request in first_requests] == [
        request["id"] for request in second_requests
    ]
    assert [request["arrival_time_ms"] for request in first_requests] == [0.0, 500.0, 1000.0]
    assert all(len(request["inputs"]) == 2 for request in first_requests)
    assert all(
        request["inputs"][0]
        == {"type": "system", "text": MSTAR_QWEN_SYSTEM_PROMPT}
        for request in first_requests
    )
    assert all(request["inputs"][1]["type"] == "text" for request in first_requests)
    assert all(
        request["outputs"]
        == [{"type": "audio", "sample_rate_hz": 24_000, "max_tokens": 256}]
        for request in first_requests
    )

    labels = _jsonl(first / "labels.jsonl")
    assert all(label["voice"]["reference_text"].startswith("Reference") for label in labels)
    assert all(label["voice"]["reference_audio"]["sha256"] for label in labels)
    assert all(
        (first / label["voice"]["reference_audio"]["path"]).resolve().is_file()
        for label in labels
    )
    manifest = json.loads((first / "manifest.json").read_text())
    second_manifest = json.loads((second / "manifest.json").read_text())
    assert manifest["request"]["conditioning"] == "target_text_only_mstar_compatible"
    assert manifest["selection"] == second_manifest["selection"]


def test_default_selection_matches_mstar_metadata_order(tmp_path: Path) -> None:
    dataset = _fixture(tmp_path)
    output = tmp_path / "output"
    materialize_seed_tts(
        SeedTTSOptions(dataset_dir=dataset, output_dir=output, limit=2, seed=99)
    )
    requests = _jsonl(output / "requests.jsonl")
    assert [request["id"] for request in requests] == [
        "seed-tts-en-utt-0",
        "seed-tts-en-utt-1",
    ]
    selection = json.loads((output / "manifest.json").read_text())["selection"]
    assert selection["strategy"] == "mstar-order"
    assert selection["seed"] is None


def test_materialization_records_an_adjacent_official_archive_hash(tmp_path: Path) -> None:
    dataset = _fixture(tmp_path)
    archive = dataset.parent / "seedtts_testset.tar"
    archive.write_bytes(b"official archive fixture")
    output = tmp_path / "output"
    materialize_seed_tts(
        SeedTTSOptions(dataset_dir=dataset, output_dir=output, limit=1)
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["dataset"]["archive_sha256"] == hashlib.sha256(
        archive.read_bytes()
    ).hexdigest()


def test_zero_shot_mode_orders_reference_text_audio_then_target(tmp_path: Path) -> None:
    dataset = _fixture(tmp_path)
    output = tmp_path / "output"
    materialize_seed_tts(
        SeedTTSOptions(
            dataset_dir=dataset,
            output_dir=output,
            limit=1,
            include_reference_audio=True,
        )
    )
    request = _jsonl(output / "requests.jsonl")[0]
    assert [part["type"] for part in request["inputs"]] == [
        "system",
        "text",
        "audio",
        "text",
    ]
    asset = request["inputs"][2]["asset"]
    assert asset["media_type"] == "audio/wav"
    assert (output / asset["path"]).resolve().is_file()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["request"]["conditioning"] == "reference_transcript_and_audio"


def test_voice_and_generation_limit_are_materialized(tmp_path: Path) -> None:
    dataset = _fixture(tmp_path)
    output = tmp_path / "output"
    materialize_seed_tts(
        SeedTTSOptions(
            dataset_dir=dataset,
            output_dir=output,
            limit=1,
            max_tokens=384,
            voice="Ethan",
        )
    )
    request = _jsonl(output / "requests.jsonl")[0]
    assert request["outputs"] == [
        {
            "type": "audio",
            "sample_rate_hz": 24_000,
            "max_tokens": 384,
            "voice": "Ethan",
        }
    ]
    request_manifest = json.loads((output / "manifest.json").read_text())["request"]
    assert request_manifest["max_tokens"] == 384
    assert request_manifest["voice"] == "Ethan"
    assert request_manifest["system_prompt"] == MSTAR_QWEN_SYSTEM_PROMPT


def test_shared_benchmark_cli_materializes_seed_tts(tmp_path: Path) -> None:
    dataset = _fixture(tmp_path)
    output = tmp_path / "output"
    assert benchmark_main(
        [
            "seed-tts",
            "--dataset-dir",
            str(dataset),
            "--output-dir",
            str(output),
            "--limit",
            "1",
            "--voice",
            "Ethan",
            "--max-tokens",
            "384",
        ]
    ) == 0
    request = _jsonl(output / "requests.jsonl")[0]
    assert request["outputs"][0]["voice"] == "Ethan"
    assert request["outputs"][0]["max_tokens"] == 384


def test_loader_rejects_escape_duplicate_and_malformed_metadata(tmp_path: Path) -> None:
    dataset = _fixture(tmp_path)
    meta = dataset / "en" / "meta.lst"
    meta.write_text("bad|prompt|../../outside.wav|target\n")
    with pytest.raises(SeedTTSError, match="escapes"):
        load_seed_tts(dataset)

    meta.write_text(
        "same|prompt|prompt-wavs/speaker-0.wav|target\n"
        "same|prompt|prompt-wavs/speaker-1.wav|target\n"
    )
    with pytest.raises(SeedTTSError, match="duplicate"):
        load_seed_tts(dataset)

    meta.write_text("too|few|columns\n")
    with pytest.raises(SeedTTSError, match="expected 4 or 5"):
        load_seed_tts(dataset)


def test_acquisition_safely_extracts_official_layout_and_records_archive_hash(
    tmp_path: Path,
) -> None:
    source = _fixture(tmp_path / "source")
    destination = tmp_path / "cache" / "seedtts_testset"

    def make_archive(_file_id: str, output: Path) -> None:
        with tarfile.open(output, "w") as archive:
            archive.add(source, arcname="seedtts_testset")

    extracted, archive_hash = acquire_seed_tts(destination, downloader=make_archive)
    assert extracted == destination.resolve()
    assert archive_hash is not None and len(archive_hash) == 64
    assert len(load_seed_tts(extracted)[1]) == 5


def test_acquisition_rejects_links(tmp_path: Path) -> None:
    destination = tmp_path / "cache" / "seedtts_testset"

    def make_archive(_file_id: str, output: Path) -> None:
        with tarfile.open(output, "w") as archive:
            link = tarfile.TarInfo("seedtts_testset/en/meta.lst")
            link.type = tarfile.SYMTYPE
            link.linkname = "/etc/passwd"
            archive.addfile(link, io.BytesIO())

    with pytest.raises(SeedTTSError, match="unsupported link/device"):
        acquire_seed_tts(destination, downloader=make_archive)
