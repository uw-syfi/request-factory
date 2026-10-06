from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from benchmarks.__main__ import main as benchmark_main
from launcher.config import load_task_config

REPO_ROOT = Path(__file__).resolve().parents[1]

PREFLIGHT_ID = "req-frontend-preflight"


def workload_rows(log: Path) -> list[dict]:
    """Server-side rows for the measured workload only.

    A multimodal run is preceded by preflight probes -- one with media and one
    without -- which the server logs like any other request. They are not the
    workload and would otherwise be read as its first rows.
    """
    return [
        row
        for row in (json.loads(line) for line in log.read_text().splitlines())
        if row.get("request_id") != PREFLIGHT_ID
    ]



def _food101_fixture(root: Path, count: int = 8) -> Path:
    dataset = root / "food-101"
    (dataset / "meta").mkdir(parents=True)
    (dataset / "images" / "pizza").mkdir(parents=True)
    identifiers = []
    for index in range(count):
        identifier = f"pizza/{index:04d}"
        identifiers.append(identifier)
        (dataset / "images" / f"{identifier}.jpg").write_bytes(
            b"\xff\xd8\xff\xe0" + bytes([index]) * 128 + b"\xff\xd9"
        )
    (dataset / "meta" / "test.txt").write_text("\n".join(identifiers) + "\n")
    return dataset


def test_food101_materializer_is_deterministic_and_launcher_needs_no_corpus(
    tmp_path: Path,
) -> None:
    dataset = _food101_fixture(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    common = [
        "food101",
        "--dataset-dir",
        str(dataset),
        "--split",
        "test",
        "--limit",
        "4",
        "--seed",
        "17",
    ]
    assert benchmark_main([*common, "--output-dir", str(first)]) == 0
    assert benchmark_main([*common, "--output-dir", str(second)]) == 0
    first_requests = (first / "requests.jsonl").read_text()
    second_requests = (second / "requests.jsonl").read_text()
    # Relative asset roots differ, but selection and content hashes do not.
    assert [json.loads(line)["id"] for line in first_requests.splitlines()] == [
        json.loads(line)["id"] for line in second_requests.splitlines()
    ]
    assert json.loads((first / "manifest.json").read_text())["selection"] == {
        "seed": 17,
        "limit": 4,
        "selected_examples": 4,
    }
    for line in first_requests.splitlines():
        request = json.loads(line)
        asset = request["inputs"][0]["asset"]
        image = (first / asset["path"]).resolve()
        assert hashlib.sha256(image.read_bytes()).hexdigest() == asset["sha256"]

    config = first / "run.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "input": {
                    "trace": "requests.jsonl",
                    "format": "multimodal-independent-v1",
                },
                "server": {
                    "backend": "openai-chat",
                "dialect": "vllm-omni",
                    "model": "bagel",
                },
                "output": {"directory": "run"},
            }
        )
    )
    specification = load_task_config("run", config)
    assert "--text-file" not in specification.arguments
    assert "--tokenizer" not in specification.arguments


def test_cpu_mock_receives_assets_and_streams_concurrent_replay(tmp_path: Path) -> None:
    dataset = _food101_fixture(tmp_path)
    artifact = tmp_path / "artifact"
    assert (
        benchmark_main(
            [
                "food101",
                "--dataset-dir",
                str(dataset),
                "--output-dir",
                str(artifact),
                "--limit",
                "8",
                "--arrival-rate",
                "1000",
                "--max-tokens",
                "4",
            ]
        )
        == 0
    )

    subprocess.run(
        ["cargo", "build", "--bin", "session_runner"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    ready = tmp_path / "ready"
    server_log = tmp_path / "server.jsonl"
    server = subprocess.Popen(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "mock_multimodal_server.py"),
            "--dialect",
            "vllm-omni",
            "--ready-file",
            str(ready),
            "--log-path",
            str(server_log),
            "--chunk-delay-ms",
            "2",
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists(), server.stderr.read() if server.poll() is not None else ""
        port = int(ready.read_text())
        summary = tmp_path / "summary.json"
        completed = subprocess.run(
            [
                str(REPO_ROOT / "target" / "debug" / "session_runner"),
                "--trace",
                str(artifact / "requests.jsonl"),
                "--input-file-format",
                "multimodal-independent-v1",
                "--base-url",
                f"http://127.0.0.1:{port}/v1",
                "--backend",
                "openai-chat",
                "--dialect",
                "vllm-omni",
                "--model",
                "bagel",
                "--arrival-mode",
                "saturated",
                "--max-concurrency",
                "4",
                "--timeline",
                "false",
                "--log-path",
                str(tmp_path / "requests-log.jsonl"),
                "--summary-path",
                str(summary),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        result = json.loads(summary.read_text())
        assert result["replay"]["kind"] == "multimodal_requests"
        assert result["replay"]["common"]["success_steps"] == 8
        records = workload_rows(server_log)
        assert len(records) == 8
        assert all(record["media_parts"] == 1 for record in records)
        assert all(record["media_bytes"] > 100 for record in records)
        assert all(record["max_tokens"] == 4 for record in records)
        assert max(record["active_at_receive"] for record in records) >= 2
    finally:
        server.terminate()
        server.wait(timeout=5)


def test_cpu_mock_validates_generated_image_and_streaming_audio_outputs(
    tmp_path: Path,
) -> None:
    subprocess.run(
        ["cargo", "build", "--bin", "session_runner"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    ready = tmp_path / "ready-media"
    server_log = tmp_path / "server-media.jsonl"
    server = subprocess.Popen(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "mock_multimodal_server.py"),
            "--dialect",
            "vllm-omni",
            "--ready-file",
            str(ready),
            "--log-path",
            str(server_log),
            "--chunk-delay-ms",
            "1",
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists(), server.stderr.read() if server.poll() is not None else ""
        port = int(ready.read_text())
        image = tmp_path / "source.png"
        image.write_bytes(base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
        ))
        image_hash = hashlib.sha256(image.read_bytes()).hexdigest()
        cases = [
            (
                "t2i",
                "openai-images",
                [{"type": "text", "text": "a red cube"}],
                [{"type": "image", "width": 64, "height": 64, "steps": 2, "count": 1}],
            ),
            (
                "i2i",
                "openai-chat",
                [
                    {"type": "text", "text": "make it blue"},
                    {"type": "image", "asset": {"path": str(image), "sha256": image_hash, "media_type": "image/png"}},
                ],
                [{"type": "image", "width": 64, "height": 64, "steps": 2, "count": 1}],
            ),
            (
                "qwen-audio",
                "openai-chat",
                [
                    {"type": "system", "text": "You are a speaking assistant."},
                    {"type": "text", "text": "say hello"},
                ],
                [
                    {
                        "type": "audio",
                        "sample_rate_hz": 24000,
                        "max_tokens": 256,
                        "voice": "Ethan",
                        # A vLLM-Omni-only knob, declared for that dialect alone
                        # and nested into its `extra_body` envelope on the wire.
                        "model_params": {"vllm-omni": {"thinker_temperature": 0.0}},
                    }
                ],
            ),
            (
                "speech-audio",
                "openai-speech",
                [{"type": "text", "text": "say hello"}],
                [{"type": "audio", "sample_rate_hz": 24000, "max_tokens": 256, "voice": "tara"}],
            ),
        ]
        for case_name, backend, inputs, outputs in cases:
            trace = tmp_path / f"{case_name}.jsonl"
            trace.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "id": f"{case_name}-{index}",
                            "arrival_time_ms": 0,
                            "inputs": inputs,
                            "outputs": outputs,
                        }
                    )
                    for index in range(2)
                )
                + "\n"
            )
            summary = tmp_path / f"{case_name}-summary.json"
            artifacts = tmp_path / f"{case_name}-artifacts"
            completed = subprocess.run(
                [
                    str(REPO_ROOT / "target" / "debug" / "session_runner"),
                    "--trace", str(trace),
                    "--input-file-format", "multimodal-independent-v1",
                    "--base-url", f"http://127.0.0.1:{port}/v1",
                    "--backend", backend,
                    "--dialect", "vllm-omni",
                    "--model", "mock-model",
                    "--arrival-mode", "saturated",
                    "--max-concurrency", "2",
                    "--timeline", "false",
                    "--output-artifact-dir", str(artifacts),
                    "--log-path", str(tmp_path / f"{case_name}-requests.jsonl"),
                    "--summary-path", str(summary),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            assert completed.returncode == 0, completed.stderr
            result = json.loads(summary.read_text())["replay"]["common"]
            assert result["success_steps"] == 2
            assert result["failed_steps"] == 0
            assert result["output_bytes"] > 0
            assert result["first_output_ms_p50"] is not None
            if "audio" in case_name:
                assert result["real_time_factor_measured_steps"] == 2
                assert result["audio_duration_s"] > 0
            assert len(list(artifacts.iterdir())) == 2
        received = workload_rows(server_log)
        qwen = [row for row in received if row["output_modality"] == "audio" and row["system_prompts"]]
        speech = [row for row in received if row["output_modality"] == "audio" and not row["system_prompts"]]
        assert len(qwen) == 2
        assert all(row["max_tokens"] == 256 for row in qwen)
        assert all(row["temperature"] == 0.0 for row in qwen)
        # Declared per-dialect in the trace and delivered where vLLM-Omni reads
        # it. `max_output_tokens` is deliberately absent: that is M*'s second
        # name for the cap, and sending it here would be another server's knob.
        assert all(row["thinker_temperature"] == 0.0 for row in qwen)
        assert all(row["max_output_tokens"] is None for row in qwen)
        assert len(speech) == 2
        assert all(row["max_tokens"] == 256 for row in speech)
        assert all(row["request_id"] for row in received)
    finally:
        server.terminate()
        server.wait(timeout=5)


def _asset(path: Path, media_type: str) -> dict:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "media_type": media_type,
    }


def _run_surface(
    tmp_path: Path, port: int, name: str, backend: str, dialect: str, inputs, outputs
) -> dict:
    trace = tmp_path / f"{name}.jsonl"
    trace.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"{name}-{index}",
                    "arrival_time_ms": float(index),
                    "inputs": inputs,
                    "outputs": outputs,
                }
            )
            for index in range(2)
        )
        + "\n"
    )
    summary = tmp_path / f"{name}-summary.json"
    completed = subprocess.run(
        [
            str(REPO_ROOT / "target" / "debug" / "session_runner"),
            "--trace", str(trace),
            "--input-file-format", "multimodal-independent-v1",
            "--base-url", f"http://127.0.0.1:{port}/v1",
            "--backend", backend,
            "--dialect", dialect,
            "--model", "mock-model",
            "--arrival-mode", "saturated",
            "--max-concurrency", "2",
            "--timeline", "false",
            "--output-artifact-dir", str(tmp_path / f"{name}-artifacts"),
            "--log-path", str(tmp_path / f"{name}-requests.jsonl"),
            "--summary-path", str(summary),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(summary.read_text())["replay"]["common"]


def _serve(tmp_path: Path, dialect: str, *, sse_no_space: bool = False):
    ready = tmp_path / f"ready-{dialect}"
    log = tmp_path / f"server-{dialect}.jsonl"
    command = [
            sys.executable,
            str(REPO_ROOT / "tools" / "mock_multimodal_server.py"),
            "--dialect", dialect,
            "--ready-file", str(ready),
            "--log-path", str(log),
        ]
    if sse_no_space:
        command.append("--sse-no-space")
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and not ready.exists():
        if process.poll() is not None:
            raise AssertionError(process.stderr.read())
        time.sleep(0.05)
    assert ready.exists(), "mock did not become ready"
    return process, int(ready.read_text()), log


def test_sse_optional_space_is_accepted_for_text_and_audio(tmp_path: Path) -> None:
    subprocess.run(
        ["cargo", "build", "--bin", "session_runner"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    )
    process, port, _ = _serve(tmp_path, "vllm-omni", sse_no_space=True)
    try:
        text = _run_surface(
            tmp_path,
            port,
            "no-space-text",
            "openai-chat",
            "vllm-omni",
            [{"type": "text", "text": "answer briefly"}],
            [{"type": "text", "max_tokens": 2}],
        )
        audio = _run_surface(
            tmp_path,
            port,
            "no-space-audio",
            "openai-chat",
            "vllm-omni",
            [{"type": "text", "text": "say hello"}],
            [{"type": "audio", "sample_rate_hz": 24000, "max_tokens": 16}],
        )
    finally:
        process.terminate()
        process.wait(timeout=10)

    assert text["success_steps"] == 2 and text["failed_steps"] == 0
    assert text["output_bytes"] > 0
    assert audio["success_steps"] == 2 and audio["failed_steps"] == 0
    assert audio["output_bytes"] > 0


@pytest.mark.parametrize(
    "dialect",
    ["openai", "vllm", "vllm-omni", "sglang-omni", "mstar", "dynamo"],
)
def test_every_chat_dialect_replays_each_supported_input_modality(
    tmp_path: Path, dialect: str
) -> None:
    """Exercise the complete serializer/server/parser path for every dialect."""
    subprocess.run(
        ["cargo", "build", "--bin", "session_runner"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    )
    inputs = [
        {"type": "text", "text": "describe all inputs"},
        {"type": "image", "synthetic": {"width": 8, "height": 8, "seed": 1}},
        {
            "type": "audio",
            "synthetic": {"sample_rate_hz": 8000, "duration_ms": 100, "seed": 2},
        },
    ]
    if dialect != "openai":
        inputs.append(
            {
                "type": "video",
                "synthetic": {
                    "width": 8,
                    "height": 8,
                    "frames": 2,
                    "fps": 2.0,
                    "seed": 3,
                },
            }
        )

    process, port, log = _serve(tmp_path, dialect)
    try:
        result = _run_surface(
            tmp_path,
            port,
            f"chat-{dialect}",
            "openai-chat",
            dialect,
            inputs,
            [{"type": "text", "max_tokens": 2}],
        )
    finally:
        process.terminate()
        process.wait(timeout=10)

    assert result["success_steps"] == 2 and result["failed_steps"] == 0
    rows = workload_rows(log)
    expected_media = 2 if dialect == "openai" else 3
    assert len(rows) == 2
    assert all(row["media_parts"] == expected_media for row in rows)
    assert all(row["media_bytes"] > 0 for row in rows)


def test_image_edit_and_video_surfaces_replay_against_mstar(tmp_path: Path) -> None:
    subprocess.run(
        ["cargo", "build", "--bin", "session_runner"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    )
    image = tmp_path / "source.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    process, port, log = _serve(tmp_path, "mstar")
    try:
        edits = _run_surface(
            tmp_path, port, "edits", "openai-image-edits", "mstar",
            [{"type": "text", "text": "make it blue"},
             {"type": "image", "asset": _asset(image, "image/png")}],
            [{"type": "image", "width": 64, "height": 64, "steps": 4, "count": 1,
              "model_params": {"mstar": {"cfg_renorm_type": "text_channel"}}}],
        )
        assert edits["success_steps"] == 2 and edits["failed_steps"] == 0

        videos = _run_surface(
            tmp_path, port, "videos", "openai-videos", "mstar",
            [{"type": "text", "text": "pan left"},
             {"type": "image", "asset": _asset(image, "image/png")}],
            [{"type": "video", "width": 64, "height": 64, "frames": 8, "steps": 4, "fps": 8.0}],
        )
        assert videos["success_steps"] == 2 and videos["failed_steps"] == 0
    finally:
        process.terminate()
        process.wait(timeout=10)

    rows = workload_rows(log)
    edit_rows = [row for row in rows if row["surface"] == "image_edits"]
    video_rows = [row for row in rows if row["surface"] == "videos"]
    assert len(edit_rows) == 2
    # The upload arrived as real bytes, and the M*-only knob arrived flat.
    assert all(row["upload_bytes"] == 40 for row in edit_rows)
    assert all(row["cfg_renorm_type"] == "text_channel" for row in edit_rows)
    assert len(video_rows) == 2
    assert all(row["num_frames"] == 8 and row["conditioned_on_image"] for row in video_rows)


def test_dynamo_video_uses_nvext_and_input_reference_end_to_end(tmp_path: Path) -> None:
    subprocess.run(
        ["cargo", "build", "--bin", "session_runner"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    )
    image = tmp_path / "source.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    process, port, log = _serve(tmp_path, "dynamo")
    try:
        result = _run_surface(
            tmp_path, port, "dynamo-video", "openai-videos", "dynamo",
            [{"type": "text", "text": "pan left"},
             {"type": "image", "asset": _asset(image, "image/png")}],
            [{"type": "video", "width": 64, "height": 64, "frames": 8,
              "steps": 4, "fps": 8.0, "guidance": 2.5, "seed": 7}],
        )
    finally:
        process.terminate()
        process.wait(timeout=10)

    assert result["success_steps"] == 2 and result["failed_steps"] == 0
    rows = workload_rows(log)
    assert len(rows) == 2
    assert all(row["surface"] == "videos" for row in rows)
    assert all(row["num_frames"] == 8 and row["fps"] == 8.0 for row in rows)
    assert all(row["conditioned_on_image"] for row in rows)
    assert all(row["steps"] == 4 for row in rows)


def test_transcription_surface_replays_against_sglang_omni(
    tmp_path: Path,
) -> None:
    subprocess.run(
        ["cargo", "build", "--bin", "session_runner"], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    )
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 60)
    process, port, log = _serve(tmp_path, "sglang-omni")
    try:
        result = _run_surface(
            tmp_path, port, "asr", "openai-transcriptions", "sglang-omni",
            [{"type": "audio", "asset": _asset(audio, "audio/wav")}],
            [{"type": "text", "max_tokens": 16}],
        )
        assert result["success_steps"] == 2
        assert result["failed_steps"] == 0
    finally:
        process.terminate()
        process.wait(timeout=10)

    rows = workload_rows(log)
    assert len([r for r in rows if r["surface"] == "audio_transcriptions"]) == 2
    assert all(row["upload_bytes"] == 64 for row in rows)


@pytest.mark.parametrize(
    ("backend", "dialect", "surface"),
    [
        ("openai-transcriptions", "mstar", "transcription"),
        ("openai-translations", "sglang-omni", "translation"),
    ],
)
def test_a_dialect_that_does_not_serve_a_surface_fails_before_running(
    tmp_path: Path, backend: str, dialect: str, surface: str
) -> None:
    """Coverage is declared, not discovered mid-run."""
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF" + b"\x00" * 60)
    trace = tmp_path / "t.jsonl"
    trace.write_text(
        json.dumps(
            {
                "id": "x-0",
                "arrival_time_ms": 0.0,
                "inputs": [{"type": "audio", "asset": _asset(audio, "audio/wav")}],
                "outputs": [{"type": "text", "max_tokens": 8}],
            }
        )
        + "\n"
    )
    completed = subprocess.run(
        [
            str(REPO_ROOT / "target" / "debug" / "session_runner"),
            "--trace", str(trace),
            "--input-file-format", "multimodal-independent-v1",
            "--base-url", "http://127.0.0.1:9/v1",
            "--backend", backend,
            "--dialect", dialect,
            "--model", "m",
            "--arrival-mode", "saturated",
            "--summary-path", str(tmp_path / "s.json"),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60, check=False,
    )
    assert completed.returncode != 0
    assert f"does not serve {surface}" in completed.stderr
    assert not (tmp_path / "s.json").exists(), "nothing should have run"
