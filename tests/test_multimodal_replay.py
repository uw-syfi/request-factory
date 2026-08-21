from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from benchmarks.__main__ import main as benchmark_main
from launcher.config import load_task_config

REPO_ROOT = Path(__file__).resolve().parents[1]


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
        records = [json.loads(line) for line in server_log.read_text().splitlines()]
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
                [{"type": "audio", "sample_rate_hz": 24000, "max_tokens": 256, "voice": "Ethan"}],
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
        received = [json.loads(line) for line in server_log.read_text().splitlines()]
        qwen = [row for row in received if row["output_modality"] == "audio" and row["system_prompts"]]
        speech = [row for row in received if row["output_modality"] == "audio" and not row["system_prompts"]]
        assert len(qwen) == 2
        assert all(row["max_tokens"] == 256 for row in qwen)
        assert all(row["max_output_tokens"] == 256 for row in qwen)
        assert all(row["temperature"] == 0.0 for row in qwen)
        assert all(row["thinker_temperature"] == 0.0 for row in qwen)
        assert len(speech) == 2
        assert all(row["max_tokens"] == 256 for row in speech)
        assert all(row["request_id"] for row in received)
    finally:
        server.terminate()
        server.wait(timeout=5)
