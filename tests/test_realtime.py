"""The /realtime WebSocket surface.

Three systems serve it and all three disagree: OpenAI renamed the audio event,
vLLM-Omni kept the old name but moved the payload into a different field and
drives a turn from an input buffer instead of a conversation item. Reading the
wrong field yields zero bytes and a run that still looks like it worked, so
these tests check both directions -- that the right pairing succeeds, and that a
wrong one fails loudly rather than quietly measuring nothing.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

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



@pytest.fixture(scope="module", autouse=True)
def _built() -> None:
    subprocess.run(
        ["cargo", "build", "--bin", "session_runner"],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    )


def _serve(tmp_path: Path, dialect: str, tag: str = ""):
    # The ready-file must be unique per server: a stale one from an earlier
    # instance would hand back a port nothing is listening on any more.
    ready = tmp_path / f"ready-{dialect}{tag}"
    log = tmp_path / f"server-{dialect}{tag}.jsonl"
    process = subprocess.Popen(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "mock_multimodal_server.py"),
            "--dialect", dialect,
            "--ready-file", str(ready),
            "--log-path", str(log),
        ],
        cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and not ready.exists():
        if process.poll() is not None:
            raise AssertionError(process.stderr.read())
        time.sleep(0.05)
    assert ready.exists(), "mock did not become ready"
    return process, int(ready.read_text()), log


def _trace(path: Path, inputs: list[dict], count: int = 2) -> Path:
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"rt-{index}",
                    "arrival_time_ms": float(index),
                    "inputs": inputs,
                    "outputs": [
                        {"type": "audio", "sample_rate_hz": 24000, "voice": "Ethan"}
                    ],
                }
            )
            for index in range(count)
        )
        + "\n"
    )
    return path


def _replay(tmp_path: Path, trace: Path, port: int, dialect: str, name: str):
    summary = tmp_path / f"{name}-summary.json"
    log = tmp_path / f"{name}-requests.jsonl"
    completed = subprocess.run(
        [
            str(REPO_ROOT / "target" / "debug" / "session_runner"),
            "--trace", str(trace),
            "--input-file-format", "multimodal-independent-v1",
            "--base-url", f"http://127.0.0.1:{port}/v1",
            "--backend", "openai-realtime",
            "--dialect", dialect,
            "--model", "m",
            "--arrival-mode", "saturated",
            "--max-concurrency", "2",
            "--timeline", "false",
            "--summary-path", str(summary),
            "--log-path", str(log),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60, check=False,
    )
    return completed, summary, log


def test_item_style_turn_streams_audio_back(tmp_path: Path) -> None:
    trace = _trace(tmp_path / "t.jsonl", [{"type": "text", "text": "say hello"}])
    process, port, log = _serve(tmp_path, "sglang-omni")
    try:
        completed, summary, requests = _replay(tmp_path, trace, port, "sglang-omni", "ok")
        assert completed.returncode == 0, completed.stderr
    finally:
        process.terminate()
        process.wait(timeout=10)

    result = json.loads(summary.read_text())["replay"]["common"]
    assert result["success_steps"] == 2 and result["failed_steps"] == 0
    outcome = workload_rows(requests)[0]["outcome"]
    assert outcome["output_chunk_count"] == 3
    assert outcome["output_bytes"] == 3 * 480
    # First-output latency is observed from the socket, not inferred.
    assert outcome["first_output_ms"] is not None
    assert outcome["real_time_factor"] is not None

    rows = workload_rows(log)
    assert all(row["surface"] == "realtime" for row in rows)
    assert rows[0]["client_events"] == [
        "session.update",
        "conversation.item.create",
        "response.create",
    ]
    assert rows[0]["voice"] == "Ethan"


def test_buffer_style_turn_streams_input_and_commits(tmp_path: Path) -> None:
    """vLLM-Omni drives a turn from an input buffer, with no item at all."""
    trace = _trace(
        tmp_path / "t.jsonl",
        # Synthetic audio: this surface needs bytes, not a recorded corpus.
        [{"type": "audio", "synthetic": {"sample_rate_hz": 16000, "duration_ms": 500, "seed": 3}}],
    )
    process, port, log = _serve(tmp_path, "vllm-omni")
    try:
        completed, summary, _ = _replay(tmp_path, trace, port, "vllm-omni", "buffer")
        assert completed.returncode == 0, completed.stderr
    finally:
        process.terminate()
        process.wait(timeout=10)

    assert json.loads(summary.read_text())["replay"]["common"]["success_steps"] == 2
    row = workload_rows(log)[0]
    assert row["turn_style"] == "buffer"
    assert "conversation.item.create" not in row["client_events"]
    assert "response.create" not in row["client_events"]
    assert row["client_events"][-1] == "input_audio_buffer.commit"
    # 16 kHz * 0.5 s * 2 bytes, plus the 44-byte WAV header the generator wrote.
    assert row["audio_input_bytes"] == 16000 // 2 * 2 + 44


def test_the_wrong_dialect_fails_instead_of_measuring_nothing(tmp_path: Path) -> None:
    """A client speaking the wrong realtime dialect must not look successful."""
    trace = _trace(tmp_path / "t.jsonl", [{"type": "text", "text": "say hello"}])
    # The server drives turns from an input buffer; the client will send an item.
    process, port, _ = _serve(tmp_path, "vllm-omni")
    try:
        completed, summary, requests = _replay(tmp_path, trace, port, "sglang-omni", "mismatch")
        assert completed.returncode == 0, completed.stderr
    finally:
        process.terminate()
        process.wait(timeout=10)

    assert json.loads(summary.read_text())["replay"]["common"]["failed_steps"] == 2
    error = workload_rows(requests)[0]["outcome"]["error"]
    assert "realtime server error" in error


def test_openai_names_the_audio_event_differently(tmp_path: Path) -> None:
    trace = _trace(tmp_path / "t.jsonl", [{"type": "text", "text": "say hello"}], count=1)
    process, port, _ = _serve(tmp_path, "openai")
    try:
        completed, summary, requests = _replay(tmp_path, trace, port, "openai", "openai")
        assert completed.returncode == 0, completed.stderr
    finally:
        process.terminate()
        process.wait(timeout=10)
    assert json.loads(summary.read_text())["replay"]["common"]["success_steps"] == 1

    # And the same server refuses to feed a client expecting the older name:
    # the event arrives, the field is right, but the type never matches.
    process, port, _ = _serve(tmp_path, "openai", tag="-crossed")
    try:
        completed, summary, requests = _replay(tmp_path, trace, port, "sglang-omni", "crossed")
        assert completed.returncode == 0, completed.stderr
    finally:
        process.terminate()
        process.wait(timeout=10)
    result = json.loads(summary.read_text())["replay"]["common"]
    assert result["failed_steps"] == 1, "a renamed event must not silently read as zero audio"
    assert "no output" in workload_rows(requests)[0]["outcome"]["error"]


def test_a_system_without_realtime_is_rejected_before_connecting(tmp_path: Path) -> None:
    trace = _trace(tmp_path / "t.jsonl", [{"type": "text", "text": "hi"}], count=1)
    completed, summary, _ = _replay(tmp_path, trace, 9, "mstar", "unsupported")
    assert completed.returncode != 0
    assert "does not serve realtime sessions" in completed.stderr
    assert not summary.exists(), "nothing should have run"
