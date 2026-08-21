"""End-to-end text transport matrix against the deterministic CPU stub."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def _built() -> None:
    subprocess.run(
        ["cargo", "build", "--bin", "session_runner"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _start_stub(protocol: str) -> tuple[subprocess.Popen, int]:
    port = _free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "stub_server.py"),
            "--port",
            str(port),
            "--protocol",
            protocol,
            "--sse-no-space",
            "--tokens-per-chunk",
            "2",
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(process.stderr.read())
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return process, port
        except OSError:
            time.sleep(0.02)
    process.terminate()
    raise AssertionError("text stub did not become ready")


def _fixtures(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text(
        json.dumps(
            {
                "version": "1.0",
                "truncation": None,
                "padding": None,
                "added_tokens": [],
                "normalizer": None,
                "pre_tokenizer": {"type": "Whitespace"},
                "post_processor": None,
                "decoder": None,
                "model": {
                    "type": "WordLevel",
                    "vocab": {"[UNK]": 0, "a": 1, "b": 2, "c": 3, "d": 4},
                    "unk_token": "[UNK]",
                },
            }
        )
    )
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("a b c d " * 5_000)

    independent = tmp_path / "independent.csv"
    independent.write_text(
        "id,input_len,output_len,arrival_time\n"
        + "\n".join(f"request-{index},8,3,{index}.0" for index in range(4))
        + "\n"
    )
    sessions = tmp_path / "sessions.csv"
    sessions.write_text(
        "request_id,session_id,round_idx,arrival_time_ms,prefix_len,input_len,"
        "output_len,tool_wait_after_ms\n"
        "session_a_round_000000,a,0,0.0,0,8,3,0.0\n"
        "session_a_round_000001,a,1,0.0,11,8,3,0.0\n"
        "session_b_round_000000,b,0,2.0,0,8,3,1.0\n"
        "session_b_round_000001,b,1,2.0,11,8,3,0.0\n"
    )
    return tokenizer, corpus, independent, sessions


def _run(
    tmp_path: Path,
    *,
    backend: str,
    port: int,
    tokenizer: Path,
    corpus: Path,
    trace: Path,
    input_format: str,
    name: str,
    saturated: bool,
) -> dict:
    summary = tmp_path / f"{name}-summary.json"
    command = [
        str(REPO_ROOT / "target" / "debug" / "session_runner"),
        "--trace",
        str(trace),
        "--input-file-format",
        input_format,
        "--text-file",
        str(corpus),
        "--tokenizer",
        str(tokenizer),
        "--model",
        "stub-model",
        "--backend",
        backend,
        "--base-url",
        f"http://127.0.0.1:{port}/v1" if backend == "openai" else f"http://127.0.0.1:{port}",
        "--arrival-mode",
        "saturated" if saturated else "trace-timed",
        "--max-concurrency",
        "3" if saturated else "1",
        "--timeline",
        "false" if saturated else "true",
        "--timeline-path",
        str(tmp_path / f"{name}-timeline.parquet"),
        "--log-path",
        str(tmp_path / f"{name}-requests.jsonl"),
        "--summary-path",
        str(summary),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(summary.read_text())


@pytest.mark.parametrize(
    ("backend", "protocol"),
    [("openai", "openai"), ("vllm-tokens", "vllm"), ("sglang-tokens", "sglang")],
)
def test_text_backends_cover_independent_and_session_configs(
    tmp_path: Path, backend: str, protocol: str
) -> None:
    tokenizer, corpus, independent, sessions = _fixtures(tmp_path)
    process, port = _start_stub(protocol)
    try:
        independent_result = _run(
            tmp_path,
            backend=backend,
            port=port,
            tokenizer=tokenizer,
            corpus=corpus,
            trace=independent,
            input_format="text-generation-independent",
            name=f"{backend}-independent",
            saturated=True,
        )
        session_result = _run(
            tmp_path,
            backend=backend,
            port=port,
            tokenizer=tokenizer,
            corpus=corpus,
            trace=sessions,
            input_format="text-generation-session-execution-v2",
            name=f"{backend}-sessions",
            saturated=False,
        )
    finally:
        process.terminate()
        process.wait(timeout=10)

    independent_common = independent_result["replay"]["common"]
    assert independent_common["success_steps"] == 4
    assert independent_common["failed_steps"] == 0

    session_common = session_result["replay"]["common"]
    assert session_common["success_steps"] == 4
    assert session_common["failed_steps"] == 0
    prefix = session_result["replay"]["prefix_cache"]
    assert prefix["measured_cache_steps"] == 4
    assert prefix["server_prefix_hit_rate"] is not None
    assert session_result["timeline"]["events_written"] > 0
