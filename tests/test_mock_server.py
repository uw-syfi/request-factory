"""Tests for the multimodal mock itself.

The mock is not a stub any more: it enforces one dialect's input encoding and
knob placement, so it has logic that can be wrong. A mock that silently accepts
everything is worse than no mock, because every replay test then passes against
a shape no real server serves. These tests hold the mock to its own contract.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MOCK = REPO_ROOT / "tools" / "mock_multimodal_server.py"

PNG_DATA_URL = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode()
WAV_B64 = base64.b64encode(b"RIFFfake").decode()
WAV_DATA_URL = "data:audio/wav;base64," + WAV_B64
MP4_DATA_URL = "data:video/mp4;base64," + base64.b64encode(b"\x00\x00\x00\x18ftypmp42").decode()


class Mock:
    def __init__(self, process: subprocess.Popen, port: int) -> None:
        self.process = process
        self.port = port

    def post_json(self, path: str, body: dict) -> tuple[int, dict]:
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1{path}",
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json", "x-request-id": "t"},
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                raw = response.read()
                if response.headers.get("content-type", "").startswith("application/json"):
                    return response.status, json.loads(raw)
                return response.status, {"_raw": raw}
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def post_multipart(self, path: str, fields: dict, files: dict) -> tuple[int, dict]:
        boundary = "----mockboundary"
        parts: list[bytes] = []
        for key, value in fields.items():
            parts.append(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n"
                f"{value}\r\n".encode()
            )
        for key, blob in files.items():
            parts.append(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"; "
                f"filename=\"{key}.bin\"\r\nContent-Type: application/octet-stream\r\n\r\n".encode()
                + blob
                + b"\r\n"
            )
        parts.append(f"--{boundary}--\r\n".encode())
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1{path}",
            data=b"".join(parts),
            headers={"content-type": f"multipart/form-data; boundary={boundary}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())


def start_mock(tmp_path: Path, dialect: str) -> Mock:
    ready = tmp_path / f"ready-{dialect}"
    process = subprocess.Popen(
        [sys.executable, str(MOCK), "--dialect", dialect, "--ready-file", str(ready)],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not ready.exists():
        if process.poll() is not None:
            raise AssertionError(process.stderr.read())
        time.sleep(0.05)
    assert ready.exists(), "mock did not become ready"
    return Mock(process, int(ready.read_text()))


@pytest.fixture
def mock(request, tmp_path):
    server = start_mock(tmp_path, request.param)
    yield server
    server.process.terminate()
    server.process.wait(timeout=10)


def chat(content, **extra) -> dict:
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 2,
        "stream": True,
    }
    body.update(extra)
    return body


# --- input encoding is enforced, not merely tolerated ----------------------


@pytest.mark.parametrize("mock", ["vllm"], indirect=True)
def test_url_parts_dialect_accepts_url_media_and_rejects_input_audio(mock: Mock) -> None:
    status, _ = mock.post_json(
        "/chat/completions",
        chat([{"type": "text", "text": "x"}, {"type": "audio_url", "audio_url": {"url": WAV_DATA_URL}}]),
    )
    assert status == 200

    status, body = mock.post_json(
        "/chat/completions",
        chat([{"type": "input_audio", "input_audio": {"data": WAV_B64, "format": "wav"}}]),
    )
    assert status == 400
    assert "input_audio" in body["error"]


@pytest.mark.parametrize("mock", ["openai"], indirect=True)
def test_openai_dialect_requires_input_audio_and_refuses_url_media(mock: Mock) -> None:
    status, _ = mock.post_json(
        "/chat/completions",
        chat(
            [
                {"type": "text", "text": "x"},
                {"type": "input_audio", "input_audio": {"data": WAV_B64, "format": "wav"}},
            ]
        ),
    )
    assert status == 200

    for kind, url in (("audio_url", WAV_DATA_URL), ("video_url", MP4_DATA_URL)):
        status, body = mock.post_json("/chat/completions", chat([{"type": kind, kind: {"url": url}}]))
        assert status == 400, kind
        assert "openai dialect has no" in body["error"]


@pytest.mark.parametrize("mock", ["sglang-omni"], indirect=True)
def test_top_level_lists_dialect_rejects_content_parts(mock: Mock) -> None:
    status, _ = mock.post_json(
        "/chat/completions", chat("describe", images=[PNG_DATA_URL], audios=[WAV_DATA_URL])
    )
    assert status == 200

    status, body = mock.post_json(
        "/chat/completions", chat([{"type": "image_url", "image_url": {"url": PNG_DATA_URL}}])
    )
    assert status == 400
    assert "top-level lists" in body["error"]


# --- knob placement is enforced -------------------------------------------


@pytest.mark.parametrize("mock", ["mstar"], indirect=True)
def test_flat_dialect_rejects_a_nested_knob_envelope(mock: Mock) -> None:
    """The failure the dialect table exists to prevent.

    M* reads unknown fields from pydantic ``model_extra`` -- flat. Knobs wrapped
    in ``extra_body`` arrive as one unrecognized field and are silently dropped,
    so the mock refuses rather than accepting a request whose knobs do nothing.
    """
    status, _ = mock.post_json(
        "/images/generations",
        {"model": "m", "prompt": "p", "n": 1, "num_inference_steps": 50},
    )
    assert status == 200

    status, body = mock.post_json(
        "/images/generations",
        {"model": "m", "prompt": "p", "n": 1, "extra_body": {"num_inference_steps": 50}},
    )
    assert status == 400
    assert "extra_body" in body["error"]


@pytest.mark.parametrize("mock", ["vllm-omni"], indirect=True)
def test_nested_dialect_accepts_its_envelope_and_rejects_another(mock: Mock) -> None:
    status, _ = mock.post_json(
        "/images/generations",
        {"model": "m", "prompt": "p", "n": 1, "extra_body": {"num_inference_steps": 50}},
    )
    assert status == 200

    status, body = mock.post_json(
        "/images/generations",
        {"model": "m", "prompt": "p", "n": 1, "nvext": {"num_inference_steps": 50}},
    )
    assert status == 400
    assert "nvext" in body["error"]


# --- every surface answers in the documented shape -------------------------


@pytest.mark.parametrize("mock", ["mstar"], indirect=True)
def test_image_generation_returns_n_images(mock: Mock) -> None:
    status, body = mock.post_json("/images/generations", {"model": "m", "prompt": "p", "n": 3})
    assert status == 200
    # A client folding only data[0] must under-report against this.
    assert len(body["data"]) == 3
    assert all("b64_json" in item for item in body["data"])


@pytest.mark.parametrize("mock", ["mstar"], indirect=True)
def test_image_edits_requires_an_upload(mock: Mock) -> None:
    status, body = mock.post_multipart(
        "/images/edits", {"model": "m", "prompt": "bluer", "n": 1}, {"image": b"\x89PNGfake"}
    )
    assert status == 200
    assert len(body["data"]) == 1

    status, body = mock.post_multipart("/images/edits", {"model": "m", "prompt": "bluer"}, {})
    assert status == 400
    assert "image" in body["error"]


@pytest.mark.parametrize("mock", ["mstar"], indirect=True)
def test_video_generation_returns_base64_and_needs_frames(mock: Mock) -> None:
    status, body = mock.post_json(
        "/videos/generations",
        {"model": "m", "prompt": "p", "size": "64x64", "num_frames": 8, "fps": 8.0},
    )
    assert status == 200
    assert base64.b64decode(body["data"][0]["b64_json"]).startswith(b"\x00\x00\x00\x18ftyp")

    status, body = mock.post_json("/videos/generations", {"model": "m", "prompt": "p"})
    assert status == 400
    assert "num_frames" in body["error"]


@pytest.mark.parametrize("mock", ["sglang-omni"], indirect=True)
def test_transcription_and_translation_return_text(mock: Mock) -> None:
    for path in ("/audio/transcriptions", "/audio/translations"):
        status, body = mock.post_multipart(
            path, {"model": "m", "response_format": "json"}, {"file": b"RIFFfake"}
        )
        assert status == 200, path
        assert body["text"] == "mock transcript"

    status, body = mock.post_multipart("/audio/transcriptions", {"model": "m"}, {})
    assert status == 400
    assert "file" in body["error"]


@pytest.mark.parametrize("mock", ["mstar"], indirect=True)
def test_speech_streams_raw_pcm(mock: Mock) -> None:
    status, body = mock.post_json(
        "/audio/speech", {"model": "m", "input": "hello", "response_format": "wav"}
    )
    assert status == 200
    assert len(body["_raw"]) == 3 * 480


@pytest.mark.parametrize("mock", ["vllm-omni"], indirect=True)
def test_speech_streams_sse_deltas_with_usage(mock: Mock) -> None:
    status, body = mock.post_json(
        "/audio/speech",
        {"model": "m", "input": "hello", "response_format": "pcm", "stream_format": "sse"},
    )
    assert status == 200
    text = body["_raw"].decode()
    assert text.count("speech.audio.delta") == 3
    assert "speech.audio.done" in text


@pytest.mark.parametrize("mock", ["mstar"], indirect=True)
def test_unknown_endpoint_is_a_404(mock: Mock) -> None:
    status, _ = mock.post_json("/nope", {})
    assert status == 404
