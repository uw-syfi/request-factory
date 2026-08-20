"""CPU-only mock for every multimodal surface req-frontend can drive.

Serves the seven HTTP surfaces exposed by M*, vLLM-Omni and SGLang-Omni:
chat completions, image generation, image edits, video generation, speech,
transcription and translation.

It answers in a chosen **dialect** and, just as importantly, *validates the
request against that dialect*. A mock that mirrors whatever the client sends can
never fail a test: both sides agree on a shape no real server serves, and a
wrong field name looks like success. So this one rejects media encoded the wrong
way, and rejects model knobs placed in the wrong envelope -- the two mistakes
that are otherwise silent, because real servers ignore what they do not
recognize.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

DIALECTS = ("openai", "vllm", "vllm-omni", "sglang-omni", "mstar", "dynamo")

# How each dialect attaches media to a chat request.
ENCODING = {
    "openai": "openai_parts",
    "vllm": "url_parts",
    "vllm-omni": "url_parts",
    "sglang-omni": "top_level_lists",
    "mstar": "url_parts",
    "dynamo": "url_parts",
}

# Where each dialect expects model knobs. `None` means flat at the request root,
# which is what a pydantic ``extra="allow"`` server reads from ``model_extra``.
KNOB_ENVELOPE = {
    "openai": None,
    "vllm": None,
    "vllm-omni": "extra_body",
    "sglang-omni": None,
    "mstar": None,
    "dynamo": "nvext",
}

# Realtime event vocabulary per dialect. Three systems serve /v1/realtime and
# all three name the audio delta differently, so the mock answers in the name
# the configured dialect expects -- and refuses a turn driven the wrong way.
REALTIME = {
    "openai": {
        "turn": "item",
        "audio_type": "response.output_audio.delta",
        "audio_field": "delta",
        "done": "response.done",
    },
    "vllm": {
        "turn": "item",
        "audio_type": "response.output_audio.delta",
        "audio_field": "delta",
        "done": "response.done",
    },
    "vllm-omni": {
        "turn": "buffer",
        "audio_type": "response.audio.delta",
        "audio_field": "audio",
        "done": "response.done",
    },
    "sglang-omni": {
        "turn": "item",
        "audio_type": "response.audio.delta",
        "audio_field": "delta",
        "done": "response.done",
    },
    "mstar": {"turn": "item", "audio_type": "response.audio.delta", "audio_field": "delta", "done": "response.done"},
    "dynamo": {"turn": "item", "audio_type": "response.audio.delta", "audio_field": "delta", "done": "response.done"},
}

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

_PNG = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
_MP4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 24


class KnobPlacementError(ValueError):
    """Knobs arrived in an envelope this dialect does not read."""


class State:
    def __init__(self, log_path: Path | None, chunk_delay_ms: float, dialect: str) -> None:
        self.dialect = dialect
        self.realtime = REALTIME[dialect]
        self.encoding = ENCODING[dialect]
        self.knob_envelope = KNOB_ENVELOPE[dialect]
        self.log_path = log_path
        self.chunk_delay_s = chunk_delay_ms / 1000.0
        self.lock = threading.Lock()
        self.requests = 0
        self.active = 0
        self.max_active = 0

    def begin(self, value: dict[str, object]) -> None:
        with self.lock:
            self.requests += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            value["active_at_receive"] = self.active
            value["max_active"] = self.max_active
            value["dialect"] = self.dialect
            if self.log_path is not None:
                with self.log_path.open("a") as writer:
                    writer.write(json.dumps(value) + "\n")

    def finish(self) -> None:
        with self.lock:
            self.active -= 1


def _data_url_bytes(url: str) -> bytes:
    if not url.startswith("data:") or "," not in url:
        raise ValueError(f"expected a data URL, got {url[:32]!r}")
    return base64.b64decode(url.split(",", 1)[1])


def _parse_multipart(body: bytes, content_type: str) -> tuple[dict[str, str], dict[str, bytes]]:
    """Minimal multipart/form-data reader: fields and file parts."""
    marker = "boundary="
    if marker not in content_type:
        raise ValueError("multipart request without a boundary")
    boundary = content_type.split(marker, 1)[1].strip().strip('"')
    sections = body.split(b"--" + boundary.encode())
    fields: dict[str, str] = {}
    files: dict[str, bytes] = {}
    for section in sections:
        section = section.strip(b"\r\n")
        if not section or section == b"--":
            continue
        head, _, payload = section.partition(b"\r\n\r\n")
        headers = head.decode("utf-8", "replace")
        name = ""
        for chunk in headers.split(";"):
            chunk = chunk.strip()
            if chunk.startswith("name="):
                name = chunk[5:].strip().strip('"').split("\r\n")[0]
        if not name:
            continue
        if "filename=" in headers:
            files[name] = payload
        else:
            fields[name] = payload.decode("utf-8", "replace")
    return fields, files


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "req-frontend-multimodal-mock/2"

    @property
    def state(self) -> State:
        return self.server.state  # type: ignore[attr-defined]

    def log_message(self, _format: str, *_args: object) -> None:
        return

    # ---- plumbing -------------------------------------------------------

    def _json_error(self, status: int, message: str) -> None:
        body = json.dumps({"error": message}).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_json(self, value: dict[str, object]) -> None:
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _start_stream(self, content_type: str) -> None:
        self.send_response(200)
        self.send_header("content-type", content_type)
        self.send_header("cache-control", "no-cache")
        # Body framed by connection close, not chunked: the mock writes raw SSE
        # and raw PCM, and announcing chunked without framing it corrupts both.
        self.send_header("connection", "close")
        self.end_headers()

    def _stream_event(self, value: dict[str, object]) -> None:
        self.wfile.write(f"data: {json.dumps(value)}\n\n".encode())
        self.wfile.flush()

    # ---- dialect enforcement --------------------------------------------

    def _knob(self, payload: dict[str, object], key: str) -> object:
        """Read a knob from where this dialect says it belongs -- only there."""
        envelope = self.state.knob_envelope
        if envelope is None:
            return payload.get(key)
        nested = payload.get(envelope)
        return nested.get(key) if isinstance(nested, dict) else None

    def _check_knob_placement(self, payload: dict[str, object]) -> None:
        """Reject knobs sent in an envelope this dialect does not read.

        This is the failure that motivated the whole dialect table: a server
        reading ``model_extra`` receives a literal ``extra_body`` key as one
        unknown field and drops every knob inside it, reporting nothing.
        """
        wrong = [name for name in ("extra_body", "nvext") if name != self.state.knob_envelope]
        for name in wrong:
            if isinstance(payload.get(name), dict):
                raise KnobPlacementError(
                    f"dialect {self.state.dialect} reads knobs from "
                    f"{self.state.knob_envelope or 'the request root'}, "
                    f"but received a {name!r} envelope"
                )

    def _collect_media(self, payload: dict[str, object]) -> tuple[int, int, int, str]:
        """Count text/media parts, enforcing this dialect's input encoding.

        Also digests the media bytes, so a test can tell "every request carried
        the same content" from "each carried its own" -- the distinction a
        synthetic-content run is configured around.
        """
        messages = payload["messages"]
        users = [message for message in messages if message.get("role") == "user"]
        if len(users) != 1:
            raise ValueError("mock requires exactly one user message")
        content = users[0]["content"]
        text_parts = media_parts = media_bytes = 0
        digest = hashlib.sha256()
        encoding = self.state.encoding

        if encoding == "top_level_lists":
            if not isinstance(content, str):
                raise ValueError(
                    "sglang-omni carries plain-text content and media in top-level lists"
                )
            text_parts = 1 if content else 0
            for key in ("images", "audios", "videos"):
                for url in payload.get(key, []) or []:
                    blob = _data_url_bytes(url)
                    media_parts += 1
                    media_bytes += len(blob)
                    digest.update(blob)
            return text_parts, media_parts, media_bytes, digest.hexdigest()

        if not isinstance(content, list) or not content:
            raise ValueError("user message content must be a non-empty list")
        for part in content:
            kind = part.get("type")
            if kind == "text":
                text_parts += 1
                continue
            if kind == "image_url":
                blob = _data_url_bytes(part["image_url"]["url"])
                media_parts += 1
                media_bytes += len(blob)
                digest.update(blob)
                continue
            if encoding == "openai_parts":
                if kind == "input_audio":
                    blob = base64.b64decode(part["input_audio"]["data"])
                    media_parts += 1
                    media_bytes += len(blob)
                    digest.update(blob)
                    continue
                raise ValueError(
                    f"the openai dialect has no {kind!r} content part "
                    "(audio is input_audio; video is not supported)"
                )
            if kind in ("audio_url", "video_url"):
                blob = _data_url_bytes(part[kind]["url"])
                media_parts += 1
                media_bytes += len(blob)
                digest.update(blob)
                continue
            raise ValueError(f"unsupported content part {kind!r}")
        return text_parts, media_parts, media_bytes, digest.hexdigest()

    # ---- websocket ------------------------------------------------------

    def _ws_accept(self) -> bool:
        """Complete the RFC 6455 handshake, or answer 400 and give up."""
        key = self.headers.get("sec-websocket-key")
        upgrade = (self.headers.get("upgrade") or "").lower()
        if upgrade != "websocket" or not key:
            self._json_error(400, "expected a websocket upgrade")
            return False
        accept = base64.b64encode(hashlib.sha1((key + _WS_GUID).encode()).digest()).decode()
        self.send_response(101)
        self.send_header("upgrade", "websocket")
        self.send_header("connection", "Upgrade")
        self.send_header("sec-websocket-accept", accept)
        self.end_headers()
        return True

    def _ws_recv(self) -> tuple[int, bytes] | None:
        """One frame as (opcode, payload). Client frames are always masked."""
        header = self.rfile.read(2)
        if len(header) < 2:
            return None
        opcode = header[0] & 0x0F
        masked = bool(header[1] & 0x80)
        length = header[1] & 0x7F
        if length == 126:
            length = struct.unpack(">H", self.rfile.read(2))[0]
        elif length == 127:
            length = struct.unpack(">Q", self.rfile.read(8))[0]
        mask = self.rfile.read(4) if masked else b""
        payload = self.rfile.read(length)
        if masked:
            payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        return opcode, payload

    def _ws_send(self, payload: bytes, opcode: int = 0x1) -> None:
        header = bytearray([0x80 | opcode])
        length = len(payload)
        if length < 126:
            header.append(length)
        elif length < 1 << 16:
            header.append(126)
            header += struct.pack(">H", length)
        else:
            header.append(127)
            header += struct.pack(">Q", length)
        self.wfile.write(bytes(header) + payload)
        self.wfile.flush()

    def _ws_event(self, value: dict[str, object]) -> None:
        self._ws_send(json.dumps(value).encode())

    def _serve_realtime(self) -> None:
        if not self._ws_accept():
            return
        shape = self.state.realtime
        seen: list[str] = []
        text_inputs = 0
        audio_input_bytes = 0
        voice = None
        try:
            while True:
                frame = self._ws_recv()
                if frame is None:
                    return
                opcode, payload = frame
                if opcode == 0x8:  # close
                    return
                if opcode in (0x9, 0xA):  # ping/pong
                    continue
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    self._ws_event({"type": "error", "error": {"message": "invalid JSON"}})
                    return
                kind = event.get("type")
                seen.append(kind)

                # Hold the client to this dialect's turn style. A server that
                # accepted either would let a wrong-dialect client look correct.
                if shape["turn"] == "buffer" and kind in (
                    "conversation.item.create",
                    "response.create",
                ):
                    self._ws_event(
                        {
                            "type": "error",
                            "error": {
                                "message": f"{self.state.dialect} drives a turn from the input "
                                f"buffer; {kind} is not accepted"
                            },
                        }
                    )
                    return
                if shape["turn"] == "item" and kind == "input_audio_buffer.append":
                    self._ws_event(
                        {
                            "type": "error",
                            "error": {
                                "message": f"{self.state.dialect} expects conversation.item.create,"
                                " not an input buffer"
                            },
                        }
                    )
                    return

                if kind == "session.update":
                    session = event.get("session") or {}
                    voice = session.get("voice")
                    modalities = session.get("modalities")
                    if modalities is not None and set(modalities) not in (
                        {"text"},
                        {"text", "audio"},
                    ):
                        self._ws_event(
                            {
                                "type": "error",
                                "error": {"message": "modalities must be text or text+audio"},
                            }
                        )
                        return
                    self._ws_event({"type": "session.updated", "session": session})
                    continue
                if kind == "conversation.item.create":
                    for part in event["item"]["content"]:
                        if part["type"] == "input_text":
                            text_inputs += 1
                        elif part["type"] == "input_audio":
                            audio_input_bytes += len(base64.b64decode(part["audio"]))
                    self._ws_event({"type": "conversation.item.created"})
                    continue
                if kind == "input_audio_buffer.append":
                    audio_input_bytes += len(base64.b64decode(event["audio"]))
                    continue
                if kind == "input_audio_buffer.commit" and not event.get("final"):
                    continue
                if kind in ("response.create", "input_audio_buffer.commit"):
                    self.state.begin(
                        {
                            "request_id": None,
                            "surface": "realtime",
                            "output_modality": "audio",
                            "turn_style": shape["turn"],
                            "client_events": seen,
                            "text_inputs": text_inputs,
                            "audio_input_bytes": audio_input_bytes,
                            "voice": voice,
                            "system_prompts": [],
                        }
                    )
                    self._ws_event({"type": "response.created"})
                    for _ in range(3):
                        self._ws_event(
                            {
                                "type": shape["audio_type"],
                                shape["audio_field"]: base64.b64encode(
                                    b"\x01\x00" * 240
                                ).decode(),
                            }
                        )
                        if self.state.chunk_delay_s:
                            time.sleep(self.state.chunk_delay_s)
                    self._ws_event({"type": shape["done"]})
                    self.state.finish()
                    return
        except (OSError, KeyError, TypeError, ValueError):
            return

    def do_GET(self) -> None:
        path = urlparse(self.path).path.removeprefix("/v1")
        if path == "/realtime":
            self._serve_realtime()
            return
        self._json_error(404, "unsupported mock endpoint")

    # ---- routing --------------------------------------------------------

    def do_POST(self) -> None:
        path = urlparse(self.path).path.removeprefix("/v1")
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length)
        content_type = self.headers.get("content-type", "")
        try:
            if path in ("/images/edits", "/audio/transcriptions", "/audio/translations"):
                fields, files = _parse_multipart(body, content_type)
                if path == "/images/edits":
                    self._serve_image_edit(fields, files)
                else:
                    self._serve_transcription(fields, files, path)
                return
            payload = json.loads(body)
            if path == "/images/generations":
                self._serve_image_generation(payload)
            elif path == "/videos/generations" or path == "/videos":
                self._serve_video_generation(payload)
            elif path == "/audio/speech":
                self._serve_speech(payload)
            elif path == "/chat/completions":
                self._serve_chat(payload)
            else:
                self._json_error(404, "unsupported mock endpoint")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._json_error(400, str(error))

    # ---- surfaces -------------------------------------------------------

    def _serve_chat(self, payload: dict[str, object]) -> None:
        self._check_knob_placement(payload)
        messages = payload["messages"]
        system_prompts = [m["content"] for m in messages if m.get("role") == "system"]
        if any(not isinstance(p, str) or not p for p in system_prompts):
            raise ValueError("system message content must be a non-empty string")
        text_parts, media_parts, media_bytes, media_sha256 = self._collect_media(payload)

        modalities = payload.get("modalities")
        output_modality = "text"
        if isinstance(modalities, list) and modalities:
            output_modality = "audio" if "audio" in modalities else modalities[0]
        if output_modality == "text":
            max_tokens = int(payload.get("max_tokens") or payload.get("max_completion_tokens") or 0)
            if media_parts == 0:
                raise ValueError("text mock requires at least one media part")
            if max_tokens <= 0 or payload.get("stream") is not True:
                raise ValueError("positive output-token cap and stream=true are required")
        else:
            max_tokens = int(payload.get("max_tokens") or 0)

        self.state.begin(
            {
                "request_id": self.headers.get("x-request-id"),
                "surface": "chat",
                "text_parts": text_parts,
                "media_parts": media_parts,
                "media_bytes": media_bytes,
                "media_sha256": media_sha256,
                "max_tokens": max_tokens,
                "max_output_tokens": self._knob(payload, "max_output_tokens"),
                "temperature": payload.get("temperature"),
                "thinker_temperature": self._knob(payload, "thinker_temperature"),
                "voice": (payload.get("audio") or {}).get("voice"),
                "system_prompts": system_prompts,
                "output_modality": output_modality,
            }
        )

        if output_modality == "image":
            self._write_json(
                {
                    "choices": [
                        {
                            "message": {
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": "data:image/png;base64,"
                                            + base64.b64encode(_PNG).decode()
                                        },
                                    }
                                ]
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }
            )
            self.state.finish()
            return

        if output_modality == "audio":
            self._start_stream("text/event-stream")
            for _ in range(3):
                encoded = base64.b64encode(b"\x01\x00" * 240).decode()
                if self.state.dialect == "vllm-omni":
                    event = {
                        "modality": "audio",
                        "choices": [{"delta": {"content": encoded}, "finish_reason": None}],
                    }
                else:
                    event = {
                        "choices": [
                            {"delta": {"audio": {"id": "a0", "data": encoded}}, "finish_reason": None}
                        ]
                    }
                self._stream_event(event)
                if self.state.chunk_delay_s:
                    time.sleep(self.state.chunk_delay_s)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.state.finish()
            self.close_connection = True
            return

        self._start_stream("text/event-stream")
        for index in range(max_tokens):
            self._stream_event(
                {
                    "choices": [
                        {
                            "delta": {"content": f"t{index} "},
                            "finish_reason": "length" if index == max_tokens - 1 else None,
                        }
                    ]
                }
            )
            if self.state.chunk_delay_s:
                time.sleep(self.state.chunk_delay_s)
        self._stream_event(
            {"choices": [], "usage": {"prompt_tokens": 8, "completion_tokens": max_tokens}}
        )
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.state.finish()
        self.close_connection = True

    def _serve_image_generation(self, payload: dict[str, object]) -> None:
        self._check_knob_placement(payload)
        if not payload.get("prompt"):
            raise ValueError("image generation requires a prompt")
        count = int(payload.get("n") or 1)
        self.state.begin(
            {
                "request_id": self.headers.get("x-request-id"),
                "surface": "images",
                "output_modality": "image",
                "count": count,
                "size": payload.get("size"),
                "steps": self._knob(payload, "num_inference_steps"),
                "guidance": self._knob(payload, "guidance_scale"),
                "cfg_text_scale": self._knob(payload, "cfg_text_scale"),
                "cfg_renorm_type": self._knob(payload, "cfg_renorm_type"),
                "system_prompts": [],
            }
        )
        # Honour `n`: a client that folds only data[0] must under-report here.
        self._write_json(
            {"data": [{"b64_json": base64.b64encode(_PNG).decode()} for _ in range(count)]}
        )
        self.state.finish()

    def _serve_image_edit(self, fields: dict[str, str], files: dict[str, bytes]) -> None:
        if "image" not in files:
            raise ValueError("images/edits requires an 'image' file upload")
        if not fields.get("prompt"):
            raise ValueError("images/edits requires a prompt")
        count = int(fields.get("n") or 1)
        self.state.begin(
            {
                "request_id": self.headers.get("x-request-id"),
                "surface": "image_edits",
                "output_modality": "image",
                "count": count,
                "upload_bytes": len(files["image"]),
                "steps": fields.get("num_inference_steps"),
                "cfg_renorm_type": fields.get("cfg_renorm_type"),
                "system_prompts": [],
            }
        )
        self._write_json(
            {"data": [{"b64_json": base64.b64encode(_PNG).decode()} for _ in range(count)]}
        )
        self.state.finish()

    def _serve_video_generation(self, payload: dict[str, object]) -> None:
        self._check_knob_placement(payload)
        frames = int(payload.get("num_frames") or 0)
        if frames <= 0:
            raise ValueError("video generation requires positive num_frames")
        self.state.begin(
            {
                "request_id": self.headers.get("x-request-id"),
                "surface": "videos",
                "output_modality": "video",
                "num_frames": frames,
                "fps": payload.get("fps"),
                "size": payload.get("size"),
                "conditioned_on_image": "image" in payload,
                "conditioned_on_video": "video" in payload,
                "steps": self._knob(payload, "num_inference_steps"),
                "system_prompts": [],
            }
        )
        self._write_json({"data": [{"b64_json": base64.b64encode(_MP4).decode()}]})
        self.state.finish()

    def _serve_transcription(
        self, fields: dict[str, str], files: dict[str, bytes], path: str
    ) -> None:
        if "file" not in files:
            raise ValueError(f"{path} requires a 'file' audio upload")
        self.state.begin(
            {
                "request_id": self.headers.get("x-request-id"),
                "surface": path.strip("/").replace("/", "_"),
                "output_modality": "text",
                "upload_bytes": len(files["file"]),
                "response_format": fields.get("response_format"),
                "system_prompts": [],
            }
        )
        self._write_json({"text": "mock transcript"})
        self.state.finish()

    def _serve_speech(self, payload: dict[str, object]) -> None:
        self._check_knob_placement(payload)
        text = payload.get("input")
        if not isinstance(text, str) or not text:
            raise ValueError("speech requires non-empty input")
        if payload.get("response_format") not in ("pcm", "wav"):
            raise ValueError("mock speech requires response_format pcm or wav")
        self.state.begin(
            {
                "request_id": self.headers.get("x-request-id"),
                "surface": "speech",
                "text_parts": 1,
                "media_parts": 0,
                "media_bytes": 0,
                "max_tokens": payload.get("max_new_tokens", 0),
                "output_modality": "audio",
                "voice": payload.get("voice"),
                "system_prompts": [],
            }
        )
        if payload.get("stream_format") == "sse":
            # vLLM-Omni's richer speech stream: explicit delta events plus a
            # terminal usage block, which raw PCM cannot carry at all.
            self._start_stream("text/event-stream")
            for _ in range(3):
                self._stream_event(
                    {
                        "type": "speech.audio.delta",
                        "audio": base64.b64encode(b"\x01\x00" * 240).decode(),
                        "response_format": "pcm",
                    }
                )
                if self.state.chunk_delay_s:
                    time.sleep(self.state.chunk_delay_s)
            self._stream_event(
                {
                    "type": "speech.audio.done",
                    "usage": {"input_tokens": 4, "output_tokens": 360, "total_tokens": 364},
                }
            )
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.state.finish()
            self.close_connection = True
            return
        self._start_stream("audio/pcm")
        for _ in range(3):
            self.wfile.write(b"\x01\x00" * 240)
            self.wfile.flush()
            if self.state.chunk_delay_s:
                time.sleep(self.state.chunk_delay_s)
        self.state.finish()
        self.close_connection = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--chunk-delay-ms", type=float, default=0.0)
    parser.add_argument(
        "--dialect",
        default="vllm-omni",
        choices=DIALECTS,
        help="wire vocabulary to answer in, and to hold the client to",
    )
    arguments = parser.parse_args()
    server = ThreadingHTTPServer((arguments.host, arguments.port), Handler)
    server.state = State(  # type: ignore[attr-defined]
        arguments.log_path, arguments.chunk_delay_ms, arguments.dialect
    )
    if arguments.ready_file:
        arguments.ready_file.write_text(str(server.server_port))
    print(server.server_port, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
