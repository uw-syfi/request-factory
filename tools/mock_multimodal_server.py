"""CPU-only OpenAI mock for text, generated-image, and streaming-audio replay."""

from __future__ import annotations

import argparse
import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


class State:
    def __init__(self, log_path: Path | None, chunk_delay_ms: float) -> None:
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
            if self.log_path is not None:
                with self.log_path.open("a") as writer:
                    writer.write(json.dumps(value) + "\n")

    def finish(self) -> None:
        with self.lock:
            self.active -= 1


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "req-frontend-multimodal-mock/1"

    @property
    def state(self) -> State:
        return self.server.state  # type: ignore[attr-defined]

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _json_error(self, status: int, message: str) -> None:
        body = json.dumps({"error": message}).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length))
            if path in {"/images/generations", "/v1/images/generations"}:
                self._serve_image_generation(payload)
            elif path in {"/audio/speech", "/v1/audio/speech"}:
                self._serve_speech(payload)
            elif path in {"/chat/completions", "/v1/chat/completions"}:
                self._serve_chat(payload)
            else:
                self._json_error(404, "unsupported mock endpoint")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._json_error(400, str(error))

    def _serve_chat(self, payload: dict[str, object]) -> None:
        try:
            messages = payload["messages"]
            system_prompts = [
                message["content"] for message in messages if message.get("role") == "system"
            ]
            users = [message for message in messages if message.get("role") == "user"]
            if any(not isinstance(prompt, str) or not prompt for prompt in system_prompts):
                raise ValueError("system message content must be a non-empty string")
            if len(users) != 1:
                raise ValueError("mock requires exactly one user message")
            content = users[0]["content"]
            if not isinstance(content, list) or not content:
                raise ValueError("user message content must be a non-empty list")
            text_parts = 0
            media_parts = 0
            media_bytes = 0
            for part in content:
                kind = part.get("type")
                if kind == "text":
                    if not part.get("text"):
                        raise ValueError("empty text part")
                    text_parts += 1
                    continue
                key = kind
                if kind not in {"image_url", "audio_url", "video_url"}:
                    raise ValueError(f"unsupported content type {kind!r}")
                url = part[key]["url"]
                prefix, encoded = url.split(",", 1)
                if not prefix.startswith("data:") or ";base64" not in prefix:
                    raise ValueError("media URL must be an inline base64 data URL")
                decoded = base64.b64decode(encoded, validate=True)
                if not decoded:
                    raise ValueError("empty media payload")
                media_parts += 1
                media_bytes += len(decoded)
            modalities = payload.get("modalities")
            output_modality = modalities[0] if isinstance(modalities, list) else "text"
            if output_modality == "text":
                if media_parts == 0:
                    raise ValueError("text mock requires at least one media part")
                max_tokens = int(payload["max_tokens"])
                if max_tokens <= 0 or payload.get("stream") is not True:
                    raise ValueError("positive max_tokens and stream=true are required")
            else:
                max_tokens = int(payload.get("max_tokens", 0))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self._json_error(400, str(error))
            return

        self.state.begin(
            {
                "request_id": self.headers.get("x-request-id"),
                "text_parts": text_parts,
                "media_parts": media_parts,
                "media_bytes": media_bytes,
                "max_tokens": max_tokens,
                "max_output_tokens": payload.get("max_output_tokens"),
                "temperature": payload.get("temperature"),
                "thinker_temperature": payload.get("thinker_temperature"),
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
                audio = b"\x01\x00" * 240
                event = {
                    "modality": "audio",
                    "choices": [
                        {
                            "delta": {"content": base64.b64encode(audio).decode()},
                            "finish_reason": None,
                        }
                    ],
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
            event = {
                "choices": [
                    {
                        "delta": {"content": "pizza" if index == 0 else " token"},
                        "finish_reason": None,
                    }
                ]
            }
            self._stream_event(event)
            if self.state.chunk_delay_s:
                time.sleep(self.state.chunk_delay_s)
        usage = {
            "choices": [{"delta": {}, "finish_reason": "length"}],
            "usage": {
                "prompt_tokens": 256 + text_parts * 8,
                "completion_tokens": max_tokens,
                "total_tokens": 256 + text_parts * 8 + max_tokens,
            },
        }
        self.wfile.write(f"data: {json.dumps(usage)}\n\ndata: [DONE]\n\n".encode())
        self.wfile.flush()
        self.state.finish()
        self.close_connection = True

    def _serve_image_generation(self, payload: dict[str, object]) -> None:
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("image generation requires a prompt")
        self.state.begin(
            {
                "request_id": self.headers.get("x-request-id"),
                "text_parts": 1,
                "media_parts": 0,
                "media_bytes": 0,
                "max_tokens": 0,
                "output_modality": "image",
                "size": payload.get("size"),
                "num_inference_steps": payload.get("num_inference_steps"),
            }
        )
        self._write_json({"data": [{"b64_json": base64.b64encode(_PNG).decode()}]})
        self.state.finish()

    def _serve_speech(self, payload: dict[str, object]) -> None:
        text = payload.get("input")
        if not isinstance(text, str) or not text:
            raise ValueError("speech requires non-empty input")
        if payload.get("response_format") != "pcm":
            raise ValueError("mock speech requires response_format=pcm")
        self.state.begin(
            {
                "request_id": self.headers.get("x-request-id"),
                "text_parts": 1,
                "media_parts": 0,
                "media_bytes": 0,
                "max_tokens": payload.get("max_new_tokens", 0),
                "output_modality": "audio",
                "voice": payload.get("voice"),
                "system_prompts": [],
            }
        )
        self._start_stream("audio/pcm")
        for _ in range(3):
            self.wfile.write(b"\x01\x00" * 240)
            self.wfile.flush()
            if self.state.chunk_delay_s:
                time.sleep(self.state.chunk_delay_s)
        self.state.finish()
        self.close_connection = True

    def _start_stream(self, content_type: str) -> None:
        self.send_response(200)
        self.send_header("content-type", content_type)
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        self.end_headers()

    def _stream_event(self, event: dict[str, object]) -> None:
        self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
        self.wfile.flush()

    def _write_json(self, value: dict[str, object]) -> None:
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--chunk-delay-ms", type=float, default=0.0)
    arguments = parser.parse_args()
    server = ThreadingHTTPServer((arguments.host, arguments.port), Handler)
    server.state = State(arguments.log_path, arguments.chunk_delay_ms)  # type: ignore[attr-defined]
    if arguments.ready_file:
        arguments.ready_file.write_text(str(server.server_port))
    print(server.server_port, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
