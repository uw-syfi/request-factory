"""CPU-only OpenAI chat mock that validates and streams multimodal requests."""

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
        if urlparse(self.path).path not in {
            "/chat/completions",
            "/v1/chat/completions",
        }:
            self._json_error(404, "expected /v1/chat/completions")
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length))
            messages = payload["messages"]
            content = messages[0]["content"]
            if not isinstance(content, list) or not content:
                raise ValueError("messages[0].content must be a non-empty list")
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
            if media_parts == 0:
                raise ValueError("mock requires at least one media part")
            max_tokens = int(payload["max_tokens"])
            if max_tokens <= 0 or payload.get("stream") is not True:
                raise ValueError("positive max_tokens and stream=true are required")
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
            }
        )
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        self.end_headers()
        for index in range(max_tokens):
            event = {
                "choices": [
                    {
                        "delta": {"content": "pizza" if index == 0 else " token"},
                        "finish_reason": None,
                    }
                ]
            }
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            self.wfile.flush()
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
