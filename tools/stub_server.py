#!/usr/bin/env python3
"""A vLLM-shaped SSE stub, for measuring the client against a server that isn't there.

It exists to answer one question that a real server cannot: does recording the
per-event timeline slow down submission? A real server's own jitter is far larger
than the effect being looked for, so the reference has to be a server whose reply
timing is fixed by construction.

Speaks just enough of the `vllm-tokens` protocol: `/inference/v1/generate`
streaming one token id per SSE chunk, a terminal usage block with prompt-token
details (so the prefix-cache preflight passes), and `[DONE]`.

    uv run python tools/stub_server.py --port 8123 [--chunk-delay-ms 0]
"""

from __future__ import annotations

import argparse
import asyncio
import json


class Stub:
    def __init__(self, chunk_delay_ms: float, tokens_per_chunk: int) -> None:
        self.chunk_delay_s = chunk_delay_ms / 1000.0
        self.tokens_per_chunk = tokens_per_chunk
        # The preflight sends one prompt twice and requires the second reply to
        # report cached tokens. Remembering what was asked is the whole feature.
        self.seen_prompts: set[int] = set()

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                request_line = await reader.readline()
                if not request_line:
                    return
                headers: dict[str, str] = {}
                while True:
                    line = await reader.readline()
                    if line in (b"\r\n", b"\n", b""):
                        break
                    name, _, value = line.decode("latin-1").partition(":")
                    headers[name.strip().lower()] = value.strip()
                body = b""
                remaining = int(headers.get("content-length", "0"))
                if remaining:
                    body = await reader.readexactly(remaining)
                await self.reply(writer, body)
        except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
            pass
        finally:
            writer.close()

    async def reply(self, writer: asyncio.StreamWriter, body: bytes) -> None:
        request = json.loads(body) if body else {}
        prompt_ids = request.get("token_ids") or []
        max_tokens = int(request.get("sampling_params", {}).get("max_tokens", 1))

        fingerprint = hash(tuple(prompt_ids))
        cached = len(prompt_ids) if fingerprint in self.seen_prompts else 0
        self.seen_prompts.add(fingerprint)

        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/event-stream\r\n"
            b"Cache-Control: no-cache\r\n"
            b"Connection: keep-alive\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n"
        )

        emitted = 0
        while emitted < max_tokens:
            count = min(self.tokens_per_chunk, max_tokens - emitted)
            ids = list(range(emitted, emitted + count))
            await self.send_event(writer, {"choices": [{"index": 0, "token_ids": ids}]})
            emitted += count
            if self.chunk_delay_s:
                await asyncio.sleep(self.chunk_delay_s)

        await self.send_event(
            writer,
            {
                "choices": [{"index": 0, "token_ids": [], "finish_reason": "length"}],
                "usage": {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": emitted,
                    "total_tokens": len(prompt_ids) + emitted,
                    # Cached detail must be present for the preflight to accept
                    # this server as one that reports prefix-cache hits at all.
                    "prompt_tokens_details": {"cached_tokens": cached},
                },
            },
        )
        await self.send_chunk(writer, b"data: [DONE]\n\n")
        await self.send_chunk(writer, b"")
        await writer.drain()

    async def send_event(self, writer: asyncio.StreamWriter, payload: dict) -> None:
        await self.send_chunk(writer, f"data: {json.dumps(payload)}\n\n".encode())

    async def send_chunk(self, writer: asyncio.StreamWriter, payload: bytes) -> None:
        writer.write(f"{len(payload):X}\r\n".encode() + payload + b"\r\n")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--chunk-delay-ms",
        type=float,
        default=0.0,
        help="Wall time between streamed chunks. Zero replies as fast as the socket allows.",
    )
    parser.add_argument("--tokens-per-chunk", type=int, default=1)
    args = parser.parse_args()

    stub = Stub(args.chunk_delay_ms, args.tokens_per_chunk)
    server = await asyncio.start_server(stub.handle, args.host, args.port)
    print(f"stub listening on http://{args.host}:{args.port}", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
