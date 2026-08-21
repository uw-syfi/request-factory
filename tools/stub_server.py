#!/usr/bin/env python3
"""A vLLM-shaped SSE stub, for measuring the client against a server that isn't there.

It exists to answer questions a real server cannot. Does recording the per-event
timeline slow down submission? A real server's own jitter is far larger than the
effect being looked for, so the reference has to be a server whose reply timing
is fixed by construction. And does the adaptive sweep find the knee? Only a
server whose saturation point is known by arithmetic can answer that -- see
`--capacity`.

Speaks just enough of the `vllm-tokens` protocol: `/inference/v1/generate`
streaming one token id per SSE chunk, a terminal usage block with prompt-token
details (so the prefix-cache preflight passes), and `[DONE]`.

    uv run python tools/stub_server.py --port 8123 \
        [--prefill-delay-ms 0] [--chunk-delay-ms 0] [--capacity 0]
"""

from __future__ import annotations

import argparse
import asyncio
import json


class Stub:
    def __init__(
        self,
        chunk_delay_ms: float,
        prefill_delay_ms: float,
        tokens_per_chunk: int,
        capacity: int,
        protocol: str,
        sse_space: bool,
    ) -> None:
        self.chunk_delay_s = chunk_delay_ms / 1000.0
        # Separate from the inter-chunk gap, and paid once before the first
        # chunk. That makes the client's TTFT and its TPOT independently
        # checkable: with one knob they are the same number, and a client that
        # confused the two would still agree with the server.
        self.prefill_delay_s = prefill_delay_ms / 1000.0
        self.tokens_per_chunk = tokens_per_chunk
        self.protocol = protocol
        self.sse_space = sse_space
        # A server with no capacity limit never saturates, so a sweep against it
        # has no knee to find. With one, the knee is arithmetic:
        #   capacity / (output_len * chunk_delay_s) requests per second.
        self.slots = asyncio.Semaphore(capacity) if capacity > 0 else None
        # A real prefix cache, not a memo of whole prompts. The difference
        # matters: the client's central claim is that round k+1 reuses round k's
        # conversation, and a server that only recognizes an exact repeat cannot
        # tell a correct client from one that rebuilds the prefix differently.
        #
        # Stored as rolling hashes of every prefix rather than a trie of ids.
        # The set is then prefix-closed by construction, so the longest hit can
        # be found by binary search, and one seen prefix costs one integer no
        # matter how many prompts share it.
        self.seen_prefixes: set[int] = set()

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
        if self.slots is None:
            await self.serve(writer, body)
            return
        # Held across the whole response, so an over-offered request queues
        # rather than being served slower -- which is what a real scheduler with
        # a fixed batch does, and what makes the knee sharp.
        async with self.slots:
            await self.serve(writer, body)

    async def serve(self, writer: asyncio.StreamWriter, body: bytes) -> None:
        request = json.loads(body) if body else {}
        sampling = request.get("sampling_params", {})
        if self.protocol == "vllm":
            prompt_ids = request.get("token_ids") or []
            max_tokens = int(sampling.get("max_tokens", 1))
        elif self.protocol == "sglang":
            prompt_ids = request.get("input_ids") or []
            max_tokens = int(sampling.get("max_new_tokens", 1))
        else:
            prompt_ids = request.get("prompt") or []
            max_tokens = int(request.get("max_tokens", 1))

        cached = self.remember(prompt_ids)
        # The two-call probe is a capability check, not workload warmup. Once
        # its second call proves a full hit, leave the cache empty so the first
        # measured request cannot inherit probe content.
        reset_after_response = (
            request.get("request_id") == "req-frontend-prefix-cache-preflight"
            and cached == len(prompt_ids)
        )

        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/event-stream\r\n"
            b"Cache-Control: no-cache\r\n"
            b"Connection: keep-alive\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n"
        )

        # Delays are paid *before* each chunk, which is what a real server does
        # and what makes the arithmetic checkable end to end:
        #   ttft  = prefill_delay
        #   gap   = chunk_delay
        #   total = prefill_delay + (chunks - 1) * chunk_delay
        # Sleeping after the last chunk instead would put one extra gap in the
        # total that no client could attribute to anything.
        if self.prefill_delay_s:
            await asyncio.sleep(self.prefill_delay_s)
        emitted = 0
        generated_ids: list[int] = []
        while emitted < max_tokens:
            if emitted and self.chunk_delay_s:
                await asyncio.sleep(self.chunk_delay_s)
            count = min(self.tokens_per_chunk, max_tokens - emitted)
            ids = list(range(emitted, emitted + count))
            if self.protocol == "sglang":
                event = {
                    "output_ids": ids,
                    "meta_info": {
                        "prompt_tokens": len(prompt_ids),
                        "completion_tokens": emitted + count,
                        "cached_tokens": cached,
                    },
                }
            else:
                choice = {"index": 0, "token_ids": ids}
                if self.protocol == "openai":
                    choice["text"] = "x" * count
                event = {"choices": [choice]}
            await self.send_event(writer, event)
            generated_ids.extend(ids)
            emitted += count

        # A serving engine retains the decode KV as well as the prefill KV. A
        # later conversation round therefore reuses the previous prompt *and*
        # its generated answer. Recording only prompt_ids would make the stub
        # report a short hit even when the client carried the exact output ids
        # it was sent.
        if not reset_after_response:
            self.store(prompt_ids + generated_ids)

        if self.protocol == "sglang":
            terminal = {
                "output_ids": [],
                "meta_info": {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": emitted,
                    "cached_tokens": cached,
                    "finish_reason": {"type": "length"},
                },
            }
        else:
            terminal = {
                "choices": [{"index": 0, "token_ids": [], "finish_reason": "length"}],
                "usage": {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": emitted,
                    "total_tokens": len(prompt_ids) + emitted,
                    # Cached detail must be present for the preflight to accept
                    # this server as one that reports prefix-cache hits at all.
                    "prompt_tokens_details": {"cached_tokens": cached},
                },
            }
        await self.send_event(writer, terminal)
        await self.send_chunk(writer, b"data: [DONE]\n\n")
        await self.send_chunk(writer, b"")
        await writer.drain()
        if reset_after_response:
            self.seen_prefixes.clear()

    # Mersenne prime modulus and a fixed base: a 61-bit rolling hash. A
    # collision would over-report a cache hit, which is why the modulus is this
    # large -- at trace scale the probability is far below the rate at which a
    # real server's cache is evicted underneath a measurement anyway.
    MODULUS = (1 << 61) - 1
    BASE = 0x1F1F_1F1F_1F1F_1F1F

    def prefix_hashes(self, prompt_ids: list[int]) -> list[int]:
        """Hash of every prefix, in one pass."""
        rolling = 0
        hashes = []
        for token_id in prompt_ids:
            rolling = (rolling * self.BASE + token_id + 1) % self.MODULUS
            hashes.append(rolling)
        return hashes

    def remember(self, prompt_ids: list[int]) -> int:
        """Longest prefix of this prompt already seen, then record the whole thing.

        Binary search is valid because the set is prefix-closed: if a prefix of
        length L was inserted, so was every shorter one. So the matched lengths
        are exactly 0..cached, and the boundary can be found in log time.
        """
        hashes = self.prefix_hashes(prompt_ids)
        low, high = 0, len(hashes)
        while low < high:
            middle = (low + high + 1) // 2
            if hashes[middle - 1] in self.seen_prefixes:
                low = middle
            else:
                high = middle - 1
        self.seen_prefixes.update(hashes)
        return low

    def store(self, token_ids: list[int]) -> None:
        """Record every prefix of a sequence whose KV is now resident."""
        self.seen_prefixes.update(self.prefix_hashes(token_ids))

    async def send_event(self, writer: asyncio.StreamWriter, payload: dict) -> None:
        separator = " " if self.sse_space else ""
        await self.send_chunk(writer, f"data:{separator}{json.dumps(payload)}\n\n".encode())

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
    parser.add_argument(
        "--prefill-delay-ms",
        type=float,
        default=0.0,
        help="Wall time before the first chunk. The ground truth a client's TTFT is checked against.",
    )
    parser.add_argument("--tokens-per-chunk", type=int, default=1)
    parser.add_argument(
        "--protocol",
        choices=("vllm", "sglang", "openai"),
        default="vllm",
        help="Request and streamed-response vocabulary to enforce.",
    )
    parser.add_argument(
        "--sse-no-space",
        action="store_true",
        help="Emit the standards-compliant data:VALUE form without the optional space.",
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=0,
        help=(
            "Concurrent requests served; further ones queue. 0 is unlimited, which "
            "never saturates. With a limit the capacity is capacity / occupancy "
            "requests per second, where occupancy is prefill_delay + (chunks - 1) * "
            "chunk_delay in seconds."
        ),
    )
    args = parser.parse_args()

    stub = Stub(
        args.chunk_delay_ms,
        args.prefill_delay_ms,
        args.tokens_per_chunk,
        args.capacity,
        args.protocol,
        not args.sse_no_space,
    )
    server = await asyncio.start_server(stub.handle, args.host, args.port)
    print(f"stub listening on http://{args.host}:{args.port}", flush=True)
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
