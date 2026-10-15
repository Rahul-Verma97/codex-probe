"""A tiny OpenAI-compatible mock server used by the tests.

It serves both wire formats, streaming and non-streaming, with fixed payloads
so the tests can check byte-for-byte passthrough and reassembly.
"""
from __future__ import annotations

import asyncio
import json

from aiohttp import web

CHAT_FULL = {
    "id": "chatcmpl-1", "object": "chat.completion", "model": "mock-qwen",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
}

CHAT_CHUNKS = [
    {"id": "c1", "object": "chat.completion.chunk", "model": "mock-qwen",
     "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Run "}}]},
    {"id": "c1", "object": "chat.completion.chunk", "model": "mock-qwen",
     "choices": [{"index": 0, "delta": {"content": "tests."}}]},
    {"id": "c1", "object": "chat.completion.chunk", "model": "mock-qwen",
     "choices": [{"index": 0, "delta": {"tool_calls": [
         {"index": 0, "id": "call_1", "type": "function", "function": {"name": "shell", "arguments": "{\"cmd\": "}}]}}]},
    {"id": "c1", "object": "chat.completion.chunk", "model": "mock-qwen",
     "choices": [{"index": 0, "delta": {"tool_calls": [
         {"index": 0, "function": {"arguments": "[\"pytest\"]}"}}]}, "finish_reason": "tool_calls"}]},
]

_MSG_DONE = {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
             "content": [{"type": "output_text", "text": "Hello", "annotations": []}]}
RESPONSES_EVENTS = [
    ("response.created", {"type": "response.created", "sequence_number": 0,
                          "response": {"id": "resp_1", "object": "response", "status": "in_progress", "output": []}}),
    ("response.output_item.added", {"type": "response.output_item.added", "sequence_number": 1, "output_index": 0,
                                    "item": {"type": "message", "id": "msg_1", "role": "assistant",
                                             "status": "in_progress", "content": []}}),
    ("response.content_part.added", {"type": "response.content_part.added", "sequence_number": 2, "item_id": "msg_1",
                                     "output_index": 0, "content_index": 0,
                                     "part": {"type": "output_text", "text": "", "annotations": []}}),
    ("response.output_text.delta", {"type": "response.output_text.delta", "sequence_number": 3, "item_id": "msg_1",
                                    "output_index": 0, "content_index": 0, "delta": "Hel"}),
    ("response.output_text.delta", {"type": "response.output_text.delta", "sequence_number": 4, "item_id": "msg_1",
                                    "output_index": 0, "content_index": 0, "delta": "lo"}),
    ("response.output_text.done", {"type": "response.output_text.done", "sequence_number": 5, "item_id": "msg_1",
                                   "output_index": 0, "content_index": 0, "text": "Hello"}),
    ("response.output_item.done", {"type": "response.output_item.done", "sequence_number": 6, "output_index": 0,
                                   "item": _MSG_DONE}),
    ("response.completed", {"type": "response.completed", "sequence_number": 7, "response": {
        "id": "resp_1", "object": "response", "status": "completed", "model": "mock-gpt", "output": [_MSG_DONE],
        "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}}}),
]


def chat_stream_bytes() -> bytes:
    return b"".join(f"data: {json.dumps(c)}\n\n".encode() for c in CHAT_CHUNKS) + b"data: [DONE]\n\n"


def responses_stream_bytes() -> bytes:
    return b"".join(f"event: {n}\ndata: {json.dumps(p)}\n\n".encode() for n, p in RESPONSES_EVENTS)


class MockBackend:
    """Records what it received so tests can compare with what Codex sent."""

    def __init__(self) -> None:
        self.received: list[dict] = []

    async def _sse(self, request: web.Request, payload: bytes) -> web.StreamResponse:
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        # Send in small pieces with pauses to imitate a real token stream.
        for i in range(0, len(payload), 37):
            await resp.write(payload[i:i + 37])
            await asyncio.sleep(0.005)
        await resp.write_eof()
        return resp

    async def handle(self, request: web.Request) -> web.StreamResponse:
        body = await request.read()
        self.received.append({"path": request.path, "headers": dict(request.headers), "body": body})
        data = json.loads(body or b"{}")
        if request.path.endswith("/chat/completions"):
            if data.get("stream"):
                return await self._sse(request, chat_stream_bytes())
            return web.json_response(CHAT_FULL)
        if request.path.endswith("/responses"):
            return await self._sse(request, responses_stream_bytes())
        if request.path.endswith("/models"):
            return web.json_response({"data": [{"id": "mock-qwen"}]})
        return web.json_response({"error": "not found"}, status=404)

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", self.handle)
        return app
