"""End-to-end tests: client -> ProxyRecorder -> mock backend."""
from __future__ import annotations

import asyncio
import json
import threading

import aiohttp
import pytest
from aiohttp import web

from codex_probe import ProxyRecorder, load_session, validate_call

from mock_backend import MockBackend, chat_stream_bytes, responses_stream_bytes


@pytest.fixture()
def backend():
    """Run the mock backend on a free port in a background thread."""
    mock = MockBackend()
    loop = asyncio.new_event_loop()
    state = {}

    async def boot():
        runner = web.AppRunner(mock.app())
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        state["runner"] = runner
        state["port"] = site._server.sockets[0].getsockname()[1]

    def run():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(boot())
        state["ready"] = True
        loop.run_forever()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    while "ready" not in state:
        pass
    mock.url = f"http://127.0.0.1:{state['port']}/v1"
    yield mock
    asyncio.run_coroutine_threadsafe(state["runner"].cleanup(), loop).result()
    loop.call_soon_threadsafe(loop.stop)


@pytest.fixture()
def recorder(backend, tmp_path, monkeypatch):
    monkeypatch.setenv("PROBE_TEST_KEY", "sk-test-1234")
    rec = ProxyRecorder({
        "backend_base_url": backend.url, "backend_name": "mock", "wire_api": "chat",
        "api_key_env": "PROBE_TEST_KEY", "listen_port": 0, "log_dir": str(tmp_path),
    })
    endpoint = rec.start()
    rec.endpoint = endpoint
    yield rec
    rec.stop()


async def _post(url: str, payload: dict) -> tuple[int, bytes, str, list[bytes]]:
    """POST and also collect chunks as they arrive, to check the stream stays live."""
    async with aiohttp.ClientSession() as s:
        async with s.post(url, json=payload) as r:
            chunks = [c async for c in r.content.iter_any()]
            return r.status, b"".join(chunks), r.headers.get("Content-Type", ""), chunks


async def test_non_streaming_round_trip(recorder, backend):
    payload = {"model": "qwen", "messages": [{"role": "user", "content": "hi"}]}
    status, body, _, _ = await _post(recorder.endpoint + "/chat/completions", payload)
    assert status == 200
    assert json.loads(body)["choices"][0]["message"]["content"] == "hello"
    # Request reached the backend byte-for-byte.
    assert json.loads(backend.received[0]["body"]) == payload
    call = recorder.calls()[0]
    assert call["request_body"] == payload
    assert call["response_body"]["choices"][0]["message"]["content"] == "hello"
    assert call["streamed"] is False


async def test_chat_stream_passthrough_and_reassembly(recorder):
    payload = {"model": "qwen", "stream": True, "messages": [{"role": "user", "content": "fix it"}],
               "tools": [{"type": "function", "function": {"name": "shell", "parameters": {}}}]}
    status, body, ctype, chunks = await _post(recorder.endpoint + "/chat/completions", payload)
    assert status == 200 and "text/event-stream" in ctype
    assert body == chat_stream_bytes()          # client sees exactly the upstream bytes
    assert len(chunks) > 1                       # and it arrived as a live stream
    call = recorder.calls()[0]
    msg = call["response_body"]["choices"][0]["message"]
    assert msg["content"] == "Run tests."
    assert msg["tool_calls"][0]["function"] == {"name": "shell", "arguments": "{\"cmd\": [\"pytest\"]}"}
    assert call["response_body"]["choices"][0]["finish_reason"] == "tool_calls"
    assert call["request_body"]["tools"] == payload["tools"]  # full tool schema captured
    assert call["raw_stream"] == chat_stream_bytes().decode()


async def test_responses_stream_reassembly(recorder):
    payload = {"model": "gpt-5", "stream": True, "instructions": "You are Codex.",
               "input": [{"role": "user", "content": "say hello"}]}
    status, body, _, _ = await _post(recorder.endpoint + "/responses", payload)
    assert status == 200 and body == responses_stream_bytes()
    resp = recorder.calls()[0]["response_body"]
    assert resp["status"] == "completed"
    assert resp["output"][0]["content"][0]["text"] == "Hello"


async def test_auth_swap_and_redaction(recorder, backend):
    await _post(recorder.endpoint + "/chat/completions", {"model": "q", "messages": []})
    assert backend.received[0]["headers"]["Authorization"] == "Bearer sk-test-1234"
    logged = recorder.calls()[0]["request_headers"]["Authorization"]
    assert "sk-test" not in logged and logged.endswith("1234")


async def test_session_log_schema_and_order(recorder):
    for i in range(3):
        await _post(recorder.endpoint + "/chat/completions", {"model": "q", "messages": [], "n": i})
    calls = load_session(recorder.log_path)       # validates every line
    assert [c["call_index"] for c in calls] == [0, 1, 2]
    assert [c["request_body"]["n"] for c in calls] == [0, 1, 2]
    assert all(c["backend_name"] == "mock" for c in calls)


async def test_new_session_rotates_file(recorder):
    await _post(recorder.endpoint + "/chat/completions", {"model": "q", "messages": []})
    first = recorder.log_path
    second = recorder.new_session("run2")
    await _post(recorder.endpoint + "/chat/completions", {"model": "q", "messages": []})
    assert first != second
    assert len(load_session(first)) == 1 and len(load_session(second)) == 1


async def test_upstream_down_returns_502_and_logs(tmp_path):
    rec = ProxyRecorder({"backend_base_url": "http://127.0.0.1:9/v1", "listen_port": 0,
                         "log_dir": str(tmp_path), "request_timeout_s": 3})
    endpoint = rec.start()
    try:
        status, _, _, _ = await _post(endpoint + "/chat/completions", {"model": "q", "messages": []})
        assert status == 502
        call = rec.calls()[0]
        validate_call(call)
        assert "error" in call
    finally:
        rec.stop()


async def test_client_disconnect_mid_stream_still_logs_full_response(recorder):
    payload = {"model": "q", "stream": True, "messages": []}
    async with aiohttp.ClientSession() as s:
        async with s.post(recorder.endpoint + "/chat/completions", json=payload) as r:
            await r.content.read(10)  # read a little, then hang up
    for _ in range(100):
        if recorder.calls():
            break
        await asyncio.sleep(0.05)
    call = recorder.calls()[0]
    assert call["raw_stream"] == chat_stream_bytes().decode()
    assert call["response_body"]["choices"][0]["message"]["content"] == "Run tests."
