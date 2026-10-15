"""Unit tests for SSE parsing, reassembly edge cases and config validation."""
from __future__ import annotations

import pytest

from codex_probe import ProbeConfig
from codex_probe.sse import parse_sse, reassemble

from mock_backend import chat_stream_bytes, responses_stream_bytes


def test_parse_sse_handles_crlf_comments_and_multiline():
    raw = b": keep-alive\r\n\r\nevent: x\r\ndata: line1\r\ndata: line2\r\n\r\n"
    events = parse_sse(raw)
    assert events == [{"event": "x", "data": "line1\nline2"}]


def test_reassembly_is_independent_of_chunk_boundaries():
    # The proxy reassembles from the concatenated buffer, so splitting the
    # stream anywhere (even mid-JSON) must give the same result.
    whole = chat_stream_bytes()
    assert reassemble(whole, "chat") == reassemble(whole[:13] + whole[13:], "chat")


def test_format_detected_from_payload_not_config():
    # A chat stream logged under wire_api="responses" is still parsed as chat.
    out = reassemble(chat_stream_bytes(), "responses")
    assert out["object"] == "chat.completion"
    out = reassemble(responses_stream_bytes(), "chat")
    assert out["status"] == "completed"


def test_truncated_responses_stream_falls_back_to_partial():
    raw = responses_stream_bytes().split(b"event: response.completed")[0]
    out = reassemble(raw, "responses")
    assert out["status"] == "partial"
    assert out["output"][0]["content"][0]["text"] == "Hello"


def test_parallel_tool_calls_are_kept_separate():
    raw = (
        b'data: {"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":['
        b'{"index":0,"id":"a","function":{"name":"read","arguments":"{}"}},'
        b'{"index":1,"id":"b","function":{"name":"write","arguments":"{\\"p\\":1}"}}]}}]}\n\n'
        b"data: [DONE]\n\n"
    )
    calls = reassemble(raw, "chat")["choices"][0]["message"]["tool_calls"]
    assert [c["function"]["name"] for c in calls] == ["read", "write"]
    assert calls[1]["function"]["arguments"] == '{"p":1}'


@pytest.mark.parametrize("bad", [
    {"backend_base_url": "api.openai.com"},
    {"backend_base_url": "http://x/v1", "wire_api": "grpc"},
    {"backend_base_url": "http://x/v1", "listen_port": 70000},
    {"backend_base_url": "http://x/v1", "typo_key": 1},
])
def test_config_validation(bad):
    with pytest.raises(ValueError):
        ProbeConfig.from_any(bad)


def test_config_strips_trailing_slash():
    assert ProbeConfig(backend_base_url="http://x/v1/").backend_base_url == "http://x/v1"
