"""Unit tests for trace analysis, using hand-built call records."""
from __future__ import annotations

import json

import pytest

from codex_probe import analyze_calls, analyze_session, markdown_report, validate_call
from codex_probe.analysis import extract_output, offered_tools
from codex_probe.cli import main

TOOLS = [{"type": "function", "name": "exec_command", "parameters": {}},
         {"type": "web_search"}]
SYS = {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "You are Codex."}]}
TASK = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "write fizzbuzz"}]}


def fn_call(args: str, name: str = "exec_command", cid: str = "c1") -> dict:
    return {"type": "function_call", "name": name, "arguments": args, "call_id": cid}


def record(i: int, history: list, output: list, usage: dict | None = None, *, status: int = 200,
           ttfb: float = 100.0, latency: float = 1100.0, model: str = "qwen2.5-coder:7b") -> dict:
    resp = {"status": "completed", "model": model, "output": output}
    if usage is not None:
        resp["usage"] = usage
    entry = {
        "session_id": "s", "call_index": i, "timestamp": "t", "backend_name": "local-qwen",
        "backend_base_url": "http://x/v1", "wire_api": "responses", "method": "POST",
        "path": "/v1/responses", "upstream_url": "http://x/v1/responses", "request_headers": {},
        "request_body": {"model": model, "stream": True, "input": history, "tools": TOOLS},
        "status": status, "streamed": True, "response_body": resp, "ttfb_ms": ttfb, "latency_ms": latency,
    }
    validate_call(entry)
    return entry


def session() -> list[dict]:
    """Three-turn agent loop: act, repeat the same action, finish with a message."""
    call1 = fn_call('{"cmd": "python test.py"}')
    out1 = {"type": "function_call_output", "call_id": "c1", "output": "ok"}
    call2 = fn_call('{ "cmd":"python test.py" }', cid="c2")  # same action, different whitespace
    out2 = {"type": "function_call_output", "call_id": "c2", "output": "ok"}
    final = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Done."}]}
    h1 = [SYS, TASK]
    h2 = h1 + [call1, out1]
    h3 = h2 + [call2, out2]
    return [
        record(0, h1, [call1], {"input_tokens": 1000, "output_tokens": 50,
                                "input_tokens_details": {"cached_tokens": 0}}),
        record(1, h2, [call2], {"input_tokens": 1100, "output_tokens": 50,
                                "input_tokens_details": {"cached_tokens": 1000}}),
        record(2, h3, [final], {"input_tokens": 1200, "output_tokens": 100,
                                "input_tokens_details": {"cached_tokens": 1100}}),
    ]


def test_offered_tools_handles_named_builtin_and_chat_shapes():
    chat = [{"type": "function", "function": {"name": "shell"}}]
    assert offered_tools({"tools": TOOLS}) == {"exec_command", "web_search"}
    assert offered_tools({"tools": chat}) == {"shell"}
    assert offered_tools({}) == set()


def test_extract_output_flags_malformed_and_unknown():
    resp = {"output": [fn_call('{"cmd": "ls"'), fn_call('{"x": 1}', name="rm_rf"),
                       {"type": "custom_tool_call", "name": "apply_patch", "input": "*** Begin Patch"},
                       {"type": "reasoning", "summary": []}]}
    calls, text = extract_output(resp, {"exec_command", "apply_patch"})
    assert [(c.name, c.valid_arguments, c.known_tool) for c in calls] == [
        ("exec_command", False, True), ("rm_rf", True, False), ("apply_patch", True, True)]
    assert text == ""


def test_extract_output_chat_format():
    resp = {"choices": [{"message": {"content": "Running.", "tool_calls": [
        {"function": {"name": "shell", "arguments": "[\"ls\"]"}}]}}]}  # JSON, but not an object
    calls, text = extract_output(resp, {"shell"})
    assert text == "Running." and calls[0].name == "shell" and calls[0].valid_arguments is False


def test_session_metrics():
    s = analyze_calls(session())
    assert (s.calls, s.failed_calls, s.tool_calls) == (3, 0, 2)
    assert (s.total_input_tokens, s.total_output_tokens, s.tokens_estimated) == (3300, 200, False)
    assert (s.first_input_tokens, s.last_input_tokens) == (1000, 1200)
    assert s.repeated_tool_call_rate == pytest.approx(0.5)      # 2nd call repeats the 1st
    assert s.malformed_tool_call_rate == 0 and s.unknown_tool_call_rate == 0
    assert s.cached_token_share == pytest.approx(2100 / 3300)
    assert s.ended_with_message is True and s.text_only_turns == 1
    assert s.tool_call_mix == {"exec_command": 2}
    assert s.served_models == ["qwen2.5-coder:7b"]
    # turn 2 re-sends all of turn 1's history, so its re-sent share is between 0 and 1
    r = [m.resent_ratio for m in s.per_call]
    assert r[0] is None and 0 < r[1] < 1 and 0 < r[2] < 1
    # decode throughput = output tokens / (latency - ttfb)
    assert s.per_call[0].decode_tok_per_s == pytest.approx(50 / 1.0)
    assert s.latency_p50_ms == 1100.0 and s.latency_p95_ms == 1100.0


def test_failed_calls_are_excluded_from_token_totals():
    calls = session()[:1]
    bad = record(1, [SYS], [], status=502)
    bad.pop("response_body")
    bad["error"] = "ClientConnectorError: refused"
    s = analyze_calls(calls + [bad])
    assert s.failed_calls == 1 and s.total_input_tokens == 1000
    assert s.ended_with_message is False  # last successful turn was a tool call


def test_tokens_estimated_without_usage():
    s = analyze_calls([record(0, [SYS, TASK], [fn_call('{"cmd": "ls"}')], usage=None)])
    assert s.tokens_estimated and s.total_input_tokens > 0 and s.total_output_tokens > 0
    assert s.cached_token_share is None


def test_empty_session():
    s = analyze_calls([])
    assert s.calls == 0 and s.malformed_tool_call_rate is None and s.latency_p50_ms is None
    assert "LLM calls" in markdown_report([s], ["empty"])


def test_report_and_cli(tmp_path, capsys):
    log = tmp_path / "s.jsonl"
    log.write_text("".join(json.dumps(c) + "\n" for c in session()))
    s = analyze_session(log)
    md = markdown_report([s, s], ["openai", "qwen"])
    assert "| Metric | openai | qwen |" in md and "1,000 → 1,200" in md and "1 repeated" in md

    out, js = tmp_path / "r.md", tmp_path / "r.json"
    main(["analyze", str(log), str(log), "--names", "a", "b", "--out", str(out), "--json", str(js)])
    assert "| Metric | a | b |" in out.read_text()
    data = json.loads(js.read_text())
    assert len(data) == 2 and data[0]["calls"] == 3 and len(data[0]["per_call"]) == 3

    with pytest.raises(SystemExit):
        main(["analyze", str(log), "--names", "a", "b"])
