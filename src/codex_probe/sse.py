"""Server-Sent Events parsing and stream reassembly.

The proxy relays raw bytes to Codex untouched. Separately, it keeps a copy of
those bytes, and after the stream ends this module turns them into one
complete response object for the log.

Two wire formats are supported:

* Chat Completions (``wire_api = "chat"``): a series of ``chat.completion.chunk``
  objects whose ``choices[i].delta`` carry pieces of ``content`` and
  ``tool_calls`` (tool-call arguments arrive as string fragments keyed by index),
  ending with ``data: [DONE]``.
* Responses API (``wire_api = "responses"``): typed events such as
  ``response.output_text.delta`` and ``response.function_call_arguments.delta``,
  ending with ``response.completed`` whose payload already holds the full
  response object.
"""
from __future__ import annotations

import json
from typing import Any


def parse_sse(raw: bytes) -> list[dict[str, Any]]:
    """Split a raw SSE byte stream into events.

    Returns a list of ``{"event": <name or None>, "data": <str>}``. Multi-line
    ``data:`` fields are joined with newlines, per the SSE spec. Comment lines
    (starting with ``:``) are ignored.
    """
    text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n")
    events: list[dict[str, Any]] = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        name: str | None = None
        data_lines: list[str] = []
        for line in block.split("\n"):
            if line.startswith(":"):
                continue
            field_name, _, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value
            if field_name == "event":
                name = value
            elif field_name == "data":
                data_lines.append(value)
        if data_lines or name:
            events.append({"event": name, "data": "\n".join(data_lines)})
    return events


def _load(data: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(data)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def reassemble_chat(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Rebuild a full ``chat.completion`` object from streamed chunks."""
    result: dict[str, Any] = {"object": "chat.completion", "choices": []}
    choices: dict[int, dict[str, Any]] = {}
    for ev in events:
        if ev["data"].strip() == "[DONE]":
            continue
        chunk = _load(ev["data"])
        if chunk is None:
            continue
        for key in ("id", "model", "created", "system_fingerprint"):
            if key in chunk:
                result[key] = chunk[key]
        if chunk.get("usage"):
            result["usage"] = chunk["usage"]
        for choice in chunk.get("choices", []):
            idx = choice.get("index", 0)
            slot = choices.setdefault(
                idx,
                {"index": idx, "message": {"role": "assistant", "content": None}, "finish_reason": None},
            )
            delta = choice.get("delta", {})
            msg = slot["message"]
            if delta.get("role"):
                msg["role"] = delta["role"]
            if delta.get("content"):
                msg["content"] = (msg["content"] or "") + delta["content"]
            if delta.get("reasoning_content"):
                msg["reasoning_content"] = msg.get("reasoning_content", "") + delta["reasoning_content"]
            for tc in delta.get("tool_calls") or []:
                calls = msg.setdefault("tool_calls", [])
                t_idx = tc.get("index", len(calls))
                while len(calls) <= t_idx:
                    calls.append({"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
                target = calls[t_idx]
                if tc.get("id"):
                    target["id"] = tc["id"]
                if tc.get("type"):
                    target["type"] = tc["type"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    target["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    target["function"]["arguments"] += fn["arguments"]
            if choice.get("finish_reason"):
                slot["finish_reason"] = choice["finish_reason"]
    result["choices"] = [choices[i] for i in sorted(choices)]
    return result


def reassemble_responses(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Rebuild a full Responses API object.

    Preferred path: the terminal ``response.completed`` (or ``.failed`` /
    ``.incomplete``) event already carries the whole response, so we use it.
    Fallback: if the stream was cut short, rebuild text and function calls from
    the delta events so the log is still useful, and mark it ``partial``.
    """
    final: dict[str, Any] | None = None
    text_parts: dict[str, str] = {}
    fn_args: dict[str, str] = {}
    items: dict[str, dict[str, Any]] = {}
    event_types: list[str] = []
    for ev in events:
        payload = _load(ev["data"])
        if payload is None:
            continue
        etype = payload.get("type") or ev.get("event") or ""
        event_types.append(etype)
        if etype in ("response.completed", "response.failed", "response.incomplete"):
            final = payload.get("response", payload)
        elif etype == "response.output_item.added" or etype == "response.output_item.done":
            item = payload.get("item", {})
            if item.get("id"):
                items[item["id"]] = item
        elif etype == "response.output_text.delta":
            key = payload.get("item_id", "text")
            text_parts[key] = text_parts.get(key, "") + payload.get("delta", "")
        elif etype == "response.function_call_arguments.delta":
            key = payload.get("item_id", "fn")
            fn_args[key] = fn_args.get(key, "") + payload.get("delta", "")
    if final is not None:
        return final
    output: list[dict[str, Any]] = []
    for item_id, item in items.items():
        item = dict(item)
        if item.get("type") == "function_call" and item_id in fn_args:
            item["arguments"] = fn_args[item_id]
        if item.get("type") == "message" and item_id in text_parts:
            item["content"] = [{"type": "output_text", "text": text_parts[item_id]}]
        output.append(item)
    for item_id, text in text_parts.items():
        if item_id not in items:
            output.append({"type": "message", "id": item_id, "content": [{"type": "output_text", "text": text}]})
    return {"object": "response", "status": "partial", "output": output, "event_types": event_types}


def reassemble(raw: bytes, wire_api: str) -> dict[str, Any]:
    """Pick the reassembler from the payload itself, falling back to ``wire_api``.

    Detecting by content (not only config) keeps logs correct even if a
    backend answers in the other format.
    """
    events = parse_sse(raw)
    for ev in events:
        payload = _load(ev["data"])
        if payload is None:
            continue
        if payload.get("object") == "chat.completion.chunk":
            return reassemble_chat(events)
        if str(payload.get("type", "")).startswith("response."):
            return reassemble_responses(events)
        break
    return reassemble_chat(events) if wire_api == "chat" else reassemble_responses(events)
