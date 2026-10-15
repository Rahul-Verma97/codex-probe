"""JSON Schema for one logged call, plus helpers to load and validate session logs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

CALL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "session_id", "call_index", "timestamp", "backend_name", "backend_base_url", "wire_api",
        "method", "path", "upstream_url", "request_headers", "request_body", "status",
        "streamed", "latency_ms",
    ],
    "properties": {
        "session_id": {"type": "string"},
        "call_index": {"type": "integer", "minimum": 0},
        "timestamp": {"type": "string"},
        "backend_name": {"type": "string"},
        "backend_base_url": {"type": "string"},
        "wire_api": {"enum": ["responses", "chat"]},
        "method": {"type": "string"},
        "path": {"type": "string"},
        "upstream_url": {"type": "string"},
        "request_headers": {"type": "object"},
        "request_body": {},
        "status": {"type": "integer"},
        "streamed": {"type": "boolean"},
        "response_headers": {"type": "object"},
        "response_body": {},
        "raw_stream": {"type": "string"},
        "ttfb_ms": {"type": "number"},
        "latency_ms": {"type": "number"},
        "error": {"type": "string"},
        "client_disconnected": {"type": "boolean"},
    },
}


def validate_call(entry: dict[str, Any]) -> None:
    """Raise ``jsonschema.ValidationError`` if ``entry`` is not a well-formed call record."""
    jsonschema.validate(entry, CALL_SCHEMA)


def load_session(path: str | Path) -> list[dict[str, Any]]:
    """Read a session JSONL file and validate every line."""
    calls = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            validate_call(entry)
            calls.append(entry)
    return calls
