"""Trace analysis: turn a recorded session into per-call and per-session metrics.

The recorder answers *what* Codex sent and received. This module answers the
questions that matter when the backend is swapped for a smaller open model:

* **Cost.** Input/output tokens per turn, total tokens, and how fast the prompt
  grows. Codex is stateless per request (``store: false``), so every turn
  re-sends the whole history; ``resent_ratio`` measures how much of each
  request is an exact repeat of the previous request's prefix. This is the
  share that prefix caching or trajectory reduction could save.
* **Latency.** Time to first byte, end-to-end latency, and decode throughput.
* **Agent behaviour.** Tool calls per turn, malformed tool-call arguments,
  calls to tools that were never offered, identical actions repeated (a loop
  signal), turns that answered in plain text instead of acting, and whether the
  session ended with a final message.

Everything here is computed from the JSONL log alone, so a run can be analysed
long after it happened and on a different machine. Token counts come from the
backend's ``usage`` field when present; otherwise they are estimated as
``bytes / 4`` and marked ``tokens_estimated``.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .schema import load_session

BYTES_PER_TOKEN_ESTIMATE = 4


# --------------------------------------------------------------------------- helpers
def _dumps(obj: Any) -> str:
    """Canonical JSON used for size and equality checks."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _nbytes(obj: Any) -> int:
    return len(_dumps(obj).encode("utf-8"))


def _percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile; ``None`` for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _history(req: dict[str, Any]) -> list[Any]:
    """Conversation items sent in a request (Responses ``input`` or Chat ``messages``)."""
    items = req.get("input")
    if items is None:
        items = req.get("messages")
    if isinstance(items, str):  # Responses API also accepts a bare string
        return [items]
    return list(items or [])


def offered_tools(req: dict[str, Any]) -> set[str]:
    """Names of the tools a request offered the model.

    Responses tools look like ``{"type": "function", "name": "exec_command"}``;
    built-in ones like ``{"type": "web_search"}`` have no name, so the type is
    used. Chat tools look like ``{"type": "function", "function": {"name": ...}}``.
    """
    names: set[str] = set()
    for tool in req.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            names.add(fn["name"])
        elif tool.get("name"):
            names.add(tool["name"])
        elif tool.get("type"):
            names.add(tool["type"])
    return names


@dataclass
class ToolCall:
    name: str
    arguments: str
    kind: str                 # function_call, custom_tool_call, chat_tool_call, ...
    valid_arguments: bool     # JSON object for function tools; non-empty for freeform tools
    known_tool: bool          # the name was in the request's tool list

    @property
    def signature(self) -> str:
        """Identity of the action, insensitive to JSON key order and whitespace."""
        try:
            args = _dumps(json.loads(self.arguments))
        except (json.JSONDecodeError, TypeError):
            args = (self.arguments or "").strip()
        return f"{self.name}:{args}"


def _check_json_object(text: Any) -> bool:
    if not isinstance(text, str):
        return False
    try:
        return isinstance(json.loads(text), dict)
    except json.JSONDecodeError:
        return False


def extract_output(resp: Any, offered: set[str]) -> tuple[list[ToolCall], str]:
    """Return the tool calls and the assistant text from a response body."""
    calls: list[ToolCall] = []
    texts: list[str] = []
    if not isinstance(resp, dict):
        return calls, ""

    # Responses API
    for item in resp.get("output") or []:
        if not isinstance(item, dict):
            continue
        kind = item.get("type", "")
        if kind == "message":
            for part in item.get("content") or []:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    texts.append(part["text"])
        elif kind == "function_call":
            name = item.get("name") or ""
            args = item.get("arguments")
            calls.append(ToolCall(name, args if isinstance(args, str) else _dumps(args), kind,
                                  _check_json_object(args), name in offered))
        elif kind == "custom_tool_call":  # freeform tools, e.g. apply_patch
            name = item.get("name") or ""
            raw = item.get("input")
            calls.append(ToolCall(name, raw if isinstance(raw, str) else "", kind,
                                  isinstance(raw, str) and bool(raw.strip()), name in offered))
        elif kind.endswith("_call") and kind != "reasoning":  # built-ins: web_search_call, local_shell_call
            base = kind[: -len("_call")]
            calls.append(ToolCall(base, _dumps(item.get("action", {})), kind, True,
                                  base in offered or kind in offered))

    # Chat Completions
    for choice in resp.get("choices") or []:
        msg = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(msg, dict):
            continue
        if isinstance(msg.get("content"), str):
            texts.append(msg["content"])
        for tc in msg.get("tool_calls") or []:
            fn = (tc or {}).get("function") or {}
            name = fn.get("name") or ""
            args = fn.get("arguments")
            calls.append(ToolCall(name, args if isinstance(args, str) else "", "chat_tool_call",
                                  _check_json_object(args), name in offered))
    return calls, "".join(texts)


def _usage(resp: Any) -> tuple[int | None, int | None, int | None]:
    """(input, output, cached input) tokens reported by the backend, if any."""
    if not isinstance(resp, dict) or not isinstance(resp.get("usage"), dict):
        return None, None, None
    u = resp["usage"]
    inp = u.get("input_tokens", u.get("prompt_tokens"))
    out = u.get("output_tokens", u.get("completion_tokens"))
    details = u.get("input_tokens_details") or u.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    return inp, out, cached


def _shared_prefix(a: list[Any], b: list[Any]) -> int:
    """Number of leading items that are identical in both lists."""
    n = 0
    for x, y in zip(a, b):
        if _dumps(x) != _dumps(y):
            break
        n += 1
    return n


# --------------------------------------------------------------------------- metrics
@dataclass
class CallMetrics:
    call_index: int
    ok: bool
    status: int
    served_model: str | None
    history_items: int
    request_bytes: int
    resent_ratio: float | None      # share of this request's history bytes identical to the previous request's prefix
    input_tokens: int
    output_tokens: int
    cached_tokens: int | None
    tokens_estimated: bool
    ttfb_ms: float | None
    latency_ms: float
    decode_tok_per_s: float | None
    tool_calls: list[str]
    malformed_tool_calls: int
    unknown_tool_calls: int
    repeated_tool_calls: int
    text_only: bool                 # replied with text and no tool call


@dataclass
class SessionMetrics:
    session_id: str
    backend: str
    served_models: list[str]
    calls: int
    failed_calls: int
    total_input_tokens: int
    total_output_tokens: int
    tokens_estimated: bool
    first_input_tokens: int
    last_input_tokens: int
    mean_resent_ratio: float | None
    cached_token_share: float | None
    total_latency_s: float
    ttfb_p50_ms: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    mean_decode_tok_per_s: float | None
    tool_calls: int
    tool_call_mix: dict[str, int]
    malformed_tool_call_rate: float | None
    unknown_tool_call_rate: float | None
    repeated_tool_call_rate: float | None
    text_only_turns: int
    ended_with_message: bool
    per_call: list[CallMetrics] = field(default_factory=list)

    def to_dict(self, include_calls: bool = True) -> dict[str, Any]:
        d = asdict(self)
        if not include_calls:
            d.pop("per_call")
        return d


def analyze_calls(calls: Iterable[dict[str, Any]]) -> SessionMetrics:
    """Compute metrics for one session's ordered call records."""
    calls = list(calls)
    per_call: list[CallMetrics] = []
    seen_actions: set[str] = set()
    prev_history: list[Any] | None = None
    estimated_any = False

    for c in calls:
        req = c.get("request_body") if isinstance(c.get("request_body"), dict) else {}
        resp = c.get("response_body")
        history = _history(req)
        offered = offered_tools(req)
        tool_calls, text = extract_output(resp, offered)

        resent_ratio = None
        if prev_history is not None and history:
            k = _shared_prefix(prev_history, history)
            total = _nbytes(history)
            resent_ratio = _nbytes(history[:k]) / total if total else None
        prev_history = history

        in_tok, out_tok, cached = _usage(resp)
        estimated = in_tok is None or out_tok is None
        if in_tok is None:
            in_tok = _nbytes(req) // BYTES_PER_TOKEN_ESTIMATE
        if out_tok is None:
            out_bytes = len(text.encode("utf-8")) + sum(len(t.arguments.encode("utf-8")) for t in tool_calls)
            out_tok = out_bytes // BYTES_PER_TOKEN_ESTIMATE
        estimated_any |= estimated

        repeated = 0
        for t in tool_calls:
            if t.signature in seen_actions:
                repeated += 1
            seen_actions.add(t.signature)

        ttfb = c.get("ttfb_ms")
        latency = float(c.get("latency_ms") or 0.0)
        decode_s = (latency - ttfb) / 1000 if ttfb is not None else None
        tps = out_tok / decode_s if decode_s and decode_s > 0 and out_tok else None
        ok = c.get("status") == 200 and "error" not in c

        per_call.append(CallMetrics(
            call_index=c.get("call_index", len(per_call)),
            ok=ok,
            status=int(c.get("status") or 0),
            served_model=resp.get("model") if isinstance(resp, dict) else None,
            history_items=len(history),
            request_bytes=_nbytes(req),
            resent_ratio=resent_ratio,
            input_tokens=int(in_tok),
            output_tokens=int(out_tok),
            cached_tokens=cached,
            tokens_estimated=estimated,
            ttfb_ms=ttfb,
            latency_ms=latency,
            decode_tok_per_s=tps,
            tool_calls=[t.name for t in tool_calls],
            malformed_tool_calls=sum(not t.valid_arguments for t in tool_calls),
            unknown_tool_calls=sum(not t.known_tool for t in tool_calls),
            repeated_tool_calls=repeated,
            text_only=ok and not tool_calls and bool(text.strip()),
        ))

    ok_calls = [m for m in per_call if m.ok]
    n_tools = sum(len(m.tool_calls) for m in per_call)
    mix: dict[str, int] = {}
    for m in per_call:
        for name in m.tool_calls:
            mix[name] = mix.get(name, 0) + 1
    rate = (lambda k: k / n_tools) if n_tools else (lambda k: None)
    cached_known = [m for m in ok_calls if m.cached_tokens is not None]
    cached_share = (sum(m.cached_tokens or 0 for m in cached_known) / max(1, sum(m.input_tokens for m in cached_known))
                    if cached_known else None)
    last_ok = ok_calls[-1] if ok_calls else None

    first = calls[0] if calls else {}
    return SessionMetrics(
        session_id=str(first.get("session_id", "")),
        backend=str(first.get("backend_name", "")),
        served_models=sorted({m.served_model for m in per_call if m.served_model}),
        calls=len(per_call),
        failed_calls=len(per_call) - len(ok_calls),
        total_input_tokens=sum(m.input_tokens for m in ok_calls),
        total_output_tokens=sum(m.output_tokens for m in ok_calls),
        tokens_estimated=estimated_any,
        first_input_tokens=ok_calls[0].input_tokens if ok_calls else 0,
        last_input_tokens=ok_calls[-1].input_tokens if ok_calls else 0,
        mean_resent_ratio=_mean([m.resent_ratio for m in per_call if m.resent_ratio is not None]),
        cached_token_share=cached_share,
        total_latency_s=sum(m.latency_ms for m in per_call) / 1000,
        ttfb_p50_ms=_percentile([m.ttfb_ms for m in ok_calls if m.ttfb_ms is not None], 50),
        latency_p50_ms=_percentile([m.latency_ms for m in ok_calls], 50),
        latency_p95_ms=_percentile([m.latency_ms for m in ok_calls], 95),
        mean_decode_tok_per_s=_mean([m.decode_tok_per_s for m in ok_calls if m.decode_tok_per_s]),
        tool_calls=n_tools,
        tool_call_mix=dict(sorted(mix.items(), key=lambda kv: -kv[1])),
        malformed_tool_call_rate=rate(sum(m.malformed_tool_calls for m in per_call)),
        unknown_tool_call_rate=rate(sum(m.unknown_tool_calls for m in per_call)),
        repeated_tool_call_rate=rate(sum(m.repeated_tool_calls for m in per_call)),
        text_only_turns=sum(m.text_only for m in per_call),
        ended_with_message=bool(last_ok and not last_ok.tool_calls),
        per_call=per_call,
    )


def analyze_session(path: str | Path) -> SessionMetrics:
    """Load (and schema-validate) a session JSONL file and analyse it."""
    return analyze_calls(load_session(path))


# --------------------------------------------------------------------------- report
_ROWS: list[tuple[str, str, str]] = [
    # (label, attribute, format)
    ("Served model(s)", "served_models", "list"),
    ("LLM calls (turns)", "calls", "int"),
    ("Failed calls", "failed_calls", "int"),
    ("Total input tokens", "total_input_tokens", "int"),
    ("Total output tokens", "total_output_tokens", "int"),
    ("Input tokens, first → last turn", "_growth", "raw"),
    ("Re-sent history per request (mean)", "mean_resent_ratio", "pct"),
    ("Input served from cache", "cached_token_share", "pct"),
    ("Wall time in LLM calls (s)", "total_latency_s", "f1"),
    ("TTFB p50 (ms)", "ttfb_p50_ms", "f0"),
    ("Latency p50 / p95 (ms)", "_lat", "raw"),
    ("Decode throughput (tok/s)", "mean_decode_tok_per_s", "f1"),
    ("Tool calls", "tool_calls", "int"),
    ("Tool mix", "tool_call_mix", "mix"),
    ("Malformed tool-call args", "malformed_tool_call_rate", "pct"),
    ("Calls to tools never offered", "unknown_tool_call_rate", "pct"),
    ("Repeated identical actions", "repeated_tool_call_rate", "pct"),
    ("Text-only turns (no action)", "text_only_turns", "int"),
    ("Ended with a final message", "ended_with_message", "bool"),
]


def _fmt(s: SessionMetrics, attr: str, kind: str) -> str:
    if attr == "_growth":
        return f"{s.first_input_tokens:,} → {s.last_input_tokens:,}"
    if attr == "_lat":
        p50, p95 = s.latency_p50_ms, s.latency_p95_ms
        return "–" if p50 is None else f"{p50:,.0f} / {p95:,.0f}"
    v = getattr(s, attr)
    if v is None:
        return "–"
    if kind == "int":
        return f"{v:,}"
    if kind == "pct":
        return f"{100 * v:.1f}%"
    if kind == "f0":
        return f"{v:,.0f}"
    if kind == "f1":
        return f"{v:,.1f}"
    if kind == "bool":
        return "yes" if v else "no"
    if kind == "list":
        return ", ".join(v) or "–"
    if kind == "mix":
        return ", ".join(f"{k}×{n}" for k, n in v.items()) or "–"
    return str(v)


def markdown_report(sessions: list[SessionMetrics], titles: list[str] | None = None) -> str:
    """Side-by-side comparison table plus a per-turn table for each session."""
    titles = titles or [s.backend or s.session_id for s in sessions]
    lines = ["| Metric | " + " | ".join(titles) + " |", "|---|" + "---|" * len(sessions)]
    for label, attr, kind in _ROWS:
        lines.append(f"| {label} | " + " | ".join(_fmt(s, attr, kind) for s in sessions) + " |")
    if any(s.tokens_estimated for s in sessions):
        lines.append("")
        lines.append("Token counts for a backend that returned no `usage` are estimated as bytes / 4.")
    for title, s in zip(titles, sessions):
        lines += ["", f"### Per turn: {title}", "",
                  "| # | status | input tok | output tok | re-sent | latency ms | action |",
                  "|---|---|---|---|---|---|---|"]
        for m in s.per_call:
            action = ", ".join(m.tool_calls) or ("text" if m.text_only else "–")
            flags = []
            if m.malformed_tool_calls:
                flags.append(f"{m.malformed_tool_calls} malformed")
            if m.unknown_tool_calls:
                flags.append(f"{m.unknown_tool_calls} unknown")
            if m.repeated_tool_calls:
                flags.append(f"{m.repeated_tool_calls} repeated")
            if flags:
                action += " (" + ", ".join(flags) + ")"
            resent = "–" if m.resent_ratio is None else f"{100 * m.resent_ratio:.0f}%"
            lines.append(f"| {m.call_index} | {m.status} | {m.input_tokens:,} | {m.output_tokens:,} | "
                         f"{resent} | {m.latency_ms:,.0f} | {action} |")
    return "\n".join(lines) + "\n"
