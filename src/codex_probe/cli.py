"""Command line entry point.

    codex-probe serve --config configs/openai.json      # run the proxy until Ctrl+C
    codex-probe show  codex_probe_logs/<session>.jsonl  # turn-by-turn summary of a session
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import threading
from typing import Any

from .recorder import ProxyRecorder
from .schema import load_session


def summarize(entry: dict[str, Any]) -> str:
    """One line per call: who served it and what the model did (tool calls or text)."""
    req = entry.get("request_body") if isinstance(entry.get("request_body"), dict) else {}
    resp = entry.get("response_body") if isinstance(entry.get("response_body"), dict) else {}
    out = []
    for item in resp.get("output") or []:                                   # Responses API
        label = item.get("name") or item.get("type", "?")
        if item.get("type") == "function_call":                             # show the command it ran
            try:
                label += f"({json.loads(item.get('arguments') or '{}').get('cmd', '')!s:.60})"
            except (json.JSONDecodeError, AttributeError):
                label += "(<unparseable arguments>)"
        out.append(label)
    for choice in resp.get("choices") or []:                                # Chat Completions
        msg = choice.get("message") or {}
        out += [tc["function"]["name"] for tc in msg.get("tool_calls") or []] or ["message"]
    return (f"#{entry['call_index']:>3} {entry['path']} status={entry['status']} "
            f"backend={entry['backend_name']} served_by={resp.get('model', '?')} "
            f"history_items={len(req.get('input') or req.get('messages') or [])} "
            f"tools={len(req.get('tools') or [])} output={out} latency={entry['latency_ms']:.0f}ms")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="codex-probe")
    sub = parser.add_subparsers(dest="cmd", required=True)
    serve = sub.add_parser("serve", help="run the recording proxy")
    serve.add_argument("--config", required=True, help="path to a JSON config")
    show = sub.add_parser("show", help="summarize a session log")
    show.add_argument("log", help="session .jsonl file")
    show.add_argument("--full", action="store_true", help="print each full call as JSON")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.cmd == "serve":
        recorder = ProxyRecorder(args.config)
        endpoint = recorder.start()
        print(f"Set model_providers.<name>.base_url = \"{endpoint}\"  (log: {recorder.log_path})")
        done = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: done.set())
        signal.signal(signal.SIGTERM, lambda *_: done.set())
        done.wait()
        calls = recorder.stop()
        print(f"Recorded {len(calls)} calls -> {recorder.log_path}")
    else:
        for entry in load_session(args.log):
            print(json.dumps(entry, indent=2, ensure_ascii=False) if args.full else summarize(entry))


if __name__ == "__main__":
    main()
