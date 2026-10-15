"""Command line entry point.

    codex-probe serve --config configs/openai.json      # run the proxy until Ctrl+C
    codex-probe show  codex_probe_logs/<session>.jsonl  # turn-by-turn summary of a session
    codex-probe analyze a.jsonl b.jsonl --names openai qwen --out report.md --json report.json
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import threading
from typing import Any

from .recorder import ProxyRecorder
from .analysis import analyze_session, markdown_report
from .schema import load_session


def _summarize(entry: dict[str, Any]) -> str:
    req = entry.get("request_body") or {}
    resp = entry.get("response_body") or {}
    model = req.get("model", "?") if isinstance(req, dict) else "?"
    n_in = len(req.get("input") or req.get("messages") or []) if isinstance(req, dict) else 0
    n_tools = len(req.get("tools") or []) if isinstance(req, dict) else 0
    served_by = resp.get("model", "?") if isinstance(resp, dict) else "?"
    return (f"#{entry['call_index']:>3} {entry['method']} {entry['path']} status={entry['status']} "
            f"backend={entry['backend_name']} model_req={model} model_resp={served_by} "
            f"input_items={n_in} tools={n_tools} streamed={entry['streamed']} "
            f"latency={entry['latency_ms']:.0f}ms")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="codex-probe")
    sub = parser.add_subparsers(dest="cmd", required=True)
    serve = sub.add_parser("serve", help="run the recording proxy")
    serve.add_argument("--config", required=True, help="path to a JSON config")
    show = sub.add_parser("show", help="summarize a session log")
    show.add_argument("log", help="session .jsonl file")
    show.add_argument("--full", action="store_true", help="print each full call as JSON")
    analyze = sub.add_parser("analyze", help="compare cost, latency and agent behaviour across sessions")
    analyze.add_argument("logs", nargs="+", help="session .jsonl files, one column each")
    analyze.add_argument("--names", nargs="+", help="column titles (default: backend names)")
    analyze.add_argument("--out", help="write the Markdown report here (default: stdout)")
    analyze.add_argument("--json", dest="json_out", help="also write all metrics as JSON")
    args = parser.parse_args(argv)
    if args.cmd == "analyze":
        if args.names and len(args.names) != len(args.logs):
            parser.error("--names needs one title per log")
        sessions = [analyze_session(p) for p in args.logs]
        report = markdown_report(sessions, args.names)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(report)
        else:
            print(report)
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as fh:
                json.dump([s.to_dict() for s in sessions], fh, indent=2)
        return

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
            print(json.dumps(entry, indent=2, ensure_ascii=False) if args.full else _summarize(entry))


if __name__ == "__main__":
    main()
