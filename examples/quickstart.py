"""Run Codex through CodexProbe once and show what was recorded.

Same script for both runs; only the config (and model name) changes:

    # (a) default OpenAI backend
    python examples/quickstart.py --config configs/openai.json --model gpt-5-codex

    # (b) local Qwen served by Ollama (must expose /v1/responses)
    python examples/quickstart.py --config configs/qwen_ollama.json --model qwen2.5-coder:7b-32k \
        --env-key OLLAMA_API_KEY

Steps: start the recorder, write a throwaway CODEX_HOME/config.toml pointing a
provider at it, run `codex exec` on a small task, stop the recorder, print one
line per recorded call, then re-run the task's test ourselves to see whether
the task was really completed (Codex's exit code alone does not tell us).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import tempfile
from pathlib import Path

from codex_probe import ProbeConfig, ProxyRecorder
from codex_probe.cli import summarize

TASK = ("Create fizzbuzz.py with a function fizzbuzz(n) that returns the FizzBuzz list for 1..n, "
        "and test_fizzbuzz.py that checks it with plain asserts. Run `python test_fizzbuzz.py` "
        "and fix any failure.")
CHECK = "python test_fizzbuzz.py"

CODEX_TOML = """\
model = "{model}"
model_provider = "probe"

[model_providers.probe]
name = "CodexProbe -> {backend}"
base_url = "{endpoint}"
wire_api = "responses"
env_key = "{env_key}"
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="CodexProbe JSON config")
    ap.add_argument("--model", required=True, help="model name Codex requests")
    ap.add_argument("--env-key", default="OPENAI_API_KEY", help="env var Codex reads the API key from")
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--sandbox", default="workspace-write", help="use danger-full-access inside Kaggle/Docker")
    args = ap.parse_args()

    os.environ.setdefault(args.env_key, "unused-for-local-models")
    session = args.model.replace("/", "_").replace(":", "_")
    config = dataclasses.replace(ProbeConfig.from_any(args.config), session_name=session)
    recorder = ProxyRecorder(config)
    endpoint = recorder.start()

    codex_home = Path(tempfile.mkdtemp(prefix="codex_home_"))
    toml = CODEX_TOML.format(model=args.model, backend=config.backend_name, endpoint=endpoint,
                             env_key=args.env_key)
    (codex_home / "config.toml").write_text(toml)
    workdir = Path(tempfile.mkdtemp(prefix="codex_task_"))
    print(f"[probe] {endpoint} -> {config.backend_base_url}\n[probe] config.toml:\n{toml}")

    cmd = ["codex", "exec", "--skip-git-repo-check", "-s", args.sandbox, "-C", str(workdir), TASK]
    try:
        proc = subprocess.run(cmd, env=dict(os.environ, CODEX_HOME=str(codex_home)), text=True,
                              capture_output=True, timeout=args.timeout, stdin=subprocess.DEVNULL)
        print(f"[codex] exit {proc.returncode}\n{proc.stdout[-2000:]}{proc.stderr[-2000:] if proc.returncode else ''}")
    except subprocess.TimeoutExpired:
        print(f"[codex] timed out after {args.timeout}s; every call so far is still in the log")
    finally:
        calls = recorder.stop()

    print(f"\n[probe] {len(calls)} calls -> {recorder.log_path}")
    for c in calls:
        print("  " + summarize(c))
    solved = subprocess.run(CHECK, shell=True, cwd=workdir, capture_output=True).returncode == 0
    outcome = {"model": args.model, "calls": len(calls), "files": sorted(p.name for p in workdir.iterdir()),
               "task_solved": solved, "upstreams": sorted({c["upstream_url"] for c in calls})}
    recorder.log_path.with_suffix(".outcome.json").write_text(json.dumps(outcome, indent=2))
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    main()
