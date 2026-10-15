"""End-to-end demo: run Codex through CodexProbe and inspect what was recorded.

Usage (same script, only the config changes between the two runs):

    # (a) default OpenAI backend
    export OPENAI_API_KEY=sk-...
    python examples/quickstart.py --config configs/openai.json --model gpt-5-codex

    # (b) local Qwen served by Ollama or vLLM (must expose /v1/responses)
    python examples/quickstart.py --config configs/qwen_ollama.json --model qwen2.5-coder:7b

The script:
  1. starts the recorder from the JSON config,
  2. writes a throwaway CODEX_HOME/config.toml whose provider points at the recorder,
  3. runs `codex exec` on a small coding task in an empty work directory,
  4. stops the recorder and prints a turn-by-turn summary plus which backend served it,
  5. independently re-runs the task's check command (Codex's own claim is not trusted)
     and writes <session>.outcome.json next to the log, so `codex-probe analyze`
     results can be read alongside whether the task was actually solved.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from codex_probe import ProxyRecorder, analyze_calls, markdown_report

DEFAULT_TASK = (
    "Create a file fizzbuzz.py with a function fizzbuzz(n) that returns the FizzBuzz list for 1..n, "
    "plus a test_fizzbuzz.py using plain asserts. Run `python test_fizzbuzz.py` and fix any failure."
)

CODEX_CONFIG = """\
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
    ap.add_argument("--model", required=True, help="model name Codex should request")
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--workdir", default=None, help="where Codex works (default: new temp dir)")
    ap.add_argument("--env-key", default="OPENAI_API_KEY",
                    help="env var Codex reads its API key from (use a dummy one for local models)")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--check", default=None,
                    help="shell command that decides success, run in the workdir after Codex exits "
                         "(default for the built-in task: python test_fizzbuzz.py)")
    ap.add_argument("--sandbox", default="workspace-write",
                    help="codex sandbox mode; use danger-full-access inside containers like Kaggle")
    args = ap.parse_args()

    os.environ.setdefault(args.env_key, "dummy-key-for-local-model")
    recorder = ProxyRecorder(args.config)
    endpoint = recorder.start()
    print(f"[probe] {endpoint} -> {recorder.config.backend_base_url}  log={recorder.log_path}")

    codex_home = Path(tempfile.mkdtemp(prefix="codex_home_"))
    toml = CODEX_CONFIG.format(model=args.model, backend=recorder.config.backend_name,
                               endpoint=endpoint, env_key=args.env_key)
    (codex_home / "config.toml").write_text(toml)
    print("[probe] Codex config.toml:\n" + toml)

    workdir = Path(args.workdir or tempfile.mkdtemp(prefix="codex_task_"))
    workdir.mkdir(parents=True, exist_ok=True)
    cmd = ["codex", "exec", "--skip-git-repo-check", "-s", args.sandbox, "-C", str(workdir), args.task]
    env = dict(os.environ, CODEX_HOME=str(codex_home))
    codex_exit: int | None = None
    try:
        proc = subprocess.run(cmd, env=env, timeout=args.timeout, text=True, capture_output=True,
                              stdin=subprocess.DEVNULL)
        codex_exit = proc.returncode
        print("[codex] exit code", proc.returncode)
        print(proc.stdout[-3000:])
        if proc.returncode:
            print(proc.stderr[-3000:])
    except subprocess.TimeoutExpired:
        print(f"[codex] timed out after {args.timeout}s (logs still contain every call so far)")
    finally:
        calls = recorder.stop()

    print(f"\n[probe] {len(calls)} LLM calls recorded in {recorder.log_path}")
    for c in calls:
        req, resp = c.get("request_body") or {}, c.get("response_body") or {}
        n_items = len(req.get("input") or []) if isinstance(req, dict) else 0
        outputs = [o.get("type") for o in resp.get("output", [])] if isinstance(resp, dict) else []
        print(f"  #{c['call_index']} status={c['status']} backend={c['backend_name']} "
              f"upstream={c['upstream_url']} input_items={n_items} output={outputs} "
              f"served_model={resp.get('model') if isinstance(resp, dict) else '?'} {c['latency_ms']:.0f}ms")
    files = sorted(p.name for p in workdir.iterdir())
    print("\n[probe] files Codex left in", workdir, "->", files)

    check = args.check or ("python test_fizzbuzz.py" if args.task == DEFAULT_TASK else None)
    solved: bool | None = None
    if check:
        try:
            res = subprocess.run(check, shell=True, cwd=workdir, text=True, capture_output=True, timeout=120)
            solved = res.returncode == 0
            print(f"[check] `{check}` -> exit {res.returncode}")
        except subprocess.TimeoutExpired:
            solved = False
            print(f"[check] `{check}` timed out")

    outcome = {"session": recorder.session_id, "model": args.model, "task": args.task,
               "codex_exit_code": codex_exit, "check": check, "solved": solved,
               "files": files, "calls": len(calls),
               "backends": sorted({c["upstream_url"] for c in calls})}
    assert recorder.log_path is not None
    recorder.log_path.with_suffix(".outcome.json").write_text(json.dumps(outcome, indent=2))
    print(json.dumps(outcome, indent=2))
    if calls:
        print(markdown_report([analyze_calls(calls)], [recorder.config.backend_name]))


if __name__ == "__main__":
    main()
