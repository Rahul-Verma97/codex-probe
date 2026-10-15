# CodexProbe

A small local reverse proxy that records **every LLM call the Codex CLI makes**, verbatim, and lets you swap Codex's backend (OpenAI → a self-hosted Qwen coder model) by changing **only a config file**.

```
codex CLI ──HTTP──▶ ProxyRecorder (127.0.0.1:8135/v1) ──HTTP──▶ real backend (OpenAI / Ollama / vLLM)
                           │
                           └──▶ codex_probe_logs/<session>.jsonl  (one line per LLM call)
```

## Install

```bash
pip install -e .            # runtime: aiohttp, jsonschema (pinned in pyproject.toml)
pip install -e ".[test]"    # + pytest
npm i -g @openai/codex      # the agent being probed (tested with codex-cli 0.162.1)
```

## Usage

```python
from codex_probe import ProxyRecorder

recorder = ProxyRecorder("configs/openai.json")   # dict, JSON path, or ProbeConfig
endpoint = recorder.start()                        # e.g. "http://127.0.0.1:8135/v1"
# point model_providers.<name>.base_url in Codex's config.toml at `endpoint`,
# then run Codex normally (e.g. `codex exec "<task prompt>"`)
calls = recorder.stop()                            # list[dict], one per LLM call, in order
```

Or from the shell:

```bash
codex-probe serve --config configs/openai.json          # runs until Ctrl+C
codex-probe show codex_probe_logs/<session>.jsonl       # turn-by-turn summary (--full for raw JSON)
codex-probe analyze openai-….jsonl qwen-….jsonl --names openai qwen --out comparison.md --json comparison.json
```

`examples/quickstart.py` does the whole loop (start the recorder → write a temporary Codex config → `codex exec` a small FizzBuzz task → print what was captured → independently re-run the task's test and write `<session>.outcome.json`). `examples/kaggle_codex_probe.ipynb` runs both the OpenAI and the local-Qwen runs on a free Kaggle T4 x2 GPU.

## Configuration

Everything about *where* traffic goes is in JSON (`configs/`); swapping backends never touches code.

| key | meaning |
|---|---|
| `backend_base_url` | real backend incl. `/v1`, e.g. `https://api.openai.com/v1`, `http://127.0.0.1:11434/v1` |
| `backend_name` | label written into every log entry (`openai`, `local-qwen-ollama`) |
| `wire_api` | `responses` or `chat`; picks the default stream reassembler (the payload itself is checked first) |
| `api_key_env` | if set, the proxy injects this env var as `Authorization: Bearer …`; if `null`, Codex's own header is forwarded |
| `listen_host`, `listen_port` | where the proxy listens (`0` = any free port) |
| `log_dir`, `session_name` | where session logs go and their filename prefix |
| `seed`, `request_timeout_s`, `extra_headers` | reproducibility note, upstream timeout, extra upstream headers |

## Codex config changes

Put this in `$CODEX_HOME/config.toml` (default `~/.codex/config.toml`; a project-level `.codex/config.toml` cannot override providers):

```toml
model = "qwen2.5-coder:7b"            # or gpt-5-codex for the OpenAI run
model_provider = "probe"

[model_providers.probe]
name = "CodexProbe"
base_url = "http://127.0.0.1:8136/v1" # the endpoint returned by recorder.start()
wire_api = "responses"
env_key = "OPENAI_API_KEY"            # any dummy variable for local models
```

**Finding: the Chat Completions route no longer exists in Codex.** The task text suggests `wire_api = "chat"` for local servers, but current Codex rejects it at startup:

```
Error loading config.toml: `wire_api = "chat"` is no longer supported.
How to fix: set `wire_api = "responses"` in your provider config.
```

So the local model server must expose the **Responses API** (`POST /v1/responses`). Recent Ollama and vLLM both do. CodexProbe itself is format-agnostic: it forwards any path and can reassemble both formats.

## Serving the local Qwen model

| option | command | hardware |
|---|---|---|
| Ollama (used on Kaggle) | `ollama serve` then `ollama pull qwen2.5-coder:7b` | ~6 GB VRAM (4-bit) — one T4 or a recent laptop |
| vLLM | `vllm serve Qwen/Qwen2.5-Coder-7B-Instruct --dtype half --tensor-parallel-size 2 --max-model-len 32768 --enable-auto-tool-choice --tool-call-parser hermes` | ~16 GB fp16 weights; 2× T4 or one 24 GB GPU |

Codex prompts are large (≈10–20k tokens of instructions and tool schema), so give the server at least a 32k context window (`OLLAMA_CONTEXT_LENGTH=32768`).

Then run:

```bash
python examples/quickstart.py --config configs/qwen_ollama.json --model qwen2.5-coder:7b --env-key OLLAMA_DUMMY_KEY
```

The log proves which server answered: every entry records `backend_name`, `upstream_url` (e.g. `http://127.0.0.1:11434/v1/responses`), and the `model` returned in the response body.

The tool names and request fields below are what codex-cli 0.162.1 actually sent in testing; other versions may differ.

## What a captured call looks like

One JSONL line per LLM call (schema in `src/codex_probe/schema.py`, validated by `load_session`):

```json
{
  "session_id": "qwen-20261015T101500Z-3fa9c1",
  "call_index": 0,
  "timestamp": "2026-10-15T10:15:02.114Z",
  "backend_name": "local-qwen-ollama",
  "upstream_url": "http://127.0.0.1:11434/v1/responses",
  "wire_api": "responses",
  "method": "POST", "path": "/v1/responses",
  "request_headers": {"Authorization": "***-key", "...": "..."},
  "request_body": {
    "model": "qwen2.5-coder:7b", "stream": true,
    "input": ["developer: Codex system instructions", "developer: permissions/sandbox", "user: environment context", "user: the task", "...earlier function_call / function_call_output items..."],
    "tools": ["exec_command", "write_stdin", "request_user_input", "view_image", "web_search", "..."],
    "tool_choice": "auto", "parallel_tool_calls": true, "store": false, "...": "..."
  },
  "status": 200, "streamed": true,
  "response_body": {"status": "completed", "model": "qwen2.5-coder:7b",
                    "output": [{"type": "function_call", "name": "exec_command", "arguments": "{\"cmd\": \"python test_fizzbuzz.py\"}"}],
                    "usage": {"...": "..."}},
  "raw_stream": "event: response.created\ndata: {...}\n\n...",
  "ttfb_ms": 812.4, "latency_ms": 6210.9
}
```

Each Codex turn is one call: the request carries the **whole** running history (Codex is stateless per request: `store: false`), so call *n*'s `input` contains every earlier tool call and its output. Replaying a session means reading the JSONL top to bottom.

## Analysing traces: what changes when the backend changes

Recording is the means; the question is what swapping a frontier model for a 7B open model does to an agent. `codex_probe.analysis` computes, from the JSONL log alone:

| Group | Metric | Why it matters |
|---|---|---|
| Cost | input / output tokens per turn and in total; input tokens first → last turn | Codex is stateless per request (`store: false`), so the prompt grows every turn; cost is roughly quadratic in the number of turns. |
| Cost | **re-sent history share**: bytes of each request that are an exact repeat of the previous request's prefix | The upper bound on what prefix caching or trajectory reduction could save. |
| Cost | input served from cache (when the backend reports `cached_tokens`) | How much of that redundancy the backend already absorbs. |
| Latency | TTFB p50, latency p50/p95, decode tokens/s | Prefill dominates TTFB as the history grows; local serving shifts the bottleneck. |
| Behaviour | malformed tool-call arguments; calls to tools never offered | Whether the model can use Codex's tool schema at all. |
| Behaviour | repeated identical actions (same tool, same canonical arguments) | A cheap signal for loops, where cost grows without progress. |
| Behaviour | text-only turns; ended with a final message | Answering instead of acting, and whether the agent terminated on its own. |
| Outcome | `solved` from an independent re-run of the task's check | Codex's own claim of success is not trusted. |

Token counts use the backend's `usage` when present and fall back to `bytes / 4` (flagged in the report). The output is a Markdown table comparing sessions side by side, a per-turn table for each session, and the full metrics as JSON.

This framing follows the efficiency view of agents: the same task can cost very different amounts of computation depending on the model's behaviour (loops, malformed calls, longer trajectories), and the log makes that cost attributable turn by turn.

## How it works (design notes)

- **Faithful passthrough.** The request body is forwarded byte-for-byte to `backend_base_url + <path after /v1>`. Only hop-by-hop headers (`Host`, `Content-Length`, `Connection`, …) are dropped, `Authorization` is replaced only if `api_key_env` is set, and `Accept-Encoding: identity` is sent so the relayed bytes are also readable for the log. Response bytes are relayed unchanged.
- **Streaming.** If the upstream answers `text/event-stream`, the proxy opens a streaming response to Codex and writes each chunk the moment it arrives (`iter_any()`), so the CLI shows tokens live. Every chunk is also appended to a buffer. Only after the stream ends is the buffer parsed into SSE events and reassembled into one complete response. Reassembly never sits on the live path, so it cannot delay or alter what Codex sees. If Codex hangs up early, the proxy keeps draining upstream so the log is still complete (`client_disconnected: true`).
- **Responses vs Chat Completions.**
  - *Chat Completions* (`/v1/chat/completions`): input is a `messages` list; the stream is `chat.completion.chunk` objects whose `delta`s carry text and tool-call fragments (arguments arrive as string pieces keyed by `index`), ending with `data: [DONE]`.
  - *Responses* (`/v1/responses`): input is typed `input` items (Codex 0.162 sends its system prompt as `developer` messages rather than `instructions`) (messages, `function_call`, `function_call_output`, reasoning); the stream is typed events (`response.output_item.added`, `response.output_text.delta`, `response.function_call_arguments.delta`, …) ending with `response.completed`, which already contains the full response object. CodexProbe uses that final object, and falls back to rebuilding from deltas (marked `"status": "partial"`) if the stream is cut.
- **Threading.** The aiohttp server runs on its own event loop in a daemon thread, so `start()`/`stop()` work from plain scripts and notebooks while Codex runs as a separate process.
- **Sessions.** Each `start()` (or `new_session(name)`) opens a new `<session>.jsonl` plus a `<session>.meta.json` with the config echo, so each Codex invocation can be replayed on its own.
- **Errors.** If the backend is unreachable or times out, Codex receives a 502 with a JSON error, and the call is still logged with an `error` field.
- **Secrets.** Keys are never written to logs; auth headers are masked to their last 4 characters.

## Tests

```bash
pytest -q
```

26 tests. The proxy tests run against an in-process mock OpenAI-compatible server (`tests/mock_backend.py`):

- passthrough round-trip, streaming and non-streaming, with byte-for-byte equality of what the client receives;
- the stream really arrives in multiple chunks (live relay);
- reassembly of text, parallel tool calls and fragmented arguments (Chat) and of Responses events, including truncated streams and arbitrary chunk boundaries;
- auth injection and redaction, ordered session logs validated against the JSON schema, session rotation, upstream-down handling, client disconnect mid-stream, and config validation;
- trace analysis: tool-schema parsing for both API shapes, malformed and unknown tool calls, repeated actions under different JSON whitespace, cache share, failed calls, missing `usage`, empty sessions, and the `analyze` CLI.

It was also checked end-to-end with the real `codex exec` (codex-cli 0.162.1) through the proxy against the mock backend.

## Known limitations

- Small local models (Qwen2.5-Coder-7B) often struggle with Codex's large tool schema and multi-step tool use: malformed `function_call` arguments, answering in plain text instead of calling `exec_command`, or looping. The logs make these failures visible call by call; they are a model limitation, not a proxy one.
- Only HTTP is proxied. Codex's optional WebSocket transport to OpenAI is not used with a custom `base_url`.
- Logs store full prompts, which can include file contents from the working directory; treat them as sensitive.
