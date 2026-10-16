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
```

`examples/quickstart.py` does the whole loop (start the recorder → write a temporary Codex config → `codex exec` a small FizzBuzz task → print one line per recorded call → re-run the task's test to check the task was really done, saved as `<session>.outcome.json`). `examples/kaggle_codex_probe.ipynb` runs everything on a free Kaggle T4 x2 GPU.

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
| Ollama, `qwen2.5-coder:7b` (the model named in the task) | `ollama pull qwen2.5-coder:7b` | ~5 GB VRAM (4-bit): one T4 or a laptop GPU |
| Ollama, `qwen3-coder:30b` (MoE, 3B active) | `ollama pull qwen3-coder:30b` | ~19 GB weights (4-bit) + KV cache: 2× T4 or one 24 GB GPU |
| vLLM | `vllm serve Qwen/Qwen2.5-Coder-7B-Instruct --dtype half --max-model-len 32768 --enable-auto-tool-choice --tool-call-parser hermes` | ~16 GB fp16 |

**Give the model a 32k context.** Codex's first request is about 10k tokens (instructions + 9 tool schemas). Ollama's default context is 4k and it truncates the prompt silently; in our first run the log showed only 3,836 prompt tokens reaching the 7B model. Bake the context into the model so no server flag can be missed:

```bash
printf 'FROM qwen2.5-coder:7b\nPARAMETER num_ctx 32768\n' > Modelfile
ollama create qwen2.5-coder:7b-32k -f Modelfile
python examples/quickstart.py --config configs/qwen_ollama.json --model qwen2.5-coder:7b-32k --env-key OLLAMA_API_KEY
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

18 tests. The proxy tests run against an in-process mock OpenAI-compatible server (`tests/mock_backend.py`):

- passthrough round-trip, streaming and non-streaming, with byte-for-byte equality of what the client receives;
- the stream really arrives in multiple chunks (live relay);
- reassembly of text, parallel tool calls and fragmented arguments (Chat) and of Responses events, including truncated streams and arbitrary chunk boundaries;
- auth injection and redaction, ordered session logs validated against the JSON schema, session rotation, upstream-down handling, client disconnect mid-stream, and config validation.

It was also checked end-to-end with the real `codex exec` (codex-cli 0.162.1) through the proxy against the mock backend.

## Findings

1. **`wire_api = "chat"` is gone.** Codex 0.162 refuses to start with it, so the local server must speak the Responses API. Ollama and vLLM both do.
2. **Silent prompt truncation.** With Ollama's default 4k context, Codex's ~10k-token request was cut to 3,836 tokens before reaching Qwen. Nothing errors, and only the recorded `usage` shows it. Fixed with `num_ctx 32768` (above).
3. **Qwen2.5-Coder-7B does not emit structured tool calls.** It writes the call as JSON inside a Markdown code block instead of the `<tool_call>` format the server parses, so Codex receives plain text, treats it as the final answer, and stops after one turn with exit code 0 while no file was created. This still happens with the full 32k context, so truncation was not the cause. It is the "tool-skip" failure mode described in recent tool-use failure studies. Codex's exit code therefore cannot be trusted as success; the quickstart re-runs the task's test itself.
4. **Qwen3-Coder-30B completes the task.** Same recorder, same config file, only the model name changes. It wrote `fizzbuzz.py`, wrote `test_fizzbuzz.py`, ran the tests, spot-checked edge cases and finished, all through real `function_call` items, every one served by the local Ollama server.

Kaggle, 2× T4, codex-cli 0.162.1, Ollama, task: write FizzBuzz and its test, then run it.

| backend | model | LLM calls | structured tool calls | task solved (our own test run) |
|---|---|---|---|---|
| OpenAI | gpt-5-codex | not yet run (needs an API key) | | |
| local Ollama | qwen2.5-coder:7b-32k | 1 | 0 (call written as text) | no |
| local Ollama | qwen3-coder:30b-32k | 6 | 5 `exec_command` | yes |

`codex-probe show` on the Qwen3 session:

```
#0 output=['message', "exec_command(cat > fizzbuzz.py << 'EOF' ...)"]
#1 output=['message', "exec_command(cat > test_fizzbuzz.py << 'EOF' ...)"]
#2 output=['message', 'exec_command(python test_fizzbuzz.py)']
#3 output=['message', 'exec_command(python -c "from fizzbuzz import fizzbuzz; print(fizzbuzz(15)) ...")']
#4 output=['message', 'exec_command(python -c "... fizzbuzz(0) ...")']
#5 output=['message']                      <- final answer
```

## Known limitations

- Only HTTP is proxied; Codex's optional WebSocket transport to OpenAI is not used with a custom `base_url`.
- Logs store full prompts, which can include file contents from the working directory; treat them as sensitive.
