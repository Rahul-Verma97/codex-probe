"""Local reverse proxy that records every LLM call Codex makes.

How it fits together::

    codex CLI --HTTP--> ProxyRecorder (127.0.0.1:8135/v1) --HTTP--> real backend
                               |                                     (OpenAI or local vLLM/Ollama)
                               +--> session JSONL log (one line per call)

Design choices worth knowing:

* The proxy runs an aiohttp server on its own event loop in a background
  thread, so ``start()``/``stop()`` can be called from ordinary blocking code
  (a script, a notebook) while Codex runs as a separate process.
* Request bodies are forwarded byte-for-byte. Response bytes are relayed to
  Codex chunk-by-chunk *as they arrive* (so streaming stays live) and copied
  into a buffer that is reassembled into one object only after the stream ends.
* The proxy never interprets or rewrites payloads. The only header changes are
  dropping hop-by-hop headers, optionally swapping ``Authorization`` for the
  configured key, and asking upstream for an uncompressed body so the logged
  bytes are readable.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import aiohttp
from aiohttp import web

from .config import ProbeConfig
from .sse import reassemble

logger = logging.getLogger("codex_probe")

# Headers that describe a single hop and must not be copied across the proxy.
HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
    "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}
SECRET_HEADERS = {"authorization", "api-key", "x-api-key", "openai-api-key"}


def _redact(headers: Mapping[str, str]) -> dict[str, str]:
    """Copy headers for the log with secrets masked (keep last 4 chars for debugging)."""
    out = {}
    for k, v in headers.items():
        out[k] = ("***" + v[-4:]) if k.lower() in SECRET_HEADERS and v else v
    return out


def _decode_body(raw: bytes) -> Any:
    """Return parsed JSON when possible, otherwise the text, so logs stay readable."""
    if not raw:
        return None
    text = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


class ProxyRecorder:
    """Record-and-forward proxy for an OpenAI-compatible backend.

    Example::

        recorder = ProxyRecorder(config)
        endpoint = recorder.start()      # "http://127.0.0.1:8135/v1"
        ...run codex with model_providers.<name>.base_url = endpoint...
        calls = recorder.stop()
    """

    def __init__(self, config: ProbeConfig | Mapping[str, Any] | str | Path) -> None:
        self.config = ProbeConfig.from_any(config)
        self._calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._runner: web.AppRunner | None = None
        self._client: aiohttp.ClientSession | None = None
        self._ready = threading.Event()
        self._start_error: BaseException | None = None
        self.port: int | None = None
        self.session_id: str | None = None
        self.log_path: Path | None = None

    # ------------------------------------------------------------------ public API
    def start(self) -> str:
        """Start the proxy in a background thread and return its ``/v1`` endpoint."""
        if self._thread is not None:
            raise RuntimeError("recorder already started")
        self._open_session()
        self._thread = threading.Thread(target=self._run_loop, name="codex-probe", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=30)
        if self._start_error is not None:
            raise RuntimeError(f"proxy failed to start: {self._start_error}") from self._start_error
        endpoint = f"http://{self.config.listen_host}:{self.port}/v1"
        logger.info("CodexProbe listening on %s -> %s (%s)", endpoint, self.config.backend_base_url,
                    self.config.backend_name)
        return endpoint

    def new_session(self, name: str | None = None) -> Path:
        """Start a fresh JSONL file, e.g. one per ``codex exec`` invocation."""
        with self._lock:
            self._open_session(name)
            assert self.log_path is not None
            return self.log_path

    def calls(self) -> list[dict[str, Any]]:
        """Snapshot of all calls recorded so far (across sessions), in order."""
        with self._lock:
            return list(self._calls)

    def stop(self) -> list[dict[str, Any]]:
        """Stop the proxy and return every recorded call, in order."""
        if self._loop is not None and self._thread is not None:
            fut = asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
            fut.result(timeout=30)
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=30)
        self._thread = None
        self._loop = None
        return self.calls()

    # ------------------------------------------------------------- session logs
    def _open_session(self, name: str | None = None) -> None:
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        label = name or self.config.session_name or self.config.backend_name
        self.session_id = f"{label}-{stamp}-{uuid.uuid4().hex[:6]}"
        self.log_path = log_dir / f"{self.session_id}.jsonl"
        meta = {
            "session_id": self.session_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "config": self.config.to_dict(),
        }
        (log_dir / f"{self.session_id}.meta.json").write_text(json.dumps(meta, indent=2))
        self.log_path.touch()

    def _append(self, entry: dict[str, Any]) -> None:
        with self._lock:
            entry["session_id"] = self.session_id
            entry["call_index"] = len(self._calls)
            self._calls.append(entry)
            assert self.log_path is not None
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------- event loop
    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._startup())
        except BaseException as exc:  # surface bind errors to start()
            self._start_error = exc
            self._ready.set()
            return
        self._ready.set()
        self._loop.run_forever()
        self._loop.close()

    async def _startup(self) -> None:
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout_s)
        # auto_decompress=False: we relay exactly the bytes upstream sent.
        self._client = aiohttp.ClientSession(timeout=timeout, auto_decompress=False)
        app = web.Application(client_max_size=1024 ** 3)
        app.router.add_route("*", "/{tail:.*}", self._handle)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.config.listen_host, self.config.listen_port)
        await site.start()
        server = site._server  # type: ignore[attr-defined]
        self.port = server.sockets[0].getsockname()[1]

    async def _shutdown(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
        if self._client is not None:
            await self._client.close()

    # ------------------------------------------------------------- forwarding
    def _upstream_url(self, request: web.Request) -> str:
        """Map ``/v1/<rest>`` on the proxy to ``<backend_base_url>/<rest>``."""
        path = request.path
        if path.startswith("/v1/"):
            path = path[3:]
        elif path == "/v1":
            path = ""
        url = self.config.backend_base_url + path
        if request.query_string:
            url += "?" + request.query_string
        return url

    def _upstream_headers(self, request: web.Request) -> dict[str, str]:
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
        # Ask for an uncompressed body so what we relay is also what we can log.
        headers["Accept-Encoding"] = "identity"
        if self.config.api_key_env:
            key = os.environ.get(self.config.api_key_env)
            if key:
                headers["Authorization"] = f"Bearer {key}"
            else:
                logger.warning("api_key_env %s is not set; forwarding client auth", self.config.api_key_env)
        headers.update(self.config.extra_headers)
        return headers

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        assert self._client is not None
        started = time.perf_counter()
        req_body = await request.read()
        url = self._upstream_url(request)
        headers = self._upstream_headers(request)
        entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "backend_name": self.config.backend_name,
            "backend_base_url": self.config.backend_base_url,
            "wire_api": self.config.wire_api,
            "method": request.method,
            "path": request.path,
            "upstream_url": url,
            "request_headers": _redact(headers),
            "request_body": _decode_body(req_body),
        }
        try:
            async with self._client.request(request.method, url, headers=headers, data=req_body) as up:
                resp_headers = {k: v for k, v in up.headers.items() if k.lower() not in HOP_BY_HOP}
                streamed = "text/event-stream" in up.headers.get("Content-Type", "")
                entry.update(status=up.status, response_headers=resp_headers, streamed=streamed)
                if streamed:
                    out = web.StreamResponse(status=up.status, headers=resp_headers)
                    await out.prepare(request)
                    buf = bytearray()
                    first_byte_ms: float | None = None
                    client_gone = False
                    async for chunk in up.content.iter_any():
                        if first_byte_ms is None:
                            first_byte_ms = (time.perf_counter() - started) * 1000
                        buf.extend(chunk)
                        if not client_gone:
                            try:
                                await out.write(chunk)  # relay live, unmodified
                            except (ConnectionResetError, aiohttp.ClientConnectionResetError):
                                # Codex hung up (e.g. it decided to stop or retry). Keep
                                # draining upstream so the log still holds the full answer.
                                client_gone = True
                    if client_gone:
                        entry["client_disconnected"] = True
                    else:
                        try:
                            await out.write_eof()
                        except (ConnectionResetError, aiohttp.ClientConnectionResetError):
                            # Codex closes the socket as soon as it reads the final
                            # event; every byte was already delivered, so this is benign.
                            pass
                    entry["ttfb_ms"] = round(first_byte_ms or 0.0, 2)
                    entry["response_body"] = reassemble(bytes(buf), self.config.wire_api)
                    entry["raw_stream"] = buf.decode("utf-8", errors="replace")
                    result: web.StreamResponse = out
                else:
                    body = await up.read()
                    entry["response_body"] = _decode_body(body)
                    result = web.Response(status=up.status, body=body, headers=resp_headers)
        except Exception as exc:  # upstream unreachable, timeout, client hung up...
            logger.exception("upstream call failed: %s", url)
            entry.update(status=502, streamed=False, error=f"{type(exc).__name__}: {exc}")
            entry["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            self._append(entry)
            return web.json_response({"error": {"message": f"codex_probe upstream error: {exc}"}}, status=502)
        entry["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        self._append(entry)
        logger.info("call #%d %s %s -> %s in %.0f ms", len(self._calls) - 1, request.method,
                    request.path, entry["status"], entry["latency_ms"])
        return result
