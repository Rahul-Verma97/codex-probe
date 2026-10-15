"""Configuration for the CodexProbe recorder.

Everything that decides *where* traffic goes lives here, so swapping the real
backend (OpenAI -> local Qwen) is a config change and never a code change.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

VALID_WIRE_APIS = ("responses", "chat")


@dataclass
class ProbeConfig:
    """Validated settings for one recorder instance.

    Attributes:
        backend_base_url: Real backend to forward to, including the ``/v1``
            prefix, e.g. ``https://api.openai.com/v1`` or ``http://127.0.0.1:8000/v1``.
        backend_name: Human label stored in every log entry (``"openai"``, ``"local-qwen"``).
        wire_api: ``"responses"`` or ``"chat"``. Recorded for the reviewer and used
            to pick the stream reassembler; the proxy itself is path-agnostic.
        api_key_env: Optional environment variable holding the backend key. If set,
            the proxy replaces the ``Authorization`` header with this key. If ``None``,
            the client's own ``Authorization`` header is forwarded unchanged.
        listen_host / listen_port: Where the local proxy listens. Port 0 = pick a free port.
        log_dir: Directory for per-session JSONL logs.
        session_name: Optional prefix for the session log file.
        seed: Stored in the session header for reproducibility notes.
        request_timeout_s: Total timeout for one upstream call.
        extra_headers: Extra headers added to every upstream request.
    """

    backend_base_url: str
    backend_name: str = "backend"
    wire_api: str = "responses"
    api_key_env: str | None = None
    listen_host: str = "127.0.0.1"
    listen_port: int = 8135
    log_dir: str = "codex_probe_logs"
    session_name: str | None = None
    seed: int = 0
    request_timeout_s: float = 600.0
    extra_headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.backend_base_url.startswith(("http://", "https://")):
            raise ValueError("backend_base_url must start with http:// or https://")
        self.backend_base_url = self.backend_base_url.rstrip("/")
        if self.wire_api not in VALID_WIRE_APIS:
            raise ValueError(f"wire_api must be one of {VALID_WIRE_APIS}, got {self.wire_api!r}")
        if not (0 <= int(self.listen_port) <= 65535):
            raise ValueError("listen_port must be between 0 and 65535")
        if self.request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive")

    @classmethod
    def from_any(cls, config: "ProbeConfig | Mapping[str, Any] | str | Path") -> "ProbeConfig":
        """Build a config from a dict, a JSON file path, or an existing ProbeConfig."""
        if isinstance(config, ProbeConfig):
            return config
        if isinstance(config, (str, Path)):
            config = json.loads(Path(config).read_text())
        unknown = set(config) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**dict(config))

    def to_dict(self) -> dict[str, Any]:
        """Config echo for the logs. Never contains the secret itself."""
        return asdict(self)
