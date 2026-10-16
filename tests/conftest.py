"""Shared fixtures for the SupportAgent test suite.

Rules:
- Never bind the fixed dev ports (8001/8002/8003/8501). Servers here use ephemeral ports.
- Builder modules may still be stubs: use `require(module, attr)` so tests skip instead of failing.
"""

from __future__ import annotations

import importlib
import os
import socket
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

# macOS can re-flag the editable-install .pth as hidden (see TASKS P0.10), which silently drops
# src/ from sys.path. Don't depend on it.
_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from support_agent.common.config import Settings  # noqa: E402
from support_agent.observability import tracing  # noqa: E402
from tests.fakes import FakeLLMClient, FakePolicyGate, FakeTicketPortal, FakeUserInfoClient  # noqa: E402

FORBIDDEN_PORTS = {8001, 8002, 8003, 8501}


# --------------------------------------------------------------------------- skip-if-missing


def require(module: str, *attrs: str) -> Any:
    """Import `module` and return it (or its single requested attr); skip if it is still a stub."""
    try:
        mod = importlib.import_module(module)
    except ImportError as e:
        pytest.skip(f"{module} not importable yet: {e}", allow_module_level=True)
    for a in attrs:
        if not hasattr(mod, a):
            pytest.skip(f"{module}.{a} not implemented yet", allow_module_level=True)
    if len(attrs) == 1:
        return getattr(mod, attrs[0])
    return mod


# --------------------------------------------------------------------------- env / settings


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip SA_* and ANTHROPIC_API_KEY-driven provider overrides so the dev's shell can't leak in."""
    for k in list(os.environ):
        if k.startswith("SA_"):
            monkeypatch.delenv(k, raising=False)


@pytest.fixture(autouse=True)
def _reset_span_sink() -> Iterator[None]:
    yield
    tracing.set_span_sink(None)


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "support_agent.db")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert port not in FORBIDDEN_PORTS
    return port


@pytest.fixture
def settings(tmp_path: Path, db_path: str) -> Settings:
    """Fast, isolated settings: tmp paths, mock LLM, tiny intervals/backoffs, ephemeral ops port."""
    return Settings(
        db_path=db_path,
        log_file=str(tmp_path / "logs" / "agent.jsonl"),
        artifacts_dir=str(tmp_path / "artifacts"),
        llm_provider="mock",
        mock_llm_latency_ms=0,
        queue_max_size=1000,
        worker_count=2,
        max_retries=3,
        ticket_timeout_seconds=10,
        retry_backoff_base_seconds=0.01,
        retry_backoff_max_seconds=0.05,
        shutdown_grace_seconds=2,
        poll_interval_seconds=0.2,
        poll_jitter_ratio=0.0,
        review_dispatch_interval_seconds=0.1,
        metrics_sample_interval_seconds=0.2,
        browser_timeout_ms=5000,
        user_service_timeout_seconds=1.0,
        user_cache_ttl_seconds=60,
        ops_api_host="127.0.0.1",
        ops_api_port=free_port(),
        ticket_ui_url="http://127.0.0.1:1",  # deliberately unreachable unless a server fixture overrides
        user_service_url="http://127.0.0.1:1",
    )


# --------------------------------------------------------------------------- fakes


@pytest.fixture
def fake_portal() -> FakeTicketPortal:
    return FakeTicketPortal()


@pytest.fixture
def fake_users() -> FakeUserInfoClient:
    return FakeUserInfoClient()


@pytest.fixture
def fake_llm() -> FakeLLMClient:
    return FakeLLMClient()


@pytest.fixture
def fake_policy() -> FakePolicyGate:
    return FakePolicyGate()


# --------------------------------------------------------------------------- uvicorn in a thread


class ServerThread:
    """Runs an ASGI app with uvicorn on a pre-bound ephemeral socket in a daemon thread."""

    def __init__(self, app: Any) -> None:
        import uvicorn

        self.app = app  # tests may poke app.state (chaos knobs, session secret)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.port = self.sock.getsockname()[1]
        assert self.port not in FORBIDDEN_PORTS
        config = uvicorn.Config(app, log_level="warning", lifespan="auto", access_log=False)
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, kwargs={"sockets": [self.sock]}, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout: float = 10.0) -> ServerThread:
        self.thread.start()
        deadline = time.monotonic() + timeout
        while not self.server.started:
            if not self.thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("uvicorn test server failed to start")
            time.sleep(0.02)
        return self

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)
        self.sock.close()


@pytest.fixture
def serve() -> Iterator[Callable[[Any], ServerThread]]:
    """Factory: `srv = serve(app)` → started ServerThread; all stopped at teardown."""
    started: list[ServerThread] = []

    def _serve(app: Any) -> ServerThread:
        srv = ServerThread(app).start()
        started.append(srv)
        return srv

    yield _serve
    for s in reversed(started):
        s.stop()


@pytest.fixture
def start_mock_ui(serve) -> Callable[..., ServerThread]:
    """Factory: `start_mock_ui(seed=True, error_rate=0.2, ...)`. Skips if the mock UI isn't built yet."""

    def _start(**kwargs: Any) -> ServerThread:
        create_app = require("support_agent.mocks.ticket_ui.app", "create_app")
        return serve(create_app(**kwargs))

    return _start


@pytest.fixture
def start_user_service(serve) -> Callable[..., ServerThread]:
    """Factory: `start_user_service(latency_ms=1000, ...)`. Skips if the user service isn't built yet."""

    def _start(**kwargs: Any) -> ServerThread:
        create_app = require("support_agent.mocks.user_service.app", "create_app")
        return serve(create_app(**kwargs))

    return _start


@pytest.fixture
def mock_ui(start_mock_ui) -> ServerThread:
    return start_mock_ui()


@pytest.fixture
def user_service(start_user_service) -> ServerThread:
    return start_user_service()


@pytest.fixture
def live_settings(settings: Settings, mock_ui: ServerThread, user_service: ServerThread) -> Settings:
    """`settings` pointed at freshly started mock UI + user service."""
    return settings.model_copy(update={"ticket_ui_url": mock_ui.url, "user_service_url": user_service.url})
