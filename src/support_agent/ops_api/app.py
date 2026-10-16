"""Ops API (PLAN §5.11), served in-process by the runtime on settings.ops_api_host:ops_api_port.

  GET /health      200 {"status": "ok"} while the process is up
  GET /ready       200 when DB ok + portal started + last successful poll within 3x poll interval,
                   else 503 {"ready": false, "reasons": [...]}
  GET /metrics     Prometheus text exposition (observability.metrics.REGISTRY)
  GET /api/status  JSON: queue depth/capacity, workers busy/configured, last poll, counts by state, ...

`create_ops_app(runtime)` only needs duck-typed attributes: `settings`, `repo`, `portal_started`,
`portal_error`, `last_poll_at` and `async status()`.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST

from support_agent.observability import metrics as m
from support_agent.observability.logging import get_logger

log = get_logger("ops_api")


async def readiness(runtime: Any) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    repo = getattr(runtime, "repo", None)
    db_ok = False
    if repo is not None:
        try:
            db_ok = bool(await repo.ping())
        except Exception:
            db_ok = False
    if not db_ok:
        reasons.append("database not reachable")
    if not getattr(runtime, "portal_started", False):
        err = getattr(runtime, "portal_error", None)
        reasons.append(f"ticket portal not started{': ' + err if err else ''}")
    last = getattr(runtime, "last_poll_at", None)
    max_age = 3 * float(runtime.settings.poll_interval_seconds)
    if last is None:
        reasons.append("no successful poll yet")
    elif time.time() - last > max_age:
        reasons.append(f"last successful poll {time.time() - last:.1f}s ago (> {max_age:.1f}s)")
    poller = getattr(runtime, "poller", None)
    last_err = (getattr(poller, "last_result", None) or {}).get("error")
    if last_err and reasons:
        reasons.append(f"last poll error: {last_err}")
    return not reasons, reasons


def create_ops_app(runtime: Any) -> FastAPI:
    app = FastAPI(title="SupportAgent ops API", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "time": time.time()}

    @app.get("/ready")
    async def ready() -> JSONResponse:
        ok, reasons = await readiness(runtime)
        return JSONResponse({"ready": ok, "reasons": reasons}, status_code=200 if ok else 503)

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(m.render_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        data = await runtime.status()
        ok, reasons = await readiness(runtime)
        data["ready"] = {"ready": ok, "reasons": reasons}
        return data

    return app


class _InProcessServer(uvicorn.Server):
    """uvicorn.Server that leaves signal handling to the runtime."""

    @contextlib.contextmanager
    def capture_signals(self):  # type: ignore[override]
        yield

    def install_signal_handlers(self) -> None:  # older uvicorn versions
        pass


class OpsServer:
    """Runs `create_ops_app(runtime)` with uvicorn inside the runtime's event loop. Port 0 = ephemeral."""

    def __init__(self, runtime: Any, host: str, port: int) -> None:
        self.host = host
        self.requested_port = port
        config = uvicorn.Config(
            create_ops_app(runtime),
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
            lifespan="off",
        )
        self.server = _InProcessServer(config)
        self._task: asyncio.Task[None] | None = None

    @property
    def port(self) -> int | None:
        servers = getattr(self.server, "servers", None) or []
        for srv in servers:
            for sock in srv.sockets or []:
                return int(sock.getsockname()[1])
        return None

    @property
    def url(self) -> str | None:
        return f"http://{self.host}:{self.port}" if self.port else None

    async def start(self, start_timeout_s: float = 10.0) -> None:
        self._task = asyncio.create_task(self.server.serve(), name="ops_api")
        deadline = time.monotonic() + start_timeout_s
        while not self.server.started:
            if self._task.done():
                exc = self._task.exception() if not self._task.cancelled() else None
                raise RuntimeError(f"ops API failed to start on {self.host}:{self.requested_port}: {exc}")
            if time.monotonic() > deadline:
                raise RuntimeError("ops API start timed out")
            await asyncio.sleep(0.02)
        log.info("ops_api.started", url=self.url)

    async def stop(self) -> None:
        if self._task is None:
            return
        self.server.should_exit = True
        try:
            await asyncio.wait_for(self._task, timeout=5)
        except (TimeoutError, asyncio.CancelledError):
            self._task.cancel()
        except BaseException as e:  # uvicorn raises SystemExit on bind failure
            log.warning("ops_api.stop_error", error=repr(e))
        self._task = None
