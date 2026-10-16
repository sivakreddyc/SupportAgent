"""Mock User Information Service: a FastAPI JSON API (docs/PLAN.md §5.2).

Run:  support-agent user-service --port 8002   (uvicorn target ``support_agent.mocks.user_service.app:app``)
Tests: ``create_app(seed=True, latency_ms=None, error_rate=None, api_key=None)``.

Endpoints (all but ``/health`` and the OpenAPI docs need header ``X-API-Key``, default ``dev-key``; 401 otherwise):
  GET /users/by-email/{email}   -> UserInfo        (case-insensitive; 404 if unknown)
  GET /users/{user_id}          -> UserInfo        (404 if unknown)
  GET /users/{user_id}/orders   -> list[Order]     (all orders, newest first; 404 if unknown user)
  GET /health                   -> {"status": "ok"}

``UserInfo.open_orders`` holds the orders that are still pending, paid or shipped.

Seed data (``seed_users.json``) is derived from the seed tickets' ``meta.customer_profile``: one user per
known ticket email (standard / enterprise / vip / suspended / closed); "unknown" customers have no
record, so they get a 404. Refund tickets have a paid order matching the amount in the ticket text.

Chaos knobs: ``latency_ms`` / ``error_rate`` arguments, falling back to env ``MOCK_US_LATENCY_MS`` /
``MOCK_US_ERROR_RATE``; ``api_key`` falls back to env ``MOCK_US_API_KEY``. Every request except
``/health`` and the docs first sleeps ``latency_ms``, then with probability ``error_rate`` returns
HTTP 500 JSON. Both values live on ``app.state`` and can be changed on a running app in tests.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from support_agent.common.models import Order, UserInfo

SEED_FILE = Path(__file__).parent / "seed_users.json"
DEFAULT_API_KEY = "dev-key"
OPEN_ORDER_STATUSES = {"pending", "paid", "shipped"}
CHAOS_EXEMPT = ("/health", "/docs", "/openapi.json", "/redoc")  # prefixes skipped by the chaos middleware


class UserRecord:
    """One seed user: the public ``UserInfo`` plus the full order history."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.orders = sorted(
            (Order(**o) for o in raw.get("orders", [])), key=lambda o: o.created_at, reverse=True
        )
        fields = {k: v for k, v in raw.items() if k in UserInfo.model_fields and k != "open_orders"}
        self.info = UserInfo(
            **fields, open_orders=[o for o in self.orders if o.status in OPEN_ORDER_STATUSES]
        )


def load_seed() -> list[UserRecord]:
    if not SEED_FILE.exists():
        return []
    return [UserRecord(u) for u in json.loads(SEED_FILE.read_text())]


def _env_number(name: str, cast: type, default: float) -> Any:
    raw = os.environ.get(name, "").strip()
    try:
        return cast(raw) if raw else cast(default)
    except ValueError:
        return cast(default)


def create_app(
    seed: bool = True,
    latency_ms: int | None = None,
    error_rate: float | None = None,
    api_key: str | None = None,
) -> FastAPI:
    app = FastAPI(title="Mock User Info Service")
    records = load_seed() if seed else []
    app.state.by_id = {r.info.user_id: r for r in records}
    app.state.by_email = {r.info.email.lower(): r for r in records}
    app.state.latency_ms = latency_ms if latency_ms is not None else _env_number("MOCK_US_LATENCY_MS", int, 0)
    app.state.error_rate = (
        error_rate if error_rate is not None else _env_number("MOCK_US_ERROR_RATE", float, 0.0)
    )
    app.state.api_key = api_key or os.environ.get("MOCK_US_API_KEY") or DEFAULT_API_KEY
    chaos_rng = random.Random()

    @app.middleware("http")
    async def chaos(request: Request, call_next):
        if request.url.path.startswith(CHAOS_EXEMPT):
            return await call_next(request)
        if app.state.latency_ms and app.state.latency_ms > 0:
            await asyncio.sleep(app.state.latency_ms / 1000)
        if app.state.error_rate and chaos_rng.random() < app.state.error_rate:
            return JSONResponse({"detail": "injected server error"}, status_code=500)
        return await call_next(request)

    async def require_key(x_api_key: str | None = Header(default=None)) -> None:
        if x_api_key != app.state.api_key:
            raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")

    def record(user_id: str) -> UserRecord:
        rec = app.state.by_id.get(user_id)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"user {user_id} not found")
        return rec

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/users/by-email/{email}", response_model=UserInfo, dependencies=[Depends(require_key)])
    async def user_by_email(email: str) -> UserInfo:
        rec = app.state.by_email.get(email.strip().lower())
        if rec is None:
            raise HTTPException(status_code=404, detail="user not found")
        return rec.info

    @app.get("/users/{user_id}", response_model=UserInfo, dependencies=[Depends(require_key)])
    async def user_by_id(user_id: str) -> UserInfo:
        return record(user_id).info

    @app.get("/users/{user_id}/orders", response_model=list[Order], dependencies=[Depends(require_key)])
    async def user_orders(user_id: str) -> list[Order]:
        return record(user_id).orders

    return app


app = create_app()
