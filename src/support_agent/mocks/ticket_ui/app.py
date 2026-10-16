"""Mock Ticket UI: a server-rendered HTML portal with no JSON API (docs/PLAN.md §5.1).

Run:  support-agent ticket-ui --port 8001   (uvicorn target ``support_agent.mocks.ticket_ui.app:app``)
Tests: ``create_app(seed=True, latency_ms=None, error_rate=None, page_size=20)``.

All state is in memory and per app instance. Login is ``agent`` / ``agent``; the session is an
itsdangerous-signed cookie. Unauthenticated requests are redirected (303) to ``/login?next=<path>``.

Admin (no login required, for demos and tests):
  ``POST /admin/generate`` form field ``n`` (1..5000), optional ``seed`` and ``policy_share``: inject
  random tickets (see ``generator.py``); ``POST /admin/reset``: restore the seed state.

Chaos knobs: ``latency_ms`` / ``error_rate`` arguments, falling back to env ``MOCK_UI_LATENCY_MS`` /
``MOCK_UI_ERROR_RATE``. Every request except ``/health`` and ``/admin/*`` first sleeps ``latency_ms``,
then with probability ``error_rate`` returns an HTTP 500 HTML page (``data-testid="error-page"``)
*before* the handler runs, so an injected error never half-applies a write. Both values live on
``app.state.latency_ms`` / ``app.state.error_rate`` and can be changed on a running app in tests.

The selector contract (``data-testid`` attributes) is documented in
``support_agent.browser.ticket_portal``; the templates in ``templates/`` are its source of truth.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from itsdangerous import BadSignature, URLSafeSerializer

from . import generator

HERE = Path(__file__).parent
SEED_FILE = HERE / "seed_tickets.json"
TEMPLATES = Jinja2Templates(directory=str(HERE / "templates"))

USERNAME = "agent"
PASSWORD = "agent"
SESSION_COOKIE = "sa_session"
STATUSES = ("open", "pending", "escalated", "closed")
MAX_GENERATE = 5000
CHAOS_EXEMPT = ("/health", "/admin/")  # prefixes skipped by the chaos middleware


def _now() -> datetime:
    return datetime.now(UTC)


def _parse_dt(value: str | datetime) -> datetime:
    dt = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def load_seed() -> list[dict[str, Any]]:
    """Seed tickets as normalised dicts (empty list if the seed file is absent)."""
    if not SEED_FILE.exists():
        return []
    return [normalise_ticket(t) for t in json.loads(SEED_FILE.read_text())]


def normalise_ticket(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticket_id": raw["ticket_id"],
        "subject": raw["subject"],
        "customer_email": raw["customer_email"],
        "created_at": _parse_dt(raw["created_at"]),
        "priority": raw.get("priority"),
        "status": raw.get("status", "open"),
        "body": raw["body"],
        "messages": [
            {"author": m["author"], "body": m["body"], "at": _parse_dt(m["at"])}
            for m in raw.get("messages", [])
        ],
        "drafts": [
            {"body": d["body"], "idempotency_key": d["idempotency_key"], "at": _parse_dt(d["at"])}
            for d in raw.get("drafts", [])
        ],
        "notes": [{"body": n["body"], "at": _parse_dt(n["at"])} for n in raw.get("notes", [])],
    }


class TicketStore:
    """In-memory tickets, ordered by (created_at, ticket_id) so pagination is stable."""

    def __init__(self, tickets: list[dict[str, Any]]) -> None:
        self.tickets: dict[str, dict[str, Any]] = {t["ticket_id"]: t for t in tickets}

    def ordered(self, status: str | None) -> list[dict[str, Any]]:
        rows = sorted(self.tickets.values(), key=lambda t: (t["created_at"], t["ticket_id"]))
        return [t for t in rows if status is None or t["status"] == status]

    def get(self, ticket_id: str) -> dict[str, Any] | None:
        return self.tickets.get(ticket_id)

    def next_number(self) -> int:
        nums = [int(m.group(1)) for t in self.tickets if (m := re.fullmatch(r"TCK-(\d+)", t))]
        return max(nums, default=1000) + 1

    def next_created_at(self) -> datetime:
        """Now, or just after the newest ticket, so a new batch always sorts after existing tickets."""
        newest = max((t["created_at"] for t in self.tickets.values()), default=None)
        now = _now()
        return now if newest is None or now > newest else newest + timedelta(milliseconds=1)

    def add(self, tickets: list[dict[str, Any]]) -> None:
        for t in tickets:
            self.tickets[t["ticket_id"]] = t


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
    page_size: int = 20,
) -> FastAPI:
    app = FastAPI(title="Mock Ticket UI", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store = TicketStore(load_seed() if seed else [])
    app.state.secret = uuid.uuid4().hex
    app.state.page_size = page_size
    app.state.latency_ms = latency_ms if latency_ms is not None else _env_number("MOCK_UI_LATENCY_MS", int, 0)
    app.state.error_rate = (
        error_rate if error_rate is not None else _env_number("MOCK_UI_ERROR_RATE", float, 0.0)
    )
    app.state.seeded = seed
    chaos_rng = random.Random()

    def serializer() -> URLSafeSerializer:
        return URLSafeSerializer(app.state.secret, salt="mock-ticket-ui")

    def current_user(request: Request) -> str | None:
        raw = request.cookies.get(SESSION_COOKIE)
        if not raw:
            return None
        try:
            data = serializer().loads(raw)
        except BadSignature:
            return None
        return data.get("u") if isinstance(data, dict) else None

    def to_login(next_path: str) -> RedirectResponse:
        return RedirectResponse(f"/login?next={quote(next_path, safe='')}", status_code=303)

    def render(request: Request, name: str, ctx: dict[str, Any], status_code: int = 200) -> HTMLResponse:
        ctx = {"user": current_user(request), **ctx}
        return TEMPLATES.TemplateResponse(request, name, ctx, status_code=status_code)

    def not_found(request: Request, what: str) -> HTMLResponse:
        return render(request, "error.html", {"code": 404, "testid": "not-found", "message": what}, 404)

    def bad_request(request: Request, what: str) -> HTMLResponse:
        return render(request, "error.html", {"code": 400, "testid": "bad-request", "message": what}, 400)

    def back_to(ticket_id: str) -> RedirectResponse:
        return RedirectResponse(f"/tickets/{ticket_id}", status_code=303)

    # ---------- chaos ----------

    @app.middleware("http")
    async def chaos(request: Request, call_next):
        if request.url.path.startswith(CHAOS_EXEMPT):
            return await call_next(request)
        if app.state.latency_ms and app.state.latency_ms > 0:
            await asyncio.sleep(app.state.latency_ms / 1000)
        if app.state.error_rate and chaos_rng.random() < app.state.error_rate:
            ctx = {"user": None, "code": 500, "testid": "error-page", "message": "Injected server error"}
            return TEMPLATES.TemplateResponse(request, "error.html", ctx, status_code=500)
        return await call_next(request)

    # ---------- admin ----------

    @app.post("/admin/generate", response_model=None)
    async def admin_generate(
        request: Request,
        n: int = Form(10),
        seed: int | None = Form(None),
        policy_share: float = Form(generator.DEFAULT_POLICY_SHARE),
    ) -> Response:
        if not 1 <= n <= MAX_GENERATE:
            return bad_request(request, f"n must be between 1 and {MAX_GENERATE}")
        if not 0.0 <= policy_share <= 1.0:
            return bad_request(request, "policy_share must be between 0 and 1")
        store: TicketStore = app.state.store
        raw = generator.generate(
            n, seed, policy_share=policy_share, start_number=store.next_number(), now=store.next_created_at()
        )
        store.add([normalise_ticket(t) for t in raw])
        return RedirectResponse(f"/tickets?status=open&generated={n}", status_code=303)

    @app.post("/admin/reset", response_model=None)
    async def admin_reset() -> Response:
        app.state.store = TicketStore(load_seed() if app.state.seeded else [])
        return RedirectResponse("/tickets", status_code=303)

    # ---------- auth ----------

    @app.get("/", include_in_schema=False)
    async def root() -> Response:
        return RedirectResponse("/tickets", status_code=303)

    @app.get("/health", include_in_schema=False)
    async def health() -> Response:
        return Response("ok", media_type="text/plain")

    @app.get("/login", response_class=HTMLResponse)
    async def login_form(request: Request, next: str = "/tickets") -> HTMLResponse:
        return render(request, "login.html", {"next": next, "error": None})

    @app.post("/login", response_model=None)
    async def login_submit(
        request: Request,
        username: str = Form(""),
        password: str = Form(""),
        next: str = Form("/tickets"),
    ) -> Response:
        if username != USERNAME or password != PASSWORD:
            return render(
                request, "login.html", {"next": next, "error": "Invalid username or password."}, 401
            )
        target = next if next.startswith("/") and not next.startswith("//") else "/tickets"
        resp = RedirectResponse(target, status_code=303)
        resp.set_cookie(SESSION_COOKIE, serializer().dumps({"u": username}), httponly=True, samesite="lax")
        return resp

    @app.post("/logout")
    async def logout() -> Response:
        resp = RedirectResponse("/login", status_code=303)
        resp.delete_cookie(SESSION_COOKIE)
        return resp

    # ---------- tickets ----------

    @app.get("/tickets", response_model=None)
    async def list_tickets(request: Request, page: int = 1, status: str | None = None) -> Response:
        if current_user(request) is None:
            return to_login(str(request.url.path) + (f"?{request.url.query}" if request.url.query else ""))
        status = status or None
        if status is not None and status not in STATUSES and status != "all":
            return bad_request(request, f"unknown status {status!r}")
        rows = app.state.store.ordered(None if status in (None, "all") else status)
        size = app.state.page_size
        total_pages = max(1, -(-len(rows) // size))
        page_no = max(1, page)
        chunk = rows[(page_no - 1) * size : page_no * size]
        return render(
            request,
            "tickets.html",
            {
                "tickets": chunk,
                "page_no": page_no,
                "total_pages": total_pages,
                "total": len(rows),
                "status": status,
            },
        )

    @app.get("/tickets/{ticket_id}", response_model=None)
    async def ticket_detail(request: Request, ticket_id: str) -> Response:
        if current_user(request) is None:
            return to_login(f"/tickets/{ticket_id}")
        t = app.state.store.get(ticket_id)
        if t is None:
            return not_found(request, f"Ticket {ticket_id} not found")
        return render(
            request, "ticket_detail.html", {"t": t, "statuses": STATUSES, "default_key": uuid.uuid4().hex}
        )

    @app.post("/tickets/{ticket_id}/draft", response_model=None)
    async def post_draft(
        request: Request, ticket_id: str, body: str = Form(""), idempotency_key: str = Form("")
    ) -> Response:
        if current_user(request) is None:
            return to_login(f"/tickets/{ticket_id}")
        t = app.state.store.get(ticket_id)
        if t is None:
            return not_found(request, f"Ticket {ticket_id} not found")
        if not body.strip():
            return bad_request(request, "draft body is empty")
        key = idempotency_key.strip() or uuid.uuid4().hex
        if not any(d["idempotency_key"] == key for d in t["drafts"]):  # duplicate key = no-op
            t["drafts"].append({"body": body, "idempotency_key": key, "at": _now()})
        return back_to(ticket_id)

    @app.post("/tickets/{ticket_id}/note", response_model=None)
    async def post_note(request: Request, ticket_id: str, body: str = Form("")) -> Response:
        if current_user(request) is None:
            return to_login(f"/tickets/{ticket_id}")
        t = app.state.store.get(ticket_id)
        if t is None:
            return not_found(request, f"Ticket {ticket_id} not found")
        if not body.strip():
            return bad_request(request, "note body is empty")
        t["notes"].append({"body": body, "at": _now()})
        return back_to(ticket_id)

    @app.post("/tickets/{ticket_id}/status", response_model=None)
    async def set_status(request: Request, ticket_id: str, status: str = Form("")) -> Response:
        if current_user(request) is None:
            return to_login(f"/tickets/{ticket_id}")
        t = app.state.store.get(ticket_id)
        if t is None:
            return not_found(request, f"Ticket {ticket_id} not found")
        if status not in STATUSES:
            return bad_request(request, f"unknown status {status!r}")
        t["status"] = status
        return back_to(ticket_id)

    return app


app = create_app()
