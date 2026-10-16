"""Playwright page object for the ticket UI (docs/PLAN.md §5.3). Implements ``common.contracts.TicketPortal``.

    portal = PlaywrightTicketPortal(settings)
    await portal.start()            # one headless Chromium + two contexts + writer page pool
    ...
    await portal.close()            # idempotent

Browser layout
--------------
* One ``Browser`` (Chromium, ``settings.browser_headless``) per portal, launched once in ``start()``
  (``BROWSER_LAUNCHES`` +1). Default action/navigation timeout is ``settings.browser_timeout_ms``.
* **Discovery context**: one page, serialised by an ``asyncio.Lock``; used only by ``list_open_tickets``
  so posting can never starve discovery.
* **Writer context**: a pool of ``settings.browser_page_pool`` pages (min 1) guarded by an
  ``asyncio.Semaphore``; used by ``get_ticket`` and the write operations. A page that crashed or was
  closed is replaced when it is returned to the pool.
* Both contexts use ``settings.ticket_ui_url`` as ``base_url`` and have their own cookie jar, so each
  logs in on its own (and again automatically whenever it is redirected to ``/login``).

Calling an operation before ``start()`` or after ``close()`` raises ``RuntimeError`` (a wiring bug,
not a portal failure). Unrecoverable UI failures raise ``common.contracts.PortalError``.

Resilience and instrumentation
------------------------------
* Every public op runs in ``tracing.span("browser.<op>")`` (trace_id taken from the structlog context
  bound by the worker) and records ``browser_ops_total{op,result=ok|error}`` and ``browser_op_seconds{op}``.
* Retries: up to ``retry_attempts`` (3) per op, and per list page for ``list_open_tickets``, with backoff
  ``retry_base_delay * 2**(attempt-1)`` (0.5 s, 1 s). Retried: HTTP 5xx pages, unexpected pages,
  Playwright timeouts and navigation errors, and a session that expired between loading a form and
  posting it. Not retried (``PortalError`` at once): 404 / other 4xx, rejected login, invalid input.
* After the final failed attempt a full-page screenshot and an HTML dump are written to
  ``settings.artifacts_dir`` (``<utc>_<op>_<ticket>_<rand>.png|.html``); the paths are logged
  (``browser.op_failed``) and included in the ``PortalError`` message.
* Idempotency: ``post_draft`` re-reads the detail page on *every* attempt and skips the submit when a
  ``draft`` with the same ``data-idempotency-key`` is present, so a retry, or a retried worker, never
  double-posts. ``post_internal_note`` skips a retry when the newest note already has the same text.
  ``set_status`` is a no-op when the status already matches.

Selector contract (mock ticket UI, ``mocks/ticket_ui/templates``)
-------------------------------------------------------------------
This module is the only code that knows these selectors. ``[data-testid=X]`` is written ``X`` below.

Routes:
  ``GET  /login?next=<path>``          login page (any unauthenticated request is 303-redirected here)
  ``GET  /tickets?status=open&page=N`` list page, 20 rows per page, oldest first; a page past the end is empty
  ``GET  /tickets/{id}``               detail page (404 page if unknown)
  ``POST /tickets/{id}/draft|note|status`` form posts, 303 back to the detail page (PRG)

Login page:
  ``login-form``; inputs ``login-username`` (name=username), ``login-password`` (name=password),
  hidden ``login-next`` (name=next); button ``login-submit``; ``login-error`` on bad credentials.

List page:
  ``ticket-table``; one ``tr`` ``ticket-row`` per ticket with attributes ``data-ticket-id`` and
  ``data-status``; cells ``ticket-link`` (the id, ``a href=/tickets/{id}``), ``ticket-row-subject``,
  ``ticket-row-email``, ``ticket-row-priority``, ``ticket-status-badge``, and ``ticket-row-created-at``
  (``<time datetime="ISO-8601">``). ``empty-state`` when the page has no rows. ``pagination`` with
  ``data-page`` / ``data-total-pages``; ``prev-page`` / ``next-page`` links. ``ticket-count``
  (``data-total``).

Detail page (root ``ticket-detail`` with ``data-ticket-id``):
  ``ticket-id``, ``ticket-subject``, ``ticket-customer-email``,
  ``ticket-created-at`` (``<time datetime="ISO-8601">``), ``ticket-priority`` (``data-priority``,
  empty if none), ``ticket-status`` (``data-status``), ``ticket-body`` (exact text via textContent).
  Thread: ``message-thread`` > ``message`` (``data-author``, ``data-at`` ISO) > ``message-body``.
  Drafts: ``draft-list`` > ``draft`` (``data-idempotency-key``, ``data-at``) > ``draft-body``.
  Notes:  ``note-list`` > ``note`` (``data-at``) > ``note-body``.
  Draft form ``draft-form``: textarea ``draft-body-input`` (``#draft-body``, name=body), hidden
  ``draft-idempotency-key`` (name=idempotency_key), button ``draft-submit``.
  Note form ``note-form``: textarea ``note-body-input`` (``#note-body``, name=body), button ``note-submit``.
  Status form ``status-form``: select ``status-select`` (name=status; open|pending|escalated|closed),
  button ``status-submit``.

Error pages: ``error-page`` (HTTP 500, also injected by the chaos knob), ``not-found`` (404),
``bad-request`` (400); each has ``data-status-code`` and an ``error-message``.

Not used by the agent: ``nav-tickets``, ``nav-open-tickets``, ``current-user``, ``logout``,
``admin-panel``, ``admin-generate-*``, ``admin-reset-*``.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import quote, urlparse

from playwright.async_api import Browser, BrowserContext, Page, Playwright, Response, async_playwright
from playwright.async_api import Error as PlaywrightError
from structlog.contextvars import get_contextvars

from support_agent.common.config import Settings
from support_agent.common.contracts import PortalError
from support_agent.common.models import Ticket, TicketSummary
from support_agent.observability import metrics, tracing
from support_agent.observability.logging import get_logger
from support_agent.observability.tracing import Span

log = get_logger("browser")

T = TypeVar("T")
MAX_LIST_PAGES = 10_000  # safety stop for list pagination
STATUSES = ("open", "pending", "escalated", "closed")

_LIST_JS = """() => {
  if (!document.querySelector('[data-testid="ticket-table"]')) return null;
  const txt = (r, id) => { const el = r.querySelector(`[data-testid="${id}"]`); return el ? el.textContent : ""; };
  return Array.from(document.querySelectorAll('[data-testid="ticket-row"]')).map(r => ({
    ticket_id: r.dataset.ticketId,
    status: r.dataset.status,
    subject: txt(r, "ticket-row-subject"),
    customer_email: txt(r, "ticket-row-email"),
    created_at: r.querySelector('[data-testid="ticket-row-created-at"]').getAttribute("datetime"),
  }));
}"""

_DETAIL_JS = """() => {
  const root = document.querySelector('[data-testid="ticket-detail"]');
  if (!root) return null;
  const q = (sel, el) => (el || root).querySelector(`[data-testid="${sel}"]`);
  const all = (sel) => Array.from(root.querySelectorAll(`[data-testid="${sel}"]`));
  return {
    ticket_id: root.dataset.ticketId,
    subject: q("ticket-subject").textContent,
    customer_email: q("ticket-customer-email").textContent,
    created_at: q("ticket-created-at").getAttribute("datetime"),
    priority: q("ticket-priority").dataset.priority || null,
    status: q("ticket-status").dataset.status,
    body: q("ticket-body").textContent,
    messages: all("message").map(m => ({
      author: m.dataset.author, at: m.dataset.at, body: q("message-body", m).textContent })),
    drafts: all("draft").map(d => ({ key: d.dataset.idempotencyKey, body: q("draft-body", d).textContent })),
    notes: all("note").map(n => q("note-body", n).textContent),
  };
}"""


class _RetryableError(Exception):
    """Transient UI failure (5xx page, unexpected page); the operation may be retried."""


class _FatalError(PortalError):
    """Permanent failure (unknown ticket, bad credentials, invalid input); never retried."""


class PlaywrightTicketPortal:
    """``TicketPortal`` backed by headless Chromium driving the server-rendered ticket UI."""

    def __init__(self, settings: Settings, *, retry_attempts: int = 3, retry_base_delay: float = 0.5) -> None:
        """``retry_attempts`` tries per operation (per list page for listing), backoff
        ``retry_base_delay * 2**(attempt-1)`` seconds between tries (0.5 s, 1 s by default)."""
        self.settings = settings
        self.retry_attempts = max(1, retry_attempts)
        self.retry_base_delay = max(0.0, retry_base_delay)
        self.base_url = settings.ticket_ui_url.rstrip("/")
        self.pool_size = max(1, int(settings.browser_page_pool))
        self._pw: Playwright | None = None
        self._browser: Browser | None = None
        self._discovery_ctx: BrowserContext | None = None
        self._writer_ctx: BrowserContext | None = None
        self._discovery_page: Page | None = None
        self._discovery_lock = asyncio.Lock()
        self._pool_sem = asyncio.Semaphore(self.pool_size)
        self._free_pages: asyncio.Queue[Page] = asyncio.Queue()
        self._start_lock = asyncio.Lock()
        self._started = False

    # ------------------------------------------------------------------ lifecycle

    @property
    def started(self) -> bool:
        return self._started

    async def start(self) -> None:
        """Launch Chromium once and build both contexts and the writer page pool. Idempotent."""
        async with self._start_lock:
            if self._started:
                return
            try:
                self._pw = await async_playwright().start()
                self._browser = await self._pw.chromium.launch(headless=self.settings.browser_headless)
                metrics.BROWSER_LAUNCHES.inc()
                self._discovery_ctx = await self._new_context()
                self._writer_ctx = await self._new_context()
                self._discovery_page = await self._discovery_ctx.new_page()
                self._pool_sem = asyncio.Semaphore(self.pool_size)
                self._free_pages = asyncio.Queue()
                for _ in range(self.pool_size):
                    self._free_pages.put_nowait(await self._writer_ctx.new_page())
            except Exception as e:
                await self._teardown()
                raise PortalError(f"failed to start browser: {type(e).__name__}: {e}") from e
            self._started = True
            log.info(
                "browser.started",
                base_url=self.base_url,
                headless=self.settings.browser_headless,
                page_pool=self.pool_size,
            )

    async def close(self) -> None:
        """Close pages, contexts, the browser and Playwright. Safe to call more than once."""
        async with self._start_lock:
            was_started = self._started
            self._started = False
            await self._teardown()
            if was_started:
                log.info("browser.closed")

    async def _new_context(self) -> BrowserContext:
        assert self._browser is not None
        ctx = await self._browser.new_context(base_url=self.base_url)
        ctx.set_default_timeout(self.settings.browser_timeout_ms)
        ctx.set_default_navigation_timeout(self.settings.browser_timeout_ms)
        return ctx

    async def _teardown(self) -> None:
        for name in ("_discovery_ctx", "_writer_ctx", "_browser"):
            obj = getattr(self, name)
            setattr(self, name, None)
            if obj is not None:
                try:
                    await obj.close()
                except Exception as e:  # already gone (browser crashed, etc.)
                    log.warning("browser.close_failed", target=name, error=str(e))
        if self._pw is not None:
            pw, self._pw = self._pw, None
            try:
                await pw.stop()
            except Exception as e:
                log.warning("browser.close_failed", target="playwright", error=str(e))
        self._discovery_page = None
        self._free_pages = asyncio.Queue()

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("PlaywrightTicketPortal is not started: call `await portal.start()` first")

    # ------------------------------------------------------------------ page access

    @asynccontextmanager
    async def _writer_page(self) -> AsyncIterator[Page]:
        """Borrow a page from the writer pool; at most ``pool_size`` are in use at once."""
        self._require_started()
        async with self._pool_sem:
            self._require_started()  # close() may have run while we waited
            page = self._free_pages.get_nowait()
            try:
                yield page
            finally:
                if self._started and self._writer_ctx is not None:
                    if page.is_closed():
                        log.warning("browser.page_replaced")
                        page = await self._writer_ctx.new_page()
                    self._free_pages.put_nowait(page)

    @asynccontextmanager
    async def _discovery(self) -> AsyncIterator[Page]:
        """The single discovery page, one user at a time."""
        self._require_started()
        async with self._discovery_lock:
            self._require_started()
            assert self._discovery_ctx is not None
            if self._discovery_page is None or self._discovery_page.is_closed():
                self._discovery_page = await self._discovery_ctx.new_page()
            yield self._discovery_page

    # ------------------------------------------------------------------ navigation + auth

    @staticmethod
    def _on_login_page(page: Page) -> bool:
        return urlparse(page.url).path.rstrip("/") == "/login"

    async def _login(self, page: Page) -> Response | None:
        """Submit the login form on the current /login page; the UI redirects back to ``next``."""
        log.info("browser.login", context="discovery" if page.context is self._discovery_ctx else "writer")
        await page.get_by_test_id("login-username").fill(self.settings.ticket_ui_user)
        await page.get_by_test_id("login-password").fill(self.settings.ticket_ui_password)
        async with page.expect_navigation(wait_until="domcontentloaded") as nav:
            await page.get_by_test_id("login-submit").click()
        resp = await nav.value
        # Classify the response before looking at the URL: an (injected) 5xx on POST /login is also
        # served at /login, and must be retried, not mistaken for rejected credentials (BUG-1).
        if resp is not None and resp.status >= 500:
            raise _RetryableError(f"POST /login: HTTP {resp.status}")
        if self._on_login_page(page):
            if await page.get_by_test_id("login-error").count():
                raise _FatalError(
                    "login rejected by the ticket UI (check ticket_ui_user / ticket_ui_password)"
                )
            raise _RetryableError(
                f"POST /login: still on the login page (HTTP {resp.status if resp else '?'})"
            )
        return resp

    def _check_response(self, resp: Response | None, what: str) -> None:
        if resp is None:
            return
        if resp.status >= 500:
            raise _RetryableError(f"{what}: HTTP {resp.status}")
        if resp.status == 404:
            raise _FatalError(f"{what}: not found (HTTP 404)")
        if resp.status >= 400:
            raise _FatalError(f"{what}: HTTP {resp.status}")

    async def _goto(self, page: Page, path: str) -> Response | None:
        """Navigate to ``path``; log in again if redirected to /login; classify error pages."""
        resp = await page.goto(path, wait_until="domcontentloaded")
        self._check_response(resp, f"GET {path}")
        if self._on_login_page(page):
            resp = await self._login(page)
            self._check_response(resp, f"GET {path} (after login)")
        return resp

    async def _submit(self, page: Page, submit_testid: str, what: str) -> None:
        """Click a form's submit button and wait for the PRG redirect back to the detail page."""
        async with page.expect_navigation(wait_until="domcontentloaded") as nav:
            await page.get_by_test_id(submit_testid).click()
        self._check_response(await nav.value, what)
        if self._on_login_page(page):
            # the session expired between loading the form and posting it: nothing was written.
            # Log in again now; the retry re-runs the whole operation (including its pre-checks).
            await self._login(page)
            raise _RetryableError(f"{what}: session expired during submit")

    async def _parse_current_ticket(self, page: Page, ticket_id: str) -> tuple[Ticket, list[str]]:
        data: dict[str, Any] | None = await page.evaluate(_DETAIL_JS)
        if data is None or data.get("ticket_id") != ticket_id:
            raise _RetryableError(f"ticket {ticket_id}: detail page not found (url={page.url})")
        drafts = data.pop("drafts")
        return Ticket(**data, drafts=[d["body"] for d in drafts]), [d["key"] for d in drafts]

    # ------------------------------------------------------------------ page readers

    async def _read_list_page(self, page: Page, page_no: int) -> list[TicketSummary]:
        await self._goto(page, f"/tickets?status=open&page={page_no}")
        rows = await page.evaluate(_LIST_JS)
        if rows is None:
            raise _RetryableError(f"list page {page_no}: ticket table not found (url={page.url})")
        return [TicketSummary(**r) for r in rows]

    async def _read_ticket(self, page: Page, ticket_id: str) -> tuple[Ticket, list[str]]:
        """Parse the detail page; returns the ticket and the idempotency keys of its drafts."""
        await self._goto(page, f"/tickets/{quote(ticket_id, safe='')}")
        return await self._parse_current_ticket(page, ticket_id)

    # ------------------------------------------------------------------ resilience + instrumentation

    @asynccontextmanager
    async def _op(self, op: str, ticket_id: str | None = None) -> AsyncIterator[Span]:
        """Span ``browser.<op>`` + ``browser_ops_total{op,result}`` + ``browser_op_seconds{op}``."""
        trace_id = get_contextvars().get("trace_id")
        t0 = time.perf_counter()
        result = "error"
        try:
            async with tracing.span(f"browser.{op}", trace_id=trace_id, ticket_id=ticket_id) as sp:
                yield sp
            result = "ok"
        finally:
            metrics.BROWSER_OPS.labels(op=op, result=result).inc()
            metrics.BROWSER_OP_SECONDS.labels(op=op).observe(time.perf_counter() - t0)

    async def _attempts(
        self,
        op: str,
        page: Page,
        fn: Callable[[int], Awaitable[T]],
        *,
        ticket_id: str | None = None,
        sp: Span | None = None,
    ) -> T:
        """Run ``fn(attempt)`` on ``page`` up to ``retry_attempts`` times with exponential backoff.

        Retried: 5xx / unexpected pages, Playwright timeouts and navigation errors. ``_FatalError``
        (404, bad input, bad credentials) is raised at once. After the last failed attempt a screenshot
        and an HTML dump are written to ``settings.artifacts_dir`` and ``PortalError`` is raised.
        """
        last: Exception | None = None
        for attempt in range(1, self.retry_attempts + 1):
            if sp is not None:
                sp.set(attempts=attempt)
            try:
                return await fn(attempt)
            except _FatalError:
                raise
            except (_RetryableError, PlaywrightError) as e:
                last = e
                if page.is_closed():
                    break  # the pool replaces the page; retrying on a dead page is pointless
                if attempt < self.retry_attempts:
                    delay = self.retry_base_delay * (2 ** (attempt - 1))
                    log.warning(
                        "browser.retry",
                        op=op,
                        ticket_id=ticket_id,
                        attempt=attempt,
                        delay_s=round(delay, 3),
                        error=f"{type(e).__name__}: {e}",
                    )
                    await asyncio.sleep(delay)
        artifacts = await self._save_artifacts(page, op, ticket_id)
        msg = f"{op} failed after {attempt} attempt(s): {type(last).__name__}: {last}"
        log.error("browser.op_failed", op=op, ticket_id=ticket_id, error=msg, artifacts=artifacts)
        raise PortalError(f"{msg} (artifacts: {', '.join(artifacts) or 'none'})") from last

    async def _save_artifacts(self, page: Page, op: str, ticket_id: str | None) -> list[str]:
        """Screenshot + HTML dump of the page at failure time. Never raises."""
        if page.is_closed():
            return []
        out_dir = Path(self.settings.artifacts_dir)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        stem = out_dir / f"{stamp}_{op}_{ticket_id or 'na'}_{uuid.uuid4().hex[:6]}"
        saved: list[str] = []
        try:
            await asyncio.to_thread(out_dir.mkdir, parents=True, exist_ok=True)
            await page.screenshot(path=f"{stem}.png", full_page=True, timeout=5000)
            saved.append(f"{stem}.png")
        except Exception as e:
            log.warning("browser.screenshot_failed", op=op, error=str(e))
        try:
            html = await page.content()
            await asyncio.to_thread(Path(f"{stem}.html").write_text, f"<!-- url: {page.url} -->\n{html}")
            saved.append(f"{stem}.html")
        except Exception as e:
            log.warning("browser.html_dump_failed", op=op, error=str(e))
        if saved:
            log.info("browser.artifacts_saved", op=op, ticket_id=ticket_id, paths=saved)
        return saved

    # ------------------------------------------------------------------ TicketPortal ops

    async def list_open_tickets(self) -> list[TicketSummary]:
        """All open tickets, paginating ``/tickets?status=open`` until a page comes back empty.

        Each list page is retried on its own, so a transient error does not restart the walk.
        """
        async with self._op("list_open_tickets") as sp:
            self._require_started()
            found: dict[str, TicketSummary] = {}
            async with self._discovery() as page:
                pages = 0
                for page_no in range(1, MAX_LIST_PAGES + 1):
                    rows = await self._attempts(
                        "list_open_tickets", page, lambda _a, n=page_no: self._read_list_page(page, n)
                    )
                    pages += 1
                    if not rows:
                        break
                    for r in rows:  # dedupe: rows can shift between pages while tickets change
                        found.setdefault(r.ticket_id, r)
            sp.set(pages=pages, found=len(found))
            return list(found.values())

    async def get_ticket(self, ticket_id: str) -> Ticket:
        async with self._op("get_ticket", ticket_id) as sp:
            self._require_started()
            async with self._writer_page() as page:

                async def attempt(_a: int) -> Ticket:
                    ticket, _ = await self._read_ticket(page, ticket_id)
                    return ticket

                return await self._attempts("get_ticket", page, attempt, ticket_id=ticket_id, sp=sp)

    async def post_draft(self, ticket_id: str, body: str, *, idempotency_key: str) -> None:
        """Post a draft reply once per ``idempotency_key``.

        Every attempt reads the detail page first: if a draft with this key is already there (an earlier
        attempt, or a retried worker, posted it), nothing is submitted. The UI also ignores duplicate keys.
        """
        async with self._op("post_draft", ticket_id) as sp:
            self._require_started()
            if not body.strip():
                raise _FatalError(f"ticket {ticket_id}: refusing to post an empty draft")
            if not idempotency_key:
                raise _FatalError(f"ticket {ticket_id}: idempotency_key is required")
            async with self._writer_page() as page:

                async def attempt(_a: int) -> None:
                    _, keys = await self._read_ticket(page, ticket_id)
                    if idempotency_key in keys:
                        sp.set(skipped_duplicate=True)
                        log.info(
                            "browser.draft_already_posted",
                            ticket_id=ticket_id,
                            idempotency_key=idempotency_key,
                        )
                        return
                    await page.get_by_test_id("draft-body-input").fill(body)
                    await page.get_by_test_id("draft-idempotency-key").evaluate(
                        "(el, v) => { el.value = v; }", idempotency_key
                    )
                    await self._submit(page, "draft-submit", f"POST draft {ticket_id}")
                    _, keys = await self._parse_current_ticket(page, ticket_id)
                    if idempotency_key not in keys:
                        raise _RetryableError(f"ticket {ticket_id}: draft not visible after submit")

                await self._attempts("post_draft", page, attempt, ticket_id=ticket_id, sp=sp)

    async def post_internal_note(self, ticket_id: str, note: str) -> None:
        """Add an internal note.

        Notes have no idempotency key. On a retry, if the newest note already equals ``note`` (the
        previous attempt's POST landed but its response was lost), it is not posted again.
        """
        async with self._op("post_internal_note", ticket_id) as sp:
            self._require_started()
            if not note.strip():
                raise _FatalError(f"ticket {ticket_id}: refusing to post an empty note")
            async with self._writer_page() as page:

                async def attempt(n: int) -> None:
                    ticket, _ = await self._read_ticket(page, ticket_id)
                    if n > 1 and ticket.notes and ticket.notes[-1] == note:
                        sp.set(skipped_duplicate=True)
                        return
                    before = len(ticket.notes)
                    await page.get_by_test_id("note-body-input").fill(note)
                    await self._submit(page, "note-submit", f"POST note {ticket_id}")
                    ticket, _ = await self._parse_current_ticket(page, ticket_id)
                    if len(ticket.notes) <= before:
                        raise _RetryableError(f"ticket {ticket_id}: note not visible after submit")

                await self._attempts("post_internal_note", page, attempt, ticket_id=ticket_id, sp=sp)

    async def set_status(self, ticket_id: str, status: str) -> None:
        """Set the ticket status (open | pending | escalated | closed). No-op if already set."""
        async with self._op("set_status", ticket_id) as sp:
            self._require_started()
            if status not in STATUSES:
                raise _FatalError(
                    f"ticket {ticket_id}: invalid status {status!r} (expected one of {STATUSES})"
                )
            async with self._writer_page() as page:

                async def attempt(_a: int) -> None:
                    ticket, _ = await self._read_ticket(page, ticket_id)
                    if ticket.status == status:
                        return
                    await page.get_by_test_id("status-select").select_option(status)
                    await self._submit(page, "status-submit", f"POST status {ticket_id}")
                    ticket, _ = await self._parse_current_ticket(page, ticket_id)
                    if ticket.status != status:
                        raise _RetryableError(f"ticket {ticket_id}: status is {ticket.status!r} after submit")

                await self._attempts("set_status", page, attempt, ticket_id=ticket_id, sp=sp)
