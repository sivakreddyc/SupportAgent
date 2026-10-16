"""HTTP client for the User Information Service (docs/PLAN.md §5.4). Implements ``common.contracts.UserInfoClient``.

    client = HttpUserInfoClient(settings)
    user = await client.get_user_by_email("maria.gonzalez@example.com")   # UserInfo | None
    await client.close()

* ``GET {user_service_url}/users/by-email/{email}`` with header ``X-API-Key: {user_service_api_key}`` and
  timeout ``user_service_timeout_seconds``. The email is stripped, lower-cased and URL-encoded.
* 200 -> ``UserInfo`` (validated); 404 -> ``None``.
* 5xx, timeouts and connection errors are retried ``retries`` times (default 3, so up to 4 requests) with
  backoff ``backoff_base * 2**n`` (0.2 s, 0.4 s, 0.8 s). Then ``UserServiceError``.
* Other 4xx (401, 422, ...) and invalid response bodies raise ``UserServiceError`` at once (no retry).
* TTL cache (``user_cache_ttl_seconds``) keyed by the lower-cased email. It holds found users *and*
  404s (negative caching); errors are never cached.
* ``list_users(q=None, limit=100, offset=0) -> (total, [UserInfo])`` wraps ``GET /users`` (same timeout and
  retries, no cache, no lookup metrics, span ``user_info.list``).
* Metrics: ``user_lookups_total{result=found|not_found|error|cache_hit}`` (once per call) and
  ``user_lookup_seconds`` (calls that went to the network, including retries). Span ``user_info.get``.
"""

from __future__ import annotations

import asyncio
import time
from urllib.parse import quote

import httpx
from pydantic import ValidationError
from structlog.contextvars import get_contextvars

from support_agent.common.config import Settings
from support_agent.common.contracts import UserServiceError
from support_agent.common.models import UserInfo
from support_agent.observability import metrics, tracing
from support_agent.observability.logging import get_logger

log = get_logger("user_info")

MAX_CACHE_ENTRIES = 10_000


class _Retryable(Exception):
    pass


class HttpUserInfoClient:
    """``UserInfoClient`` over httpx with retries and a small TTL cache."""

    def __init__(
        self,
        settings: Settings,
        *,
        retries: int = 3,
        backoff_base: float = 0.2,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.settings = settings
        self.retries = max(0, retries)
        self.backoff_base = max(0.0, backoff_base)
        self.ttl = float(settings.user_cache_ttl_seconds)
        self._cache: dict[str, tuple[float, UserInfo | None]] = {}
        self._client = httpx.AsyncClient(
            base_url=settings.user_service_url.rstrip("/"),
            timeout=httpx.Timeout(settings.user_service_timeout_seconds),
            headers={"X-API-Key": settings.user_service_api_key, "Accept": "application/json"},
            transport=transport,
        )
        self._closed = False

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            await self._client.aclose()

    def clear_cache(self) -> None:
        self._cache.clear()

    # ------------------------------------------------------------------ cache

    def _cache_get(self, key: str) -> tuple[bool, UserInfo | None]:
        entry = self._cache.get(key)
        if entry is None:
            return False, None
        expires, value = entry
        if time.monotonic() >= expires:
            self._cache.pop(key, None)
            return False, None
        return True, value

    def _cache_put(self, key: str, value: UserInfo | None) -> None:
        if self.ttl <= 0:
            return
        if len(self._cache) >= MAX_CACHE_ENTRIES:
            now = time.monotonic()
            for k in [k for k, (exp, _) in self._cache.items() if exp <= now]:
                del self._cache[k]
            while len(self._cache) >= MAX_CACHE_ENTRIES:  # still full: drop the oldest insertions
                del self._cache[next(iter(self._cache))]
        self._cache[key] = (time.monotonic() + self.ttl, value)

    # ------------------------------------------------------------------ lookup

    async def get_user_by_email(self, email: str) -> UserInfo | None:
        key = email.strip().lower()
        trace_id = get_contextvars().get("trace_id")
        ticket_id = get_contextvars().get("ticket_id")
        async with tracing.span("user_info.get", trace_id=trace_id, ticket_id=ticket_id) as sp:
            hit, cached = self._cache_get(key)
            if hit:
                metrics.USER_LOOKUPS.labels(result="cache_hit").inc()
                sp.set(result="cache_hit", found=cached is not None)
                return cached
            t0 = time.perf_counter()
            try:
                user = await self._fetch(key, sp)
            except UserServiceError:
                metrics.USER_LOOKUPS.labels(result="error").inc()
                sp.set(result="error")
                raise
            finally:
                metrics.USER_LOOKUP_SECONDS.observe(time.perf_counter() - t0)
            self._cache_put(key, user)
            result = "found" if user is not None else "not_found"
            metrics.USER_LOOKUPS.labels(result=result).inc()
            sp.set(result=result)
            return user

    async def _fetch(self, key: str, sp: tracing.Span) -> UserInfo | None:
        if not key:
            return None
        resp = await self._get(f"/users/by-email/{quote(key, safe='@')}", sp=sp)
        if resp is None:
            return None
        try:
            return UserInfo.model_validate(resp.json())
        except (ValueError, ValidationError) as e:
            raise UserServiceError(f"invalid user service response: {e}") from e

    async def list_users(
        self, q: str | None = None, limit: int = 100, offset: int = 0
    ) -> tuple[int, list[UserInfo]]:
        """``GET /users``: (total matches, one page of users sorted by user_id). Never cached.

        ``q`` is a case-insensitive substring of email, name or user_id; ``limit`` 1..1000, ``offset`` >= 0
        (out of range raises ``UserServiceError``, the service answers 422). Same timeout and retries as
        ``get_user_by_email``.
        """
        params: dict[str, str | int] = {"limit": limit, "offset": offset}
        if q:
            params["q"] = q
        trace_id = get_contextvars().get("trace_id")
        async with tracing.span(
            "user_info.list", trace_id=trace_id, q=q or "", limit=limit, offset=offset
        ) as sp:
            resp = await self._get("/users", params=params, sp=sp)
            if resp is None:
                raise UserServiceError("user service returned 404 for GET /users")
            try:
                data = resp.json()
                total, items = int(data["total"]), [UserInfo.model_validate(u) for u in data["items"]]
            except (ValueError, KeyError, TypeError, ValidationError) as e:
                raise UserServiceError(f"invalid user service response: {e}") from e
            sp.set(total=total, returned=len(items))
            return total, items

    async def _get(
        self, path: str, *, params: dict[str, str | int] | None = None, sp: tracing.Span | None = None
    ) -> httpx.Response | None:
        """GET with retries on 5xx / transport errors. 200 -> response, 404 -> None, other -> error."""
        if self._closed:
            raise UserServiceError("HttpUserInfoClient is closed")
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            if sp is not None:
                sp.set(attempts=attempt + 1)
            try:
                return await self._request_once(path, params)
            except _Retryable as e:
                last = e
            if attempt < self.retries:
                delay = self.backoff_base * (2**attempt)
                log.warning(
                    "user_info.retry",
                    path=path,
                    attempt=attempt + 1,
                    delay_s=round(delay, 3),
                    error=str(last),
                )
                await asyncio.sleep(delay)
        raise UserServiceError(
            f"user service request failed after {self.retries + 1} attempts: {last}"
        ) from last

    async def _request_once(self, path: str, params: dict[str, str | int] | None) -> httpx.Response | None:
        try:
            resp = await self._client.get(path, params=params)
        except httpx.TransportError as e:  # connect errors, timeouts, protocol errors
            raise _Retryable(f"{type(e).__name__}: {e}") from e
        if resp.status_code == 404:
            return None
        if resp.status_code >= 500:
            raise _Retryable(f"HTTP {resp.status_code}")
        if resp.status_code != 200:
            raise UserServiceError(f"user service returned HTTP {resp.status_code}: {resp.text[:200]}")
        return resp
