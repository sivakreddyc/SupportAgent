"""HttpUserInfoClient (PLAN §5.4, A9): X-API-Key, 404 -> None, 5xx/connect retries then UserServiceError,
timeout, TTL cache, URL-encoding, metrics.

Most tests talk to a scripted FastAPI server on an ephemeral port, so they don't depend on whether the
client is built on httpx or httpx2. The respx tests run only if the module uses plain httpx.
Skips at module level until A9 lands.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections import deque

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from support_agent.common.contracts import UserInfoClient, UserServiceError
from support_agent.observability import metrics as m
from tests.conftest import require
from tests.fakes import make_user

HttpUserInfoClient = require("support_agent.clients.user_info", "HttpUserInfoClient")
user_info_mod = require("support_agent.clients.user_info")
USES_HTTPX2 = "httpx2" in inspect.getsource(user_info_mod)


class Script:
    """Per-request behaviour queue. Items: int status | ("delay", s) | ("garbage",). Empty -> normal."""

    def __init__(self) -> None:
        self.queue: deque = deque()
        self.requests: list[dict] = []
        self.users = {u.email: u for u in [make_user("alice@example.com"), make_user("a+b@example.com")]}


def scripted_app(script: Script) -> FastAPI:
    app = FastAPI()

    @app.get("/users/by-email/{email}")
    async def by_email(email: str, request: Request):
        script.requests.append({"email": email, "api_key": request.headers.get("x-api-key")})
        action = script.queue.popleft() if script.queue else None
        if isinstance(action, tuple) and action[0] == "delay":
            await asyncio.sleep(action[1])
        elif isinstance(action, int):
            return JSONResponse({"detail": "scripted"}, status_code=action)
        if request.headers.get("x-api-key") != "dev-key":
            return JSONResponse({"detail": "bad key"}, status_code=401)
        u = script.users.get(email.lower())
        if u is None:
            return JSONResponse({"detail": "not found"}, status_code=404)
        return JSONResponse(u.model_dump(mode="json"))

    return app


@pytest.fixture
def script():
    return Script()


@pytest.fixture
def server(serve, script):
    return serve(scripted_app(script))


@pytest.fixture
async def make_client(settings, server):
    made = []

    def _make(**overrides):
        s = settings.model_copy(update={"user_service_url": server.url, **overrides})
        c = HttpUserInfoClient(s)
        made.append(c)
        return c

    yield _make
    for c in made:
        await c.close()


def metric(name: str, **labels) -> float:
    return m.REGISTRY.get_sample_value(name, labels) or 0.0


async def test_protocol(make_client):
    assert isinstance(make_client(), UserInfoClient)


async def test_found_sends_api_key(make_client, script):
    u = await make_client().get_user_by_email("alice@example.com")
    assert u == script.users["alice@example.com"]
    assert script.requests[0]["api_key"] == "dev-key"


async def test_custom_api_key(make_client, script):
    c = make_client(user_service_api_key="other-key")
    with pytest.raises(UserServiceError):
        await c.get_user_by_email("alice@example.com")  # server answers 401
    assert script.requests[0]["api_key"] == "other-key"
    assert len(script.requests) == 1  # 4xx is not retried


async def test_404_returns_none_without_retry(make_client, script):
    assert await make_client().get_user_by_email("nobody@example.com") is None
    assert len(script.requests) == 1


async def test_email_is_url_encoded(make_client, script):
    u = await make_client().get_user_by_email("a+b@example.com")
    assert u is not None and script.requests[0]["email"] == "a+b@example.com"


@pytest.mark.parametrize("code", [500, 502, 503])
async def test_5xx_retried_then_succeeds(make_client, script, code):
    script.queue.extend([code, code])
    u = await make_client().get_user_by_email("alice@example.com")
    assert u is not None
    assert len(script.requests) == 3


async def test_5xx_exhausts_retries(make_client, script):
    script.queue.extend([503] * 10)
    with pytest.raises(UserServiceError):
        await make_client().get_user_by_email("alice@example.com")
    assert len(script.requests) in (3, 4)  # "3 retries": 3 attempts or 1 + 3 retries


@pytest.mark.parametrize("code", [400, 401, 403, 422])
async def test_4xx_not_retried(make_client, script, code):
    script.queue.append(code)
    with pytest.raises(UserServiceError):
        await make_client().get_user_by_email("alice@example.com")
    assert len(script.requests) == 1


async def test_connection_error(settings):
    c = HttpUserInfoClient(settings.model_copy(update={"user_service_url": "http://127.0.0.1:1"}))
    try:
        t0 = time.perf_counter()
        with pytest.raises(UserServiceError):
            await c.get_user_by_email("alice@example.com")
        assert time.perf_counter() - t0 < 15
    finally:
        await c.close()


async def test_timeout_then_recovery(make_client, script):
    script.queue.append(("delay", 1.0))
    c = make_client(user_service_timeout_seconds=0.2)
    t0 = time.perf_counter()
    u = await c.get_user_by_email("alice@example.com")  # first attempt times out, retry succeeds
    assert u is not None
    assert len(script.requests) == 2
    assert time.perf_counter() - t0 < 5


async def test_persistent_timeout_raises(make_client, script):
    script.queue.extend([("delay", 1.0)] * 10)
    c = make_client(user_service_timeout_seconds=0.15)
    with pytest.raises(UserServiceError):
        await c.get_user_by_email("alice@example.com")


async def test_cache_hit(make_client, script):
    c = make_client()
    a = await c.get_user_by_email("alice@example.com")
    b = await c.get_user_by_email("alice@example.com")
    assert a == b and len(script.requests) == 1


async def test_cache_ttl_expiry(make_client, script):
    c = make_client(user_cache_ttl_seconds=0.15)
    await c.get_user_by_email("alice@example.com")
    await asyncio.sleep(0.25)
    await c.get_user_by_email("alice@example.com")
    assert len(script.requests) == 2


async def test_errors_not_cached(make_client, script):
    script.queue.extend([500] * 10)
    c = make_client()
    with pytest.raises(UserServiceError):
        await c.get_user_by_email("alice@example.com")
    script.queue.clear()
    assert await c.get_user_by_email("alice@example.com") is not None


async def test_metrics(make_client, script):
    found0 = metric("user_lookups_total", result="found")
    nf0 = metric("user_lookups_total", result="not_found")
    err0 = metric("user_lookups_total", result="error")
    hit0 = metric("user_lookups_total", result="cache_hit")
    lat0 = metric("user_lookup_seconds_count")
    c = make_client()
    await c.get_user_by_email("alice@example.com")
    await c.get_user_by_email("alice@example.com")
    await c.get_user_by_email("nobody@example.com")
    script.queue.extend([500] * 10)
    with pytest.raises(UserServiceError):
        await c.get_user_by_email("x@example.com")
    assert metric("user_lookups_total", result="found") - found0 == 1
    assert metric("user_lookups_total", result="not_found") - nf0 == 1
    assert metric("user_lookups_total", result="error") - err0 == 1
    assert metric("user_lookups_total", result="cache_hit") - hit0 == 1
    assert metric("user_lookup_seconds_count") - lat0 >= 2


async def test_concurrent_lookups(make_client, script):
    c = make_client()
    emails = ["alice@example.com", "nobody@example.com"] * 10
    results = await asyncio.gather(*(c.get_user_by_email(e) for e in emails))
    assert sum(r is not None for r in results) == 10


# --------------------------------------------------------------------------- respx (only if plain httpx)

respx_only = pytest.mark.skipif(USES_HTTPX2, reason="client uses httpx2; respx cannot intercept it")


@respx_only
async def test_respx_5xx_then_ok(settings):
    import httpx
    import respx

    s = settings.model_copy(update={"user_service_url": "http://users.test"})
    user = make_user()
    with respx.mock(assert_all_called=True) as mock:
        route = mock.get(url__regex=r"http://users\.test/users/by-email/.*").mock(
            side_effect=[httpx.Response(503), httpx.Response(200, json=user.model_dump(mode="json"))]
        )
        c = HttpUserInfoClient(s)
        try:
            assert await c.get_user_by_email(user.email) == user
        finally:
            await c.close()
        assert route.call_count == 2


@respx_only
async def test_respx_timeout(settings):
    import httpx
    import respx

    s = settings.model_copy(update={"user_service_url": "http://users.test"})
    with respx.mock() as mock:
        mock.get(url__regex=r".*").mock(side_effect=httpx.ReadTimeout("slow"))
        c = HttpUserInfoClient(s)
        try:
            with pytest.raises(UserServiceError):
                await c.get_user_by_email("alice@example.com")
        finally:
            await c.close()
