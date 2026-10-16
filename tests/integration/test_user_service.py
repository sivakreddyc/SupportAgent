"""Mock User Info Service (PLAN §5.2, A4) over real HTTP: auth, seed users per profile, 404 path,
orders, chaos knobs. Plus HttpUserInfoClient against the real service once A9 lands."""

from __future__ import annotations

import time

import httpx
import pytest

from support_agent.common.contracts import UserServiceError
from support_agent.common.models import Order, UserInfo
from tests.conftest import require
from tests.fakes import load_seed_cases

KEY = {"X-API-Key": "dev-key"}
SEEDS = load_seed_cases()
KNOWN = [c for c in SEEDS if c.profile != "unknown"]
UNKNOWN = [c for c in SEEDS if c.profile == "unknown"]


@pytest.fixture
def svc(user_service):
    with httpx.Client(base_url=user_service.url, timeout=5) as c:
        yield c


def test_health_needs_no_key(svc):
    r = svc.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}])
def test_api_key_required(svc, headers):
    assert svc.get(f"/users/by-email/{KNOWN[0].ticket.customer_email}", headers=headers).status_code == 401


def test_custom_api_key(start_user_service):
    srv = start_user_service(api_key="s3cret")
    with httpx.Client(base_url=srv.url) as c:
        email = KNOWN[0].ticket.customer_email
        assert c.get(f"/users/by-email/{email}", headers=KEY).status_code == 401
        assert c.get(f"/users/by-email/{email}", headers={"X-API-Key": "s3cret"}).status_code == 200


@pytest.mark.parametrize("case", KNOWN, ids=[c.ticket.ticket_id for c in KNOWN])
def test_seed_user_matches_ticket_profile(svc, case):
    r = svc.get(f"/users/by-email/{case.ticket.customer_email}", headers=KEY)
    assert r.status_code == 200
    u = UserInfo(**r.json())
    assert u.email.lower() == case.ticket.customer_email.lower()
    if case.profile == "enterprise":
        assert u.plan == "enterprise"
    elif case.profile == "vip":
        assert u.vip is True
    elif case.profile in ("suspended", "closed"):
        assert u.account_status == case.profile
    else:  # standard: nothing that would trip VIP / ACCOUNT_NOT_ACTIVE
        assert u.plan != "enterprise" and not u.vip and u.account_status == "active"


@pytest.mark.parametrize("case", UNKNOWN, ids=[c.ticket.ticket_id for c in UNKNOWN])
def test_unknown_customers_404(svc, case):
    assert svc.get(f"/users/by-email/{case.ticket.customer_email}", headers=KEY).status_code == 404


def test_email_case_insensitive(svc):
    email = KNOWN[0].ticket.customer_email
    assert svc.get(f"/users/by-email/{email.upper()}", headers=KEY).status_code == 200


def test_user_by_id_and_orders(svc):
    u = UserInfo(**svc.get(f"/users/by-email/{KNOWN[0].ticket.customer_email}", headers=KEY).json())
    assert UserInfo(**svc.get(f"/users/{u.user_id}", headers=KEY).json()) == u
    orders = [Order(**o) for o in svc.get(f"/users/{u.user_id}/orders", headers=KEY).json()]
    assert orders == sorted(orders, key=lambda o: o.created_at, reverse=True)
    open_ids = {o.order_id for o in u.open_orders}
    assert open_ids == {o.order_id for o in orders if o.status in ("pending", "paid", "shipped")}


def test_unknown_user_id_404(svc):
    assert svc.get("/users/U-nope", headers=KEY).status_code == 404
    assert svc.get("/users/U-nope/orders", headers=KEY).status_code == 404


def test_refund_tickets_have_a_paid_order(svc):
    refund_cases = [c for c in KNOWN if "REFUND" in c.expected_rules]
    assert refund_cases
    for c in refund_cases:
        u = UserInfo(**svc.get(f"/users/by-email/{c.ticket.customer_email}", headers=KEY).json())
        uid = u.user_id
        orders = [Order(**o) for o in svc.get(f"/users/{uid}/orders", headers=KEY).json()]
        assert any(o.status in ("paid", "delivered", "shipped", "refunded") for o in orders), (
            c.ticket.ticket_id
        )


def test_error_rate_knob(start_user_service):
    srv = start_user_service(error_rate=1.0)
    with httpx.Client(base_url=srv.url) as c:
        assert c.get("/health").status_code == 200  # exempt
        assert c.get(f"/users/by-email/{KNOWN[0].ticket.customer_email}", headers=KEY).status_code == 500
        srv.app.state.error_rate = 0.0
        assert c.get(f"/users/by-email/{KNOWN[0].ticket.customer_email}", headers=KEY).status_code == 200


def test_latency_knob(start_user_service):
    srv = start_user_service(latency_ms=200)
    with httpx.Client(base_url=srv.url) as c:
        t0 = time.perf_counter()
        c.get(f"/users/by-email/{KNOWN[0].ticket.customer_email}", headers=KEY)
        assert time.perf_counter() - t0 >= 0.19


def test_unseeded_service_is_empty(start_user_service):
    srv = start_user_service(seed=False)
    with httpx.Client(base_url=srv.url) as c:
        assert c.get(f"/users/by-email/{KNOWN[0].ticket.customer_email}", headers=KEY).status_code == 404


# --------------------------------------------------------------------------- real client vs real service


@pytest.fixture
async def client(settings, user_service):
    HttpUserInfoClient = require("support_agent.clients.user_info", "HttpUserInfoClient")
    c = HttpUserInfoClient(settings.model_copy(update={"user_service_url": user_service.url}))
    yield c
    await c.close()


async def test_client_against_service_every_seed(client):
    for case in SEEDS:
        u = await client.get_user_by_email(case.ticket.customer_email)
        assert (u is None) == (case.profile == "unknown"), case.ticket.ticket_id


async def test_client_with_chaos(settings, start_user_service):
    HttpUserInfoClient = require("support_agent.clients.user_info", "HttpUserInfoClient")
    srv = start_user_service(error_rate=1.0)
    c = HttpUserInfoClient(settings.model_copy(update={"user_service_url": srv.url}))
    try:
        with pytest.raises(UserServiceError):
            await c.get_user_by_email(KNOWN[0].ticket.customer_email)
    finally:
        await c.close()


# --------------------------------------------------------------------------- GET /users (Phase 5 U1)


def test_list_users_default_page(svc):
    r = svc.get("/users", headers=KEY)
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 50 and len(body["items"]) == 50  # default limit 100 > 50 users
    ids = [u["user_id"] for u in body["items"]]
    assert ids == sorted(ids)
    UserInfo(**body["items"][0])
    assert {u["email"].lower() for u in body["items"]} == {c.ticket.customer_email.lower() for c in KNOWN}


def test_list_users_requires_key(svc):
    assert svc.get("/users").status_code == 401


def test_list_users_pagination(svc):
    first = svc.get("/users", params={"limit": 20, "offset": 0}, headers=KEY).json()
    second = svc.get("/users", params={"limit": 20, "offset": 20}, headers=KEY).json()
    tail = svc.get("/users", params={"limit": 20, "offset": 40}, headers=KEY).json()
    past = svc.get("/users", params={"limit": 20, "offset": 500}, headers=KEY).json()
    assert first["total"] == second["total"] == 50
    assert len(first["items"]) == len(second["items"]) == 20 and len(tail["items"]) == 10
    assert past["items"] == [] and past["total"] == 50
    all_ids = [u["user_id"] for p in (first, second, tail) for u in p["items"]]
    assert len(set(all_ids)) == 50


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 1001}, {"offset": -1}, {"limit": "x"}])
def test_list_users_bad_params_422(svc, params):
    assert svc.get("/users", params=params, headers=KEY).status_code == 422


def test_list_users_search(svc):
    u = UserInfo(**svc.get(f"/users/by-email/{KNOWN[0].ticket.customer_email}", headers=KEY).json())
    for q in (u.email.upper(), u.name.split()[0].lower(), u.user_id.lower()):
        body = svc.get("/users", params={"q": q}, headers=KEY).json()
        assert u.user_id in [x["user_id"] for x in body["items"]], q
        assert body["total"] == len(body["items"])
    assert svc.get("/users", params={"q": "zz-no-such-user"}, headers=KEY).json() == {"total": 0, "items": []}


def test_list_users_chaos_applies(start_user_service):
    srv = start_user_service(error_rate=1.0)
    with httpx.Client(base_url=srv.url) as c:
        assert c.get("/users", headers=KEY).status_code == 500


async def test_client_list_users(client):
    total, items = await client.list_users()
    assert total == 50 and len(items) == 50 and all(isinstance(u, UserInfo) for u in items)
    total, page = await client.list_users(limit=10, offset=45)
    assert total == 50 and len(page) == 5
    total, found = await client.list_users(q=KNOWN[0].ticket.customer_email)
    assert total == 1 and found[0].email.lower() == KNOWN[0].ticket.customer_email.lower()


async def test_client_list_users_errors(client, settings):
    with pytest.raises(UserServiceError):
        await client.list_users(limit=0)  # 422
    HttpUserInfoClient = require("support_agent.clients.user_info", "HttpUserInfoClient")
    down = HttpUserInfoClient(
        settings.model_copy(update={"user_service_url": "http://127.0.0.1:1"}), retries=0
    )
    try:
        with pytest.raises(UserServiceError):
            await down.list_users()
    finally:
        await down.close()


async def test_client_list_users_not_cached(client, user_service):
    await client.list_users()
    user_service.app.state.error_rate = 1.0
    with pytest.raises(UserServiceError):
        await client.list_users()
