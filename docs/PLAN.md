# SupportAgent — Design Spec & Build Plan

> Status tracking lives in [TASKS.md](TASKS.md). This file is the source of truth for *what* we build;
> TASKS.md is the source of truth for *where we are*.

## 1. Goal

An autonomous support agent that:

1. Opens a ticket web UI that has **no API** using a headless Chromium browser (Playwright + Python), and discovers new tickets.
2. Pushes discovered tickets onto a **bounded, configurable queue** (default size 1000).
3. Runs a pool of **workers** that take tickets off the queue, enrich them with user data from a **User Information Service** (FastAPI), and call an **LLM** for a structured JSON analysis + draft reply.
4. Runs every result through a **policy gate** (low confidence, refunds, legal, security, angry sentiment → human review).
5. **Posts the draft back** to the ticket UI through Playwright, and stores every draft and every state transition in **SQLite**.
6. Exposes **observability** (structured JSON logs, metrics, per-ticket traces) and a **Streamlit** ops dashboard with a human-review queue.

Target from README: respond within ~1s of pickup where possible. The LLM call dominates latency, so model and effort are configurable (see §6).

## 2. Architecture

```
                        ┌──────────────────────────────────────────────────────────────┐
                        │                    Agent Runtime (asyncio)                   │
                        │                                                              │
 ┌──────────────┐  HTML │  ┌───────────┐   ┌──────────────┐   ┌──────────────────┐     │
 │ Mock Ticket  │◄──────┼──┤ Discovery │──►│ Bounded Queue│──►│  Worker Pool (N) │     │
 │ UI  :8001    │ (PW)  │  │  Poller   │   │ maxsize=1000 │   │                  │     │
 │ (no JSON API)│       │  └─────▲─────┘   └──────────────┘   │ 1. read ticket   │     │
 └──────▲───────┘       │        │ tick                       │ 2. get user info ├─────┼──► User Info Svc :8002 (FastAPI)
        │               │  ┌─────┴─────┐                      │ 3. call LLM      ├─────┼──► LLM (Anthropic / Mock)
        │  post draft   │  │ Scheduler │                      │ 4. policy gate   │     │
        └───────────────┼──┤ (interval)│                      │ 5. post draft/   │     │
          (Playwright)  │  └───────────┘                      │    escalate      │     │
                        │                                     └────────┬─────────┘     │
                        │   Observability: JSON logs · metrics · spans │               │
                        │   Ops API :8003  (/health, /metrics)         │               │
                        └──────────────────────────────────────────────┼───────────────┘
                                                                       ▼
                                                           ┌──────────────────────┐
                                                           │ SQLite (WAL)         │
                                                           │ tickets, transitions,│
                                                           │ drafts, llm_calls,   │
                                                           │ spans, metric_samples│
                                                           └──────────▲───────────┘
                                                                      │ read + review actions
                                                           ┌──────────┴───────────┐
                                                           │ Streamlit UI :8501   │
                                                           └──────────────────────┘
```

**Processes**

| Process | Port | Tech |
|---|---|---|
| Mock Ticket UI | 8001 | FastAPI + Jinja2, server-rendered HTML only |
| Mock User Info Service | 8002 | FastAPI, JSON API |
| Agent runtime (scheduler, poller, queue, workers, ops API) | 8003 | asyncio, Playwright async API, httpx, uvicorn |
| Ops dashboard | 8501 | Streamlit (reads SQLite, writes review decisions) |

SQLite is the only shared state between the runtime and the dashboard. It runs in WAL mode so the dashboard can read while workers write.

## 3. Repository layout

```
pyproject.toml                    # one package, extras: [dev], [llm]
config/
  settings.yaml                   # defaults; every key overridable by env var SA_<KEY>
  policy.yaml                     # policy-gate rules (thresholds, keyword lists)
src/support_agent/
  common/
    config.py                     # pydantic-settings Settings
    models.py                     # shared pydantic models (Ticket, UserInfo, LLMAnalysis, PolicyDecision, ...)
    states.py                     # TicketState enum + allowed transitions
    contracts.py                  # Protocols: TicketPortal, UserInfoClient, LLMClient, PolicyGate, Repository
  mocks/
    ticket_ui/   app.py, templates/*.html, seed_tickets.json, generator.py
    user_service/ app.py, seed_users.json
  browser/
    ticket_portal.py              # Playwright page object implementing TicketPortal
  clients/
    user_info.py                  # httpx async client implementing UserInfoClient
  storage/
    schema.sql, db.py, repository.py
  pipeline/
    queue.py                      # BoundedTicketQueue
    scheduler.py                  # interval scheduler with jitter + overlap guard
    poller.py                     # DiscoveryPoller
    worker.py                     # Worker + WorkerPool
    runtime.py                    # wires everything; entrypoint `support-agent run`
  llm/
    schema.py                     # LLMAnalysis output schema
    prompts.py
    anthropic_client.py           # real provider
    mock_client.py                # deterministic offline provider (default in tests)
  policy/
    gate.py, rules.py
  observability/
    logging.py                    # structlog JSON
    metrics.py                    # prometheus_client registry + SQLite sampler
    tracing.py                    # span context manager -> spans table + log
  ops_api/app.py                  # /health, /ready, /metrics (Prometheus text)
  dashboard/
    app.py, pages/*.py
tests/
  unit/ integration/ e2e/ load/ conftest.py
scripts/
  run_all.sh                      # start mocks + runtime + dashboard
docs/
  PLAN.md, TASKS.md
```

## 4. Shared contracts (Phase 0 — written before agents split)

These are frozen before parallel work starts so the three build agents and the test agent code against the same interfaces.

### 4.1 Models (`common/models.py`)

```python
class TicketSummary(BaseModel):  # what the list page shows
    ticket_id: str
    subject: str
    customer_email: str
    status: str
    created_at: datetime


class Ticket(TicketSummary):  # what the detail page shows
    body: str
    messages: list[TicketMessage]
    priority: str | None


class UserInfo(BaseModel):
    user_id: str
    email: str
    name: str
    plan: Literal["free", "pro", "enterprise"]
    account_status: Literal["active", "suspended", "closed"]
    signup_date: date
    lifetime_value: float
    open_orders: list[Order]
    recent_refunds: int
    vip: bool


class LLMAnalysis(BaseModel):  # the JSON the LLM must return
    category: Literal["billing", "refund", "technical", "account", "legal", "security", "feedback", "other"]
    intent: str
    sentiment: Literal["positive", "neutral", "frustrated", "angry"]
    urgency: Literal["low", "medium", "high", "critical"]
    confidence: float  # 0..1
    refund_requested: bool
    refund_amount: float | None
    legal_flag: bool
    security_flag: bool
    draft_reply: str
    reasoning_summary: str
    suggested_actions: list[str]


class PolicyDecision(BaseModel):
    outcome: Literal["auto_approve", "human_review", "reject"]
    reasons: list[str]  # rule ids that fired, e.g. ["LOW_CONFIDENCE","ANGRY"]


class WorkItem(BaseModel):  # what sits in the queue
    ticket_id: str
    enqueued_at: datetime
    attempt: int = 0
    trace_id: str
```

### 4.2 Protocols (`common/contracts.py`)

```python
class TicketPortal(Protocol):
    async def list_open_tickets(self) -> list[TicketSummary]: ...
    async def get_ticket(self, ticket_id: str) -> Ticket: ...
    async def post_draft(self, ticket_id: str, body: str, *, idempotency_key: str) -> None: ...
    async def post_internal_note(self, ticket_id: str, note: str) -> None: ...
    async def set_status(self, ticket_id: str, status: str) -> None: ...


class UserInfoClient(Protocol):
    async def get_user_by_email(self, email: str) -> UserInfo | None: ...


class LLMClient(Protocol):
    async def analyze(self, ticket: Ticket, user: UserInfo | None) -> LLMResult: ...

    # LLMResult = analysis + usage(input/output tokens) + latency_ms + model + stop_reason


class PolicyGate(Protocol):
    def evaluate(self, ticket: Ticket, result: LLMResult, user: UserInfo | None) -> PolicyDecision: ...
```

The authoritative versions are in `src/support_agent/common/` (`models.py`, `contracts.py`, `states.py`). That includes the `Repository` Protocol and the `PortalError` / `UserServiceError` / `LLMError(retryable=...)` exception types.

### 4.3 Ticket state machine (`common/states.py`)

```
DISCOVERED ─► QUEUED ─► PROCESSING ─► ENRICHED ─► ANALYZED ─► POLICY_CHECKED ─┬─► DRAFT_POSTED ─► DONE
     ▲            │                                                            └─► NEEDS_REVIEW ─┬─► APPROVED ─► DRAFT_POSTED
     │            │                                                                              └─► REJECTED ─► DONE
     └──(queue    └─► any step can go ─► FAILED ─(attempt < MAX_RETRIES)─► QUEUED
         full)                               └─(attempt ≥ MAX_RETRIES)──► DEAD_LETTER
```

Every transition is written to `ticket_transitions` in the same transaction that updates `tickets.state`. Illegal transitions raise `InvalidTransition`.

## 5. Components

### 5.1 Mock Ticket UI (`mocks/ticket_ui`) — no JSON API on purpose

- `GET /login` + `POST /login` — simple form login (cookie session), so the browser layer must handle auth like a real portal.
- `GET /tickets?page=N` — HTML table, paginated (20/page), each row has `data-ticket-id`, status badge.
- `GET /tickets/{id}` — detail page: subject, customer email, body, message thread, existing drafts/notes.
- `POST /tickets/{id}/draft` — form submit (textarea `#draft-body`, hidden `idempotency_key`); duplicate key = no-op.
- `POST /tickets/{id}/note` — internal note form.
- `POST /tickets/{id}/status` — status dropdown.
- `POST /admin/generate?n=K` (HTML form button) — inject K new random tickets for demos/load tests.
- Optional chaos knobs via env: `MOCK_UI_LATENCY_MS`, `MOCK_UI_ERROR_RATE` (returns 500 page) to exercise retries.
- Seed: ~50 tickets covering every policy path (refund, legal threat, security/breach, angry, vague/low-confidence, happy path, unknown customer).
- Stable selectors: every element the agent touches gets a `data-testid`.

### 5.2 Mock User Info Service (`mocks/user_service`)

- `GET /users/by-email/{email}` → `UserInfo` or 404.
- `GET /users/{user_id}` → `UserInfo`.
- `GET /users/{user_id}/orders` → orders list.
- `GET /health`.
- API-key header `X-API-Key`; latency/error injection env vars like the UI mock.
- Seed users matching the seeded ticket emails, plus some tickets whose email has no user (→ 404 path).

### 5.3 Browser layer (`browser/ticket_portal.py`)

- One Playwright `Browser` (headless Chromium) per runtime; separate `BrowserContext` for poller vs. writers.
- **Page pool** (size `BROWSER_PAGE_POOL`, default 3) guarded by an `asyncio.Semaphore`, so workers can post in parallel without spawning unbounded tabs.
- Auto re-login when redirected to `/login`.
- Paginates the list page until it hits an empty page.
- Retries with exponential backoff on navigation timeout / 5xx; every call is wrapped in a span and counted in metrics (`browser_ops_total{op,result}`).
- Screenshot + HTML dump to `artifacts/` on failure (path logged).
- `post_draft` checks the detail page for the idempotency key before submitting, so a retried worker never double-posts.

### 5.4 User info client (`clients/user_info.py`)

httpx `AsyncClient` with timeout (2s), 3 retries on 5xx/connect errors, returns `None` on 404. Short TTL in-memory cache (60s) keyed by email.

### 5.5 Storage (`storage/`)

SQLite, WAL, `busy_timeout=5000`, accessed through `aiosqlite` from the runtime and plain `sqlite3` from Streamlit.

```sql
tickets(ticket_id PK, subject, customer_email, state, attempt, priority,
        discovered_at, updated_at, last_error, trace_id)
ticket_transitions(id PK, ticket_id FK, from_state, to_state, at, worker_id, detail_json)
ticket_snapshots(ticket_id PK, body, messages_json, fetched_at)
user_lookups(ticket_id PK, found BOOL, user_json, latency_ms, at)
llm_calls(id PK, ticket_id, model, input_tokens, output_tokens, latency_ms,
          stop_reason, success BOOL, error, analysis_json, at)
policy_decisions(ticket_id PK, outcome, reasons_json, at)
drafts(id PK, ticket_id, body, source  -- 'llm' | 'human_edit'
       , status  -- 'pending_review' | 'posted' | 'rejected'
       , idempotency_key UNIQUE, posted_at, reviewer, created_at)
review_actions(id PK, ticket_id, action, reviewer, edited_body, at)
spans(id PK, trace_id, ticket_id, name, start_ts, duration_ms, status, attrs_json)
metric_samples(ts, name, labels_json, value)      -- periodic snapshot for the dashboard
scheduler_runs(id PK, started_at, finished_at, found, enqueued, skipped_full, error)
```

`Repository` exposes intent-level methods (`transition(ticket_id, to, detail)`, `upsert_discovered(...)`, `record_llm_call(...)`, `pending_reviews()`, ...). No raw SQL outside `storage/`.

### 5.6 Bounded queue (`pipeline/queue.py`)

- Wraps `asyncio.Queue(maxsize=QUEUE_MAX_SIZE)`; `QUEUE_MAX_SIZE` defaults to **1000** when unset, empty, or ≤0.
- `try_put(item) -> bool` (non-blocking; used by the poller) and `put(item, timeout)` (used by retries).
- When full, the poller does **not** block: the ticket stays `DISCOVERED`, `queue_rejections_total` is incremented, and the next poll picks it up again (backpressure).
- On startup, the runtime re-enqueues tickets stuck in `QUEUED`/`PROCESSING`/`FAILED`(retryable) so nothing is lost on a crash. SQLite is the durable record; the queue is just the work buffer.
- Gauges: `queue_depth`, `queue_capacity`, `queue_utilization`; histogram `queue_wait_seconds` (enqueue → dequeue).

### 5.7 Scheduler + Discovery poller (`pipeline/scheduler.py`, `pipeline/poller.py`)

- Scheduler fires every `POLL_INTERVAL_SECONDS` (default 15) with ±10% jitter; never overlaps (skips a tick if the previous poll is still running, counted as `scheduler_skipped_total`).
- Poller: `list_open_tickets()` → upsert unseen ids as `DISCOVERED` → `try_put` each → `QUEUED`.
- Dedup on `ticket_id`: a ticket already past `DISCOVERED` is never re-enqueued by the poller.
- Each run is written to `scheduler_runs` and emits `scheduler_runs_total`, `poller_runs_total{result}`, `poller_tickets_found`, `poller_duration_seconds`, `browser_launches_total`.

### 5.8 Workers (`pipeline/worker.py`)

- `WORKER_COUNT` async workers (default 4), each with an id, a loop `get → process → task_done`.
- Pipeline per ticket, each step a span and a state transition:
  1. `PROCESSING` — `portal.get_ticket` → snapshot saved.
  2. `ENRICHED` — `user_info.get_user_by_email` (missing user is allowed; policy handles it).
  3. `ANALYZED` — `llm.analyze` → `llm_calls` row (tokens, latency, stop_reason).
  4. `POLICY_CHECKED` — `policy.evaluate` → `policy_decisions` row.
  5a. auto_approve → `drafts(status=posted)` + `portal.post_draft` → `DRAFT_POSTED` → `DONE`.
  5b. human_review → `drafts(status=pending_review)` + `portal.post_internal_note("Escalated: <reasons>")` → `NEEDS_REVIEW`.
- Failures: exception → `FAILED` with `last_error`, `attempt += 1`, re-queued after backoff (`2^attempt` s, capped) until `MAX_RETRIES` (default 3), then `DEAD_LETTER`.
- Per-ticket timeout `TICKET_TIMEOUT_SECONDS` (default 60).
- Graceful shutdown: stop scheduler → drain queue up to `SHUTDOWN_GRACE_SECONDS` → cancel workers → close browser.
- Metrics: `workers_configured`, `workers_busy`, `tickets_processed_total{outcome}`, `ticket_e2e_seconds` (discovered → final), `ticket_processing_seconds` (dequeue → final), `step_seconds{step}`.

### 5.9 LLM integration (`llm/`)

- **Provider: Anthropic Python SDK** (`anthropic`), model `claude-opus-5-5` by default, configurable via `LLM_MODEL` (e.g. `claude-sonnet-5-5` or `claude-haiku-5-5` for lower latency, given the ~1s target).
- Structured output via `client.messages.parse(..., output_format=LLMAnalysis)` so the response is validated against the pydantic schema; no hand-rolled JSON parsing.
- `output_config.effort` configurable (`LLM_EFFORT`, default `medium`); `max_tokens` 4096.
- Server-side refusal fallback enabled (`betas=["server-side-fallback-2026-07-01"]`, `fallbacks="default"`); any `stop_reason == "refusal"` that still comes back routes the ticket to human review.
- System prompt (frozen, cache-friendly): role, tone guide, company policy summary, "never promise refunds/legal outcomes", "output must follow the schema". Volatile data (ticket, user JSON) goes in the user message only.
- Typed error handling: `RateLimitError`/5xx/`APIConnectionError` → retry; `BadRequestError` → fail the ticket without retry.
- Records `usage.input_tokens`, `usage.output_tokens`, cache read tokens, latency, model.
- **`MockLLMClient`** (`LLM_PROVIDER=mock`, the default for tests and local demo without a key): deterministic keyword rules produce a valid `LLMAnalysis` with fake token counts, so the whole system runs offline.

### 5.10 Policy gate (`policy/`)

Rules are loaded from `config/policy.yaml`; each has an id, and every rule that fires is recorded.

| Rule id | Condition | Outcome |
|---|---|---|
| `LOW_CONFIDENCE` | `confidence < 0.75` | human_review |
| `REFUND` | `refund_requested` or category == refund | human_review |
| `REFUND_HIGH_VALUE` | `refund_amount > 100` | human_review (priority high) |
| `LEGAL` | `legal_flag` or category == legal or keywords (lawyer, lawsuit, sue, GDPR request…) | human_review |
| `SECURITY` | `security_flag` or category == security or keywords (breach, hacked, password leak…) | human_review |
| `ANGRY` | sentiment == angry | human_review |
| `CRITICAL_URGENCY` | urgency == critical | human_review |
| `VIP` | `user.vip` or plan == enterprise | human_review |
| `UNKNOWN_USER` | user lookup returned none | human_review |
| `ACCOUNT_NOT_ACTIVE` | account suspended/closed | human_review |
| `UNSAFE_DRAFT` | draft contains PII of other users, card numbers, or promises ("we will refund", "guarantee") | human_review |
| `LLM_REFUSAL` | stop_reason == refusal | human_review |

Anything with no rule firing → `auto_approve`. Keyword checks run on the raw ticket text as well as LLM flags, so the gate doesn't depend only on the model getting it right.

### 5.11 Observability (`observability/`)

- **Logs**: structlog, JSON to stdout + rotating file `logs/agent.jsonl`. Every log line carries `ts, level, event, component, trace_id, ticket_id, worker_id` where applicable.
- **Tracing**: one `trace_id` per ticket created at discovery; `span(name)` context manager writes to `spans` and logs `span.end` with duration. Gives a per-ticket timeline (poll → queue wait → fetch → user → llm → policy → post).
- **Metrics** (prometheus_client, served at `:8003/metrics`, and sampled every 5s into `metric_samples` for Streamlit history):

| Area | Metrics |
|---|---|
| Scheduler | `scheduler_ticks_total`, `scheduler_skipped_total`, `scheduler_last_run_timestamp` |
| Poller / browser | `poller_runs_total{result}`, `poller_duration_seconds`, `poller_tickets_found`, `browser_launches_total`, `browser_ops_total{op,result}`, `browser_op_seconds{op}` |
| Queue | `queue_depth`, `queue_capacity`, `queue_enqueued_total`, `queue_rejections_total`, `queue_wait_seconds` |
| Workers | `workers_configured`, `workers_busy`, `worker_utilization` |
| Tickets | `tickets_discovered_total`, `tickets_processed_total{outcome}`, `tickets_failed_total{step}`, `tickets_dead_letter_total`, `ticket_e2e_seconds`, `ticket_processing_seconds`, `step_seconds{step}` |
| User service | `user_lookups_total{result}`, `user_lookup_seconds` |
| LLM | `llm_calls_total{model,result}`, `llm_tokens_total{model,direction}`, `llm_latency_seconds`, `llm_refusals_total` |
| Policy | `policy_decisions_total{outcome}`, `policy_rule_hits_total{rule}` |
| Drafts | `drafts_posted_total{source}`, `reviews_total{action}` |

Health: `/health` (process up), `/ready` (DB ok, browser ok, last poll < 3× interval).

### 5.12 Streamlit dashboard (`dashboard/`)

Auto-refresh every 5s. Pages:

1. **Overview** — KPI tiles (discovered, processed, auto-approved %, in review, failed, DLQ), throughput over time, p50/p95 ticket latency.
2. **Tickets** — filterable table by state; click a ticket → timeline of transitions + spans, ticket text, user info, LLM JSON, policy reasons, drafts.
3. **Human Review** — pending queue sorted by priority; show draft + reasons; **Approve / Edit & Approve / Reject**. Approval writes `review_actions` and sets `APPROVED`; the runtime's review dispatcher (polls the DB every few seconds) posts the draft via Playwright and moves the ticket to `DRAFT_POSTED`.
4. **Queue & Workers** — depth vs. capacity over time, rejections, queue wait, busy/configured workers, a "workers saturated" hint when utilization stays > 90%.
5. **LLM** — calls, failures, refusals, tokens in/out over time, latency histogram, tokens per ticket.
6. **Scheduler & Poller** — runs, duration, tickets found per run, browser op success rate, last error + screenshot link.
7. **Logs** — tail of `agent.jsonl` with level/component/ticket filters.

### 5.13 Configuration (`config/settings.yaml`, env `SA_*`)

| Key | Default |
|---|---|
| `QUEUE_MAX_SIZE` | 1000 |
| `WORKER_COUNT` | 4 |
| `POLL_INTERVAL_SECONDS` | 15 |
| `MAX_RETRIES` | 3 |
| `TICKET_TIMEOUT_SECONDS` | 60 |
| `BROWSER_HEADLESS` | true |
| `BROWSER_PAGE_POOL` | 3 |
| `TICKET_UI_URL` / `TICKET_UI_USER` / `TICKET_UI_PASSWORD` | http://localhost:8001 / agent / agent |
| `USER_SERVICE_URL` / `USER_SERVICE_API_KEY` | http://localhost:8002 / dev-key |
| `LLM_PROVIDER` | mock (`anthropic` for real calls; reads `ANTHROPIC_API_KEY`) |
| `LLM_MODEL` | claude-opus-5-5 |
| `LLM_EFFORT` | medium |
| `DB_PATH` | data/support_agent.db |
| `LOG_LEVEL` | INFO |
| `POLICY_FILE` | config/policy.yaml |

## 6. Tech stack

Python 3.11+, Playwright (async), FastAPI + uvicorn, Jinja2, httpx, aiosqlite, pydantic v2 + pydantic-settings, anthropic, structlog, prometheus_client, Streamlit + pandas/altair, pytest + pytest-asyncio + respx, ruff. `uv` for env management.

## 7. Agent team & work split

Four agents. Phase 0 is done first by the lead (single agent) so everyone builds against frozen contracts; then the three builders and the tester run in parallel on **disjoint directories**.

### Phase 0 — Lead (sequential, before fan-out)
Scaffold `pyproject.toml`, layout, `common/` (config, models, states, contracts), `config/*.yaml`, `storage/schema.sql`, empty module stubs, `scripts/run_all.sh` skeleton, CI command (`ruff` + `pytest`).

### Agent A — "Edge" : mocks + browser + user client
Owns: `mocks/`, `browser/`, `clients/`.
Delivers: Mock Ticket UI, Mock User Info Service, seed data covering every policy path, `PlaywrightTicketPortal`, `HttpUserInfoClient`.

### Agent B — "Core" : storage + pipeline
Owns: `storage/`, `pipeline/`, `ops_api/`.
Delivers: Repository + state machine enforcement, BoundedTicketQueue, Scheduler, DiscoveryPoller, Worker/WorkerPool, retries/DLQ, review dispatcher, runtime wiring + CLI, crash recovery.

### Agent C — "Brain & Eyes" : LLM + policy + observability + dashboard
Owns: `llm/`, `policy/`, `observability/`, `dashboard/`.
Delivers: Anthropic client + MockLLM, prompts, policy gate + YAML rules, logging/metrics/tracing modules, Streamlit dashboard with human review.

> Observability is a shared dependency: Agent C ships `observability/` APIs (`get_logger`, `metrics.*`, `span()`) **first** (stub-complete in the first hour) so A and B can instrument as they go.

### Agent D — "QA" : tests
Owns: `tests/`.
Starts immediately from the contracts in §4 (writes tests against Protocols and fakes), then runs them against real components as they land and files failures into TASKS.md under "Bugs found".

**Coordination rules**
- Only edit files in your owned directories. Need a contract change? Add it under "Contract change requests" in TASKS.md; the lead applies it to `common/`.
- When a task is done: tick it in TASKS.md (your section only), add a one-line note (what/where), and make sure `ruff check` and your tests pass.
- Integration (Phase 2) starts when A, B, C core tasks are ticked.

## 8. Test plan (Agent D)

| Level | Coverage |
|---|---|
| Unit | queue bounds (default 1000 when unset/0/negative; custom size; `try_put` false when full); state machine legal/illegal transitions; policy every rule + combinations + clean path; MockLLM determinism; LLM client error mapping & refusal handling (mocked SDK); user client 404/5xx retry (respx); repository transactions; scheduler overlap guard & jitter; metrics registered |
| Integration | mock UI + Playwright portal: login, pagination, read ticket, post draft idempotent, re-login on expiry, 500 retry; user service API; worker pipeline with fakes → correct states/rows; retry → DLQ; crash recovery re-enqueue |
| End-to-end | `run_all` stack with MockLLM: seed 50 tickets → all reach `DONE`/`NEEDS_REVIEW`; drafts visible in ticket UI; each seeded policy case lands in the expected outcome; approve from DB → draft posted; metrics endpoint values match DB counts |
| Load / backpressure | generate 2000 tickets with queue size 100 and slow LLM → no ticket lost, rejections counted, queue never exceeds capacity, all eventually processed |
| Chaos | UI error rate 20%, user service latency 1s → retries work, no double-posts |
| Dashboard smoke | Streamlit `AppTest` renders every page against a seeded DB; review action writes `review_actions` |
| Optional live | `@pytest.mark.live` single real Anthropic call validating schema (skipped without `ANTHROPIC_API_KEY`) |

## 9. Milestones

| Phase | Exit criteria |
|---|---|
| 0. Contracts & scaffold | package installs, `pytest` runs (empty), contracts reviewed |
| 1. Parallel build | A/B/C core tasks ticked, D unit tests green |
| 2. Integration | `scripts/run_all.sh` brings up all 4 processes; e2e with MockLLM green |
| 3. Hardening | load + chaos tests green, dashboard complete, README run guide |
| 4. Real LLM | `LLM_PROVIDER=anthropic` run on seed set, token/latency metrics verified, prompt tuned |

## 10. Risks & decisions

- **No API on the ticket UI** → selectors are the contract; mock uses `data-testid` attributes, and the portal class is the only code that knows about them.
- **Latency vs. quality** → Opus 5.5 default; switch `LLM_MODEL` to Sonnet/Haiku or lower `LLM_EFFORT` if the ~1s target matters more than reply quality. Measured on the LLM dashboard page.
- **In-memory queue durability** → SQLite is authoritative; startup recovery re-enqueues non-terminal tickets.
- **Browser concurrency** → page pool caps tabs; poller has its own context so posting can't starve discovery.
- **SQLite write contention** → WAL + short transactions + single aiosqlite connection per process; fine at this scale.
