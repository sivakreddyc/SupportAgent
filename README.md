# SupportAgent

An autonomous support agent. It reads tickets from a web UI that has **no API** (by driving headless Chromium with
Playwright), enriches each ticket with customer data from a User Info Service, asks an LLM for a structured analysis
and a draft reply, and runs the result through a policy gate. Safe, routine replies are posted back to the ticket UI.
Anything risky (refunds, legal, security, angry customers, low confidence, VIPs, …) is escalated to a human reviewer
in a Streamlit dashboard. Every state change, draft, LLM call and span is stored in SQLite.

- Design spec: [docs/PLAN.md](docs/PLAN.md)
- Task tracker and decisions log: [docs/TASKS.md](docs/TASKS.md)
- Test report: [docs/TEST_REPORT.md](docs/TEST_REPORT.md)

## Architecture

```
 Mock Ticket UI :8001 ◄── Playwright (headless Chromium) ──┐
 (HTML only, no API)                                       │
                                                           │
  Scheduler ─► Discovery poller ─► Bounded queue ─► Worker pool (N)
  (interval    (dedup, backpressure)  (default 1000)   1. read ticket        (Playwright)
   + jitter)                                           2. get user info  ──► User Info Service :8002 (FastAPI)
                                                       3. analyze        ──► LLM (Anthropic Claude, or offline mock)
                                                       4. policy gate
                                                       5. post draft  or  escalate with an internal note
                                                                │
   Review dispatcher ◄── APPROVED ◄── Streamlit dashboard :8501 ◄┤  SQLite (WAL): tickets, transitions,
   (posts human-approved drafts)      (human review, ops pages)  │  drafts, LLM calls, spans, metrics
                                                                 │
   Ops API :8003  /health  /ready  /metrics  /api/status ◄───────┘  + structured JSON logs
```

Ticket lifecycle: `DISCOVERED → QUEUED → PROCESSING → ENRICHED → ANALYZED → POLICY_CHECKED →`
`DRAFT_POSTED → DONE`, or `→ NEEDS_REVIEW → APPROVED/REJECTED → DONE`. Failures go `FAILED →` retry with backoff
`→ QUEUED`, or `→ DEAD_LETTER` after `max_retries`. Every transition is checked against the state machine in
`src/support_agent/common/states.py` and logged.

## Quick start

Requirements: Python 3.11+ (3.12 recommended) and [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.12
uv pip install -e '.[dev]'
.venv/bin/playwright install chromium

scripts/run_all.sh          # starts all four processes; Ctrl-C stops them
```

**Open the console: http://localhost:8501.** It's the only page you need, with four tabs:

| Tab | What it's for |
|---|---|
| **Observability** | KPIs, throughput, end-to-end latency, queue and workers, LLM calls and tokens, scheduler and poller runs, and the JSON log tail |
| **Tickets** | every ticket (100 rows by default, then *Load more*), each ticket's timeline, spans, LLM output, policy reasons and drafts; the **Review queue** sub-tab to approve, edit or reject escalated drafts; the **New ticket** sub-tab to create a ticket and watch it go through the pipeline; requeue dead-letter tickets |
| **User Info** | look up a customer by email or user id (profile, plan, status, VIP, orders, their tickets), or browse all users (100 by default) |
| **SQL** | read-only SQL console on the agent's SQLite DB: first 100 rows by default, *Load next 100*, schema browser, example queries, CSV download |

The other processes run in the background, and you don't need to open them: the mock ticket UI on :8001 (the "external"
portal the agent reads through Playwright; login `agent` / `agent`), the User Info Service on :8002
(`X-API-Key: dev-key`), and the agent's ops API on :8003 (`/health`, `/ready`, `/metrics`, `/api/status`, `POST /api/poll`).

With the default settings, the 53 seed tickets settle within a few seconds: 19 drafts are posted and 34 tickets
wait in **Tickets → Review queue**. To push more load, use **Generate** on the mock ticket UI
(or `curl -X POST -d n=500 localhost:8001/admin/generate`).

> The mock ticket UI keeps its tickets in memory, while the agent's DB persists in `data/`. For a clean start,
> stop the stack and move or delete `data/` and `logs/`.

You can also run each process on its own: `support-agent ticket-ui | user-service | run | dashboard`.

> **macOS note:** if `import support_agent` fails inside the venv, macOS has set the hidden flag on the editable
> install's `.pth` file, and Python skips hidden `.pth` files. Fix it with
> `ln -sfn "$PWD/src/support_agent" .venv/lib/python3.12/site-packages/support_agent`.

## Test the flow yourself

1. Open the console at http://localhost:8501 and go to **Tickets → New ticket**.
2. Pick a template, or write your own ticket (customer email, subject, message, priority). Templates cover each path:
   how-to question (auto-approved), small refund, large refund + angry, legal threat, security incident, enterprise
   (VIP) customer, suspended account, unknown customer, and an LLM refusal.
3. Leave **Poll now** ticked and click **Create ticket**. The ticket is posted to the mock ticket UI like a real
   customer ticket, then the agent discovers it through the browser right away instead of waiting for the next poll.
4. **Pipeline progress** updates every second: `DISCOVERED → QUEUED → PROCESSING → ENRICHED → ANALYZED →
   POLICY_CHECKED →` `DONE` (reply posted) or `NEEDS_REVIEW` (with the rules that fired).
5. For escalated tickets, go to **Tickets → Review queue** to approve, edit or reject the draft. The approved reply is
   posted to the ticket within a few seconds. Click **Open ticket detail** to see every step's timing, the user
   lookup, the LLM output and the policy decision.

Use the email of a known seed customer (e.g. `maria.gonzalez@example.com`, see **User Info**) to get a user record.
Any other email takes the `UNKNOWN_USER` path. Without the console, you can create tickets with **+ New ticket** on
the mock ticket UI (http://localhost:8001) and trigger a poll with `curl -X POST localhost:8003/api/poll`.

## Using the real LLM

The default provider is an offline, deterministic mock, so everything runs without an API key. To use Claude:

```bash
export ANTHROPIC_API_KEY=...
SA_LLM_PROVIDER=anthropic scripts/run_all.sh
```

The client uses the Anthropic Python SDK with structured outputs (the `LLMAnalysis` schema), a cached system prompt,
configurable `effort`, and server-side refusal fallback. Refusals and truncated or invalid output are routed to human
review. Errors are mapped to retryable (429, 5xx, timeouts) or non-retryable (400, 401, 403, …). The default model is
`claude-opus-5-5`. For lower latency, set `SA_LLM_MODEL=claude-sonnet-5-5` or `claude-haiku-5-5`, or lower
`SA_LLM_EFFORT`.

## Configuration

Defaults are in [config/settings.yaml](config/settings.yaml). Any key can be overridden with an `SA_<KEY>`
environment variable (precedence: env > YAML > built-in default).

| Key | Default | Notes |
|---|---|---|
| `queue_max_size` | 1000 | bounded queue; unset, empty or ≤ 0 means 1000 |
| `worker_count` | 4 | async workers |
| `browser_page_pool` | 3 | concurrent Playwright pages used for reads and writes |
| `poll_interval_seconds` | 15 | discovery interval (±10% jitter) |
| `max_retries` | 3 | then `DEAD_LETTER` |
| `ticket_timeout_seconds` | 60 | per-ticket processing timeout |
| `llm_provider` | `mock` | `mock` or `anthropic` |
| `llm_model` / `llm_effort` | `claude-opus-5-5` / `medium` | |
| `db_path` | `data/support_agent.db` | SQLite (WAL) |
| `log_file` | `logs/agent.jsonl` | structured JSON logs |

Policy rules, thresholds, keywords and unsafe-draft patterns are in [config/policy.yaml](config/policy.yaml).

## Querying the database

Use the **SQL** tab. It's read-only by design: it accepts one `SELECT` / `WITH` / `EXPLAIN` / read-only `PRAGMA`
statement at a time, runs it on a `mode=ro` + `query_only` connection with a 5 s timeout, and loads 100 rows at a time.
Writes are blocked because ticket state must only change through the state machine. Approve or reject drafts in
**Tickets → Review queue**. For ad-hoc access outside the console:
`sqlite3 -readonly data/support_agent.db`.

## Policy gate

If any rule fires, the ticket goes to human review. Rules marked high priority sort first in the review queue.

| Rule | Fires when | Priority |
|---|---|---|
| `LOW_CONFIDENCE` | LLM confidence < 0.75 | normal |
| `REFUND` / `REFUND_HIGH_VALUE` | refund requested / amount > $100 | normal / high |
| `LEGAL` | legal flag, category, or keyword (lawyer, sue, GDPR, …) | high |
| `SECURITY` | security flag, category, or keyword (breach, hacked, phishing, …) | high |
| `ANGRY` | angry sentiment | normal |
| `CRITICAL_URGENCY` | critical urgency | high |
| `VIP` | VIP customer or enterprise plan | normal |
| `UNKNOWN_USER` / `ACCOUNT_NOT_ACTIVE` | no user record / account suspended or closed | normal |
| `UNSAFE_DRAFT` | draft contains card-like numbers, refund promises, guarantees, or another person's email | high |
| `LLM_REFUSAL` / `LLM_NO_ANALYSIS` | model refused / output truncated or invalid | normal |

Keyword checks run on the raw ticket text as well as on the LLM's flags, so the gate doesn't rely only on the model.

## Observability

- **Metrics** (`:8003/metrics`, also sampled into SQLite for the dashboard):
  - tickets discovered, processed and failed, and dead-lettered;
  - queue depth, capacity, rejections and wait time;
  - busy and configured workers;
  - LLM calls, tokens, latency and refusals;
  - policy decisions and rule hits;
  - browser operations and launches;
  - scheduler ticks, and per-step and end-to-end latency.
- **Tracing:** one trace id per ticket; every pipeline step is a span stored in the `spans` table.
- **Logs:** structured JSON with `trace_id`, `ticket_id` and `worker_id` bound.
- **Console:** one page at :8501 with the tabs Observability, Tickets (with the Review queue), User Info and SQL.

## Tests

```bash
.venv/bin/pytest -q -m "not live"            # ~880 tests, about 4 minutes (includes load and chaos)
.venv/bin/pytest -q -m "not live and not load"   # faster loop
ANTHROPIC_API_KEY=... .venv/bin/pytest -m live   # one real API call to check the schema
```

Unit, integration (Playwright against the mock UI), end-to-end (full stack), load and backpressure (2000 tickets,
queue 100) and chaos (injected UI errors and lost responses) suites. All servers in the tests bind ephemeral ports,
so the suite can run alongside a live stack. See [docs/TEST_REPORT.md](docs/TEST_REPORT.md).

## Project layout

```
config/                 settings.yaml, policy.yaml
scripts/run_all.sh      start the whole stack
src/support_agent/
  common/               settings, shared models, state machine, interfaces (Protocols)
  mocks/ticket_ui/      HTML-only ticket portal (+ seed tickets, generator, chaos knobs)
  mocks/user_service/   FastAPI User Info Service (+ seed users)
  browser/              Playwright ticket portal client
  clients/              User Info HTTP client
  storage/              SQLite schema, repository, review helpers
  pipeline/             queue, scheduler, poller, workers, review dispatcher, runtime
  llm/                  prompts, Anthropic client, mock client
  policy/               rule-based policy gate
  observability/        JSON logging, Prometheus metrics, tracing, metric sampler
  ops_api/              /health /ready /metrics /api/status
  dashboard/            Streamlit console (one page, tabs: Observability / Tickets / User Info / SQL)
tests/                  unit / integration / e2e / load
docs/                   PLAN.md, TASKS.md, TEST_REPORT.md
```

## Known limitations

- **Throughput is limited by the browser.** Each UI write takes about 150 ms. A burst of 500 tickets takes about 37 s
  with 4 workers and 3 pages, so a ticket's end-to-end time is mostly queue wait under load. Raise `worker_count` and
  `browser_page_pool` to scale.
- **Internal notes are at-least-once.** They have no idempotency key, so in a rare case a note can appear twice.
  Drafts are exactly-once.
- **Review rejections don't reach runtime metrics.** They happen in the dashboard process, so `reviews_total{reject}`
  isn't counted by the runtime.
- **The mocks are mocks.** The ticket UI's admin routes need no login, and the company name and status page
  ("Acme Cloud", `status.acme.example`) are placeholders.
