# SupportAgent — Test Report

_Agent D (QA), 2026-10-16, updated for Phase 5 (single-page console). Spec: [PLAN.md](PLAN.md) §8. Tracker: [TASKS.md](TASKS.md) (D1–D12, "Bugs found")._

## Result

```
.venv/bin/pytest -q -m "not live"
1010 passed, 1 deselected in 254.58s (0:04:14)
```

- **Failures: 0. Skips: 0. Xfails: 0.** Every builder module has landed and is tested for real. All three bugs found are fixed, and their regression tests pass.
- **1 deselected:** the optional live Anthropic test. It skips cleanly without `ANTHROPIC_API_KEY`.
- **Lint:** `ruff check tests` and `ruff format --check tests` are clean.
- **Ports:** every server under test binds an ephemeral port. 8001/8002/8003/8501 are never bound, so the suite runs alongside a live stack.
- **Flakiness:** the timing-sensitive files (sampler, scheduler, portal chaos, BUG-1 regression) were each run 2–3× with no flakes, and the full suite gave identical results on two consecutive runs.

Markers (registered in `pyproject.toml`): `browser` (Playwright Chromium), `e2e` (full stack), `load`, `live`.
Faster loop without the load/chaos runs: `pytest -m "not live and not load"`.

| Level | Files | Tests | Status |
|---|---|---|---|
| Unit | `tests/unit/` (14 files) | 788 | all pass |
| Integration | `tests/integration/` (5 files + `conftest.py`) | 216 | all pass |
| E2E | `tests/e2e/test_full_stack.py` (+ live) | 2 (+1) | pass (live deselected) |
| Load / chaos | `tests/load/` | 4 | all pass |

## What's covered

### Harness (D1)
- **`tests/conftest.py`:**
  - `settings` fixture: tmp db/log/artifacts paths, mock LLM, millisecond backoffs, ephemeral ops port. It strips `SA_*` from the env.
  - `db_path`.
  - `require(module, *attrs)`: skip-until-landed, works at module level.
  - `serve(app)`: uvicorn on a pre-bound ephemeral socket in a thread. `ServerThread.app` exposes `app.state` for chaos knobs and session rotation.
  - `start_mock_ui(**kw)`, `start_user_service(**kw)`, `live_settings`.
  - Resets the span sink after every test.
- **`tests/fakes.py`:**
  - `FakeTicketPortal` (dedups drafts on idempotency key), `FakeUserInfoClient`, `FakeLLMClient` (tracks concurrency), `FakePolicyGate`.
  - `Faults` failure injection: fail N times or forever, retryable or fatal, delays.
  - Model factories.
  - `load_seed_cases()` / `seed_case_from_raw()`: seed and generated tickets with their `meta` expectations and a user derived from `customer_profile`.
- **`tests/integration/conftest.py`:** `Harness`, a real repository, queue, poller and `WorkerPool` with fakes, plus `wait_settled`.

### Unit (D2–D4)
| File | Covers |
|---|---|
| `test_config.py` | PLAN §5.13 defaults; `QUEUE_MAX_SIZE` None/""/blank/0/negative/non-numeric → 1000 via kwarg, env and YAML; env override (case-insensitive); precedence kwargs > env > YAML > defaults; invalid literals |
| `test_states.py` | all 169 (from, to) pairs vs a hand-written PLAN §4.3 table; `InvalidTransition` fields; `TERMINAL_STATES`, `RECOVERABLE_STATES`; nothing re-enters DISCOVERED; reachability |
| `test_models.py` | confidence bounds, every Literal, required/optional fields, JSON round-trip, shared-default safety, trace ids |
| `test_observability.py` | spans → sink, error/cancel marked, sink failure swallowed and never masks the original error; every PLAN §5.11 metric registered with the right type; `snapshot()`; `sample_once` / `run_metric_sampler` (period, prompt stop, final sample, errors swallowed, interval clamp); JSON log context |
| `test_queue.py` | default 1000, `try_put` false when full, FIFO, `put` waits/times out, 10 concurrent producers never exceed capacity, enqueued/rejection/depth/capacity/wait metrics |
| `test_anthropic_client.py` | mocked transport, no real calls. The SDK uses `httpx2`, so the tests inject `httpx2.MockTransport` rather than respx. Request body: model, max_tokens, cached frozen system block, `output_config` effort + json_schema, `fallbacks` + beta header on/off, no thinking/temperature/budget_tokens/prefill. Refusal (empty/partial/valid JSON), `max_tokens`, schema violation, prose → `analysis=None` with usage kept; served model on fallback; 14 HTTP statuses + connect/timeout/missing key → `LLMError(retryable)`; metric deltas; span |
| `test_mock_llm.py` | determinism, all 53 seeds schema-valid, the 3 markers, `$` amount parsing, keyword flags, drafts never unsafe, latency, telemetry |
| `test_policy.py` | each of the 13 rules alone (+ priority); boundaries; keywords on subject/body/thread with word boundaries; other-user email PII + allowed domains; combos; YAML toggles/thresholds/keywords/priorities/patterns; `PolicyConfigError`; **53-seed cross-check** (MockLLM + gate == `meta.expected_rules` / `expected_outcome`) |
| `test_repository.py` | pragmas, upsert dedup, every transition logged in order, illegal/missing → no writes, concurrency (one winner; 200 concurrent upserts), error/attempt/finished_at, all step records, FK enforcement, idempotent drafts, review queries, extras, spans/metric samples/scheduler_runs, WAL reader |
| `test_review.py` | `apply_review` approve/edit/reject, bad input and wrong state are atomic, double review rejected, `requeue_dead_letter` resets attempt |
| `test_scheduler.py` | fire-on-start, interval, jitter bounds/determinism, overlap guard + `scheduler_skipped_total`, errors keep the loop alive, `last_run_at` only advances on successful runs, stop waits then cancels, restart |
| `test_sql_guard.py` | Phase 5 SQL console data layer. **38 rejected statements:** UPDATE / DELETE / DROP / ALTER / CREATE / INSERT / REPLACE INTO / ATTACH / DETACH / VACUUM [INTO] / REINDEX / BEGIN, multiple statements, `WITH … DELETE/UPDATE/INSERT`, `PRAGMA query_only=OFF` and other PRAGMA writes or non-allow-listed PRAGMAs, `;` hidden in comments and string literals, `load_extension`, `EXPLAIN DELETE`, empty or incomplete statements. **20 accepted:** SELECT, WITH, VALUES, EXPLAIN [QUERY PLAN], allow-listed PRAGMAs, keywords inside literals and comments, `replace()`, `updated_at`/`created_at` columns. Also: 100-row default + `has_more`; column types; DB byte-identical after rejected writes; **guard bypassed → the read-only connection still refuses every write**; timeout → `SqlTimeout`; a missing DB is never created; `schema_overview` |
| `test_user_client.py` | scripted ephemeral server: API key, 404 → None (no retry), 5xx retried, 4xx not, connect error, timeout + recovery, TTL cache, errors not cached, URL-encoding, metrics; respx 5xx/timeout |

### Integration (D5, D6, D10 / U6)
| File | Covers |
|---|---|
| `test_portal.py` (`browser`) | Playwright against the real mock UI: lifecycle; bad credentials; all 53 seeds across pages (sizes 20 and 7) + 173 after `/admin/generate`; exact `get_ticket`; `post_draft` idempotent (repeat, 6 concurrent, across a portal restart), visible in raw HTML; notes; status; small page pool; **re-login** after cookies are cleared and after the server rotates its secret; persistent 500 → `PortalError` + png/html artifacts; transient 500s retried; BUG-1 regression (exactly N failed `POST /login`); drafts under 30% chaos not duplicated; spans + `browser_ops_total` |
| `test_user_service.py` | mock user service API: auth, every seed user matches its profile, unknown → 404, orders, chaos knobs; real `HttpUserInfoClient` against it. **Phase 5 `GET /users`:** default page (50 users, limit 100, sorted by id), key required, limit/offset pagination, 422 on bad params, case-insensitive `q` on email/name/id, chaos applies. `HttpUserInfoClient.list_users`: pages, search, 422 → `UserServiceError`, service down, not cached |
| `test_pipeline.py` | poller dedup, transition-before-enqueue invariant, **backpressure** (stays DISCOVERED, rejections counted, picked up next poll; 40 tickets through queue 5 with nothing lost); **happy path → DONE with one draft** and every step row and span; **each of the 13 rules → NEEDS_REVIEW + "Escalated" note** (real gate); **retry → DEAD_LETTER after max_retries** (LLM, portal fetch, user service, post); **non-retryable → DLQ immediately**; timeout; poison ticket isolated; no double-post on retry (BUG-2 regression); **crash recovery** (crash mid-LLM, then `Runtime` restart → DONE once each; every recoverable state re-enqueued, NEEDS_REVIEW untouched) |
| `test_review_flow.py` | `apply_review` + `ReviewDispatcher`: approve / edit / reject, metrics, idempotent second pass, post failure → retry → DONE, persistent → DLQ, failure after post doesn't double-post, background loop, DLQ requeue → workers → DONE |
| `test_dashboard.py` | **Single-page console** via `AppTest`, with tabs picked through `session_state`. The DB is seeded by the real pipeline with 204 tickets. **Rendering:** all 9 views (5 Observability sections, All tickets, Review queue, User Info, SQL) × seeded / empty / missing DB, 0 exceptions; a missing DB shows "No database" and is never created. Also the default tab, header counts, log lines and every time range. **Tickets:** 100 rows by default, Load more → 200 → 204, page size, state + search filters, DLQ requeue. Review queue approve / edit / reject write the right rows; reviewer name required. **SQL:** 100 rows + "more available", Load next → 200 → 204, page size, every example runs, 7 write attempts rejected with the DB file byte-identical, SQL errors shown, timeout gives a friendly warning. **User Info** (service on an ephemeral port): look up by email and by id, unknown-user warning, browse list (50 of 50, no Load more), search. Service down → DB fallback list, and an error on lookup; no DB and service down → error |

### End to end, load, chaos (D7–D9, D11)
- **E2E (`test_full_stack.py`).** Mock UI and user service run on ephemeral ports, with the real `Runtime` and nothing injected.
  - **Outcomes:** all 53 seeds settle as **19 DONE / 34 NEEDS_REVIEW**, and every ticket's rules equal its `meta.expected_rules`.
  - **UI state:** each DONE ticket has exactly one draft in the UI; each review ticket has one note and status `escalated`.
  - **Ops API:** `/health` and `/ready` return 200. **`/metrics` deltas match the DB:** discovered, processed, decisions, drafts, LLM calls and tokens, and every rule-hit count.
  - **Review:** approve and edit get posted by the dispatcher.
  - **UI down:** `/ready` returns 503 with reasons.
- **Load (`test_backpressure.py`).** 2000 generated tickets, queue 100, slow LLM, 8 workers.
  - The run is done twice: real `Runtime` over an in-memory portal (~11 s), then the full stack through Playwright (~110 s).
  - **Results:** no ticket lost, and queue depth (sampled every 3 ms) never exceeded 100. `queue_rejections_total` equalled Σ `scheduler_runs.skipped_full` (> 0). There were no DLQ tickets: 1574 tickets were DONE and 426 went to NEEDS_REVIEW, each as expected, with one draft per DONE ticket.
- **Chaos (`test_chaos.py`).**
  - **Run 1:** UI error rate 0.2 plus a 1 s user service.
  - **Run 2:** a wrapper that **applies the draft/note POST and then answers 500** (response lost after the write) 30% of the time, plus error rate 0.1.
  - **Both runs:** every ticket reaches its expected outcome, with no double-posts and no DLQ. Notes are at-least-once by design (they have no idempotency key).
- **Live (`test_live_anthropic.py`, `live`):** one real `AnthropicLLMClient` call validated against the schema. Skips without `ANTHROPIC_API_KEY`. It has **not been run live** in this environment.

## Bugs found

| ID | Owner | Summary | Status |
|---|---|---|---|
| BUG-1 | A | `_login()` treated a 5xx on `POST /login` as bad credentials and never retried, so anything needing a (re)login failed permanently under chaos | ✅ fixed by A; deterministic regression tests pass |
| BUG-2 | B | Draft idempotency key = `hash(draft body)`, so a retry after a failure that followed a successful post re-ran the LLM. A non-deterministic LLM then produced a different draft, which got a new key, and the customer received two replies (MockLLM hid it) | ✅ fixed by B; regression test passes |
| BUG-3 | C | A non-numeric `confidence_threshold` in policy.yaml raised a bare `ValueError` instead of `PolicyConfigError` | ✅ fixed by the lead (`policy/gate.py::_number`); regression test passes |

Details and repro test ids are in TASKS.md → "Bugs found".

## Observations (not bugs)
- The mock UI stores browser-submitted drafts with CRLF line breaks (textarea form semantics). The rendered HTML and `get_ticket` show LF. The E2E test normalises them.
- At 20–30% UI chaos, a single portal op can exhaust its 3 attempts. Worker-level retries recover, so chaos tests use `max_retries=6` and assert eventual success, not zero portal errors.
- Phase 5 console: with the user service down but a DB present, User Info quietly falls back to customer emails from tickets (a caption, no error). The error appears on a lookup, or when there is also no DB. The Logs section renders without a DB. Both are by design, and the tests assert them.
- `app.py` reads the `lru_cache`d `get_settings()`, and AppTest runs the script in-process, so the console fixture clears that cache per test. Production is unaffected (one env per process).
- The installed `anthropic` SDK (1.15) is built on `httpx2`, so respx cannot intercept it. Use `httpx2.MockTransport`. The user client is plain `httpx`, so respx works there.

## Pending
- **D11 live run:** needs an `ANTHROPIC_API_KEY`. Run `ANTHROPIC_API_KEY=... .venv/bin/pytest -m live`.
- Nothing else is blocked.
