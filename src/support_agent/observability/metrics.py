"""Prometheus metrics. The names here are the contract (docs/PLAN.md §5.11).

Import the objects directly:
    from support_agent.observability import metrics as m
    m.QUEUE_DEPTH.set(q.qsize())
    m.LLM_TOKENS.labels(model=..., direction="input").inc(n)
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

REGISTRY = CollectorRegistry(auto_describe=True)

_LAT = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120)


def _c(name, doc, labels=()):
    return Counter(name, doc, labels, registry=REGISTRY)


def _g(name, doc, labels=()):
    return Gauge(name, doc, labels, registry=REGISTRY)


def _h(name, doc, labels=(), buckets=_LAT):
    return Histogram(name, doc, labels, registry=REGISTRY, buckets=buckets)


# scheduler
SCHEDULER_TICKS = _c("scheduler_ticks_total", "Scheduler ticks fired")
SCHEDULER_SKIPPED = _c("scheduler_skipped_total", "Ticks skipped because previous poll still running")
SCHEDULER_LAST_RUN = _g("scheduler_last_run_timestamp", "Unix ts of last completed poll")

# poller / browser
POLLER_RUNS = _c("poller_runs_total", "Discovery poll runs", ("result",))  # ok|error
POLLER_DURATION = _h("poller_duration_seconds", "Discovery poll duration")
POLLER_TICKETS_FOUND = _g("poller_tickets_found", "Open tickets seen in last poll")
BROWSER_LAUNCHES = _c("browser_launches_total", "Chromium launches")
BROWSER_OPS = _c("browser_ops_total", "Browser operations", ("op", "result"))
BROWSER_OP_SECONDS = _h("browser_op_seconds", "Browser operation latency", ("op",))

# queue
QUEUE_DEPTH = _g("queue_depth", "Items currently in the queue")
QUEUE_CAPACITY = _g("queue_capacity", "Queue max size")
QUEUE_ENQUEUED = _c("queue_enqueued_total", "Items enqueued")
QUEUE_REJECTIONS = _c("queue_rejections_total", "Enqueue attempts rejected because queue full")
QUEUE_WAIT = _h("queue_wait_seconds", "Time from enqueue to dequeue")

# workers
WORKERS_CONFIGURED = _g("workers_configured", "Configured worker count")
WORKERS_BUSY = _g("workers_busy", "Workers currently processing a ticket")

# tickets
TICKETS_DISCOVERED = _c("tickets_discovered_total", "New tickets discovered")
TICKETS_PROCESSED = _c("tickets_processed_total", "Tickets finished processing", ("outcome",))
TICKETS_FAILED = _c("tickets_failed_total", "Ticket step failures", ("step",))
TICKETS_DEAD_LETTER = _c("tickets_dead_letter_total", "Tickets moved to dead letter")
TICKET_E2E = _h("ticket_e2e_seconds", "Discovered -> final state")
TICKET_PROCESSING = _h("ticket_processing_seconds", "Dequeue -> final state")
STEP_SECONDS = _h("step_seconds", "Per-step latency", ("step",))

# user service
USER_LOOKUPS = _c("user_lookups_total", "User info lookups", ("result",))  # found|not_found|error|cache_hit
USER_LOOKUP_SECONDS = _h("user_lookup_seconds", "User info lookup latency")

# llm
LLM_CALLS = _c("llm_calls_total", "LLM calls", ("model", "result"))  # ok|error|refusal|parse_error
LLM_TOKENS = _c("llm_tokens_total", "LLM tokens", ("model", "direction"))  # input|output|cache_read
LLM_LATENCY = _h("llm_latency_seconds", "LLM call latency", ("model",))
LLM_REFUSALS = _c("llm_refusals_total", "LLM refusals", ("model",))

# policy
POLICY_DECISIONS = _c("policy_decisions_total", "Policy decisions", ("outcome",))
POLICY_RULE_HITS = _c("policy_rule_hits_total", "Policy rule hits", ("rule",))

# drafts / review
DRAFTS_POSTED = _c("drafts_posted_total", "Drafts posted to the ticket UI", ("source",))  # llm|human_edit
REVIEWS = _c("reviews_total", "Human review actions", ("action",))  # approve|edit|reject


def render_latest() -> bytes:
    return generate_latest(REGISTRY)


def snapshot() -> list[tuple[str, dict[str, str], float]]:
    """Flatten current values for persisting into metric_samples (histograms: _count and _sum only)."""
    out: list[tuple[str, dict[str, str], float]] = []
    for family in REGISTRY.collect():
        for s in family.samples:
            if s.name.endswith("_bucket") or s.name.endswith("_created"):
                continue
            out.append((s.name, dict(s.labels), float(s.value)))
    return out
