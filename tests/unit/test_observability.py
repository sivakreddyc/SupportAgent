"""Observability (PLAN §5.11): tracing sink, error marking, sink failures, metrics registry/snapshot, sampler, logs."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid

import pytest

from support_agent.observability import metrics as m
from support_agent.observability import tracing
from support_agent.observability.logging import bind_context, clear_context, configure_logging, get_logger
from support_agent.observability.sampler import run_metric_sampler, sample_once
from support_agent.observability.tracing import set_span_sink, span

# --------------------------------------------------------------------------- tracing


@pytest.fixture
def sink():
    records: list[dict] = []

    async def _sink(rec):
        records.append(rec)

    set_span_sink(_sink)
    return records


async def test_span_reaches_sink(sink):
    async with span("llm.analyze", trace_id="tr1", ticket_id="T-1", model="m") as sp:
        await asyncio.sleep(0.01)
        sp.set(output_tokens=12)
    assert len(sink) == 1
    rec = sink[0]
    assert rec["name"] == "llm.analyze"
    assert rec["trace_id"] == "tr1" and rec["ticket_id"] == "T-1"
    assert rec["status"] == "ok"
    assert rec["duration_ms"] >= 9
    assert rec["attrs"] == {"model": "m", "output_tokens": 12}
    assert "T" in rec["start_ts"]  # ISO timestamp


async def test_nested_spans_each_recorded(sink):
    async with span("outer", trace_id="t"):
        async with span("inner", trace_id="t"):
            pass
    assert [r["name"] for r in sink] == ["inner", "outer"]


async def test_span_error_is_marked_and_reraised(sink):
    with pytest.raises(ValueError, match="boom"):
        async with span("step", ticket_id="T-2"):
            raise ValueError("boom")
    assert sink[0]["status"] == "error"
    assert sink[0]["attrs"]["error"] == "ValueError: boom"


async def test_span_error_keeps_explicit_error_attr(sink):
    with pytest.raises(RuntimeError):
        async with span("step") as sp:
            sp.set(error="custom")
            raise RuntimeError("x")
    assert sink[0]["attrs"]["error"] == "custom"


async def test_span_cancellation_marked_error(sink):
    async def work():
        async with span("slow"):
            await asyncio.sleep(10)

    task = asyncio.create_task(work())
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sink[0]["status"] == "error"
    assert "CancelledError" in sink[0]["attrs"]["error"]


async def test_sink_failure_is_swallowed():
    calls = 0

    async def bad_sink(rec):
        nonlocal calls
        calls += 1
        raise RuntimeError("db locked")

    set_span_sink(bad_sink)
    async with span("ok-step") as sp:
        sp.set(a=1)
    assert calls == 1


async def test_sink_failure_does_not_mask_original_error():
    async def bad_sink(rec):
        raise RuntimeError("db locked")

    set_span_sink(bad_sink)
    with pytest.raises(KeyError):
        async with span("step"):
            raise KeyError("orig")


async def test_no_sink_is_fine():
    set_span_sink(None)
    async with span("x"):
        pass
    assert tracing._sink is None


def test_span_record_shape():
    rec = tracing.Span("n", "tr", "T", {"k": "v"}).to_record()
    assert set(rec) == {"trace_id", "ticket_id", "name", "start_ts", "duration_ms", "status", "attrs"}


# --------------------------------------------------------------------------- metrics

PLAN_METRICS = {
    "counter": [
        "scheduler_ticks_total",
        "scheduler_skipped_total",
        "poller_runs_total",
        "browser_launches_total",
        "browser_ops_total",
        "queue_enqueued_total",
        "queue_rejections_total",
        "tickets_discovered_total",
        "tickets_processed_total",
        "tickets_failed_total",
        "tickets_dead_letter_total",
        "user_lookups_total",
        "llm_calls_total",
        "llm_tokens_total",
        "llm_refusals_total",
        "policy_decisions_total",
        "policy_rule_hits_total",
        "drafts_posted_total",
        "reviews_total",
    ],
    "gauge": [
        "scheduler_last_run_timestamp",
        "poller_tickets_found",
        "queue_depth",
        "queue_capacity",
        "workers_configured",
        "workers_busy",
    ],
    "histogram": [
        "poller_duration_seconds",
        "browser_op_seconds",
        "queue_wait_seconds",
        "ticket_e2e_seconds",
        "ticket_processing_seconds",
        "step_seconds",
        "user_lookup_seconds",
        "llm_latency_seconds",
    ],
}


@pytest.mark.parametrize("kind,name", [(k, n) for k, ns in PLAN_METRICS.items() for n in ns])
def test_plan_metric_registered(kind, name):
    text = m.render_latest().decode()
    assert f"# TYPE {name} {kind}" in text


def test_render_latest_is_prometheus_text():
    out = m.render_latest()
    assert isinstance(out, bytes)
    assert b"# HELP queue_depth" in out


def _snap(name: str, **labels) -> float | None:
    for n, lb, v in m.snapshot():
        if n == name and lb == labels:
            return v
    return None


def test_snapshot_reflects_counter_and_gauge():
    label = f"test-{uuid.uuid4().hex[:6]}"
    m.REVIEWS.labels(action=label).inc(3)
    m.QUEUE_DEPTH.set(42)
    assert _snap("reviews_total", action=label) == 3.0
    assert _snap("queue_depth") == 42.0


def test_snapshot_histograms_only_count_and_sum():
    model = f"m-{uuid.uuid4().hex[:6]}"
    m.LLM_LATENCY.labels(model=model).observe(0.3)
    m.LLM_LATENCY.labels(model=model).observe(0.7)
    names = {n for n, _, _ in m.snapshot()}
    assert not any(n.endswith("_bucket") or n.endswith("_created") for n in names)
    assert _snap("llm_latency_seconds_count", model=model) == 2.0
    assert _snap("llm_latency_seconds_sum", model=model) == pytest.approx(1.0)


def test_snapshot_types():
    for name, labels, value in m.snapshot():
        assert isinstance(name, str)
        assert isinstance(labels, dict) and all(isinstance(v, str) for v in labels.values())
        assert isinstance(value, float)


# --------------------------------------------------------------------------- sampler


class FakeMetricRepo:
    def __init__(self, fail_times: int = 0) -> None:
        self.batches: list[list] = []
        self.fail_times = fail_times
        self.attempts = 0

    async def record_metric_samples(self, samples):
        self.attempts += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("database is locked")
        self.batches.append(list(samples))


async def test_sample_once_writes_snapshot():
    repo = FakeMetricRepo()
    n = await sample_once(repo)
    assert n == len(repo.batches[0]) > 0
    assert any(name == "queue_capacity" for name, _, _ in repo.batches[0])


async def test_sample_once_propagates_errors():
    with pytest.raises(RuntimeError):
        await sample_once(FakeMetricRepo(fail_times=1))


async def test_sampler_runs_periodically_and_final_sample_on_stop():
    repo = FakeMetricRepo()
    stop = asyncio.Event()
    task = asyncio.create_task(run_metric_sampler(repo, 0.05, stop))
    await asyncio.sleep(0.28)
    stop.set()
    await asyncio.wait_for(task, 2)
    # ~6 periodic samples + 1 final
    assert 4 <= len(repo.batches) <= 9


async def test_sampler_stops_promptly():
    repo = FakeMetricRepo()
    stop = asyncio.Event()
    task = asyncio.create_task(run_metric_sampler(repo, 60, stop))
    await asyncio.sleep(0.05)
    stop.set()
    await asyncio.wait_for(task, 1)  # does not wait out the 60s interval
    assert len(repo.batches) == 2  # first + final


async def test_sampler_swallows_repo_errors():
    repo = FakeMetricRepo(fail_times=2)
    stop = asyncio.Event()
    task = asyncio.create_task(run_metric_sampler(repo, 0.05, stop))
    await asyncio.sleep(0.25)
    stop.set()
    await asyncio.wait_for(task, 2)
    assert not task.cancelled() and task.exception() is None
    assert repo.attempts >= 3 and len(repo.batches) >= 1


async def test_sampler_final_failure_swallowed():
    repo = FakeMetricRepo(fail_times=100)
    stop = asyncio.Event()
    stop.set()
    await asyncio.wait_for(run_metric_sampler(repo, 0.05, stop), 1)
    assert repo.attempts == 1 and repo.batches == []


async def test_sampler_clamps_tiny_interval():
    repo = FakeMetricRepo()
    stop = asyncio.Event()
    task = asyncio.create_task(run_metric_sampler(repo, 0, stop))
    await asyncio.sleep(0.2)
    stop.set()
    await asyncio.wait_for(task, 1)
    assert len(repo.batches) <= 7  # min interval 0.05s, so no hot loop


# --------------------------------------------------------------------------- logging


def test_json_log_lines_carry_context(tmp_path):
    log_file = tmp_path / "agent.jsonl"
    configure_logging("INFO", str(log_file))
    try:
        bind_context(trace_id="tr9", ticket_id="T-9", worker_id="w1")
        get_logger("worker").info("ticket.done", outcome="auto_approve")
        get_logger("worker").debug("hidden")
        clear_context()
        for h in logging.getLogger().handlers:
            h.flush()
        lines = [json.loads(line) for line in log_file.read_text().splitlines() if line.strip()]
        rec = next(r for r in lines if r["event"] == "ticket.done")
        assert rec["component"] == "worker"
        assert rec["level"] == "info"
        assert rec["trace_id"] == "tr9" and rec["ticket_id"] == "T-9" and rec["worker_id"] == "w1"
        assert rec["outcome"] == "auto_approve"
        assert "ts" in rec
        assert not any(r["event"] == "hidden" for r in lines)
    finally:
        clear_context()
        configure_logging("INFO", None)
