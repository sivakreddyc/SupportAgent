"""Settings: defaults (PLAN §5.13), QUEUE_MAX_SIZE fallback, precedence kwargs > env > YAML > defaults."""

from __future__ import annotations

from pathlib import Path

import pytest

from support_agent.common import config as config_mod
from support_agent.common.config import DEFAULT_QUEUE_MAX_SIZE, Settings, get_settings


@pytest.fixture
def no_yaml(monkeypatch, tmp_path):
    """Point the YAML source at a missing file so only code defaults apply."""
    monkeypatch.setattr(config_mod, "DEFAULT_SETTINGS_FILE", tmp_path / "missing.yaml")


@pytest.fixture
def yaml_file(monkeypatch, tmp_path) -> Path:
    p = tmp_path / "settings.yaml"
    monkeypatch.setattr(config_mod, "DEFAULT_SETTINGS_FILE", p)
    return p


def test_default_queue_size_constant():
    assert DEFAULT_QUEUE_MAX_SIZE == 1000


def test_code_defaults_match_plan(no_yaml):
    s = Settings()
    assert s.queue_max_size == 1000
    assert s.worker_count == 4
    assert s.poll_interval_seconds == 15
    assert s.max_retries == 3
    assert s.ticket_timeout_seconds == 60
    assert s.browser_headless is True
    assert s.browser_page_pool == 3
    assert s.ticket_ui_url == "http://localhost:8001"
    assert (s.ticket_ui_user, s.ticket_ui_password) == ("agent", "agent")
    assert s.user_service_url == "http://localhost:8002"
    assert s.user_service_api_key == "dev-key"
    assert s.llm_provider == "mock"
    assert s.llm_model == "claude-opus-5-5"
    assert s.llm_effort == "medium"
    assert s.db_path == "data/support_agent.db"
    assert s.log_level == "INFO"
    assert s.ops_api_port == 8003
    assert Path(s.policy_file).name == "policy.yaml"
    assert s.poll_jitter_ratio == pytest.approx(0.1)


def test_repo_settings_yaml_agrees_with_plan():
    """The shipped config/settings.yaml must not silently change PLAN defaults."""
    s = Settings()
    assert s.queue_max_size == 1000
    assert s.worker_count == 4
    assert s.poll_interval_seconds == 15
    assert s.llm_provider == "mock"
    assert Path(s.policy_file).exists()


@pytest.mark.parametrize("raw", [None, "", "   ", 0, "0", -1, "-5", -1000, "not-a-number"])
def test_queue_max_size_falls_back_to_1000_kwarg(no_yaml, raw):
    assert Settings(queue_max_size=raw).queue_max_size == 1000


@pytest.mark.parametrize("raw", ["", "0", "-3", "abc"])
def test_queue_max_size_falls_back_to_1000_env(no_yaml, monkeypatch, raw):
    monkeypatch.setenv("SA_QUEUE_MAX_SIZE", raw)
    assert Settings().queue_max_size == 1000


def test_queue_max_size_unset_env(no_yaml, monkeypatch):
    monkeypatch.delenv("SA_QUEUE_MAX_SIZE", raising=False)
    assert Settings().queue_max_size == 1000


@pytest.mark.parametrize("raw,expected", [(1, 1), ("100", 100), (5000, 5000), (" 250 ", 250)])
def test_queue_max_size_custom(no_yaml, raw, expected):
    assert Settings(queue_max_size=raw).queue_max_size == expected


def test_env_override(no_yaml, monkeypatch):
    monkeypatch.setenv("SA_QUEUE_MAX_SIZE", "100")
    monkeypatch.setenv("SA_WORKER_COUNT", "8")
    monkeypatch.setenv("sa_llm_model", "claude-haiku-5-5")  # case-insensitive
    monkeypatch.setenv("SA_BROWSER_HEADLESS", "false")
    s = Settings()
    assert s.queue_max_size == 100
    assert s.worker_count == 8
    assert s.llm_model == "claude-haiku-5-5"
    assert s.browser_headless is False


def test_yaml_overrides_defaults(yaml_file):
    yaml_file.write_text("queue_max_size: 50\nWORKER_COUNT: 7\nunknown_key: 1\n")
    s = Settings()
    assert s.queue_max_size == 50
    assert s.worker_count == 7  # keys are case-insensitive
    assert not hasattr(s, "unknown_key")


def test_yaml_zero_queue_size_falls_back(yaml_file):
    yaml_file.write_text("queue_max_size: 0\n")
    assert Settings().queue_max_size == 1000


def test_empty_yaml_is_ok(yaml_file):
    yaml_file.write_text("")
    assert Settings().queue_max_size == 1000


def test_precedence_kwargs_over_env_over_yaml(yaml_file, monkeypatch):
    yaml_file.write_text("worker_count: 2\nmax_retries: 9\npoll_interval_seconds: 30\n")
    monkeypatch.setenv("SA_WORKER_COUNT", "5")
    monkeypatch.setenv("SA_MAX_RETRIES", "6")
    s = Settings(max_retries=1)
    assert s.max_retries == 1  # kwarg > env > yaml
    assert s.worker_count == 5  # env > yaml
    assert s.poll_interval_seconds == 30  # yaml > default


def test_invalid_literal_rejected(no_yaml):
    with pytest.raises(ValueError):
        Settings(llm_provider="openai")
    with pytest.raises(ValueError):
        Settings(llm_effort="extreme")


def test_get_settings_is_cached():
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()
